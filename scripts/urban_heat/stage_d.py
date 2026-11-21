#!/usr/bin/env python3
"""
Stage D — repackage Stage C (reference SOLWEIG) labels into V5 training tensors.
================================================================================

V5 replaces the LABELS. The old label engine had no surface energy balance
(`Tground = Ta + 8`); Stage C regenerated T_mrt with the published SOLWEIG-GPU.
This script pairs those labels with the existing V4 inputs and writes the
training corpus.

WHAT THIS FIXES (each verified against data, see docs/.../memory.md 2026-08-16)
------------------------------------------------------------------------------
1. BAND MAPPING. The (doy, hour) -> band index is read from
   DONE.json["saved_bands"][i]["index_in_file"] and never re-derived. Two
   separate errors in this project came from re-deriving it. The layout is
   identical across patches, so a mismatch is a hard error, not a surprise.

2. VALIDITY MASK. Reference SOLWEIG writes a finite value EVERYWHERE, including
   pixels where the input DSM is nodata -- it computes T_mrt from the -9999
   sentinel and returns a plausible-looking number. The old V4 labels marked
   those pixels NaN; naively swapping labels would turn 12% of the corpus from
   "correctly excluded" into "invented and trained on". So the mask is derived
   from the INPUT rasters and cross-checked against V4's own finite mask.

3. NO MEAN-FILL. package_training_data.py:369-374 replaced missing pixels with
   the patch mean, teaching the model invented numbers as measurements. Invalid
   pixels here are NaN in `tmrt` plus an explicit `valid` array.

4. SOLAR CHANNELS. sin_alt (ch3) and alt_norm (ch4) are identically ZERO in the
   whole V3/V4 corpus while inference feeds real values -- a train/serve skew,
   and those first-layer weights never received a gradient. Recomputed here.

5. UTC OFFSET = +11. SOLWEIG derives its own timezone (+11) and reads met `it`
   as local time, so corpus hour H pairs with local hour H+1. Verified: at
   local hour 14 this reproduces SOLWEIG's own reported solar altitude of
   71.4 deg (utc_offset=10 gives 62.9). Melbourne is on AEDT for all five dates.

6. WEATHER. era5_* channels are re-read from the SAME met file that drove the
   labels, at `it` = corpus_hour + 1. Verified against V4: d015_h09 carries
   tair 21.07 / rh 71.37 / wind 1.73 / ssrd 460.9, which is met row it=10
   exactly. So V4's weather was already on the label's clock.

7. LST HEAD MASKED FOR DOY 351. The heat date (16 Dec 2024, 39.2 C) has no
   coincident satellite scene. Substituting a nearby one would mislabel the one
   date added for its heat. `lst` is NaN and `lst_valid` is 0 for that date.

KNOWN DEFECT NOT FIXED HERE (deliberately) -- see --shadow-channel
------------------------------------------------------------------------------
compute_shadow_fraction() traces its ray AWAY from the sun (the array shift in
urban_heat_feature_engineering.py:190-200 samples dsm[p-d] where it needs
dsm[p+d]). Measured against SOLWEIG's own shadow raster it scores IoU 0.052
as-is and 0.218 with the azimuth flipped 180 deg. It is what LIVE inference
feeds into channel 10.

Filling channel 10 from SOLWEIG's (correct) shadow while inference keeps
computing a backwards one would CREATE a train/serve skew rather than remove
one. So the default is `zeros` -- the status quo, which at least matches the
V4 corpus -- and switching to `solweig` is a deliberate choice that must be
paired with an inference fix.

USAGE
-----
    python3 stage_d.py --stage-c-dir <dir> --out-dir <dir> [--limit N] [--dry-run]

Writes only under --out-dir. Reads the live corpus read-only. Touches nothing
under urban_heat_data/models/.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform as warp_transform

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

DATA = Path("/data") if Path("/data/training").exists() else Path("/app/urban_heat_data")
V4_DIR = DATA / "training/patches_v4"
PATCH_DIR = DATA / "training/patches"

NUM_CHANNELS = 23
PATCH_PX = 512

# Channel slots we touch. Names mirror V4_CHANNEL_ORDER.
CH_DSM = 0
CH_SIN_ALT = 3
CH_ALT_NORM = 4
CH_SHADOW_FRACTION = 10
CH_ERA5_TAIR = 11
CH_ERA5_RH = 12
CH_ERA5_WIND = 13
CH_ERA5_SSRD = 14

# The 16 slots that do not vary with date or hour. Verified: identical across all
# 12 date-hours for 6 sampled patches, 0 mismatches. This is what makes the
# portable donor pack ~12x smaller than shipping the V4 corpus.
STATIC_SLOTS = (0, 1, 2, 5, 6, 7, 8, 9, 15, 16, 17, 18, 19, 20, 21, 22)
DYNAMIC_SLOTS = (3, 4, 10, 11, 12, 13, 14)

# met_day.txt columns (0-based) for the UMEP format:
# iy id it imin qn qh qe qs qf U RH Tair pres rain kdown snow ldown ...
MET_DOY, MET_HOUR = 1, 2
MET_WIND, MET_RH, MET_TAIR, MET_KDOWN = 9, 10, 11, 14

# SOLWEIG derives +11 for Melbourne and reads met `it` as local time.
UTC_OFFSET = 11

# DSM denormalisation factor, matching urban_heat_inference.py:802
DSM_SCALE = 300.0

# The heat date has no coincident Landsat scene.
DOY_NO_LST = 351


# ---------------------------------------------------------------------------
# Solar helpers — verbatim from app/services/urban_heat_solar.py so this script
# can run in a container without the app package mounted. Kept byte-identical;
# any divergence would silently desynchronise training from inference.
# ---------------------------------------------------------------------------

def compute_solar_position(lat: float, lon: float, doy: int, hour: int,
                           utc_offset: int = UTC_OFFSET) -> tuple[float, float]:
    solar_hour = hour - utc_offset + lon / 15.0
    dec = -23.45 * math.cos(math.radians(360 / 365 * (doy + 10)))
    ha = 15 * (solar_hour - 12)

    lat_r, dec_r, ha_r = map(math.radians, (lat, dec, ha))
    sin_alt = (math.sin(lat_r) * math.sin(dec_r)
               + math.cos(lat_r) * math.cos(dec_r) * math.cos(ha_r))
    altitude = math.degrees(math.asin(max(-1, min(1, sin_alt))))

    if math.cos(math.radians(altitude)) < 0.001:
        return altitude, 180.0

    cos_az = ((math.sin(dec_r) - math.sin(lat_r) * sin_alt)
              / (math.cos(lat_r) * math.cos(math.radians(altitude))))
    azimuth = math.degrees(math.acos(max(-1, min(1, cos_az))))
    if ha > 0:
        azimuth = 360 - azimuth
    return altitude, azimuth


def get_solar_channels(sun_altitude: float) -> tuple[float, float]:
    alt = max(sun_altitude, 0.0)
    return math.sin(math.radians(alt)), alt / 90.0


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

@dataclass
class Stats:
    patches: int = 0
    written: int = 0
    skipped_invalid: int = 0
    skipped_no_v4: int = 0
    doy351: int = 0
    mask_checked: int = 0
    mask_mismatch: int = 0
    invalid_fracs: list[float] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def report(self) -> None:
        print("\n" + "=" * 66)
        print(f"patches processed        : {self.patches}")
        print(f"samples written          : {self.written}")
        print(f"  of which DOY 351       : {self.doy351}  (lst head masked)")
        print(f"skipped, too much invalid: {self.skipped_invalid}")
        print(f"skipped, no V4 donor     : {self.skipped_no_v4}")
        if self.invalid_fracs:
            a = np.array(self.invalid_fracs)
            print(f"invalid pixel fraction   : median {np.median(a)*100:.2f}%  "
                  f"mean {a.mean()*100:.2f}%  max {a.max()*100:.2f}%")
        print(f"mask cross-checks vs V4  : {self.mask_checked} checked, "
              f"{self.mask_mismatch} mismatched")
        if self.errors:
            print(f"\nerrors ({len(self.errors)}), first 10:")
            for e in self.errors[:10]:
                print(f"  {e}")
        print("=" * 66)


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def load_met(path: Path) -> dict[tuple[int, int], dict[str, float]]:
    """(doy, local_hour) -> weather. This is the file that drove the labels."""
    out: dict[tuple[int, int], dict[str, float]] = {}
    for line in path.read_text().splitlines()[1:]:
        f = line.split()
        if len(f) <= MET_KDOWN:
            continue
        key = (int(f[MET_DOY]), int(f[MET_HOUR]))
        out[key] = {
            "wind": float(f[MET_WIND]),
            "rh": float(f[MET_RH]),
            "tair": float(f[MET_TAIR]),
            "ssrd": float(f[MET_KDOWN]),
        }
    return out


def patch_lonlat(path: Path) -> tuple[float, float]:
    with rasterio.open(path) as s:
        t, w, h, crs = s.transform, s.width, s.height, s.crs
    lon, lat = warp_transform(crs, "EPSG:4326",
                              [t.c + t.a * w / 2], [t.f + t.e * h / 2])
    return lon[0], lat[0]


def validity_mask(patch: str, edge_buffer_px: int) -> np.ndarray | None:
    """Valid where the INPUT DSM has real data. Never derived from the label:
    reference SOLWEIG writes finite values over nodata, so the label cannot
    reveal what is missing."""
    f = PATCH_DIR / patch / "building_dsm.tif"
    if not f.exists():
        return None
    with rasterio.open(f) as s:
        dsm = s.read(1)
    valid = dsm > -9000

    if edge_buffer_px > 0:
        # A missing building removes the shadow it would have cast, so pixels
        # NEAR a hole are wrong too even though they look valid.
        from scipy import ndimage
        valid = ~ndimage.binary_dilation(~valid, iterations=edge_buffer_px)
    return valid


@dataclass
class Donor:
    """One patch's portable inputs. Replaces the 44 GB V4 corpus + patch rasters
    on machines that only have the pack (i.e. Windows)."""
    static: np.ndarray                       # (16, 512, 512)
    valid: np.ndarray                        # (512, 512) bool
    lst: dict[tuple[int, int], float]        # (doy, corpus_hour) -> scalar
    mask_verified: bool                      # cross-checked against V4 at build time
    # V7 packs (apply_shadow_v7.py) carry per-hour shade-age planes for slot
    # 10: corpus_hour -> consecutive hours shaded. None on pre-V7 packs.
    age: dict[int, np.ndarray] | None = None

    @classmethod
    def load(cls, path: Path) -> "Donor":
        d = np.load(path)
        lst = {(int(a), int(b)): float(c)
               for a, b, c in zip(d["lst_doys"], d["lst_hours"], d["lst_values"])}
        age = None
        age_keys = [k for k in d.files if k.startswith("shade_age_h")]
        if age_keys:
            age = {int(k[-2:]): d[k].astype(np.float32) for k in age_keys}
        return cls(
            static=d["static"].astype(np.float32),
            valid=d["valid"].astype(bool),
            lst=lst,
            mask_verified=bool(d["mask_verified"]),
            age=age,
        )


def process_patch(patch_dir: Path, out_dir: Path, args: argparse.Namespace,
                  st: Stats) -> None:
    patch = patch_dir.name
    done = json.loads((patch_dir / "DONE.json").read_text())
    bands = done["saved_bands"]

    with rasterio.open(patch_dir / "TMRT_0_0.tif") as s:
        tmrt_all = s.read()

    # (1) the band mapping is read, never re-derived
    if tmrt_all.shape[0] != len(bands):
        st.errors.append(f"{patch}: TMRT has {tmrt_all.shape[0]} bands, "
                         f"DONE.json declares {len(bands)}")
        return

    shadow_all = None
    if args.shadow_channel == "solweig":
        with rasterio.open(patch_dir / "Shadow_0_0.tif") as s:
            shadow_all = s.read()

    met = load_met(patch_dir / "met_day.txt")

    pack: Donor | None = None
    if args.donor_pack:
        # Portable mode: everything comes from the pack plus Stage C's own output.
        # Geo-reference is taken from the Stage C raster, which is always present.
        f = args.donor_pack / f"{patch}.npz"
        if not f.exists():
            st.errors.append(f"{patch}: no donor at {f}")
            return
        pack = Donor.load(f)
        if not pack.mask_verified:
            st.errors.append(f"{patch}: donor mask was NOT verified at build time")
            return
        valid = pack.valid
        lon, lat = patch_lonlat(patch_dir / "TMRT_0_0.tif")
    else:
        lon, lat = patch_lonlat(PATCH_DIR / patch / "building_dsm.tif")
        valid = validity_mask(patch, args.edge_buffer_px)
        if valid is None:
            st.errors.append(f"{patch}: no input DSM at {PATCH_DIR / patch}")
            return

    invalid_frac = 1.0 - valid.mean()
    donor = None   # any V4 npz for this patch supplies the static channels

    for b in bands:
        doy, ch = b["doy"], b["corpus_hour"]
        local_hour, idx = b["local_hour"], b["index_in_file"]
        stem = f"{patch}_d{doy:03d}_h{ch:02d}"

        # (2) trust the recorded local hour rather than recomputing ch+1
        if local_hour != ch + 1:
            st.errors.append(f"{stem}: local_hour {local_hour} != corpus_hour+1")
            continue

        label = tmrt_all[idx].astype(np.float32)

        # (3) the new labels should carry no sentinels; assert rather than assume
        if (label <= -9000).any():
            n = int((label <= -9000).sum())
            st.errors.append(f"{stem}: {n} sentinel pixels in a Stage C label")

        # NaN everywhere: tair_sparse is 100% NaN in 40/40 sampled V4 files, so
        # it is a dead target. Carried through for shape compatibility only.
        tair_sparse = np.full((PATCH_PX, PATCH_PX), np.nan, dtype=np.float32)

        if pack is not None:
            # Portable mode. `legacy` means "this date-hour existed in V4", which
            # is exactly the set for which the pack carries an lst scalar.
            legacy = (doy, ch) in pack.lst
            x = np.zeros((NUM_CHANNELS, PATCH_PX, PATCH_PX), dtype=np.float32)
            for slot, plane in zip(STATIC_SLOTS, pack.static):
                x[slot] = plane
            lst_scalar = pack.lst.get((doy, ch))
        else:
            v4_path = V4_DIR / f"{stem}.npz"
            legacy = v4_path.exists()

            if legacy:
                src = np.load(v4_path)
                donor = donor or src
            elif donor is not None:
                src = donor                   # DOY 351: static channels only
            else:
                cand = sorted(V4_DIR.glob(f"{patch}_d*.npz"))
                if not cand:
                    st.skipped_no_v4 += 1
                    continue
                donor = src = np.load(cand[0])

            x = np.array(src["input"], dtype=np.float32, copy=True)
            if x.shape != (NUM_CHANNELS, PATCH_PX, PATCH_PX):
                st.errors.append(f"{stem}: input shape {x.shape}")
                continue
            tair_sparse = np.array(src["tair_sparse"], dtype=np.float32)
            lst_scalar = float(src["lst"].flat[0]) if legacy else None

            # (4) cross-check our mask against V4's implicit NaN mask. This is
            # the check that would fail if the mask were built wrongly.
            if legacy and args.check_mask:
                v4_finite = np.isfinite(src["tmrt"])
                st.mask_checked += 1
                agree = (v4_finite == valid).mean()
                if agree < args.mask_agreement:
                    st.mask_mismatch += 1
                    st.errors.append(f"{stem}: mask agreement {agree*100:.2f}% "
                                     f"(v4 finite {100*v4_finite.mean():.2f}%, "
                                     f"ours {100*valid.mean():.2f}%)")

        if invalid_frac > args.max_invalid_frac:
            st.skipped_invalid += 1
            continue

        # (5) solar channels — the three that were dead
        alt, az = compute_solar_position(lat, lon, doy, local_hour)
        if args.solar_channels == "computed":
            sin_alt, alt_norm = get_solar_channels(alt)
            x[CH_SIN_ALT] = sin_alt
            x[CH_ALT_NORM] = alt_norm
        else:
            x[CH_SIN_ALT] = 0.0
            x[CH_ALT_NORM] = 0.0

        if args.shadow_channel == "solweig":
            # SOLWEIG: 1 sunlit | 0.03 vegetation | 0 building. Channel 10 is a
            # shadow FRACTION, so invert to 1 = shaded.
            x[CH_SHADOW_FRACTION] = (1.0 - shadow_all[idx]).astype(np.float32)
        elif args.shadow_channel == "computed":
            from app.services.urban_heat_feature_engineering import (
                compute_shadow_fraction,
            )
            dsm_m = x[CH_DSM] * DSM_SCALE
            x[CH_SHADOW_FRACTION] = compute_shadow_fraction(
                dsm_m, sun_altitude_deg=max(alt, 1.0), sun_azimuth_deg=az,
            ).astype(np.float32)
        elif args.shadow_channel == "age":
            # V7: slot 10 = consecutive hours shaded before (and including)
            # this sample's corpus hour, from the donor pack's per-hour
            # planes (apply_shadow_v7.py). Raw hours, like every other raw-
            # unit channel. Fail loudly: silently zero-filling would train
            # V7 on a dead channel while claiming otherwise.
            if pack is None or pack.age is None or ch not in pack.age:
                st.errors.append(
                    f"{stem}: --shadow-channel age but donor pack has no "
                    f"shade_age_h{ch:02d} (need a donors_v7 pack)")
                continue
            x[CH_SHADOW_FRACTION] = pack.age[ch]
        else:
            x[CH_SHADOW_FRACTION] = 0.0

        # (6) weather from the met file that drove the label, at local hour
        w = met.get((doy, local_hour))
        if w is None:
            st.errors.append(f"{stem}: no met row for doy {doy} it {local_hour}")
            continue
        x[CH_ERA5_TAIR] = w["tair"]
        x[CH_ERA5_RH] = w["rh"]
        x[CH_ERA5_WIND] = w["wind"]
        x[CH_ERA5_SSRD] = w["ssrd"]

        # (7) label: NaN where invalid, never mean-filled
        tmrt = np.where(valid, label, np.nan).astype(np.float32)

        # (8) lst: per-patch scalar; masked entirely for the heat date
        if doy == DOY_NO_LST:
            lst = np.full((PATCH_PX, PATCH_PX), np.nan, dtype=np.float32)
            lst_valid = np.uint8(0)
            st.doy351 += 1
        elif legacy and lst_scalar is not None:
            lst = np.where(valid, np.float32(lst_scalar), np.nan).astype(np.float32)
            lst_valid = np.uint8(1)
        else:
            lst = np.full((PATCH_PX, PATCH_PX), np.nan, dtype=np.float32)
            lst_valid = np.uint8(0)

        st.invalid_fracs.append(float(invalid_frac))
        if args.dry_run:
            st.written += 1
            continue

        np.savez_compressed(
            out_dir / f"{stem}.npz",
            input=x,
            tmrt=tmrt,
            lst=lst,
            tair_sparse=tair_sparse,
            valid=valid.astype(np.uint8),
            lst_valid=lst_valid,
            doy=np.int16(doy),
            corpus_hour=np.int16(ch),
            local_hour=np.int16(local_hour),
            solar_altitude=np.float32(alt),
            solar_azimuth=np.float32(az),
            band_index=np.int16(idx),
        )
        st.written += 1

    st.patches += 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage-c-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--donor-pack", type=Path, default=None,
                    help="portable mode: read static channels, validity masks and "
                         "lst scalars from a donor pack instead of the 44 GB V4 "
                         "corpus. Built by build_donor_pack.py; required on Windows.")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true",
                    help="validate everything, write nothing")
    ap.add_argument("--solar-channels", choices=("computed", "zeros"),
                    default="computed",
                    help="'zeros' reproduces the V4 corpus for a clean ablation")
    ap.add_argument("--shadow-channel", choices=("zeros", "solweig", "computed", "age"),
                    default="zeros",
                    help="'solweig' is physically correct but requires an "
                         "inference-side fix first (see module docstring)")
    ap.add_argument("--max-invalid-frac", type=float, default=0.5,
                    help="drop samples whose patch exceeds this invalid fraction")
    ap.add_argument("--edge-buffer-px", type=int, default=0,
                    help="also invalidate pixels within N px of a hole; the "
                         "shadow-contamination width, pending measurement")
    ap.add_argument("--check-mask", action="store_true", default=True)
    ap.add_argument("--mask-agreement", type=float, default=0.999)
    args = ap.parse_args()

    if not args.stage_c_dir.is_dir():
        print(f"no such stage-c dir: {args.stage_c_dir}", file=sys.stderr)
        return 2
    args.out_dir.mkdir(parents=True, exist_ok=True)

    patches = sorted(p for p in args.stage_c_dir.iterdir()
                     if p.is_dir() and (p / "DONE.json").exists())
    if args.limit:
        patches = patches[:args.limit]

    print(f"Stage D - {len(patches)} patches from {args.stage_c_dir}")
    print(f"  solar_channels={args.solar_channels}  shadow_channel={args.shadow_channel}")
    print(f"  utc_offset=+{UTC_OFFSET}  max_invalid_frac={args.max_invalid_frac}"
          f"  edge_buffer_px={args.edge_buffer_px}")
    print(f"  source={'donor pack ' + str(args.donor_pack) if args.donor_pack else 'V4 corpus + patch rasters'}")
    print(f"  out={args.out_dir}{'  [DRY RUN]' if args.dry_run else ''}\n")

    st = Stats()
    for i, p in enumerate(patches, 1):
        try:
            process_patch(p, args.out_dir, args, st)
        except Exception as exc:                      # noqa: BLE001
            st.errors.append(f"{p.name}: {type(exc).__name__}: {exc}")
        if i % 25 == 0:
            print(f"  [{i}/{len(patches)}] written {st.written}, "
                  f"errors {len(st.errors)}")

    st.report()
    return 1 if st.mask_mismatch or st.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
