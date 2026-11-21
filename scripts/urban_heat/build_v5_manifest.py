#!/usr/bin/env python3
"""
Build manifest.parquet for the V5 (Stage D) corpus.
===================================================

`UMAMI_UrbanHeat_V4_Training.ipynb` requires a manifest beside the npz files:

    MANIFEST_PATH = PATCHES_DIR / 'manifest.parquet'
    npz_path = self.patches_dir / Path(row['npz_path']).name

Only the FILENAME of `npz_path` is used and the directory comes from
`PATCHES_DIR`, so this manifest is portable: build it here, ship the parquet, drop
it beside either corpus on any machine.

Rows: 482 patches x 16 date-hours = 7,712.

  * the 12 legacy date-hours are copied VERBATIM from the V4 manifest, so every
    field the notebook reads keeps the exact provenance it already had
  * DOY 351 (the 39.2 C heat day) has no V4 counterpart, so its rows are built:
      - era5_* from the SAME met file that drove the labels, at LOCAL hour
        (corpus_hour + 1 -- SOLWEIG derives +11 and reads met `it` as local)
      - lst_celsius = NaN and has_lst = False, because there is no coincident
        satellite scene. Substituting a nearby one would mislabel the one date
        added for its heat.
      - geometry / centroid / v2_patch_id copied from that patch's legacy rows

`has_lst=False` <=> `lst_celsius` is null, which is V4's own convention (2,386
such rows there, all of DOY 46 and 306).

Verification is built in and every check can fail; see --verify-npz.

Read-only except for the output parquet.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("/data") if Path("/data/training").exists() else Path("/app/urban_heat_data")
V4_MANIFEST = DATA / "training/patches_v4/manifest.parquet"

# SOLWEIG derives +11 (AEDT) for Melbourne and reads met `it` as local time.
UTC_OFFSET_HOURS = 11
NEW_DOY = 351
YEAR = 2024
HOURS_351 = (9, 11, 13, 15)

MET_DOY, MET_HOUR = 1, 2
MET_WIND, MET_RH, MET_TAIR, MET_KDOWN = 9, 10, 11, 14


def load_met(path: Path) -> dict[tuple[int, int], dict[str, float]]:
    out = {}
    for line in path.read_text().splitlines()[1:]:
        f = line.split()
        if len(f) > MET_KDOWN:
            out[(int(f[MET_DOY]), int(f[MET_HOUR]))] = {
                "wind": float(f[MET_WIND]), "rh": float(f[MET_RH]),
                "tair": float(f[MET_TAIR]), "ssrd": float(f[MET_KDOWN]),
            }
    return out


def utc_stamp(doy: int, local_hour: int) -> str:
    """Local (AEDT, +11) -> the UTC string format V4 uses."""
    local = datetime(YEAR, 1, 1, tzinfo=timezone(timedelta(hours=UTC_OFFSET_HOURS)))
    local += timedelta(days=doy - 1, hours=local_hour)
    return local.astimezone(timezone.utc).isoformat()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patch-list", type=Path, required=True)
    ap.add_argument("--met", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--verify-npz", type=Path, default=None,
                    help="a Stage D output dir; cross-checks manifest era5 and "
                         "lst against what is actually inside the npz files")
    args = ap.parse_args()

    patches = json.loads(args.patch_list.read_text())
    met = load_met(args.met)
    v4 = pd.read_parquet(V4_MANIFEST)
    print(f"patches      : {len(patches)}")
    print(f"V4 manifest  : {len(v4)} rows, {len(v4.columns)} columns")

    keep = set(patches)
    legacy = v4[v4["v2_patch_id"].isin(keep)].copy()
    print(f"legacy rows  : {len(legacy)}  (expect {len(patches)*12})")
    if len(legacy) != len(patches) * 12:
        print("  MISMATCH -- some patches are missing V4 rows", file=sys.stderr)

    # --- DOY 351 rows ------------------------------------------------------
    # Per-patch geometry comes from that patch's own legacy rows, so nothing is
    # invented; only the time-varying fields are computed.
    first = legacy.sort_values("patch_id").groupby("v2_patch_id").first()
    new_rows = []
    for patch in patches:
        if patch not in first.index:
            continue
        g = first.loc[patch]
        for hour in HOURS_351:
            local_hour = hour + 1
            w = met.get((NEW_DOY, local_hour))
            if w is None:
                print(f"  no met row for doy {NEW_DOY} it {local_hour}", file=sys.stderr)
                continue
            pid = f"{patch}_d{NEW_DOY:03d}_h{hour:02d}"
            new_rows.append({
                "patch_id": pid,
                "window_col": g["window_col"], "window_row": g["window_row"],
                "patch_w": g["patch_w"], "patch_h": g["patch_h"],
                "timestamp_utc": utc_stamp(NEW_DOY, local_hour),
                "doy": NEW_DOY, "hour": hour,
                "centroid_lon": g["centroid_lon"], "centroid_lat": g["centroid_lat"],
                "era5_tair_c": w["tair"], "era5_rh_pct": w["rh"],
                "era5_wind_speed_ms": w["wind"], "era5_ssrd_wm2": w["ssrd"],
                "lst_celsius": np.nan,          # no coincident scene for 16 Dec
                "has_tmrt": True, "has_lst": False,
                "num_sensors_in_patch": g["num_sensors_in_patch"],
                "sensor_tair_mean_c": g["sensor_tair_mean_c"],
                "v2_patch_id": patch, "year": YEAR,
                "npz_path": f"{pid}.npz",
            })
    new = pd.DataFrame(new_rows)
    print(f"DOY 351 rows : {len(new)}  (expect {len(patches)*4})")

    legacy["npz_path"] = legacy["patch_id"] + ".npz"
    out = pd.concat([legacy, new], ignore_index=True)
    out = out[v4.columns.tolist()]                       # lock column order
    out = out.sort_values("patch_id").reset_index(drop=True)
    for col in ("has_tmrt", "has_lst"):
        out[col] = out[col].astype(bool)

    # --- checks that can fail ---------------------------------------------
    print("\n" + "=" * 62)
    fails = []

    def check(name, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  -- ' + detail) if detail else ''}")
        if not ok:
            fails.append(name)

    check("row count == patches x 16", len(out) == len(patches) * 16,
          f"{len(out)} vs {len(patches)*16}")
    check("patch_id unique", out["patch_id"].is_unique,
          f"{len(out) - out['patch_id'].nunique()} duplicates")
    check("columns match V4 exactly", list(out.columns) == list(v4.columns))
    check("has_lst False exactly where lst_celsius is null",
          bool((out["has_lst"] == out["lst_celsius"].notna()).all()))
    check("every DOY 351 row has has_lst False",
          bool((~out.loc[out.doy == NEW_DOY, "has_lst"]).all()))
    check("has_tmrt True everywhere", bool(out["has_tmrt"].all()))

    # The adversarial one: weather must match the met file at LOCAL hour.
    #
    # That assertion is only MEANINGFUL where the two candidate hours actually
    # differ -- in the afternoon plateau consecutive hours are near-identical
    # (DOY 15: 26.69 at 15:00, 26.83 at 16:00), so the check would pass whichever
    # clock were used. Requiring every row to be decisive is therefore wrong: it
    # tests the weather, not the manifest. Report the decisive fraction instead,
    # and name the combos the check cannot speak to.
    bad_local, decisive, blind = 0, 0, {}
    for _, r in out.iterrows():
        w = met.get((int(r.doy), int(r.hour) + 1))
        c = met.get((int(r.doy), int(r.hour)))
        if w is None:
            continue
        if abs(float(r.era5_tair_c) - w["tair"]) > 0.011:
            bad_local += 1
        if c and abs(w["tair"] - c["tair"]) > 0.5:
            decisive += 1
        elif c:
            blind[(int(r.doy), int(r.hour))] = round(abs(w["tair"] - c["tair"]), 2)

    check("era5_tair matches met at LOCAL hour for every row", bad_local == 0,
          f"{bad_local} rows differ")
    frac = decisive / max(len(out), 1)
    check("the clock check is decisive for most rows", frac >= 0.5,
          f"{decisive}/{len(out)} = {100*frac:.1f}% can distinguish local from corpus hour")
    if blind:
        print(f"         blind combos (consecutive hours within 0.5 C) — the clock")
        print(f"         cannot be confirmed from temperature alone for these:")
        for (d, h), gap in sorted(blind.items()):
            print(f"           doy {d:>3} hour {h:>2}  gap {gap:.2f} C")

    # --- optional: cross-check against real npz files ----------------------
    if args.verify_npz and args.verify_npz.is_dir():
        n_checked = n_bad = 0
        for f in sorted(args.verify_npz.glob("*.npz"))[:60]:
            row = out[out.patch_id == f.stem]
            if row.empty:
                continue
            r = row.iloc[0]
            d = np.load(f)
            n_checked += 1
            if abs(float(d["input"][11].mean()) - float(r.era5_tair_c)) > 0.011:
                n_bad += 1
            elif int(d["doy"]) != int(r.doy) or int(d["corpus_hour"]) != int(r.hour):
                n_bad += 1
        check(f"manifest agrees with {n_checked} real npz files", n_bad == 0,
              f"{n_bad} disagree")
    else:
        print("  [SKIP] npz cross-check -- pass --verify-npz <stage_d out dir>")

    print("=" * 62)
    if fails:
        print(f"{len(fails)} FAILED, not writing")
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    size = args.out.stat().st_size
    print(f"\nwrote {args.out}  ({len(out)} rows, {size/1e6:.2f} MB)")
    print(f"  has_lst True : {int(out['has_lst'].sum())} / {len(out)}")
    print(f"  by doy       : {out.groupby('doy').size().to_dict()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
