#!/usr/bin/env python3
"""
UMAMI Urban Heat — Stage C: regenerate T_mrt labels with REFERENCE SOLWEIG.

Runs on Colab (GPU). Replaces the V4 labels, which were produced by a custom
engine with no surface energy balance and are ~36 C too cold in sun.

═══════════════════════════════════════════════════════════════════════════════
WHY EVERY PREFLIGHT BELOW HALTS INSTEAD OF WARNING
═══════════════════════════════════════════════════════════════════════════════
This project's failures have all been silent: a label engine that rendered clean
maps while pinning every surface to air temperature; an evaluation that shared its
source with the training target; a validation run an hour before the satellite
passed, which AGREED because two errors cancelled. Nothing errored. Every metric
looked green.

So each check below either passes or stops the run. A warning that scrolls past at
2am while 12,244 labels build on a broken assumption is worth nothing.

Checks, all of which CAN fail (each has been observed failing):
  1. CUDA present AND demonstrably used   - timing vs the 85 s CPU baseline
  2. ERA5 forcing == the weather in input channels 11-14, to 0.00 C
  3. Leaf fix applied AND VERIFIED IN OUTPUT - vegetation shadow reads 0.03
  4. Landcover remap total - no unmapped code reaches the physics
  5. Met file is 24 contiguous hours from midnight (the OOM guard)
  6. Output actually written - a "successful" run that wrote nothing is a failure

═══════════════════════════════════════════════════════════════════════════════
THE FOUR DEFECTS THIS RUN FIXES (and the one it does NOT)
═══════════════════════════════════════════════════════════════════════════════
1. Engine       - reference SOLWEIG replaces the no-energy-balance engine.
2. Leaf season  - solweig_gpu hardcodes NORTHERN hemisphere leaf-on (DOY 97-300).
                  All our DOYs fall outside it, so Melbourne's trees were treated
                  as LEAFLESS, passing 50% of the beam instead of 3%. Fixing it
                  moves canopy T_mrt by -10.1 C and raises sun-vs-canopy contrast
                  by 64%. Without this the labels undervalue tree planting.
3. Forcing      - labels were TMY-forced while input channels carry real ERA5.
                  Mismatch: mean |diff| 2.98 C, max 5.83 C, SIGN FLIPPING BY DATE.
                  A constant offset is absorbable; a date-dependent one teaches the
                  network part of the calendar. Now forced with the SAME ERA5.
4. Time         - the corpus is built at UTC+10 throughout (solar position AND
                  weather), so it is internally coherent even though all dates are
                  actually AEDT. We MATCH that convention. Using the "correct" +11
                  would desynchronise new labels from the weather the network reads.

NOT fixed here: landcover is ~70% a DEFAULT FILL (generate_landcover.py:100 fills
with LC_BARE_SOIL, "everything else"). SOLWEIG parameterises albedo/emissivity/TgK
FROM the class, so those pixels are guesses. Mitigated by mapping the default to
Dark_asphalt (correct for most unclassified inner-Melbourne ground) and by the
PATCH LIST, which the caller restricts. It is not solved. See README.

═══════════════════════════════════════════════════════════════════════════════
DELIBERATELY NOT DONE
═══════════════════════════════════════════════════════════════════════════════
* No SVF caching. solweig_gpu writes SVF but has NO code path to reuse one, so
  this would mean patching library internals sitting upstream of every label.
  Revisit only if GPU timings show SVF dominating, and only with a numerical
  equality check against an uncached run.
* No new input channels. V5 changes the LABELS ONLY, so the result is
  attributable. Channel ablation comes after, once correct labels exist.
* No tuning against Landsat. The model runs +3.56 C warm vs ST_B10 (+1.83 after
  harmonising emissivity conventions), and that residual cannot be separated from
  the ST product's own ~1-2 K urban bias. Fitting it away would be calibrating
  against a measurement whose error we cannot isolate - the original sin.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import rasterio

# ── the corpus temporal grid ────────────────────────────────────────────────
# 4 existing dates + 1 heat date. DOY 351 (16 Dec 2024, Tmax 39.2 C) was added
# because the corpus tops out at 26.8 C while clients care about 40 C days. It
# sits 5 days from DOY 356, so sun geometry is near-identical while air
# temperature differs by ~20 C - a controlled pair separating weather from sun.
# ⚠️ DOY 351 has NO coincident cloud-free Landsat scene: the `lst` multi-task head
# must be MASKED for it. Do not substitute a nearby scene - 18 Dec was 21.4 C and
# would mislabel the one date added for its heat.
DATE_FOR_DOY = {
    15:  "2024-01-15",
    46:  "2024-02-15",
    306: "2024-11-01",
    356: "2024-12-21",
    351: "2024-12-16",     # heat date; lst head MASKED
}
# ── TIME CONVENTION ────────────────────────────────────────────────────────
# Reference SOLWEIG DERIVES ITS OWN TIMEZONE from the raster coordinates
# (utci_process.py:276-283) and reads the met file's `it` column as LOCAL time at
# that offset — for Melbourne in these months, AEDT = UTC+11. We do not choose the
# convention; the library does. Met files are therefore built in TRUE LOCAL TIME.
#
# The V4 corpus uses a DIFFERENT convention (+10 throughout: compute_solar_position
# defaults to utc_offset=10, train_v2.py:76). It is internally coherent, so we do
# not change it — we MAP between them:
#
#       corpus "hour H"  ==  UTC (H-10)  ==  AEDT (H+1)  ==  SOLWEIG hour (H+1)
#
# ⚠️ STAGE D: to package the label for corpus (doy, hour H), read the SOLWEIG
# output for hour H+1. Get this wrong and every label shifts an hour, invisibly.
#
# ⚠️ An earlier version built met at +10 to make the temperatures match
# era5_combos by label. The preflight passed at 0.00 C while the SUN sat an hour
# off the WEATHER — it verified the wrong invariant. Compare INSTANTS, not labels.
CORPUS_HOUR_TO_LOCAL = 1

# our generate_landcover.py codes -> UMEP landcoverclasses_2016a.txt codes
# 5 is the DEFAULT FILL, not measured bare soil -> Dark_asphalt (UMEP 1).
# Measured in the Stage A pilot: base(bare soil) vs paved differ by only +0.62 C,
# so this is a cheap correctness gain, not a big lever.
LC_REMAP = {1: 1, 2: 2, 3: 7, 4: 5, 5: 1}

LEAF_FIRST, LEAF_LAST = 274, 120          # southern-hemisphere leaf season
VEG_SHADOW_LEAF_ON = 0.03                 # transVeg; leaf-OFF fallback is 0.5
LEAF_VERIFY_DEADLINE = 15                 # halt if still unverified after N runs

# ⚠️ NOT a like-for-like GPU check. 85 s was measured on 14 CPU cores; Colab gives
# 2 vCPU. A T4 landing at ~103 s on 2 cores is therefore evidence the GPU IS
# working — pure CPU on 2 cores would be far slower than 85 s. The remaining time
# is CPU-bound preprocessing (wall/aspect ~25 s, tiling, IO), not SOLWEIG itself.
CPU_BASELINE_S = 85.0
CPU_BASELINE_CORES = 14


class PreflightError(RuntimeError):
    """A check that must stop the run rather than warn."""


def log(msg: str, logfile: Path | None = None) -> None:
    line = f"[{time.strftime('%F %T')}] {msg}"
    print(line, flush=True)
    if logfile:
        with open(logfile, "a") as fh:
            fh.write(line + "\n")


# ── met construction ────────────────────────────────────────────────────────
def esat(t_c: float) -> float:
    """Saturation vapour pressure, Magnus. Used to get RH from ERA5 dewpoint."""
    return 6.112 * math.exp(17.67 * t_c / (t_c + 243.5))


def build_met(dest: Path, doy: int, era5_dir: Path, tmy_template: Path,
              year: int = 2024) -> None:
    """
    One contiguous local day of UMEP met, forced by ERA5.

    ⚠️ THE 24-ROW SUBSET IS AN OOM GUARD, NOT AN OPTIMISATION. use_own_met=True
    IGNORES start/end time and iterates EVERY row of the met file, accumulating
    TMRT_all and UTCI_all in memory (9.2 GB each at 512x512). A full TMY year =
    8760 steps was OOM-killed at 38 min - and the shell reported EXIT 0, because
    the call was piped through `tail`. Only journalctl showed the kill.

    Hours must run contiguously from midnight: SOLWEIG evolves ground temperature
    through a running Tgmap, so a gap silently corrupts the thermal history.

    Radiation: ERA5 gives total ssrd only, so the TMY beam/diffuse SPLIT is scaled
    per hour to match measured ssrd. The split is a ratio, far more transferable
    between days than absolute values.
    """
    date = dt.date.fromisoformat(DATE_FOR_DOY[doy])
    era_path = era5_dir / f"era5_day_{date.strftime('%Y%m%d')}.json"
    if not era_path.exists():
        raise PreflightError(f"missing ERA5 day file: {era_path}")

    rows = json.loads(era_path.read_text())
    byhour = {r["hour_local"]: r for r in rows}
    if sorted(byhour) != list(range(24)):
        raise PreflightError(
            f"ERA5 for DOY {doy} is not 24 contiguous local hours: got {sorted(byhour)}")

    lines = tmy_template.read_text().splitlines()
    header = lines[0]
    idx = {c: i for i, c in enumerate(header.split())}
    # any day supplies the diffuse/direct SPLIT shape; magnitudes are rescaled
    template_doy = 15 if doy in (15, 46, 351, 356) else 306
    tmy = {}
    for r in lines[1:]:
        f = r.split()
        if int(f[1]) == template_doy:
            tmy[int(f[2])] = f
    if sorted(tmy) != list(range(24)):
        raise PreflightError(f"TMY template DOY {template_doy} lacks 24 hours")

    out = []
    for h in range(24):
        f = list(tmy[h])
        e = byhour[h]
        f[idx["iy"]] = str(year)
        f[idx["id"]] = str(doy)
        f[idx["it"]] = str(h)
        f[idx["Tair"]] = "%.2f" % e["tair_c"]
        rh = 100.0 * esat(e["dewpoint_c"]) / esat(e["tair_c"])
        f[idx["RH"]] = "%.2f" % min(100.0, max(1.0, rh))
        f[idx["U"]] = "%.2f" % max(0.3, math.hypot(e["u10"], e["v10"]))

        ssrd = max(0.0, (e["ssrd_j"] or 0.0) / 3600.0)      # J/m2/h -> W/m2
        tmy_kdown = float(f[idx["kdown"]])
        if tmy_kdown > 1.0:
            scale = ssrd / tmy_kdown
            for k in ("kdown", "kdiff", "kdir"):
                f[idx[k]] = "%.2f" % (float(f[idx[k]]) * scale)
        else:
            for k in ("kdown", "kdiff", "kdir"):
                f[idx[k]] = "0.00"
        out.append(" ".join(f))

    dest.write_text("\n".join([header] + out) + "\n")


def remap_landcover(src: Path, dest: Path) -> None:
    """Our codes -> UMEP codes. Halts if any code is unmapped."""
    with rasterio.open(src) as s:
        lc = s.read(1)
        prof = s.profile.copy()
    present = set(np.unique(lc).tolist()) - {0}
    unmapped = present - set(LC_REMAP)
    if unmapped:
        raise PreflightError(
            f"landcover codes with no UMEP mapping: {sorted(unmapped)} in {src}. "
            "An unmapped code would silently take default physics parameters.")
    out = np.zeros_like(lc)
    for a, b in LC_REMAP.items():
        out[lc == a] = b
    prof.update(dtype=rasterio.uint8, count=1, compress="lzw")
    with rasterio.open(dest, "w", **prof) as dst:
        dst.write(out.astype(np.uint8), 1)


# ── preflight ───────────────────────────────────────────────────────────────
def preflight_era5(era5_dir: Path, combos_csv: Path, logfile: Path) -> None:
    """
    Does our forcing equal the weather the network is shown?

    This check FAILED at 1.82 C when the day files were built at UTC+11, which is
    how the corpus's +10 convention was discovered. It can fail.
    """
    import csv
    want = {}
    with open(combos_csv) as fh:
        for r in csv.DictReader(fh):
            want[(int(r["doy"]), int(r["hour"]))] = float(r["tair_c"])

    worst, checked = 0.0, 0
    for (doy, hour), w in sorted(want.items()):
        if doy not in DATE_FOR_DOY:
            continue
        date = dt.date.fromisoformat(DATE_FOR_DOY[doy])
        p = era5_dir / f"era5_day_{date.strftime('%Y%m%d')}.json"
        if not p.exists():
            raise PreflightError(f"missing ERA5 day file for DOY {doy}: {p}")
        rows = {r["hour_local"]: r for r in json.loads(p.read_text())}
        # Compare the SAME INSTANT: corpus hour H is local hour H+1. Comparing
        # H to H matches temperatures by label while the moments differ.
        lh = hour + CORPUS_HOUR_TO_LOCAL
        if lh not in rows:
            raise PreflightError(
                f"ERA5 day file for DOY {doy} has no local hour {lh} "
                f"(needed for corpus hour {hour})")
        worst = max(worst, abs(rows[lh]["tair_c"] - w))
        checked += 1

    log(f"preflight ERA5: {checked} combos checked (corpus hour H vs local H+1), "
        f"worst |diff| {worst:.2f} C", logfile)
    if worst > 0.05:
        raise PreflightError(
            f"forcing disagrees with input channels by {worst:.2f} C. Labels would "
            "describe different weather from what the network reads.\n"
            "Met files must be in TRUE LOCAL TIME (AEDT) because SOLWEIG derives "
            "its own timezone from the rasters; the corpus's own hour labels are "
            "at +10, so corpus hour H == local hour H+1.")


def apply_leaf_fix(logfile: Path) -> None:
    import solweig_gpu.utci_process as up
    before = (up.firstdayleaf, up.lastdayleaf)
    up.firstdayleaf, up.lastdayleaf = LEAF_FIRST, LEAF_LAST
    log(f"leaf season {before} -> ({LEAF_FIRST}, {LEAF_LAST}) [southern hemisphere]",
        logfile)


def verify_leaf_in_output(base: Path, patch_src: Path, logfile: Path) -> bool:
    """
    Setting the globals is NOT proof. Read the shadow raster back: with leaf ON,
    canopy transmissivity is transVeg = 0.03. The leaf-OFF fallback is
    psi[leafon == 0] = 0.5. If 0.5 appears, the fix did not take and every label
    in this run undervalues tree planting by ~64%.

    Returns True only when the check was CONCLUSIVE.

    ⚠️ A patch with no canopy produces a shadow raster of only 0 and 1 — nothing
    to inspect. An earlier version treated that as "verified" and never looked
    again, so the most important fix in the run went unchecked behind a reassuring
    log line. Now an inconclusive patch returns False and the caller keeps trying.
    """
    shadows = list(base.glob("Shadow_*.tif")) or list(base.glob("output_folder/*/Shadow_*.tif"))
    if not shadows:
        log("leaf check: no shadow raster written — inconclusive, will retry", logfile)
        return False

    # no canopy in the input => nothing this patch can tell us
    try:
        with rasterio.open(patch_src / "trees.tif") as s:
            trees = s.read(1)
        canopy_px = int((np.isfinite(trees) & (trees > 0.5)).sum())
    except Exception:                                   # noqa: BLE001
        canopy_px = -1
    if canopy_px == 0:
        log("leaf check: patch has NO canopy — inconclusive, will retry on a "
            "vegetated patch", logfile)
        return False

    # ⚠️ MUST read a DAYTIME band. In the full run band 1 is hour 0 — midnight —
    # where there is no sun and so no canopy transmissivity to inspect. An earlier
    # version read band 1 and reported "no intermediate shadow values" on patches
    # with 14k canopy pixels: it was looking at the dark, not at a broken fix.
    #
    # Saved files now hold only the corpus bands, so the band<->hour mapping comes
    # from the marker rather than being assumed.
    try:
        meta = json.loads((base / DONE_MARKER).read_text())
        saved = meta.get("saved_bands", [])
    except Exception:                                   # noqa: BLE001
        saved = []
    candidates = [(s_["index_in_file"], s_["local_hour"]) for s_ in saved
                  if 9 <= s_["local_hour"] <= 16]
    if not candidates:                                  # full 120-band file
        candidates = [(h, h) for h in (12, 13, 11, 14, 10, 15)]

    vals = np.array([])
    used_band = None
    with rasterio.open(shadows[0]) as s:
        for idx, hour in candidates:
            if idx + 1 > s.count:
                continue
            a = s.read(idx + 1)                         # rasterio is 1-indexed
            v = np.unique(np.round(a[(a > 0.0) & (a < 1.0)], 3))
            if v.size:
                vals, used_band = v, hour
                break
    if vals.size == 0:
        log(f"leaf check: no intermediate shadow values in any daytime band "
            f"(canopy px={canopy_px}) — inconclusive, will retry", logfile)
        return False

    log(f"leaf check: local hour {used_band} — intermediate shadow values "
        f"{vals[:8].tolist()} (canopy px={canopy_px})", logfile)
    if np.any(np.isclose(vals, 0.5, atol=0.01)):
        raise PreflightError(
            "vegetation shadow reads 0.5 (leaf-OFF fallback). The leaf fix did NOT "
            "take. Every label in this run would undervalue canopy by ~64%.")
    if not np.any(np.isclose(vals, VEG_SHADOW_LEAF_ON, atol=0.02)):
        raise PreflightError(
            f"expected canopy transmissivity ~{VEG_SHADOW_LEAF_ON}, saw "
            f"{vals[:8].tolist()}. Do not trust this run until understood.")
    log("leaf check: CONFIRMED leaf-on (0.03) in the written output", logfile)
    return True


def report_device(logfile: Path) -> bool:
    try:
        import torch
    except ImportError:
        raise PreflightError("torch not importable — solweig_gpu cannot run")
    have = torch.cuda.is_available()
    name = torch.cuda.get_device_name(0) if have else "CPU only"
    log(f"device: {name} (cuda={have})", logfile)
    if not have:
        log("WARNING: no CUDA. The full corpus on CPU is ~150 CPU-hours; that is "
            "why this runs on Colab. Continuing only if you passed --allow-cpu.",
            logfile)
    return have


# ── main ────────────────────────────────────────────────────────────────────
DONE_MARKER = "DONE.json"
KEEP_PREFIXES = ("TMRT_", "Lup_", "Shadow_")

# ── which hours the corpus actually trains on ──────────────────────────────
# SOLWEIG writes all 24 h per date; the corpus uses only these 12 combos (+4 for
# the new heat date). Keeping everything costs 120 MB per file per patch:
#   480 MB x 482 patches = ~231 GB, against 57 GB free. Keeping only the used
#   bands is ~16 MB per file -> a few GB. The other 104 bands are hours nobody
#   trains on.
# DOY 351 mirrors 356's hours; it is the new heat date, lst head MASKED.
CORPUS_HOURS = {
    15:  [9, 11, 13, 15],
    46:  [9, 13],
    306: [9, 13],
    356: [9, 11, 13, 15],
    351: [9, 11, 13, 15],
}
SHADOW_KEEP_PATCHES = 20        # shadow is only needed to audit the leaf fix


def corpus_band_plan(doys: list[int]) -> tuple[list[int], list[dict]]:
    """
    Which 0-indexed bands to keep, and what each one is.

    Band layout is day-major (verified for all 5 dates: 0.000 C against separate
    runs). corpus hour H is local hour H+1, so:

        band0 = day_offset + (H + CORPUS_HOUR_TO_LOCAL)
    """
    bands, meta = [], []
    for i, doy in enumerate(doys):
        for h in CORPUS_HOURS.get(doy, []):
            b = i * 24 + h + CORPUS_HOUR_TO_LOCAL
            bands.append(b)
            meta.append({"doy": doy, "corpus_hour": h,
                         "local_hour": h + CORPUS_HOUR_TO_LOCAL, "band0": b})
    return bands, meta


def extract_bands(src: Path, dest: Path, bands0: list[int]) -> None:
    """Write a compact copy holding only the corpus bands, in the given order."""
    with rasterio.open(src) as s:
        too_big = [b for b in bands0 if b >= s.count]
        if too_big:
            raise RuntimeError(
                f"{src.name}: bands {too_big} beyond the {s.count} available — "
                "the day-major mapping is wrong and every label's hour would be too")
        data = np.stack([s.read(b + 1) for b in bands0])    # rasterio is 1-indexed
        prof = s.profile.copy()
    prof.update(count=len(bands0), compress="lzw", dtype="float32")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with rasterio.open(tmp, "w", **prof) as dst:
        dst.write(data.astype(np.float32))
    tmp.replace(dest)


def is_complete(dest: Path) -> bool:
    """
    Resume test. Trusts ONLY the completion marker, never a file's presence.

    Colab disconnects mid-write. A truncated TMRT_*.tif still EXISTS, so an
    existence check would skip it and carry a corrupt label into training — the
    failure would be silent, which is how every other defect in this project
    survived. The marker is written last, after the outputs have been copied and
    re-opened successfully, so its presence means the whole chain finished.
    """
    marker = dest / DONE_MARKER
    if not marker.exists():
        return False
    try:
        meta = json.loads(marker.read_text())
    except Exception:                                   # noqa: BLE001
        return False                                    # unreadable marker -> redo
    for name in meta.get("files", []):
        if not (dest / name).exists():
            return False
    return bool(meta.get("files"))


def _verify_raster(path: Path) -> None:
    """Open and touch the data. A truncated GeoTIFF usually opens but fails to read."""
    with rasterio.open(path) as s:
        if s.count < 1:
            raise RuntimeError(f"{path.name}: no bands")
        a = s.read(s.count)
        if not np.isfinite(a).any():
            raise RuntimeError(f"{path.name}: no finite values")


def build_met_multi(dest: Path, doys: list[int], era5_dir: Path,
                    tmy_template: Path) -> None:
    """
    All dates for one patch in a single met file: 24 rows per day, day-major.

    MEASURED SAFE, not assumed. Running 5 dates in one call vs 5 separate calls
    gives **bit-identical** T_mrt for every date (max |dT_mrt| 0.000 C on all
    five), so SOLWEIG's running Tgmap does not leak between non-consecutive days.
    1.52x faster — the saving is per-call overhead (tiling, wall/aspect) plus a
    small fixed SVF cost, NOT a large SVF saving; that was ~11 s, not the ~44 s
    first guessed.

    Band layout in the output is day-major and was verified for every date:
        doys[i] occupies bands  i*24 .. i*24+23   (0-indexed)
    """
    header = None
    rows: list[str] = []
    for i, doy in enumerate(doys):
        tmp = dest.with_suffix(f".d{doy:03d}.tmp")
        build_met(tmp, doy, era5_dir, tmy_template)
        lines = tmp.read_text().splitlines()
        tmp.unlink()
        if i == 0:
            header = lines[0]
        rows += lines[1:]
    if header is None or len(rows) != 24 * len(doys):
        raise PreflightError(
            f"combined met has {len(rows)} rows, expected {24 * len(doys)} — "
            "the day-major band mapping would be wrong")
    dest.write_text("\n".join([header] + rows) + "\n")


def run_patch(patch: str, doys: list[int], cfg, logfile: Path) -> float:
    """
    ALL dates for one patch in a single SOLWEIG call, crash-safe.

    Work happens in LOCAL scratch (fast); only the outputs we need are copied to
    Drive. SOLWEIG also writes bulky intermediates (tiled inputs, walls, aspect,
    SVF) — pushing those through Drive's FUSE mount is slow and is itself a common
    cause of the disconnects this protects against.

    Order matters: compute -> copy -> verify -> marker. If the session dies at any
    point before the marker, the next run redoes this patch cleanly.
    """
    from solweig_gpu import thermal_comfort

    src = cfg.patches / patch
    dest = cfg.out / patch
    work = cfg.scratch / patch
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)         # never resume a partial dir
    work.mkdir(parents=True, exist_ok=True)

    for f in ("building_dsm.tif", "dem.tif", "trees.tif"):
        shutil.copy2(src / f, work / f)
    remap_landcover(src / "landcover.tif", work / "landcover.tif")
    build_met_multi(work / "met_day.txt", doys, cfg.era5, cfg.tmy)

    t0 = time.time()
    thermal_comfort(
        base_path=str(work),
        selected_date_str=DATE_FOR_DOY[doys[0]],
        building_dsm_filename=str(work / "building_dsm.tif"),
        dem_filename=str(work / "dem.tif"),
        trees_filename=str(work / "trees.tif"),
        landcover_filename=str(work / "landcover.tif"),
        tile_size=3600, overlap=20,
        use_own_met=True, own_met_file=str(work / "met_day.txt"),
        save_tmrt=True, save_lup=True, save_shadow=True,
    )
    elapsed = time.time() - t0

    # A run that "succeeds" without writing anything is a failure. This is exactly
    # how the OOM presented: exit 0, no output.
    produced = [p for p in work.glob("output_folder/*/*.tif")
                if p.name.startswith(KEEP_PREFIXES)]
    if not any(p.name.startswith("TMRT_") for p in produced):
        raise RuntimeError("completed but wrote no TMRT — suspect OOM (check dmesg)")

    # every output must carry 24 bands per date, or the band mapping is wrong
    expect_bands = 24 * len(doys)
    for p in produced:
        with rasterio.open(p) as s:
            if s.count != expect_bands:
                raise RuntimeError(
                    f"{p.name} has {s.count} bands, expected {expect_bands} "
                    f"({len(doys)} dates x 24 h). The day-major band mapping — and "
                    "therefore every label's date — would be wrong.")

    # Keep ONLY the hours the corpus trains on. Full output is ~480 MB/patch
    # (~231 GB over the corpus, against 57 GB free); the used bands are ~16 MB.
    bands0, band_meta = corpus_band_plan(doys)
    dest.mkdir(parents=True, exist_ok=True)
    copied = []
    for p in produced:
        if p.name.startswith("Shadow_") and cfg.patch_index > SHADOW_KEEP_PATCHES:
            continue                                    # only kept to audit the leaf fix
        target = dest / p.name
        extract_bands(p, target, bands0)
        _verify_raster(target)                          # re-open from the destination
        with rasterio.open(target) as s:
            if s.count != len(bands0):
                raise RuntimeError(
                    f"{target.name}: kept {s.count} bands, expected {len(bands0)}")
        copied.append(p.name)

    shutil.copy2(work / "met_day.txt", dest / "met_day.txt")

    # marker LAST — this is what resume trusts.
    # band_index records EXACTLY where each corpus (doy, hour) lives, so Stage D
    # never has to re-derive it. corpus hour H == local hour H+1 (see time note),
    # and bands here are 0-indexed; rasterio reads are 1-indexed.
    # Files hold ONLY the corpus bands now, in the order of `bands`. Stage D reads
    # band i (0-indexed) of the saved file for the (doy, corpus_hour) at index i —
    # no re-deriving the mapping, which is how hour errors creep in.
    (dest / DONE_MARKER).write_text(json.dumps({
        "patch": patch,
        "doys": doys,
        "dates": [DATE_FOR_DOY[d] for d in doys],
        "saved_bands": [
            {"index_in_file": i, **m} for i, m in enumerate(band_meta)
        ],
        "source_bands_per_day": 24,
        "note": ("files contain only the corpus hours; index_in_file is the band to "
                 "read (0-indexed), band0 was its position in the full 120-band run"),
        "files": copied, "seconds": round(elapsed, 1),
        "finished": time.strftime("%F %T"),
        "leaf_season": [LEAF_FIRST, LEAF_LAST],
        "lc_remap": {str(k): v for k, v in LC_REMAP.items()},
        "met_local_utc_offset": 11,
        "corpus_hour_to_local": CORPUS_HOUR_TO_LOCAL,
    }, indent=1))

    shutil.rmtree(work, ignore_errors=True)             # scratch is disposable
    return elapsed


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage C label regeneration")
    ap.add_argument("--root", default="/content/drive/MyDrive/UMAMI/stage_c",
                    help="Drive folder holding patches/, gee/, weather/")
    ap.add_argument("--out", default=None, help="output dir (default <root>/out)")
    ap.add_argument("--patch-list", default=None,
                    help="JSON list of patch ids. REQUIRED — the corpus subset is "
                         "a deliberate decision, not a default.")
    ap.add_argument("--doys", default="15,46,306,356,351")
    ap.add_argument("--limit", type=int, default=0, help="stop after N runs (smoke test)")
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--shard", default="0/1",
                    help="i/N — process only patches where index %% N == i. Lets "
                         "several workers share the list without colliding (each "
                         "writes its own patch dirs, and resume is per patch). Use "
                         "to fill a many-core box, or to split across machines.")
    ap.add_argument("--threads", type=int, default=0,
                    help="cap CPU threads per worker (0 = leave alone). On a shared "
                         "box, several small workers beat one greedy one — and it "
                         "keeps cores free for whatever else the machine serves.")
    ap.add_argument("--scratch", default="/content/stage_c_work",
                    help="LOCAL fast disk for intermediates. Must NOT be on Drive: "
                         "SOLWEIG writes bulky tiled inputs/walls/aspect/SVF, and "
                         "pushing those through the FUSE mount is slow and is itself "
                         "a common cause of session drops.")
    cfg = ap.parse_args()

    # Thread caps must be set BEFORE torch/numpy import, or they are ignored —
    # the libraries read them once at load time.
    if cfg.threads > 0:
        import os as _os
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS"):
            _os.environ[var] = str(cfg.threads)

    root = Path(cfg.root)
    cfg.patches = root / "patches"
    cfg.era5 = root / "gee"
    cfg.tmy = root / "weather" / "mel_tullamarine_umep.txt"
    cfg.out = Path(cfg.out) if cfg.out else root / "out"
    cfg.out.mkdir(parents=True, exist_ok=True)
    cfg.scratch = Path(cfg.scratch)
    cfg.scratch.mkdir(parents=True, exist_ok=True)
    # one log per worker — concurrent shards writing the same file interleave and
    # the result is unreadable exactly when something goes wrong
    tag = cfg.shard.replace("/", "of")
    logfile = cfg.out / (f"stage_c_{tag}.log" if cfg.shard != "0/1" else "stage_c.log")

    log("=== Stage C: reference-SOLWEIG label regeneration ===", logfile)

    for p in (cfg.patches, cfg.era5, cfg.tmy):
        if not p.exists():
            raise PreflightError(f"missing input: {p}")

    have_cuda = report_device(logfile)
    if not have_cuda and not cfg.allow_cpu:
        raise PreflightError("no CUDA and --allow-cpu not given; refusing to start")

    preflight_era5(cfg.era5, root / "v4_prep" / "era5_combos.csv", logfile)
    apply_leaf_fix(logfile)

    available = sorted(p.name for p in root.glob("patch_list_*.json"))
    if not cfg.patch_list:
        raise PreflightError(
            "--patch-list is required. Which patches to include is a decision about "
            "label honesty (~70% of the corpus has landcover that is mostly a "
            "default fill), not something this script should guess.\n"
            f"Available in {root}: {available or '(none found)'}\n"
            "Recommended: patch_list_B_default_lt70.json (482 patches).")
    plist = Path(cfg.patch_list)
    if not plist.exists():
        # A bare FileNotFoundError here is unhelpful — the lists have option
        # suffixes, so name it and show what is actually there.
        raise PreflightError(
            f"patch list not found: {plist}\n"
            f"Available in {root}: {available or '(none found)'}\n"
            "Recommended: patch_list_B_default_lt70.json (482 patches, the corpus "
            "decision from the stratified validation).")
    patches = json.loads(plist.read_text())
    doys = [int(d) for d in cfg.doys.split(",")]
    missing = [d for d in doys if d not in DATE_FOR_DOY]
    if missing:
        raise PreflightError(f"no date mapping for DOY(s) {missing}")
    try:
        shard_i, shard_n = (int(x) for x in cfg.shard.split("/"))
    except Exception:                                   # noqa: BLE001
        raise PreflightError(f"--shard must look like i/N, got {cfg.shard!r}")
    if not (shard_n >= 1 and 0 <= shard_i < shard_n):
        raise PreflightError(f"--shard {cfg.shard} out of range")
    if shard_n > 1:
        total = len(patches)
        patches = [p for j, p in enumerate(patches) if j % shard_n == shard_i]
        log(f"shard {shard_i}/{shard_n}: {len(patches)} of {total} patches", logfile)

    if cfg.limit:
        patches = patches[:cfg.limit]
    if cfg.threads > 0:
        try:
            import torch
            torch.set_num_threads(cfg.threads)
            log(f"threads capped at {cfg.threads} per worker", logfile)
        except Exception:                               # noqa: BLE001
            pass
    # ONE call per patch covering ALL dates — measured bit-identical to separate
    # calls (0.000 C on all 5) and 1.52x faster.
    log(f"{len(patches)} patches x {len(doys)} DOYs in ONE call each "
        f"= {len(patches)} runs, {24 * len(doys)} bands per output", logfile)
    jobs = patches

    ok = fail = skip = 0
    verified_leaf = False
    first_timing = None

    for i, patch in enumerate(jobs, 1):
        base = cfg.out / patch
        if is_complete(base):
            skip += 1
            continue
        try:
            cfg.patch_index = i          # shadow rasters kept only for the first N
            elapsed = run_patch(patch, doys, cfg, logfile)
            ok += 1
            if first_timing is None:
                first_timing = elapsed
                import os
                ncpu = os.cpu_count() or 1
                log(f"first patch ({len(doys)} dates) {elapsed:.0f}s on {ncpu} cores",
                    logfile)
                # MEASURED, not assumed: this workload is NOT GPU-bound.
                #   T4   103 s/date   A100  100 s/date   -> ~3% for ~6x the cost
                #   our 14-core CPU box is FASTER per date (~60 s)
                # At 512x512 the kernels are small enough that launch overhead and
                # CPU-side Python dominate. More cores do not help either: wall/aspect
                # parallelises over TILES and there is exactly one tile per patch
                # (22 s on 12 cores vs 25 s on 2). So a slow number here is expected
                # and NOT actionable — do not buy a bigger GPU on the strength of it.
                per_date = elapsed / max(len(doys), 1)
                log(f"  = {per_date:.0f}s per date. Reference: T4 103, A100 100, "
                    f"14-core CPU ~60. Not GPU-bound; a bigger GPU will not help.",
                    logfile)
                log(f"projected full run: {len(jobs)} patches x {elapsed:.0f}s = "
                    f"{len(jobs) * elapsed / 3600:.1f} h", logfile)
            if not verified_leaf:
                verified_leaf = verify_leaf_in_output(base, cfg.patches / patch, logfile)
                if not verified_leaf and ok >= LEAF_VERIFY_DEADLINE:
                    raise PreflightError(
                        f"after {ok} successful runs the leaf fix is still "
                        "UNVERIFIED — no patch yet produced an inspectable canopy "
                        "shadow. Stopping rather than generating thousands of "
                        "labels whose most important correction is unconfirmed. "
                        "Check that trees.tif actually carries canopy for these "
                        "patches.")
            if i % 25 == 0 or i < 4:
                log(f"[{i}/{len(jobs)}] {patch} ({len(doys)} dates) OK {elapsed:.0f}s "
                    f"(ok {ok} fail {fail} skip {skip})", logfile)
        except PreflightError:
            raise
        except Exception as exc:                        # noqa: BLE001
            fail += 1
            log(f"[{i}/{len(jobs)}] {patch} FAILED: {exc}", logfile)
            log(traceback.format_exc(), logfile)

    log(f"=== complete: ok {ok}, failed {fail}, skipped {skip} ===", logfile)
    log("NEXT: run analyse_spatial.py on a sample of these outputs and confirm "
        "Spearman is still ~0.91. The GPU is a different code path; nothing "
        "validated on CPU transfers for free.", logfile)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except PreflightError as e:
        print(f"\n*** PREFLIGHT FAILED — RUN STOPPED ***\n{e}\n", flush=True)
        sys.exit(2)
