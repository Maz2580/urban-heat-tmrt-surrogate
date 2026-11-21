#!/usr/bin/env python3
"""Generate V7's shadow inputs: corrected planes + shade-age, ONE world (2018).

V7's premise (memory.md §35-39 + the 2026-08-20 world-audit): the corpus mixes
two worlds -- ch0 geometry from Geoscape Dec 2025, SOLWEIG labels from the
2018 LiDAR. The model is trained that some buildings it sees cast no shadow,
so it hedges: soft, unreliable building shadows. V7 makes every input agree
with the LABELS' world (2018), because the labels are the physics:

  ch0    dsm_1m (2018)            -- swapped at pack time, not here
  ch17-19 shadow_h09/13/16        -- corrected ray, dsm_1m, 400 m  (this script)
  ch10   shade_age at sample hour -- consecutive hours shaded, dsm_1m (this
         script); reuses the dead slot that is currently served as zero

Reuses regen_shadow_v6.py's caster verbatim -- the one that earned trust by
reproducing the legacy artefact before producing the corrected one.

Writes ONLY to --out-dir (default urban_heat_data/v4_prep/shadow_multi_v7).
Never touches v4_prep/shadow_multi/ (live V4/V4.1/V5 serving).

Run (server, ~15 GB free required, nice'd):
    cd /mnt/apps && nice -n 19 uv run --with rasterio --with numpy python \
        spatial-llm/src/umami_spatial-llm/scripts/urban_heat/regen_shadow_v7.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent))
import regen_shadow_v6 as R                      # noqa: E402

SAMPLE_HOURS = (9, 11, 13, 15)                   # corpus hours -> age rasters
PLANE_HOURS = (9, 13, 16)                        # ch17/18/19 convention
ALL_HOURS = tuple(range(6, 17))                  # hourly stack for age


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path,
                    default=R.DATA / "v4_prep" / "shadow_multi_v7")
    ap.add_argument("--max-distance", type=int, default=400)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    surface, prof = R.load_surface(R.LEGACY_SURFACE, dem_fill=True)
    print(f"surface {R.LEGACY_SURFACE.name} {surface.shape}  "
          f"voids {prof['void_px']:,} ({prof['void_pct']:.2f}%) -> DEM-filled")

    def write(name: str, arr: np.ndarray) -> None:
        with rasterio.open(args.out_dir / name, "w", driver="GTiff",
                           height=arr.shape[0], width=arr.shape[1], count=1,
                           dtype="float32", crs=prof["crs"],
                           transform=prof["transform"], compress="deflate",
                           tiled=True, blockxsize=256, blockysize=256) as ds:
            ds.write(arr.astype(np.float32), 1)

    summary = {"surface": str(R.LEGACY_SURFACE), "max_distance": args.max_distance,
               "backwards": False, "doy": R.DOY, "hours": {}, "age": {}}

    hourly: dict[int, np.ndarray] = {}
    for hour in ALL_HOURS:
        alt, az = R.solar_position(R.CENTRE_LAT, R.CENTRE_LON, R.DOY, hour)
        t0 = time.time()
        if alt <= 1.0:
            sh = np.zeros(surface.shape, dtype=np.uint8)   # sun down: no shade
            note = "sun below horizon -> all sunlit"
        else:
            sh = R.shadow_map(surface, alt, az, args.max_distance,
                              backwards=False).astype(np.uint8)
            note = ""
        hourly[hour] = sh
        cov = 100.0 * float(sh.mean())
        el = time.time() - t0
        print(f"  h{hour:02d}  alt={alt:5.1f} az={az:6.1f}  "
              f"shadow={cov:5.2f}%  {el:5.1f}s  {note}")
        summary["hours"][f"h{hour:02d}"] = {
            "sun_alt_deg": round(alt, 2), "sun_az_deg": round(az, 2),
            "shadow_pct": round(cov, 3), "seconds": round(el, 1)}
        if hour in PLANE_HOURS:
            write(f"shadow_h{hour:02d}.tif", sh)

    # shade age: consecutive shaded hours ending at (and including) H.
    # A pixel sunlit at H has age 0 regardless of its morning.
    for H in SAMPLE_HOURS:
        age = np.zeros(surface.shape, dtype=np.float32)
        run = np.ones(surface.shape, dtype=bool)
        for h in range(H, ALL_HOURS[0] - 1, -1):
            run &= hourly[h].astype(bool)
            if not run.any():
                break
            age[run] += 1.0
        write(f"shade_age_h{H:02d}.tif", age)
        m = age[age > 0]
        summary["age"][f"h{H:02d}"] = {
            "shaded_pct": round(100.0 * float((age > 0).mean()), 3),
            "mean_age_h": round(float(m.mean()) if m.size else 0.0, 3),
            "max_age_h": float(age.max())}
        print(f"  age h{H:02d}: shaded {summary['age'][f'h{H:02d}']['shaded_pct']:.2f}%  "
              f"mean {summary['age'][f'h{H:02d}']['mean_age_h']:.2f} h  "
              f"max {age.max():.0f} h")

    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {args.out_dir}/summary.json")


if __name__ == "__main__":
    main()
