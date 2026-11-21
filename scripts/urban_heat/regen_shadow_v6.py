#!/usr/bin/env python3
"""Regenerate the three static shadow channels (slots 17/18/19) for V6.

The rasters currently feeding channels 17-19 carry three independent defects,
all measured (see V6_PLAN.md):

  1. DIRECTION REVERSED. `compute_shadow_fraction` builds a ray direction
     (dx, dy) toward the sun, then its array shift reads dsm[p - d] instead of
     dsm[p + d], painting shadow on the SUNLIT face of every building.
     Measured alignment with the true shadow direction: -0.988 / -1.000 /
     -1.000 at 09/13/16. Because dx = sin(az) and dy = -cos(az) both flip sign
     under az -> az+180, the correction is a 180 deg rotation of the azimuth.

  2. WRONG SURFACE. Generated from `dsm_1m.tif` (2018 LiDAR) while channel 0
     of the same tensor reads `geoscape_heights.tif` (Dec 2025). The model was
     therefore told "a 250 m tower stands here" and "nothing casts a shadow
     here" in the same sample. 40.2% of buildings over 80 m are absent or
     >5 m short in the 2018 surface, against 4.0% of 3-10 m buildings.

  3. RAY TRUNCATED AT 100 m. Shadow length is height / tan(altitude); at the
     09:00 and 16:00 sun altitudes (~43 and ~40 deg) that truncates every
     building above roughly 85-95 m -- precisely the towers defect 2 is about.

`--legacy` reproduces the CURRENT rasters (backwards ray, dsm_1m, 100 m). It
exists so the corrected output is only trusted once this same code has been
shown to regenerate the artefact it replaces. A generator that cannot
reproduce the old file has no standing to produce the new one.

Writes ONLY to --out-dir. Never touches `v4_prep/shadow_multi/`, which the
live V4/V4.1/V5 inference path reads.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import rasterio

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "urban_heat_data"
RAW = DATA / "rasters" / "raw"
COGS = DATA / "rasters" / "cogs"

LEGACY_SURFACE = RAW / "dsm_1m.tif"
V6_SURFACE = RAW / "geoscape_heights.tif"
DEM = COGS / "dem_cog.tif"

# Must match scripts/urban_heat/v4_prep.py task_shadow().
DOY = 22
HOURS = (9, 13, 16)
CENTRE_LON, CENTRE_LAT = 144.93, -37.83


def solar_position(lat: float, lon: float, doy: int, hour: int,
                   utc_offset: int = 10) -> tuple[float, float]:
    """Altitude and azimuth in degrees, from the DEPLOYED solar service.

    Imported rather than reimplemented: an approximate formula of my own
    disagreed with this one by 7 deg of azimuth at 13:00, which would have
    shifted every shadow and broken the legacy control for a reason that had
    nothing to do with the defect under test. `app/services/urban_heat_solar.py`
    and `scripts/urban_heat/stage_d.py` were checked against each other and
    agree exactly, so this is the single convention the corpus already uses.
    """
    import importlib.util
    import sys as _sys
    name = "_uh_solar_regen"
    if name not in _sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, REPO / "app/services/urban_heat_solar.py")
        mod = importlib.util.module_from_spec(spec)
        _sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return _sys.modules[name].compute_solar_position(
        lat, lon, doy, hour, utc_offset=utc_offset)


def shadow_map(surface: np.ndarray, alt_deg: float, az_deg: float,
               max_distance: int, backwards: bool) -> np.ndarray:
    """Array-shift ray cast. `backwards=True` reproduces the shipped defect.

    Buffers are preallocated and reused: at 9000x9000 float32 each temporary
    is 324 MB, and this box runs with ~15 GB free alongside production
    containers.
    """
    alt = math.radians(alt_deg)
    dx, dy = math.sin(math.radians(az_deg)), -math.cos(math.radians(az_deg))
    if not backwards:
        # The shift below reads surface[p - shift]; negating the direction
        # makes that surface[p + d_toward_sun], which is the true occluder.
        dx, dy = -dx, -dy
    tan_alt = math.tan(alt) if alt > 0.01 else 100.0

    h, w = surface.shape
    acc = np.zeros((h, w), dtype=bool)
    shifted = np.empty_like(surface)
    diff = np.empty_like(surface)
    hit = np.empty((h, w), dtype=bool)
    fill = float(surface.min())

    for dist in range(1, max_distance + 1, 2):
        sy, sx = int(round(dist * dy)), int(round(dist * dx))
        if abs(sy) >= h or abs(sx) >= w:
            break
        shifted[:] = fill
        shifted[max(0, sy):min(h, h + sy), max(0, sx):min(w, w + sx)] = \
            surface[max(0, -sy):min(h, h - sy), max(0, -sx):min(w, w - sx)]
        np.subtract(shifted, surface, out=diff)
        np.greater(diff, dist * tan_alt, out=hit)
        np.logical_or(acc, hit, out=acc)
    return acc.astype(np.float32)


def load_surface(path: Path, dem_fill: bool) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as s:
        arr = s.read(1).astype(np.float32)
        prof = {"transform": s.transform, "crs": s.crs}
    bad = ~np.isfinite(arr) | (arr < -100.0)
    n_bad = int(bad.sum())
    if dem_fill and n_bad:
        # Filling voids with 0 would sink them ~10-50 m below the surrounding
        # terrain and cast shadows off the resulting cliff. Terrain is the
        # physical answer: no structure, just ground.
        with rasterio.open(DEM) as s:
            dem = s.read(1).astype(np.float32)
        ok = np.isfinite(dem) & (dem > -9000)
        arr[bad & ok] = dem[bad & ok]
        arr[bad & ~ok] = 0.0
    else:
        arr[bad] = 0.0
    return arr, {**prof, "void_px": n_bad, "void_pct": 100.0 * n_bad / arr.size}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--legacy", action="store_true",
                    help="reproduce the CURRENT rasters (control, not for training)")
    ap.add_argument("--max-distance", type=int, default=400,
                    help="ray length in metres; 400 covers a 300 m tower at 39 deg sun")
    args = ap.parse_args()

    backwards = args.legacy
    surface_path = LEGACY_SURFACE if args.legacy else V6_SURFACE
    max_dist = 100 if args.legacy else args.max_distance
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"mode        : {'LEGACY (reproduce current)' if args.legacy else 'V6 (corrected)'}")
    print(f"surface     : {surface_path.name}")
    print(f"ray         : {'backwards' if backwards else 'toward sun'}, max {max_dist} m")
    print(f"out         : {args.out_dir}\n")

    surface, prof = load_surface(surface_path, dem_fill=not args.legacy)
    print(f"surface loaded {surface.shape}  voids {prof['void_px']:,} ({prof['void_pct']:.2f}%)"
          f"  -> filled with {'DEM' if not args.legacy else '0.0'}\n")

    summary = {"mode": "legacy" if args.legacy else "v6", "surface": str(surface_path),
               "max_distance": max_dist, "backwards": backwards, "hours": {}}
    for hour in HOURS:
        alt, az = solar_position(CENTRE_LAT, CENTRE_LON, DOY, hour)
        t0 = time.time()
        sh = shadow_map(surface, max(alt, 1.0), az, max_dist, backwards)
        el = time.time() - t0
        out = args.out_dir / f"shadow_h{hour:02d}.tif"
        with rasterio.open(out, "w", driver="GTiff", height=sh.shape[0],
                           width=sh.shape[1], count=1, dtype="float32",
                           crs=prof["crs"], transform=prof["transform"],
                           compress="deflate", tiled=True,
                           blockxsize=256, blockysize=256) as ds:
            ds.write(sh, 1)
        cov = 100.0 * float((sh > 0.5).mean())
        print(f"  h{hour:02d}  alt={alt:5.1f} az={az:6.1f}  shadow={cov:5.2f}%  {el:5.1f}s  -> {out.name}")
        summary["hours"][f"h{hour:02d}"] = {
            "sun_alt_deg": round(alt, 2), "sun_az_deg": round(az, 2),
            "shadow_pct": round(cov, 3), "seconds": round(el, 1),
        }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote summary.json")


if __name__ == "__main__":
    main()
