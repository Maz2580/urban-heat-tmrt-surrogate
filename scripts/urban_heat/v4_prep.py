"""V4 input prep — runs alongside V3 training to stage future model inputs.

Four sub-tasks (each with its own --task flag):

  ndvi      — Sentinel-2 median NDVI raster aligned to the 9000x9000 grid
  era5      — Pre-fetch ERA5 4-tuple {tair, rh, wind, dni} for all 12 combos
  shadow    — Multi-time-of-day shadow rasters (9am / 1pm / 4pm solar pos)
  landsat   — Per-patch Landsat LST sampling (11,844 patch x combo pairs)
  all       — Run all four

All outputs land in urban_heat_data/v4_prep/. Tasks are independent and
resumable (skip if output exists, override with --force).

Usage:
    docker exec -e PYTHONPATH=/app umami_test python /app/scripts/urban_heat/v4_prep.py --task all
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import sys
import time
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import requests
from rasterio.warp import Resampling, reproject

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.services.urban_heat.data_pipeline import gee_client as gc  # noqa: E402
from app.services.urban_heat.data_pipeline import patch_builder as pb  # noqa: E402
from app.services.urban_heat_feature_engineering import compute_shadow_fraction  # noqa: E402
from app.services.urban_heat_solar import compute_solar_position  # noqa: E402

logger = logging.getLogger("umami.urban_heat.v4_prep")

OUT_DIR = REPO_ROOT / "urban_heat_data" / "v4_prep"
RASTER_DIR = REPO_ROOT / "urban_heat_data" / "rasters" / "raw"
GRID_TEMPLATE = RASTER_DIR / "geoscape_heights.tif"
DSM_PATH = RASTER_DIR / "dsm_1m.tif"

# Same 12 combos as V3 corpus
COMBOS = [
    (15, 9), (15, 11), (15, 13), (15, 15),
    (46, 9), (46, 13),
    (306, 9), (306, 13),
    (356, 9), (356, 11), (356, 13), (356, 15),
]
DEFAULT_YEAR = 2024
AOI_CENTRE_LONLAT = (144.93, -37.83)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _doy_hour_to_utc(doy: int, hour: int, year: int) -> datetime:
    """Convert (DOY, hour) AEST to UTC datetime — same as build_v21_corpus."""
    aest = datetime(year, 1, 1, hour=hour, tzinfo=timezone.utc) \
        + timedelta(days=doy - 1)
    return aest - timedelta(hours=10)


def _read_grid_template():
    """Return (transform, crs, width, height) of the 9000x9000 master grid."""
    with rasterio.open(GRID_TEMPLATE) as ds:
        return ds.transform, ds.crs, ds.width, ds.height


def _dsm_grid_lonlat_bbox() -> tuple[float, float, float, float]:
    """The actual V3 training-grid bbox in WGS84 (W, S, E, N).

    Computed from dsm_1m.tif so it stays in sync with whatever the master
    grid actually covers. DON'T use gc.MELB_FB_BBOX here — that constant
    is for Smart Auditor's FB-specific work and doesn't match the V3 grid
    (it's offset south + a bit smaller).
    """
    from rasterio.warp import transform_bounds
    with rasterio.open(GRID_TEMPLATE) as ds:
        return transform_bounds(ds.crs, "EPSG:4326", *ds.bounds)


# ---------------------------------------------------------------------------
# Task 1 — Sentinel-2 NDVI raster
# ---------------------------------------------------------------------------

def task_ndvi(out_path: Path, force: bool = False) -> dict:
    """Median NDVI from Sentinel-2 around mid-summer Melbourne, aligned to grid."""
    if out_path.exists() and not force:
        logger.info("ndvi: %s exists, skipping (use --force to rebuild)", out_path)
        return {"skipped": True, "path": str(out_path)}

    gc.init()
    import ee
    bbox = _dsm_grid_lonlat_bbox()
    logger.info("ndvi: fetching Sentinel-2 median NDVI for mid-summer 2024 over %s", bbox)
    ndvi_img = gc.sentinel2_ndvi_for_bbox(
        bbox, "2024-01-15", window_days=30, max_cloud=20,
    )
    if ndvi_img is None:
        raise RuntimeError("no Sentinel-2 imagery returned for V3 grid bbox")

    transform, crs, width, height = _read_grid_template()
    region = ee.Geometry.Rectangle(list(bbox), 'EPSG:4326', False)
    url = ndvi_img.getDownloadURL({
        'region': region.getInfo(),
        'scale': 10,                     # Sentinel-2 native
        'crs': 'EPSG:32755',
        'format': 'GEO_TIFF',
    })
    logger.info("ndvi: downloading from GEE …")
    r = requests.get(url, timeout=180)
    r.raise_for_status()

    # GEE returns either a raw tif or a zip with one tif inside
    raw = r.content
    if raw[:2] == b'PK':
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            tif_name = next(n for n in zf.namelist() if n.endswith('.tif'))
            raw = zf.read(tif_name)
    tmp_path = out_path.with_suffix('.s2_native.tif')
    tmp_path.write_bytes(raw)

    # Reproject to the master grid (1m, 9000x9000, EPSG:32755)
    logger.info("ndvi: reprojecting to %dx%d @ 1m …", width, height)
    out = np.zeros((height, width), dtype=np.float32)
    with rasterio.open(tmp_path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=out,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=transform, dst_crs=crs,
            resampling=Resampling.bilinear,
        )

    profile = {
        "driver": "GTiff", "height": height, "width": width,
        "count": 1, "dtype": "float32",
        "crs": crs, "transform": transform,
        "compress": "deflate", "tiled": True, "blockxsize": 256, "blockysize": 256,
        "nodata": -9999.0,
    }
    with rasterio.open(out_path, "w", **profile) as ds:
        ds.write(out, 1)
    tmp_path.unlink(missing_ok=True)

    finite = float(np.isfinite(out).sum()) / out.size * 100
    return {
        "path": str(out_path),
        "size_mb": round(out_path.stat().st_size / 1e6, 1),
        "shape": list(out.shape),
        "ndvi_min": float(out.min()), "ndvi_max": float(out.max()),
        "ndvi_mean": float(out.mean()), "finite_pct": round(finite, 1),
    }


# ---------------------------------------------------------------------------
# Task 2 — Full ERA5 4-tuple cache
# ---------------------------------------------------------------------------

def task_era5(out_path: Path, year: int = DEFAULT_YEAR, force: bool = False) -> dict:
    """Sample {tair, rh, wind_speed, dni} for all 12 combos at AOI centre."""
    if out_path.exists() and not force:
        logger.info("era5: %s exists, skipping", out_path)
        return {"skipped": True, "path": str(out_path)}

    gc.init()
    rows = []
    for doy, hour in COMBOS:
        ts = _doy_hour_to_utc(doy, hour, year)
        sample = gc.era5_hourly_at_point(*AOI_CENTRE_LONLAT, ts)
        if sample is None:
            logger.warning("era5: no data for d%03d_h%02d", doy, hour)
            continue
        rows.append({
            "doy": doy, "hour": hour, "year": year,
            "timestamp_utc": ts.isoformat(),
            "tair_c": sample.get("tair_c"),
            "rh_pct": sample.get("rh_pct"),
            "wind_speed_ms": sample.get("wind_speed_ms"),
            "wind_dir_deg": sample.get("wind_dir_deg"),
            "ssrd_wm2": sample.get("ssrd_wm2"),
            "era5_image_ts": sample.get("image_ts"),
        })
        logger.info("era5: d%03d_h%02d → tair=%.2f, rh=%.1f, wind=%.2f m/s @ %.0f°, ssrd=%.0f W/m²",
                    doy, hour,
                    sample.get("tair_c") or float('nan'),
                    sample.get("rh_pct") or float('nan'),
                    sample.get("wind_speed_ms") or float('nan'),
                    sample.get("wind_dir_deg") or float('nan'),
                    sample.get("ssrd_wm2") or float('nan'))
    df = pd.DataFrame(rows)
    df.to_parquet(out_path)
    df.to_csv(out_path.with_suffix('.csv'), index=False)
    return {"path": str(out_path), "rows": len(df)}


# ---------------------------------------------------------------------------
# Task 3 — Multi-time-of-day shadow rasters
# ---------------------------------------------------------------------------

def task_shadow(out_dir: Path, force: bool = False) -> dict:
    """Compute shadow_fraction for 3 solar positions (9am, 1pm, 4pm Jan 22)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cen_lat, cen_lon = AOI_CENTRE_LONLAT[1], AOI_CENTRE_LONLAT[0]
    DOY = 22  # Jan 22 — peak summer reference
    HOURS = [9, 13, 16]

    logger.info("shadow: reading DSM …")
    with rasterio.open(DSM_PATH) as ds:
        dsm = ds.read(1).astype(np.float32)
        # Replace sentinel NoData with 0 so ray-tracing sees flat ground
        dsm[dsm < -100] = 0.0
        transform, crs = ds.transform, ds.crs
    h, w = dsm.shape

    out = {}
    for hour in HOURS:
        out_path = out_dir / f"shadow_h{hour:02d}.tif"
        if out_path.exists() and not force:
            logger.info("shadow: %s exists, skipping", out_path)
            out[f"h{hour:02d}"] = {"skipped": True, "path": str(out_path)}
            continue

        sun_alt, sun_az = compute_solar_position(cen_lat, cen_lon, DOY, hour, utc_offset=10)
        logger.info("shadow: hour=%d sun_alt=%.1f° sun_az=%.1f° — computing …",
                    hour, sun_alt, sun_az)
        t0 = time.time()
        shadow = compute_shadow_fraction(
            dsm, sun_altitude_deg=max(sun_alt, 1.0), sun_azimuth_deg=sun_az,
        )
        elapsed = time.time() - t0
        logger.info("shadow: hour=%d done in %.1fs (mean=%.3f, frac_in_shadow=%.2f%%)",
                    hour, elapsed, float(shadow.mean()),
                    100 * float((shadow > 0.5).mean()))

        profile = {
            "driver": "GTiff", "height": h, "width": w,
            "count": 1, "dtype": "float32",
            "crs": crs, "transform": transform,
            "compress": "deflate", "tiled": True, "blockxsize": 256, "blockysize": 256,
        }
        with rasterio.open(out_path, "w", **profile) as ds:
            ds.write(shadow.astype(np.float32), 1)
        out[f"h{hour:02d}"] = {
            "path": str(out_path),
            "sun_alt_deg": round(sun_alt, 1), "sun_az_deg": round(sun_az, 1),
            "size_mb": round(out_path.stat().st_size / 1e6, 1),
            "frac_in_shadow_pct": round(100 * float((shadow > 0.5).mean()), 2),
        }
    return out


# ---------------------------------------------------------------------------
# Task 4 — Per-patch Landsat LST sampling (batched)
# ---------------------------------------------------------------------------

def _patch_centroid_lonlat(patch_id: str) -> tuple[float, float] | None:
    """Read patch DSM, return centroid in EPSG:4326."""
    patch_dir = REPO_ROOT / "urban_heat_data" / "training" / "patches" / patch_id
    dsm_path = patch_dir / "building_dsm.tif"
    if not dsm_path.exists():
        return None
    with rasterio.open(dsm_path) as ds:
        bounds = ds.bounds
        cx = (bounds.left + bounds.right) / 2
        cy = (bounds.top + bounds.bottom) / 2
        from rasterio.warp import transform as warp_transform
        lon, lat = warp_transform(ds.crs, "EPSG:4326", [cx], [cy])
    return lon[0], lat[0]


def task_landsat(out_path: Path, year: int = DEFAULT_YEAR, force: bool = False) -> dict:
    """Sample Landsat 8/9 LST at each (V2 patch, combo) centroid via batched sampleRegions."""
    if out_path.exists() and not force:
        logger.info("landsat: %s exists, skipping", out_path)
        return {"skipped": True, "path": str(out_path)}

    gc.init()
    import ee

    # Load V2 valid patch list
    valid_path = REPO_ROOT / "urban_heat_data" / "training" / "valid_patches.json"
    with valid_path.open() as f:
        valid = json.load(f)
    patch_ids = valid["valid_patches"]
    logger.info("landsat: %d V2 patches in scope", len(patch_ids))

    # Pre-compute centroids in EPSG:4326 for batch sampling
    logger.info("landsat: computing patch centroids …")
    centroids = {}
    for pid in patch_ids:
        c = _patch_centroid_lonlat(pid)
        if c is not None:
            centroids[pid] = c
    logger.info("landsat: %d centroids ready", len(centroids))

    # Build EE FeatureCollection of points (one per patch)
    features = [
        ee.Feature(ee.Geometry.Point([lon, lat]), {"patch_id": pid})
        for pid, (lon, lat) in centroids.items()
    ]
    points_fc = ee.FeatureCollection(features)

    rows = []
    for doy, hour in COMBOS:
        ts = _doy_hour_to_utc(doy, hour, year)
        # Search for cloud-free Landsat scene within ±16 days of this combo
        start = (ts - timedelta(days=16)).strftime("%Y-%m-%d")
        end = (ts + timedelta(days=16)).strftime("%Y-%m-%d")
        scenes = list(gc.landsat_lst_for_bbox(
            _dsm_grid_lonlat_bbox(), start, end, max_cloud=30.0,
        ))
        if not scenes:
            logger.warning("landsat: no scenes for d%03d_h%02d (window %s..%s)",
                           doy, hour, start, end)
            continue

        # Pick scene closest in time
        best = min(scenes, key=lambda s: abs((s.timestamp - ts).total_seconds()))
        logger.info("landsat: d%03d_h%02d → %s (cloud=%.1f%%, time_delta=%.1fd)",
                    doy, hour, best.image_id, best.cloud_cover,
                    abs((best.timestamp - ts).total_seconds()) / 86400)

        # Batch sample all patch centroids on this scene at 30m
        sampled = best.image.sampleRegions(
            collection=points_fc, scale=30, projection='EPSG:4326',
        ).getInfo()
        for f in sampled.get('features', []):
            props = f.get('properties', {})
            # gee_client._landsat_st_celsius renames the band to "LST_C"
            lst_c = props.get('LST_C')
            if lst_c is None:
                continue
            rows.append({
                "v2_patch_id": props.get('patch_id'),
                "doy": doy, "hour": hour, "year": year,
                "scene_id": best.image_id, "scene_cloud_cover": best.cloud_cover,
                "scene_timestamp_utc": best.timestamp.isoformat(),
                "lst_celsius": float(lst_c),
            })
        logger.info("landsat: d%03d_h%02d → %d patches sampled", doy, hour, len(sampled.get('features', [])))

    df = pd.DataFrame(rows)
    df.to_parquet(out_path)
    return {
        "path": str(out_path), "rows": len(df),
        "unique_patches": int(df["v2_patch_id"].nunique()) if len(df) else 0,
        "unique_combos": int(df[["doy","hour"]].drop_duplicates().shape[0]) if len(df) else 0,
        "lst_min": float(df["lst_celsius"].min()) if len(df) else None,
        "lst_max": float(df["lst_celsius"].max()) if len(df) else None,
        "lst_mean": float(df["lst_celsius"].mean()) if len(df) else None,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--task", choices=["ndvi", "era5", "shadow", "landsat", "all"],
                   default="all")
    p.add_argument("--force", action="store_true",
                   help="Rebuild outputs even if they already exist")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}

    tasks = ["ndvi", "era5", "shadow", "landsat"] if args.task == "all" else [args.task]
    for t in tasks:
        t0 = time.time()
        logger.info("=" * 60)
        logger.info("starting task: %s", t)
        if t == "ndvi":
            summary[t] = task_ndvi(OUT_DIR / "geoscape_ndvi.tif", force=args.force)
        elif t == "era5":
            summary[t] = task_era5(OUT_DIR / "era5_combos.parquet", force=args.force)
        elif t == "shadow":
            summary[t] = task_shadow(OUT_DIR / "shadow_multi", force=args.force)
        elif t == "landsat":
            summary[t] = task_landsat(OUT_DIR / "landsat_per_patch.parquet", force=args.force)
        logger.info("task %s done in %.1fs", t, time.time() - t0)

    summary_path = OUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    logger.info("=" * 60)
    logger.info("all done. summary written to %s", summary_path)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
