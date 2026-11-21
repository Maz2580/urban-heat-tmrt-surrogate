#!/usr/bin/env python3
"""
Urban Heat Tool — DSM Processing Script (Phase 1.6.1)

Processes the City of Melbourne 0.1m DSM (2018) into SOLWEIG-ready rasters:
  1. Merge DSM tiles → single GeoTIFF
  2. Resample from 0.1m to 1m resolution (100x reduction)
  3. Reproject to EPSG:4326 if needed
  4. Clip to CoM bounding box
  5. Generate DEM (bare-earth) by masking buildings and interpolating
  6. Convert outputs to Cloud Optimized GeoTIFF (COG)

Usage:
  python process_dsm.py                    # Full pipeline
  python process_dsm.py --step merge       # Only merge tiles
  python process_dsm.py --step resample    # Only resample
  python process_dsm.py --step dem         # Only generate DEM
  python process_dsm.py --step cog         # Only convert to COG

Requires: rasterio, numpy, scipy (for DEM interpolation)
"""

import argparse
import glob
import logging
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("urban_heat.process_dsm")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_ROOT = PROJECT_ROOT / "urban_heat_data"
RAW_DIR = DATA_ROOT / "rasters" / "raw"
COG_DIR = DATA_ROOT / "rasters" / "cogs"
AUX_DIR = DATA_ROOT / "rasters" / "auxiliary"
VECTOR_DIR = DATA_ROOT / "vectors"

# Target resolution (meters)
TARGET_RES = 1.0

# CoM bounding box (EPSG:4326)
COM_BBOX_4326 = {
    "xmin": 144.90, "ymin": -37.84,
    "xmax": 144.99, "ymax": -37.78,
}

# NoData value
NODATA = -9999.0


# ---------------------------------------------------------------------------
# Step 1: Extract and merge DSM tiles
# ---------------------------------------------------------------------------

def extract_dsm_zip():
    """Extract CoM_DSM_2018.zip into raw/dsm_tiles/."""
    zip_path = RAW_DIR / "CoM_DSM_2018.zip"
    if not zip_path.exists():
        logger.error(f"DSM zip not found: {zip_path}")
        return False

    extract_dir = RAW_DIR / "dsm_tiles"
    extract_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Extracting {zip_path} ({zip_path.stat().st_size / 1e9:.1f} GB)...")

    import zipfile
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if m.lower().endswith('.tif')]
        logger.info(f"Found {len(members)} GeoTIFF files in archive")
        extracted = 0
        skipped = 0
        for i, member in enumerate(members):
            filename = os.path.basename(member)
            if not filename:
                continue
            target = extract_dir / filename
            if target.exists() and target.stat().st_size > 0:
                skipped += 1
                continue
            with zf.open(member) as src, open(target, 'wb') as dst:
                shutil.copyfileobj(src, dst)
            extracted += 1
            if extracted % 10 == 0:
                logger.info(f"  Extracted {extracted} new tiles ({skipped} already exist)...")
        logger.info(f"  New: {extracted}, Already existed: {skipped}")

    tifs = list(extract_dir.glob("*.tif"))
    logger.info(f"Extracted {len(tifs)} tiles to {extract_dir}")
    return True


def merge_and_resample_dsm():
    """
    Merge DSM tiles and resample from 0.1m to 1m using GDAL VRT + translate.
    This avoids loading 12GB into memory.
    """
    import rasterio
    from rasterio.merge import merge
    from rasterio.enums import Resampling

    tiles_dir = RAW_DIR / "dsm_tiles"
    output_path = RAW_DIR / "dsm_1m.tif"

    if output_path.exists():
        logger.info(f"Resampled DSM already exists: {output_path}")
        return True

    tif_files = sorted(tiles_dir.glob("*.tif"))
    if not tif_files:
        logger.error(f"No .tif files found in {tiles_dir}")
        return False

    logger.info(f"Found {len(tif_files)} DSM tiles")

    # Read first tile to get CRS and check resolution
    with rasterio.open(tif_files[0]) as src:
        src_crs = src.crs
        src_res = src.res
        logger.info(f"Source CRS: {src_crs}, Resolution: {src_res[0]:.4f}m x {src_res[1]:.4f}m")

    # Calculate the resample factor
    resample_factor = TARGET_RES / abs(src_res[0])
    logger.info(f"Resample factor: {resample_factor:.1f}x (from {abs(src_res[0]):.2f}m to {TARGET_RES}m)")

    # Use GDAL buildvrt + translate for memory-efficient processing
    logger.info("Building VRT from all tiles...")
    vrt_path = RAW_DIR / "dsm_merged.vrt"

    # Build file list for GDAL
    filelist_path = RAW_DIR / "dsm_tiles_list.txt"
    with open(filelist_path, 'w') as f:
        for tf in tif_files:
            f.write(str(tf) + '\n')

    # Use GDAL CLI for memory-efficient merge + resample
    import subprocess

    # Step 1: Build VRT
    cmd_vrt = [
        'gdalbuildvrt',
        str(vrt_path),
        '-input_file_list', str(filelist_path),
    ]
    logger.info(f"Running: {' '.join(cmd_vrt[:3])}...")
    result = subprocess.run(cmd_vrt, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        logger.error(f"gdalbuildvrt failed: {result.stderr}")
        return False
    logger.info("VRT built successfully")

    # Step 2: Translate VRT → resampled GeoTIFF at 1m
    cmd_translate = [
        'gdal_translate',
        '-of', 'GTiff',
        '-tr', str(TARGET_RES), str(TARGET_RES),
        '-r', 'average',  # Average resampling for elevation data
        '-co', 'COMPRESS=DEFLATE',
        '-co', 'TILED=YES',
        '-co', 'BLOCKXSIZE=256',
        '-co', 'BLOCKYSIZE=256',
        '-co', 'NUM_THREADS=ALL_CPUS',
        str(vrt_path),
        str(output_path),
    ]
    logger.info(f"Resampling to {TARGET_RES}m resolution (this may take a while)...")
    result = subprocess.run(cmd_translate, capture_output=True, text=True, timeout=1800)
    if result.returncode != 0:
        logger.error(f"gdal_translate failed: {result.stderr}")
        return False

    # Check output
    with rasterio.open(output_path) as dst:
        logger.info(f"Output DSM: {dst.width}x{dst.height}, CRS={dst.crs}, res={dst.res}")
        logger.info(f"Bounds: {dst.bounds}")
        size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"File size: {size_mb:.1f} MB")

    return True


# ---------------------------------------------------------------------------
# Step 2: Generate DEM (bare-earth)
# ---------------------------------------------------------------------------

def generate_dem():
    """
    Generate bare-earth DEM from DSM by:
    1. Rasterizing building footprints as a mask
    2. Setting building pixels to NaN
    3. Interpolating the gaps using scipy.ndimage
    """
    import rasterio
    from rasterio.features import rasterize
    import json

    dsm_path = RAW_DIR / "dsm_1m.tif"
    dem_path = RAW_DIR / "dem_1m.tif"

    if dem_path.exists():
        logger.info(f"DEM already exists: {dem_path}")
        return True

    if not dsm_path.exists():
        logger.error(f"DSM not found: {dsm_path}. Run merge_and_resample_dsm first.")
        return False

    logger.info("Generating DEM (bare-earth) from DSM...")

    # Load building footprints
    buildings_path = VECTOR_DIR / "buildings_com_2023.geojson"
    if not buildings_path.exists():
        logger.error(f"Building footprints not found: {buildings_path}")
        return False

    logger.info("Loading building footprints...")
    with open(buildings_path) as f:
        buildings = json.load(f)

    # Extract building geometries
    building_geoms = []
    for feat in buildings.get('features', []):
        geom = feat.get('geometry')
        if geom and geom.get('type') in ('Polygon', 'MultiPolygon'):
            building_geoms.append(geom)

    logger.info(f"Loaded {len(building_geoms)} building geometries")

    with rasterio.open(dsm_path) as src:
        dsm_data = src.read(1).astype(np.float64)
        profile = src.profile.copy()
        transform = src.transform
        dsm_crs = src.crs

        logger.info(f"DSM shape: {dsm_data.shape}, dtype: {dsm_data.dtype}, CRS: {dsm_crs}")

        # Reproject building geometries from EPSG:4326 to DSM CRS if needed
        from shapely.geometry import shape as shapely_shape, mapping as shapely_mapping
        from shapely.ops import transform as shapely_transform

        need_reproject = dsm_crs and str(dsm_crs) != "EPSG:4326"
        if need_reproject:
            from pyproj import Transformer
            proj_transformer = Transformer.from_crs("EPSG:4326", dsm_crs, always_xy=True)
            logger.info(f"Reprojecting buildings from EPSG:4326 → {dsm_crs}")

        # Rasterize buildings as a binary mask
        logger.info("Rasterizing building footprints...")
        shapes_with_val = []
        for geom in building_geoms:
            try:
                if need_reproject:
                    shp = shapely_shape(geom)
                    shp_proj = shapely_transform(proj_transformer.transform, shp)
                    geom = shapely_mapping(shp_proj)
                shapes_with_val.append((geom, 1))
            except Exception:
                continue

        building_mask = rasterize(
            shapes_with_val,
            out_shape=dsm_data.shape,
            transform=transform,
            fill=0,
            dtype=np.uint8,
        )

        building_pixels = np.sum(building_mask > 0)
        total_pixels = dsm_data.size
        logger.info(f"Building pixels: {building_pixels:,} / {total_pixels:,} ({100*building_pixels/total_pixels:.1f}%)")

        # Set building pixels to NaN for interpolation
        dem_data = dsm_data.copy()
        dem_data[building_mask > 0] = np.nan

        # Also set nodata values
        if src.nodata is not None:
            dem_data[dsm_data == src.nodata] = np.nan

        nan_count = np.sum(np.isnan(dem_data))
        logger.info(f"NaN pixels to interpolate: {nan_count:,}")

        # Interpolate gaps using nearest-neighbor (fast, preserves terrain)
        logger.info("Interpolating bare-earth surface...")
        from scipy.ndimage import distance_transform_edt

        valid_mask = ~np.isnan(dem_data)
        if np.any(valid_mask):
            # Use distance transform to find nearest valid pixel
            indices = distance_transform_edt(
                ~valid_mask,
                return_distances=False,
                return_indices=True,
            )
            dem_data = dem_data[tuple(indices)]
        else:
            logger.warning("No valid pixels in DEM — using DSM as-is")
            dem_data = dsm_data.copy()

        # Write DEM
        profile.update(dtype='float32', nodata=NODATA)
        dem_data = dem_data.astype(np.float32)
        dem_data[np.isnan(dem_data)] = NODATA

        with rasterio.open(dem_path, 'w', **profile) as dst:
            dst.write(dem_data, 1)

        size_mb = dem_path.stat().st_size / (1024 * 1024)
        logger.info(f"DEM written: {dem_path} ({size_mb:.1f} MB)")

    return True


# ---------------------------------------------------------------------------
# Step 3: Convert to COG
# ---------------------------------------------------------------------------

def convert_to_cog(input_path: Path, output_path: Path):
    """Convert a GeoTIFF to Cloud Optimized GeoTIFF."""
    import subprocess

    if output_path.exists():
        logger.info(f"COG already exists: {output_path}")
        return True

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        'gdal_translate',
        '-of', 'COG',
        '-co', 'COMPRESS=DEFLATE',
        '-co', 'OVERVIEW_RESAMPLING=AVERAGE',
        '-co', 'BLOCKSIZE=256',
        '-co', 'NUM_THREADS=ALL_CPUS',
        str(input_path),
        str(output_path),
    ]

    logger.info(f"Converting to COG: {input_path.name} → {output_path.name}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        logger.error(f"COG conversion failed: {result.stderr}")
        return False

    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"COG created: {output_path} ({size_mb:.1f} MB)")
    return True


def convert_all_to_cog():
    """Convert DSM and DEM to COGs."""
    success = True

    dsm_raw = RAW_DIR / "dsm_1m.tif"
    dsm_cog = COG_DIR / "dsm_cog.tif"
    if dsm_raw.exists():
        success &= convert_to_cog(dsm_raw, dsm_cog)

    dem_raw = RAW_DIR / "dem_1m.tif"
    dem_cog = COG_DIR / "dem_cog.tif"
    if dem_raw.exists():
        success &= convert_to_cog(dem_raw, dem_cog)

    return success


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Process CoM DSM into SOLWEIG-ready rasters")
    parser.add_argument(
        "--step",
        choices=["extract", "merge", "dem", "cog", "all"],
        default="all",
        help="Which processing step to run (default: all)",
    )
    args = parser.parse_args()

    # Ensure directories exist
    for d in [RAW_DIR, COG_DIR, AUX_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    if args.step in ("extract", "all"):
        logger.info("\n" + "=" * 60)
        logger.info("Step 1: Extract DSM tiles")
        if not extract_dsm_zip():
            return 1

    if args.step in ("merge", "all"):
        logger.info("\n" + "=" * 60)
        logger.info("Step 2: Merge and resample to 1m")
        if not merge_and_resample_dsm():
            return 1

    if args.step in ("dem", "all"):
        logger.info("\n" + "=" * 60)
        logger.info("Step 3: Generate DEM (bare-earth)")
        if not generate_dem():
            return 1

    if args.step in ("cog", "all"):
        logger.info("\n" + "=" * 60)
        logger.info("Step 4: Convert to COG")
        if not convert_all_to_cog():
            return 1

    logger.info("\n" + "=" * 60)
    logger.info("DSM processing complete!")
    return 0


if __name__ == "__main__":
    exit(main())
