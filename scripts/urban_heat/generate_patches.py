#!/usr/bin/env python3
"""
Urban Heat Tool — Patch Generator (Phase 1.6.2)

Slices the 9000x9000 city-wide rasters into overlapping 512x512 patches
for SOLWEIG simulation. Each patch directory contains cropped versions of
all input layers aligned to the same spatial window.

Strategy (per supervisor's proposal — Task 2.1):
  - Patch size:  512x512 pixels (512m x 512m at 1m resolution)
  - Stride:      256 pixels (50% overlap)
  - Validity:    Skip patches where >80% is NoData or water

Input COG rasters (all 9000x9000 @ 1m, EPSG:32755):
  urban_heat_data/rasters/cogs/dsm_cog.tif
  urban_heat_data/rasters/cogs/dem_cog.tif
  urban_heat_data/rasters/cogs/cdsm_cog.tif
  urban_heat_data/rasters/cogs/albedo_master_cog.tif
  urban_heat_data/rasters/auxiliary/landcover.tif

Output:
  urban_heat_data/training/patches/patch_NNNN/
    building_dsm.tif    (512x512)
    dem.tif             (512x512)
    trees.tif           (512x512)
    landcover.tif       (512x512)
    albedo.tif          (512x512)
  urban_heat_data/training/metadata.json

Usage:
  python generate_patches.py
  python generate_patches.py --patch-size 512 --stride 256 --min-valid 0.2
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.windows import Window
from pyproj import Transformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Paths
DATA_ROOT = Path("/data")
if not DATA_ROOT.exists():
    DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "urban_heat_data"

COG_DIR = DATA_ROOT / "rasters" / "cogs"
AUX_DIR = DATA_ROOT / "rasters" / "auxiliary"
TRAINING_DIR = DATA_ROOT / "training"
PATCHES_DIR = TRAINING_DIR / "patches"

# Layer definitions: (filename, output_name, source_dir)
LAYERS = [
    ("dsm_cog.tif", "building_dsm.tif", COG_DIR),
    ("dem_cog.tif", "dem.tif", COG_DIR),
    ("cdsm_cog.tif", "trees.tif", COG_DIR),
    ("albedo_master_cog.tif", "albedo.tif", COG_DIR),
    ("landcover.tif", "landcover.tif", AUX_DIR),
]

# UMEP land cover code for water
LC_WATER = 3


def get_raster_info(path: Path) -> Dict:
    """Get basic raster metadata."""
    with rasterio.open(path) as src:
        return {
            "width": src.width,
            "height": src.height,
            "crs": str(src.crs),
            "transform": src.transform,
            "nodata": src.nodata,
            "bounds": src.bounds,
        }


def check_patch_validity(
    dsm_path: Path,
    landcover_path: Path,
    window: Window,
    min_valid_fraction: float,
) -> Tuple[bool, float]:
    """
    Check if a patch has enough valid data to be useful for simulation.
    Returns (is_valid, valid_fraction).
    """
    # Read DSM to check for NoData
    with rasterio.open(dsm_path) as src:
        dsm_data = src.read(1, window=window)
        nodata = src.nodata if src.nodata is not None else -9999.0
        nodata_mask = (dsm_data == nodata) | np.isnan(dsm_data)

    # Read land cover to check for water
    with rasterio.open(landcover_path) as src:
        lc_data = src.read(1, window=window)
        water_mask = lc_data == LC_WATER

    # Invalid = NoData OR water
    invalid_mask = nodata_mask | water_mask
    total_pixels = window.width * window.height
    valid_fraction = 1.0 - (np.sum(invalid_mask) / total_pixels)

    return valid_fraction >= min_valid_fraction, float(valid_fraction)


def extract_patch(
    src_path: Path,
    out_path: Path,
    window: Window,
    ref_profile: Dict,
):
    """Extract a window from a raster and write as a new GeoTIFF."""
    with rasterio.open(src_path) as src:
        data = src.read(1, window=window)

        # Compute the transform for this window
        win_transform = rasterio.windows.transform(window, src.transform)

        profile = ref_profile.copy()
        profile.update(
            width=window.width,
            height=window.height,
            count=1,
            transform=win_transform,
            dtype=data.dtype,
            compress="deflate",
        )
        if src.nodata is not None:
            profile["nodata"] = src.nodata

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data, 1)


def main():
    parser = argparse.ArgumentParser(description="Generate 512x512 training patches")
    parser.add_argument("--patch-size", type=int, default=512, help="Patch size in pixels")
    parser.add_argument("--stride", type=int, default=256, help="Stride in pixels (overlap)")
    parser.add_argument("--min-valid", type=float, default=0.2, help="Min fraction of valid pixels")
    args = parser.parse_args()

    patch_size = args.patch_size
    stride = args.stride
    min_valid = args.min_valid

    logger.info(f"Patch generation: size={patch_size}, stride={stride}, min_valid={min_valid}")

    # Verify all input layers exist
    for filename, _, src_dir in LAYERS:
        path = src_dir / filename
        if not path.exists():
            logger.error(f"Missing input: {path}")
            sys.exit(1)
        logger.info(f"  Found: {filename} ({path.stat().st_size / (1024*1024):.1f} MB)")

    # Get grid dimensions from DSM
    dsm_path = COG_DIR / "dsm_cog.tif"
    info = get_raster_info(dsm_path)
    grid_width = info["width"]
    grid_height = info["height"]
    crs = info["crs"]
    transform = info["transform"]
    logger.info(f"Grid: {grid_width}x{grid_height}, CRS: {crs}")

    # Set up UTM → WGS84 transformer for center lat/lon
    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

    # Create output directory
    PATCHES_DIR.mkdir(parents=True, exist_ok=True)

    # Get reference profile for writing patches
    with rasterio.open(dsm_path) as src:
        ref_profile = src.profile.copy()

    # Generate patch grid
    landcover_path = AUX_DIR / "landcover.tif"
    patch_positions: List[Tuple[int, int]] = []

    for row_start in range(0, grid_height - patch_size + 1, stride):
        for col_start in range(0, grid_width - patch_size + 1, stride):
            patch_positions.append((row_start, col_start))

    logger.info(f"Total grid positions: {len(patch_positions)}")

    # Filter valid patches and extract
    metadata = {
        "patch_size": patch_size,
        "stride": stride,
        "min_valid_fraction": min_valid,
        "grid_width": grid_width,
        "grid_height": grid_height,
        "crs": crs,
        "patches": [],
    }

    valid_count = 0
    skipped_count = 0

    for idx, (row_start, col_start) in enumerate(patch_positions):
        window = Window(col_start, row_start, patch_size, patch_size)

        # Check validity
        is_valid, valid_frac = check_patch_validity(
            dsm_path, landcover_path, window, min_valid
        )

        if not is_valid:
            skipped_count += 1
            continue

        # Create patch directory
        patch_name = f"patch_{valid_count:04d}"
        patch_dir = PATCHES_DIR / patch_name
        patch_dir.mkdir(parents=True, exist_ok=True)

        # Extract all layers
        for filename, out_name, src_dir in LAYERS:
            src_path = src_dir / filename
            out_path = patch_dir / out_name
            extract_patch(src_path, out_path, window, ref_profile)

        # Compute patch extent in UTM and WGS84
        x_min = transform.c + col_start * transform.a
        y_max = transform.f + row_start * transform.e
        x_max = x_min + patch_size * transform.a
        y_min = y_max + patch_size * transform.e

        # Center in WGS84
        cx_utm = (x_min + x_max) / 2
        cy_utm = (y_min + y_max) / 2
        cx_lon, cy_lat = transformer.transform(cx_utm, cy_utm)

        patch_meta = {
            "name": patch_name,
            "index": valid_count,
            "grid_position": {"row": row_start, "col": col_start},
            "extent_utm": {
                "x_min": round(x_min, 2),
                "y_min": round(y_min, 2),
                "x_max": round(x_max, 2),
                "y_max": round(y_max, 2),
            },
            "center_wgs84": {
                "lon": round(cx_lon, 6),
                "lat": round(cy_lat, 6),
            },
            "valid_fraction": round(valid_frac, 4),
        }
        metadata["patches"].append(patch_meta)

        valid_count += 1

        if valid_count % 25 == 0:
            logger.info(f"  Generated {valid_count} patches ({skipped_count} skipped)...")

    metadata["total_valid"] = valid_count
    metadata["total_skipped"] = skipped_count

    # Write metadata
    meta_path = TRAINING_DIR / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"\nPatch generation complete:")
    logger.info(f"  Valid patches: {valid_count}")
    logger.info(f"  Skipped (invalid): {skipped_count}")
    logger.info(f"  Output: {PATCHES_DIR}")
    logger.info(f"  Metadata: {meta_path}")

    # Quick size estimate
    if valid_count > 0:
        sample_dir = PATCHES_DIR / "patch_0000"
        patch_size_mb = sum(f.stat().st_size for f in sample_dir.iterdir()) / (1024 * 1024)
        total_mb = patch_size_mb * valid_count
        logger.info(f"  Size per patch: {patch_size_mb:.1f} MB")
        logger.info(f"  Total patches size: {total_mb:.0f} MB")


if __name__ == "__main__":
    main()
