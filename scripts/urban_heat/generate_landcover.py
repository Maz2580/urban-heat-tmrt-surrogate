#!/usr/bin/env python3
"""
Urban Heat Tool — Land Cover Classification Generator (Phase 1.6.2)

Combines the 6 auxiliary rasters (from Phase 1.6.1) into a single land cover
classification raster using the UMEP-style scheme required by solweig-gpu.

UMEP Land Cover Codes:
  1 = Paved surfaces (roads, footpaths, hard surfaces)
  2 = Buildings
  3 = Water
  4 = Vegetation (tree canopy, grass, nature strips)
  5 = Bare soil (default — everything else)

Priority (highest wins): Water > Buildings > Paved > Vegetation > Bare Soil

Inputs:
  urban_heat_data/rasters/auxiliary/buildings.tif
  urban_heat_data/rasters/auxiliary/carriageway.tif
  urban_heat_data/rasters/auxiliary/footway.tif
  urban_heat_data/rasters/auxiliary/nature_strip.tif
  urban_heat_data/rasters/auxiliary/tree_canopy.tif
  urban_heat_data/rasters/auxiliary/water.tif

Output:
  urban_heat_data/rasters/auxiliary/landcover.tif  (9000x9000, uint8, EPSG:32755)

Usage:
  python generate_landcover.py
"""

import logging
import sys
from pathlib import Path

import numpy as np
import rasterio

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

AUX_DIR = DATA_ROOT / "rasters" / "auxiliary"
OUTPUT_PATH = AUX_DIR / "landcover.tif"

# UMEP land cover codes
LC_BARE_SOIL = 5
LC_VEGETATION = 4
LC_PAVED = 1
LC_BUILDINGS = 2
LC_WATER = 3


def load_mask(path: Path) -> np.ndarray:
    """Load a raster and return a boolean mask where values > 0."""
    with rasterio.open(path) as src:
        data = src.read(1)
        nodata = src.nodata
        if nodata is not None:
            mask = (data > 0) & (data != nodata)
        else:
            mask = data > 0
        return mask


def main():
    logger.info("Generating UMEP land cover classification raster")

    # Check inputs exist
    required = {
        "buildings": AUX_DIR / "buildings.tif",
        "carriageway": AUX_DIR / "carriageway.tif",
        "footway": AUX_DIR / "footway.tif",
        "nature_strip": AUX_DIR / "nature_strip.tif",
        "tree_canopy": AUX_DIR / "tree_canopy.tif",
        "water": AUX_DIR / "water.tif",
    }

    for name, path in required.items():
        if not path.exists():
            logger.error(f"Missing input: {path}")
            sys.exit(1)
        logger.info(f"  Found: {name} ({path.stat().st_size / 1024:.0f} KB)")

    # Get profile from reference raster
    with rasterio.open(required["buildings"]) as ref:
        profile = ref.profile.copy()
        height, width = ref.height, ref.width
        logger.info(f"Grid: {width}x{height}, CRS: {ref.crs}, transform: {ref.transform}")

    # Initialize with bare soil
    landcover = np.full((height, width), LC_BARE_SOIL, dtype=np.uint8)
    logger.info(f"Initialized {width}x{height} grid with bare soil (code {LC_BARE_SOIL})")

    # Apply in priority order (lowest to highest — last write wins)

    # 1. Vegetation (tree canopy + nature strips)
    tree_mask = load_mask(required["tree_canopy"])
    nature_mask = load_mask(required["nature_strip"])
    veg_mask = tree_mask | nature_mask
    veg_count = np.sum(veg_mask)
    landcover[veg_mask] = LC_VEGETATION
    logger.info(f"  Vegetation: {veg_count:,} pixels ({veg_count / (height * width) * 100:.1f}%)")

    # 2. Paved surfaces (carriageway + footway)
    road_mask = load_mask(required["carriageway"])
    foot_mask = load_mask(required["footway"])
    paved_mask = road_mask | foot_mask
    paved_count = np.sum(paved_mask)
    landcover[paved_mask] = LC_PAVED
    logger.info(f"  Paved: {paved_count:,} pixels ({paved_count / (height * width) * 100:.1f}%)")

    # 3. Buildings (highest priority after water)
    bldg_mask = load_mask(required["buildings"])
    bldg_count = np.sum(bldg_mask)
    landcover[bldg_mask] = LC_BUILDINGS
    logger.info(f"  Buildings: {bldg_count:,} pixels ({bldg_count / (height * width) * 100:.1f}%)")

    # 4. Water (highest priority)
    water_mask = load_mask(required["water"])
    water_count = np.sum(water_mask)
    landcover[water_mask] = LC_WATER
    logger.info(f"  Water: {water_count:,} pixels ({water_count / (height * width) * 100:.1f}%)")

    # Summary
    bare_count = np.sum(landcover == LC_BARE_SOIL)
    logger.info(f"  Bare soil: {bare_count:,} pixels ({bare_count / (height * width) * 100:.1f}%)")

    # Write output
    out_profile = profile.copy()
    out_profile.update(
        dtype=rasterio.uint8,
        count=1,
        compress="deflate",
        nodata=0,
    )

    with rasterio.open(OUTPUT_PATH, "w", **out_profile) as dst:
        dst.write(landcover, 1)
        dst.update_tags(
            description="UMEP Land Cover Classification",
            codes="1=Paved, 2=Buildings, 3=Water, 4=Vegetation, 5=Bare Soil",
        )

    size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    logger.info(f"Written: {OUTPUT_PATH} ({size_mb:.1f} MB)")
    logger.info("Land cover generation complete.")


if __name__ == "__main__":
    main()
