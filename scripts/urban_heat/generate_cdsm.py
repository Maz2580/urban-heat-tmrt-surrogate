#!/usr/bin/env python3
"""
Urban Heat Tool — CDSM (Canopy Digital Surface Model) Generation (Phase 1.6.1)

Generates a tree canopy height raster from the City of Melbourne Urban Forest dataset.
Trees are point features with species + DBH; we estimate height and canopy radius
using allometric equations, then rasterize circular canopies onto a 1m grid.

Input:  urban_forest.geojson (82K trees with species, DBH, age)
        dsm_1m.tif (as spatial template for bounds/resolution/CRS)
Output: cdsm_1m.tif (canopy height above ground, 0 where no trees)

Usage:
  python generate_cdsm.py
  python generate_cdsm.py --template /path/to/dsm_1m.tif

References for allometric equations:
  - Australian urban tree allometry (City of Melbourne Urban Forest Strategy)
  - i-Tree Eco equations for Australian species
  - Generic H = a * DBH^b (power-law scaling)
"""

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("urban_heat.generate_cdsm")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_ROOT = PROJECT_ROOT / "urban_heat_data"
RAW_DIR = DATA_ROOT / "rasters" / "raw"
COG_DIR = DATA_ROOT / "rasters" / "cogs"
VECTOR_DIR = DATA_ROOT / "vectors"

NODATA = 0.0  # 0 means no canopy


# ---------------------------------------------------------------------------
# Allometric Equations: DBH (cm) → Height (m) and Canopy Radius (m)
# ---------------------------------------------------------------------------

# Height estimation: H = a * DBH^b (power law)
# These are calibrated for common Melbourne urban tree species/groups.
# DBH in cm, Height in meters.
#
# Groups:
#   eucalyptus: Eucalyptus spp. (River Red Gum, Spotted Gum, Lemon-scented Gum, etc.)
#   deciduous_large: Large deciduous (London Plane, English Elm, Dutch Elm)
#   acacia: Wattles (Black Wattle, Golden Wattle, Silver Wattle, Lightwood)
#   casuarina: Sheoaks (Drooping Sheoak, River Sheoak)
#   evergreen_medium: Medium evergreen (Brush Box, Moreton Bay Fig, Smooth-barked Apple)
#   small_tree: Small trees, shrubs (Sweet Bursaria, Coastal Banksia)
#   default: Generic urban tree

ALLOMETRY: Dict[str, Dict[str, float]] = {
    "eucalyptus":       {"h_a": 1.8, "h_b": 0.55, "cr_a": 0.8, "cr_b": 0.50, "max_h": 35},
    "deciduous_large":  {"h_a": 2.0, "h_b": 0.52, "cr_a": 1.0, "cr_b": 0.48, "max_h": 30},
    "acacia":           {"h_a": 1.5, "h_b": 0.50, "cr_a": 0.7, "cr_b": 0.48, "max_h": 15},
    "casuarina":        {"h_a": 1.6, "h_b": 0.52, "cr_a": 0.5, "cr_b": 0.45, "max_h": 20},
    "evergreen_medium": {"h_a": 1.7, "h_b": 0.53, "cr_a": 0.9, "cr_b": 0.50, "max_h": 25},
    "small_tree":       {"h_a": 1.2, "h_b": 0.48, "cr_a": 0.6, "cr_b": 0.45, "max_h": 10},
    "default":          {"h_a": 1.6, "h_b": 0.52, "cr_a": 0.7, "cr_b": 0.48, "max_h": 25},
}

# Species → allometric group mapping
SPECIES_GROUP: Dict[str, str] = {
    # Eucalyptus
    "River red gum": "eucalyptus",
    "Spotted Gum": "eucalyptus",
    "Lemon Scented Gum": "eucalyptus",
    "Yellow Box": "eucalyptus",
    "Yellow Gum": "eucalyptus",
    "Sugar Gum": "eucalyptus",
    "Manna Gum": "eucalyptus",
    "Swamp Gum": "eucalyptus",
    "Red Ironbark": "eucalyptus",
    "White Gum": "eucalyptus",
    # Deciduous large
    "London Plane": "deciduous_large",
    "English Elm": "deciduous_large",
    "Dutch Elm": "deciduous_large",
    "Tulip Tree": "deciduous_large",
    "English Oak": "deciduous_large",
    "Pin Oak": "deciduous_large",
    "Claret Ash": "deciduous_large",
    "Desert Ash": "deciduous_large",
    "Chinaberry Tree": "deciduous_large",
    # Acacia (wattles)
    "Black Wattle": "acacia",
    "Golden Wattle": "acacia",
    "Silver Wattle": "acacia",
    "Lightwood Wattle": "acacia",
    "Blackwood": "acacia",
    # Casuarina (sheoaks)
    "Drooping sheoak": "casuarina",
    "River Sheoak": "casuarina",
    # Evergreen medium
    "Smooth-barked apple": "evergreen_medium",
    "Brush Box": "evergreen_medium",
    "Moreton Bay Fig": "evergreen_medium",
    "Port Jackson Fig": "evergreen_medium",
    "Lilly Pilly": "evergreen_medium",
    # Small trees
    "Sweet Bursaria": "small_tree",
    "Coastal Banksia": "small_tree",
    "Bottlebrush": "small_tree",
}


def estimate_tree_dimensions(species: str, dbh_cm: float, age: str) -> Tuple[float, float]:
    """
    Estimate tree height (m) and canopy radius (m) from species, DBH, and age.
    Returns (height_m, canopy_radius_m).
    """
    group = SPECIES_GROUP.get(species, "default")
    params = ALLOMETRY[group]

    # Clamp DBH to reasonable range (some data has outliers like 26000cm)
    dbh_cm = min(max(dbh_cm, 1.0), 300.0)

    # Height: H = a * DBH^b, capped at max_h
    height = params["h_a"] * (dbh_cm ** params["h_b"])
    height = min(height, params["max_h"])

    # Canopy radius: CR = a * DBH^b
    canopy_r = params["cr_a"] * (dbh_cm ** params["cr_b"])
    canopy_r = min(canopy_r, 15.0)  # Max 15m radius

    # Age adjustment
    if age == "Unestablished":
        height *= 0.3
        canopy_r *= 0.3
    elif age == "Semi-mature":
        height *= 0.7
        canopy_r *= 0.7

    # Minimum canopy radius of 1m for any tree
    canopy_r = max(canopy_r, 1.0)
    height = max(height, 1.5)

    return round(height, 1), round(canopy_r, 1)


# ---------------------------------------------------------------------------
# Rasterization
# ---------------------------------------------------------------------------

def generate_cdsm(template_path: Path, output_path: Path) -> bool:
    """
    Generate CDSM raster from Urban Forest point data.
    Uses the DSM as a spatial template (bounds, CRS, resolution).
    """
    import rasterio
    from rasterio.transform import rowcol

    # Load template raster properties
    if not template_path.exists():
        logger.error(f"Template raster not found: {template_path}")
        return False

    with rasterio.open(template_path) as src:
        profile = src.profile.copy()
        transform = src.transform
        width = src.width
        height = src.height
        crs = src.crs
        logger.info(f"Template: {width}x{height}, CRS={crs}, res={src.res}")

    # Set up coordinate transformer if CRS is not EPSG:4326
    coord_transformer = None
    if crs and str(crs) != "EPSG:4326":
        from pyproj import Transformer
        coord_transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        logger.info(f"Will transform tree coords from EPSG:4326 → {crs}")

    # Initialize canopy array
    cdsm = np.zeros((height, width), dtype=np.float32)

    # Load trees
    trees_path = VECTOR_DIR / "urban_forest.geojson"
    if not trees_path.exists():
        logger.error(f"Urban Forest data not found: {trees_path}")
        return False

    logger.info(f"Loading Urban Forest data...")
    with open(trees_path) as f:
        data = json.load(f)

    features = data.get('features', [])
    logger.info(f"Total trees: {len(features)}")

    # Process each tree
    trees_placed = 0
    trees_skipped = 0

    for feat in features:
        props = feat.get('properties', {})
        geom = feat.get('geometry', {})

        # Get coordinates
        if geom.get('type') != 'Point':
            trees_skipped += 1
            continue

        coords = geom.get('coordinates', [])
        if len(coords) < 2:
            trees_skipped += 1
            continue

        lon, lat = coords[0], coords[1]

        # Transform to template CRS if needed
        if coord_transformer:
            try:
                lon, lat = coord_transformer.transform(lon, lat)
            except Exception:
                trees_skipped += 1
                continue

        # Get tree attributes
        species = props.get('common_name', 'Unknown')
        dbh = props.get('diameter_breast_height')
        age = props.get('age_description', 'Mature')

        if dbh is None or float(dbh) <= 0:
            # No DBH — estimate from age
            if age == 'Mature':
                dbh = 30.0
            elif age == 'Semi-mature':
                dbh = 15.0
            else:
                dbh = 5.0
        else:
            dbh = float(dbh)

        # Estimate dimensions
        tree_height, canopy_r = estimate_tree_dimensions(species, dbh, age)

        # Convert lon/lat to pixel coordinates
        try:
            row, col = rowcol(transform, lon, lat)
        except Exception:
            trees_skipped += 1
            continue

        # Check bounds
        if row < 0 or row >= height or col < 0 or col >= width:
            trees_skipped += 1
            continue

        # Rasterize circular canopy (radius in pixels, 1 pixel = 1m)
        r_px = int(round(canopy_r))

        for dy in range(-r_px, r_px + 1):
            for dx in range(-r_px, r_px + 1):
                # Circular mask
                if dx * dx + dy * dy > r_px * r_px:
                    continue
                py, px = row + dy, col + dx
                if 0 <= py < height and 0 <= px < width:
                    # Use max height (tallest tree wins at each pixel)
                    cdsm[py, px] = max(cdsm[py, px], tree_height)

        trees_placed += 1

        if trees_placed % 10000 == 0:
            logger.info(f"  Processed {trees_placed} trees...")

    logger.info(f"Trees placed: {trees_placed}, skipped: {trees_skipped}")

    canopy_pixels = np.sum(cdsm > 0)
    canopy_pct = 100 * canopy_pixels / cdsm.size
    logger.info(f"Canopy coverage: {canopy_pixels:,} pixels ({canopy_pct:.1f}%)")
    logger.info(f"Height range: {cdsm[cdsm > 0].min():.1f} - {cdsm.max():.1f}m")

    # Write CDSM
    profile.update(dtype='float32', count=1, nodata=NODATA)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(cdsm, 1)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"CDSM written: {output_path} ({size_mb:.1f} MB)")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate CDSM from Urban Forest data")
    parser.add_argument(
        "--template",
        type=str,
        default=str(RAW_DIR / "dsm_1m.tif"),
        help="Path to template raster (DSM) for bounds/CRS/resolution",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(RAW_DIR / "cdsm_1m.tif"),
        help="Output CDSM raster path",
    )
    args = parser.parse_args()

    template = Path(args.template)
    output = Path(args.output)

    if not template.exists():
        logger.error(f"Template raster not found: {template}")
        logger.error("Run process_dsm.py first to create the 1m DSM.")
        return 1

    if not generate_cdsm(template, output):
        return 1

    # Also convert to COG
    cog_output = COG_DIR / "cdsm_cog.tif"
    if not cog_output.exists():
        import subprocess
        logger.info(f"Converting to COG: {cog_output}")
        cmd = [
            'gdal_translate', '-of', 'COG',
            '-co', 'COMPRESS=DEFLATE',
            '-co', 'OVERVIEW_RESAMPLING=AVERAGE',
            '-co', 'BLOCKSIZE=256',
            '-co', 'NUM_THREADS=ALL_CPUS',
            str(output), str(cog_output),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            logger.error(f"COG conversion failed: {result.stderr}")
            return 1
        logger.info(f"CDSM COG created: {cog_output}")

    logger.info("CDSM generation complete!")
    return 0


if __name__ == "__main__":
    exit(main())
