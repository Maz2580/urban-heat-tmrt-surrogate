#!/usr/bin/env python3
"""
Urban Heat Tool — Albedo Map Generation (Phase 1.6.1)

Generates the Albedo Map (0.0 – 1.0) by stacking rasterized vector layers
with a priority-based Look-Up Table (LUT). This is NOT derived from satellite
imagery — it's computed from known surface properties.

The UMEP/SOLWEIG physics engine requires this as input for T_mrt simulation.

Priority hierarchy (last written wins):
  Bare Soil (0.15) → Grass (0.25) → Tree Canopy (0.20) →
  Building Roof (0.15) → Sidewalk (0.30) → Road (0.12) → Water (0.06)

Inputs:
  urban_heat_data/rasters/auxiliary/buildings.tif       (building footprints)
  urban_heat_data/rasters/auxiliary/carriageway.tif     (roads, tramway, bridges)
  urban_heat_data/rasters/auxiliary/footway.tif         (sidewalks, kerbs, channels)
  urban_heat_data/rasters/auxiliary/nature_strip.tif    (medians, nature strips)
  urban_heat_data/rasters/auxiliary/tree_canopy.tif     (LiDAR canopy polygons 2018)
  urban_heat_data/rasters/auxiliary/water.tif           (water bodies)
  urban_heat_data/rasters/cogs/cdsm_cog.tif            (optional, for tree height)

Output:
  urban_heat_data/rasters/raw/albedo_master.tif

Usage:
  python generate_albedo.py
  python generate_albedo.py --with-cdsm urban_heat_data/rasters/cogs/cdsm_cog.tif
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("urban_heat.albedo")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_ROOT = PROJECT_ROOT / "urban_heat_data"
AUX_DIR = DATA_ROOT / "rasters" / "auxiliary"
RAW_DIR = DATA_ROOT / "rasters" / "raw"

# ---------------------------------------------------------------------------
# Albedo Look-Up Table (Melbourne)
# ---------------------------------------------------------------------------
# Values from literature + supervisor's proposal

ALBEDO_LUT = {
    "bare_soil":    0.15,   # Default background — dry earth
    "nature_strip": 0.23,   # Grass/vegetation median strip
    "grass":        0.25,   # Standard dry-season parkland
    "tree_canopy":  0.18,   # Dark green leaves (from LiDAR polygons)
    "roof":         0.15,   # Dark tiles/bitumen (common in CBD)
    "footway":      0.30,   # Light grey concrete sidewalks
    "carriageway":  0.12,   # Standard aged asphalt roads
    "water":        0.06,   # Very low; absorbs most heat
    # Interventions (for what-if scenarios in Phase 1.6.7)
    "cool_roof":    0.70,   # White/reflective paint
    "cool_road":    0.35,   # Reflective road coating
}


def generate_albedo(
    output_path: Path = None,
    cdsm_path: Path = None,
) -> bool:
    """
    Generate albedo master raster by stacking layers with priority LUT.
    Reads rasterized layers from AUX_DIR. Returns True on success.
    """
    import rasterio

    if output_path is None:
        output_path = RAW_DIR / "albedo_master.tif"

    # Determine reference profile from the first available auxiliary raster
    ref_path = None
    for name in ["buildings.tif", "carriageway.tif", "water.tif", "tree_canopy.tif"]:
        p = AUX_DIR / name
        if p.exists():
            ref_path = p
            break

    if ref_path is None:
        logger.error("No input rasters found. Run rasterize_vectors.py first.")
        return False

    logger.info(f"Reference raster: {ref_path}")

    with rasterio.open(str(ref_path)) as src:
        profile = src.profile.copy()
        height, width = src.shape

    logger.info(f"Grid: {width}x{height} pixels")

    def _apply_layer(albedo_arr, raster_path, lut_key, label):
        """Apply a raster mask to the albedo array."""
        if not raster_path.exists():
            logger.info(f"  [{label}] not found — skipping")
            return
        with rasterio.open(str(raster_path)) as src:
            data = src.read(1)
            if data.shape != (height, width):
                logger.warning(f"  [{label}] shape mismatch: {data.shape} vs {(height, width)} — skipping")
                return
            mask = data > 0.5
            count = np.count_nonzero(mask)
            if count > 0:
                albedo_arr[mask] = ALBEDO_LUT[lut_key]
                pct = 100 * count / albedo_arr.size
                logger.info(f"  [{label}] {count:,} px ({pct:.1f}%) → albedo {ALBEDO_LUT[lut_key]}")

    # Start with bare soil as default
    albedo = np.full((height, width), ALBEDO_LUT["bare_soil"], dtype=np.float32)
    logger.info(f"  Base: bare soil ({ALBEDO_LUT['bare_soil']})")

    # Priority stack: apply from lowest to highest priority
    # Each layer overwrites previous where its mask is present

    # 1. Nature strips / medians (grass-like)
    _apply_layer(albedo, AUX_DIR / "nature_strip.tif", "nature_strip", "Nature strips")

    # 2. Tree canopy from LiDAR-mapped polygons (2018)
    _apply_layer(albedo, AUX_DIR / "tree_canopy.tif", "tree_canopy", "Tree canopy (LiDAR)")

    # 2b. Also use CDSM for canopy if provided (fills gaps in LiDAR polygons)
    if cdsm_path and Path(cdsm_path).exists():
        logger.info(f"  Applying tree canopy from CDSM (gap fill)...")
        with rasterio.open(str(cdsm_path)) as src:
            cdsm = src.read(1)
            if cdsm.shape == (height, width):
                tree_mask = cdsm > 1.0
                # Only apply where not already set to tree_canopy
                new_trees = tree_mask & (albedo != ALBEDO_LUT["tree_canopy"])
                albedo[new_trees] = ALBEDO_LUT["tree_canopy"]
                count = np.count_nonzero(new_trees)
                logger.info(f"  [Tree canopy (CDSM)] {count:,} extra px → albedo {ALBEDO_LUT['tree_canopy']}")

    # 3. Building roofs
    _apply_layer(albedo, AUX_DIR / "buildings.tif", "roof", "Building roofs")

    # 4. Footways / sidewalks (concrete, higher albedo than asphalt)
    _apply_layer(albedo, AUX_DIR / "footway.tif", "footway", "Footways/sidewalks")

    # 5. Carriageways / roads (asphalt, low albedo)
    _apply_layer(albedo, AUX_DIR / "carriageway.tif", "carriageway", "Carriageways/roads")

    # 6. Water bodies (highest priority — lowest albedo)
    _apply_layer(albedo, AUX_DIR / "water.tif", "water", "Water bodies")

    # Statistics
    logger.info(f"\n  Albedo statistics:")
    logger.info(f"    Min:  {albedo.min():.3f}")
    logger.info(f"    Max:  {albedo.max():.3f}")
    logger.info(f"    Mean: {albedo.mean():.3f}")
    logger.info(f"    Std:  {albedo.std():.3f}")

    # Unique values distribution
    unique_vals, counts = np.unique(albedo, return_counts=True)
    for val, cnt in zip(unique_vals, counts):
        pct = (cnt / albedo.size) * 100
        # Reverse-lookup surface type
        surface = next((k for k, v in ALBEDO_LUT.items() if abs(v - val) < 0.001), "unknown")
        logger.info(f"    {val:.2f} ({surface}): {cnt:>10,} px ({pct:.1f}%)")

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_profile = profile.copy()
    write_profile.update(
        dtype="float32",
        count=1,
        nodata=-9999.0,
        compress="deflate",
    )

    with rasterio.open(str(output_path), "w", **write_profile) as dst:
        dst.write(albedo, 1)
        dst.update_tags(
            description="Albedo Map for City of Melbourne (Phase 1.6.1)",
            source="LUT-based vector stacking",
            unit="ratio (0.0-1.0)",
        )

    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"\n  Output: {output_path} ({size_mb:.1f} MB)")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate Albedo Map from rasterized vector layers",
    )
    parser.add_argument(
        "--with-cdsm",
        type=str,
        default=None,
        help="Path to CDSM COG (optional, for tree canopy gap-fill)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(RAW_DIR / "albedo_master.tif"),
        help="Output path for albedo master raster",
    )

    args = parser.parse_args()

    logger.info("Urban Heat Albedo Map Generation")
    logger.info(f"Albedo LUT: {ALBEDO_LUT}")
    logger.info(f"Auxiliary rasters dir: {AUX_DIR}")

    success = generate_albedo(
        cdsm_path=args.with_cdsm,
        output_path=Path(args.output),
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
