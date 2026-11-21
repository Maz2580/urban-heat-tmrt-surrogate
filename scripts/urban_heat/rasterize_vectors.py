#!/usr/bin/env python3
"""
Urban Heat Tool — Vector Rasterization Script (Phase 1.6.1)

Converts GeoJSON vector layers to 1m resolution GeoTIFFs aligned
to a template raster grid (DSM). Used as intermediate step for
albedo map generation.

Outputs:
  urban_heat_data/rasters/auxiliary/buildings.tif
  urban_heat_data/rasters/auxiliary/roads.tif
  urban_heat_data/rasters/auxiliary/water.tif

Usage:
  # Rasterize all vectors using DSM as template grid:
  python rasterize_vectors.py --template urban_heat_data/rasters/cogs/dsm_cog.tif

  # Rasterize a single layer:
  python rasterize_vectors.py --template dsm_cog.tif --layer buildings

  # Custom resolution (default 1m):
  python rasterize_vectors.py --template dsm_cog.tif --resolution 2.0
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("urban_heat.rasterize")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_ROOT = PROJECT_ROOT / "urban_heat_data"
VECTOR_DIR = DATA_ROOT / "vectors"
AUX_DIR = DATA_ROOT / "rasters" / "auxiliary"

# Vector layers to rasterize
# filter_property/filter_values: only rasterize features matching these property values
RASTER_LAYERS = {
    "buildings": {
        "input": "buildings_com_2023.geojson",
        "output": "buildings.tif",
        "burn_value": 1.0,
        "description": "Building footprint mask (1=building, 0=no building)",
    },
    "carriageway": {
        "input": "road_segments_surface.geojson",
        "output": "carriageway.tif",
        "burn_value": 1.0,
        "filter_property": "type",
        "filter_values": ["Carriageway", "Tramway", "Bridge"],
        "description": "Carriageway/road surface mask (asphalt)",
    },
    "footway": {
        "input": "road_segments_surface.geojson",
        "output": "footway.tif",
        "burn_value": 1.0,
        "filter_property": "type",
        "filter_values": ["Footway", "Road Channel", "Road Kerb"],
        "description": "Footway/sidewalk surface mask (concrete)",
    },
    "nature_strip": {
        "input": "road_segments_surface.geojson",
        "output": "nature_strip.tif",
        "burn_value": 1.0,
        "filter_property": "type",
        "filter_values": ["Median"],
        "description": "Nature strip / median mask (grass/vegetation)",
    },
    "tree_canopy": {
        "input": "tree_canopies_2018.geojson",
        "output": "tree_canopy.tif",
        "burn_value": 1.0,
        "description": "LiDAR-mapped tree canopy mask (2018)",
    },
    "water": {
        "input": "water_areas.geojson",
        "output": "water.tif",
        "burn_value": 1.0,
        "description": "Water body mask (1=water, 0=no water)",
    },
}


def get_raster_profile(
    template_path: Optional[Path] = None,
    bounds: Optional[Tuple[float, float, float, float]] = None,
    resolution: float = 1.0,
) -> dict:
    """
    Build a rasterio profile from a template raster or from bounds+resolution.
    Returns profile dict compatible with rasterio.open().
    """
    import rasterio
    from rasterio.transform import from_bounds

    if template_path and template_path.exists():
        with rasterio.open(str(template_path)) as src:
            profile = src.profile.copy()
            profile.update(
                dtype="float32",
                count=1,
                nodata=0.0,
                compress="deflate",
            )
            return profile

    # Fallback: create profile from CoM bounding box
    if bounds is None:
        # City of Melbourne approximate bounds (EPSG:28355 / MGA Zone 55)
        # For 1m resolution, we work in projected CRS
        bounds = (144.90, -37.84, 144.99, -37.78)

    west, south, east, north = bounds
    # Approximate pixel count at given resolution (degrees)
    # 1 degree lat ~ 111,000m, 1 degree lon ~ 82,000m at -37.8 lat
    width = int((east - west) * 82000 / resolution)
    height = int((north - south) * 111000 / resolution)

    transform = from_bounds(west, south, east, north, width, height)

    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "width": width,
        "height": height,
        "count": 1,
        "crs": "EPSG:4326",
        "transform": transform,
        "nodata": 0.0,
        "compress": "deflate",
    }
    return profile


def rasterize_geojson(
    geojson_path: Path,
    output_path: Path,
    profile: dict,
    burn_value: float = 1.0,
    filter_property: str = None,
    filter_values: list = None,
) -> bool:
    """
    Rasterize a GeoJSON file to a GeoTIFF using rasterio.features.rasterize.
    Burns `burn_value` for all features (or filtered subset), 0 elsewhere.
    """
    import rasterio
    from rasterio.features import rasterize
    import json

    if not geojson_path.exists():
        logger.error(f"  Input not found: {geojson_path}")
        return False

    # Load geometries from GeoJSON
    logger.info(f"  Loading {geojson_path.name}...")
    with open(geojson_path, "r") as f:
        data = json.load(f)

    features = data.get("features", [])
    if not features:
        logger.warning(f"  No features in {geojson_path.name}")
        return False

    logger.info(f"  Total features in file: {len(features)}")

    # Check if we need to reproject geometries (GeoJSON is EPSG:4326)
    target_crs = profile.get("crs")
    need_reproject = target_crs and str(target_crs) != "EPSG:4326"
    transformer = None
    if need_reproject:
        from pyproj import Transformer
        from shapely.geometry import shape as shapely_shape
        from shapely.ops import transform as shapely_transform
        from shapely.geometry import mapping as shapely_mapping
        transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
        logger.info(f"  Reprojecting geometries: EPSG:4326 → {target_crs}")

    # Extract geometry shapes for rasterize(), optionally filtering
    geometries = []
    for feat in features:
        # Apply property filter if specified
        if filter_property and filter_values:
            props = feat.get("properties", {})
            val = props.get(filter_property)
            if val not in filter_values:
                continue

        geom = feat.get("geometry")
        if geom and geom.get("type") and geom.get("coordinates"):
            # Reproject geometry if needed
            if transformer:
                try:
                    shp = shapely_shape(geom)
                    shp_proj = shapely_transform(transformer.transform, shp)
                    geom = shapely_mapping(shp_proj)
                except Exception:
                    continue
            geometries.append((geom, burn_value))

    if not geometries:
        logger.warning(f"  No valid geometries found")
        return False

    logger.info(f"  Rasterizing {len(geometries)} geometries -> {output_path.name}")
    logger.info(f"  Grid: {profile['width']}x{profile['height']} pixels")

    # Rasterize
    raster = rasterize(
        shapes=geometries,
        out_shape=(profile["height"], profile["width"]),
        transform=profile["transform"],
        fill=0.0,
        dtype="float32",
        all_touched=True,
    )

    # Count burned pixels
    burned = np.count_nonzero(raster)
    total = raster.size
    pct = (burned / total) * 100 if total > 0 else 0
    logger.info(f"  Burned {burned:,} pixels ({pct:.1f}% coverage)")

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(str(output_path), "w", **profile) as dst:
        dst.write(raster, 1)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"  Written: {output_path} ({size_mb:.1f} MB)")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Rasterize vector layers for Urban Heat Tool albedo generation",
    )
    parser.add_argument(
        "--template",
        type=str,
        help="Template raster (e.g., DSM COG) to match grid/resolution/CRS",
    )
    parser.add_argument(
        "--layer",
        choices=list(RASTER_LAYERS.keys()),
        help="Rasterize a single layer. If omitted, rasterizes all.",
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=1.0,
        help="Target resolution in meters (default: 1.0). Only used when no template.",
    )

    args = parser.parse_args()

    logger.info("Urban Heat Vector Rasterization")
    logger.info(f"Vector dir: {VECTOR_DIR}")
    logger.info(f"Output dir: {AUX_DIR}")

    # Get raster profile
    template = Path(args.template) if args.template else None
    if template and not template.is_absolute():
        template = PROJECT_ROOT / template

    profile = get_raster_profile(
        template_path=template,
        resolution=args.resolution,
    )

    logger.info(f"Grid: {profile['width']}x{profile['height']}, CRS: {profile.get('crs', 'unknown')}")

    # Rasterize layers
    layers = {args.layer: RASTER_LAYERS[args.layer]} if args.layer else RASTER_LAYERS
    results = {}

    for key, layer_def in layers.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"Layer: {key} — {layer_def['description']}")

        input_path = VECTOR_DIR / layer_def["input"]
        output_path = AUX_DIR / layer_def["output"]

        results[key] = rasterize_geojson(
            geojson_path=input_path,
            output_path=output_path,
            profile=profile,
            burn_value=layer_def["burn_value"],
            filter_property=layer_def.get("filter_property"),
            filter_values=layer_def.get("filter_values"),
        )

    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("RASTERIZATION SUMMARY")
    for key, success in results.items():
        status = "OK" if success else "FAILED"
        logger.info(f"  [{status}] {key}")

    failed = [k for k, v in results.items() if not v]
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
