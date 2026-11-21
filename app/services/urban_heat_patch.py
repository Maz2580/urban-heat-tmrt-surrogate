"""
Urban Heat Patch Extraction Service (Phase 1.6.4)
==================================================

Extracts a 512x512 pixel (1m/pixel) window from the DSM, CDSM, and
Albedo COGs given a WGS84 center coordinate.  Returns a normalized
(3, 512, 512) float32 array ready for solar channel injection and
ONNX inference.

COG CRS: EPSG:32755 (UTM Zone 55S)
Input CRS: EPSG:4326 (WGS84)
Resolution: 1m/pixel → 512m x 512m coverage
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger("umami.urban_heat_patch")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "urban_heat_data"
COG_DIR = DATA_ROOT / "rasters" / "cogs"
DATASET_STATS_PATH = DATA_ROOT / "models" / "dataset_stats.json"

PATCH_SIZE = 512  # pixels (1m/pixel)

# COG filenames matching LAYER_CATALOG in urban_heat.py
COG_FILES = {
    "dsm": "dsm_cog.tif",
    "dem": "dem_cog.tif",
    "cdsm": "cdsm_cog.tif",
    "albedo": "albedo_master_cog.tif",
}

# Default normalization ranges (overridden by dataset_stats.json if available)
DEFAULT_NORM = {
    "dsm_range": [0, 300.0],
    "cdsm_range": [0, 50.0],
    "albedo_range": [0, 1.0],
}


@dataclass
class PatchResult:
    """Result of patch extraction."""
    array: np.ndarray           # (3, 512, 512) float32 normalized
    bbox_4326: Tuple[float, float, float, float]  # (west, south, east, north) in EPSG:4326
    bbox_32755: Tuple[float, float, float, float]  # (west, south, east, north) in EPSG:32755
    center_pixel: Tuple[int, int]  # (row, col) in the source raster


def _load_normalization() -> dict:
    """Load normalization ranges from dataset_stats.json."""
    if DATASET_STATS_PATH.exists():
        with open(DATASET_STATS_PATH) as f:
            stats = json.load(f)
            return stats.get("normalization", DEFAULT_NORM)
    return DEFAULT_NORM


def _reproject_point(lon: float, lat: float) -> Tuple[float, float]:
    """Reproject WGS84 (lon, lat) to EPSG:32755 (easting, northing)."""
    from pyproj import Transformer
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32755", always_xy=True)
    easting, northing = transformer.transform(lon, lat)
    return easting, northing


def _reproject_bbox_to_4326(
    west: float, south: float, east: float, north: float
) -> Tuple[float, float, float, float]:
    """Reproject EPSG:32755 bbox to EPSG:4326."""
    from pyproj import Transformer
    transformer = Transformer.from_crs("EPSG:32755", "EPSG:4326", always_xy=True)
    lon_w, lat_s = transformer.transform(west, south)
    lon_e, lat_n = transformer.transform(east, north)
    return (lon_w, lat_s, lon_e, lat_n)


def extract_patch(
    center_lat: float,
    center_lon: float,
) -> PatchResult:
    """
    Extract a 512x512 normalized patch from the three COGs.

    Args:
        center_lat: Center latitude (EPSG:4326)
        center_lon: Center longitude (EPSG:4326)

    Returns:
        PatchResult with (3, 512, 512) array [DSM/300, CDSM/50, Albedo/1.0]

    Raises:
        ValueError: If center point is outside COG extent
        FileNotFoundError: If COG files are missing
    """
    import rasterio
    from rasterio.windows import Window

    norm = _load_normalization()
    half = PATCH_SIZE // 2  # 256 pixels

    # Reproject center to UTM
    easting, northing = _reproject_point(center_lon, center_lat)

    # UTM bbox (512m x 512m)
    bbox_32755 = (
        easting - half,   # west
        northing - half,  # south
        easting + half,   # east
        northing + half,  # north
    )

    # Reproject bbox to WGS84 for frontend
    bbox_4326 = _reproject_bbox_to_4326(*bbox_32755)

    channels = []
    center_pixel = None

    for layer_name, norm_key in [("dsm", "dsm_range"), ("cdsm", "cdsm_range"), ("albedo", "albedo_range")]:
        cog_path = COG_DIR / COG_FILES[layer_name]
        if not cog_path.exists():
            raise FileNotFoundError(f"COG not found: {cog_path}")

        with rasterio.open(str(cog_path)) as src:
            # Convert UTM center to pixel coordinates
            row, col = src.index(easting, northing)

            if center_pixel is None:
                center_pixel = (row, col)

            # Calculate window (top-left origin)
            row_start = row - half
            col_start = col - half

            # Check bounds
            if (
                row_start < 0
                or col_start < 0
                or row_start + PATCH_SIZE > src.height
                or col_start + PATCH_SIZE > src.width
            ):
                # Handle partial coverage: read what we can, pad with zeros
                # Calculate valid region
                r_start = max(0, row_start)
                c_start = max(0, col_start)
                r_end = min(src.height, row_start + PATCH_SIZE)
                c_end = min(src.width, col_start + PATCH_SIZE)

                if r_start >= r_end or c_start >= c_end:
                    raise ValueError(
                        f"Center ({center_lat}, {center_lon}) is completely outside "
                        f"COG extent for {layer_name}"
                    )

                window = Window(c_start, r_start, c_end - c_start, r_end - r_start)
                data = src.read(1, window=window).astype(np.float32)

                # Pad to 512x512
                padded = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
                pad_row = r_start - row_start
                pad_col = c_start - col_start
                padded[pad_row:pad_row + data.shape[0], pad_col:pad_col + data.shape[1]] = data
                data = padded

                logger.warning(
                    "Partial coverage for %s at (%f, %f) — padded with zeros",
                    layer_name, center_lat, center_lon,
                )
            else:
                window = Window(col_start, row_start, PATCH_SIZE, PATCH_SIZE)
                data = src.read(1, window=window).astype(np.float32)

            # Handle nodata
            if src.nodata is not None:
                data = np.where(data == src.nodata, 0.0, data)

            # Normalize
            norm_range = norm.get(norm_key, DEFAULT_NORM[norm_key])
            norm_max = norm_range[1]
            if norm_max > 0:
                data = np.clip(data / norm_max, 0.0, 1.0)

            channels.append(data)

    # Stack to (3, 512, 512)
    patch_array = np.stack(channels, axis=0)

    return PatchResult(
        array=patch_array,
        bbox_4326=bbox_4326,
        bbox_32755=bbox_32755,
        center_pixel=center_pixel,
    )


def get_cog_bounds() -> Optional[Tuple[float, float, float, float]]:
    """
    Get the WGS84 bounding box of the DSM COG.
    Used for validating input coordinates.
    Returns (west, south, east, north) or None if COG not found.
    """
    dsm_path = COG_DIR / COG_FILES["dsm"]
    if not dsm_path.exists():
        return None

    try:
        import rasterio
        with rasterio.open(str(dsm_path)) as src:
            bounds = src.bounds
            # COG is in EPSG:32755, need to reproject bounds
            return _reproject_bbox_to_4326(
                bounds.left, bounds.bottom, bounds.right, bounds.top
            )
    except Exception as e:
        logger.error("Failed to read COG bounds: %s", e)
        return None
