"""
Urban Heat Feature Engineering (Phase 1.6.9)
=============================================

Computes additional input features from existing raster data (DSM, DEM, CDSM,
Albedo) for each training patch. These features are suggested by the LLM error
advisor to improve the V2 model.

Available features:
  - svf: Sky View Factor (16-azimuth horizon scanning)
  - building_density: Fraction of building pixels within 50m radius
  - tree_distance: Euclidean distance to nearest tree canopy
  - aspect: Terrain aspect (sin + cos components from DEM gradient)
  - shadow_fraction: Approximate shadow coverage for a given solar position

Each feature is saved as {patch_name}_{feature}.npy alongside existing data.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rasterio
from scipy import ndimage

logger = logging.getLogger("umami.urban_heat.feature_engineering")

PATCHES_DIR = Path("urban_heat_data/training/patches")
FEATURES_MANIFEST = Path("urban_heat_data/training/features_manifest.json")

# Feature normalization ranges (for input to neural network)
FEATURE_NORMS = {
    "svf": (0.0, 1.0),
    "building_density": (0.0, 1.0),
    "tree_distance": (0.0, 200.0),   # clamp at 200m
    "aspect_sin": (-1.0, 1.0),
    "aspect_cos": (-1.0, 1.0),
    "shadow_fraction": (0.0, 1.0),
}


# ---------------------------------------------------------------------------
# Raster loading helpers
# ---------------------------------------------------------------------------

def _load_patch_rasters(patch_dir: Path) -> Dict[str, np.ndarray]:
    """Load DSM, DEM, CDSM, Albedo from a patch directory."""
    rasters = {}
    for name, filename in [
        ("dsm", "building_dsm.tif"),
        ("dem", "dem.tif"),
        ("cdsm", "trees.tif"),
        ("albedo", "albedo.tif"),
    ]:
        fpath = patch_dir / filename
        if fpath.exists():
            with rasterio.open(fpath) as ds:
                data = ds.read(1).astype(np.float32)
                nodata = ds.nodata
                if nodata is not None:
                    data[data == nodata] = 0.0
                # Handle sentinel values
                data[data < -9000] = 0.0
                rasters[name] = data
        else:
            rasters[name] = None
    return rasters


# ---------------------------------------------------------------------------
# Feature computation functions
# ---------------------------------------------------------------------------

def compute_svf(dsm: np.ndarray, n_azimuths: int = 16, max_radius: int = 100, step: int = 2) -> np.ndarray:
    """
    Sky View Factor: fraction of visible sky hemisphere.

    Uses simplified horizon scanning at n_azimuths directions.
    SVF=1.0 means fully open sky, SVF→0 means heavily obstructed.
    """
    h, w = dsm.shape
    svf = np.zeros((h, w), dtype=np.float32)

    for i in range(n_azimuths):
        angle = 2 * np.pi * i / n_azimuths
        dx = np.cos(angle)
        dy = np.sin(angle)

        max_elev = np.zeros((h, w), dtype=np.float32)

        for dist in range(step, max_radius + 1, step):
            shift_y = int(round(dist * dy))
            shift_x = int(round(dist * dx))

            shifted = np.full_like(dsm, dsm.min())
            sy0 = max(0, -shift_y)
            sy1 = min(h, h - shift_y)
            sx0 = max(0, -shift_x)
            sx1 = min(w, w - shift_x)
            dy0 = max(0, shift_y)
            dy1 = min(h, h + shift_y)
            dx0 = max(0, shift_x)
            dx1 = min(w, w + shift_x)

            shifted[dy0:dy1, dx0:dx1] = dsm[sy0:sy1, sx0:sx1]

            elev_angle = np.arctan2(shifted - dsm, float(dist))
            max_elev = np.maximum(max_elev, elev_angle)

        svf += np.cos(max_elev)

    svf /= n_azimuths
    return svf


def compute_building_density(dsm: np.ndarray, dem: np.ndarray, radius: int = 50) -> np.ndarray:
    """
    Building density: fraction of building pixels within a given radius.

    Buildings detected as DSM - DEM > 2m.
    Uses 'reflect' padding to avoid edge artifacts from the uniform filter.
    """
    building_height = dsm - dem
    building_mask = (building_height > 2.0).astype(np.float32)
    kernel_size = 2 * radius + 1
    # Pad with reflection to avoid edge artifacts (pixels near borders
    # would otherwise get artificially low density)
    padded = np.pad(building_mask, radius, mode='reflect')
    density = ndimage.uniform_filter(padded, size=kernel_size)
    # Crop back to original size
    density = density[radius:-radius, radius:-radius]
    return density.astype(np.float32)


def compute_tree_distance(cdsm: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Distance to nearest tree canopy pixel (in meters at 1m resolution).

    Clamped to 200m and normalized to [0, 1].
    """
    tree_mask = cdsm > threshold
    if not tree_mask.any():
        return np.full(cdsm.shape, 200.0, dtype=np.float32)

    dist = ndimage.distance_transform_edt(~tree_mask).astype(np.float32)
    return np.clip(dist, 0.0, 200.0)


def compute_aspect(dem: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Terrain aspect from DEM gradient, returned as (sin, cos) components.

    Using sin/cos avoids the discontinuity at 0°/360°.
    """
    dy, dx = np.gradient(dem, 1.0)
    aspect_rad = np.arctan2(-dx, dy)
    return np.sin(aspect_rad).astype(np.float32), np.cos(aspect_rad).astype(np.float32)


def compute_shadow_fraction(
    dsm: np.ndarray,
    sun_altitude_deg: float = 45.0,
    sun_azimuth_deg: float = 0.0,
    max_distance: int = 100,
) -> np.ndarray:
    """
    Approximate shadow fraction using array-shift ray tracing.

    For each pixel, trace a ray toward the sun and check if any DSM
    pixel along the ray is tall enough to cast a shadow.
    """
    alt_rad = np.radians(sun_altitude_deg)
    az_rad = np.radians(sun_azimuth_deg)

    # Shadow direction (toward sun, in image coords: east=+x, north=-y)
    dx = np.sin(az_rad)
    dy = -np.cos(az_rad)
    tan_alt = np.tan(alt_rad) if alt_rad > 0.01 else 100.0

    h, w = dsm.shape
    shadow = np.zeros((h, w), dtype=np.float32)

    for dist in range(1, max_distance + 1, 2):
        shift_y = int(round(dist * dy))
        shift_x = int(round(dist * dx))

        shifted = np.full_like(dsm, dsm.min())
        sy0 = max(0, -shift_y)
        sy1 = min(h, h - shift_y)
        sx0 = max(0, -shift_x)
        sx1 = min(w, w - shift_x)
        dy0 = max(0, shift_y)
        dy1 = min(h, h + shift_y)
        dx0 = max(0, shift_x)
        dx1 = min(w, w + shift_x)

        shifted[dy0:dy1, dx0:dx1] = dsm[sy0:sy1, sx0:sx1]

        # Shadow threshold: shifted DSM must be taller by dist*tan(alt)
        height_needed = dist * tan_alt
        in_shadow = (shifted - dsm) > height_needed
        shadow = np.maximum(shadow, in_shadow.astype(np.float32))

    return shadow


# ---------------------------------------------------------------------------
# Feature registry
# ---------------------------------------------------------------------------

FEATURE_REGISTRY = {
    "svf": {
        "description": "Sky View Factor (0=obstructed, 1=open sky)",
        "requires": ["dsm"],
        "channels": 1,
    },
    "building_density": {
        "description": "Building pixel fraction within 50m radius",
        "requires": ["dsm", "dem"],
        "channels": 1,
    },
    "tree_distance": {
        "description": "Distance to nearest tree canopy (meters, clamped 200m)",
        "requires": ["cdsm"],
        "channels": 1,
    },
    "aspect": {
        "description": "Terrain aspect as sin/cos pair from DEM gradient",
        "requires": ["dem"],
        "channels": 2,  # sin + cos
    },
    "shadow_fraction": {
        "description": "Binary shadow map at 45° altitude (static proxy for shadowability; dynamic shadow is captured by solar input channels)",
        "requires": ["dsm"],
        "channels": 1,
    },
}


def get_available_features() -> Dict[str, Any]:
    """Return registry of computable features."""
    return FEATURE_REGISTRY


# ---------------------------------------------------------------------------
# Batch computation
# ---------------------------------------------------------------------------

def _compute_feature_for_patch(
    patch_name: str,
    feature_name: str,
    rasters: Dict[str, np.ndarray],
) -> Optional[np.ndarray]:
    """Compute a single feature for a single patch. Returns (C, H, W) array."""
    if feature_name == "svf":
        if rasters["dsm"] is None:
            return None
        result = compute_svf(rasters["dsm"])
        return result[np.newaxis, :, :]

    elif feature_name == "building_density":
        if rasters["dsm"] is None or rasters["dem"] is None:
            return None
        result = compute_building_density(rasters["dsm"], rasters["dem"])
        return result[np.newaxis, :, :]

    elif feature_name == "tree_distance":
        if rasters["cdsm"] is None:
            return None
        result = compute_tree_distance(rasters["cdsm"])
        # Normalize to [0, 1]
        result = result / 200.0
        return result[np.newaxis, :, :]

    elif feature_name == "aspect":
        if rasters["dem"] is None:
            return None
        sin_asp, cos_asp = compute_aspect(rasters["dem"])
        return np.stack([sin_asp, cos_asp], axis=0)

    elif feature_name == "shadow_fraction":
        if rasters["dsm"] is None:
            return None
        result = compute_shadow_fraction(rasters["dsm"])
        return result[np.newaxis, :, :]

    else:
        logger.warning("Unknown feature: %s", feature_name)
        return None


def compute_features_batch(
    feature_name: str,
    patch_limit: Optional[int] = None,
    patch_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute a feature for multiple patches and save as .npy files.

    Args:
        feature_name: Name from FEATURE_REGISTRY
        patch_limit: Max patches to process (None = all)
        patch_names: Specific patches to process (None = all with input.npy)

    Returns:
        Summary dict with counts and timing.
    """
    if feature_name not in FEATURE_REGISTRY:
        return {"error": f"Unknown feature: {feature_name}", "computed": 0}

    # Find patches that have input.npy files
    if patch_names:
        patches = [PATCHES_DIR / f"{name}_input.npy" for name in patch_names]
        patches = [p for p in patches if p.exists()]
    else:
        patches = sorted(PATCHES_DIR.glob("*_input.npy"))

    if patch_limit:
        patches = patches[:patch_limit]

    total = len(patches)
    computed = 0
    skipped = 0
    errors = 0
    t_start = time.time()

    for i, input_path in enumerate(patches):
        patch_name = input_path.name.replace("_input.npy", "")
        output_path = PATCHES_DIR / f"{patch_name}_{feature_name}.npy"

        # Skip if already computed
        if output_path.exists():
            skipped += 1
            continue

        patch_dir = PATCHES_DIR / patch_name
        if not patch_dir.exists():
            skipped += 1
            continue

        try:
            rasters = _load_patch_rasters(patch_dir)

            # Check for valid data (skip patches with all-zero/nodata DSM)
            if rasters["dsm"] is not None and rasters["dsm"].max() < 0.01:
                skipped += 1
                continue

            result = _compute_feature_for_patch(patch_name, feature_name, rasters)
            if result is not None:
                np.save(output_path, result)
                computed += 1
            else:
                skipped += 1

        except Exception as e:
            logger.warning("Failed to compute %s for %s: %s", feature_name, patch_name, e)
            errors += 1

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            logger.info(
                "[FeatureEng] %s: %d/%d patches (%.1f/s), computed=%d, skipped=%d",
                feature_name, i + 1, total, rate, computed, skipped,
            )

    elapsed = time.time() - t_start

    # Update manifest
    _update_manifest(feature_name, computed)

    result = {
        "feature": feature_name,
        "total_patches": total,
        "computed": computed,
        "skipped": skipped,
        "errors": errors,
        "elapsed_seconds": round(elapsed, 1),
        "patches_per_second": round(total / max(elapsed, 0.001), 2),
    }

    logger.info("[FeatureEng] %s complete: %s", feature_name, result)
    return result


def _update_manifest(feature_name: str, count: int):
    """Update the features manifest with computation results."""
    manifest = {}
    if FEATURES_MANIFEST.exists():
        with open(FEATURES_MANIFEST) as f:
            manifest = json.load(f)

    if feature_name not in manifest:
        manifest[feature_name] = {
            "info": FEATURE_REGISTRY.get(feature_name, {}),
            "normalization": FEATURE_NORMS.get(feature_name, (0.0, 1.0)),
            "computed_patches": 0,
        }

    manifest[feature_name]["computed_patches"] = (
        manifest[feature_name].get("computed_patches", 0) + count
    )
    manifest[feature_name]["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    FEATURES_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(FEATURES_MANIFEST, "w") as f:
        json.dump(manifest, f, indent=2)


def get_computed_features() -> Dict[str, Any]:
    """List all features and their computation status."""
    manifest = {}
    if FEATURES_MANIFEST.exists():
        with open(FEATURES_MANIFEST) as f:
            manifest = json.load(f)

    result = {}
    for name, info in FEATURE_REGISTRY.items():
        computed = manifest.get(name, {}).get("computed_patches", 0)
        result[name] = {
            **info,
            "computed_patches": computed,
            "normalization": FEATURE_NORMS.get(name, (0.0, 1.0)),
        }

    return result
