#!/usr/bin/env python3
"""
Urban Heat Tool — Training Data Packager (Phase 1.6.2)

Converts GPU-generated T_mrt GeoTIFFs into normalized .npy training pairs
ready for Phase 1.6.3 (surrogate model training).

Input tensor (3, 512, 512) float32 — per supervisor's Task 3.1:
  Channel 0 (R): DSM normalized (0-300m → 0.0-1.0)
  Channel 1 (G): CDSM normalized (0-50m → 0.0-1.0)
  Channel 2 (B): Albedo (already 0.0-1.0)

Target tensor (1, 512, 512) float32:
  Channel 0: T_mrt normalized (0-63°C → 0.0-1.0)

For mutation patches (_green, _cool), input rasters are recreated from
the base patch using the same deterministic logic as fix.ipynb on Colab.

Output:
  training/patches/patch_NNNN_input.npy             (3, 512, 512)
  training/patches/patch_NNNN_d015_h13_target.npy   (1, 512, 512)
  training/patches/patch_NNNN_green_input.npy        (3, 512, 512)
  ...
  training/dataset_stats.json

Usage:
  python package_training_data.py
  python package_training_data.py --dsm-max 300 --cdsm-max 50 --tmrt-max 63
"""

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

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

TRAINING_DIR = DATA_ROOT / "training"
PATCHES_DIR = TRAINING_DIR / "patches"

# Normalization ranges (per supervisor's Task 3.1)
DEFAULT_DSM_MAX = 300.0    # meters
DEFAULT_CDSM_MAX = 50.0    # meters
DEFAULT_ALBEDO_MAX = 1.0   # ratio (already 0-1)
DEFAULT_TMRT_MAX = 63.0    # degrees C

# Nodata threshold — skip patches with more than this fraction of nodata
MAX_NODATA_FRACTION = 0.70

# Mutation constants (must match fix.ipynb exactly)
ALBEDO_ASPHALT = 0.12
ALBEDO_GRASS = 0.25
ALBEDO_ROOF_COOL = 0.70
ALBEDO_ROOF_STD = 0.15
LC_PAVED = 1
LC_BUILDINGS = 2
LC_VEGETATION = 4


def read_raster(path: Path) -> np.ndarray:
    """Read single-band raster as float32 array, preserving nodata as -9999."""
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        nodata = src.nodata
        if nodata is not None:
            data[data == nodata] = -9999.0
        data[np.isnan(data)] = -9999.0
        return data


def read_raster_clean(path: Path) -> np.ndarray:
    """Read single-band raster, replacing nodata/NaN with 0."""
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        nodata = src.nodata
        if nodata is not None:
            data[data == nodata] = 0.0
        data[np.isnan(data)] = 0.0
        return data


def normalize(data: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Normalize to [0, 1] range, clamping outliers."""
    return np.clip((data - vmin) / (vmax - vmin), 0.0, 1.0)


def find_tmrt_files(output_dir: Path) -> List[Tuple[str, Path]]:
    """Find T_mrt output files and extract temporal labels.

    Handles filename format: Tmrt_d015_h13.tif → label 'd015_h13'
    """
    results = []
    if not output_dir.exists():
        return results

    for f in sorted(output_dir.glob("Tmrt_*.tif")):
        parts = f.stem.split("_")
        # Format: Tmrt_d015_h13 → parts = ['Tmrt', 'd015', 'h13']
        if len(parts) == 3 and parts[1].startswith("d") and parts[2].startswith("h"):
            label = f"{parts[1]}_{parts[2]}"
            results.append((label, f))

    return results


def get_base_patch_name(patch_name: str) -> str:
    """Extract base patch name from a mutation name.

    'patch_0007_green' → 'patch_0007'
    'patch_0007_cool'  → 'patch_0007'
    'patch_0007'       → 'patch_0007'
    """
    for suffix in ("_green", "_cool"):
        if patch_name.endswith(suffix):
            return patch_name[: -len(suffix)]
    return patch_name


def recreate_greening_mutation(base_dir: Path, mut_dir: Path) -> None:
    """Recreate greening mutation inputs (deterministic, matches fix.ipynb)."""
    patch_name = mut_dir.name
    seed = hash(get_base_patch_name(patch_name)) % (2**31)
    rng = np.random.RandomState(seed)

    # Copy DSM and DEM as-is
    for f in ["building_dsm.tif", "dem.tif"]:
        shutil.copy2(base_dir / f, mut_dir / f)

    # Modify trees: add 50 random trees
    with rasterio.open(base_dir / "trees.tif") as src:
        trees = src.read(1).astype(np.float32)
        profile = src.profile.copy()

    # Load landcover for plantable area detection
    lc_path = base_dir / "landcover.tif"
    if lc_path.exists():
        with rasterio.open(lc_path) as src:
            lc = src.read(1)
        plantable = ((lc == LC_PAVED) | (lc == 5)) & (trees < 1.0)
    else:
        plantable = trees < 1.0

    coords = np.argwhere(plantable)
    if len(coords) > 0:
        n_trees = min(50, len(coords))
        for idx in rng.choice(len(coords), n_trees, replace=False):
            cy, cx = coords[idx]
            h = rng.uniform(8, 15)
            r = rng.randint(3, 7)
            y0, y1 = max(0, cy - r), min(trees.shape[0], cy + r + 1)
            x0, x1 = max(0, cx - r), min(trees.shape[1], cx + r + 1)
            yy, xx = np.ogrid[y0:y1, x0:x1]
            mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r**2
            trees[y0:y1, x0:x1][mask] = np.maximum(trees[y0:y1, x0:x1][mask], h)

    with rasterio.open(mut_dir / "trees.tif", "w", **profile) as dst:
        dst.write(trees, 1)

    # Modify albedo: 30% asphalt → grass
    with rasterio.open(base_dir / "albedo.tif") as src:
        albedo = src.read(1).astype(np.float32)
        ap = src.profile.copy()

    asphalt = np.abs(albedo - ALBEDO_ASPHALT) < 0.02
    ac = np.argwhere(asphalt)
    if len(ac) > 0:
        n_convert = int(len(ac) * 0.3)
        for idx in rng.choice(len(ac), n_convert, replace=False):
            y, x = ac[idx]
            albedo[y, x] = ALBEDO_GRASS

    with rasterio.open(mut_dir / "albedo.tif", "w", **ap) as dst:
        dst.write(albedo, 1)

    # Copy landcover if exists
    if lc_path.exists():
        shutil.copy2(lc_path, mut_dir / "landcover.tif")


def recreate_cool_roof_mutation(base_dir: Path, mut_dir: Path) -> None:
    """Recreate cool roof mutation inputs (deterministic, matches fix.ipynb)."""
    # Copy DSM, DEM, trees as-is
    for f in ["building_dsm.tif", "dem.tif", "trees.tif"]:
        shutil.copy2(base_dir / f, mut_dir / f)

    # Modify albedo: building roofs 0.15 → 0.70
    with rasterio.open(base_dir / "albedo.tif") as src:
        albedo = src.read(1).astype(np.float32)
        profile = src.profile.copy()

    lc_path = base_dir / "landcover.tif"
    if lc_path.exists():
        with rasterio.open(lc_path) as src:
            lc = src.read(1)
        roof_mask = (lc == LC_BUILDINGS) & (np.abs(albedo - ALBEDO_ROOF_STD) < 0.03)
        albedo[roof_mask] = ALBEDO_ROOF_COOL
    else:
        # Fallback: change all ~0.15 albedo to cool roof
        roof_mask = np.abs(albedo - ALBEDO_ROOF_STD) < 0.03
        albedo[roof_mask] = ALBEDO_ROOF_COOL

    with rasterio.open(mut_dir / "albedo.tif", "w", **profile) as dst:
        dst.write(albedo, 1)

    if lc_path.exists():
        shutil.copy2(lc_path, mut_dir / "landcover.tif")


def main():
    parser = argparse.ArgumentParser(description="Package training data as .npy pairs")
    parser.add_argument("--dsm-max", type=float, default=DEFAULT_DSM_MAX)
    parser.add_argument("--cdsm-max", type=float, default=DEFAULT_CDSM_MAX)
    parser.add_argument("--tmrt-max", type=float, default=DEFAULT_TMRT_MAX)
    args = parser.parse_args()

    dsm_max = args.dsm_max
    cdsm_max = args.cdsm_max
    tmrt_max = args.tmrt_max

    logger.info(f"Normalization: DSM 0-{dsm_max}m, CDSM 0-{cdsm_max}m, T_mrt 0-{tmrt_max}°C")
    logger.info(f"Nodata skip threshold: {MAX_NODATA_FRACTION * 100:.0f}%")
    logger.info(f"Data root: {DATA_ROOT}")
    logger.info(f"Patches dir: {PATCHES_DIR}")

    # Find all patch directories
    all_dirs = sorted([
        d for d in PATCHES_DIR.iterdir()
        if d.is_dir() and d.name.startswith("patch_")
    ])

    if not all_dirs:
        logger.error(f"No patch directories found in {PATCHES_DIR}")
        sys.exit(1)

    # Separate base and mutation directories
    base_dirs = [d for d in all_dirs if "_green" not in d.name and "_cool" not in d.name]
    green_dirs = [d for d in all_dirs if d.name.endswith("_green")]
    cool_dirs = [d for d in all_dirs if d.name.endswith("_cool")]

    logger.info(f"Found {len(base_dirs)} base, {len(green_dirs)} green, {len(cool_dirs)} cool patches")

    # Step 1: Recreate mutation input rasters
    logger.info("Recreating mutation input rasters...")
    mut_recreated = 0
    for mut_dir in green_dirs + cool_dirs:
        base_name = get_base_patch_name(mut_dir.name)
        base_dir = PATCHES_DIR / base_name

        if not base_dir.exists():
            logger.warning(f"  Base patch {base_name} not found for {mut_dir.name}, skipping")
            continue

        # Check if inputs already exist (idempotent)
        if (mut_dir / "building_dsm.tif").exists():
            continue

        if mut_dir.name.endswith("_green"):
            recreate_greening_mutation(base_dir, mut_dir)
        elif mut_dir.name.endswith("_cool"):
            recreate_cool_roof_mutation(base_dir, mut_dir)

        mut_recreated += 1

    logger.info(f"  Recreated inputs for {mut_recreated} mutation patches")

    # Step 2: Package all patches into .npy pairs
    logger.info("Packaging .npy training pairs...")

    stats = {
        "normalization": {
            "dsm_range": [0, dsm_max],
            "cdsm_range": [0, cdsm_max],
            "albedo_range": [0, 1.0],
            "tmrt_range": [0, tmrt_max],
        },
        "input_pairs": 0,
        "input_tensors": 0,
        "base_patches": 0,
        "green_mutations": 0,
        "cool_mutations": 0,
        "skipped_nodata": 0,
        "skipped_missing": 0,
        "temporal_combos": set(),
        "dsm_stats": {"min": float("inf"), "max": float("-inf"), "sum": 0, "count": 0},
        "cdsm_stats": {"min": float("inf"), "max": float("-inf"), "sum": 0, "count": 0},
        "albedo_stats": {"min": float("inf"), "max": float("-inf"), "sum": 0, "count": 0},
        "tmrt_stats": {"min": float("inf"), "max": float("-inf"), "sum": 0, "count": 0},
    }

    # Re-scan all dirs (mutations now have input rasters)
    all_dirs = sorted([
        d for d in PATCHES_DIR.iterdir()
        if d.is_dir() and d.name.startswith("patch_")
    ])

    for patch_dir in all_dirs:
        patch_name = patch_dir.name

        # Check required input files
        dsm_path = patch_dir / "building_dsm.tif"
        cdsm_path = patch_dir / "trees.tif"
        albedo_path = patch_dir / "albedo.tif"

        if not all(p.exists() for p in [dsm_path, cdsm_path, albedo_path]):
            stats["skipped_missing"] += 1
            continue

        # Find T_mrt output files
        output_dir = patch_dir / "output_folder"
        tmrt_files = find_tmrt_files(output_dir)

        if not tmrt_files:
            continue

        # Read input rasters (clean: nodata → 0)
        dsm = read_raster_clean(dsm_path)
        cdsm = read_raster_clean(cdsm_path)
        albedo = read_raster_clean(albedo_path)

        # Normalize and stack input tensor
        dsm_norm = normalize(dsm, 0, dsm_max)
        cdsm_norm = normalize(cdsm, 0, cdsm_max)
        albedo_norm = normalize(albedo, 0, 1.0)

        input_tensor = np.stack([dsm_norm, cdsm_norm, albedo_norm], axis=0)  # (3, H, W)

        # Save input tensor
        input_path = PATCHES_DIR / f"{patch_name}_input.npy"
        np.save(input_path, input_tensor)
        stats["input_tensors"] += 1

        # Update input stats
        for raw, stat_key in [(dsm, "dsm_stats"), (cdsm, "cdsm_stats"), (albedo, "albedo_stats")]:
            valid = raw[raw > 0] if stat_key != "albedo_stats" else raw[raw >= 0]
            if len(valid) > 0:
                stats[stat_key]["min"] = min(stats[stat_key]["min"], float(np.min(valid)))
                stats[stat_key]["max"] = max(stats[stat_key]["max"], float(np.max(valid)))
                stats[stat_key]["sum"] += float(np.sum(valid))
                stats[stat_key]["count"] += len(valid)

        # Process each T_mrt output
        for temporal_label, tmrt_path in tmrt_files:
            tmrt_raw = read_raster(tmrt_path)  # preserves -9999

            # Check nodata fraction
            nodata_mask = tmrt_raw <= -9000
            nodata_frac = nodata_mask.sum() / tmrt_raw.size

            if nodata_frac > MAX_NODATA_FRACTION:
                stats["skipped_nodata"] += 1
                continue

            # Fill nodata with patch valid-pixel mean
            valid_mask = ~nodata_mask
            if valid_mask.any():
                patch_mean = tmrt_raw[valid_mask].mean()
                tmrt_filled = tmrt_raw.copy()
                tmrt_filled[nodata_mask] = patch_mean
            else:
                stats["skipped_nodata"] += 1
                continue

            # Normalize T_mrt
            tmrt_norm = normalize(tmrt_filled, 0, tmrt_max)
            target_tensor = tmrt_norm[np.newaxis, :, :]  # (1, H, W)

            # Save target tensor
            target_path = PATCHES_DIR / f"{patch_name}_{temporal_label}_target.npy"
            np.save(target_path, target_tensor)

            stats["input_pairs"] += 1
            stats["temporal_combos"].add(temporal_label)

            # T_mrt stats (on valid raw values, exclude nodata)
            valid_tmrt = tmrt_raw[valid_mask]
            if len(valid_tmrt) > 0:
                stats["tmrt_stats"]["min"] = min(stats["tmrt_stats"]["min"], float(np.min(valid_tmrt)))
                stats["tmrt_stats"]["max"] = max(stats["tmrt_stats"]["max"], float(np.max(valid_tmrt)))
                stats["tmrt_stats"]["sum"] += float(np.sum(valid_tmrt))
                stats["tmrt_stats"]["count"] += len(valid_tmrt)

        # Count type
        if "_green" in patch_name:
            stats["green_mutations"] += 1
        elif "_cool" in patch_name:
            stats["cool_mutations"] += 1
        else:
            stats["base_patches"] += 1

        total_done = stats["base_patches"] + stats["green_mutations"] + stats["cool_mutations"]
        if total_done % 100 == 0:
            logger.info(f"  Processed {total_done} patches, {stats['input_pairs']} training pairs...")

    # Compute means
    for key in ["dsm_stats", "cdsm_stats", "albedo_stats", "tmrt_stats"]:
        s = stats[key]
        if s["count"] > 0:
            s["mean"] = round(s["sum"] / s["count"], 4)
        else:
            s["mean"] = 0.0
        s["min"] = round(s["min"], 4) if s["min"] != float("inf") else 0.0
        s["max"] = round(s["max"], 4) if s["max"] != float("-inf") else 0.0
        del s["sum"]
        del s["count"]

    stats["temporal_combos"] = sorted(stats["temporal_combos"])

    # Generate train/val/test split (80/10/10, by geographic blocking using patch names)
    base_patch_names = sorted([
        d.name for d in all_dirs
        if "_green" not in d.name and "_cool" not in d.name
        and (d / "output_folder").exists()
        and len(find_tmrt_files(d / "output_folder")) > 0
    ])
    n_base = len(base_patch_names)
    n_train = int(n_base * 0.8)
    n_val = int(n_base * 0.1)

    stats["splits"] = {
        "train_patches": base_patch_names[:n_train],
        "val_patches": base_patch_names[n_train : n_train + n_val],
        "test_patches": base_patch_names[n_train + n_val :],
        "train_count": n_train,
        "val_count": n_val,
        "test_count": n_base - n_train - n_val,
        "note": "Geographic block split by sorted patch name — train=first 80%, val=next 10%, test=last 10%",
    }

    # Save stats
    stats_path = TRAINING_DIR / "dataset_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    # Summary
    logger.info(f"\n{'=' * 50}")
    logger.info(f"Training data packaging complete:")
    logger.info(f"  Total training pairs: {stats['input_pairs']}")
    logger.info(f"  Input tensors: {stats['input_tensors']}")
    logger.info(f"  Base patches: {stats['base_patches']}")
    logger.info(f"  Greening mutations: {stats['green_mutations']}")
    logger.info(f"  Cool roof mutations: {stats['cool_mutations']}")
    logger.info(f"  Temporal combos: {stats['temporal_combos']}")
    logger.info(f"  T_mrt range: {stats['tmrt_stats']['min']:.1f} - {stats['tmrt_stats']['max']:.1f} °C")
    logger.info(f"  Skipped (nodata): {stats['skipped_nodata']}")
    logger.info(f"  Skipped (missing): {stats['skipped_missing']}")
    logger.info(f"  Train/Val/Test: {stats['splits']['train_count']}/{stats['splits']['val_count']}/{stats['splits']['test_count']}")
    logger.info(f"  Dataset stats: {stats_path}")

    # Estimate storage
    npy_files = list(PATCHES_DIR.glob("*.npy"))
    total_npy_mb = sum(f.stat().st_size for f in npy_files) / (1024 * 1024)
    logger.info(f"  .npy files: {len(npy_files)} ({total_npy_mb:.0f} MB)")
    logger.info(f"{'=' * 50}")


if __name__ == "__main__":
    main()
