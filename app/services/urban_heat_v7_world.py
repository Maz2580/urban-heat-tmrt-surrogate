"""V7's input world: serve V7 EXACTLY what its corpus was built from.

WHY THIS EXISTS
---------------------------------------------------------------------------
V7 (2026-08-21) is V5's architecture, labels, split and loss, retrained on
inputs that finally agree with the labels' world (the 2018 LiDAR that the
SOLWEIG labels were computed on):

    slot 0   dsm_corrected   <- rasters/raw/dsm_1m.tif      (was Geoscape 2025)
    slot 10  shadow_fraction <- shade AGE in hours           (was identically 0)
    slot 17  shadow_h09      <- v4_prep/shadow_multi_v7/     (corrected ray,
    slot 18  shadow_h13                                       dsm_1m surface,
    slot 19  shadow_h16                                       400 m)

The corpus was built by taking V5's Stage-D tensor and OVERWRITING those
rows (scripts/urban_heat/apply_shadow_v7.py + stage_d.py --shadow-channel
age). Serving must do the identical overwrite on top of the V5 tensor --
NOT swap the `dsm_corrected` raster path, because patch_builder derives svf
and building_density from that raster and the corpus kept V5's values there.

HOW MUCH IT MATTERS (sensitivity_probe.py, test split, 2026-08-21)
    age := 0      -> building-shade RMSE 4.92 -> 13.72, +9.4 C in every shadow
    planes := 0   -> +0.19 RMSE in shade
Channel 10 is the most load-bearing input V7 has. An absent age raster must
therefore REFUSE to serve V7, never fall back to zeros.

Strictly version-conditional: nothing here runs unless use_v7 is True.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)

DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "urban_heat_data"
V7_DSM_PATH = DATA_ROOT / "rasters" / "raw" / "dsm_1m.tif"
V7_SHADOW_DIR = DATA_ROOT / "v4_prep" / "shadow_multi_v7"
V7_SHADOW_PATHS: Dict[int, Path] = {
    17: V7_SHADOW_DIR / "shadow_h09.tif",
    18: V7_SHADOW_DIR / "shadow_h13.tif",
    19: V7_SHADOW_DIR / "shadow_h16.tif",
}
AGE_HOURS = (9, 11, 13, 15)           # the corpus sample hours
V7_AGE_PATHS: Dict[int, Path] = {
    h: V7_SHADOW_DIR / f"shade_age_h{h:02d}.tif" for h in AGE_HOURS
}
CH_DSM = 0
CH_AGE = 10
PATCH = 512


def snap_age_hour(hour: int) -> int:
    """Nearest corpus hour; ties resolve LATER (same rule as the shade
    calibration and the audit scripts, so every consumer agrees)."""
    return min(AGE_HOURS, key=lambda h: (abs(h - hour), -h))


def missing_rasters() -> list[str]:
    """Every V7 raster that is absent. Empty list == servable."""
    paths = [V7_DSM_PATH, *V7_SHADOW_PATHS.values(), *V7_AGE_PATHS.values()]
    return [str(p) for p in paths if not p.exists()]


def rasters_available() -> bool:
    return not missing_rasters()


def _read(path: Path, col_off: int, row_off: int) -> np.ndarray:
    """Read one 512x512 window; fill nodata / sentinels with 0 exactly as
    build_shadow_v7_pack.py did for the corpus (nodata -> 0, < -100 -> 0).
    Reads outside the raster come back zero-padded (boundless), matching
    patch_builder._read_window's fill for out-of-coverage margins."""
    import rasterio
    from rasterio.windows import Window

    with rasterio.open(path) as ds:
        a = ds.read(1, window=Window(col_off, row_off, PATCH, PATCH),
                    boundless=True, fill_value=0.0).astype(np.float32)
        if ds.nodata is not None:
            a[a == ds.nodata] = 0.0
    a[~np.isfinite(a)] = 0.0
    a[a < -100.0] = 0.0
    return a


def apply_v7_world(
    tensor23: np.ndarray, col_off: int, row_off: int, hour: int,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Overwrite slots 0 / 10 / 17-19 of a V5-style 23-channel tensor with
    V7's world at the given master-grid window. Returns (new tensor, meta).

    Raises RuntimeError if any V7 raster is missing -- serving V7 without
    its age channel is a +9.4 C skew, which must never happen silently.
    """
    missing = missing_rasters()
    if missing:
        raise RuntimeError(f"V7 rasters missing: {missing}")
    if tensor23.ndim != 3 or tensor23.shape[0] < 20:
        raise RuntimeError(f"apply_v7_world expects (23,H,W), got {tensor23.shape}")

    out = tensor23.copy()
    out[CH_DSM] = _read(V7_DSM_PATH, col_off, row_off)
    for slot, path in V7_SHADOW_PATHS.items():
        out[slot] = _read(path, col_off, row_off)
    age_hour = snap_age_hour(int(hour))
    out[CH_AGE] = _read(V7_AGE_PATHS[age_hour], col_off, row_off)

    meta = {
        "world": "2018 LiDAR (dsm_1m) + shadow_multi_v7 + shade age",
        "age_hour_used": age_hour,
        "age_max_h": float(out[CH_AGE].max()),
        "shaded_now_pct": float((out[CH_AGE] > 0).mean() * 100.0),
    }
    return out, meta
