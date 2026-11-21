"""Serving-time shade calibration for V5 (Lane 1, 2026-08-20).

WHY THIS EXISTS
---------------------------------------------------------------------------
shade_age_audit.py (workstation, test split) measured V5 running +1 to
+2.3 C warm in TRUE, current building shade, growing with shade depth, at
h13/h15 -- small in corpus RMSE (removing it buys 0.014 C) but large in
perceived shadow contrast: a midday shadow rendered 2 C too warm looks
weak. The mechanism is history-blindness: V5's only shade-history inputs
are the mirrored planes it learned to ignore (shadow_lie_audit.py).

This module subtracts the per-(hour x trajectory) bias fitted on the VAL
split (fit_shade_calib.py -> v5_shade_calib.json), where trajectory is the
pixel's true shading pattern at 09/13/16 from the CORRECTED planes
(v4_prep/shadow_multi_v6). Canopy pixels (cdsm > 3 m) are untouched -- the
fit excluded them. Cells without a confident fit carry bias 0.

It is honest statistics, not faked cooling: fitted on held-out val data,
gated per cell on >= 50k pixels, fully reversible, and reported in the
response metadata. Kill switch: env UH_V5_SHADE_CALIB=0.

V7 (retrained on one consistent world with real history channels) is the
proper fix; this keeps live V5 honest until then.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "urban_heat_data"
CALIB_PATH = DATA_ROOT / "models" / "v5_shade_calib.json"
PLANES_DIR = DATA_ROOT / "v4_prep" / "shadow_multi_v6"   # corrected planes
CDSM_CANOPY_M = 3.0
SNAP_HOURS = (9, 11, 13, 15)
PLANE_HOURS = (9, 13, 16)

_table: Optional[Dict[int, np.ndarray]] = None
_disabled_reason: Optional[str] = None
_load_attempted = False


def _snap_hour(hour: int) -> int:
    """Nearest fitted hour; ties resolve later (matches the audit scripts)."""
    return min(SNAP_HOURS, key=lambda h: (abs(h - hour), -h))


def _load() -> None:
    global _table, _disabled_reason, _load_attempted
    _load_attempted = True
    if os.environ.get("UH_V5_SHADE_CALIB", "1") == "0":
        _disabled_reason = "disabled via UH_V5_SHADE_CALIB=0"
        logger.info("shade calibration: %s", _disabled_reason)
        return
    if not CALIB_PATH.exists():
        _disabled_reason = f"no calibration table at {CALIB_PATH}"
        logger.info("shade calibration: %s", _disabled_reason)
        return
    missing = [h for h in PLANE_HOURS
               if not (PLANES_DIR / f"shadow_h{h:02d}.tif").exists()]
    if missing:
        _disabled_reason = f"corrected planes missing: {missing}"
        logger.warning("shade calibration: %s", _disabled_reason)
        return
    try:
        raw = json.loads(CALIB_PATH.read_text())
        table: Dict[int, np.ndarray] = {}
        for h in SNAP_HOURS:
            row = np.zeros(8, dtype=np.float32)
            for pat, b in raw["bias"][str(h)].items():
                row[int(pat, 2)] = float(b)
            table[h] = row
        _table = table
        logger.info("shade calibration loaded: fit=%s, hours=%s",
                    raw.get("fit_split"), sorted(table.keys()))
    except Exception as e:                        # malformed table: refuse
        _disabled_reason = f"failed to parse {CALIB_PATH.name}: {e}"
        logger.warning("shade calibration: %s", _disabled_reason)


def _read_planes_window(col_off: float, row_off: float,
                        height: int, width: int) -> Optional[np.ndarray]:
    """(3,H,W) trajectory planes on the master grid; None on any failure."""
    import rasterio
    from rasterio.windows import Window
    out = np.zeros((3, height, width), dtype=np.float32)
    try:
        for i, h in enumerate(PLANE_HOURS):
            with rasterio.open(PLANES_DIR / f"shadow_h{h:02d}.tif") as src:
                out[i] = src.read(
                    1, window=Window(col_off, row_off, width, height),
                    boundless=True, fill_value=0.0).astype(np.float32)
        return out
    except Exception as e:
        logger.warning("shade calibration: plane read failed (%s)", e)
        return None


def _apply(tmrt: np.ndarray, cdsm_m: np.ndarray, planes: np.ndarray,
           hour: int) -> Tuple[np.ndarray, Dict]:
    snap = _snap_hour(hour)
    bias_row = _table[snap]                                   # type: ignore
    pat = ((planes[0] > 0.5).astype(np.int8) * 4
           + (planes[1] > 0.5).astype(np.int8) * 2
           + (planes[2] > 0.5).astype(np.int8))
    delta = bias_row[pat]
    delta = np.where(cdsm_m > CDSM_CANOPY_M, 0.0, delta).astype(np.float32)
    touched = float((np.abs(delta) > 1e-6).mean())
    meta = {
        "applied": True,
        "fit": "val-split per-(hour x trajectory) bias, fit_shade_calib.py",
        "hour_used": snap,
        "pct_pixels_adjusted": round(100.0 * touched, 2),
        "mean_delta_C": round(float(delta.mean()), 4),
        "max_abs_delta_C": round(float(np.abs(delta).max()), 3),
    }
    return tmrt - delta, meta


def apply_by_window(tmrt: np.ndarray, cdsm_m: np.ndarray,
                    col_off: float, row_off: float,
                    hour: int) -> Tuple[np.ndarray, Dict]:
    """Calibrate a tile addressed by master-grid pixel offsets (stitch path).

    cdsm_m must be canopy height in METRES. Never raises: on any problem the
    input comes back unchanged with meta.applied False -- a render must not
    die for its calibration.
    """
    if not _load_attempted:
        _load()
    if _table is None:
        return tmrt, {"applied": False, "reason": _disabled_reason}
    planes = _read_planes_window(col_off, row_off,
                                 tmrt.shape[0], tmrt.shape[1])
    if planes is None:
        return tmrt, {"applied": False, "reason": "plane read failed"}
    return _apply(tmrt, cdsm_m, planes, hour)


def apply_by_bounds(tmrt: np.ndarray, cdsm_m: np.ndarray,
                    bbox_32755: Tuple[float, float, float, float],
                    hour: int) -> Tuple[np.ndarray, Dict]:
    """Calibrate a patch addressed by its EPSG:32755 bounds (patch path)."""
    if not _load_attempted:
        _load()
    if _table is None:
        return tmrt, {"applied": False, "reason": _disabled_reason}
    try:
        import rasterio
        west, south, east, north = bbox_32755
        with rasterio.open(PLANES_DIR / "shadow_h09.tif") as src:
            row_off, col_off = src.index(west, north)
    except Exception as e:
        logger.warning("shade calibration: bounds->window failed (%s)", e)
        return tmrt, {"applied": False, "reason": "bounds->window failed"}
    planes = _read_planes_window(col_off, row_off,
                                 tmrt.shape[0], tmrt.shape[1])
    if planes is None:
        return tmrt, {"applied": False, "reason": "plane read failed"}
    return _apply(tmrt, cdsm_m, planes, hour)
