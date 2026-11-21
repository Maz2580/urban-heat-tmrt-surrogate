"""
Urban Heat Inference Service (Phase 1.6.3 + 1.6.4 + V2 + V3 + V4)
==================================================================

ONNX Runtime wrapper for U-Net surrogate models.
Predicts T_mrt (Mean Radiant Temperature) from raster input.

Model versions:
  V1/V1.1: 5-channel input (DSM, CDSM, Albedo, sin_alt, alt_norm)
  V2:      11-channel input (+ SVF, building_density, tree_distance,
                              aspect_sin, aspect_cos, shadow_fraction)
  V3:      12-channel multi-task (V2's 11 + ERA5 Tair scalar plane;
                                  output 3 heads, T_mrt is channel 0)
  V4:      23-channel multi-task (V3's 12 + ERA5 RH/wind/SSRD + NDVI +
                                  Geoscape solar-panel + multi-shadow
                                  h09/h13/h16 + roof-type one-hot)

- Loads FP32 ONNX models on startup; V4 uses external-data format
  (graph + .onnx.data both required, co-located).
- Uses ProcessPoolExecutor (2 workers) for CPU-parallel inference
- Semaphore(2) limits concurrent inference requests
- ~1.8s per 512x512 patch on CPU (V1), ~2.1s (V2 with feature computation),
  V3 ~5s end-to-end (live ERA5 GEE call dominates), V4 ~5s similar
  (one ERA5 GEE call yields all 4 fields; per-channel raster reads add
  ~50 ms total).

Model paths:
  V1:  urban_heat_data/models/umami_unet_fp32.onnx (97 MB)
  V2:  urban_heat_data/models/umami_unet_v2_fp32.onnx (0.6 MB graph + 97 MB data)
  V3:  urban_heat_data/models/umami_unet_v3_fp32.onnx (97 MB single file)
  V3.1: urban_heat_data/models/umami_unet_v31_fp32.onnx (0.3 MB graph + 94 MB data)
        — identical 12-channel input as V3; only the trained weights differ.
        Trained with L1 + variance-preservation loss + flat-patch filter to
        fix V3's mode collapse on building-vs-canopy spatial contrast.
  V4:  urban_heat_data/models/umami_unet_v4_fp32.onnx (0.27 MB graph + 93 MB data)
"""

import asyncio
import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger("umami.urban_heat_inference")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "urban_heat_data"
MODEL_DIR = DATA_ROOT / "models"
ONNX_MODEL_PATH = MODEL_DIR / "umami_unet_int8.onnx"
ONNX_MODEL_FP32_PATH = MODEL_DIR / "umami_unet_fp32.onnx"
ONNX_V2_MODEL_PATH = MODEL_DIR / "umami_unet_v2_fp32.onnx"
ONNX_V3_MODEL_PATH = MODEL_DIR / "umami_unet_v3_fp32.onnx"
# V3.1 ONNX uses external-data format (same as V4): graph stub + .onnx.data
# weights file MUST be co-located. Same input layout as V3 (12 channels).
ONNX_V31_MODEL_PATH = MODEL_DIR / "umami_unet_v31_fp32.onnx"
# V4 ONNX uses external-data format: graph stub + .onnx.data weights file.
# Both files MUST be co-located (the graph file references the data file by
# relative name). Loading the graph alone produces silent garbage output.
ONNX_V4_MODEL_PATH = MODEL_DIR / "umami_unet_v4_fp32.onnx"
ONNX_V41_MODEL_PATH = MODEL_DIR / "umami_unet_v41_fp32.onnx"
# V5: single-file ONNX (no .onnx.data sidecar). Same 23-channel input layout as
# V4/V4.1 -- V5 changed the LABELS, not the inputs. Trained on reference
# SOLWEIG labels instead of the old `Tground = Ta + 8` engine.
# See V5_SHADOW_CHANNEL below: V5 must be served with channel 10 zeroed.
ONNX_V5_MODEL_PATH = MODEL_DIR / "umami_unet_v5_fp32.onnx"

# V6: identical architecture and input layout to V5. The ONLY difference is
# that channels 17/18/19 (shadow_h09/h13/h16) were CORRECTED before training --
# the shipped rasters traced their ray away from the sun, painting shadow on the
# sunlit face of every building (alignment -0.988/-1.000/-1.000; corrected
# +1.000/+0.998/+0.969). V6 must therefore be SERVED the corrected rasters; see
# V6_SHADOW_*_PATH. Serving it the old ones is a train/serve skew that measures
# +4.34 C on building-shaded ground.
ONNX_V6_MODEL_PATH = MODEL_DIR / "umami_unet_v6_fp32.onnx"

# V7 (2026-08-21): V5's architecture/labels/split/loss retrained on inputs
# that agree with the labels' world (2018 LiDAR). Slots 0 / 10 / 17-19 are
# OVERWRITTEN on top of the V5 tensor at serve time by
# app/services/urban_heat_v7_world.apply_v7_world -- exactly how the corpus
# was built. Channel 10 carries shade AGE in hours and is V7's most
# load-bearing input (zeroing it = +9.4 C in every shadow), so the
# V5/V6 channel-10 zeroing below must NEVER apply to the V7 session, and an
# absent age raster refuses to serve V7 rather than falling back to zeros.
# Test RMSE 3.758 C vs V5 4.182; building shade 5.91 -> 4.92 C on the same
# mask. Selectable, not default.
ONNX_V7_MODEL_PATH = MODEL_DIR / "umami_unet_v7_fp32.onnx"
DATASET_STATS_PATH = MODEL_DIR / "dataset_stats.json"

# Channel 10 (shadow_fraction) is identically ZERO in every V3/V4/V5 training
# sample, so its first-layer weights never received a gradient. The live path
# computes a shadow map that traces its ray AWAY from the sun (IoU 0.052 against
# SOLWEIG's own shadow raster; it claims 70% of a scene is shaded at 71 deg
# solar altitude). Feeding that through untrained weights injects structured
# noise -- measured at up to +26.8 C on individual pixels while leaving the mean
# unchanged. V5 is therefore served what it was trained on: zero.
# Fixing compute_shadow_fraction() and retraining with a real shadow channel is
# a separate piece of work; until then this keeps V5 honest.
V5_SHADOW_CHANNEL = 10

# V3 reads two new master rasters that V1/V2 don't use
GEOSCAPE_HEIGHTS_PATH = DATA_ROOT / "rasters" / "raw" / "geoscape_heights.tif"
GEOSCAPE_ALBEDO_PATH = DATA_ROOT / "rasters" / "raw" / "geoscape_albedo.tif"

# V4 reads 6 additional rasters for slots 12-22 (NDVI + multi-shadow +
# Geoscape solar-panel mask + roof-type one-hot). All on the same UTM grid
# as the DSM master raster.
V4_NDVI_PATH = DATA_ROOT / "v4_prep" / "geoscape_ndvi.tif"
V4_SOLAR_PANEL_PATH = DATA_ROOT / "rasters" / "raw" / "geoscape_solar_panel_mask.tif"
V4_ROOF_TYPE_PATH = DATA_ROOT / "rasters" / "raw" / "geoscape_roof_type.tif"
V4_SHADOW_H09_PATH = DATA_ROOT / "v4_prep" / "shadow_multi" / "shadow_h09.tif"
V4_SHADOW_H13_PATH = DATA_ROOT / "v4_prep" / "shadow_multi" / "shadow_h13.tif"
V4_SHADOW_H16_PATH = DATA_ROOT / "v4_prep" / "shadow_multi" / "shadow_h16.tif"

# V6 ONLY. Regenerated 2026-08-18 by scripts/urban_heat/regen_shadow_v6.py with
# three fixes: ray direction (was traced away from the sun), surface
# (geoscape_heights.tif Dec 2025 instead of dsm_1m.tif 2018, matching channel 0),
# and ray length (400 m instead of 100 m, which truncated anything above ~85 m).
#
# STRICTLY VERSION-CONDITIONAL. V4/V4.1/V5 learned the mirrored convention and
# reproduce their labels well under it; serving them these would break them. The
# same lesson as _apply_v3_geoscape_overrides -- a blanket swap breaks the older
# models.
V6_SHADOW_H09_PATH = DATA_ROOT / "v4_prep" / "shadow_multi_v6" / "shadow_h09.tif"
V6_SHADOW_H13_PATH = DATA_ROOT / "v4_prep" / "shadow_multi_v6" / "shadow_h13.tif"
V6_SHADOW_H16_PATH = DATA_ROOT / "v4_prep" / "shadow_multi_v6" / "shadow_h16.tif"

MAX_INFERENCE_WORKERS = 2


# ---------------------------------------------------------------------------
# Module-level inference function (must be top-level for ProcessPoolExecutor)
# ---------------------------------------------------------------------------

_ort_session = None
_ort_session_v2 = None
_ort_session_v3 = None
_ort_session_v31 = None
_ort_session_v4 = None
_ort_session_v41 = None
_ort_session_v5 = None
_ort_session_v6 = None
_ort_session_v7 = None
_normalization = None


def _init_worker():
    """Initialize ONNX Runtime sessions (V1 + V2 + V3 + V3.1 + V4) in each worker process."""
    global _ort_session, _ort_session_v2, _ort_session_v3, _ort_session_v31, _ort_session_v4, _ort_session_v41, _ort_session_v5, _ort_session_v6, _ort_session_v7, _normalization
    import onnxruntime as ort

    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 4
    sess_opts.inter_op_num_threads = 2
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opts.enable_mem_pattern = True
    sess_opts.enable_mem_reuse = True

    # V1 model (prefer FP32)
    model_path = str(ONNX_MODEL_FP32_PATH) if ONNX_MODEL_FP32_PATH.exists() else str(ONNX_MODEL_PATH)
    _ort_session = ort.InferenceSession(
        model_path,
        sess_options=sess_opts,
        providers=["CPUExecutionProvider"],
    )

    # V2 model (11-channel)
    if ONNX_V2_MODEL_PATH.exists():
        v2_opts = ort.SessionOptions()
        v2_opts.intra_op_num_threads = 4
        v2_opts.inter_op_num_threads = 2
        v2_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        _ort_session_v2 = ort.InferenceSession(
            str(ONNX_V2_MODEL_PATH),
            sess_options=v2_opts,
            providers=["CPUExecutionProvider"],
        )

    # V3 model (12-channel multi-task — V2's 11 + ERA5 Tair scalar plane;
    # output has 3 heads but only the T_mrt one is used at serving time)
    if ONNX_V3_MODEL_PATH.exists():
        v3_opts = ort.SessionOptions()
        v3_opts.intra_op_num_threads = 4
        v3_opts.inter_op_num_threads = 2
        v3_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        _ort_session_v3 = ort.InferenceSession(
            str(ONNX_V3_MODEL_PATH),
            sess_options=v3_opts,
            providers=["CPUExecutionProvider"],
        )

    # V3.1 model (12-channel — identical input layout to V3, only trained
    # weights differ. Uses external-data format like V4: graph stub +
    # .onnx.data MUST be co-located).
    if ONNX_V31_MODEL_PATH.exists():
        v31_data_path = ONNX_V31_MODEL_PATH.with_suffix(".onnx.data")
        if not v31_data_path.exists():
            logger.warning(
                "V3.1 ONNX graph %s found but external-data file missing at %s — "
                "V3.1 inference disabled until both files are co-located",
                ONNX_V31_MODEL_PATH.name, v31_data_path.name,
            )
        else:
            v31_opts = ort.SessionOptions()
            v31_opts.intra_op_num_threads = 4
            v31_opts.inter_op_num_threads = 2
            v31_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            _ort_session_v31 = ort.InferenceSession(
                str(ONNX_V31_MODEL_PATH),
                sess_options=v31_opts,
                providers=["CPUExecutionProvider"],
            )

    # V4 model (23-channel multi-task — V3's 12 + ERA5 RH/wind/SSRD + NDVI +
    # multi-shadow + Geoscape solar-panel + roof-type one-hot; output has
    # 3 heads same as V3, only T_mrt is used at serving time).
    # V4 uses ONNX external-data format — graph stub + .onnx.data weights
    # MUST be co-located. We fail-explicit (don't load V4) if data file is
    # missing rather than silently load a stub that produces garbage.
    if ONNX_V4_MODEL_PATH.exists():
        v4_data_path = ONNX_V4_MODEL_PATH.with_suffix(".onnx.data")
        if not v4_data_path.exists():
            logger.warning(
                "V4 ONNX graph %s found but external-data file missing at %s — "
                "V4 inference disabled until both files are co-located",
                ONNX_V4_MODEL_PATH.name, v4_data_path.name,
            )
        else:
            v4_opts = ort.SessionOptions()
            v4_opts.intra_op_num_threads = 4
            v4_opts.inter_op_num_threads = 2
            v4_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            _ort_session_v4 = ort.InferenceSession(
                str(ONNX_V4_MODEL_PATH),
                sess_options=v4_opts,
                providers=["CPUExecutionProvider"],
            )

    # V4.1 model (23-channel — same input layout as V4, retrained with
    # Option B annealed variance-preservation loss. Exported as a SINGLE
    # ONNX file via the legacy dynamo=False exporter — no .onnx.data
    # sidecar to co-locate, unlike V4/V3.1).
    if ONNX_V41_MODEL_PATH.exists():
        v41_opts = ort.SessionOptions()
        v41_opts.intra_op_num_threads = 4
        v41_opts.inter_op_num_threads = 2
        v41_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        _ort_session_v41 = ort.InferenceSession(
            str(ONNX_V41_MODEL_PATH),
            sess_options=v41_opts,
            providers=["CPUExecutionProvider"],
        )

    # V5 — reference-SOLWEIG labels. Single-file ONNX like V4.1. Guarded by
    # exists() so an absent file leaves _ort_session_v5 as None and every other
    # path behaves exactly as before.
    if ONNX_V5_MODEL_PATH.exists():
        v5_opts = ort.SessionOptions()
        v5_opts.intra_op_num_threads = 4
        v5_opts.inter_op_num_threads = 2
        v5_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        _ort_session_v5 = ort.InferenceSession(
            str(ONNX_V5_MODEL_PATH),
            sess_options=v5_opts,
            providers=["CPUExecutionProvider"],
        )

    # V6 -- same guard: an absent file leaves _ort_session_v6 as None and every
    # existing path behaves exactly as before.
    if ONNX_V6_MODEL_PATH.exists():
        v6_opts = ort.SessionOptions()
        v6_opts.intra_op_num_threads = 4
        v6_opts.inter_op_num_threads = 2
        v6_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        _ort_session_v6 = ort.InferenceSession(
            str(ONNX_V6_MODEL_PATH),
            sess_options=v6_opts,
            providers=["CPUExecutionProvider"],
        )

    # V7 -- same guard again; its own session object so the V5/V6 channel-10
    # zeroing in _run_inference cannot reach it.
    if ONNX_V7_MODEL_PATH.exists():
        v7_opts = ort.SessionOptions()
        v7_opts.intra_op_num_threads = 4
        v7_opts.inter_op_num_threads = 2
        v7_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        _ort_session_v7 = ort.InferenceSession(
            str(ONNX_V7_MODEL_PATH),
            sess_options=v7_opts,
            providers=["CPUExecutionProvider"],
        )

    stats_path = str(DATASET_STATS_PATH)
    if Path(stats_path).exists():
        with open(stats_path) as f:
            _normalization = json.load(f)["normalization"]


def _run_inference(
    input_array: np.ndarray,
    use_v2: bool = False,
    use_v3: bool = False,
    use_v4: bool = False,
    use_v31: bool = False,
    use_v41: bool = False,
    use_v5: bool = False,
    use_v6: bool = False,
    use_v7: bool = False,
) -> Dict[str, Any]:
    """
    Run inference on a normalized input tensor.

    V1:    (5, 512, 512)  — DSM, CDSM, Albedo, sin_alt, alt_norm
    V2:    (11, 512, 512) — above + SVF, building_density, tree_distance,
                            aspect_sin, aspect_cos, shadow_fraction
    V3:    (12, 512, 512) — V2's 11 + ERA5 Tair scalar plane;
                            output is (1, 3, 512, 512) (multi-task heads
                            tmrt/lst/tair) and we keep only channel 0 = T_mrt
    V3.1:  (12, 512, 512) — same input layout as V3; different ONNX weights
                            (anti-collapse loss + flat-patch filter).
    V4:    (23, 512, 512) — V3's 12 + ERA5 RH/wind/SSRD + NDVI +
                            Geoscape solar-panel + multi-shadow (h09/h13/h16)
                            + roof-type one-hot. Output (1, 3, 512, 512)
                            same shape as V3, channel 0 is T_mrt.
    V4.1:  (23, 512, 512) — same input layout as V4; retrained with
                            Option B annealed variance-preservation loss.
                            Single-file ONNX (not external-data like V4/V3.1).

    Returns dict with tmrt_normalized, tmrt_celsius, stats, inference_ms.
    """
    global _ort_session, _ort_session_v2, _ort_session_v3, _ort_session_v31, _ort_session_v4, _ort_session_v41, _ort_session_v5, _ort_session_v6, _ort_session_v7, _normalization

    if _ort_session is None:
        _init_worker()

    # Branch order: V5 > V4.1 > V4 > V3.1 > V3 > V2 > V1 (each opt-in via its
    # own bool; V1 is the implicit fallback). V4.1 takes priority over V4 so
    # that if both flags accidentally fire, V4.1's annealed weights win. V5 sits
    # at the top for the same reason, and is only reachable when explicitly
    # requested -- V4.1 remains the default everywhere it already was.
    if use_v7 and _ort_session_v7 is not None:
        session = _ort_session_v7
    elif use_v6 and _ort_session_v6 is not None:
        session = _ort_session_v6
    elif use_v5 and _ort_session_v5 is not None:
        session = _ort_session_v5
    elif use_v41 and _ort_session_v41 is not None:
        session = _ort_session_v41
    elif use_v4 and _ort_session_v4 is not None:
        session = _ort_session_v4
    elif use_v31 and _ort_session_v31 is not None:
        session = _ort_session_v31
    elif use_v3 and _ort_session_v3 is not None:
        session = _ort_session_v3
    elif use_v2 and _ort_session_v2 is not None:
        session = _ort_session_v2
    else:
        session = _ort_session

    if input_array.ndim == 3:
        input_array = input_array[np.newaxis, ...]

    # V5 was trained with shadow_fraction identically zero in every sample, so
    # those weights are untrained. Serving the live computed shadow map (which
    # traces backwards) would push structured noise through them. Give V5 what
    # it saw in training. Copy first: the caller's array is not ours to mutate.
    # V6 was built by Stage D with `--shadow-channel zeros`, exactly as V5 was,
    # so channel 10 is untrained for it too and gets the same treatment. What V6
    # changed is channels 17/18/19, which come from rasters (see V6_SHADOW_*).
    if (session is _ort_session_v5 or session is _ort_session_v6) and session is not None:
        if input_array.shape[1] > V5_SHADOW_CHANNEL:
            input_array = input_array.copy()
            input_array[:, V5_SHADOW_CHANNEL] = 0.0

    t0 = time.perf_counter()
    outputs = session.run(None, {"input": input_array.astype(np.float32)})
    inference_ms = (time.perf_counter() - t0) * 1000

    # V1/V2: outputs[0] shape = (1, 1, 512, 512); V3/V3.1/V4/V4.1/V5: (1, 3, 512, 512).
    # Index [0, 0] gives T_mrt for all — V3+/V4+/V5's channel 0 is the T_mrt head.
    raw_output = outputs[0][0, 0]
    if use_v3 or use_v31 or use_v4 or use_v41 or use_v5 or use_v6 or use_v7:
        # V3, V3.1, V4, V4.1, V5 all trained on raw T_mrt °C values (no [0, 1]
        # normalisation), so the model output is already in °C. V1/V2 trained
        # on tmrt/63 so we have to denormalise. Omitting V5 here would multiply
        # its ~50 °C output by 63.
        tmrt_norm = raw_output / 63.0  # for backward-compat field
        tmrt_celsius = raw_output
    else:
        tmrt_norm = raw_output
        tmrt_max = _normalization["tmrt_range"][1] if _normalization else 63.0
        tmrt_celsius = tmrt_norm * tmrt_max

    return {
        "tmrt_normalized": tmrt_norm,
        "tmrt_celsius": tmrt_celsius,
        "stats": {
            "min_temp": float(np.min(tmrt_celsius)),
            "max_temp": float(np.max(tmrt_celsius)),
            "avg_temp": float(np.mean(tmrt_celsius)),
        },
        "inference_ms": inference_ms,
    }


# ---------------------------------------------------------------------------
# Service Class
# ---------------------------------------------------------------------------

class UrbanHeatInferenceService:
    """
    Singleton service wrapping ONNX inference with ProcessPoolExecutor.

    Usage:
        svc = get_inference_service()
        if svc.is_ready:
            result = await svc.predict(input_array)
    """

    def __init__(self):
        self._pool: Optional[ProcessPoolExecutor] = None
        self._model_loaded: bool = False
        self._normalization: Optional[Dict] = None
        self._warmup_done: bool = False
        self._semaphore = asyncio.Semaphore(MAX_INFERENCE_WORKERS)

    def _find_model_path(self) -> Optional[Path]:
        """Find the best available V1 ONNX model (prefer FP32 for accuracy)."""
        if ONNX_MODEL_FP32_PATH.exists():
            return ONNX_MODEL_FP32_PATH
        if ONNX_MODEL_PATH.exists():
            return ONNX_MODEL_PATH
        return None

    @property
    def v2_available(self) -> bool:
        """Check if V2 model is available."""
        return ONNX_V2_MODEL_PATH.exists()

    async def initialize(self) -> Dict[str, Any]:
        """Load model and warm up inference pool."""
        result = {"model_found": False, "pool_started": False, "warmup_done": False}

        model_path = self._find_model_path()
        if model_path is None:
            logger.info(
                "ONNX model not found — inference disabled (deploy to %s)",
                MODEL_DIR,
            )
            return result

        result["model_found"] = True
        model_size_mb = model_path.stat().st_size / (1024 * 1024)
        logger.info(
            "Loading ONNX model: %s (%.1f MB)", model_path.name, model_size_mb
        )

        if DATASET_STATS_PATH.exists():
            with open(DATASET_STATS_PATH) as f:
                self._normalization = json.load(f)["normalization"]

        self._pool = ProcessPoolExecutor(
            max_workers=MAX_INFERENCE_WORKERS,
            initializer=_init_worker,
        )
        self._model_loaded = True
        result["pool_started"] = True

        try:
            dummy = np.random.randn(5, 512, 512).astype(np.float32)
            loop = asyncio.get_event_loop()
            warmup_result = await loop.run_in_executor(self._pool, _run_inference, dummy)
            self._warmup_done = True
            result["warmup_done"] = True
            result["warmup_ms"] = warmup_result["inference_ms"]
            logger.info("V1 inference warmup: %.0f ms", warmup_result["inference_ms"])
        except Exception as e:
            logger.error("V1 warmup failed: %s", e)

        # Warm up V2 if available
        result["v2_available"] = ONNX_V2_MODEL_PATH.exists()
        if result["v2_available"]:
            try:
                dummy_v2 = np.random.randn(11, 512, 512).astype(np.float32)
                v2_result = await loop.run_in_executor(
                    self._pool, _run_inference, dummy_v2, True, False
                )
                result["v2_warmup_ms"] = v2_result["inference_ms"]
                logger.info("V2 inference warmup: %.0f ms", v2_result["inference_ms"])
            except Exception as e:
                logger.warning("V2 warmup failed: %s", e)

        # Warm up V3 if available
        result["v3_available"] = ONNX_V3_MODEL_PATH.exists()
        if result["v3_available"]:
            try:
                dummy_v3 = np.random.randn(12, 512, 512).astype(np.float32)
                v3_result = await loop.run_in_executor(
                    self._pool, _run_inference, dummy_v3, False, True
                )
                result["v3_warmup_ms"] = v3_result["inference_ms"]
                logger.info("V3 inference warmup: %.0f ms", v3_result["inference_ms"])
            except Exception as e:
                logger.warning("V3 warmup failed: %s", e)

        # Warm up V4 if available (graph + .onnx.data both present)
        result["v4_available"] = self.v4_available
        if result["v4_available"]:
            try:
                dummy_v4 = np.random.randn(23, 512, 512).astype(np.float32)
                v4_result = await loop.run_in_executor(
                    self._pool, _run_inference, dummy_v4, False, False, True
                )
                result["v4_warmup_ms"] = v4_result["inference_ms"]
                logger.info("V4 inference warmup: %.0f ms", v4_result["inference_ms"])
            except Exception as e:
                logger.warning("V4 warmup failed: %s", e)

        # Warm up V3.1 if available (graph + .onnx.data both present)
        result["v3.1_available"] = self.v31_available
        if result["v3.1_available"]:
            try:
                dummy_v31 = np.random.randn(12, 512, 512).astype(np.float32)
                v31_result = await loop.run_in_executor(
                    self._pool, _run_inference, dummy_v31, False, False, False, True
                )
                result["v31_warmup_ms"] = v31_result["inference_ms"]
                logger.info("V3.1 inference warmup: %.0f ms", v31_result["inference_ms"])
            except Exception as e:
                logger.warning("V3.1 warmup failed: %s", e)

        # Warm up V4.1 if available (single-file ONNX, no external-data needed)
        result["v4.1_available"] = self.v41_available
        if result["v4.1_available"]:
            try:
                dummy_v41 = np.random.randn(23, 512, 512).astype(np.float32)
                v41_result = await loop.run_in_executor(
                    self._pool, _run_inference, dummy_v41, False, False, False, False, True
                )
                result["v41_warmup_ms"] = v41_result["inference_ms"]
                logger.info("V4.1 inference warmup: %.0f ms", v41_result["inference_ms"])
            except Exception as e:
                logger.warning("V4.1 warmup failed: %s", e)

        return result

    @property
    def is_ready(self) -> bool:
        return self._model_loaded and self._pool is not None

    @property
    def model_loaded(self) -> bool:
        return self._model_loaded

    @property
    def normalization(self) -> Optional[Dict]:
        return self._normalization

    async def predict(
        self,
        input_array: np.ndarray,
        use_v2: bool = False,
        use_v3: bool = False,
        use_v4: bool = False,
        use_v31: bool = False,
        use_v41: bool = False,
        use_v5: bool = False,
        use_v6: bool = False,
        use_v7: bool = False,
    ) -> Dict[str, Any]:
        """
        Run T_mrt prediction on a normalized input.

        V1: (5, 512, 512), V2: (11, 512, 512), V3: (12, 512, 512),
        V3.1: (12, 512, 512) — identical input shape to V3,
        V4: (23, 512, 512), V4.1: (23, 512, 512) — same input as V4,
        V5: (23, 512, 512) — same input as V4, but channel 10 is zeroed on the
        way in (see V5_SHADOW_CHANNEL).

        Returns dict with tmrt_normalized, tmrt_celsius, stats, inference_ms.
        Raises RuntimeError if model not loaded.
        """
        if not self.is_ready:
            raise RuntimeError("Inference service not initialized — ONNX model not loaded")

        async with self._semaphore:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self._pool, _run_inference,
                input_array, use_v2, use_v3, use_v4, use_v31, use_v41, use_v5,
                use_v6, use_v7,
            )

    # Training solar altitude range (from dataset_stats.json temporal combos)
    TRAIN_ALT_MIN = 39.5
    TRAIN_ALT_MAX = 73.3

    async def simulate_patch(
        self,
        center_lat: float,
        center_lon: float,
        hour: int = 13,
        doy: int = 356,
        delta_image_b64: Optional[str] = None,
        colormap: str = "turbo",
        llm_analysis: bool = False,
        model_version: str = "v1.1",
        edit_tools: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Full simulation pipeline: lat/lon → colorized PNG + statistics.

        edit_tools names the tools that painted delta_image_b64. Only the V8 doorway reads it (task 7c); the V1-V7
        path below never sees it.

        model_version:
          "v1"   — raw model output (original behavior, 5-channel)
          "v1.1" — physics-corrected for out-of-range solar angles (5-channel)
          "v2"   — 11-channel model with engineered features (SVF, density, etc.)

        Steps:
        1. Extract 3-channel patch from COGs (DSM, CDSM, Albedo)
        2. Compute solar position channels
        3. Stack to 5-channel input
        3b. (V2) Compute 6 extra features → stack to 11-channel input
        4. (Optional) Merge delta image from user edits
        5. Run ONNX inference (V1 or V2 model)
        6. Denormalize to Celsius
        6b. (V1.1) Apply physics correction for out-of-range solar angles
        7. Colorize to PNG
        8. Compute statistics + UTCI distribution

        Returns dict with heat_map_png (bytes), statistics, bbox, timing, solar.
        """
        import base64

        from app.services.urban_heat_colormap import colorize_tmrt, compute_utci_distribution
        from app.services.urban_heat_patch import extract_patch
        from app.services.urban_heat_solar import compute_solar_position, get_solar_channels

        # V8 is the 2025 world (plan 5): its own module, grid and hour planes. A v8 request leaves here and never runs
        # the V1-V7 path below, and no V1-V7 request can reach the V8 module.
        if model_version == "v8":
            from app.services import urban_heat_v8_world as v8_world
            return await v8_world.simulate_async(
                center_lat=center_lat, center_lon=center_lon, hour=hour, doy=doy,
                colormap=colormap, delta_image_b64=delta_image_b64, edit_tools=edit_tools,
            )

        t_total_start = time.perf_counter()

        # 1. Extract base patch from COGs
        patch_result = extract_patch(center_lat, center_lon)
        base_array = patch_result.array  # (3, 512, 512) normalized

        # 2. Compute solar position
        sun_alt, sun_az = compute_solar_position(center_lat, center_lon, doy, hour)

        # V1.1: Clamp solar altitude to training range for model input,
        # then correct the output using physics scaling
        actual_sun_alt = sun_alt
        is_out_of_range = False
        use_v2 = model_version == "v2"
        use_v3 = model_version == "v3"
        use_v31 = model_version == "v3.1"
        use_v4 = model_version == "v4"
        use_v41 = model_version == "v4.1"
        use_v5 = model_version == "v5"
        use_v6 = model_version == "v6"
        use_v7 = model_version == "v7"

        if model_version == "v1.1":
            if sun_alt < self.TRAIN_ALT_MIN:
                sun_alt_clamped = self.TRAIN_ALT_MIN
                is_out_of_range = True
            elif sun_alt > self.TRAIN_ALT_MAX:
                sun_alt_clamped = self.TRAIN_ALT_MAX
                is_out_of_range = True
            else:
                sun_alt_clamped = sun_alt
            sin_alt, alt_norm = get_solar_channels(sun_alt_clamped)
        else:
            sin_alt, alt_norm = get_solar_channels(sun_alt)

        # 3. Stack solar channels → (5, 512, 512)
        h, w = base_array.shape[1], base_array.shape[2]
        ch_sin_alt = np.full((1, h, w), sin_alt, dtype=np.float32)
        ch_alt_norm = np.full((1, h, w), alt_norm, dtype=np.float32)
        input_5ch = np.concatenate([base_array, ch_sin_alt, ch_alt_norm], axis=0)

        # 3b. V2/V3/V3.1: Compute 6 additional engineered features → (11, 512, 512)
        feature_ms = 0.0
        if use_v2 or use_v3 or use_v31:
            t_feat = time.perf_counter()
            input_11ch = self._compute_v2_features(
                input_5ch, patch_result, sun_alt, sun_az,
            )
            feature_ms = (time.perf_counter() - t_feat) * 1000
            logger.info("%s feature computation: %.0f ms",
                        "V3.1" if use_v31 else ("V3" if use_v3 else "V2"), feature_ms)

        # 3c. V3 / V3.1: swap slots 0 and 2 to Geoscape-derived rasters, append
        #     ERA5 Tair scalar plane → (12, 512, 512). The encoder learned
        #     these specific input distributions during training; reusing
        #     the original DSM/albedo would introduce a small but
        #     systematic bias. V3.1 was trained on the same corpus so it
        #     also expects Geoscape + ERA5 in this layout.
        era5_ms = 0.0
        era5_tair_c: Optional[float] = None
        era5_dict_v4: Optional[Dict[str, float]] = None
        input_23ch: Optional[np.ndarray] = None
        if use_v3 or use_v31:
            tag = "V3.1" if use_v31 else "V3"
            t_v3 = time.perf_counter()
            try:
                input_11ch = self._apply_v3_geoscape_overrides(
                    input_11ch, patch_result,
                )
            except Exception as e:
                logger.warning(
                    "%s: Geoscape override failed (%s); using original DSM/albedo",
                    tag, e,
                )

            era5_tair_c = self._fetch_era5_tair(center_lat, center_lon, doy, hour)
            era5_plane = np.full((1, h, w), float(era5_tair_c), dtype=np.float32)
            input_12ch = np.concatenate([input_11ch, era5_plane], axis=0)
            era5_ms = (time.perf_counter() - t_v3) * 1000
            logger.info("%s Geoscape+ERA5: %.0f ms (Tair=%.2f °C)",
                        tag, era5_ms, era5_tair_c)

        # 3d. V4 / V4.1: Build the 23-channel input via
        #     patch_builder.extract_input_tensor_v4 (Strategy C wrapper). V4.1
        #     shares V4's input layout exactly — only the trained ONNX weights
        #     differ (V4.1 uses Option B annealed variance-preservation loss).
        if use_v4 or use_v41 or use_v5 or use_v6 or use_v7:
            t_v4 = time.perf_counter()
            era5_dict_v4 = self._fetch_era5_full(center_lat, center_lon, doy, hour)
            era5_tair_c = era5_dict_v4["tair_c"]  # for backward-compat output field
            input_23ch = self._build_v4_input(
                patch_result, hour, doy, era5_dict_v4, use_v6=use_v6,
                use_v7=use_v7,
            )
            era5_ms = (time.perf_counter() - t_v4) * 1000
            tag = ("V7" if use_v7 else "V6" if use_v6 else
                   "V5" if use_v5 else "V4.1" if use_v41 else "V4")
            logger.info(
                "%s 23-ch input: %.0f ms (Tair=%.2f °C, RH=%.1f%%, "
                "wind=%.2f m/s, SSRD=%.0f W/m²)",
                tag, era5_ms, era5_dict_v4["tair_c"], era5_dict_v4["rh_pct"],
                era5_dict_v4["wind_speed_ms"], era5_dict_v4["ssrd_wm2"],
            )

        # 4. Optional delta image merge (Phase 1.6.5)
        if delta_image_b64:
            input_5ch = self._merge_delta(input_5ch, delta_image_b64)
            if use_v2 or use_v3 or use_v31:
                input_11ch = self._compute_v2_features(
                    input_5ch, patch_result, sun_alt, sun_az,
                )
            if use_v3 or use_v31:
                # Re-apply V3 overrides + ERA5 channel after delta merge.
                # V3.1 follows V3's exact post-delta layout.
                try:
                    input_11ch = self._apply_v3_geoscape_overrides(
                        input_11ch, patch_result,
                    )
                except Exception:
                    pass
                era5_plane = np.full((1, h, w), float(era5_tair_c or 20.0), dtype=np.float32)
                input_12ch = np.concatenate([input_11ch, era5_plane], axis=0)

            # V4 / V4.1 / V5 build their own 23-ch tensor from the rasters, so
            # the merge above never reaches them. Apply the same edit there,
            # in raw units.
            if (use_v4 or use_v41 or use_v5 or use_v6 or use_v7) and input_23ch is not None:
                input_23ch = self._merge_delta_raw(input_23ch, delta_image_b64)

        # 5. Run ONNX inference
        if use_v4 or use_v41 or use_v5 or use_v6 or use_v7:
            model_input = input_23ch
        elif use_v3 or use_v31:
            # V3.1 shares V3's 12-channel input tensor exactly.
            model_input = input_12ch
        elif use_v2:
            model_input = input_11ch
        else:
            model_input = input_5ch
        result = await self.predict(
            model_input,
            use_v2=use_v2, use_v3=use_v3, use_v4=use_v4, use_v31=use_v31, use_v41=use_v41,
            use_v5=use_v5, use_v6=use_v6, use_v7=use_v7,
        )

        # 6. Denormalize (already done in _run_inference)
        tmrt_celsius = result["tmrt_celsius"]

        # 6a. V5 shade calibration (2026-08-20). V5 runs +1 to +2.3 C warm in
        # true, current building shade (measured per shading trajectory on the
        # test split); subtract the per-(hour x trajectory) bias fitted on the
        # VAL split. Uses cdsm from the tensor (slot 1, raw metres) so Scene
        # Editor canopy edits are respected. Applied before colorize AND
        # statistics/UTCI so every consumer sees one consistent field.
        # No-op with meta.reason if the table or planes are absent.
        shade_calib_meta: Dict[str, Any] = {"applied": False, "reason": "not v5"}
        if use_v5 and input_23ch is not None:
            from app.services.urban_heat_shade_calib import apply_by_bounds
            tmrt_celsius, shade_calib_meta = apply_by_bounds(
                tmrt_celsius, input_23ch[1], patch_result.bbox_32755, hour,
            )

        # 6b. V1.1 Physics correction for out-of-range solar angles
        # V2 does not need this — it was trained with engineered features
        if model_version == "v1.1" and is_out_of_range and actual_sun_alt > 0:
            import math
            correction_factor = math.sin(math.radians(actual_sun_alt)) / math.sin(math.radians(sun_alt_clamped))
            ambient_baseline = 15.0
            tmrt_celsius = ambient_baseline + (tmrt_celsius - ambient_baseline) * correction_factor
            logger.info("V1.1 correction: alt %.1f→%.1f, factor=%.3f",
                        actual_sun_alt, sun_alt_clamped, correction_factor)
        elif model_version == "v1.1" and actual_sun_alt <= 0:
            ambient_night = 12.0
            tmrt_celsius = np.full_like(tmrt_celsius, ambient_night)
            is_out_of_range = True

        # 7. Colorize to PNG — adaptive stretch to this patch's actual range
        # for vivid within-AOI contrast (climatology constants remain the
        # opt-in fallback for an "absolute scale" mode).
        patch_vmin = float(np.min(tmrt_celsius))
        patch_vmax = float(np.max(tmrt_celsius))
        png_bytes = colorize_tmrt(
            tmrt_celsius,
            colormap=colormap,
            vmin=patch_vmin,
            vmax=patch_vmax,
        )

        # 8. Statistics + UTCI
        #
        # Real UTCI needs air temperature, wind and humidity as well as T_mrt.
        # V4/V4.1/V5 fetch all four from ERA5; V3/V3.1 only have Tair, and
        # V1/V2 have nothing. Pass whatever this run actually has and let the
        # UTCI service refuse rather than classify T_mrt as if it were UTCI.
        utci = compute_utci_distribution(
            tmrt_celsius,
            tair_c=era5_dict_v4["tair_c"] if era5_dict_v4 else None,
            wind_ms=era5_dict_v4["wind_speed_ms"] if era5_dict_v4 else None,
            rh_pct=era5_dict_v4["rh_pct"] if era5_dict_v4 else None,
        )

        total_ms = (time.perf_counter() - t_total_start) * 1000

        # Use corrected stats (after physics adjustment)
        corrected_stats = {
            "min_temp": float(np.min(tmrt_celsius)),
            "max_temp": float(np.max(tmrt_celsius)),
            "avg_temp": float(np.mean(tmrt_celsius)),
            "utci_distribution": utci["distribution"],
            # Consumers must be able to tell "no heat stress" from "we could
            # not compute heat stress". Before this, both looked like a
            # distribution.
            "utci_available": utci["available"],
            "utci_basis": utci["basis"],
            "avg_utci": utci.get("avg_utci"),
            "min_utci": utci.get("min_utci"),
            "max_utci": utci.get("max_utci"),
            "utci_inputs": utci.get("inputs"),
        }

        output = {
            "heat_map_png": base64.b64encode(png_bytes).decode("ascii"),
            "statistics": corrected_stats,
            "bbox": list(patch_result.bbox_4326),
            "colormap": colormap,
            # Transparency: what the V5 shade calibration did to this patch
            # (applied False + reason when inactive). See
            # urban_heat_shade_calib.py for the measurement it rests on.
            "shade_calibration": shade_calib_meta,
            "inference_ms": result["inference_ms"],
            "feature_ms": round(feature_ms, 1) if (use_v2 or use_v3 or use_v31) else 0,
            # V4.1 and V5 fetch ERA5 exactly like V4 — they were missing from
            # these three conditions, so the default model reported era5_v4 as
            # null despite having the values in hand. UTCI needs them on the
            # wire, not just inside the tensor.
            "era5_ms": (
                round(era5_ms, 1)
                if (use_v3 or use_v31 or use_v4 or use_v41 or use_v5 or use_v6 or use_v7) else 0
            ),
            "era5_tair_c": (
                round(era5_tair_c, 2)
                if ((use_v3 or use_v31 or use_v4 or use_v41 or use_v5 or use_v6 or use_v7)
                    and era5_tair_c is not None) else None
            ),
            # The full ERA5 4-tuple — V3 only had tair_c.
            "era5_v4": (
                {
                    "tair_c": round(era5_dict_v4["tair_c"], 2),
                    "rh_pct": round(era5_dict_v4["rh_pct"], 1),
                    "wind_speed_ms": round(era5_dict_v4["wind_speed_ms"], 2),
                    "ssrd_wm2": round(era5_dict_v4["ssrd_wm2"], 0),
                }
                if ((use_v4 or use_v41 or use_v5 or use_v6 or use_v7) and era5_dict_v4 is not None)
                else None
            ),
            "total_ms": round(total_ms, 1),
            "model_version": model_version,
            "out_of_training_range": is_out_of_range,
            "solar_position": {
                "altitude": round(actual_sun_alt, 2),
                "azimuth": round(sun_az, 2),
                "sin_altitude": round(sin_alt, 4),
                "altitude_normalized": round(alt_norm, 4),
            },
        }

        # 9. Optional LLM post-processing (Phase 1.6.9)
        if llm_analysis:
            from app.services.urban_heat_llm_postprocess import analyze_prediction

            llm_result = await analyze_prediction(
                heat_map_png_b64=output["heat_map_png"],
                statistics=output["statistics"],
                solar_position=output["solar_position"],
                bbox=output["bbox"],
            )
            output["llm_analysis"] = llm_result
            # Update total_ms to include LLM latency
            output["total_ms"] = round(
                (time.perf_counter() - t_total_start) * 1000, 1
            )

            # Log inference + LLM result for fine-tune dataset curation
            try:
                import uuid as _uuid

                log_dir = Path("urban_heat_data/inference_log")
                log_dir.mkdir(parents=True, exist_ok=True)
                log_entry = {
                    "statistics": output["statistics"],
                    "solar_position": output["solar_position"],
                    "bbox": output["bbox"],
                    "llm_analysis": llm_result,
                    "timestamp": time.time(),
                }
                log_file = log_dir / f"{int(time.time())}_{_uuid.uuid4().hex[:6]}.json"
                log_file.write_text(json.dumps(log_entry, indent=2, default=str))
            except Exception:
                logger.debug("Failed to write inference log", exc_info=True)

        return output

    @staticmethod
    def _compute_v2_features(
        input_5ch: np.ndarray,
        patch_result,
        sun_alt: float,
        sun_az: float,
    ) -> np.ndarray:
        """
        Compute 6 engineered features from the base 5-channel input and
        stack to an 11-channel tensor for V2 inference.

        Channel order matches V2 training:
          0-4:  DSM, CDSM, Albedo, sin_alt, alt_norm  (from input_5ch)
          5:    SVF (sky view factor)
          6:    building_density
          7:    tree_distance / 200
          8:    aspect_sin  (rescaled to 0-1)
          9:    aspect_cos  (rescaled to 0-1)
          10:   shadow_fraction  (computed for actual sun_alt/sun_az —
                must match training/stitch/polygon paths)
        """
        import rasterio
        from rasterio.windows import Window
        from scipy import ndimage

        from app.services.urban_heat_feature_engineering import (
            compute_aspect,
            compute_building_density,
            compute_shadow_fraction,
            compute_svf,
        )
        from app.services.urban_heat_patch import COG_DIR, COG_FILES, PATCH_SIZE

        # Denormalize DSM/CDSM from the base array for feature computation
        dsm = input_5ch[0] * 300.0
        cdsm = input_5ch[1] * 50.0

        # Extract DEM from COG at the same window as the base patch
        dem_path = COG_DIR / COG_FILES["dem"]
        if dem_path.exists():
            with rasterio.open(str(dem_path)) as src:
                row, col = patch_result.center_pixel
                half = PATCH_SIZE // 2
                row_start = max(0, row - half)
                col_start = max(0, col - half)
                window = Window(col_start, row_start, PATCH_SIZE, PATCH_SIZE)
                dem = src.read(1, window=window).astype(np.float32)
                if src.nodata is not None:
                    dem = np.where(dem == src.nodata, 0.0, dem)
                # Handle size mismatch from edge patches
                if dem.shape != (PATCH_SIZE, PATCH_SIZE):
                    padded = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
                    padded[:dem.shape[0], :dem.shape[1]] = dem
                    dem = padded
        else:
            # Fallback: approximate DEM as smoothed DSM
            dem = ndimage.uniform_filter(dsm, size=50).astype(np.float32)

        # Compute features
        svf = compute_svf(dsm, n_azimuths=16, max_radius=100, step=2)
        building_density = compute_building_density(dsm, dem)
        tree_mask = cdsm > 0.5
        tree_distance = ndimage.distance_transform_edt(~tree_mask).astype(np.float32)
        tree_distance = np.clip(tree_distance, 0, 200) / 200.0
        aspect_sin, aspect_cos = compute_aspect(dem)
        # Shadow channel must use the actual sun position the user selected,
        # otherwise V2/V3 are fed a noon-from-north shadow at every hour
        # (training/stitch/polygon all pass real sun args; this used to default).
        shadow = compute_shadow_fraction(
            dsm, sun_altitude_deg=max(sun_alt, 1.0), sun_azimuth_deg=sun_az,
        )

        # Aspect is kept in raw [-1,1] to match training + the
        # polygon-analysis worker + the stitch worker. The previous
        # rescaling here (commented as "to match training normalization")
        # was the actual mismatch: training and every other inference path
        # feed raw [-1,1]. simulate_patch was the only outlier.

        # Stack: (5, H, W) + 6 features → (11, H, W)
        features = np.stack([
            svf, building_density, tree_distance,
            aspect_sin, aspect_cos, shadow
        ], axis=0).astype(np.float32)

        return np.concatenate([input_5ch, features], axis=0)

    @staticmethod
    def _merge_delta(input_5ch: np.ndarray, delta_b64: str) -> np.ndarray:
        """
        Merge user delta edits onto the base tensor.

        Delta RGBA PNG encoding:
          R = DSM height (0-255 → 0-300m, normalized 0-1)
          G = Vegetation/CDSM (0-255 → 0-50m, normalized 0-1)
          B = Albedo (0-255 → 0-1.0)
          A = Mask (255 = use delta pixel, 0 = keep base)

        Formula: Final = Base * (1-alpha) + Delta * alpha
        """
        alpha, delta_3ch = UrbanHeatInferenceService._decode_delta(delta_b64)

        # Merge: only first 3 channels (DSM/CDSM/Albedo), solar channels unchanged
        merged = input_5ch.copy()
        merged[:3] = UrbanHeatInferenceService._blend_base_slots(
            input_5ch[:3], delta_3ch, alpha,
        )

        return merged

    @staticmethod
    def _blend_base_slots(
        base_3ch: np.ndarray, delta_3ch: np.ndarray, alpha: np.ndarray,
    ) -> np.ndarray:
        """Alpha-blend DSM/CDSM/albedo, without demolishing buildings.

        The delta PNG carries ONE alpha mask for all three channels, so an
        intervention that only means to change the roof surface still writes a
        DSM value. Every rooftop and planting preset in InterventionPalette.tsx
        ships `dsm: 0` — cool_roof, green_roof, street_tree, tree_grove — so a
        straight alpha blend flattens whatever building the user painted on.
        "Cool Roof" would then show cooling because the tower was removed, not
        because the roof got brighter: a plausible number produced by the wrong
        mechanism.

        DSM is therefore blended with max(), so an edit may RAISE terrain but
        never lower it. That is exactly right for the current tool set — only
        shade_shelter (dsm: 4) intends a height change, and it adds structure.
        CDSM and albedo blend normally, so "Remove Trees" still clears canopy.

        If a demolition tool is ever added, this rule needs a per-channel edit
        mask in the PNG protocol rather than a looser blend here.
        """
        blended = base_3ch * (1.0 - alpha) + delta_3ch * alpha
        out = blended.copy()
        out[0] = np.maximum(base_3ch[0], blended[0])
        return out

    @staticmethod
    def _decode_delta(delta_b64: str) -> Tuple[np.ndarray, np.ndarray]:
        """Decode the delta RGBA PNG into (alpha, delta_3ch).

        delta_3ch is in the NORMALISED units the PNG encodes:
        DSM/300, CDSM/50, albedo as-is. Callers working in raw metres must
        scale — see _merge_delta_raw.
        """
        import base64
        import io

        from PIL import Image

        png_bytes = base64.b64decode(delta_b64)
        img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
        img = img.resize((512, 512), Image.NEAREST)
        delta_rgba = np.array(img, dtype=np.float32)  # (512, 512, 4)

        alpha = delta_rgba[:, :, 3:4].transpose(2, 0, 1) / 255.0  # (1, 512, 512)
        delta_dsm = delta_rgba[:, :, 0:1].transpose(2, 0, 1) / 255.0
        delta_cdsm = delta_rgba[:, :, 1:2].transpose(2, 0, 1) / 255.0
        delta_albedo = delta_rgba[:, :, 2:3].transpose(2, 0, 1) / 255.0
        delta_3ch = np.concatenate([delta_dsm, delta_cdsm, delta_albedo], axis=0)
        return alpha, delta_3ch

    @staticmethod
    def _merge_delta_raw(input_23ch: np.ndarray, delta_b64: str) -> np.ndarray:
        """Merge Scene Editor edits into the V4/V4.1/V5 23-channel tensor.

        Until this existed, delta edits were merged only into input_5ch /
        11ch / 12ch, while V4/V4.1/V5 ran on input_23ch built straight from
        the rasters. The user's interventions were silently discarded, and
        because V4.1 is the default, that was the behaviour almost everyone
        got: "AVG REDUCTION 0.0 °C" was the baseline being returned twice.

        The 23-channel tensor is in raw physical units, so the normalised
        PNG values are scaled back up before merging.

        Only slots 0-2 are edited. The geometry-derived features (SVF,
        building density, tree distance, shadow) and the scene-level extras
        (NDVI, roof type, multi-shadow) are left at their unedited values.
        That is deliberate: recomputing them would route the edit through
        compute_shadow_fraction, whose ray-trace is known to run backwards,
        and slots 0-2 alone already carry the intervention — planting 8 m of
        canopy through this path moves V4.1 by about -10 °C and V5 by about
        -25 °C in the edited area.
        """
        alpha, delta_3ch = UrbanHeatInferenceService._decode_delta(delta_b64)

        # PNG encoding -> raw units: R*300 m, G*50 m, B as a 0-1 fraction.
        scale = np.array([300.0, 50.0, 1.0], dtype=np.float32).reshape(3, 1, 1)
        delta_raw = delta_3ch * scale

        merged = input_23ch.copy()
        merged[:3] = UrbanHeatInferenceService._blend_base_slots(
            input_23ch[:3], delta_raw, alpha,
        )
        return merged

    @property
    def v3_available(self) -> bool:
        return ONNX_V3_MODEL_PATH.exists()

    @property
    def v4_available(self) -> bool:
        """V4 needs BOTH the .onnx graph and the .onnx.data weights file
        to be co-located. A graph-only file would load but produce garbage.
        """
        return (
            ONNX_V4_MODEL_PATH.exists()
            and ONNX_V4_MODEL_PATH.with_suffix(".onnx.data").exists()
        )

    @property
    def v31_available(self) -> bool:
        """V3.1 uses the same external-data ONNX format as V4 — both files
        must be co-located. Graph-alone would load but produce garbage.
        """
        return (
            ONNX_V31_MODEL_PATH.exists()
            and ONNX_V31_MODEL_PATH.with_suffix(".onnx.data").exists()
        )

    @property
    def v41_available(self) -> bool:
        """V4.1 was exported as a SINGLE-file ONNX (not external-data like
        V4/V3.1) via the legacy dynamo=False exporter — just the .onnx file
        is enough, no .onnx.data sidecar.
        """
        return ONNX_V41_MODEL_PATH.exists()

    @property
    def v6_available(self) -> bool:
        """V6: V5's architecture retrained on CORRECTED shadow channels 17/18/19.

        Selectable but NOT default. On the held-out test split V6 scores
        RMSE 4.469 C against V5's 4.182 -- it does NOT beat V5 on aggregate, and
        is offered for render comparison, not as a replacement.
        """
        return ONNX_V6_MODEL_PATH.exists()

    @property
    def v7_available(self) -> bool:
        """V7: one consistent 2018 world + shade-age channel. Needs the ONNX
        AND every V7 raster (dsm_1m, shadow_multi_v7 planes, 4 age planes):
        without the age raster V7 is +9.4 C wrong in every shadow, so a
        partial install is reported as unavailable, not served degraded."""
        from app.services.urban_heat_v7_world import rasters_available
        return ONNX_V7_MODEL_PATH.exists() and rasters_available()

    @property
    def v5_available(self) -> bool:
        """V5: single-file ONNX, no .onnx.data sidecar. Trained on reference
        SOLWEIG labels rather than the old `Tground = Ta + 8` engine.
        Selectable but NOT default -- V4.1 stays default until V5's renders are
        approved.
        """
        return ONNX_V5_MODEL_PATH.exists()

    # In-memory cache for ERA5 lookups: (doy, hour, year) → (fetch_ts, dict).
    # Keyed at AOI level (not per-pixel) — ERA5_LAND varies < 0.5 °C across
    # the 9 km × 9 km grid so AOI-level caching is fine.
    _ERA5_CACHE: Dict[tuple, tuple] = {}
    _ERA5_TTL_SECONDS = 3600  # 1 hr — refresh once per hour
    _AOI_CENTRE_LONLAT = (144.93, -37.83)
    _ERA5_FALLBACK_TAIR = 20.0  # used if GEE call fails

    def _fetch_era5_tair(
        self, lat: float, lon: float, doy: int, hour: int, year: int = 2024,
    ) -> float:
        """Sample ERA5-Land Tair (°C) for a (doy, hour) at the AOI centre.

        Cached for 1 hour — ERA5 values for fixed (doy, hour) are
        identical across the 2024 reanalysis, so the cache is never
        stale within a single Colab run. The user said live GEE is
        worth waiting a few seconds; we still cache to avoid hammering
        GEE on repeated clicks at the same time-of-day.

        Falls back to _ERA5_FALLBACK_TAIR if GEE init / sampling fails
        — V3 still produces reasonable predictions in that case
        (channel 11 just becomes a constant 20 °C).
        """
        from datetime import datetime, timedelta, timezone

        key = (doy, hour, year)
        cached = self._ERA5_CACHE.get(key)
        if cached is not None:
            cached_ts, val = cached
            if time.time() - cached_ts < self._ERA5_TTL_SECONDS:
                return val

        try:
            from app.services.urban_heat.data_pipeline import gee_client as gc

            gc.init()
            # AEST hour → UTC datetime
            aest = datetime(year, 1, 1, hour=hour, tzinfo=timezone.utc) \
                + timedelta(days=doy - 1)
            ts_utc = aest - timedelta(hours=10)
            # Sample at AOI centre (not the user's click) — ERA5 is
            # ~9 km native and barely varies across our 9 km grid.
            sample = gc.era5_hourly_at_point(
                self._AOI_CENTRE_LONLAT[0], self._AOI_CENTRE_LONLAT[1], ts_utc,
            )
            if sample and sample.get("tair_c") is not None:
                val = float(sample["tair_c"])
                self._ERA5_CACHE[key] = (time.time(), val)
                return val
        except Exception as e:
            logger.warning("ERA5 fetch failed for (doy=%d, hour=%d): %s", doy, hour, e)

        return self._ERA5_FALLBACK_TAIR

    # V4-only ERA5 cache + fallback. Kept fully independent of the V3 cache
    # above so failure modes are isolated — V3 should never be affected by
    # changes to V4's ERA5 fetcher and vice versa.
    _ERA5_FULL_CACHE: Dict[tuple, tuple] = {}
    _ERA5_FULL_FALLBACK: Dict[str, float] = {
        "tair_c": 20.0,
        "rh_pct": 60.0,
        "wind_speed_ms": 2.5,
        "ssrd_wm2": 600.0,
    }

    def _fetch_era5_full(
        self, lat: float, lon: float, doy: int, hour: int, year: int = 2024,
    ) -> Dict[str, float]:
        """V4: Sample ERA5-Land 4-tuple {tair_c, rh_pct, wind_speed_ms, ssrd_wm2}
        at the AOI centre.

        V4 needs all 4 fields (V3 only consumed tair_c). The underlying GEE
        call already returns the 4-tuple — V3's _fetch_era5_tair simply
        discarded 3 of the 4. This method preserves all 4.

        Cached independently of _fetch_era5_tair (separate _ERA5_FULL_CACHE)
        so V3 behaviour is unaffected by V4 cache state.
        """
        from datetime import datetime, timedelta, timezone

        key = (doy, hour, year)
        cached = self._ERA5_FULL_CACHE.get(key)
        if cached is not None:
            cached_ts, val = cached
            if time.time() - cached_ts < self._ERA5_TTL_SECONDS:
                return val

        try:
            from app.services.urban_heat.data_pipeline import gee_client as gc

            gc.init()
            aest = datetime(year, 1, 1, hour=hour, tzinfo=timezone.utc) \
                + timedelta(days=doy - 1)
            ts_utc = aest - timedelta(hours=10)
            sample = gc.era5_hourly_at_point(
                self._AOI_CENTRE_LONLAT[0], self._AOI_CENTRE_LONLAT[1], ts_utc,
            )
            if sample:
                # GEE returns None for missing fields — fall back per-field
                # to defaults rather than failing the whole tuple.
                val = {
                    "tair_c": float(sample.get("tair_c") or self._ERA5_FULL_FALLBACK["tair_c"]),
                    "rh_pct": float(sample.get("rh_pct") or self._ERA5_FULL_FALLBACK["rh_pct"]),
                    "wind_speed_ms": float(
                        sample.get("wind_speed_ms")
                        or self._ERA5_FULL_FALLBACK["wind_speed_ms"]
                    ),
                    "ssrd_wm2": float(sample.get("ssrd_wm2") or self._ERA5_FULL_FALLBACK["ssrd_wm2"]),
                }
                self._ERA5_FULL_CACHE[key] = (time.time(), val)
                return val
        except Exception as e:
            logger.warning(
                "ERA5 full fetch failed for (doy=%d, hour=%d): %s — using fallback",
                doy, hour, e,
            )

        return dict(self._ERA5_FULL_FALLBACK)

    def _build_v4_input(
        self,
        patch_result,
        hour: int,
        doy: int,
        era5_dict: Dict[str, float],
        use_v6: bool = False,
        use_v7: bool = False,
    ) -> np.ndarray:
        """V4: Build the 23-channel input tensor (Strategy C wrapper).

        Composes patch_builder.extract_input_tensor_v4 — the canonical
        23-channel constructor used by the offline corpus build script —
        by translating inference-side (lat, lon, hour, doy) into the
        PatchSpec/Window form it expects. This keeps V4_CHANNEL_ORDER as
        a single source of truth: any future channel change touches only
        patch_builder.py, not this file.

        The window is computed from patch_result.center_pixel (which lies
        on the master DSM grid — the same grid all V4 rasters share).

        Note: V4 reads from the Geoscape rasters for slots 0/2 (DSM/albedo)
        directly, so user delta_image edits to those channels are silently
        ignored — this is inherited V3 behaviour, not a new bug.
        """
        from datetime import datetime, timedelta, timezone

        from rasterio.windows import Window

        from app.services.urban_heat.data_pipeline.patch_builder import (
            PatchSpec, extract_input_tensor_v4,
        )
        from app.services.urban_heat_patch import COG_DIR, COG_FILES, PATCH_SIZE

        # Compute window on the master DSM grid from the centre pixel
        # (extract_patch already located the centre in this same grid).
        row, col = patch_result.center_pixel
        half = PATCH_SIZE // 2
        win = Window(col - half, row - half, PATCH_SIZE, PATCH_SIZE)

        # PatchSpec timestamp must yield .hour == hour and tm_yday == doy.
        # Year is arbitrary (2024 matches the V4 corpus build year).
        timestamp = (
            datetime(2024, 1, 1, hour=hour, tzinfo=timezone.utc)
            + timedelta(days=doy - 1)
        )
        spec = PatchSpec(
            patch_id=f"sim_{int(time.time() * 1000)}",
            window=win,
            timestamp=timestamp,
        )
        raster_paths = {
            "dsm_corrected": GEOSCAPE_HEIGHTS_PATH,
            "cdsm": COG_DIR / COG_FILES["cdsm"],
            "dem": COG_DIR / COG_FILES["dem"],
            "albedo": GEOSCAPE_ALBEDO_PATH,
        }
        v4_raster_paths = {
            "ndvi": V4_NDVI_PATH,
            "solar_panel_mask": V4_SOLAR_PANEL_PATH,
            # VERSION-CONDITIONAL. V6 trained on the corrected rasters and must
            # be served them; V4/V4.1/V5 learned the mirrored convention and
            # must keep it. Serving either the other one is a train/serve skew.
            "shadow_h09": V6_SHADOW_H09_PATH if use_v6 else V4_SHADOW_H09_PATH,
            "shadow_h13": V6_SHADOW_H13_PATH if use_v6 else V4_SHADOW_H13_PATH,
            "shadow_h16": V6_SHADOW_H16_PATH if use_v6 else V4_SHADOW_H16_PATH,
            "roof_type": V4_ROOF_TYPE_PATH,
        }
        tensor, _ = extract_input_tensor_v4(
            spec, raster_paths=raster_paths,
            era5_dict=era5_dict, v4_raster_paths=v4_raster_paths,
        )
        if use_v7:
            # V7's world is laid OVER the V5 tensor (slots 0/10/17-19), the
            # same way its corpus was built; svf/density stay V5's. Raises if
            # any V7 raster is missing -- never serve V7 zeros in slot 10.
            from app.services.urban_heat_v7_world import apply_v7_world
            tensor, v7_meta = apply_v7_world(
                tensor, int(win.col_off), int(win.row_off), hour,
            )
            logger.info("V7 world applied: age hour %s, shaded now %.1f%%",
                        v7_meta["age_hour_used"], v7_meta["shaded_now_pct"])
        return tensor

    def _apply_v3_geoscape_overrides(self, input_11ch: np.ndarray, patch_result) -> np.ndarray:
        """Replace channels 0 (DSM) and 2 (albedo) with Geoscape-derived values.

        V3 was trained with `geoscape_heights.tif` (clean per-building
        DSM correction) and `geoscape_albedo.tif` (per-building material
        × colour blend) in those slots. Using the original noisy DSM /
        flat albedo at inference would produce a small but systematic
        bias because the encoder learned the Geoscape distribution.

        Reads at the same window patch_result was extracted from, then
        normalises to the [0, 1] range V3 trained against (DSM/300,
        Albedo/1.0) — same as V1/V2's normalisation for these slots.
        """
        import rasterio
        from rasterio.windows import Window

        out = input_11ch.copy()
        # PatchResult exposes the UTM bbox as `bbox_32755`. Compute the raster
        # pixel window per-Geoscape-raster from those coords so spatial
        # alignment is exact regardless of each raster's transform. The
        # previous code referenced `.window` and `.bbox_utm` — neither exists
        # on PatchResult — causing every call to silently fall back to the
        # V1/V2 inputs at the wrapping try/except (a train/inference
        # distribution mismatch that hid V3's intended Geoscape signal).
        bbox = patch_result.bbox_32755  # (west, south, east, north) in EPSG:32755

        # Channel 0: dsm_corrected
        if GEOSCAPE_HEIGHTS_PATH.exists():
            with rasterio.open(GEOSCAPE_HEIGHTS_PATH) as ds:
                col, row = ~ds.transform * (bbox[0], bbox[3])  # left, top
                win = Window(int(round(col)), int(round(row)), 512, 512)
                dsm = ds.read(1, window=win).astype(np.float32)
                if dsm.shape == (512, 512):
                    nodata = ds.nodata if ds.nodata is not None else -32767.0
                    dsm[dsm == nodata] = 0.0
                    dsm[dsm < -100.0] = 0.0  # bleed sentinel cleanup
                    out[0] = dsm / 300.0  # same normalisation V1/V2 used

        # Channel 2: albedo
        if GEOSCAPE_ALBEDO_PATH.exists():
            with rasterio.open(GEOSCAPE_ALBEDO_PATH) as ds:
                col, row = ~ds.transform * (bbox[0], bbox[3])  # left, top
                win = Window(int(round(col)), int(round(row)), 512, 512)
                alb = ds.read(1, window=win).astype(np.float32)
                if alb.shape == (512, 512):
                    out[2] = alb  # already in [0, 1] range

        return out

    def get_status(self) -> Dict[str, Any]:
        """Status dict for the /urban-heat/status endpoint."""
        model_path = self._find_model_path()
        available = ["v1", "v1.1"]
        if self.v2_available:
            available.append("v2")
        if self.v3_available:
            available.append("v3")
        if self.v31_available:
            available.append("v3.1")
        if self.v4_available:
            available.append("v4")
        if self.v41_available:
            available.append("v4.1")
        if self.v5_available:
            available.append("v5")
        if self.v6_available:
            available.append("v6")
        if self.v7_available:
            available.append("v7")
        status = {
            "model_loaded": self._model_loaded,
            "model_exists": model_path is not None,
            "warmup_done": self._warmup_done,
            "max_workers": MAX_INFERENCE_WORKERS,
            "v5_available": self.v5_available,
            "v6_available": self.v6_available,
            "v2_available": self.v2_available,
            "v3_available": self.v3_available,
            "v3.1_available": self.v31_available,
            "v4_available": self.v4_available,
            "v4.1_available": self.v41_available,
            "available_versions": available,
        }
        if model_path and model_path.exists():
            status["model_path"] = model_path.name
            status["model_size_mb"] = round(model_path.stat().st_size / (1024 * 1024), 1)
        if self.v2_available:
            status["v2_model_path"] = ONNX_V2_MODEL_PATH.name
            status["v2_model_size_mb"] = round(ONNX_V2_MODEL_PATH.stat().st_size / (1024 * 1024), 1)
        if self.v3_available:
            status["v3_model_path"] = ONNX_V3_MODEL_PATH.name
            status["v3_model_size_mb"] = round(ONNX_V3_MODEL_PATH.stat().st_size / (1024 * 1024), 1)
        if self.v4_available:
            # V4 size = graph stub + weights blob (external-data format)
            v4_data = ONNX_V4_MODEL_PATH.with_suffix(".onnx.data")
            v4_total_bytes = (
                ONNX_V4_MODEL_PATH.stat().st_size
                + (v4_data.stat().st_size if v4_data.exists() else 0)
            )
            status["v4_model_path"] = ONNX_V4_MODEL_PATH.name
            status["v4_model_size_mb"] = round(v4_total_bytes / (1024 * 1024), 1)
        if self.v31_available:
            # V3.1 also uses external-data format (graph stub + .onnx.data)
            v31_data = ONNX_V31_MODEL_PATH.with_suffix(".onnx.data")
            v31_total_bytes = (
                ONNX_V31_MODEL_PATH.stat().st_size
                + (v31_data.stat().st_size if v31_data.exists() else 0)
            )
            status["v3.1_model_path"] = ONNX_V31_MODEL_PATH.name
            status["v3.1_model_size_mb"] = round(v31_total_bytes / (1024 * 1024), 1)
        if self.v41_available:
            # V4.1 is single-file (legacy exporter), unlike V4/V3.1 external-data.
            status["v4.1_model_path"] = ONNX_V41_MODEL_PATH.name
            status["v4.1_model_size_mb"] = round(ONNX_V41_MODEL_PATH.stat().st_size / (1024 * 1024), 1)
        if self._normalization:
            status["normalization"] = self._normalization
        return status

    async def shutdown(self):
        """Shutdown the worker pool."""
        if self._pool:
            self._pool.shutdown(wait=False)
            self._pool = None
            self._model_loaded = False
            logger.info("Inference pool shut down")


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: Optional[UrbanHeatInferenceService] = None


def get_inference_service() -> UrbanHeatInferenceService:
    """Get or create the singleton inference service."""
    global _instance
    if _instance is None:
        _instance = UrbanHeatInferenceService()
    return _instance
