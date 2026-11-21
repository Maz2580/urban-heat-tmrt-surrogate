"""
Urban Heat Colormap & UTCI Service (Phase 1.6.4)
=================================================

Converts raw T_mrt output (512x512 float32 °C) to:
1. Colorized PNG using thermal colormaps (turbo/inferno/FLIR ironbow)
2. UTCI stress category distribution

T_mrt range: 0-63°C (from dataset_stats.json normalization)
"""

import io
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("umami.urban_heat_colormap")

# ---------------------------------------------------------------------------
# Melbourne climatology scale — fixed range for all T_mrt visualization paths.
#
# Locks colormap stretch to a fixed physical span so "same red = same °C"
# across models (V1/V1.1/V2/V3/V4), across jobs (different stitches), and
# across baseline/modified pairs in polygon analysis. Without this, each
# rendering path stretched the turbo gradient over each job's own min/max,
# making cross-model and before/after comparisons visually meaningless even
# when the underlying numbers agreed.
#
# Range chosen for Melbourne TMRT envelope: ~15 °C (pre-dawn shaded surface)
# to ~55 °C (peak summer sunlit asphalt). Values outside clamp to ramp ends.
# ---------------------------------------------------------------------------
MELBOURNE_TMRT_VMIN: float = 15.0
MELBOURNE_TMRT_VMAX: float = 55.0


# ---------------------------------------------------------------------------
# Absolute display ranges, per MODEL FAMILY
# ---------------------------------------------------------------------------
# V5 onward were trained on regenerated SOLWEIG labels with a corrected surface
# energy balance, and predict roughly 25 C higher than V4.1 for the same place.
# Measured on V5's held-out test split (2026-08-20, 192,801,072 label pixels):
# labels average 48.74 C and reach 69.35 C, where the V4.1-era labels averaged
# about 20 C. One absolute scale cannot serve both families -- 55 C is a
# sensible ceiling for V4.1 and far too low for V5.
#
# V5 AND V6 DELIBERATELY SHARE ONE RANGE. They predict the same quantity
# against the same labels; only the shadow channels they were trained on
# differ. Giving them separate ranges would make the V5-vs-V6 comparison
# meaningless in exactly the way per-image auto-stretch already does -- the
# same colour would mean a different temperature in each half of the split.
#
# These serve two purposes and neither is the default:
#   1. the ABSOLUTE display mode (opt-in, strict cross-model colour identity)
#   2. the fallback when a job carries no statistics
# The default everywhere remains per-AOI adaptive stretch, which is correct for
# "which street is hotter here" and is why single-AOI views look vivid.
NEW_LABEL_MODEL_VERSIONS = frozenset({"v5", "v6", "v7", "v8"})
NEW_LABEL_TMRT_VMIN: float = 15.0
NEW_LABEL_TMRT_VMAX: float = 75.0


def absolute_range_for(model_version: Optional[str]) -> Tuple[float, float]:
    """Fixed display range for a model version's family.

    Unknown or missing versions fall back to the Melbourne climatology scale,
    which is the pre-V5 behaviour -- a new model version that nobody added here
    renders on the OLD scale and looks obviously wrong (saturated at the top)
    rather than silently rescaling and looking plausible.
    """
    if model_version in NEW_LABEL_MODEL_VERSIONS:
        return NEW_LABEL_TMRT_VMIN, NEW_LABEL_TMRT_VMAX
    return MELBOURNE_TMRT_VMIN, MELBOURNE_TMRT_VMAX


# ---------------------------------------------------------------------------
# UTCI Stress Categories
# ---------------------------------------------------------------------------

# Re-exported so existing importers keep working. The real definition, and the
# real UTCI computation, now live in urban_heat_utci.py.
from app.services.urban_heat_utci import UTCI_CATEGORIES  # noqa: E402,F401
from app.services.urban_heat_utci import (  # noqa: E402
    compute_utci_distribution as _compute_utci_distribution,
)


def compute_utci_distribution(
    tmrt_celsius: np.ndarray,
    tair_c: float | None = None,
    wind_ms: float | None = None,
    rh_pct: float | None = None,
) -> Dict[str, Any]:
    """Classify a T_mrt field into UTCI stress categories.

    This used to apply UTCI's thresholds directly to T_mrt and call the result
    UTCI. It is not: UTCI also depends on air temperature, wind and humidity,
    so the same T_mrt in still humid air and in a dry breeze are different
    thermal stresses. The docstring here promised the fix "when weather data
    integration is complete" -- it is, so this now delegates to the real
    Broede et al. (2012) polynomial.

    Callers that cannot supply meteorology get `available: False` rather than a
    T_mrt-based distribution wearing a UTCI label.
    """
    return _compute_utci_distribution(
        tmrt_celsius, tair_c=tair_c, wind_ms=wind_ms, rh_pct=rh_pct,
    )


# ---------------------------------------------------------------------------
# Colormaps
# ---------------------------------------------------------------------------

def _get_flir_ironbow_cmap():
    """Create FLIR Ironbow-style colormap (dark purple → blue → yellow → red → white)."""
    from matplotlib.colors import LinearSegmentedColormap

    colors = [
        (0.0, "#000020"),   # Deep dark blue/black
        (0.15, "#1a0a4e"),  # Dark purple
        (0.30, "#3b0f8c"),  # Purple
        (0.40, "#6a1fb0"),  # Blue-purple
        (0.50, "#b03060"),  # Magenta-red
        (0.60, "#d04020"),  # Orange-red
        (0.70, "#e88010"),  # Orange
        (0.80, "#f0c000"),  # Yellow-orange
        (0.90, "#f8e838"),  # Yellow
        (1.0, "#ffffff"),   # White (hottest)
    ]
    positions = [c[0] for c in colors]
    hex_colors = [c[1] for c in colors]

    # Convert hex to RGB tuples
    rgb_colors = []
    for h in hex_colors:
        h = h.lstrip("#")
        rgb_colors.append(tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4)))

    return LinearSegmentedColormap.from_list("flir_ironbow", list(zip(positions, rgb_colors)))


AVAILABLE_COLORMAPS = {
    "turbo": {
        "name": "Turbo",
        "description": "Perceptually uniform rainbow — good for heat maps",
        "mpl_name": "turbo",
    },
    "inferno": {
        "name": "Inferno",
        "description": "Dark background thermal — high contrast",
        "mpl_name": "inferno",
    },
    "flir_ironbow": {
        "name": "FLIR Ironbow",
        "description": "Thermal camera palette — familiar to thermographers",
        "mpl_name": None,  # Custom colormap
    },
}


def _get_colormap(name: str):
    """Get matplotlib colormap by name."""
    import matplotlib.pyplot as plt

    if name == "flir_ironbow":
        return _get_flir_ironbow_cmap()

    info = AVAILABLE_COLORMAPS.get(name)
    if info and info.get("mpl_name"):
        return plt.get_cmap(info["mpl_name"])

    # Fallback to turbo
    return plt.get_cmap("turbo")


def colorize_tmrt(
    tmrt_celsius: np.ndarray,
    colormap: str = "turbo",
    vmin: float = MELBOURNE_TMRT_VMIN,
    vmax: float = MELBOURNE_TMRT_VMAX,
) -> bytes:
    """
    Apply a thermal colormap to T_mrt array and return PNG bytes.

    Args:
        tmrt_celsius: (H, W) array of T_mrt in degrees Celsius
        colormap: Name of colormap (turbo/inferno/flir_ironbow)
        vmin: Minimum temperature for color scaling
        vmax: Maximum temperature for color scaling

    Returns:
        PNG image bytes (RGBA, 512x512)
    """
    from PIL import Image

    cmap = _get_colormap(colormap)

    # Normalize to [0, 1]; guard against degenerate adaptive ranges (e.g. an
    # entirely uniform patch) where vmin == vmax would otherwise divide by zero.
    denom = max(vmax - vmin, 1e-6)
    normalized = np.clip((tmrt_celsius - vmin) / denom, 0.0, 1.0)

    # Apply colormap → (H, W, 4) RGBA float [0,1]
    colored = cmap(normalized)

    # Convert to uint8 RGBA
    rgba = (colored * 255).astype(np.uint8)

    # Encode to PNG
    img = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def generate_colormap_preview(colormap: str, width: int = 256, height: int = 20) -> bytes:
    """Generate a small gradient preview image for a colormap."""
    from PIL import Image

    cmap = _get_colormap(colormap)
    gradient = np.linspace(0, 1, width)
    gradient = np.tile(gradient, (height, 1))
    colored = (cmap(gradient) * 255).astype(np.uint8)

    img = Image.fromarray(colored, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def list_colormaps() -> List[Dict[str, str]]:
    """List available colormaps with metadata."""
    import base64

    result = []
    for cmap_id, info in AVAILABLE_COLORMAPS.items():
        preview_bytes = generate_colormap_preview(cmap_id)
        result.append({
            "id": cmap_id,
            "name": info["name"],
            "description": info["description"],
            "preview_gradient": base64.b64encode(preview_bytes).decode("ascii"),
        })
    return result
