"""Plan UH-1 / A4 — patch-level feature builder.

Composes the 12-channel V2.1 input tensor + 3 multi-task labels for one
(window, timestamp) tuple. Caller is responsible for choosing which
windows × timestamps to build (typically a driver script in
`scripts/urban_heat/`).

Channel order (must match `urban_heat_inference._stack_v2_input` and
the V2.1 training notebook):

    0  DSM-corrected     — geoscape_heights.tif
    1  CDSM              — cdsm_1m.tif
    2  Albedo            — geoscape_albedo.tif
    3  sin(altitude)     — solar position from (lat, lon, ts)
    4  altitude / 90     — solar position
    5  SVF               — derived from DSM
    6  building_density  — derived from DSM-DEM
    7  tree_distance     — derived from CDSM
    8  aspect_sin        — derived from DEM
    9  aspect_cos        — derived from DEM
    10 shadow_fraction   — derived from DSM + solar
    11 ERA5 Tair (°C)    — broadcast scalar

Labels:
    tmrt        — SOLWEIG synthetic (dense)
    lst         — Landsat LST resampled (NaN where outside scene)
    tair_sparse — sensor air temperature (NaN except at sensor cells)
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import rasterio
from rasterio.windows import Window

from app.services.urban_heat_feature_engineering import (
    compute_aspect,
    compute_building_density,
    compute_shadow_fraction,
    compute_svf,
    compute_tree_distance,
)
from app.services.urban_heat_solar import (
    compute_solar_position,
    get_solar_channels,
)

logger = logging.getLogger("umami.urban_heat.patch_builder")

CHANNEL_ORDER = (
    "dsm_corrected", "cdsm", "albedo",
    "sin_alt", "alt_norm",
    "svf", "building_density", "tree_distance",
    "aspect_sin", "aspect_cos", "shadow_fraction",
    "era5_tair_c",
)
NUM_CHANNELS = len(CHANNEL_ORDER)
assert NUM_CHANNELS == 12, "V2.1 must be exactly 12 channels"

# Default patch size — must match V2 model's expected input
DEFAULT_PATCH_SIZE = 512

# Sensor sample tolerance (matches sensor_archive.to_training_label)
SENSOR_MAX_DIST_M = 50.0
SENSOR_MAX_TIME_MIN = 15.0


# ---------------------------------------------------------------------------
# Spec dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PatchSpec:
    """One (window, timestamp) tuple to build a patch for."""
    patch_id: str
    window: Window
    timestamp: datetime  # tz-aware
    # Optional metadata copied through to the manifest row
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None:
            object.__setattr__(
                self, "timestamp",
                self.timestamp.replace(tzinfo=timezone.utc),
            )


# ---------------------------------------------------------------------------
# Window I/O helpers
# ---------------------------------------------------------------------------

def _read_window(
    raster_path: Path, window: Window, *, fill_nodata: float = 0.0,
) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    """Read a window from a raster, replacing NoData with fill value.

    Returns (array, (lon_min, lat_min, lon_max, lat_max)) where the bbox
    is in EPSG:4326 (reprojected from the raster's CRS).
    """
    with rasterio.open(raster_path) as ds:
        arr = ds.read(1, window=window).astype(np.float32)
        nodata = ds.nodata
        if nodata is not None:
            arr[arr == nodata] = fill_nodata
        # Catch undeclared sentinels (e.g., DSM uses -32767 with nodata=None)
        arr[arr < -1000.0] = fill_nodata

        # Compute bbox in raster CRS, then reproject corner to lon/lat
        bounds = rasterio.windows.bounds(window, ds.transform)
        from rasterio.warp import transform as warp_transform
        xs = [bounds[0], bounds[2]]
        ys = [bounds[1], bounds[3]]
        lon, lat = warp_transform(ds.crs, "EPSG:4326", xs, ys)
        bbox = (min(lon), min(lat), max(lon), max(lat))
    return arr, bbox


def _patch_centroid_lonlat(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    return (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2


# ---------------------------------------------------------------------------
# Input tensor builder
# ---------------------------------------------------------------------------

def extract_input_tensor(
    spec: PatchSpec,
    *,
    raster_paths: Dict[str, Path],
    era5_tair_c: float,
    utc_offset_hours: int = 10,  # AEST
) -> Tuple[np.ndarray, Tuple[float, float]]:
    """Build the (12, H, W) input tensor for one patch.

    raster_paths must contain keys:
        dsm_corrected, cdsm, dem, albedo
    (DEM is needed for aspect + building density, even though it isn't
    a model input itself.)

    Returns (tensor, (centroid_lon, centroid_lat)).
    """
    required = {"dsm_corrected", "cdsm", "dem", "albedo"}
    missing = required - set(raster_paths)
    if missing:
        raise ValueError(f"raster_paths missing required keys: {missing}")

    dsm, bbox = _read_window(raster_paths["dsm_corrected"], spec.window)
    cdsm, _ = _read_window(raster_paths["cdsm"], spec.window)
    dem, _ = _read_window(raster_paths["dem"], spec.window)
    albedo, _ = _read_window(raster_paths["albedo"], spec.window)

    h, w = dsm.shape
    if (cdsm.shape, dem.shape, albedo.shape) != ((h, w), (h, w), (h, w)):
        raise ValueError(
            f"raster windows produced mismatched shapes: dsm={dsm.shape}, "
            f"cdsm={cdsm.shape}, dem={dem.shape}, albedo={albedo.shape}",
        )

    # Solar channels — use patch centroid lat/lon
    cen_lon, cen_lat = _patch_centroid_lonlat(bbox)
    doy = spec.timestamp.timetuple().tm_yday
    hour = spec.timestamp.hour
    sun_alt, sun_az = compute_solar_position(
        cen_lat, cen_lon, doy, hour, utc_offset=utc_offset_hours,
    )
    sin_alt, alt_norm = get_solar_channels(sun_alt)

    # Engineered features — match V2 inference's normalisation
    svf = compute_svf(dsm)
    bldg_density = compute_building_density(dsm, dem)
    tree_dist = compute_tree_distance(cdsm) / 200.0  # normalise to [0,1]
    aspect_sin, aspect_cos = compute_aspect(dem)
    shadow = compute_shadow_fraction(
        dsm, sun_altitude_deg=max(sun_alt, 0.0), sun_azimuth_deg=sun_az,
    )

    # ERA5 Tair — broadcast scalar across the full patch
    era5_plane = np.full((h, w), float(era5_tair_c), dtype=np.float32)
    sin_alt_plane = np.full((h, w), float(sin_alt), dtype=np.float32)
    alt_norm_plane = np.full((h, w), float(alt_norm), dtype=np.float32)

    tensor = np.stack(
        [
            dsm, cdsm, albedo,
            sin_alt_plane, alt_norm_plane,
            svf, bldg_density, tree_dist,
            aspect_sin, aspect_cos, shadow,
            era5_plane,
        ],
        axis=0,
    ).astype(np.float32)
    if tensor.shape[0] != NUM_CHANNELS:
        raise RuntimeError(
            f"channel count mismatch: got {tensor.shape[0]}, want {NUM_CHANNELS}",
        )
    return tensor, (cen_lon, cen_lat)


# ---------------------------------------------------------------------------
# Label builders
# ---------------------------------------------------------------------------

def extract_solweig_label(
    tmrt_path: Optional[Path], spec: PatchSpec, *, h: int, w: int,
) -> np.ndarray:
    """Crop a SOLWEIG T_mrt raster to the patch window.

    Two patterns are supported:
      1. **Per-patch SOLWEIG output** (V2-style): the file is already
         the patch's exact (h, w) shape. Read the whole file.
      2. **Master-grid SOLWEIG output** (Phase 2): the file covers a
         larger area and we extract via spec.window.

    Returns NaN-filled (h, w) when path is missing or covers neither
    case — the trainer treats NaN cells as "no supervision here".
    """
    if tmrt_path is None or not Path(tmrt_path).exists():
        return np.full((h, w), np.nan, dtype=np.float32)
    with rasterio.open(tmrt_path) as ds:
        if ds.shape == (h, w):
            # Per-patch file: read whole, normalise NoData sentinels
            arr = ds.read(1).astype(np.float32)
            if ds.nodata is not None:
                arr[arr == ds.nodata] = np.nan
            arr[arr < -9000.0] = np.nan  # SOLWEIG legacy sentinel
            return arr
    # Fallback: try as a master-grid raster
    arr, _ = _read_window(tmrt_path, spec.window, fill_nodata=np.nan)
    if arr.shape != (h, w):
        return np.full((h, w), np.nan, dtype=np.float32)
    return arr.astype(np.float32)


def extract_lst_label(
    spec: PatchSpec, bbox: Tuple[float, float, float, float], *,
    h: int, w: int, max_age_days: int = 16,
    cloud_max: float = 30.0,
) -> np.ndarray:
    """Sample Landsat LST closest in time to spec.timestamp.

    Returns (h, w) float32 with NaN where no cloud-free Landsat covers
    the patch within ±max_age_days.

    Live calls are routed through `gee_client`; if init fails (no creds
    in test env), we return all-NaN so training continues with just the
    other two labels.
    """
    try:
        from app.services.urban_heat.data_pipeline import gee_client as gc
    except ImportError:
        return np.full((h, w), np.nan, dtype=np.float32)
    try:
        gc.init()
    except (RuntimeError, FileNotFoundError) as e:
        logger.debug("GEE init unavailable, skipping LST label: %s", e)
        return np.full((h, w), np.nan, dtype=np.float32)

    from datetime import timedelta
    start = (spec.timestamp - timedelta(days=max_age_days)).strftime("%Y-%m-%d")
    end = (spec.timestamp + timedelta(days=max_age_days)).strftime("%Y-%m-%d")
    try:
        scenes = list(gc.landsat_lst_for_bbox(
            bbox, start, end, max_cloud=cloud_max,
        ))
    except Exception as e:  # pragma: no cover — GEE network errors
        logger.warning("Landsat fetch failed for %s: %s", spec.patch_id, e)
        return np.full((h, w), np.nan, dtype=np.float32)
    if not scenes:
        return np.full((h, w), np.nan, dtype=np.float32)

    # Pick scene closest in time
    best = min(
        scenes,
        key=lambda s: abs((s.timestamp - spec.timestamp).total_seconds()),
    )
    sample = gc.sample_image_at_point(
        best.image, *_patch_centroid_lonlat(bbox), scale=30,
    )
    if sample is None or "ST_celsius" not in sample:
        return np.full((h, w), np.nan, dtype=np.float32)
    # Phase 1: broadcast the centroid LST across the patch (Landsat's
    # 30 m resolution is much coarser than our 1 m grid; per-pixel
    # bilinear resample lands in Phase 2).
    return np.full((h, w), float(sample["ST_celsius"]), dtype=np.float32)


def extract_sensor_sparse_label(
    spec: PatchSpec, bbox: Tuple[float, float, float, float], *,
    sensor_df, raster_paths: Dict[str, Path], h: int, w: int,
) -> Tuple[np.ndarray, int]:
    """Build a sparse Tair label: NaN everywhere except where sensors fall.

    Returns ((h, w) array, num_sensors_in_patch). The array carries the
    sensor's air-temperature reading at the patch pixel under the sensor
    location, NaN elsewhere.
    """
    label = np.full((h, w), np.nan, dtype=np.float32)
    if sensor_df is None or len(sensor_df) == 0:
        return label, 0

    lon_min, lat_min, lon_max, lat_max = bbox
    # Pre-filter sensors to bbox + temporal window (cheap)
    from datetime import timedelta
    t_lo = spec.timestamp - timedelta(minutes=SENSOR_MAX_TIME_MIN)
    t_hi = spec.timestamp + timedelta(minutes=SENSOR_MAX_TIME_MIN)
    sub = sensor_df[
        (sensor_df["received_at"] >= t_lo)
        & (sensor_df["received_at"] <= t_hi)
        & (sensor_df["lon"] >= lon_min) & (sensor_df["lon"] <= lon_max)
        & (sensor_df["lat"] >= lat_min) & (sensor_df["lat"] <= lat_max)
        & sensor_df["airtemperature"].notna()
    ]
    if sub.empty:
        return label, 0

    # Map sensor lon/lat to patch pixel via the raster transform
    with rasterio.open(raster_paths["dsm_corrected"]) as ds:
        from rasterio.warp import transform as warp_transform
        for _, row in sub.iterrows():
            xs, ys = warp_transform(
                "EPSG:4326", ds.crs, [float(row["lon"])], [float(row["lat"])],
            )
            col, rr = ~ds.transform * (xs[0], ys[0])
            # Convert to patch-local pixel coords
            col_local = int(round(col)) - int(spec.window.col_off)
            row_local = int(round(rr)) - int(spec.window.row_off)
            if 0 <= row_local < h and 0 <= col_local < w:
                label[row_local, col_local] = float(row["airtemperature"])
    return label, int(np.isfinite(label).sum())


# ---------------------------------------------------------------------------
# Compose
# ---------------------------------------------------------------------------

def build_patch(
    spec: PatchSpec,
    *,
    raster_paths: Dict[str, Path],
    era5_tair_c: float,
    sensor_df=None,
    tmrt_path: Optional[Path] = None,
    fetch_lst: bool = True,
) -> Tuple[Dict[str, np.ndarray], Dict]:
    """Build one full patch: input tensor + 3 labels + manifest row.

    Returns (npz_arrays, manifest_row).
    """
    tensor, (cen_lon, cen_lat) = extract_input_tensor(
        spec, raster_paths=raster_paths, era5_tair_c=era5_tair_c,
    )
    _, _, h, w = (1, 1, tensor.shape[1], tensor.shape[2])

    # Recompute bbox for the labels (cheap, single open)
    with rasterio.open(raster_paths["dsm_corrected"]) as ds:
        bounds = rasterio.windows.bounds(spec.window, ds.transform)
        from rasterio.warp import transform as warp_transform
        lon, lat = warp_transform(
            ds.crs, "EPSG:4326",
            [bounds[0], bounds[2]], [bounds[1], bounds[3]],
        )
        bbox = (min(lon), min(lat), max(lon), max(lat))

    tmrt = extract_solweig_label(tmrt_path, spec, h=h, w=w)
    lst = (
        extract_lst_label(spec, bbox, h=h, w=w)
        if fetch_lst else np.full((h, w), np.nan, dtype=np.float32)
    )
    sensor_sparse, num_sensors = extract_sensor_sparse_label(
        spec, bbox, sensor_df=sensor_df, raster_paths=raster_paths,
        h=h, w=w,
    )

    has_tmrt = bool(np.isfinite(tmrt).any())
    has_lst = bool(np.isfinite(lst).any())
    sensor_mean = (
        float(np.nanmean(sensor_sparse))
        if num_sensors > 0 else float("nan")
    )

    npz = {"input": tensor, "tmrt": tmrt, "lst": lst, "tair_sparse": sensor_sparse}
    manifest = {
        "patch_id": spec.patch_id,
        "window_col": int(spec.window.col_off),
        "window_row": int(spec.window.row_off),
        "patch_w": w, "patch_h": h,
        "timestamp_utc": spec.timestamp.astimezone(timezone.utc).isoformat(),
        "doy": spec.timestamp.timetuple().tm_yday,
        "hour": spec.timestamp.hour,
        "centroid_lon": cen_lon, "centroid_lat": cen_lat,
        "era5_tair_c": float(era5_tair_c),
        "has_tmrt": has_tmrt,
        "has_lst": has_lst,
        "num_sensors_in_patch": num_sensors,
        "sensor_tair_mean_c": sensor_mean,
        **spec.meta,
    }
    return npz, manifest


def write_patch_npz(npz_arrays: Dict[str, np.ndarray], out_path: Path) -> Path:
    """Save a built patch to disk as compressed .npz."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **npz_arrays)
    return out_path


# ===========================================================================
# V4 — 20-channel multi-task model with full ERA5 + NDVI + multi-shadow
# ===========================================================================
#
# V4 = V3's 12 channels + 11 new (slots 12-22):
#   12 era5_rh_pct              — broadcast scalar
#   13 era5_wind_speed_ms       — broadcast scalar
#   14 era5_ssrd_wm2            — broadcast scalar (global horizontal solar
#                                  irradiance, hour-delta of accumulated SSRD)
#   15 ndvi                     — Sentinel-2 median NDVI raster
#   16 geoscape_solar_panel     — binary mask raster
#   17 shadow_h09               — multi-time-of-day shadow (Jan 22 9 AM)
#   18 shadow_h13               — Jan 22 1 PM
#   19 shadow_h16               — Jan 22 4 PM
#   20 roof_type_flat           — Geoscape one-hot band 0 (binary)
#   21 roof_type_moderate       — Geoscape one-hot band 1 (binary)
#   22 roof_type_steep          — Geoscape one-hot band 2 (binary)
#
# Labels (3 heads, NaN-aware loss, same as V3 but LST is now REAL Landsat
# values from per-patch parquet instead of all-NaN):
#   tmrt        — SOLWEIG synthetic (dense)
#   lst         — Landsat per-patch broadcast value (not NaN — real signal)
#   tair_sparse — sensor air temperature (sparse cells)
#
# Backward-compat: V3 functions (extract_input_tensor, build_patch) are
# unchanged and remain callable.

V4_CHANNEL_ORDER = (
    # V3's 12 channels (slots 0-11) — verbatim
    "dsm_corrected", "cdsm", "albedo",
    "sin_alt", "alt_norm",
    "svf", "building_density", "tree_distance",
    "aspect_sin", "aspect_cos", "shadow_fraction",
    "era5_tair_c",
    # V4 additions (slots 12-22)
    "era5_rh_pct",
    "era5_wind_speed_ms",
    "era5_ssrd_wm2",
    "ndvi",
    "geoscape_solar_panel_mask",
    "shadow_h09",
    "shadow_h13",
    "shadow_h16",
    "roof_type_flat",
    "roof_type_moderate",
    "roof_type_steep",
)
NUM_CHANNELS_V4 = len(V4_CHANNEL_ORDER)
assert NUM_CHANNELS_V4 == 23, "V4 must be exactly 23 channels"


def extract_input_tensor_v4(
    spec: PatchSpec,
    *,
    raster_paths: Dict[str, Path],
    era5_dict: Dict[str, float],
    v4_raster_paths: Dict[str, Path],
    utc_offset_hours: int = 10,
) -> Tuple[np.ndarray, Tuple[float, float]]:
    """Build the (20, H, W) V4 input tensor for one patch.

    raster_paths: same as V3 (dsm_corrected, cdsm, dem, albedo).
    era5_dict: must contain {tair_c, rh_pct, wind_speed_ms, ssrd_wm2}.
    v4_raster_paths: must contain {ndvi, solar_panel_mask, shadow_h09,
        shadow_h13, shadow_h16} — the V4 prep outputs.
    """
    required_v4 = {
        "ndvi", "solar_panel_mask",
        "shadow_h09", "shadow_h13", "shadow_h16",
        "roof_type",  # 3-band raster: Flat / Moderate / Steep one-hot
    }
    missing = required_v4 - set(v4_raster_paths)
    if missing:
        raise ValueError(f"v4_raster_paths missing required keys: {missing}")

    # Build V3's 12-channel tensor first (no duplication of the V3 logic)
    v3_tensor, (cen_lon, cen_lat) = extract_input_tensor(
        spec, raster_paths=raster_paths,
        era5_tair_c=float(era5_dict["tair_c"]),
        utc_offset_hours=utc_offset_hours,
    )
    h, w = v3_tensor.shape[1], v3_tensor.shape[2]

    # ERA5 broadcast scalars (channels 12-14)
    rh = float(era5_dict.get("rh_pct") or 0.0)
    wind = float(era5_dict.get("wind_speed_ms") or 0.0)
    ssrd = float(era5_dict.get("ssrd_wm2") or 0.0)
    rh_plane = np.full((h, w), rh, dtype=np.float32)
    wind_plane = np.full((h, w), wind, dtype=np.float32)
    ssrd_plane = np.full((h, w), ssrd, dtype=np.float32)

    # Spatial rasters (channels 15-19) — read window from each
    ndvi, _ = _read_window(v4_raster_paths["ndvi"], spec.window, fill_nodata=0.0)
    solar_panel, _ = _read_window(
        v4_raster_paths["solar_panel_mask"], spec.window, fill_nodata=0.0,
    )
    shadow_09, _ = _read_window(v4_raster_paths["shadow_h09"], spec.window, fill_nodata=0.0)
    shadow_13, _ = _read_window(v4_raster_paths["shadow_h13"], spec.window, fill_nodata=0.0)
    shadow_16, _ = _read_window(v4_raster_paths["shadow_h16"], spec.window, fill_nodata=0.0)

    # Roof-type one-hot — 3 bands (Flat / Moderate / Steep) read at once
    with rasterio.open(v4_raster_paths["roof_type"]) as ds:
        rt_arr = ds.read(window=spec.window).astype(np.float32)
        # Replace nodata if declared
        if ds.nodata is not None:
            rt_arr[rt_arr == ds.nodata] = 0.0
    if rt_arr.shape != (3, h, w):
        raise ValueError(
            f"roof_type raster window shape {rt_arr.shape} != expected (3, {h}, {w})",
        )
    rt_flat, rt_mod, rt_steep = rt_arr[0], rt_arr[1], rt_arr[2]

    if not all(arr.shape == (h, w) for arr in (ndvi, solar_panel, shadow_09, shadow_13, shadow_16)):
        raise ValueError(
            "V4 raster windows produced mismatched shapes vs V3 base tensor",
        )

    extras = np.stack([
        rh_plane, wind_plane, ssrd_plane,
        ndvi, solar_panel,
        shadow_09, shadow_13, shadow_16,
        rt_flat, rt_mod, rt_steep,
    ], axis=0).astype(np.float32)

    full_tensor = np.concatenate([v3_tensor, extras], axis=0)
    if full_tensor.shape[0] != NUM_CHANNELS_V4:
        raise RuntimeError(
            f"V4 channel count mismatch: got {full_tensor.shape[0]}, "
            f"want {NUM_CHANNELS_V4}",
        )
    return full_tensor, (cen_lon, cen_lat)


def extract_lst_label_from_value(
    lst_celsius: Optional[float], h: int, w: int,
) -> np.ndarray:
    """Broadcast a single Landsat LST value (from per-patch parquet) across the patch.

    For V4: real Landsat LST is sampled at patch centroid (Landsat is
    30 m vs our 1 m grid; per-pixel resample is a Phase-5 idea), so the
    entire patch gets the same value.

    Returns NaN-filled (h, w) when lst_celsius is None — the trainer's
    NaN-aware loss skips this head when there's no signal.
    """
    if lst_celsius is None or not np.isfinite(lst_celsius):
        return np.full((h, w), np.nan, dtype=np.float32)
    return np.full((h, w), float(lst_celsius), dtype=np.float32)


def build_patch_v4(
    spec: PatchSpec,
    *,
    raster_paths: Dict[str, Path],
    era5_dict: Dict[str, float],
    v4_raster_paths: Dict[str, Path],
    sensor_df=None,
    tmrt_path: Optional[Path] = None,
    lst_celsius: Optional[float] = None,
) -> Tuple[Dict[str, np.ndarray], Dict]:
    """Build one V4 patch: 20-channel input + 3 labels + manifest row."""
    tensor, (cen_lon, cen_lat) = extract_input_tensor_v4(
        spec, raster_paths=raster_paths,
        era5_dict=era5_dict, v4_raster_paths=v4_raster_paths,
    )
    h, w = tensor.shape[1], tensor.shape[2]

    # Compute bbox for sensor sampling (same approach as V3)
    with rasterio.open(raster_paths["dsm_corrected"]) as ds:
        bounds = rasterio.windows.bounds(spec.window, ds.transform)
        from rasterio.warp import transform as warp_transform
        lon, lat = warp_transform(
            ds.crs, "EPSG:4326",
            [bounds[0], bounds[2]], [bounds[1], bounds[3]],
        )
        bbox = (min(lon), min(lat), max(lon), max(lat))

    tmrt = extract_solweig_label(tmrt_path, spec, h=h, w=w)
    lst = extract_lst_label_from_value(lst_celsius, h, w)
    sensor_sparse, num_sensors = extract_sensor_sparse_label(
        spec, bbox, sensor_df=sensor_df, raster_paths=raster_paths,
        h=h, w=w,
    )

    has_tmrt = bool(np.isfinite(tmrt).any())
    has_lst = bool(np.isfinite(lst).any())
    sensor_mean = (
        float(np.nanmean(sensor_sparse))
        if num_sensors > 0 else float("nan")
    )

    npz = {"input": tensor, "tmrt": tmrt, "lst": lst, "tair_sparse": sensor_sparse}
    manifest = {
        "patch_id": spec.patch_id,
        "window_col": int(spec.window.col_off),
        "window_row": int(spec.window.row_off),
        "patch_w": w, "patch_h": h,
        "timestamp_utc": spec.timestamp.astimezone(timezone.utc).isoformat(),
        "doy": spec.timestamp.timetuple().tm_yday,
        "hour": spec.timestamp.hour,
        "centroid_lon": cen_lon, "centroid_lat": cen_lat,
        # ERA5 4-tuple (V4)
        "era5_tair_c": float(era5_dict["tair_c"]),
        "era5_rh_pct": float(era5_dict.get("rh_pct") or 0.0),
        "era5_wind_speed_ms": float(era5_dict.get("wind_speed_ms") or 0.0),
        "era5_ssrd_wm2": float(era5_dict.get("ssrd_wm2") or 0.0),
        # Real Landsat LST per patch
        "lst_celsius": float(lst_celsius) if lst_celsius is not None else None,
        # Label availability flags
        "has_tmrt": has_tmrt,
        "has_lst": has_lst,
        "num_sensors_in_patch": num_sensors,
        "sensor_tair_mean_c": sensor_mean,
        **spec.meta,
    }
    return npz, manifest
