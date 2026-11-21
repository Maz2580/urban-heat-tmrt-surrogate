"""Plan UH-1 / A2 — Google Earth Engine service-account client.

Server-side GEE access for Phase-1 data preparation. Uses the service
account credentials configured in `env/app.env` (no browser popup —
the Venice notebook's Colab-popup pattern is for runtime only).

Three datasets exposed for the V3 training pipeline:

  - Landsat 8/9 surface temperature (ST_B10 → °C)
      auxiliary training label
  - ECMWF ERA5-Land hourly meteorology (Tair / RH / wind / DNI)
      replaces the sparse 2-station EPW in model input
  - Sentinel-2 SR (B8 / B4 → NDVI)
      time-aware vegetation channel

The module is import-safe even if `ee` is not installed; everything
hot-imports inside functions so unit tests on adjacent code don't
require the Earth Engine package.

Example usage:
  from app.services.urban_heat.data_pipeline.gee_client import (
      init, landsat_lst_for_bbox, era5_hourly_at_point, sentinel2_ndvi_for_bbox,
  )
  init()
  scenes = list(landsat_lst_for_bbox(
      MELB_FB_BBOX, "2024-01-01", "2024-12-31", max_cloud=20
  ))
  print(f"{len(scenes)} cloud-filtered Landsat scenes for FB 2024")
"""
from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Iterator, Optional

logger = logging.getLogger(__name__)

# Fishermans Bend covers a ~9 km × 9 km box around -37.83° / 144.93°.
# Slightly padded so the model's training grid sits inside.
MELB_FB_BBOX = (144.880, -37.860, 144.985, -37.805)  # (W, S, E, N)

# Landsat L2 surface temperature: ST_B10 is scaled int16; convert to
# Kelvin via `K = pixel × 0.00341802 + 149.0`, then to °C.
LANDSAT_ST_SCALE = 0.00341802
LANDSAT_ST_OFFSET = 149.0  # Kelvin

# QA_PIXEL bit positions (Landsat Collection 2 L2 spec)
_QA_BIT_DILATED_CLOUD = 1
_QA_BIT_CIRRUS = 2
_QA_BIT_CLOUD = 3
_QA_BIT_CLOUD_SHADOW = 4
_QA_BIT_SNOW = 5

# Sentinel-2 SCL classes to mask out (cloud + shadow + snow).
_S2_BAD_SCL_CLASSES = (3, 8, 9, 10, 11)  # cloud_shadows, cloud_med, cloud_high, cirrus, snow

# Module-level "have we initialised" flag. ee.Initialize() is idempotent
# but doing it twice still hits the auth machinery; cheap to gate.
_initialised: bool = False


@dataclass
class LandsatScene:
    """One Landsat 8/9 surface-temperature scene."""
    image_id: str           # e.g. 'LANDSAT/LC08/C02/T1_L2/LC08_093086_20240115'
    sensor: str             # 'LC08' or 'LC09'
    timestamp: datetime     # scene acquisition timestamp (UTC)
    cloud_cover: float      # CLOUD_COVER property, %
    image: object           # ee.Image (LST band only, in °C, cloud-masked)


def init(
    *,
    service_account: Optional[str] = None,
    key_path: Optional[str] = None,
    project: Optional[str] = None,
    force: bool = False,
) -> None:
    """Initialise the Earth Engine client with a service account.

    Reads from environment by default:
      - GEE_SERVICE_ACCOUNT
      - GOOGLE_APPLICATION_CREDENTIALS  (path to JSON key file)
      - GCP_PROJECT_ID

    Idempotent — call freely; subsequent calls are a no-op unless
    `force=True`.

    Browser-popup auth is intentionally NOT supported here — that's
    a Colab-runtime pattern. On the server we always use the
    service-account JSON.
    """
    global _initialised
    if _initialised and not force:
        return

    import ee

    sa = service_account or os.environ.get("GEE_SERVICE_ACCOUNT")
    key = key_path or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    proj = project or os.environ.get("GCP_PROJECT_ID")
    if not sa or not key or not proj:
        raise RuntimeError(
            "GEE init: GEE_SERVICE_ACCOUNT + GOOGLE_APPLICATION_CREDENTIALS "
            "+ GCP_PROJECT_ID must all be set (in env/app.env or kwargs)",
        )
    if not Path(key).exists():
        raise FileNotFoundError(
            f"GEE service-account JSON not found: {key}",
        )

    creds = ee.ServiceAccountCredentials(sa, key)
    ee.Initialize(credentials=creds, project=proj)
    _initialised = True
    logger.info("Earth Engine initialised (project=%s, service_account=%s)",
                proj, sa)


# -------------------------------------------------------------------- #
# Landsat 8/9 surface temperature
# -------------------------------------------------------------------- #


def _landsat_st_celsius(image, qa_band: str = "QA_PIXEL"):
    """Convert raw ST_B10 to °C and apply the Collection-2 cloud mask."""
    import ee
    st_kelvin = image.select("ST_B10").multiply(LANDSAT_ST_SCALE).add(LANDSAT_ST_OFFSET)
    st_celsius = st_kelvin.subtract(273.15).rename("LST_C")
    qa = image.select(qa_band)
    bad = (
        qa.bitwiseAnd(1 << _QA_BIT_DILATED_CLOUD).neq(0)
        .Or(qa.bitwiseAnd(1 << _QA_BIT_CIRRUS).neq(0))
        .Or(qa.bitwiseAnd(1 << _QA_BIT_CLOUD).neq(0))
        .Or(qa.bitwiseAnd(1 << _QA_BIT_CLOUD_SHADOW).neq(0))
        .Or(qa.bitwiseAnd(1 << _QA_BIT_SNOW).neq(0))
    )
    return st_celsius.updateMask(bad.Not())


def landsat_lst_for_bbox(
    bbox: tuple[float, float, float, float],
    start: str | datetime,
    end: str | datetime,
    *,
    max_cloud: float = 30.0,
) -> Iterator[LandsatScene]:
    """Yield cloud-filtered Landsat 8/9 LST scenes intersecting `bbox`
    in the inclusive date range [start, end].

    bbox: (west, south, east, north) in EPSG:4326 decimal degrees.
    start, end: ISO-8601 strings ('YYYY-MM-DD') or datetime objects.
    max_cloud: scenes with CLOUD_COVER above this % are skipped.

    Each yielded `LandsatScene` carries an `ee.Image` whose only band
    is `LST_C` (°C, cloud-masked). The image is lazy — no GEE compute
    runs until you `.sample()` / `.reduceRegion()` / etc.
    """
    if not _initialised:
        init()
    import ee

    if isinstance(start, datetime):
        start = start.strftime("%Y-%m-%d")
    if isinstance(end, datetime):
        end = end.strftime("%Y-%m-%d")
    geom = ee.Geometry.Rectangle(list(bbox))

    collections = [
        ("LC08", "LANDSAT/LC08/C02/T1_L2"),
        ("LC09", "LANDSAT/LC09/C02/T1_L2"),
    ]
    scenes: list[LandsatScene] = []
    for sensor, coll_id in collections:
        coll = (
            ee.ImageCollection(coll_id)
            .filterBounds(geom)
            .filterDate(start, end)
            .filter(ee.Filter.lt("CLOUD_COVER", max_cloud))
        )
        # Force materialisation of properties — cheap (only metadata).
        info = coll.getInfo()
        for feat in info.get("features", []) or []:
            props = feat.get("properties", {}) or {}
            ts_ms = props.get("system:time_start")
            ts = (
                datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
                if ts_ms is not None else None
            )
            img_id = feat.get("id")
            scenes.append(LandsatScene(
                image_id=img_id, sensor=sensor, timestamp=ts,
                cloud_cover=float(props.get("CLOUD_COVER", 0.0)),
                image=_landsat_st_celsius(ee.Image(img_id)),
            ))
    # Yield sorted by acquisition timestamp so consumers can iterate
    # naturally across the calendar.
    scenes.sort(key=lambda s: s.timestamp or datetime.min.replace(tzinfo=timezone.utc))
    yield from scenes


# -------------------------------------------------------------------- #
# ERA5-Land hourly meteorology
# -------------------------------------------------------------------- #


def _kelvin_to_c(k):
    return k - 273.15


def _rh_from_dewpoint(temp_c, dew_c):
    """Magnus formula — relative humidity % from Tair and Tdew (both °C)."""
    a, b = 17.625, 243.04
    es = math.exp((a * temp_c) / (b + temp_c))
    e = math.exp((a * dew_c) / (b + dew_c))
    return max(0.0, min(100.0, 100.0 * e / es))


def era5_hourly_at_point(
    lon: float, lat: float, ts: datetime | str,
    *,
    bracket_minutes: int = 30,
) -> dict:
    """Sample ERA5-Land hourly meteo at one point + timestamp.

    Returns a dict:
      {
        'tair_c':   float,      # 2 m air temperature, °C
        'rh_pct':   float,      # derived from Tair + Tdew via Magnus
        'wind_speed_ms': float, # sqrt(u² + v²) at 10 m
        'wind_dir_deg':  float, # bearing the wind is FROM, 0=N, 90=E
        'ssrd_wm2': float,      # global horizontal solar irradiance, W/m²
                                # — computed as (SSRD[h] - SSRD[h-1]) / 3600
                                # because ERA5_LAND SSRD is accumulated
                                # since 00:00 UTC of each day
        'image_ts': str,        # ISO timestamp of the matched ERA5 hour
      }

    bracket_minutes: half-window around `ts` to find the matching ERA5
    hourly image. ERA5-Land has hourly cadence, so 30 min is enough.
    """
    if not _initialised:
        init()
    import ee

    if isinstance(ts, str):
        ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)

    pt = ee.Geometry.Point([lon, lat])

    def _fetch(ts_target: datetime):
        lo = ee.Date(ts_target.timestamp() * 1000 - bracket_minutes * 60_000)
        hi = ee.Date(ts_target.timestamp() * 1000 + bracket_minutes * 60_000)
        coll = (
            ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY")
            .filterDate(lo, hi).filterBounds(pt)
        )
        if coll.size().getInfo() == 0:
            return None, None
        img = ee.Image(coll.first())
        sample = img.select([
            "temperature_2m", "dewpoint_temperature_2m",
            "u_component_of_wind_10m", "v_component_of_wind_10m",
            "surface_solar_radiation_downwards",
        ]).reduceRegion(
            reducer=ee.Reducer.first(), geometry=pt, scale=11132,
        ).getInfo()
        return img, sample

    img, sample = _fetch(ts)
    if not sample:
        return {}

    # SSRD is accumulated J/m² since 00:00 UTC, reset daily. To get the
    # average W/m² for the hour ENDING at ts, fetch the previous hour and
    # take the delta. If ts is at hour 0 (or prev image not found), the
    # accumulated value at ts == in-hour energy already.
    ssrd_now = sample.get("surface_solar_radiation_downwards") or 0.0
    ts_prev = ts - timedelta(hours=1)
    ssrd_wm2 = ssrd_now / 3600.0
    if ts.hour > 0:
        _, sample_prev = _fetch(ts_prev)
        if sample_prev:
            ssrd_prev = sample_prev.get("surface_solar_radiation_downwards") or 0.0
            delta = ssrd_now - ssrd_prev
            if delta >= 0:
                ssrd_wm2 = delta / 3600.0
            # else: midnight reset happened between H-1 and H — fallback to ssrd_now/3600

    t_k = sample.get("temperature_2m")
    td_k = sample.get("dewpoint_temperature_2m")
    u = sample.get("u_component_of_wind_10m") or 0.0
    v = sample.get("v_component_of_wind_10m") or 0.0

    tair_c = _kelvin_to_c(t_k) if t_k is not None else None
    tdew_c = _kelvin_to_c(td_k) if td_k is not None else None
    rh = (
        _rh_from_dewpoint(tair_c, tdew_c)
        if tair_c is not None and tdew_c is not None else None
    )
    wind_speed = math.sqrt(u * u + v * v)
    wind_dir = (math.degrees(math.atan2(-u, -v)) + 360.0) % 360.0

    return {
        "tair_c": round(tair_c, 2) if tair_c is not None else None,
        "rh_pct": round(rh, 2) if rh is not None else None,
        "wind_speed_ms": round(wind_speed, 2),
        "wind_dir_deg": round(wind_dir, 1),
        "ssrd_wm2": round(ssrd_wm2, 1),
        "image_ts": img.get("system:index").getInfo(),
    }


# -------------------------------------------------------------------- #
# Sentinel-2 NDVI
# -------------------------------------------------------------------- #


def _s2_mask_clouds(image):
    """Mask Sentinel-2 SCL bad classes (cloud, shadow, cirrus, snow)."""
    import ee
    scl = image.select("SCL")
    mask = ee.Image(1)
    for cls in _S2_BAD_SCL_CLASSES:
        mask = mask.And(scl.neq(cls))
    return image.updateMask(mask)


def sentinel2_ndvi_for_bbox(
    bbox: tuple[float, float, float, float],
    around_ts: datetime | str,
    *,
    window_days: int = 15,
    max_cloud: float = 30.0,
):
    """Median NDVI image for a bbox in a window centred on `around_ts`.

    Returns an `ee.Image` with single band `NDVI`. Lazy — caller must
    `.sample()` / `.reduceRegion()` / `.sampleRectangle()` to fetch.
    Returns None if no scenes in window.
    """
    if not _initialised:
        init()
    import ee

    if isinstance(around_ts, str):
        around_ts = datetime.fromisoformat(around_ts.replace("Z", "+00:00"))
    if around_ts.tzinfo is None:
        around_ts = around_ts.replace(tzinfo=timezone.utc)
    half_ms = window_days * 12 * 3600_000
    centre_ms = around_ts.timestamp() * 1000

    coll = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(ee.Date(centre_ms - half_ms), ee.Date(centre_ms + half_ms))
        .filterBounds(ee.Geometry.Rectangle(list(bbox)))
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud))
        .map(_s2_mask_clouds)
    )
    if coll.size().getInfo() == 0:
        return None
    median = coll.median()
    nir = median.select("B8")
    red = median.select("B4")
    ndvi = nir.subtract(red).divide(nir.add(red)).rename("NDVI")
    return ndvi


# -------------------------------------------------------------------- #
# Convenience: per-patch sampler used by patch_builder (Phase 1 / A4)
# -------------------------------------------------------------------- #


def sample_image_at_point(
    image, lon: float, lat: float, *, scale_m: float = 30.0,
) -> Optional[float]:
    """Single-value `reduceRegion(first)` sample. Convenience wrapper
    used by the patch builder when assembling per-pixel labels."""
    if not _initialised:
        init()
    import ee
    pt = ee.Geometry.Point([lon, lat])
    out = image.reduceRegion(
        reducer=ee.Reducer.first(), geometry=pt, scale=scale_m,
    ).getInfo() or {}
    if not out:
        return None
    # Pick whichever band the caller passed
    for v in out.values():
        if v is not None:
            return float(v)
    return None
