"""
Urban Heat Solar Position Service (Phase 1.6.4)
================================================

Exact replica of the training notebook's solar position calculator.
Must match Cell 3 of UMAMI_UNet_Training.ipynb to produce identical
model input channels.

Critical: The `+ lon/15.0` solar time correction is essential —
without it, the model receives wrong solar angles and produces
incorrect T_mrt predictions.
"""

import math
from typing import Tuple


def compute_solar_position(
    lat: float,
    lon: float,
    doy: int,
    hour: int,
    utc_offset: int = 10,
) -> Tuple[float, float]:
    """
    Compute solar altitude and azimuth for a given location and time.

    This is a verbatim copy of the training notebook function.
    EPW files use AEST (UTC+10), so utc_offset=10 is the default.

    Args:
        lat: Latitude in degrees (negative for southern hemisphere)
        lon: Longitude in degrees
        doy: Day of year (1-366)
        hour: Hour of day in local time (0-23)
        utc_offset: UTC offset for local time (default 10 for AEST)

    Returns:
        (altitude_degrees, azimuth_degrees)
    """
    solar_hour = hour - utc_offset + lon / 15.0
    dec = -23.45 * math.cos(math.radians(360 / 365 * (doy + 10)))
    ha = 15 * (solar_hour - 12)

    lat_r = math.radians(lat)
    dec_r = math.radians(dec)
    ha_r = math.radians(ha)

    sin_alt = (
        math.sin(lat_r) * math.sin(dec_r)
        + math.cos(lat_r) * math.cos(dec_r) * math.cos(ha_r)
    )
    altitude = math.degrees(math.asin(max(-1, min(1, sin_alt))))

    if math.cos(math.radians(altitude)) < 0.001:
        return altitude, 180.0

    cos_az = (math.sin(dec_r) - math.sin(lat_r) * sin_alt) / (
        math.cos(lat_r) * math.cos(math.radians(altitude))
    )
    azimuth = math.degrees(math.acos(max(-1, min(1, cos_az))))
    if ha > 0:
        azimuth = 360 - azimuth

    return altitude, azimuth


def get_solar_channels(sun_altitude: float) -> Tuple[float, float]:
    """
    Convert solar altitude to the two model input channels.

    Channel 3: sin(altitude) — 0 at horizon, 1 at zenith
    Channel 4: altitude / 90  — linear 0-1

    Args:
        sun_altitude: Solar altitude in degrees

    Returns:
        (sin_alt_value, alt_normalized_value)
    """
    alt_clamped = max(sun_altitude, 0.0)
    sin_alt = math.sin(math.radians(alt_clamped))
    alt_norm = alt_clamped / 90.0
    return sin_alt, alt_norm


def get_default_temporal_params() -> dict:
    """
    Default simulation parameters: peak summer afternoon in Melbourne.

    DOY 356 = December 22 (summer solstice, longest day)
    Hour 13 = 1 PM AEST (near peak solar radiation)
    """
    return {"doy": 356, "hour": 13}
