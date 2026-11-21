#!/usr/bin/env python3
"""
Urban Heat Tool — EPW to UMEP Meteorological Format Converter (Phase 1.6.2)

Converts EnergyPlus Weather (.epw) files to the UMEP text format required
by solweig-gpu. The UMEP format is a space-separated text file with 24 columns
of hourly meteorological data.

UMEP column order (per UMEP MetPreprocessor documentation):
  iy id it imin qn qh qe qs qf U RH Tair pres rain kdown snow ldown fcld wuh xsmd lai kdiff kdir wdir

EPW data rows start at line 9 (0-indexed: line 8), comma-delimited.
EPW column indices (from ASHRAE Standard 91010):
  0=Year, 1=Month, 2=Day, 3=Hour, 4=Minute,
  6=DryBulbTemp, 8=RelHumidity, 9=AtmPressure(Pa),
  12=HorizInfraredRad, 13=GlobalHorizRad, 14=DirectNormalRad,
  15=DiffuseHorizRad, 20=WindDirection, 21=WindSpeed,
  22=TotalSkyCover(0-10), 33=LiquidPrecipDepth

Input:  urban_heat_data/weather/*.epw
Output: urban_heat_data/weather/*_umep.txt

Usage:
  python convert_epw_to_umep.py
  python convert_epw_to_umep.py --input /data/weather/specific_file.epw
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Paths (container mount points)
DATA_ROOT = Path("/data")
WEATHER_DIR = DATA_ROOT / "weather"

# Fallback for running outside container
if not DATA_ROOT.exists():
    DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "urban_heat_data"
    WEATHER_DIR = DATA_ROOT / "weather"

# UMEP column header
UMEP_HEADER = "iy id it imin qn qh qe qs qf U RH Tair pres rain kdown snow ldown fcld wuh xsmd lai kdiff kdir wdir"
MISSING = -999.0
EPW_HEADER_LINES = 8  # EPW has 8 header lines before data


def day_of_year(year: int, month: int, day: int) -> int:
    """Convert year/month/day to day of year (1-366)."""
    return datetime(year, month, day).timetuple().tm_yday


def convert_epw_to_umep(epw_path: Path, output_path: Path) -> int:
    """
    Convert a single EPW file to UMEP text format.

    Returns the number of hourly records written.
    """
    logger.info(f"Reading EPW: {epw_path.name}")

    rows = []
    with open(epw_path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i < EPW_HEADER_LINES:
                continue
            parts = line.strip().split(",")
            if len(parts) < 34:
                continue

            try:
                year = int(parts[0])
                month = int(parts[1])
                day = int(parts[2])
                hour = int(parts[3])
                minute = int(parts[4])
                doy = day_of_year(year, month, day)

                # EPW hour convention: 1-24 (hour ending)
                # UMEP convention: 0-23 (hour beginning)
                # EPW hour 1 = data for 00:00-01:00, which in UMEP is hour 0
                umep_hour = hour - 1 if hour > 0 else 23

                # Meteorological variables from EPW
                tair = float(parts[6])            # Dry bulb temp (C)
                rh = float(parts[8])              # Relative humidity (%)
                pres_pa = float(parts[9])          # Atmospheric pressure (Pa)
                pres_kpa = pres_pa / 1000.0        # Convert to kPa for UMEP

                ldown = float(parts[12])           # Horiz infrared radiation (W/m2)
                kdown = float(parts[13])           # Global horizontal radiation (W/m2)
                kdir = float(parts[14])            # Direct normal radiation (W/m2)
                kdiff = float(parts[15])           # Diffuse horizontal radiation (W/m2)

                wdir = float(parts[20])            # Wind direction (degrees)
                wspd = float(parts[21])            # Wind speed (m/s)
                cloud = float(parts[22])           # Total sky cover (0-10 tenths)

                # Liquid precipitation depth (mm) — column 33
                rain_str = parts[33].strip() if len(parts) > 33 else "0"
                rain = float(rain_str) if rain_str and rain_str != "999" else 0.0

                # Build UMEP row (24 columns)
                row = [
                    year,           # iy
                    doy,            # id (day of year)
                    umep_hour,      # it (hour 0-23)
                    minute,         # imin
                    MISSING,        # qn (net radiation — not available)
                    MISSING,        # qh (sensible heat flux)
                    MISSING,        # qe (latent heat flux)
                    MISSING,        # qs (storage heat flux)
                    MISSING,        # qf (anthropogenic heat flux)
                    wspd,           # U (wind speed m/s)
                    rh,             # RH (relative humidity %)
                    tair,           # Tair (air temperature C)
                    pres_kpa,       # pres (pressure kPa)
                    rain,           # rain (mm)
                    kdown,          # kdown (global shortwave W/m2)
                    MISSING,        # snow (not available)
                    ldown,          # ldown (longwave down W/m2)
                    cloud,          # fcld (cloud fraction tenths)
                    MISSING,        # wuh (external water use)
                    MISSING,        # xsmd (soil moisture)
                    MISSING,        # lai (leaf area index)
                    kdiff,          # kdiff (diffuse shortwave W/m2)
                    kdir,           # kdir (direct shortwave W/m2)
                    wdir,           # wdir (wind direction degrees)
                ]
                rows.append(row)

            except (ValueError, IndexError) as e:
                logger.warning(f"Skipping EPW line {i}: {e}")
                continue

    if not rows:
        logger.error(f"No valid data rows found in {epw_path.name}")
        return 0

    # Write UMEP file
    logger.info(f"Writing UMEP file: {output_path.name} ({len(rows)} hourly records)")
    with open(output_path, "w") as f:
        f.write(UMEP_HEADER + "\n")
        for row in rows:
            formatted = []
            for val in row:
                if isinstance(val, int):
                    formatted.append(str(val))
                elif val == MISSING:
                    formatted.append("-999")
                else:
                    formatted.append(f"{val:.2f}")
            f.write(" ".join(formatted) + "\n")

    logger.info(f"Done: {len(rows)} records, date range DOY {rows[0][1]}-{rows[-1][1]}")
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description="Convert EPW to UMEP met format")
    parser.add_argument("--input", type=str, help="Specific EPW file to convert")
    args = parser.parse_args()

    if args.input:
        epw_files = [Path(args.input)]
    else:
        epw_files = sorted(WEATHER_DIR.glob("*.epw"))

    if not epw_files:
        logger.error(f"No EPW files found in {WEATHER_DIR}")
        sys.exit(1)

    logger.info(f"Found {len(epw_files)} EPW file(s) to convert")

    for epw_path in epw_files:
        # Derive output name
        stem = epw_path.stem.lower()
        if "tullamarine" in stem or "948660" in stem:
            out_name = "mel_tullamarine_umep.txt"
        elif "olympic" in stem or "959360" in stem:
            out_name = "mel_olympic_park_umep.txt"
        else:
            out_name = epw_path.stem + "_umep.txt"

        output_path = WEATHER_DIR / out_name
        count = convert_epw_to_umep(epw_path, output_path)
        if count > 0:
            logger.info(f"SUCCESS: {epw_path.name} -> {out_name} ({count} rows)")
        else:
            logger.error(f"FAILED: {epw_path.name}")

    logger.info("All conversions complete.")


if __name__ == "__main__":
    main()
