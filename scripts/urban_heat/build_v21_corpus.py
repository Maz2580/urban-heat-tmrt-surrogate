"""Plan UH-1 / B0 — V2.1 corpus driver.

Walks the V2 patch list (`urban_heat_data/training/valid_patches.json`)
and for each (patch, temporal_combo) tuple builds a 12-channel V2.1
input tensor + 3 multi-task labels via `patch_builder.build_patch`,
then saves an .npz to the output dir + appends a manifest row.

Reuses V2's geographic windows (so the train/test split stays
comparable) and V2's SOLWEIG T_mrt outputs as one of the three labels.

Usage:
    docker exec umami_test python /app/scripts/urban_heat/build_v21_corpus.py \\
        --max-patches 16 --combos d306_h13   # smoke test
    docker exec umami_test python /app/scripts/urban_heat/build_v21_corpus.py
        # full sweep — ~8,640 patches, ~15 min with 8 workers
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

# Make the package importable when run via `python -m` or directly
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.services.urban_heat.data_pipeline import patch_builder as pb  # noqa: E402

logger = logging.getLogger("umami.urban_heat.build_v21_corpus")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

V2_PATCHES_DIR = REPO_ROOT / "urban_heat_data/training/patches"
V2_VALID_JSON = REPO_ROOT / "urban_heat_data/training/valid_patches.json"
RASTER_DIR = REPO_ROOT / "urban_heat_data/rasters/raw"
SENSOR_PARQUET = REPO_ROOT / "urban_heat_data/sensors/sensors_data.parquet"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "urban_heat_data/training/patches_v21"

RASTER_PATHS = {
    "dsm_corrected": RASTER_DIR / "geoscape_heights.tif",
    "cdsm": RASTER_DIR / "cdsm_1m.tif",
    "dem": RASTER_DIR / "dem_1m.tif",
    "albedo": RASTER_DIR / "geoscape_albedo.tif",
}

# Default year for ERA5 + Landsat sampling. SOLWEIG T_mrt was generated
# from TMY data (2009-2023 typical year), so any specific year is a
# small drift; 2024 is recent + has plenty of cloud-free Landsat scenes.
DEFAULT_YEAR = 2024

# Filename pattern for V2 SOLWEIG outputs: Tmrt_d{DOY:03d}_h{HOUR:02d}.tif
TMRT_PATTERN = re.compile(r"Tmrt_d(\d{3})_h(\d{2})\.tif")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_combo(combo_str: str) -> tuple[int, int]:
    """Parse 'd306_h13' → (306, 13)."""
    m = re.match(r"d(\d{1,3})_h(\d{1,2})", combo_str)
    if not m:
        raise ValueError(f"bad combo string: {combo_str!r}")
    return int(m.group(1)), int(m.group(2))


def doy_hour_to_utc(doy: int, hour: int, year: int) -> datetime:
    """Convert (DOY, hour) AEST to a tz-aware UTC datetime.

    The V2 SOLWEIG runs used local AEST (UTC+10). For ERA5 sampling we
    need UTC, so we shift -10 h.
    """
    from datetime import timedelta
    aest = datetime(year, 1, 1, hour=hour, tzinfo=timezone.utc) \
        + timedelta(days=doy - 1)
    return aest - timedelta(hours=10)


def patch_window_in_master(patch_dsm_path: Path) -> Window:
    """Compute the V2 patch's window in the V2.1 master grid."""
    with rasterio.open(patch_dsm_path) as patch_ds:
        patch_bounds = patch_ds.bounds
        patch_h, patch_w = patch_ds.shape
    with rasterio.open(RASTER_PATHS["dsm_corrected"]) as master_ds:
        # Convert top-left of patch to master pixel coords
        col, row = ~master_ds.transform * (patch_bounds.left, patch_bounds.top)
        return Window(
            col_off=int(round(col)), row_off=int(round(row)),
            width=patch_w, height=patch_h,
        )


def list_temporal_combos(patch_id: str) -> list[tuple[int, int]]:
    """Discover which (DOY, hour) combos have a SOLWEIG output for this patch."""
    out_dir = V2_PATCHES_DIR / patch_id / "output_folder"
    if not out_dir.is_dir():
        return []
    combos = []
    for fp in out_dir.glob("Tmrt_d*_h*.tif"):
        m = TMRT_PATTERN.match(fp.name)
        if m:
            combos.append((int(m.group(1)), int(m.group(2))))
    return sorted(combos)


def load_sensor_archive() -> Optional[pd.DataFrame]:
    if not SENSOR_PARQUET.exists():
        logger.warning("sensor archive not found at %s", SENSOR_PARQUET)
        return None
    df = pd.read_parquet(SENSOR_PARQUET)
    df["received_at"] = pd.to_datetime(df["received_at"], utc=True)
    return df


# ---------------------------------------------------------------------------
# ERA5 cache — sample once per (DOY, hour) at AOI centre
# ---------------------------------------------------------------------------

_ERA5_CACHE: dict[tuple[int, int], float] = {}
# Approximate centre of FB AOI (degrees)
AOI_CENTRE_LONLAT = (144.93, -37.83)


def get_era5_tair(doy: int, hour: int, year: int) -> float:
    """Return ERA5 Tair (°C) for a temporal combo, cached per-combo."""
    key = (doy, hour)
    if key in _ERA5_CACHE:
        return _ERA5_CACHE[key]
    try:
        from app.services.urban_heat.data_pipeline import gee_client as gc
        gc.init()
        ts = doy_hour_to_utc(doy, hour, year)
        sample = gc.era5_hourly_at_point(*AOI_CENTRE_LONLAT, ts)
        if sample is None or sample.get("tair_c") is None:
            raise RuntimeError("ERA5 returned no Tair")
        val = float(sample["tair_c"])
    except Exception as e:
        # Fallback to a TMY-ish approximation by season
        logger.warning(
            "ERA5 unavailable for d%03d_h%02d (%s); using TMY fallback",
            doy, hour, e,
        )
        val = _tmy_fallback_tair(doy, hour)
    _ERA5_CACHE[key] = val
    return val


def _tmy_fallback_tair(doy: int, hour: int) -> float:
    """Crude Melbourne TMY Tair fallback when ERA5 is unavailable.

    Annual cycle: peak ~22 °C late January, trough ~10 °C late July.
    Diurnal cycle: ±5 °C around the daily mean, peak at hour 14.
    """
    annual = 16.0 + 6.0 * np.cos(2 * np.pi * (doy - 30) / 365)
    diurnal = 5.0 * np.sin(np.pi * (hour - 6) / 12)
    return float(np.clip(annual + diurnal, 0.0, 45.0))


# ---------------------------------------------------------------------------
# Single-task worker
# ---------------------------------------------------------------------------

_WORKER_SENSOR_DF: Optional[pd.DataFrame] = None


def _worker_init() -> None:
    """Load sensor_df once per worker process — avoids 13 MB
    serialised on every task call."""
    global _WORKER_SENSOR_DF
    _WORKER_SENSOR_DF = load_sensor_archive()


def build_one_task(
    patch_id: str, doy: int, hour: int, year: int,
    output_dir: Path, era5_tair_c: float, fetch_lst: bool,
    skip_existing: bool, sensor_df=None,
) -> dict:
    """Build one (patch, combo) tuple and save its .npz."""
    out_path = output_dir / f"{patch_id}_d{doy:03d}_h{hour:02d}.npz"
    if skip_existing and out_path.exists():
        return {"patch_id": patch_id, "doy": doy, "hour": hour, "skipped": True}

    if sensor_df is None:
        sensor_df = _WORKER_SENSOR_DF  # populated by _worker_init in pool

    patch_dir = V2_PATCHES_DIR / patch_id
    dsm_path = patch_dir / "building_dsm.tif"
    if not dsm_path.exists():
        return {"patch_id": patch_id, "doy": doy, "hour": hour, "missing_dsm": True}

    window = patch_window_in_master(dsm_path)
    spec = pb.PatchSpec(
        patch_id=f"{patch_id}_d{doy:03d}_h{hour:02d}",
        window=window,
        timestamp=doy_hour_to_utc(doy, hour, year),
        meta={"v2_patch_id": patch_id, "doy": doy, "hour": hour, "year": year},
    )
    tmrt_path = patch_dir / "output_folder" / f"Tmrt_d{doy:03d}_h{hour:02d}.tif"
    npz, manifest = pb.build_patch(
        spec, raster_paths=RASTER_PATHS, era5_tair_c=era5_tair_c,
        sensor_df=sensor_df, tmrt_path=tmrt_path, fetch_lst=fetch_lst,
    )
    pb.write_patch_npz(npz, out_path)
    manifest["npz_path"] = str(out_path.relative_to(REPO_ROOT))
    manifest["v2_patch_id"] = patch_id
    return manifest


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_corpus(
    *,
    patch_ids: list[str],
    combo_filter: Optional[set[tuple[int, int]]] = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    year: int = DEFAULT_YEAR,
    fetch_lst: bool = True,
    skip_existing: bool = True,
    workers: int = 1,
    manifest_every: int = 100,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    sensor_df = load_sensor_archive()
    if sensor_df is not None:
        logger.info("loaded %d sensor rows", len(sensor_df))

    # Build the (patch, combo) work list
    tasks = []
    for patch_id in patch_ids:
        for doy, hour in list_temporal_combos(patch_id):
            if combo_filter and (doy, hour) not in combo_filter:
                continue
            tasks.append((patch_id, doy, hour))
    logger.info("planned %d patch×combo tasks", len(tasks))

    # Pre-warm ERA5 cache
    needed_combos = sorted({(d, h) for _, d, h in tasks})
    logger.info("pre-fetching ERA5 Tair for %d unique combos", len(needed_combos))
    for d, h in needed_combos:
        v = get_era5_tair(d, h, year)
        logger.info("  d%03d_h%02d → Tair = %.2f °C", d, h, v)

    manifest_path = output_dir / "manifest.parquet"
    manifest_rows: list[dict] = []
    t_start = time.time()
    last_log = t_start

    if workers == 1:
        for i, (patch_id, doy, hour) in enumerate(tasks, start=1):
            era5 = _ERA5_CACHE[(doy, hour)]
            row = build_one_task(
                patch_id, doy, hour, year, output_dir, era5,
                fetch_lst, skip_existing, sensor_df,
            )
            if not row.get("skipped") and not row.get("missing_dsm"):
                manifest_rows.append(row)
            now = time.time()
            if now - last_log > 30 or i == len(tasks):
                rate = i / max(now - t_start, 0.001)
                eta = (len(tasks) - i) / max(rate, 0.001)
                logger.info(
                    "%d/%d tasks (%.1f patches/s, ~%d min ETA)",
                    i, len(tasks), rate, int(eta // 60),
                )
                last_log = now
            if i % manifest_every == 0 and manifest_rows:
                pd.DataFrame(manifest_rows).to_parquet(manifest_path)
    else:
        # Multiprocess pool — each worker re-imports patch_builder + loads
        # sensor_df once via _worker_init (avoids 13 MB serialised per task).
        era5_lookup = dict(_ERA5_CACHE)
        with ProcessPoolExecutor(
            max_workers=workers, initializer=_worker_init,
        ) as ex:
            futures = [
                ex.submit(
                    build_one_task,
                    patch_id, doy, hour, year, output_dir,
                    era5_lookup[(doy, hour)], fetch_lst, skip_existing,
                    None,  # sensor_df loaded inside worker
                )
                for patch_id, doy, hour in tasks
            ]
            for i, fut in enumerate(as_completed(futures), start=1):
                row = fut.result()
                if not row.get("skipped") and not row.get("missing_dsm"):
                    manifest_rows.append(row)
                now = time.time()
                if now - last_log > 30 or i == len(tasks):
                    rate = i / max(now - t_start, 0.001)
                    eta = (len(tasks) - i) / max(rate, 0.001)
                    logger.info(
                        "%d/%d tasks (%.1f patches/s, ~%d min ETA)",
                        i, len(tasks), rate, int(eta // 60),
                    )
                    last_log = now
                if i % manifest_every == 0 and manifest_rows:
                    pd.DataFrame(manifest_rows).to_parquet(manifest_path)

    # Final manifest
    if manifest_rows:
        df = pd.DataFrame(manifest_rows)
        df.to_parquet(manifest_path)
    else:
        df = pd.DataFrame()

    elapsed = time.time() - t_start
    logger.info(
        "Done. %d patches built in %.1f s; manifest at %s",
        len(manifest_rows), elapsed, manifest_path,
    )
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--max-patches", type=int, default=None,
        help="Smoke-test cap (e.g., 16). Default: full V2 patch list.",
    )
    p.add_argument(
        "--combos", default=None,
        help='Comma-sep combo filter, e.g. "d306_h13,d356_h13". Default: all.',
    )
    p.add_argument(
        "--output-dir", default=str(DEFAULT_OUTPUT_DIR),
        help=f"Output directory for .npz + manifest. Default: {DEFAULT_OUTPUT_DIR}",
    )
    p.add_argument("--year", type=int, default=DEFAULT_YEAR)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument(
        "--no-lst", action="store_true",
        help="Skip Landsat LST sampling (fetches via GEE — slow).",
    )
    p.add_argument(
        "--no-skip-existing", action="store_true",
        help="Rebuild patches even if .npz already exists.",
    )
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    with V2_VALID_JSON.open() as f:
        valid = json.load(f)
    patch_ids = valid["valid_patches"]
    if args.max_patches:
        patch_ids = patch_ids[: args.max_patches]
    logger.info("V2 patches in scope: %d", len(patch_ids))

    combo_filter = None
    if args.combos:
        combo_filter = {parse_combo(c.strip()) for c in args.combos.split(",")}

    df = build_corpus(
        patch_ids=patch_ids,
        combo_filter=combo_filter,
        output_dir=Path(args.output_dir),
        year=args.year,
        fetch_lst=not args.no_lst,
        skip_existing=not args.no_skip_existing,
        workers=args.workers,
    )
    logger.info("manifest summary: %d rows", len(df))
    if len(df):
        print(df.head().to_string())
        print()
        print("totals:")
        print(f"  has_tmrt        : {int(df['has_tmrt'].sum())} / {len(df)}")
        print(f"  has_lst         : {int(df['has_lst'].sum())} / {len(df)}")
        print(f"  num_sensors > 0 : {int((df['num_sensors_in_patch'] > 0).sum())} / {len(df)}")


if __name__ == "__main__":
    main()
