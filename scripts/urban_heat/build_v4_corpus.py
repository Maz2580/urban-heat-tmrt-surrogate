"""Build V4 corpus — 23-channel multi-task with full ERA5 + NDVI + multi-shadow + roof-type + real Landsat LST.

Same approach as V3's build_v21_corpus.py: walks the V2 valid_patches.json
list × 12 SOLWEIG temporal combos and builds patches via
patch_builder.build_patch_v4. The differences:

  * 23-channel input (V3's 12 + 11 new — see V4_CHANNEL_ORDER in
    patch_builder.py for the canonical slot order)
      Slots 12-14: ERA5 RH%, wind speed (m/s), SSRD (W/m²)
      Slot 15:    NDVI (Sentinel-2 median)
      Slot 16:    Geoscape solar_panel_mask
      Slots 17-19: Multi-time-of-day shadows (h09 / h13 / h16)
      Slots 20-22: Geoscape roof_type one-hot (flat / moderate / steep)
  * Real Landsat LST per (patch, combo) sourced from
    urban_heat_data/v4_prep/landsat_per_patch.parquet
  * Full ERA5 4-tuple {tair_c, rh_pct, wind_speed_ms, ssrd_wm2} per
    combo sourced from urban_heat_data/v4_prep/era5_combos.parquet
  * V4 raster paths point to v4_prep/ NDVI + multi-shadow + the existing
    Geoscape solar_panel_mask + roof_type rasters

Output: urban_heat_data/training/patches_v4/*.npz + manifest.parquet
Wall time: ~30 min on 8 workers (slightly slower than V3 due to extra
raster reads per patch).

Usage:
    docker exec -e PYTHONPATH=/app umami_test python /app/scripts/urban_heat/build_v4_corpus.py \\
        --max-patches 16 --combos d306_h13   # smoke
    docker exec -e PYTHONPATH=/app umami_test python /app/scripts/urban_heat/build_v4_corpus.py \\
        --workers 8                           # full sweep
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.services.urban_heat.data_pipeline import patch_builder as pb  # noqa: E402

logger = logging.getLogger("umami.urban_heat.build_v4_corpus")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

V2_PATCHES_DIR = REPO_ROOT / "urban_heat_data/training/patches"
V2_VALID_JSON = REPO_ROOT / "urban_heat_data/training/valid_patches.json"
RASTER_DIR = REPO_ROOT / "urban_heat_data/rasters/raw"
V4_PREP_DIR = REPO_ROOT / "urban_heat_data/v4_prep"
SENSOR_PARQUET = REPO_ROOT / "urban_heat_data/sensors/sensors_data.parquet"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "urban_heat_data/training/patches_v4"

# V3-compatible raster paths (same as V3 corpus build)
RASTER_PATHS = {
    "dsm_corrected": RASTER_DIR / "geoscape_heights.tif",
    "cdsm": RASTER_DIR / "cdsm_1m.tif",
    "dem": RASTER_DIR / "dem_1m.tif",
    "albedo": RASTER_DIR / "geoscape_albedo.tif",
}

# V4-only raster paths (the new spatial channels added on top of V3's 12)
V4_RASTER_PATHS = {
    "ndvi": V4_PREP_DIR / "geoscape_ndvi.tif",
    "solar_panel_mask": RASTER_DIR / "geoscape_solar_panel_mask.tif",
    "shadow_h09": V4_PREP_DIR / "shadow_multi" / "shadow_h09.tif",
    "shadow_h13": V4_PREP_DIR / "shadow_multi" / "shadow_h13.tif",
    "shadow_h16": V4_PREP_DIR / "shadow_multi" / "shadow_h16.tif",
    "roof_type": RASTER_DIR / "geoscape_roof_type.tif",  # 3-band one-hot
}

# Required prep parquets
ERA5_PARQUET = V4_PREP_DIR / "era5_combos.parquet"
LANDSAT_PARQUET = V4_PREP_DIR / "landsat_per_patch.parquet"

DEFAULT_YEAR = 2024
TMRT_PATTERN = re.compile(r"Tmrt_d(\d{3})_h(\d{2})\.tif")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_combo(combo_str: str) -> tuple[int, int]:
    m = re.match(r"d(\d{1,3})_h(\d{1,2})", combo_str)
    if not m:
        raise ValueError(f"bad combo: {combo_str!r}")
    return int(m.group(1)), int(m.group(2))


def doy_hour_to_utc(doy: int, hour: int, year: int) -> datetime:
    aest = datetime(year, 1, 1, hour=hour, tzinfo=timezone.utc) + timedelta(days=doy - 1)
    return aest - timedelta(hours=10)


def patch_window_in_master(patch_dsm_path: Path) -> Window:
    with rasterio.open(patch_dsm_path) as patch_ds:
        b = patch_ds.bounds
        h, w = patch_ds.shape
    with rasterio.open(RASTER_PATHS["dsm_corrected"]) as master_ds:
        col, row = ~master_ds.transform * (b.left, b.top)
        return Window(col_off=int(round(col)), row_off=int(round(row)), width=w, height=h)


def list_temporal_combos(patch_id: str) -> list[tuple[int, int]]:
    out_dir = V2_PATCHES_DIR / patch_id / "output_folder"
    if not out_dir.is_dir():
        return []
    combos = []
    for fp in out_dir.glob("Tmrt_d*_h*.tif"):
        m = TMRT_PATTERN.match(fp.name)
        if m:
            combos.append((int(m.group(1)), int(m.group(2))))
    return sorted(combos)


def load_era5_lookup() -> dict[tuple[int, int], dict]:
    """Load the v4_prep ERA5 cache → {(doy, hour): full_dict}."""
    if not ERA5_PARQUET.exists():
        raise FileNotFoundError(
            f"ERA5 cache not found: {ERA5_PARQUET}. Run scripts/urban_heat/v4_prep.py --task era5 first."
        )
    df = pd.read_parquet(ERA5_PARQUET)
    return {
        (int(r.doy), int(r.hour)): {
            "tair_c": r.tair_c, "rh_pct": r.rh_pct,
            "wind_speed_ms": r.wind_speed_ms, "ssrd_wm2": r.ssrd_wm2,
        }
        for r in df.itertuples()
    }


def load_landsat_lookup() -> dict[tuple[str, int, int], float]:
    """Load v4_prep Landsat LST → {(patch_id, doy, hour): lst_celsius}."""
    if not LANDSAT_PARQUET.exists():
        logger.warning("Landsat parquet missing — V4 LST labels will all be NaN")
        return {}
    df = pd.read_parquet(LANDSAT_PARQUET)
    return {
        (r.v2_patch_id, int(r.doy), int(r.hour)): float(r.lst_celsius)
        for r in df.itertuples()
    }


def load_sensor_archive() -> Optional[pd.DataFrame]:
    if not SENSOR_PARQUET.exists():
        logger.warning("sensor archive not found at %s", SENSOR_PARQUET)
        return None
    df = pd.read_parquet(SENSOR_PARQUET)
    df["received_at"] = pd.to_datetime(df["received_at"], utc=True)
    return df


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

_WORKER_SENSOR_DF: Optional[pd.DataFrame] = None


def _worker_init() -> None:
    global _WORKER_SENSOR_DF
    _WORKER_SENSOR_DF = load_sensor_archive()


def build_one_task(
    patch_id: str, doy: int, hour: int, year: int,
    output_dir: Path, era5_dict: dict, lst_celsius: Optional[float],
    skip_existing: bool, sensor_df=None,
) -> dict:
    out_path = output_dir / f"{patch_id}_d{doy:03d}_h{hour:02d}.npz"
    if skip_existing and out_path.exists():
        return {"patch_id": patch_id, "doy": doy, "hour": hour, "skipped": True}

    if sensor_df is None:
        sensor_df = _WORKER_SENSOR_DF

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
    npz, manifest = pb.build_patch_v4(
        spec, raster_paths=RASTER_PATHS,
        era5_dict=era5_dict, v4_raster_paths=V4_RASTER_PATHS,
        sensor_df=sensor_df, tmrt_path=tmrt_path,
        lst_celsius=lst_celsius,
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
    skip_existing: bool = True,
    workers: int = 1,
    manifest_every: int = 100,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    sensor_df = load_sensor_archive()
    if sensor_df is not None:
        logger.info("loaded %d sensor rows", len(sensor_df))

    era5_lookup = load_era5_lookup()
    logger.info("loaded ERA5 cache: %d combos", len(era5_lookup))
    landsat_lookup = load_landsat_lookup()
    logger.info("loaded Landsat per-patch LST: %d (patch, combo) entries", len(landsat_lookup))

    # Plan tasks
    tasks = []
    for patch_id in patch_ids:
        for doy, hour in list_temporal_combos(patch_id):
            if combo_filter and (doy, hour) not in combo_filter:
                continue
            tasks.append((patch_id, doy, hour))
    logger.info("planned %d patch×combo tasks", len(tasks))

    # Verify ERA5 cache covers all needed combos
    needed_combos = {(d, h) for _, d, h in tasks}
    missing_era5 = needed_combos - set(era5_lookup.keys())
    if missing_era5:
        raise RuntimeError(
            f"ERA5 cache missing combos: {missing_era5}. "
            f"Re-run scripts/urban_heat/v4_prep.py --task era5"
        )

    manifest_path = output_dir / "manifest.parquet"
    manifest_rows: list[dict] = []
    t_start = time.time()
    last_log = t_start

    if workers == 1:
        for i, (patch_id, doy, hour) in enumerate(tasks, start=1):
            era5 = era5_lookup[(doy, hour)]
            lst = landsat_lookup.get((patch_id, doy, hour))
            row = build_one_task(
                patch_id, doy, hour, year, output_dir, era5, lst,
                skip_existing, sensor_df,
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
        with ProcessPoolExecutor(
            max_workers=workers, initializer=_worker_init,
        ) as ex:
            futures = [
                ex.submit(
                    build_one_task,
                    patch_id, doy, hour, year, output_dir,
                    era5_lookup[(doy, hour)],
                    landsat_lookup.get((patch_id, doy, hour)),
                    skip_existing, None,
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
    p.add_argument("--max-patches", type=int, default=None,
                   help="Smoke-test cap. Default: full V2 patch list.")
    p.add_argument("--combos", default=None,
                   help='Comma-sep combo filter, e.g. "d306_h13,d356_h13".')
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--year", type=int, default=DEFAULT_YEAR)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--no-skip-existing", action="store_true")
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
        patch_ids=patch_ids, combo_filter=combo_filter,
        output_dir=Path(args.output_dir), year=args.year,
        skip_existing=not args.no_skip_existing,
        workers=args.workers,
    )
    logger.info("manifest summary: %d rows", len(df))
    if len(df):
        print(df.head().to_string())
        print()
        print("totals:")
        print(f"  has_tmrt        : {int(df['has_tmrt'].sum())} / {len(df)}")
        print(f"  has_lst (real)  : {int(df['has_lst'].sum())} / {len(df)}")
        print(f"  num_sensors > 0 : {int((df['num_sensors_in_patch'] > 0).sum())} / {len(df)}")


if __name__ == "__main__":
    main()
