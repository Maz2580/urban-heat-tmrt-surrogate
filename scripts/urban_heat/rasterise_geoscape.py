"""Plan UH-1 / A3 — rasterise Geoscape buildings + greenspace.

Reads the two GeoJSON zips delivered to
`docs/report/urbanheat_model_improvement/`:
  - `com_buildings_geoscape_dec_2025.zip` — 32,094 polygons with rich
    per-building attributes (material, colour, roof type, solar
    panel, heights, ...)
  - `com_greenspace_polygon_geoscape_feb_2021.zip` — 1,473 polygons,
    a `PRC_TYP_CD` flag and area only.

Emits four GeoTIFFs aligned to the existing 9000 × 9000 / 1 m UTM-55S
training grid (`urban_heat_data/rasters/raw/dsm_1m.tif`):

  1. `geoscape_albedo.tif` (float32, 1 band)
       Per-building albedo blended from `primary_roof_material`
       (lookup table) and `roof_colour` (RGB hex → grayscale
       brightness). Falls back to `albedo_master.tif` value off-roof.
  2. `geoscape_solar_panel_mask.tif` (uint8, 1 band)
       Binary 1 where Geoscape `solar_panel == 'Yes'`.
  3. `geoscape_roof_type.tif` (uint8, 3 bands)
       One-hot Flat / Moderate / Steep.
  4. `geoscape_heights.tif` (float32, 1 band)
       `roof_height` blended over the noisy DSM 2018 cells (DSM
       elsewhere unchanged; tiny smoothing at building boundaries).

Run:
  python -m scripts.urban_heat.rasterise_geoscape

Or:
  python scripts/urban_heat/rasterise_geoscape.py

Reads + writes only under /mnt; no /tmp use; ~30 sec total.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import zipfile
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---- paths -------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]  # .../umami_spatial-llm/
GEOSCAPE_DIR = REPO_ROOT / "docs/report/urbanheat_model_improvement"
BUILDINGS_ZIP = GEOSCAPE_DIR / "com_buildings_geoscape_dec_2025.zip"
GREENSPACE_ZIP = GEOSCAPE_DIR / "com_greenspace_polygon_geoscape_feb_2021.zip"

RASTER_RAW_DIR = REPO_ROOT / "urban_heat_data/rasters/raw"
GRID_TEMPLATE = RASTER_RAW_DIR / "dsm_1m.tif"
ALBEDO_FALLBACK = RASTER_RAW_DIR / "albedo_master.tif"

OUT_ALBEDO = RASTER_RAW_DIR / "geoscape_albedo.tif"
OUT_SOLAR_MASK = RASTER_RAW_DIR / "geoscape_solar_panel_mask.tif"
OUT_ROOF_TYPE = RASTER_RAW_DIR / "geoscape_roof_type.tif"
OUT_HEIGHTS = RASTER_RAW_DIR / "geoscape_heights.tif"

# ---- parameters --------------------------------------------------------

# Per-material reflectance constants. Sourced from public roofing /
# urban-heat literature mid-points; we'll blend with roof_colour
# brightness so light-coloured tiles read higher than dark concrete.
MATERIAL_ALBEDO = {
    "Metal": 0.55,                 # bright metal sheets, broad mid-point
    "Tile": 0.22,                  # terracotta typical
    "Flat Concrete": 0.27,
    "Fiberglass/Plastic": 0.30,
}
MATERIAL_DEFAULT = 0.30

# Mix weight: 0.6 × material constant + 0.4 × colour-brightness.
MATERIAL_WEIGHT = 0.6
COLOUR_WEIGHT = 0.4

# Roof type → one-hot band index (0=Flat, 1=Moderate, 2=Steep).
ROOF_TYPE_BAND = {
    "Flat": 0,
    "Moderate pitch or complexity": 1,
    "Steep pitch or high complexity": 2,
}


# ---- helpers -----------------------------------------------------------


def _hex_to_brightness(hex_str: Optional[str]) -> Optional[float]:
    """Convert '#CDC5C2' → 0..1 grayscale luminance.

    Uses Rec. 709 weights (0.2126 R, 0.7152 G, 0.0722 B). The
    upstream Geoscape data sometimes has empty / placeholder colours;
    return None for those so callers fall back to the material value.
    """
    if not hex_str or not isinstance(hex_str, str):
        return None
    s = hex_str.strip().lstrip("#")
    if len(s) != 6:
        return None
    try:
        r = int(s[0:2], 16) / 255.0
        g = int(s[2:4], 16) / 255.0
        b = int(s[4:6], 16) / 255.0
    except ValueError:
        return None
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _albedo_for_feature(props: dict) -> float:
    """Blend material + colour into a single albedo value."""
    mat = props.get("primary_roof_material") or ""
    colour = props.get("roof_colour")
    mat_val = MATERIAL_ALBEDO.get(mat, MATERIAL_DEFAULT)
    bright = _hex_to_brightness(colour)
    if bright is None:
        return mat_val
    # Bias the blend toward the material constant so colour acts as a
    # modulator, not an override.
    return MATERIAL_WEIGHT * mat_val + COLOUR_WEIGHT * bright


def _load_geojson_from_zip(zip_path: Path) -> dict:
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.endswith(".geojson")]
        if not names:
            raise RuntimeError(f"no .geojson in {zip_path}")
        with zf.open(names[0]) as f:
            return json.load(f)


# ---- main rasterisation -------------------------------------------------


def main(out_dir: Path = RASTER_RAW_DIR) -> dict:
    import rasterio
    from rasterio import features
    from rasterio.warp import transform_geom
    import shapely.geometry as sg

    if not GRID_TEMPLATE.exists():
        raise FileNotFoundError(
            f"grid template not found: {GRID_TEMPLATE}\n"
            f"(this script needs the 9000×9000 1 m UTM-55S DSM as alignment ref)",
        )
    if not BUILDINGS_ZIP.exists():
        raise FileNotFoundError(f"missing: {BUILDINGS_ZIP}")
    if not GREENSPACE_ZIP.exists():
        raise FileNotFoundError(f"missing: {GREENSPACE_ZIP}")

    out_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(GRID_TEMPLATE) as ref:
        transform = ref.transform
        width = ref.width
        height = ref.height
        crs = ref.crs
        ref_dtype = ref.dtypes[0]
    logger.info("Aligning to grid: %dx%d, %s, transform=%s",
                width, height, crs, transform)

    # ---- buildings ---------------------------------------------------

    logger.info("Reading buildings GeoJSON …")
    bldg = _load_geojson_from_zip(BUILDINGS_ZIP)
    bfeats = bldg.get("features", []) or []
    logger.info("  %d building polygons", len(bfeats))

    # Reproject every polygon from EPSG:4326 → grid CRS (32755) once,
    # caching the projected geometry alongside derived properties.
    bldg_records: list[tuple[dict, dict]] = []  # (projected_geom, props)
    for feat in bfeats:
        geom_4326 = feat.get("geometry")
        if not geom_4326:
            continue
        try:
            geom_utm = transform_geom("EPSG:4326", crs.to_string(), geom_4326)
        except Exception:
            continue
        bldg_records.append((geom_utm, feat.get("properties") or {}))
    logger.info("  %d projected building geometries", len(bldg_records))

    # ---- albedo ------------------------------------------------------

    logger.info("Rasterising albedo …")
    if ALBEDO_FALLBACK.exists():
        with rasterio.open(ALBEDO_FALLBACK) as a:
            albedo = a.read(1).astype(np.float32)
    else:
        albedo = np.full((height, width), MATERIAL_DEFAULT, dtype=np.float32)

    shapes_albedo = [
        (geom, _albedo_for_feature(props))
        for geom, props in bldg_records
    ]
    if shapes_albedo:
        building_albedo = features.rasterize(
            shapes=shapes_albedo,
            out_shape=(height, width),
            transform=transform,
            fill=0.0,
            dtype="float32",
            all_touched=False,
        )
        # Where building_albedo > 0 we overwrite the master raster.
        # Treat 0 as "no building here" (some buildings might legitimately
        # have albedo 0 but our material table has none below 0.22).
        mask = building_albedo > 0.0
        albedo = np.where(mask, building_albedo, albedo)

    _write_geotiff(
        OUT_ALBEDO, albedo, transform, crs, dtype="float32", nodata=None,
    )

    # ---- solar panel mask -------------------------------------------

    logger.info("Rasterising solar-panel mask …")
    shapes_solar = [
        (geom, 1)
        for geom, props in bldg_records
        if props.get("solar_panel") == "Yes"
    ]
    solar_mask = (
        features.rasterize(
            shapes=shapes_solar,
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype="uint8",
            all_touched=False,
        )
        if shapes_solar else np.zeros((height, width), dtype="uint8")
    )
    _write_geotiff(
        OUT_SOLAR_MASK, solar_mask, transform, crs, dtype="uint8", nodata=None,
    )

    # ---- roof type one-hot ------------------------------------------

    logger.info("Rasterising roof-type one-hot …")
    roof_bands = np.zeros((3, height, width), dtype="uint8")
    for band_idx in range(3):
        roof_type_label = [k for k, v in ROOF_TYPE_BAND.items() if v == band_idx][0]
        shapes = [
            (geom, 1)
            for geom, props in bldg_records
            if props.get("roof_type") == roof_type_label
        ]
        if shapes:
            roof_bands[band_idx] = features.rasterize(
                shapes=shapes,
                out_shape=(height, width),
                transform=transform,
                fill=0,
                dtype="uint8",
                all_touched=False,
            )
    _write_geotiff(
        OUT_ROOF_TYPE, roof_bands, transform, crs,
        dtype="uint8", nodata=None,
    )

    # ---- heights ----------------------------------------------------

    logger.info("Rasterising heights (overlay on DSM) …")
    with rasterio.open(GRID_TEMPLATE) as g:
        dsm_raw = g.read(1).astype(np.float32)
        # DSM declares nodata=None but uses -32767 as sentinel; detect both.
        dsm_nodata = float(g.nodata) if g.nodata is not None else -32767.0
    # Reprojection bleed turned ~0.1% of cells into values like -32439 …
    # -1174 (sentinel mixed with neighbours by bilinear). Anything below
    # -100 m is impossible in the FB AOI (deepest natural point ≈ -30 m).
    BOGUS_THRESHOLD = -100.0
    heights = np.where(dsm_raw < BOGUS_THRESHOLD, dsm_nodata, dsm_raw)
    shapes_height = [
        (geom, float(props.get("roof_height") or 0.0))
        for geom, props in bldg_records
        if props.get("roof_height") is not None
    ]
    if shapes_height:
        bldg_heights = features.rasterize(
            shapes=shapes_height,
            out_shape=(height, width),
            transform=transform,
            fill=0.0,
            dtype="float32",
            all_touched=False,
        )
        # ground_elevation is per-building; if available, we add it
        # so the raster matches DSM (which is absolute elevation).
        ge_shapes = [
            (geom, float(props.get("ground_elevation") or 0.0))
            for geom, props in bldg_records
            if props.get("roof_height") is not None
        ]
        bldg_ground = features.rasterize(
            shapes=ge_shapes,
            out_shape=(height, width),
            transform=transform,
            fill=0.0,
            dtype="float32",
            all_touched=False,
        )
        bldg_top = bldg_heights + bldg_ground
        # Buildings always win — overrides DSM even where DSM was NoData.
        overlay_mask = bldg_heights > 0.0
        heights = np.where(overlay_mask, bldg_top, heights)
    nodata_pixels = int((heights == dsm_nodata).sum())
    logger.info(
        "heights nodata pixels: %s (%.1f%%)",
        f"{nodata_pixels:,}",
        100.0 * nodata_pixels / heights.size,
    )
    _write_geotiff(
        OUT_HEIGHTS, heights, transform, crs, dtype="float32",
        nodata=dsm_nodata,
    )

    # ---- summary ----------------------------------------------------

    summary = {
        "albedo": {
            "path": str(OUT_ALBEDO),
            "min": float(albedo.min()),
            "max": float(albedo.max()),
            "mean": float(albedo.mean()),
        },
        "solar_panel_mask": {
            "path": str(OUT_SOLAR_MASK),
            "yes_pixels": int(solar_mask.sum()),
            "yes_pct": float(solar_mask.sum()) / (height * width) * 100,
        },
        "roof_type": {
            "path": str(OUT_ROOF_TYPE),
            "flat_pixels": int(roof_bands[0].sum()),
            "moderate_pixels": int(roof_bands[1].sum()),
            "steep_pixels": int(roof_bands[2].sum()),
        },
        "heights": {
            "path": str(OUT_HEIGHTS),
            "nodata": float(dsm_nodata),
            "valid_min": float(heights[heights != dsm_nodata].min()),
            "valid_max": float(heights[heights != dsm_nodata].max()),
            "valid_mean": float(heights[heights != dsm_nodata].mean()),
            "valid_pct": 100.0 * float((heights != dsm_nodata).sum()) / heights.size,
        },
    }
    logger.info("Done. Summary:\n%s", json.dumps(summary, indent=2))
    return summary


def _write_geotiff(
    path: Path, array, transform, crs, *, dtype: str, nodata=None,
) -> None:
    import rasterio

    if array.ndim == 2:
        bands = 1
        h, w = array.shape
        write_array = array[np.newaxis, :, :]
    else:
        bands, h, w = array.shape
        write_array = array

    profile = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": bands,
        "dtype": dtype,
        "crs": crs,
        "transform": transform,
        "compress": "deflate",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "predictor": 2 if dtype.startswith("float") else 1,
        "nodata": nodata,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(write_array)
    sz_mb = path.stat().st_size / 1e6
    logger.info("wrote %s (%.1f MB)", path, sz_mb)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    p = argparse.ArgumentParser(description=__doc__)
    args = p.parse_args()
    main()
