#!/usr/bin/env python3
"""
Urban Heat Tool — Data Download Script (Phase 1.6.1)

Downloads vector datasets from public APIs for the City of Melbourne
Urban Heat Island analysis pipeline.

Sources:
  1. VicMap VaaS (ArcGIS REST) — Hydro, Transport (no auth)
  2. City of Melbourne Open Data — 2023 Building Footprints, Urban Forest (no auth)
  3. AURIN WFS — Geoscape Buildings, HVI, Roads (requires credentials via env vars)

Usage:
  # Download all public layers:
  python download_data_sources.py

  # Download specific layer only:
  python download_data_sources.py --layer water
  python download_data_sources.py --layer roads
  python download_data_sources.py --layer buildings

  # Download AURIN layers (requires AURIN_USER + AURIN_TOKEN env vars):
  python download_data_sources.py --layer aurin_buildings
  python download_data_sources.py --layer aurin_hvi
  python download_data_sources.py --layer aurin_roads

  # Download ALL layers including AURIN (requires credentials):
  python download_data_sources.py --all

  # Dry run (show what would be downloaded):
  python download_data_sources.py --dry-run
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("urban_heat.download")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# City of Melbourne bounding box (EPSG:4326)
COM_BBOX = {
    "xmin": 144.90,
    "ymin": -37.84,
    "xmax": 144.99,
    "ymax": -37.78,
}

# VicMap as a Service (VaaS) — ArcGIS REST API (no auth)
VAAS_BASE = "https://services-ap1.arcgis.com/P744lA0wf4LlBZ84/ArcGIS/rest/services"

# Output directory (relative to project root)
# Inside Docker: /app/urban_heat_data/vectors/
# On host: /mnt/apps/spatial-llm/src/umami_spatial-llm/urban_heat_data/vectors/
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # scripts/urban_heat/ -> project root
OUTPUT_DIR = PROJECT_ROOT / "urban_heat_data" / "vectors"

# Max records per ArcGIS REST request
PAGE_SIZE = 2000

# Request timeout (seconds)
TIMEOUT = 120

# Retry config
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds


# ---------------------------------------------------------------------------
# Layer Definitions
# ---------------------------------------------------------------------------

LAYERS = {
    "water": {
        "name": "Water Areas (VicMap Hydro)",
        "source": "vicmap",
        "service": "Vicmap_Hydro",
        "layer_id": 1,
        "output": "water_areas.geojson",
        "description": "Yarra River, Docklands, ponds — Albedo 0.06",
    },
    "road_infra": {
        "name": "Road Infrastructure (VicMap Transport)",
        "source": "vicmap",
        "service": "Vicmap_Transport",
        "layer_id": 0,
        "output": "road_infra.geojson",
        "description": "Road surface polygons — Albedo 0.12 (asphalt), 0.30 (concrete)",
    },
    "road_network": {
        "name": "Road Network (VicMap Transport)",
        "source": "vicmap",
        "service": "Vicmap_Transport",
        "layer_id": 1,
        "output": "road_network.geojson",
        "description": "Road centrelines with ROAD_SEAL attribute",
    },
    "buildings": {
        "name": "2023 Building Footprints (City of Melbourne)",
        "source": "com_opendata",
        "output": "buildings_com_2023.geojson",
        "description": "All building footprints within CoM — primary roof mask",
    },
    "urban_forest": {
        "name": "Urban Forest (City of Melbourne)",
        "source": "com_opendata_trees",
        "output": "urban_forest.geojson",
        "description": "Tree locations with species, canopy width — derive CDSM",
    },
    "aurin_buildings": {
        "name": "Geoscape Melbourne Buildings Jun 2022 (AURIN WFS)",
        "source": "aurin",
        "typename": "datasource-Geoscape-UoM_AURIN_DB:geoscape_melbourne_buildings_jun22",
        "output": "geoscape_buildings.geojson",
        "description": "Building heights (roof_hgt, eave_hgt, grd_elev), roof material/type/color — enhanced albedo & DSM (requires auth)",
    },
    "aurin_hvi": {
        "name": "Heat Vulnerability Index Melbourne SA1 2021 (AURIN WFS)",
        "source": "aurin",
        "typename": "gisail_rmit_updated:hvi_cityofgreatermelbourne_sa1_2021",
        "output": "hvi_melbourne_sa1_2021.geojson",
        "description": "HVI (0-5), LST, NDVI, heat exposure/sensitivity/adaptive indexes per SA1 zone — validation data (requires auth)",
    },
    "aurin_roads": {
        "name": "Geoscape Melbourne Roads Jun 2022 (AURIN WFS)",
        "source": "aurin",
        "typename": "datasource-Geoscape-UoM_AURIN_DB:geoscape_melbourne_roads_jun22",
        "output": "geoscape_roads.geojson",
        "description": "Road surface type, hierarchy, lane count — enhanced albedo by surface material (requires auth)",
    },
}

# Public layers (no authentication required)
PUBLIC_LAYERS = ["water", "road_infra", "road_network", "buildings", "urban_forest"]

# AURIN layers (require AURIN_USER + AURIN_TOKEN)
AURIN_LAYERS = ["aurin_buildings", "aurin_hvi", "aurin_roads"]


# ---------------------------------------------------------------------------
# Download Functions
# ---------------------------------------------------------------------------

def _request_with_retry(url: str, params: Dict, auth=None) -> requests.Response:
    """Make HTTP GET with retries and exponential backoff."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, auth=auth, timeout=TIMEOUT)
            resp.raise_for_status()
            return resp
        except requests.exceptions.RequestException as e:
            if attempt == MAX_RETRIES:
                raise
            wait = RETRY_DELAY * attempt
            logger.warning(f"  Attempt {attempt}/{MAX_RETRIES} failed: {e}. Retrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError("Unreachable")


def download_vicmap_layer(
    service: str,
    layer_id: int,
    output_path: Path,
) -> int:
    """
    Download a VicMap layer via ArcGIS REST API with pagination.
    Returns the number of features downloaded.
    """
    url = f"{VAAS_BASE}/{service}/FeatureServer/{layer_id}/query"

    all_features: List[Dict[str, Any]] = []
    offset = 0

    while True:
        params = {
            "where": "1=1",
            "geometry": json.dumps({
                "xmin": COM_BBOX["xmin"],
                "ymin": COM_BBOX["ymin"],
                "xmax": COM_BBOX["xmax"],
                "ymax": COM_BBOX["ymax"],
                "spatialReference": {"wkid": 4326},
            }),
            "geometryType": "esriGeometryEnvelope",
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": "*",
            "outSR": "4326",
            "f": "geojson",
            "resultRecordCount": PAGE_SIZE,
            "resultOffset": offset,
        }

        logger.info(f"  Fetching offset={offset} (page {offset // PAGE_SIZE + 1})...")
        resp = _request_with_retry(url, params)
        data = resp.json()

        # Check for ArcGIS errors
        if "error" in data:
            error_msg = data["error"].get("message", str(data["error"]))
            raise RuntimeError(f"ArcGIS API error: {error_msg}")

        features = data.get("features", [])
        all_features.extend(features)
        logger.info(f"  Got {len(features)} features (total: {len(all_features)})")

        if len(features) < PAGE_SIZE:
            break
        offset += PAGE_SIZE

    if not all_features:
        logger.warning(f"  No features returned for {service}/{layer_id}")
        return 0

    # Build GeoJSON FeatureCollection
    geojson = {
        "type": "FeatureCollection",
        "features": all_features,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(geojson, f)

    return len(all_features)


def download_com_buildings(output_path: Path) -> int:
    """
    Download 2023 Building Footprints from City of Melbourne Open Data Portal.
    Uses the v2.1 export API for GeoJSON format.
    """
    url = "https://data.melbourne.vic.gov.au/api/explore/v2.1/catalog/datasets/2023-building-footprints/exports/geojson"

    params = {
        "limit": -1,  # all records
        "timezone": "Australia/Melbourne",
    }

    logger.info("  Requesting full dataset export (this may take a moment)...")
    resp = _request_with_retry(url, params)
    data = resp.json()

    features = data.get("features", [])
    if not features:
        logger.warning("  No features in CoM buildings response")
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f)

    return len(features)


def download_com_urban_forest(output_path: Path) -> int:
    """
    Download Urban Forest data from City of Melbourne Open Data Portal.
    Dataset: trees-with-species-and-dimensions (includes canopy width, height).
    """
    url = "https://data.melbourne.vic.gov.au/api/explore/v2.1/catalog/datasets/trees-with-species-and-dimensions-urban-forest/exports/geojson"

    params = {
        "limit": -1,
        "timezone": "Australia/Melbourne",
    }

    logger.info("  Requesting Urban Forest export (large dataset)...")
    resp = _request_with_retry(url, params)
    data = resp.json()

    features = data.get("features", [])
    if not features:
        logger.warning("  No features in Urban Forest response")
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f)

    return len(features)


def download_aurin_layer(
    typename: str,
    output_path: Path,
    username: Optional[str] = None,
    token: Optional[str] = None,
) -> int:
    """
    Download a dataset from AURIN via WFS GetFeature with bounding box filter.
    Paginates using startIndex for large datasets (e.g. Geoscape Buildings).
    Requires AURIN credentials (university institutional access).
    """
    user = username or os.environ.get("AURIN_USER")
    api_token = token or os.environ.get("AURIN_TOKEN")

    if not user or not api_token:
        logger.error("  AURIN credentials not provided. Set AURIN_USER and AURIN_TOKEN env vars.")
        logger.error("  Skipping AURIN download.")
        return 0

    from requests.auth import HTTPBasicAuth
    auth = HTTPBasicAuth(user, api_token)

    url = os.environ.get("AURIN_WFS_URL", "https://adp.aurin.org.au/geoserver/wfs")
    aurin_page_size = 10000
    all_features: List[Dict[str, Any]] = []

    # First request without startIndex (some AURIN layers don't support it)
    params = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "typeNames": typename,
        "outputFormat": "application/json",
        "srsName": "EPSG:4326",
        "bbox": f"{COM_BBOX['xmin']},{COM_BBOX['ymin']},{COM_BBOX['xmax']},{COM_BBOX['ymax']},EPSG:4326",
        "count": aurin_page_size,
    }

    logger.info(f"  Querying AURIN WFS: {typename}...")
    resp = _request_with_retry(url, params, auth=auth)
    data = resp.json()

    features = data.get("features", [])
    all_features.extend(features)
    logger.info(f"  Got {len(features)} features")

    # Paginate only if we hit the count limit (layer supports pagination)
    if len(features) >= aurin_page_size:
        start_index = aurin_page_size
        while True:
            params["startIndex"] = start_index
            logger.info(f"  Paginating: startIndex={start_index}...")
            try:
                resp = _request_with_retry(url, params, auth=auth)
                data = resp.json()
                features = data.get("features", [])
                all_features.extend(features)
                logger.info(f"  Got {len(features)} features (total: {len(all_features)})")
                if len(features) < aurin_page_size:
                    break
                start_index += aurin_page_size
            except Exception as e:
                logger.warning(f"  Pagination failed at startIndex={start_index}: {e}")
                logger.warning(f"  Continuing with {len(all_features)} features collected so far")
                break

    if not all_features:
        logger.warning(f"  No features returned from AURIN for {typename}")
        return 0

    geojson = {
        "type": "FeatureCollection",
        "features": all_features,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(geojson, f)

    return len(all_features)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def download_layer(layer_key: str, dry_run: bool = False, output_dir: Path = None) -> bool:
    """Download a single layer by key. Returns True on success."""
    if layer_key not in LAYERS:
        logger.error(f"Unknown layer: {layer_key}. Available: {list(LAYERS.keys())}")
        return False

    base_dir = output_dir or OUTPUT_DIR
    layer = LAYERS[layer_key]
    output_path = base_dir / layer["output"]

    logger.info(f"\n{'='*60}")
    logger.info(f"Layer: {layer['name']}")
    logger.info(f"Description: {layer['description']}")
    logger.info(f"Output: {output_path}")

    if dry_run:
        logger.info("  [DRY RUN] Would download this layer")
        return True

    # Check if already downloaded
    if output_path.exists():
        size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"  Already exists ({size_mb:.1f} MB). Use --force to re-download.")
        return True

    try:
        start = time.time()
        source = layer["source"]

        if source == "vicmap":
            count = download_vicmap_layer(
                service=layer["service"],
                layer_id=layer["layer_id"],
                output_path=output_path,
            )
        elif source == "com_opendata":
            count = download_com_buildings(output_path)
        elif source == "com_opendata_trees":
            count = download_com_urban_forest(output_path)
        elif source == "aurin":
            count = download_aurin_layer(
                typename=layer["typename"],
                output_path=output_path,
            )
        else:
            logger.error(f"  Unknown source type: {source}")
            return False

        elapsed = time.time() - start
        size_mb = output_path.stat().st_size / (1024 * 1024) if output_path.exists() else 0

        if count > 0:
            logger.info(f"  Downloaded {count} features ({size_mb:.1f} MB) in {elapsed:.1f}s")
            return True
        else:
            logger.warning(f"  No features downloaded for {layer_key}")
            return False

    except Exception as e:
        logger.error(f"  FAILED: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download vector datasets for Urban Heat Tool (Phase 1.6.1)",
    )
    parser.add_argument(
        "--layer",
        choices=list(LAYERS.keys()),
        help="Download a specific layer only. If omitted, downloads all public layers.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all layers including AURIN (requires credentials).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if file already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without actually downloading.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help=f"Override output directory (default: {OUTPUT_DIR})",
    )

    args = parser.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR

    logger.info(f"Urban Heat Data Download Script")
    logger.info(f"Output directory: {out_dir}")
    logger.info(f"CoM bounding box: {COM_BBOX}")

    # Ensure output dir exists
    out_dir.mkdir(parents=True, exist_ok=True)

    # Handle --force by removing existing files
    if args.force and not args.dry_run:
        logger.info("Force mode: will overwrite existing files")

    if args.layer:
        # Single layer
        if args.force:
            target = out_dir / LAYERS[args.layer]["output"]
            if target.exists():
                target.unlink()
        success = download_layer(args.layer, dry_run=args.dry_run, output_dir=out_dir)
        sys.exit(0 if success else 1)

    # Multiple layers
    layers_to_download = list(LAYERS.keys()) if args.all else PUBLIC_LAYERS
    results = {}

    for layer_key in layers_to_download:
        if args.force:
            target = out_dir / LAYERS[layer_key]["output"]
            if target.exists():
                target.unlink()
        results[layer_key] = download_layer(layer_key, dry_run=args.dry_run, output_dir=out_dir)

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("DOWNLOAD SUMMARY")
    logger.info(f"{'='*60}")
    for key, success in results.items():
        status = "OK" if success else "FAILED"
        logger.info(f"  [{status}] {LAYERS[key]['name']}")

    failed = [k for k, v in results.items() if not v]
    if failed:
        logger.warning(f"\n{len(failed)} layer(s) failed. Re-run with --layer <name> to retry.")
        sys.exit(1)
    else:
        logger.info(f"\nAll {len(results)} layers downloaded successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
