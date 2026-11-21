#!/usr/bin/env python3
r"""WORLD CHANGE AUDIT -- how much of the coverage was built AFTER the LiDAR?

`dsm_1m.tif` is 2018 LiDAR (heights above ground; carries no canopy, memory.md
S24). `geoscape_heights.tif` is Geoscape Dec 2025 building heights on the SAME
grid. A building present in Geoscape but flat in the DSM was built after 2018:
it casts NO shadow anywhere in the pipeline -- not in the inputs, not in the
SOLWEIG labels, not in any eval. The model cannot know it exists, and every
metric we computed is blind to it. This is the leading suspect for the team's
"shadow missing in front of buildings" report (Parkville has many instances,
per the user).

    new_building  := geoscape >= 3 m  AND  dsm_1m < 0.5 m
    gone_changed  := dsm_1m  >= 3 m  AND  geoscape < 0.5 m
                     (demolished since 2018, or non-building structure)

Outputs totals + a 100 m-cell hotspot list (UTM + lat/lon) to check in the app.
Read-only. Server-side. Run:

    cd /mnt/apps && uv run --with rasterio --with numpy python \
        spatial-llm/src/umami_spatial-llm/scripts/urban_heat/world_change_audit.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform as rio_transform

RAW = Path("/mnt/apps/spatial-llm/src/umami_spatial-llm/urban_heat_data/rasters/raw")
DSM = RAW / "dsm_1m.tif"
GEO = RAW / "geoscape_heights.tif"
OUT = Path("/mnt/apps/spatial-llm/src/umami_spatial-llm/urban_heat_data/"
           "training/v5_calib") / "world_change_audit.json"

BUILDING_M = 3.0
GROUND_M = 0.5
CELL = 100            # hotspot cell size, px (= metres)
TOP_N = 15
STRIP = 1024          # rows per read


def main() -> int:
    with rasterio.open(DSM) as ds, rasterio.open(GEO) as gs:
        assert ds.shape == gs.shape and ds.transform == gs.transform, \
            "rasters are not on the same grid"
        H, W = ds.shape
        print(f"grid {W} x {H} @ {ds.transform.a:.1f} m, crs {ds.crs}")

        gh = -(-H // CELL)
        gw = -(-W // CELL)
        cell_new = np.zeros((gh, gw), dtype=np.int64)
        tot = {"valid": 0, "dsm_building": 0, "geo_building": 0,
               "new_building": 0, "gone_changed": 0}

        for r0 in range(0, H, STRIP):
            rows = min(STRIP, H - r0)
            win = rasterio.windows.Window(0, r0, W, rows)
            d = ds.read(1, window=win, masked=True)
            g = gs.read(1, window=win, masked=True)
            valid = (~d.mask) & (~g.mask)
            dv = np.where(valid, d.filled(0), 0)
            gv = np.where(valid, g.filled(0), 0)

            new = valid & (gv >= BUILDING_M) & (dv < GROUND_M)
            gone = valid & (dv >= BUILDING_M) & (gv < GROUND_M)
            tot["valid"] += int(valid.sum())
            tot["dsm_building"] += int((valid & (dv >= BUILDING_M)).sum())
            tot["geo_building"] += int((valid & (gv >= BUILDING_M)).sum())
            tot["new_building"] += int(new.sum())
            tot["gone_changed"] += int(gone.sum())

            # accumulate the 100 m hotspot grid
            nr, nc = new.shape
            pr = (-(-nr // CELL)) * CELL - nr
            pc = (-(-nc // CELL)) * CELL - nc
            npad = np.pad(new, ((0, pr), (0, pc)))
            blk = npad.reshape(npad.shape[0] // CELL, CELL,
                               npad.shape[1] // CELL, CELL).sum(axis=(1, 3))
            cr0 = r0 // CELL
            # a strip may straddle a cell row boundary only if STRIP % CELL
            # != 0 -- STRIP=1024, CELL=100 -> it does. Accumulate by offset:
            if r0 % CELL == 0:
                cell_new[cr0:cr0 + blk.shape[0], :blk.shape[1]] += blk
            else:
                # split first partial cell-row
                off = r0 % CELL
                first = min(CELL - off, nr)
                fnew = new[:first]
                fp = np.pad(fnew, ((0, 0), (0, pc)))
                frow = fp.reshape(1, first, -1, CELL).sum(axis=(1, 3))
                cell_new[cr0, :frow.shape[1]] += frow[0]
                rest = new[first:]
                if rest.shape[0]:
                    rp = (-(-rest.shape[0] // CELL)) * CELL - rest.shape[0]
                    rpad = np.pad(rest, ((0, rp), (0, pc)))
                    rblk = rpad.reshape(rpad.shape[0] // CELL, CELL,
                                        rpad.shape[1] // CELL,
                                        CELL).sum(axis=(1, 3))
                    cell_new[cr0 + 1:cr0 + 1 + rblk.shape[0],
                             :rblk.shape[1]] += rblk

        print()
        print("=" * 68)
        print("TOTALS (1 px = 1 m2)")
        print("=" * 68)
        v = tot["valid"]
        print(f"  valid pixels          : {v:,}")
        print(f"  building in 2018 DSM  : {tot['dsm_building']:,}  "
              f"({100 * tot['dsm_building'] / v:.2f}%)")
        print(f"  building in Geoscape25: {tot['geo_building']:,}  "
              f"({100 * tot['geo_building'] / v:.2f}%)")
        print(f"  NEW since 2018        : {tot['new_building']:,}  "
              f"({100 * tot['new_building'] / v:.2f}% of area, "
              f"{tot['new_building'] / 1e4:.1f} ha, "
              f"{100 * tot['new_building'] / max(tot['geo_building'], 1):.1f}% "
              "of today's building footprint)")
        print(f"  gone / changed        : {tot['gone_changed']:,}  "
              f"({tot['gone_changed'] / 1e4:.1f} ha)")

        print()
        print(f"TOP {TOP_N} NEW-CONSTRUCTION HOTSPOTS "
              f"({CELL} m cells; check these in the app)")
        print(f"  {'m2 new':>8}  {'easting':>9}{'northing':>11}"
              f"  {'lat':>10}{'lon':>11}")
        flat = cell_new.ravel()
        order = np.argsort(flat)[::-1][:TOP_N]
        hot = []
        for k in order:
            if flat[k] == 0:
                break
            cy, cx = divmod(int(k), gw)
            px = cx * CELL + CELL // 2
            py = cy * CELL + CELL // 2
            x, y = ds.transform * (px, py)
            lon, lat = rio_transform(ds.crs, "EPSG:4326", [x], [y])
            hot.append({"m2_new": int(flat[k]),
                        "easting": round(x, 1), "northing": round(y, 1),
                        "lat": round(lat[0], 6), "lon": round(lon[0], 6)})
            print(f"  {flat[k]:>8,}  {x:>9.0f}{y:>11.0f}"
                  f"  {lat[0]:>10.5f}{lon[0]:>11.5f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"totals": tot, "hotspots_100m": hot,
                               "thresholds": {"building_m": BUILDING_M,
                                              "ground_m": GROUND_M}}, indent=1))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
