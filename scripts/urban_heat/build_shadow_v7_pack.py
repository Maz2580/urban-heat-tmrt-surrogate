#!/usr/bin/env python3
"""Crop V7's per-patch planes: ONE consistent 2018 world for every input.

V7's premise (memory.md 2026-08-20 world-audit): the corpus mixed two worlds
-- ch0 geometry from Geoscape Dec 2025, SOLWEIG labels from the 2018 LiDAR --
so the model was trained that some buildings it sees cast no shadow, and
hedged. V7 makes every input agree with the labels' world:

  slot 0    dsm_1m (2018 LiDAR heights)      -- replaces Geoscape
  slots 17-19  corrected-direction planes on dsm_1m, 400 m ray
  slot 10   shade age (consecutive hours shaded) at the sample's own hour --
            carried as four per-hour crops; stage_d places the right one

svf / building_density / aspect etc. were ALWAYS dsm_1m-derived (v4_prep.py),
so after this swap no channel disagrees with the labels.

--control reuses the V6 gate: crop the OLD rasters through the same window
arithmetic and require bit-identity with corpus slots 17-19. If that fails,
every crop this script makes is misaligned and would surface later as a
"model problem".

Reads everything read-only; writes only --out-dir.

Server usage:
  cd /mnt/apps && uv run --with rasterio --with numpy python \
      spatial-llm/src/umami_spatial-llm/scripts/urban_heat/build_shadow_v7_pack.py \
      --patch-list <list.txt> --control          # gate first
  ... same without --control, plus --out-dir <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rasterio

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "urban_heat_data"
PATCH_DIR = DATA / "training/patches"
V4_DIR = DATA / "training/patches_v4"
SHADOW_OLD = DATA / "v4_prep/shadow_multi"
SHADOW_V7 = DATA / "v4_prep/shadow_multi_v7"
MASTER = DATA / "rasters/raw/dsm_1m.tif"

HOURS = (9, 13, 16)
SLOTS = {9: 17, 13: 18, 16: 19}
AGE_HOURS = (9, 11, 13, 15)
PX = 512


def patch_window(pid: str, master_transform):
    f = PATCH_DIR / pid / "building_dsm.tif"
    if not f.exists():
        return None
    with rasterio.open(f) as s:
        t = s.transform
    col = (t.c - master_transform.c) / master_transform.a
    row = (t.f - master_transform.f) / master_transform.e
    return int(round(row)), int(round(col))


def read_win(src, win) -> np.ndarray:
    a = src.read(1, window=win)
    if a.shape != (PX, PX):
        a = np.zeros((PX, PX), dtype=np.float32)
    return a.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch-list", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--control", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    txt = args.patch_list.read_text()
    if args.patch_list.suffix == ".json":
        pids = list(json.loads(txt))
    else:
        pids = [l.strip() for l in txt.splitlines() if l.strip()]
    if args.limit:
        pids = pids[:args.limit]

    with rasterio.open(MASTER) as s:
        mt = s.transform

    if args.control:
        src = {h: rasterio.open(SHADOW_OLD / f"shadow_h{h:02d}.tif")
               for h in HOURS}
    else:
        if args.out_dir is None:
            print("--out-dir required unless --control", file=sys.stderr)
            sys.exit(2)
        src = {h: rasterio.open(SHADOW_V7 / f"shadow_h{h:02d}.tif")
               for h in HOURS}
        age = {h: rasterio.open(SHADOW_V7 / f"shade_age_h{h:02d}.tif")
               for h in AGE_HOURS}
        dsm = rasterio.open(MASTER)
        args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"mode    : {'CONTROL (old rasters vs corpus)' if args.control else 'BUILD V7 pack'}")
    print(f"patches : {len(pids)}\n")

    ok = bad = skipped = 0
    failures = []
    for i, pid in enumerate(pids):
        rc = patch_window(pid, mt)
        if rc is None:
            skipped += 1
            failures.append(f"{pid}: no building_dsm.tif")
            continue
        row, col = rc
        win = rasterio.windows.Window(col, row, PX, PX)
        planes = {h: read_win(src[h], win) for h in HOURS}

        if args.control:
            ref_files = sorted(V4_DIR.glob(f"{pid}_*.npz"))
            if not ref_files:
                skipped += 1
                continue
            x = np.load(ref_files[0])["input"]
            mism = [h for h in HOURS
                    if not np.array_equal(planes[h], x[SLOTS[h]])]
            if mism:
                bad += 1
                d = max(float(np.abs(planes[h] - x[SLOTS[h]]).max())
                        for h in mism)
                failures.append(
                    f"{pid}: slots {[SLOTS[h] for h in mism]} differ, max {d:g}")
            else:
                ok += 1
        else:
            d = read_win(dsm, win)
            d[~np.isfinite(d)] = 0.0
            d[d < -100.0] = 0.0
            np.savez_compressed(
                args.out_dir / f"{pid}.npz",
                shadow_h09=planes[9].astype(np.uint8),
                shadow_h13=planes[13].astype(np.uint8),
                shadow_h16=planes[16].astype(np.uint8),
                dsm_2018=d.astype(np.float32),
                **{f"shade_age_h{h:02d}": read_win(age[h], win).astype(np.uint8)
                   for h in AGE_HOURS},
                row=np.int32(row), col=np.int32(col),
            )
            ok += 1
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(pids)} ...")

    print(f"\n  ok={ok}  mismatched={bad}  skipped={skipped}")
    for f in failures[:10]:
        print(f"    {f}")
    if args.control:
        if bad or ok == 0:
            print("\n  >>> CONTROL FAILED -- window arithmetic is wrong, "
                  "do NOT build the pack")
            sys.exit(1)
        print(f"\n  >>> CONTROL PASSED -- {ok} patches reproduce corpus "
              "slots 17-19 exactly")
    else:
        (args.out_dir / "MANIFEST.json").write_text(json.dumps(
            {"patches": ok, "source": str(SHADOW_V7),
             "world": "2018 LiDAR everywhere (dsm_1m)",
             "slots": {"0": "dsm_2018", "10": "shade_age (per sample hour)",
                       "17": "shadow_h09", "18": "shadow_h13",
                       "19": "shadow_h16"},
             "age_hours": list(AGE_HOURS), "size_px": PX}, indent=2))
        print(f"\n  wrote {ok} patch files + MANIFEST.json to {args.out_dir}")


if __name__ == "__main__":
    main()
