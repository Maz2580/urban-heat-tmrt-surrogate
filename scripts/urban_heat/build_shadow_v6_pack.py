#!/usr/bin/env python3
"""Crop the corrected V6 shadow rasters into per-patch planes for Stage D.

The Stage D donor pack carries 16 pre-cropped static channels per patch.
Only slots 17/18/19 (shadow_h09/h13/h16) change for V6, so rather than
rebuilding and re-uploading the whole ~2 GB pack this emits just the three
corrected planes per patch. They are binary, so they compress to a fraction
of that. `apply_shadow_v6.py` on the workstation merges them into a NEW
`donors_v6/` directory -- the existing donors are never mutated.

--control is the acceptance gate. It crops the OLD rasters through the exact
same window arithmetic and compares against slots 17-19 already sitting in the
V4 corpus. If those do not match bit-for-bit, the window mapping is wrong and
the corrected crops would be silently misaligned -- which would look like a
model problem later, not a data problem.

Reads the corpus read-only. Writes only to --out-dir.
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
SHADOW_V6 = DATA / "v4_prep/shadow_multi_v6"
MASTER = DATA / "rasters/raw/dsm_1m.tif"

HOURS = (9, 13, 16)
SLOTS = {9: 17, 13: 18, 16: 19}
PX = 512


def patch_window(pid: str, master_transform) -> tuple[int, int] | None:
    """Top-left (row, col) of this patch in the 9000x9000 master grid."""
    f = PATCH_DIR / pid / "building_dsm.tif"
    if not f.exists():
        return None
    with rasterio.open(f) as s:
        t = s.transform
    col = (t.c - master_transform.c) / master_transform.a
    row = (t.f - master_transform.f) / master_transform.e
    return int(round(row)), int(round(col))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch-list", type=Path, required=True,
                    help="text file, one patch id per line (the 482 donors)")
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--control", action="store_true",
                    help="verify window arithmetic against the corpus, write nothing")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    pids = [l.strip() for l in args.patch_list.read_text().splitlines() if l.strip()]
    if args.limit:
        pids = pids[:args.limit]
    src_dir = SHADOW_OLD if args.control else SHADOW_V6
    with rasterio.open(MASTER) as s:
        mt = s.transform
    src = {h: rasterio.open(src_dir / f"shadow_h{h:02d}.tif") for h in HOURS}

    print(f"mode     : {'CONTROL (old rasters vs corpus)' if args.control else 'BUILD V6 pack'}")
    print(f"source   : {src_dir}")
    print(f"patches  : {len(pids)}\n")

    if not args.control:
        args.out_dir.mkdir(parents=True, exist_ok=True)

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
        planes = {}
        for h in HOURS:
            a = src[h].read(1, window=win)
            if a.shape != (PX, PX):
                a = np.zeros((PX, PX), dtype=np.float32)
            planes[h] = a.astype(np.float32)

        if args.control:
            ref_files = sorted(V4_DIR.glob(f"{pid}_*.npz"))
            if not ref_files:
                skipped += 1
                continue
            x = np.load(ref_files[0])["input"]
            mism = [h for h in HOURS if not np.array_equal(planes[h], x[SLOTS[h]])]
            if mism:
                bad += 1
                d = max(float(np.abs(planes[h] - x[SLOTS[h]]).max()) for h in mism)
                failures.append(f"{pid}: slots {[SLOTS[h] for h in mism]} differ, max {d:g}")
            else:
                ok += 1
        else:
            np.savez_compressed(
                args.out_dir / f"{pid}.npz",
                shadow_h09=planes[9].astype(np.uint8),
                shadow_h13=planes[13].astype(np.uint8),
                shadow_h16=planes[16].astype(np.uint8),
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
            print("\n  >>> CONTROL FAILED -- window arithmetic is wrong, do NOT build the pack")
            sys.exit(1)
        print(f"\n  >>> CONTROL PASSED -- {ok} patches reproduce corpus slots 17-19 exactly")
    else:
        (args.out_dir / "MANIFEST.json").write_text(json.dumps(
            {"patches": ok, "source": str(SHADOW_V6), "hours": list(HOURS),
             "slots": {str(h): SLOTS[h] for h in HOURS},
             "dtype": "uint8 (binary shadow)", "size_px": PX}, indent=2))
        print(f"\n  wrote {ok} patch files + MANIFEST.json to {args.out_dir}")


if __name__ == "__main__":
    main()
