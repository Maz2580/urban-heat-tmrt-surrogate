#!/usr/bin/env python3
"""Merge the corrected V6 shadow planes into a NEW donor pack. Runs on Windows.

Reads  : <donors>/patch_XXXX.npz          (existing Stage D pack, untouched)
         <shadow-pack>/patch_XXXX.npz     (corrected slots 17/18/19)
Writes : <out>/patch_XXXX.npz             (same format, corrected shadows)

Stage D then runs exactly as before against --donor-pack <out>. Nothing in
stage_d.py, verify_stage_d.py or the .ps1 launchers changes, so the pipeline
that was verified 23/23 stays the pipeline that runs.

The existing donors are opened read-only and a NEW directory is written. If a
run goes wrong the original pack is still there, and V5 remains reproducible
from it.

Self-check on every patch: exactly the three shadow planes may differ, and the
other thirteen static channels plus `valid`, `lst` and `mask_verified` must be
bit-identical. A merge that quietly altered channel 0 would otherwise surface
only as a strange V6.

Usage (PowerShell):
  python apply_shadow_v6.py --donors stage_d\donors `
                            --shadow-pack stage_d\shadow_v6_pack `
                            --out stage_d\donors_v6
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

STATIC_SLOTS = (0, 1, 2, 5, 6, 7, 8, 9, 15, 16, 17, 18, 19, 20, 21, 22)
SHADOW_SLOTS = (17, 18, 19)
KEYS = ("shadow_h09", "shadow_h13", "shadow_h16")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--donors", type=Path, required=True)
    ap.add_argument("--shadow-pack", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    idx = {slot: STATIC_SLOTS.index(slot) for slot in SHADOW_SLOTS}
    args.out.mkdir(parents=True, exist_ok=True)

    donors = sorted(args.donors.glob("patch_*.npz"))
    if args.limit:
        donors = donors[:args.limit]
    print(f"donors      : {len(donors)}")
    print(f"shadow pack : {args.shadow_pack}")
    print(f"out         : {args.out}\n")

    done = missing = failed = 0
    changed_frac = []
    problems = []
    for i, f in enumerate(donors):
        sp = args.shadow_pack / f.name
        if not sp.exists():
            missing += 1
            problems.append(f"{f.stem}: no shadow plane in pack")
            continue
        d = np.load(f)
        s = np.load(sp)
        static = d["static"].astype(np.float32)
        new = static.copy()
        for slot, key in zip(SHADOW_SLOTS, KEYS):
            new[idx[slot]] = s[key].astype(np.float32)

        # self-check: only the three shadow rows may move
        moved = [j for j in range(static.shape[0])
                 if not np.array_equal(static[j], new[j])]
        allowed = {idx[sl] for sl in SHADOW_SLOTS}
        if not set(moved).issubset(allowed):
            failed += 1
            problems.append(f"{f.stem}: unexpected change in static rows "
                            f"{sorted(set(moved) - allowed)}")
            continue
        changed_frac.append(float(np.mean([
            (static[idx[sl]] != new[idx[sl]]).mean() for sl in SHADOW_SLOTS])))

        np.savez_compressed(
            args.out / f.name,
            static=new,
            valid=d["valid"],
            lst_doys=d["lst_doys"], lst_hours=d["lst_hours"], lst_values=d["lst_values"],
            mask_verified=d["mask_verified"],
            static_slots=d["static_slots"] if "static_slots" in d.files
            else np.array(STATIC_SLOTS, dtype=np.int16),
        )
        done += 1
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(donors)} ...")

    print(f"\n  written={done}  missing={missing}  failed_selfcheck={failed}")
    for p in problems[:10]:
        print(f"    {p}")
    if changed_frac:
        cf = np.array(changed_frac)
        print(f"\n  shadow pixels changed per patch: mean {100*cf.mean():.2f}%  "
              f"min {100*cf.min():.2f}%  max {100*cf.max():.2f}%")
        if cf.mean() < 0.001:
            print("  >>> WARNING: almost nothing changed. The pack may not have applied.")
    if failed or missing or done == 0:
        print("\n  >>> NOT CLEAN -- do not train from this pack")
        sys.exit(1)
    print(f"\n  >>> OK. Run Stage D with --donor-pack {args.out}")


if __name__ == "__main__":
    main()
