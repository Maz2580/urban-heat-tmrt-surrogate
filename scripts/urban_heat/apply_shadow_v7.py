#!/usr/bin/env python3
r"""Merge the V7 pack into a NEW donor set: one consistent 2018 world.

Reads  : <donors>/patch_XXXX.npz        (existing Stage D pack, untouched)
         <v7-pack>/patch_XXXX.npz       (dsm_2018 + corrected planes + age)
Writes : <out>/patch_XXXX.npz           (slots 0/17/18/19 swapped, four
                                         shade_age_hXX planes carried along)

Stage D then runs with --donor-pack <out> --shadow-channel age, which places
the sample-hour age plane into slot 10 (the dead, zero-served channel).

Self-check per patch: exactly rows {0, 17, 18, 19} of `static` may differ;
everything else must be bit-identical (a merge that quietly altered svf
would surface only as a strange V7).

Usage (workstation, plain venv -- numpy only):
  python stage_d\apply_shadow_v7.py --donors stage_d\donors `
      --v7-pack stage_d\shadow_v7_pack --out stage_d\donors_v7
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

STATIC_SLOTS = (0, 1, 2, 5, 6, 7, 8, 9, 15, 16, 17, 18, 19, 20, 21, 22)
SWAP = {0: "dsm_2018", 17: "shadow_h09", 18: "shadow_h13", 19: "shadow_h16"}
AGE_KEYS = ("shade_age_h09", "shade_age_h11", "shade_age_h13", "shade_age_h15")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--donors", type=Path, required=True)
    ap.add_argument("--v7-pack", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    idx = {slot: STATIC_SLOTS.index(slot) for slot in SWAP}
    args.out.mkdir(parents=True, exist_ok=True)

    donors = sorted(args.donors.glob("patch_*.npz"))
    if args.limit:
        donors = donors[:args.limit]
    print(f"donors  : {len(donors)}")
    print(f"v7 pack : {args.v7_pack}")
    print(f"out     : {args.out}\n")

    done = missing = failed = 0
    ch0_changed = []
    sh_changed = []
    problems = []
    for i, f in enumerate(donors):
        sp = args.v7_pack / f.name
        if not sp.exists():
            missing += 1
            problems.append(f"{f.stem}: not in v7 pack")
            continue
        d = np.load(f)
        s = np.load(sp)
        if any(k not in s.files for k in AGE_KEYS):
            failed += 1
            problems.append(f"{f.stem}: pack missing age planes")
            continue
        static = d["static"].astype(np.float32)
        new = static.copy()
        for slot, key in SWAP.items():
            new[idx[slot]] = s[key].astype(np.float32)

        moved = [j for j in range(static.shape[0])
                 if not np.array_equal(static[j], new[j])]
        allowed = set(idx.values())
        if not set(moved).issubset(allowed):
            failed += 1
            problems.append(f"{f.stem}: unexpected change in static rows "
                            f"{sorted(set(moved) - allowed)}")
            continue
        ch0_changed.append(float((static[idx[0]] != new[idx[0]]).mean()))
        sh_changed.append(float(np.mean(
            [(static[idx[sl]] != new[idx[sl]]).mean() for sl in (17, 18, 19)])))

        np.savez_compressed(
            args.out / f.name,
            static=new,
            valid=d["valid"],
            lst_doys=d["lst_doys"], lst_hours=d["lst_hours"],
            lst_values=d["lst_values"],
            mask_verified=d["mask_verified"],
            static_slots=d["static_slots"] if "static_slots" in d.files
            else np.array(STATIC_SLOTS, dtype=np.int16),
            **{k: s[k].astype(np.uint8) for k in AGE_KEYS},
        )
        done += 1
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(donors)} ...")

    print(f"\n  written={done}  missing={missing}  failed_selfcheck={failed}")
    if ch0_changed:
        print(f"  ch0 (geoscape->dsm2018) changed: mean "
              f"{100*np.mean(ch0_changed):.2f}%  min {100*min(ch0_changed):.2f}%"
              f"  max {100*max(ch0_changed):.2f}%")
        print(f"  ch17-19 changed:                mean "
              f"{100*np.mean(sh_changed):.2f}%")
    for p in problems[:10]:
        print(f"    {p}")


if __name__ == "__main__":
    main()
