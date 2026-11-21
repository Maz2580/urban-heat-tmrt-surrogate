#!/usr/bin/env python3
"""
Verify a Stage D corpus.
========================

Runs everywhere. Checks that need the V4 corpus (Linux only) are skipped with a
SKIP marker rather than silently dropped -- a verification run that quietly does
less on one machine than another is how two corpora diverge unnoticed.

    python verify_stage_d.py --dir <stage_d out> [--v4-dir <patches_v4>]
                             [--met <met_day.txt>] [--sample N]

Every check is written so it CAN fail. Two are adversarial on purpose: the
weather check asserts the value matches the LOCAL hour *and* differs from the
corpus hour, so a wrong clock trips it instead of silently agreeing.

Read-only.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np

STATIC_SLOTS = (0, 1, 2, 5, 6, 7, 8, 9, 15, 16, 17, 18, 19, 20, 21, 22)
DOY_NO_LST = 351

fails: list[str] = []
skips: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  -- ' + detail) if detail else ''}")
    if not ok:
        fails.append(name)


def skip(name: str, why: str) -> None:
    print(f"  [SKIP] {name}  -- {why}")
    skips.append(name)


def load_met(path: Path) -> dict[tuple[int, int], tuple[float, ...]]:
    met = {}
    for line in path.read_text().splitlines()[1:]:
        f = line.split()
        if len(f) > 14:
            met[(int(f[1]), int(f[2]))] = (float(f[11]), float(f[10]),
                                           float(f[9]), float(f[14]))
    return met


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--v4-dir", type=Path, default=None)
    ap.add_argument("--met", type=Path, default=None)
    ap.add_argument("--sample", type=int, default=200)
    args = ap.parse_args()

    files = sorted(args.dir.glob("*.npz"))
    if not files:
        print(f"no npz in {args.dir}")
        return 2
    print(f"Stage D corpus: {len(files)} files in {args.dir}\n")

    # locate a met file if not given: Stage C wrote one beside every patch
    met = None
    if args.met and args.met.exists():
        met = load_met(args.met)
    else:
        for cand in (args.dir.parent / "out").glob("*/met_day.txt"):
            met = load_met(cand)
            print(f"(using met file {cand})\n")
            break

    print("=" * 70)
    print("A. structure and physics - one representative sample")
    print("=" * 70)
    ref = next((f for f in files if f.stem.endswith("d015_h09")), files[0])
    n = np.load(ref)
    print(f"  file: {ref.name}")
    print(f"  keys: {sorted(n.files)}\n")

    for k in ("input", "tmrt", "lst", "valid", "lst_valid", "doy",
              "corpus_hour", "local_hour", "solar_altitude"):
        check(f"key '{k}' present", k in n.files)

    check("input is (23, 512, 512)", n["input"].shape == (23, 512, 512),
          str(n["input"].shape))

    alt = float(n["solar_altitude"])
    check("solar altitude is daytime", 0 < alt < 90, f"{alt:.2f} deg")
    check("ch3 sin_alt == sin(altitude)",
          abs(float(n["input"][3].mean()) - np.sin(np.radians(alt))) < 1e-5,
          f"ch3={n['input'][3].mean():.4f}")
    check("ch4 alt_norm == altitude/90",
          abs(float(n["input"][4].mean()) - alt / 90.0) < 1e-5)
    check("ch3 is NOT zero (the V3/V4 defect)", n["input"][3].mean() > 0)

    check("local_hour == corpus_hour + 1 (SOLWEIG timezone)",
          int(n["local_hour"]) == int(n["corpus_hour"]) + 1,
          f"{int(n['corpus_hour'])} -> {int(n['local_hour'])}")

    v = n["valid"].astype(bool)
    check("tmrt finite exactly where valid",
          np.array_equal(v, np.isfinite(n["tmrt"])),
          f"valid {100*v.mean():.2f}%")
    check("no mean-fill: invalid pixels are NaN",
          (~np.isfinite(n["tmrt"][~v])).all() if (~v).any() else True)

    if met:
        doy, ch, lh = int(n["doy"]), int(n["corpus_hour"]), int(n["local_hour"])
        tair_local = met[(doy, lh)][0]
        tair_corpus = met[(doy, ch)][0]
        check("era5_tair == met at LOCAL hour",
              abs(float(n["input"][11].mean()) - tair_local) < 0.01,
              f"{n['input'][11].mean():.2f} vs {tair_local}")
        check("...and DIFFERS from met at corpus hour (check discriminates)",
              abs(tair_local - tair_corpus) > 0.5,
              f"local {tair_local} vs corpus-hour {tair_corpus}")
    else:
        skip("weather clock checks", "no met file found; pass --met")

    print()
    print("=" * 70)
    print("B. the heat date - DOY 351")
    print("=" * 70)
    heat = [f for f in files if "_d351_" in f.name]
    if not heat:
        skip("heat-date checks", "no DOY 351 samples present")
    else:
        h = np.load(heat[len(heat) // 2])
        print(f"  file: {heat[len(heat)//2].name}")
        check("lst fully masked", not np.isfinite(h["lst"]).any())
        check("lst_valid == 0", int(h["lst_valid"]) == 0)
        check("tmrt still masked by validity",
              np.array_equal(h["valid"].astype(bool), np.isfinite(h["tmrt"])))
        check("heat-date label is hot", np.nanmean(h["tmrt"]) > 40,
              f"mean {np.nanmean(h['tmrt']):.2f} C")

    print()
    print("=" * 70)
    print(f"C. corpus-wide - {min(args.sample, len(files))} sampled files")
    print("=" * 70)
    random.seed(0)
    sample = random.sample(files, min(args.sample, len(files)))
    bad_solar = bad_mask = bad_lst = bad_shape = 0
    tmrts = []
    for f in sample:
        d = np.load(f)
        if d["input"].shape != (23, 512, 512):
            bad_shape += 1
            continue
        if float(d["solar_altitude"]) > 0 and d["input"][3].mean() <= 0:
            bad_solar += 1
        if not np.array_equal(d["valid"].astype(bool), np.isfinite(d["tmrt"])):
            bad_mask += 1
        if int(d["doy"]) == DOY_NO_LST and np.isfinite(d["lst"]).any():
            bad_lst += 1
        tmrts.append(float(np.nanmean(d["tmrt"])))

    check("every sample has the right shape", bad_shape == 0, f"{bad_shape} bad")
    check("solar channel populated in every daylight sample", bad_solar == 0,
          f"{bad_solar} bad")
    check("valid mask consistent everywhere", bad_mask == 0, f"{bad_mask} bad")
    check("lst masked in every DOY 351 sample", bad_lst == 0, f"{bad_lst} bad")
    if tmrts:
        a = np.array(tmrts)
        print(f"\n  label T_mrt: mean {a.mean():.2f} C  min {a.min():.2f}  "
              f"max {a.max():.2f}")
        check("labels are in a physical range", 5 < a.min() and a.max() < 95)

    print()
    print("=" * 70)
    print("D. comparison against V4 (Linux only)")
    print("=" * 70)
    if not args.v4_dir or not args.v4_dir.is_dir():
        skip("V4 static-channel equality", "no --v4-dir on this machine")
        skip("V4 label-delta", "no --v4-dir on this machine")
    else:
        deltas, static_bad, pairs = [], 0, 0
        for f in sample:
            o = args.v4_dir / f.name
            if not o.exists():
                continue
            pairs += 1
            d, ov = np.load(f), np.load(o)
            deltas.append(float(np.nanmean(d["tmrt"]) - np.nanmean(ov["tmrt"])))
            if any(not np.array_equal(d["input"][c], ov["input"][c])
                   for c in STATIC_SLOTS):
                static_bad += 1
        check("16 static channels identical to V4", static_bad == 0,
              f"{static_bad} of {pairs} differ")
        if deltas:
            a = np.array(deltas)
            print(f"\n  label change vs V4 over {len(a)} pairs:")
            print(f"    mean {a.mean():+.2f} C  median {np.median(a):+.2f} C  "
                  f"min {a.min():+.2f}  max {a.max():+.2f}")
            check("labels moved substantially warmer", a.mean() > 10)

    print()
    print("=" * 70)
    if fails:
        print(f"{len(fails)} FAILED, {len(skips)} skipped")
        for f in fails:
            print(f"  FAIL {f}")
        return 1
    print(f"ALL CHECKS PASSED ({len(skips)} skipped)")
    for s in skips:
        print(f"  skipped: {s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
