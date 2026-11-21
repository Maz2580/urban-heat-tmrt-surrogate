#!/usr/bin/env python3
"""
Build the portable Stage D donor pack.
=====================================

Stage D needs three things from this server that the Windows box does not have:

  * the 16 STATIC input channels (dsm, cdsm, albedo, svf, building_density,
    tree_distance, aspect x2, ndvi, solar_panel_mask, shadow_h09/13/16,
    roof_type x3)
  * a VALIDITY MASK derived from the input DSM
  * the per-(patch, doy, hour) `lst` scalar

Shipping the V4 corpus would mean 44 GB. But the 16 static channels are
IDENTICAL across all 12 date-hours of a patch (verified: 6 patches, 11
comparisons each, 0 mismatches), so one donor per patch is enough -- roughly a
12x reduction to ~1.3 GB.

THE MASK IS CROSS-CHECKED HERE, NOT THERE. V4 exists on this machine, so this is
the only place the mask can be validated against V4's own implicit NaN mask.
Each donor records `mask_verified`; stage_d.py REFUSES to process a patch whose
donor is not stamped. That way a mask that was never checked cannot quietly
become a training corpus on a machine that had no way to check it.

Read-only with respect to the corpus. Writes only into --out-dir.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rasterio

DATA = Path("/data") if Path("/data/training").exists() else Path("/app/urban_heat_data")
V4_DIR = DATA / "training/patches_v4"
PATCH_DIR = DATA / "training/patches"

STATIC_SLOTS = (0, 1, 2, 5, 6, 7, 8, 9, 15, 16, 17, 18, 19, 20, 21, 22)
PATCH_PX = 512


def build_one(patch: str, out_dir: Path, strict_agreement: float,
              verify_static: bool) -> tuple[bool, str]:
    files = sorted(V4_DIR.glob(f"{patch}_d*.npz"))
    if not files:
        return False, "no V4 files"

    dsm_path = PATCH_DIR / patch / "building_dsm.tif"
    if not dsm_path.exists():
        return False, "no input DSM"
    with rasterio.open(dsm_path) as s:
        dsm = s.read(1)
    valid = dsm > -9000

    ref = np.load(files[0])
    static = np.stack([ref["input"][c] for c in STATIC_SLOTS]).astype(np.float32)

    # Confirm the assumption that makes the pack small, per patch rather than
    # trusting the six-patch sample. Cheap relative to the I/O already happening.
    if verify_static:
        for f in files[1:]:
            x = np.load(f)["input"]
            for i, c in enumerate(STATIC_SLOTS):
                if not np.array_equal(static[i], x[c]):
                    return False, f"static channel {c} differs in {f.name}"

    # Cross-check the mask against V4's own finite mask on every date-hour.
    #
    # DIRECTIONAL on purpose. The two ways to disagree are not equally serious:
    #
    #   v4_only  = V4 kept the pixel, we drop it.  Our mask is TOO STRICT --
    #              a structural error (wrong threshold, wrong raster, a buffer
    #              applied by mistake). Never acceptable.
    #   ours_only= V4 dropped it, we keep it.  BENIGN: the input DSM is present,
    #              so the old engine wrote NaN there for its own reasons. The new
    #              engine computes a real value and we should keep it.
    #
    # Measured on the 5 patches that failed a symmetric check: v4_only was 0 in
    # every case, ours_only was 1-84 px of 262,144. A symmetric threshold would
    # have forced us to loosen tolerance in BOTH directions to accept those.
    doys, hours, lsts = [], [], []
    worst = 1.0
    max_v4_only = max_ours_only = 0
    for f in files:
        d = np.load(f)
        v4_finite = np.isfinite(d["tmrt"])
        worst = min(worst, float((v4_finite == valid).mean()))
        max_v4_only = max(max_v4_only, int((v4_finite & ~valid).sum()))
        max_ours_only = max(max_ours_only, int((~v4_finite & valid).sum()))
        stem = f.stem                                   # patch_XXXX_dDDD_hHH
        doys.append(int(stem.split("_d")[1].split("_")[0]))
        hours.append(int(stem.split("_h")[1]))
        lsts.append(float(d["lst"].flat[0]))

    if max_v4_only > 0:
        return False, (f"mask DROPS {max_v4_only} px that V4 kept — too strict, "
                       f"this is a structural error")
    budget = int(strict_agreement * valid.size)
    if max_ours_only > budget:
        return False, (f"mask keeps {max_ours_only} px V4 dropped "
                       f"(budget {budget}) — more than isolated engine failures")

    np.savez_compressed(
        out_dir / f"{patch}.npz",
        static=static,
        valid=valid.astype(np.uint8),
        lst_doys=np.array(doys, dtype=np.int16),
        lst_hours=np.array(hours, dtype=np.int16),
        lst_values=np.array(lsts, dtype=np.float32),
        mask_verified=np.uint8(1),
        mask_agreement=np.float32(worst),
        mask_v4_only=np.int32(max_v4_only),
        mask_ours_only=np.int32(max_ours_only),
        static_slots=np.array(STATIC_SLOTS, dtype=np.int16),
    )
    return True, ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patch-list", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--strict-agreement", type=float, default=0.001,
                    help="budget, as a fraction of pixels, for the BENIGN "
                         "direction only (pixels V4 dropped that we keep). The "
                         "dangerous direction — dropping a pixel V4 kept — is "
                         "always rejected regardless of this value.")
    ap.add_argument("--no-verify-static", action="store_true",
                    help="skip the per-patch static-channel equality check "
                         "(faster; the assumption was sampled, not proven)")
    args = ap.parse_args()

    raw = json.loads(args.patch_list.read_text())
    patches = raw if isinstance(raw, list) else (
        raw.get("patches") or raw.get("valid_patches") or [])
    if args.limit:
        patches = patches[:args.limit]
    if not patches:
        print(f"no patches in {args.patch_list}", file=sys.stderr)
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"building donor pack for {len(patches)} patches -> {args.out_dir}")
    print(f"  mask check: reject ANY pixel dropped that V4 kept; "
          f"allow up to {args.strict_agreement*100:.3f}% kept that V4 dropped")
    print(f"  verify static channels: {not args.no_verify_static}\n")

    ok = 0
    failures: list[tuple[str, str]] = []
    for i, p in enumerate(patches, 1):
        try:
            good, why = build_one(p, args.out_dir, args.strict_agreement,
                                  not args.no_verify_static)
        except Exception as exc:                          # noqa: BLE001
            good, why = False, f"{type(exc).__name__}: {exc}"
        if good:
            ok += 1
        else:
            failures.append((p, why))
        if i % 50 == 0:
            print(f"  [{i}/{len(patches)}] ok {ok}, failed {len(failures)}")

    total = sum(f.stat().st_size for f in args.out_dir.glob("*.npz"))
    print("\n" + "=" * 62)
    print(f"donors written : {ok}/{len(patches)}")
    print(f"pack size      : {total/1e9:.2f} GB  "
          f"({total/max(ok,1)/1e6:.1f} MB per patch)")
    if failures:
        print(f"\nfailures ({len(failures)}), first 15:")
        for p, why in failures[:15]:
            print(f"  {p}: {why}")
    print("=" * 62)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
