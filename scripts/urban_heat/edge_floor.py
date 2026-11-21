#!/usr/bin/env python3
r"""LABEL SELF-SHIFT FLOOR -- is the shadow-boundary error reducible at all?

scatter_diag.py located V5's remaining error at shadow EDGES:

    <=1 px of a building-shadow edge   RMSE 5.11   (V5, corrected mask)
    2-4 px                             RMSE 3.98
    5-12 px                            RMSE 3.33
    >12 px                             RMSE 3.62

SOLWEIG's shadows are binary at 1 m. Even a perfect regressor has a finite
edge width, so SOME boundary error is aliasing that no model can beat. This
script measures that floor WITHOUT ANY MODEL: the labels' RMS difference
against themselves shifted by 1 px and 2 px, in the same boundary bands.

Reading it (compare to the V5 numbers above):

  label 1-px self-shift ~ 5 C in the <=1 px band
      -> V5 already renders shadows to within ~1 px of perfect. The floor is
         rasterisation physics; a V7 cannot meaningfully improve it. Stop.
  label 1-px self-shift ~ 2-3 C
      -> V5's boundary error is 2 px+ worth of blur; real headroom exists and
         the lever is sharpness (edge-weighted loss / decoder), not data.

No checkpoint, no inference -- labels only. venv_train (for the shared
dataset code), test split, read-only, everything inside the working folder.

    & .\venv_train\Scripts\python.exe stage_d\edge_floor.py `
        --corpus stage_d\out --mask-corpus stage_d\out_v6 --tag v5
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_v5 as T                       # noqa: E402
import ceiling_test as C                   # noqa: E402
import scatter_diag as D                   # noqa: E402  (dilate, band names)

SHIFTS = (1, 2)
# V5's measured boundary RMSE (scatter_diag, corrected mask) for side-by-side
V5_REF = {"<=1": 5.11, "2-4": 3.98, "5-12": 3.33, ">12": 3.62}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--mask-corpus", type=Path, default=None,
                    help="out_v6 so the bands mean TRUE building shade")
    ap.add_argument("--tag", default="v5")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    manifest = pd.read_parquet(args.corpus / "manifest.parquet")
    manifest = manifest[manifest["has_tmrt"]].reset_index(drop=True)
    manifest = T.build_splits(manifest)
    test = manifest[manifest.split == "test"].reset_index(drop=True)
    if args.limit:
        test = test.head(args.limit)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"corpus  : {args.corpus}   mask: {args.mask_corpus or '(own ch17-19)'}")
    print(f"test    : {len(test):,} samples   device {device}   NO MODEL -- "
          "labels only")

    ds = (C.MaskSwapDataset(test, args.corpus, args.mask_corpus)
          if args.mask_corpus else T.V5PatchDataset(test, args.corpus))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers,
                        pin_memory=(device.type == "cuda"))

    nb = len(D.BND_NAMES)
    sums = {s: C.Sums(nb, device) for s in SHIFTS}

    with torch.no_grad():
        for bi, batch in enumerate(loader):
            x = batch["input"].to(device, non_blocking=True)
            tgt = batch["tmrt"].to(device, non_blocking=True)
            B, H, W = tgt.shape

            bshadow = torch.zeros((B, H, W), dtype=torch.bool, device=device)
            msk = batch.get("mask_shadow")
            if msk is not None:
                msk = msk.to(device, non_blocking=True)
            for i, h in enumerate(batch["hour"].tolist()):
                sh = C.nearest_shadow_hour(int(h))
                if msk is not None:
                    bshadow[i] = msk[i, C.SHADOW_HOURS.index(sh)] > 0.5
                else:
                    bshadow[i] = x[i, C.CH_SHADOW[sh]] > 0.5

            mf = bshadow.float()
            bnd = ((D.dilate(mf, 1) > 0.5) & (D.dilate(1.0 - mf, 1) > 0.5))
            bf = bnd.float()
            near = [D.dilate(bf, k) > 0.5 for k in D.BND_KS]
            band = torch.full((B, H, W), nb - 1, dtype=torch.long,
                              device=device)
            band[near[2]] = 2
            band[near[1]] = 1
            band[near[0]] = 0

            for s in SHIFTS:
                t0 = tgt[:, s:H - s, s:W - s]
                b0 = band[:, s:H - s, s:W - s]
                for dy, dx in ((s, 0), (-s, 0), (0, s), (0, -s)):
                    ts = tgt[:, s + dy:H - s + dy, s + dx:W - s + dx]
                    fin = torch.isfinite(t0) & torch.isfinite(ts)
                    sums[s].add(b0[fin], (ts - t0)[fin])

            if bi % 20 == 0:
                print(f"  [{bi * args.batch_size:>5}/{len(test)}]", flush=True)

    print()
    print("=" * 72)
    print("LABEL SELF-SHIFT RMS BY SHADE-BOUNDARY BAND  (no model involved)")
    print("=" * 72)
    print(f"  {'band':>6}{'1px floor':>12}{'2px floor':>12}{'V5 actual':>12}"
          f"{'verdict':>26}")
    j = {}
    for r, nm in enumerate(D.BND_NAMES):
        row = {}
        for s in SHIFTS:
            n, _, d2 = sums[s].cpu()
            row[f"rms_{s}px"] = (math.sqrt(float(d2[r] / n[r]))
                                 if n[r] else None)
        v5 = V5_REF.get(nm)
        f1 = row["rms_1px"]
        verdict = ""
        if f1 is not None and v5 is not None:
            if v5 <= f1 * 1.15:
                verdict = "at the 1px floor"
            elif v5 <= row["rms_2px"] * 1.15:
                verdict = "between 1-2px blur"
            else:
                verdict = "REAL HEADROOM"
        print(f"  {nm:>6}{row['rms_1px']:>12.2f}{row['rms_2px']:>12.2f}"
              f"{v5:>12.2f}{verdict:>26}")
        j[nm] = {**row, "v5_actual": v5, "verdict": verdict}

    print()
    print("  '1px floor' = RMS difference of the LABELS vs themselves moved")
    print("  1 px. A model whose edges are placed within ~1 px cannot score")
    print("  below this in that band. V5 close to the 1px column = shadows")
    print("  are already as sharp as the rasterisation allows.")

    out_dir = args.out_dir or args.corpus.parent
    out = out_dir / f"{args.tag}_edge_floor.json"
    out.write_text(json.dumps(j, indent=1))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
