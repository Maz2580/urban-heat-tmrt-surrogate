#!/usr/bin/env python3
r"""SHADE AGE AUDIT -- is V5 blind to HOW LONG ground has been shaded?

SOLWEIG's shade T_mrt depends on thermal history: ground shaded since 09:00 is
several degrees cooler than ground a shadow reached minutes ago. The only
input channels that could carry that history are the three shadow snapshots --
and in V5's corpus they are the mirrored ones, which shadow_lie_audit.py
proved V5 learned to IGNORE. Hypothesis: V5 is history-blind, predicting only
the average shade temperature; the real spread around it is the +-5 C interior
blotchiness the team reads as "shadow captured in places, missed in others".

Method: stratify non-canopy pixels by their TRUE shading trajectory (the
corrected out_v6 planes at 09/13/16 -> 8 patterns) x sample hour, and measure:

  mean LABEL   physics check -- long-shaded must be cooler than fresh-shaded
  V5 bias      prediction if history-blind: positive on long-shaded, negative
               on fresh-shaded, cancelling to ~0 overall (which is what every
               earlier per-class table saw)
  oracle       RMSE after removing per-(hour x pattern) bias = ceiling for a
               V7 whose ONLY change is correct multi-hour shadow planes on the
               original dsm_1m surface (single variable, unlike V6)

This structure is invisible to every partition tested so far (label bins,
exposure class, edges, scales) -- its ceiling is NOT bounded by their 0.217.

Read-only inference, test split, venv_train, everything in the working folder.

    & .\venv_train\Scripts\python.exe stage_d\shade_age_audit.py `
        --corpus stage_d\out --ckpt stage_d\models\v5_best.pt `
        --mask-corpus stage_d\out_v6 --tag v5
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

HOURS = (9, 11, 13, 15)
NPAT = 8                                    # (s09, s13, s16) bits
PAT_NAMES = {0b000: "never shaded", 0b100: "morning only", 0b010: "midday only",
             0b001: "late only", 0b110: "morning+midday", 0b011: "midday+late",
             0b101: "morning+late", 0b111: "shaded all day"}
V5_EXPECTED_RMSE = 4.1824
RMSE_TOL = 0.02


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--mask-corpus", type=Path, required=True)
    ap.add_argument("--tag", default="v5")
    ap.add_argument("--expected-rmse", type=float, default=V5_EXPECTED_RMSE,
                    help="recorded test RMSE of --ckpt (gate); default = V5")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-amp", action="store_true")
    args = ap.parse_args()

    try:
        import segmentation_models_pytorch as smp
    except ImportError:
        print("run in venv_train (inference only).", file=sys.stderr)
        return 2

    manifest = pd.read_parquet(args.corpus / "manifest.parquet")
    manifest = manifest[manifest["has_tmrt"]].reset_index(drop=True)
    manifest = T.build_splits(manifest)
    test = manifest[manifest.split == "test"].reset_index(drop=True)
    if args.limit:
        test = test.head(args.limit)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = (torch.bfloat16
                 if (device.type == "cuda" and not args.no_amp) else None)
    print(f"corpus : {args.corpus}   truth planes: {args.mask_corpus}")
    print(f"test   : {len(test):,} samples  device {device} "
          f"amp={'bf16' if amp_dtype else 'off'}")

    model = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                     in_channels=T.NUM_CHANNELS, classes=len(T.HEAD_NAMES),
                     activation=None)
    ck = torch.load(args.ckpt, map_location=device, weights_only=True)
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()

    loader = DataLoader(C.MaskSwapDataset(test, args.corpus, args.mask_corpus),
                        batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers,
                        pin_memory=(device.type == "cuda"))

    ncell = len(HOURS) * NPAT
    err = C.Sums(ncell, device)
    lab = C.Sums(ncell, device)
    overall = C.Sums(1, device)                 # all finite px, for the gate
    universe = C.Sums(1, device)                # non-canopy px, oracle base

    with torch.no_grad():
        for bi, batch in enumerate(loader):
            x = batch["input"].to(device, non_blocking=True)
            tgt = batch["tmrt"].to(device, non_blocking=True)
            msk = batch["mask_shadow"].to(device, non_blocking=True)
            if amp_dtype is not None:
                with torch.autocast(device_type=device.type, dtype=amp_dtype):
                    pred = model(x)
                pred = pred.float()
            else:
                pred = model(x)
            pred = pred[:, 0]

            B, H, W = tgt.shape
            s09 = msk[:, 0] > 0.5
            s13 = msk[:, 1] > 0.5
            s16 = msk[:, 2] > 0.5
            pat = s09.long() * 4 + s13.long() * 2 + s16.long()

            hour_i = torch.tensor(
                [HOURS.index(int(h)) for h in batch["hour"].tolist()],
                device=device).view(B, 1, 1).expand(B, H, W)

            canopy = x[:, C.CH_CDSM] > C.CANOPY_M
            finite = torch.isfinite(tgt)
            base = finite & ~canopy

            d = pred - tgt
            overall.add(torch.zeros_like(tgt[finite], dtype=torch.long),
                        d[finite])
            universe.add(torch.zeros_like(tgt[base], dtype=torch.long),
                         d[base])
            idx = (hour_i * NPAT + pat)[base]
            err.add(idx, d[base])
            lab.add(idx, tgt[base])

            if bi % 20 == 0:
                print(f"  [{bi * args.batch_size:>5}/{len(test)}]", flush=True)

    n0, _, d20 = overall.cpu()
    base_rmse = math.sqrt(d20.sum() / n0.sum())
    gate = "PASS" if abs(base_rmse - args.expected_rmse) <= RMSE_TOL else "FAIL"
    if args.limit:
        gate = "SKIPPED (smoke)"
    print()
    print("=" * 76)
    print(f"GATE : pixel RMSE {base_rmse:.4f} C vs recorded "
          f"{args.expected_rmse} -> {gate}")
    print("=" * 76)

    en, ed, ed2 = err.cpu()
    ln_, ls, _ = lab.cpu()
    j = {"gate": gate, "cells": {}}
    for hi, h in enumerate(HOURS):
        print(f"\nHOUR {h:02d}  (non-canopy; pattern = truly shaded at "
              "09/13/16)")
        print(f"  {'trajectory':<18}{'pat':>5}{'share':>8}{'label':>8}"
              f"{'bias':>8}{'rmse':>8}")
        htot = en[hi * NPAT:(hi + 1) * NPAT].sum()
        for p in range(NPAT):
            k = hi * NPAT + p
            if en[k] < 50_000:              # skip statistically empty cells
                continue
            r = {"pixels": int(en[k]),
                 "share_pct": 100.0 * float(en[k] / htot),
                 "mean_label_C": float(ls[k] / ln_[k]),
                 "bias_C": float(ed[k] / en[k]),
                 "rmse_C": math.sqrt(float(ed2[k] / en[k]))}
            j["cells"][f"h{h:02d}_{p:03b}"] = r
            nm = PAT_NAMES.get(p, f"{p:03b}")
            print(f"  {nm:<18}{format(p, '03b'):>5}{r['share_pct']:>7.1f}%"
                  f"{r['mean_label_C']:>8.2f}{r['bias_C']:>+8.2f}"
                  f"{r['rmse_C']:>8.2f}")

    nu, _, du2 = universe.cpu()
    Nu = float(nu.sum())
    u_rmse = math.sqrt(du2.sum() / Nu)
    u_after = math.sqrt(C.sse_after_own_bias(en, ed, ed2) / Nu)
    j["universe_rmse"] = u_rmse
    j["universe_rmse_after_pattern_bias"] = u_after
    print()
    print("ORACLE (non-canopy universe): remove per-(hour x trajectory) bias")
    print(f"  {u_rmse:.3f} -> {u_after:.3f} C   (gain {u_rmse - u_after:+.3f})")
    print()
    print("HOW TO READ")
    print("  Labels: 'shaded all day' should be several C cooler than a fresh")
    print("  trajectory at the same hour. If V5's bias is + on long-shaded and")
    print("  - on fresh-shaded, it predicts the AVERAGE shade temperature --")
    print("  history-blind, and correct multi-hour planes are the missing")
    print("  input. If bias is flat across trajectories, history is already")
    print("  inferred (or irrelevant) and the planes are NOT the blocker.")

    out_dir = args.out_dir or args.ckpt.parent
    out = out_dir / f"{args.tag}_shade_age.json"
    out.write_text(json.dumps(j, indent=1))
    print(f"\nwrote {out}")
    return 0 if gate != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
