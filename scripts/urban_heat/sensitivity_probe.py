#!/usr/bin/env python3
r"""SENSITIVITY / ABLATION PROBE -- does the model USE ch10 (shade age) and
ch17-19 (shadow planes), or has it learned to ignore them (the V6 failure)?

V7_PLAN.md section 4 step 2: "perturbing ch10/ch17-19 should move the output;
a V7 that ignores its history channels has failed the same way V6 did."

Two kinds of perturbation, test split only, read-only inference:

  ABLATION (accuracy): zero a channel group and re-score RMSE overall and in
      TRUE building shade (own nearest-hour plane, non-canopy). If RMSE rises,
      the channel is load-bearing. This is the clean version of "is it used".
  RESPONSE (direction): force planes to 1 / add +2 h age in shaded pixels and
      report mean delta-prediction in shade. Comparable with the historical
      "planes forced to 1" probe (V5 -15.37, V6 -1.66).

Gate: baseline RMSE must reproduce --expected-rmse (same rule as ceiling_test).

    & .\venv_train\Scripts\python.exe stage_d\sensitivity_probe.py `
        --corpus stage_d\out_v7 --ckpt stage_d\models\v7_best.pt `
        --expected-rmse 3.758 --tag v7
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_v5 as T                       # noqa: E402
import ceiling_test as C                   # noqa: E402

CH_AGE = T.CHANNEL_ORDER.index("shadow_fraction")   # slot 10, V7 = age hours

PERTURBATIONS = {
    "baseline":        lambda x, sh: x,
    "age := 0":        lambda x, sh: _set(x, CH_AGE, 0.0),
    "age += 2h (shaded)": lambda x, sh: _add_where(x, CH_AGE, 2.0, sh),
    "age := 8h (shaded)": lambda x, sh: _set_where(x, CH_AGE, 8.0, sh),
    "planes := 0":     lambda x, sh: _set(x, C.SHADOW_IDXS, 0.0),
    "planes := 1":     lambda x, sh: _set(x, C.SHADOW_IDXS, 1.0),
    "age:=0 & planes:=0": lambda x, sh: _set(_set(x, CH_AGE, 0.0), C.SHADOW_IDXS, 0.0),
}


def _set(x, idx, v):
    y = x.clone(); y[:, idx] = v; return y


def _set_where(x, ch, v, m):
    y = x.clone(); y[:, ch][m] = v; return y


def _add_where(x, ch, v, m):
    y = x.clone(); y[:, ch][m] += v; return y


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--expected-rmse", type=float, required=True)
    ap.add_argument("--tag", default="v7")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-amp", action="store_true")
    args = ap.parse_args()
    out_dir = args.out_dir or args.ckpt.parent

    import pandas as pd
    import segmentation_models_pytorch as smp
    manifest = pd.read_parquet(args.corpus / "manifest.parquet")
    manifest = manifest[manifest["has_tmrt"]].reset_index(drop=True)
    manifest = T.build_splits(manifest)               # SPLIT_SEED = 42
    test = manifest[manifest.split == "test"].reset_index(drop=True)
    if args.limit:
        test = test.head(args.limit)
        print("!! --limit set: SMOKE RUN")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = torch.bfloat16 if (device.type == "cuda" and not args.no_amp) else None
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                       in_channels=T.NUM_CHANNELS, classes=len(T.HEAD_NAMES),
                       activation=None)
    ck = torch.load(args.ckpt, map_location=device, weights_only=True)
    model.load_state_dict(ck["state_dict"]); model.to(device).eval()
    print(f"corpus : {args.corpus}\nckpt   : {args.ckpt}\ntest   : {len(test)} "
          f"samples  device {device} amp={'bf16' if amp else 'off'}")

    loader = DataLoader(T.V5PatchDataset(test, args.corpus),
                        batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers,
                        pin_memory=(device.type == "cuda"))
    names = list(PERTURBATIONS)
    # per perturbation: [n, sse] overall, [n, sse, sum_delta] in shade
    acc = {k: torch.zeros(5, dtype=torch.float64, device=device) for k in names}
    age_seen = 0.0

    def fwd(x):
        with torch.no_grad():
            if amp is not None:
                with torch.autocast(device_type=device.type, dtype=amp):
                    p = model(x)
                return p.float()[:, 0]
            return model(x)[:, 0]

    for bi, batch in enumerate(loader):
        x = batch["input"].to(device, non_blocking=True)
        tgt = batch["tmrt"].to(device, non_blocking=True)
        age_seen = max(age_seen, float(x[:, CH_AGE].max()))
        canopy = x[:, C.CH_CDSM] > C.CANOPY_M
        sh = torch.zeros_like(canopy)
        for i, h in enumerate(batch["hour"].tolist()):
            sh[i] = x[i, C.CH_SHADOW[C.nearest_shadow_hour(int(h))]] > 0.5
        shade = sh & ~canopy & torch.isfinite(tgt)
        finite = torch.isfinite(tgt)
        base = fwd(x)
        for k in names:
            p = base if k == "baseline" else fwd(PERTURBATIONS[k](x, sh))
            d = p - tgt
            a = acc[k]
            a[0] += finite.sum(); a[1] += (d[finite] ** 2).sum()
            a[2] += shade.sum(); a[3] += (d[shade] ** 2).sum()
            a[4] += (p - base)[shade].sum()
        if bi % 20 == 0:
            print(f"  [{bi*args.batch_size:5d}/{len(test)}]")

    b = acc["baseline"]
    base_rmse = math.sqrt(b[1] / b[0])
    gate = "PASS" if abs(base_rmse - args.expected_rmse) <= C.RMSE_TOL else "FAIL"
    print("\n" + "=" * 76)
    print(f"GATE : RMSE {base_rmse:.4f} C vs recorded {args.expected_rmse} -> {gate}")
    print("=" * 76)
    print(f"ch10 max seen in corpus: {age_seen:.2f}  "
          f"({'AGE hours -- V7' if age_seen > 1.5 else 'ZERO/flag -- not an age corpus'})")
    print(f"\n{'perturbation':22s} {'RMSE all':>9s} {'dRMSE':>7s} "
          f"{'RMSE shade':>11s} {'dRMSE':>7s} {'mean dPred shade':>17s}")
    rows = {}
    bs = math.sqrt(b[3] / b[2])
    for k in names:
        a = acc[k]
        r_all = math.sqrt(a[1] / a[0]); r_sh = math.sqrt(a[3] / a[2])
        dp = float(a[4] / a[2])
        rows[k] = {"rmse_all": r_all, "rmse_shade": r_sh, "mean_dpred_shade": dp}
        print(f"{k:22s} {r_all:9.3f} {r_all-base_rmse:+7.3f} "
              f"{r_sh:11.3f} {r_sh-bs:+7.3f} {dp:+17.2f}")
    print("\nHOW TO READ")
    print("  dRMSE > 0 on ablation  => channel is LOAD-BEARING (model uses it)")
    print("  dRMSE ~ 0 and dPred ~ 0 => channel IGNORED (the V6 failure mode)")
    print("  'age += 2h' dPred < 0  => model cools with longer shade (physics)")
    out = out_dir / f"{args.tag}_sensitivity.json"
    out.write_text(json.dumps({"tag": args.tag, "ckpt": str(args.ckpt),
                               "corpus": str(args.corpus), "gate": gate,
                               "baseline_rmse": base_rmse, "rows": rows},
                              indent=2))
    print(f"\nwrote {out}")
    return 0 if gate == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
