#!/usr/bin/env python3
r"""CEILING TEST -- how much could ANY bias fix improve V5, without training?

WHY THIS EXISTS (memory.md section 30)
--------------------------------------------------------------------------
V6 taught us: proving an input is WRONG does not prove it is LOAD-BEARING.
Before the next retrain we measure the ceiling: perturb / correct the thing we
intend to fix ON THE MODEL WE ALREADY HAVE, and ask what the best possible
improvement is. If the answer is small, no retrain can help and we stop after
an hour instead of after days.

The V7 hypothesis is a LOSS problem, not an input problem:
  - V5 runs warm in building shade (under-cools the coolest places)
  - by-label-bin errors show range compression (+ in cool bins, - in hot bins)
Both are BIAS patterns. The best any loss reweighting can do against a bias
pattern is remove it. Removing a group's mean error reduces that group's sum of
squared errors by exactly n * bias^2 -- so from per-group running sums (n,
sum_d, sum_d2) we can compute post-correction RMSE ANALYTICALLY, in one
inference pass, with nothing stored per-pixel and nothing trained.

WHAT IT REPORTS
--------------------------------------------------------------------------
  baseline        RMSE as-is. MUST reproduce v5_test.json (4.182) or the whole
                  run is invalid (memory.md section 28).
  deployable      per-PREDICTION-bin bias fitted on the VAL split, applied to
                  the TEST split. This is an honest post-hoc calibration we
                  could ship TODAY with zero training.
  cal. ceiling    same correction fitted on test itself (optimistic bound for
                  any output calibration).
  oracle: label   per-LABEL-bin correction. Needs the answer, so not
                  deployable -- it is the upper bound for what tail/bin
                  reweighting in the LOSS could buy.
  oracle: class   per exposure class (canopy / building shade / sunlit).
  oracle: joint   class x label-bin. Upper bound for a shade-aware loss (V7).

Plus per-class tables (the decision is NOT the headline RMSE: building shade
may be a small pixel fraction, so a real shade fix can move overall RMSE very
little -- exactly how V6 was nearly misjudged).

PRE-REGISTERED DECISION RULE (written before seeing the numbers)
--------------------------------------------------------------------------
  RETRAIN V7 only if BOTH:
    (1) oracle:joint improves overall RMSE by >= 0.15 C, OR improves
        building_shaded RMSE by >= 1.0 C; and
    (2) building_shaded |bias| >= 2.0 C at baseline (there is a real target).
  If deployable calibration alone captures >= 70% of oracle:label, prefer
  shipping the calibration layer and skip training.

RUN (workstation, venv_train -- inference only, NOTHING trains, GPU makes it
minutes; everything read-only, output is one JSON next to the checkpoint):

    & $pyt stage_d\ceiling_test.py --corpus stage_d\out `
        --ckpt stage_d\models\v5_best.pt --tag v5

RUN IT ON THE CORPUS THE MODEL WAS TRAINED ON (stage_d\out for V5).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

# train_v5.py owns the split, the dataset and the channel order. Import, never
# reimplement (a drifted copy answers a different question with a plausible
# number).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_v5 as T  # noqa: E402

CANOPY_M = 3.0
CH_CDSM = T.CHANNEL_ORDER.index("cdsm")
SHADOW_HOURS = (9, 13, 16)
CH_SHADOW = {h: T.CHANNEL_ORDER.index(f"shadow_h{h:02d}") for h in SHADOW_HOURS}
SHADOW_IDXS = [CH_SHADOW[h] for h in SHADOW_HOURS]   # in SHADOW_HOURS order

# 1-degree bins, generous range. Values outside are clamped into the end bins
# (they exist -- a handful of pixels -- and dropping them would silently change
# N between the baseline and the corrected RMSE).
BIN_LO, BIN_HI = 10.0, 85.0
NBINS = int(BIN_HI - BIN_LO)          # 75 bins of 1 C
CLASSES = ("canopy_shaded", "building_shaded", "sunlit")

V5_EXPECTED_RMSE = 4.1824             # v5_test.json (default --expected-rmse)
RMSE_TOL = 0.02


def nearest_shadow_hour(hour: int) -> int:
    """Ties resolve to the later snapshot (11 -> 13): the sun has moved past
    the earlier one by then. Same rule as eval_signed_error.py."""
    return min(SHADOW_HOURS, key=lambda h: (abs(h - hour), -h))


class MaskSwapDataset(Dataset):
    """Base sample from the model's own corpus, PLUS the three shadow planes
    from a parallel corpus (out_v6). The model still sees its own inputs --
    only the GROUPING mask changes. This answers: is V5 biased in TRUE
    building shade, once the broken planes stop deciding what 'shade' means?"""

    def __init__(self, df: pd.DataFrame, corpus: Path, mask_corpus: Path):
        self.base = T.V5PatchDataset(df, corpus)
        self.df = df.reset_index(drop=True)
        self.mask_dir = Path(mask_corpus)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> dict:
        item = self.base[idx]
        name = Path(self.df.iloc[idx]["npz_path"]).name
        with np.load(self.mask_dir / name) as d:
            sh = d["input"][SHADOW_IDXS].astype(np.float32)
        item["mask_shadow"] = torch.from_numpy(sh)
        return item


def to_bin(v: torch.Tensor) -> torch.Tensor:
    """Value -> 1-C bin index, clamped into [0, NBINS-1]."""
    return torch.clamp((v - BIN_LO).long(), 0, NBINS - 1)


class Sums:
    """Per-key running (n, sum_d, sum_d2) held as float64 GPU tensors.
    float64 because 2e8 accumulations in float32 lose real digits."""

    def __init__(self, size: int, device: torch.device) -> None:
        self.n = torch.zeros(size, dtype=torch.float64, device=device)
        self.d = torch.zeros(size, dtype=torch.float64, device=device)
        self.d2 = torch.zeros(size, dtype=torch.float64, device=device)

    def add(self, idx: torch.Tensor, d: torch.Tensor) -> None:
        d64 = d.double()
        self.n.scatter_add_(0, idx, torch.ones_like(d64))
        self.d.scatter_add_(0, idx, d64)
        self.d2.scatter_add_(0, idx, d64 * d64)

    def cpu(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (self.n.cpu().numpy(), self.d.cpu().numpy(),
                self.d2.cpu().numpy())


def sse_after_own_bias(n: np.ndarray, d: np.ndarray, d2: np.ndarray) -> float:
    """SSE after subtracting each group's OWN mean error (the oracle move):
    SSE' = sum_k (d2_k - n_k * bias_k^2). Exact, not an approximation."""
    with np.errstate(divide="ignore", invalid="ignore"):
        bias = np.where(n > 0, d / np.maximum(n, 1), 0.0)
    return float((d2 - n * bias * bias).sum())


def sse_after_foreign_bias(n: np.ndarray, d: np.ndarray, d2: np.ndarray,
                           b: np.ndarray) -> float:
    """SSE after subtracting a correction b fitted ELSEWHERE (val split):
    sum_k (d2_k - 2 b_k d_k + n_k b_k^2). This is the honest deployable
    number -- fitted on val, applied to test."""
    return float((d2 - 2.0 * b * d + n * b * b).sum())


def run_split(df: pd.DataFrame, corpus: Path, model, device,
              amp_dtype, batch_size: int, num_workers: int, label: str,
              mask_corpus: Path | None = None):
    """One inference sweep. Returns dict of Sums-derived arrays."""
    ds = (MaskSwapDataset(df, corpus, mask_corpus) if mask_corpus
          else T.V5PatchDataset(df, corpus))
    loader = DataLoader(ds, batch_size=batch_size,
                        shuffle=False, num_workers=num_workers,
                        pin_memory=(device.type == "cuda"))
    overall = Sums(1, device)
    by_label = Sums(NBINS, device)
    by_pred = Sums(NBINS, device)
    by_class = Sums(len(CLASSES), device)
    by_joint = Sums(len(CLASSES) * NBINS, device)

    checked = False
    mask_diff_px = 0
    mask_tot_px = 0
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            x = batch["input"].to(device, non_blocking=True)
            tgt = batch["tmrt"].to(device, non_blocking=True)
            if amp_dtype is not None:
                with torch.autocast(device_type=device.type, dtype=amp_dtype):
                    pred = model(x)
                pred = pred.float()
            else:
                pred = model(x)
            pred = pred[:, 0]                         # head 0 = tmrt

            if not checked:
                checked = True
                cmax = float(x[:, CH_CDSM].max())
                if cmax <= 1.5:
                    print(f"  !! cdsm max {cmax:.2f} -- looks NORMALISED, not "
                          "metres. canopy mask would be empty. STOP.",
                          file=sys.stderr)
                    raise SystemExit(3)

            canopy = x[:, CH_CDSM] > CANOPY_M
            bshadow = torch.zeros_like(canopy)
            msk = batch.get("mask_shadow")
            if msk is not None:
                msk = msk.to(device, non_blocking=True)
            for i, h in enumerate(batch["hour"].tolist()):
                sh = nearest_shadow_hour(int(h))
                own = x[i, CH_SHADOW[sh]] > 0.5
                if msk is not None:
                    swapped = msk[i, SHADOW_HOURS.index(sh)] > 0.5
                    bshadow[i] = swapped
                    mask_diff_px += int((swapped != own).sum())
                    mask_tot_px += int(own.numel())
                else:
                    bshadow[i] = own

            finite = torch.isfinite(tgt)
            cls = torch.full_like(tgt, 2.0)           # sunlit
            cls[bshadow & ~canopy] = 1.0              # building_shaded
            cls[canopy] = 0.0                         # canopy wins overlap

            t = tgt[finite]
            p = pred[finite]
            c = cls[finite].long()
            d = p - t

            zero = torch.zeros_like(c)
            overall.add(zero, d)
            lb = to_bin(t)
            pb = to_bin(p)
            by_label.add(lb, d)
            by_pred.add(pb, d)
            by_class.add(c, d)
            by_joint.add(c * NBINS + lb, d)

            if bi % 20 == 0:
                print(f"  [{label}] {bi * batch_size:>5}/{len(df)}", flush=True)

    if mask_corpus is not None:
        frac = 100.0 * mask_diff_px / max(mask_tot_px, 1)
        print(f"  [{label}] swapped mask disagrees with own mask on "
              f"{frac:.2f}% of pixels")
        if mask_diff_px == 0:
            print("  !! swapped mask is IDENTICAL to the broken one -- wrong "
                  "--mask-corpus? STOP.", file=sys.stderr)
            raise SystemExit(3)

    return {"overall": overall.cpu(), "by_label": by_label.cpu(),
            "by_pred": by_pred.cpu(), "by_class": by_class.cpu(),
            "by_joint": by_joint.cpu(),
            "mask_diff_pct": (100.0 * mask_diff_px / max(mask_tot_px, 1)
                              if mask_corpus is not None else None)}


def rmse_of(sums, key: str) -> float:
    n, _, d2 = sums[key]
    return math.sqrt(d2.sum() / n.sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--tag", default="v5")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--expected-rmse", type=float, default=V5_EXPECTED_RMSE,
                    help="the model's recorded held-out test RMSE; the gate "
                         "must reproduce it (V5 4.1824, V6 4.469)")
    ap.add_argument("--mask-corpus", type=Path, default=None,
                    help="parallel corpus (out_v6) whose ch17-19 define the "
                         "building-shade GROUPING; model inputs are untouched")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0,
                    help="smoke only: first N samples per split. Numbers from "
                         "a subset are NOT the ceiling (section 28).")
    ap.add_argument("--no-amp", action="store_true")
    args = ap.parse_args()

    try:
        import segmentation_models_pytorch as smp
    except ImportError:
        print("segmentation_models_pytorch missing -- run in venv_train "
              "(inference only; nothing trains).", file=sys.stderr)
        return 2

    mp = args.corpus / "manifest.parquet"
    if not mp.exists():
        print(f"no manifest at {mp}", file=sys.stderr)
        return 2
    if not args.ckpt.exists():
        print(f"no checkpoint at {args.ckpt}", file=sys.stderr)
        return 2

    manifest = pd.read_parquet(mp)
    manifest = manifest[manifest["has_tmrt"]].reset_index(drop=True)
    manifest = T.build_splits(manifest)               # SPLIT_SEED = 42
    splits = {}
    for s in ("val", "test"):
        df = manifest[manifest.split == s].reset_index(drop=True)
        if args.limit:
            df = df.head(args.limit)
        splits[s] = df

    print(f"corpus     : {args.corpus}")
    print(f"checkpoint : {args.ckpt}")
    print(f"val/test   : {len(splits['val']):,} / {len(splits['test']):,} "
          f"samples (SPLIT_SEED={T.SPLIT_SEED})")
    if args.limit:
        print("⚠️  --limit set: SMOKE RUN. Do not read ceilings off a subset.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = (torch.bfloat16
                 if (device.type == "cuda" and not args.no_amp) else None)
    print(f"device     : {device}  amp={'bf16' if amp_dtype else 'off'}")

    model = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                     in_channels=T.NUM_CHANNELS, classes=len(T.HEAD_NAMES),
                     activation=None)
    ck = torch.load(args.ckpt, map_location=device, weights_only=True)
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()

    # val first: it fits the deployable calibration that test then scores.
    if args.mask_corpus is not None:
        probe = args.mask_corpus / "manifest.parquet"
        if not probe.exists():
            print(f"no manifest at {probe} -- wrong --mask-corpus?",
                  file=sys.stderr)
            return 2
        print(f"mask       : ch17-19 taken from {args.mask_corpus} "
              "(grouping only; model inputs unchanged)")
    S = {s: run_split(splits[s], args.corpus, model, device, amp_dtype,
                      args.batch_size, args.num_workers, s,
                      mask_corpus=args.mask_corpus)
         for s in ("val", "test")}

    n_all, d_all, d2_all = S["test"]["overall"]
    N = float(n_all.sum())
    base_rmse = math.sqrt(d2_all.sum() / N)

    # --- gate: must reproduce the recorded test RMSE -----------------------
    baseline_ok = abs(base_rmse - args.expected_rmse) <= RMSE_TOL
    gate = "PASS" if baseline_ok else "FAIL"
    if args.limit:
        gate = "SKIPPED (smoke)"
        baseline_ok = False

    # --- deployable: per-pred-bin bias fitted on VAL, applied to TEST ------
    vn, vd, _ = S["val"]["by_pred"]
    with np.errstate(divide="ignore", invalid="ignore"):
        b_val = np.where(vn > 0, vd / np.maximum(vn, 1), 0.0)
    tn, td, td2 = S["test"]["by_pred"]
    rmse_deploy = math.sqrt(sse_after_foreign_bias(tn, td, td2, b_val) / N)

    # --- ceilings (own-bias, on test) --------------------------------------
    def ceiling(key: str) -> float:
        return math.sqrt(sse_after_own_bias(*S["test"][key]) / N)

    rmse_cal = ceiling("by_pred")
    rmse_label = ceiling("by_label")
    rmse_class = ceiling("by_class")
    rmse_joint = ceiling("by_joint")

    # --- per-class detail ---------------------------------------------------
    cn, cd, cd2 = S["test"]["by_class"]
    jn, jd, jd2 = S["test"]["by_joint"]
    per_class = {}
    for k, name in enumerate(CLASSES):
        if cn[k] == 0:
            per_class[name] = {"pixels": 0}
            continue
        nk = float(cn[k])
        bias = float(cd[k] / cn[k])
        rmse_k = math.sqrt(float(cd2[k] / cn[k]))
        sl = slice(k * NBINS, (k + 1) * NBINS)
        rmse_k_joint = math.sqrt(
            sse_after_own_bias(jn[sl], jd[sl], jd2[sl]) / nk)
        per_class[name] = {
            "pixels": int(nk), "share_pct": 100.0 * nk / N,
            "bias_C": bias, "rmse_C": rmse_k,
            "rmse_after_joint_oracle_C": rmse_k_joint,
        }

    # --- by-label-bin table (compression fingerprint), 5 C for readability --
    ln, ld, _ = S["test"]["by_label"]
    coarse = {}
    for lo in range(0, NBINS, 5):
        n5, d5 = float(ln[lo:lo+5].sum()), float(ld[lo:lo+5].sum())
        if n5 > 0:
            coarse[f"{BIN_LO+lo:.0f}-{BIN_LO+lo+5:.0f}"] = {
                "pixels": int(n5), "bias_C": d5 / n5}

    # --- report -------------------------------------------------------------
    W = 72
    print()
    print("=" * W)
    print(f"BASELINE GATE : RMSE {base_rmse:.4f} C  vs recorded "
          f"{args.expected_rmse}  -> {gate}")
    if not baseline_ok and not args.limit:
        print("  !! numbers below are NOT trustworthy until this passes.")
    print("=" * W)
    rows = [
        ("baseline (as-is)", base_rmse, ""),
        ("deployable calibration (fit val)", rmse_deploy, "ship w/o training"),
        ("ceiling: output calibration", rmse_cal, "bound for any calib"),
        ("oracle: per label bin", rmse_label, "bound for bin reweighting"),
        ("oracle: per exposure class", rmse_class, "bound for shade weight"),
        ("oracle: class x label bin", rmse_joint, "bound for V7 loss"),
    ]
    print(f"  {'correction':<34}{'RMSE':>8}{'gain':>8}   note")
    for name, r, note in rows:
        print(f"  {name:<34}{r:>8.3f}{base_rmse - r:>+8.3f}   {note}")

    print()
    print("BY EXPOSURE CLASS (test split)")
    print(f"  {'class':<18}{'share':>7}{'bias':>8}{'rmse':>8}"
          f"{'rmse after V7-oracle':>22}")
    for name in CLASSES:
        r = per_class[name]
        if not r.get("pixels"):
            print(f"  {name:<18}      (no pixels)")
            continue
        print(f"  {name:<18}{r['share_pct']:>6.1f}%{r['bias_C']:>+8.2f}"
              f"{r['rmse_C']:>8.2f}{r['rmse_after_joint_oracle_C']:>22.2f}")

    print()
    print("BY LABEL BIN (bias; + cool-bin / - hot-bin = range compression)")
    for k, v in coarse.items():
        print(f"  {k:>7} C  {v['bias_C']:>+7.2f}   ({v['pixels']:,} px)")

    print()
    print("PRE-REGISTERED DECISION RULE (fixed before this run)")
    print("  retrain V7 only if: (oracle joint gain >= 0.15 C overall OR")
    print("  building_shaded rmse gain >= 1.0 C) AND baseline building_shaded")
    print("  |bias| >= 2.0 C. If deployable calibration captures >= 70% of the")
    print("  label-bin oracle, ship the calibration and skip training.")

    out_dir = args.out_dir or args.ckpt.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.tag}_ceiling.json"
    out.write_text(json.dumps({
        "corpus": str(args.corpus), "checkpoint": str(args.ckpt),
        "mask_corpus": str(args.mask_corpus) if args.mask_corpus else None,
        "mask_diff_pct_test": S["test"].get("mask_diff_pct"),
        "split_seed": T.SPLIT_SEED, "limit": args.limit,
        "baseline_gate": gate, "pixels_test": int(N),
        "rmse": {"baseline": base_rmse,
                 "deployable_calibration_val_fit": rmse_deploy,
                 "ceiling_output_calibration": rmse_cal,
                 "oracle_label_bin": rmse_label,
                 "oracle_exposure_class": rmse_class,
                 "oracle_class_x_label": rmse_joint},
        "by_class": per_class,
        "by_label_bin_5C": coarse,
    }, indent=1))
    print(f"\nwrote {out}")
    return 0 if (baseline_ok or args.limit) else 1


if __name__ == "__main__":
    raise SystemExit(main())
