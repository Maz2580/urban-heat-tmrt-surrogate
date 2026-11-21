#!/usr/bin/env python3
r"""Measure a model's mean SIGNED error on its held-out test split.

WHY THIS EXISTS
--------------------------------------------------------------------------
`v5_test.json` records RMSE only:

    {"rmse": {"tmrt": 4.1824, "lst": 1.7116, "tair_sparse": NaN}, ...}

RMSE squares its errors, so it is blind to direction. A model uniformly +4 C
too warm and one scattering +/-4 C randomly produce the identical number. The
team's report that V5 "shows a little hot overall" is a claim about BIAS -- the
mean signed error -- which RMSE structurally cannot express.

This script answers it, and splits the answer three ways, because the report was
specific: tree cooling reads correctly while other areas look extreme.

    canopy-shaded   cdsm > CANOPY_M                  (the half said to work)
    building-shaded that hour's shadow channel == 1  (the half that was broken)
    sunlit          neither

READ THIS BEFORE INTERPRETING THE OUTPUT
--------------------------------------------------------------------------
Error here is measured against the SOLWEIG labels in the corpus, i.e. "does the
model reproduce its own ground truth". It cannot tell you whether the labels
themselves are right. If mean signed error is ~0 and the labels average ~50 C,
then the model is faithful and "too hot" is a question about the LABELS and the
DISPLAY, not about the model.

Run it in venv_train (needs torch). Example:

    & $pyt stage_d\eval_signed_error.py `
        --corpus stage_d\out --ckpt stage_d\models\v5_best.pt --tag v5

RUN IT ON THE CORPUS THE MODEL WAS TRAINED ON. Scoring V5 against the Linux
`patches_v4` labels would manufacture a fake warm bias of roughly +20 C, which
looks exactly like the reported symptom.
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
from torch.utils.data import DataLoader

# train_v5.py owns the split, the dataset and the RMSE arithmetic. Import it
# rather than reimplementing: a reimplementation that drifts would produce a
# number that looks plausible and answers a different question.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_v5 as T  # noqa: E402

CANOPY_M = 3.0                      # cdsm above this counts as canopy
CH_CDSM = T.CHANNEL_ORDER.index("cdsm")
# The three shadow planes were generated at 09/13/16 (v4_prep task_shadow), but
# the corpus samples sit at hours 9/11/13/15. They are STATIC channels -- every
# sample of a patch carries all three regardless of its own hour -- so for the
# exposure split we pick the nearest available snapshot rather than falling back
# to the union of all three, which would overstate the shaded area by counting
# morning and afternoon shadow together.
SHADOW_HOURS = (9, 13, 16)
CH_SHADOW = {h: T.CHANNEL_ORDER.index(f"shadow_h{h:02d}") for h in SHADOW_HOURS}


def nearest_shadow_hour(hour: int) -> int:
    """Ties resolve to the later snapshot (11 -> 13): the sun has moved past the
    earlier one by then, so the later shadow is the better approximation."""
    return min(SHADOW_HOURS, key=lambda h: (abs(h - hour), -h))
BIN_EDGES = np.arange(15.0, 80.1, 5.0)   # label bins, deg C


class Acc:
    """Running sums. Accumulate rather than store: the test split is ~770
    samples x 262144 px, which does not fit in memory as float32."""

    def __init__(self) -> None:
        self.n = 0
        self.sum_d = 0.0        # signed
        self.sum_d2 = 0.0
        self.sum_t = 0.0        # label
        self.sum_p = 0.0        # prediction

    def add(self, d: torch.Tensor, tgt: torch.Tensor, pred: torch.Tensor) -> None:
        if d.numel() == 0:
            return
        self.n += int(d.numel())
        self.sum_d += float(d.sum())
        self.sum_d2 += float((d ** 2).sum())
        self.sum_t += float(tgt.sum())
        self.sum_p += float(pred.sum())

    def report(self) -> dict:
        if self.n == 0:
            return {"pixels": 0}
        return {"pixels": self.n,
                "mean_signed_error_C": self.sum_d / self.n,
                "rmse_C": math.sqrt(self.sum_d2 / self.n),
                "mean_label_C": self.sum_t / self.n,
                "mean_pred_C": self.sum_p / self.n}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True,
                    help="Stage D out dir; must hold manifest.parquet")
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--tag", default="v5")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="where to write <tag>_signed_error.json (default: ckpt's dir)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="first N test samples only")
    ap.add_argument("--no-amp", action="store_true",
                    help="disable bf16 autocast; V5's own test pass USED it")
    args = ap.parse_args()

    try:
        import segmentation_models_pytorch as smp
    except ImportError:
        print("segmentation_models_pytorch missing -- are you in venv_train?",
              file=sys.stderr)
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
    manifest = T.build_splits(manifest)          # SPLIT_SEED = 42, V5's own split
    test = manifest[manifest.split == "test"].reset_index(drop=True)
    if args.limit:
        test = test.head(args.limit)

    print(f"corpus     : {args.corpus}")
    print(f"checkpoint : {args.ckpt}")
    print(f"test split : {len(test):,} samples, "
          f"{test['v2_patch_id'].nunique()} patches (SPLIT_SEED={T.SPLIT_SEED})")
    print(f"hours      : {sorted(test['hour'].unique().tolist())}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if (device.type == "cuda" and not args.no_amp) else None
    print(f"device     : {device}  amp={'bf16' if amp_dtype else 'off'}")

    model = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                     in_channels=T.NUM_CHANNELS, classes=len(T.HEAD_NAMES),
                     activation=None)
    ck = torch.load(args.ckpt, map_location=device, weights_only=True)
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()

    loader = DataLoader(
        T.V5PatchDataset(test, args.corpus), batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=True)

    overall = Acc()
    groups = {"canopy_shaded": Acc(), "building_shaded": Acc(), "sunlit": Acc()}
    bins = [Acc() for _ in range(len(BIN_EDGES) - 1)]
    hour_mapped: dict[tuple[int, int], int] = {}

    printed_diag = False
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
            pred = pred[:, 0]                     # head 0 = tmrt

            if not printed_diag:
                printed_diag = True
                cd = x[:, CH_CDSM]
                print()
                print("  --- channel sanity (first batch) ---")
                print(f"  cdsm      : min {float(cd.min()):.2f}  max {float(cd.max()):.2f}"
                      f"  mean {float(cd.mean()):.2f}   >{CANOPY_M} m: "
                      f"{float((cd > CANOPY_M).float().mean()) * 100:.2f}%")
                if float(cd.max()) <= 1.5:
                    print("  !! cdsm looks NORMALISED, not metres. CANOPY_M is wrong;")
                    print("     canopy_shaded would be empty. STOP and check.")
                for nm in ("shadow_h09", "shadow_h13", "shadow_h16"):
                    c = x[:, T.CHANNEL_ORDER.index(nm)]
                    print(f"  {nm}: mean {float(c.mean()):.4f}  "
                          f"unique~{int(len(torch.unique(c[:1])))}")
                print()

            finite = torch.isfinite(tgt)
            d_all = (pred - tgt)

            overall.add(d_all[finite], tgt[finite], pred[finite])

            # --- group masks, per sample (the shadow channel depends on hour) --
            canopy = x[:, CH_CDSM] > CANOPY_M
            bshadow = torch.zeros_like(canopy)
            for i, h in enumerate(batch["hour"].tolist()):
                h = int(h)
                sh = nearest_shadow_hour(h)
                if sh != h:
                    hour_mapped[(h, sh)] = hour_mapped.get((h, sh), 0) + 1
                bshadow[i] = x[i, CH_SHADOW[sh]] > 0.5

            m_can = finite & canopy
            m_bld = finite & bshadow & ~canopy       # canopy wins where both
            m_sun = finite & ~canopy & ~bshadow
            groups["canopy_shaded"].add(d_all[m_can], tgt[m_can], pred[m_can])
            groups["building_shaded"].add(d_all[m_bld], tgt[m_bld], pred[m_bld])
            groups["sunlit"].add(d_all[m_sun], tgt[m_sun], pred[m_sun])

            # --- bins by LABEL value -------------------------------------
            for k in range(len(BIN_EDGES) - 1):
                mb = finite & (tgt >= BIN_EDGES[k]) & (tgt < BIN_EDGES[k + 1])
                if mb.any():
                    bins[k].add(d_all[mb], tgt[mb], pred[mb])

            if bi % 20 == 0:
                print(f"  [{bi * args.batch_size:>5}/{len(test)}] ...", flush=True)

    o = overall.report()
    print()
    print("=" * 72)
    print("OVERALL (T_mrt, held-out test split)")
    print("=" * 72)
    print(f"  pixels              : {o['pixels']:,}")
    print(f"  RMSE                : {o['rmse_C']:.3f} C")
    print(f"  MEAN SIGNED ERROR   : {o['mean_signed_error_C']:+.3f} C   "
          "<-- positive = model runs HOT vs its labels")
    print(f"  mean label          : {o['mean_label_C']:.2f} C")
    print(f"  mean prediction     : {o['mean_pred_C']:.2f} C")

    print()
    print("=" * 72)
    print("BY EXPOSURE")
    print("=" * 72)
    print(f"  {'group':<18}{'pixels':>14}{'signed':>10}{'rmse':>9}"
          f"{'label':>9}{'pred':>9}")
    for name, acc in groups.items():
        r = acc.report()
        if r["pixels"] == 0:
            print(f"  {name:<18}{'0':>14}   (no pixels)")
            continue
        print(f"  {name:<18}{r['pixels']:>14,}{r['mean_signed_error_C']:>+10.2f}"
              f"{r['rmse_C']:>9.2f}{r['mean_label_C']:>9.2f}{r['mean_pred_C']:>9.2f}")

    print()
    print("=" * 72)
    print("BY LABEL VALUE  (is any bias concentrated at the hot end?)")
    print("=" * 72)
    print(f"  {'label bin':<16}{'pixels':>14}{'signed':>10}{'rmse':>9}")
    for k in range(len(BIN_EDGES) - 1):
        r = bins[k].report()
        if r["pixels"] == 0:
            continue
        lbl = f"{BIN_EDGES[k]:.0f}-{BIN_EDGES[k+1]:.0f} C"
        print(f"  {lbl:<16}{r['pixels']:>14,}{r['mean_signed_error_C']:>+10.2f}"
              f"{r['rmse_C']:>9.2f}")

    if hour_mapped:
        print("\n  NOTE: shadow snapshots exist only at 09/13/16. Samples at other")
        print("  hours used the nearest snapshot:")
        for (h, sh), n in sorted(hour_mapped.items()):
            print(f"    hour {h:>2} -> shadow_h{sh:02d}   ({n} samples)")

    print()
    print("HOW TO READ THIS")
    print("  mean signed error ~ 0  -> the model reproduces its labels. 'Too hot'")
    print("                            is then about the LABELS or the DISPLAY")
    print("                            (colour ramp clamps at 55 C), not the model.")
    print("  clearly positive       -> a real warm bias; investigate separately.")
    print("  RMSE must match the model's own *_test.json, or this eval is wrong.")

    out_dir = args.out_dir or args.ckpt.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.tag}_signed_error.json"
    out.write_text(json.dumps({
        "corpus": str(args.corpus), "checkpoint": str(args.ckpt),
        "split_seed": T.SPLIT_SEED, "samples": int(len(test)),
        "canopy_threshold_m": CANOPY_M,
        "overall": o,
        "by_exposure": {k: v.report() for k, v in groups.items()},
        "by_label_bin": {f"{BIN_EDGES[k]:.0f}-{BIN_EDGES[k+1]:.0f}": bins[k].report()
                         for k in range(len(BIN_EDGES) - 1) if bins[k].n},
    }, indent=1))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
