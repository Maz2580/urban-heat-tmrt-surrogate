#!/usr/bin/env python3
r"""SHADOW LIE AUDIT -- does V5's error concentrate where its inputs LIE?

The team reports V5 "not acting good for the shadow in front of buildings".
Three measured facts make one specific hypothesis:

  1. V5 RELIES on ch17-19 (forcing them moved T_mrt -15.37 C).
  2. Those planes are mirrored/truncated; the corrected planes (out_v6)
     disagree with them on 19.4% of pixels.
  3. V6 does NOT disprove fixing them: it changed direction AND the source
     surface (dsm_1m -> Geoscape) at once, and learned to ignore the planes
     (-1.66 C sensitivity). A confounded run, not a verdict.

Hypothesis: V5 runs HOT exactly where the planes say "sun" but the ground is
truly shaded (the missing shadow in front of a building), and COLD on phantom
shade. Every earlier diagnostic averaged across these cells; this one splits
them. Non-canopy pixels only, four cells at the nearest snapshot hour:

  agree_sun       planes 0, truth 0
  missing_shade   planes 0, truth 1   <- "shadow in front of the building"
  phantom_shade   planes 1, truth 0
  agree_shade     planes 1, truth 1

Reported twice: ALL pixels, and INTERIOR only (>4 px from the boundary of
EITHER mask) so the 1-px rasterisation floor cannot masquerade as signal.

Also prints the RETRAIN CEILING: corpus RMSE if a corrected-planes V7 brought
each lying cell down to its honest counterpart's RMSE (missing_shade ->
agree_shade level, phantom_shade -> agree_sun level). If that number is not
meaningfully below 4.182, correcting the planes cannot help even in principle.

Read-only inference, test split, venv_train, everything in the working folder.

    & .\venv_train\Scripts\python.exe stage_d\shadow_lie_audit.py `
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
import scatter_diag as D                   # noqa: E402  (dilate)

CELLS = ("agree_sun", "missing_shade", "phantom_shade", "agree_shade")
VIEWS = ("all", "interior")
INTERIOR_PX = 4
V5_EXPECTED_RMSE = 4.1824
RMSE_TOL = 0.02


def boundary_zone(m: torch.Tensor, k: int) -> torch.Tensor:
    """Pixels within k px of the mask's edge. m: (B,H,W) bool."""
    mf = m.float()
    bnd = ((D.dilate(mf, 1) > 0.5) & (D.dilate(1.0 - mf, 1) > 0.5)).float()
    return D.dilate(bnd, k) > 0.5


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--mask-corpus", type=Path, required=True,
                    help="out_v6 -- the corrected planes are the TRUTH side")
    ap.add_argument("--tag", default="v5")
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

    nc = len(CELLS)
    err = C.Sums(len(VIEWS) * nc, device)      # signed error d = pred - label
    lab = C.Sums(len(VIEWS) * nc, device)      # label values (for mean label)
    overall = C.Sums(1, device)

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
            old = torch.zeros((B, H, W), dtype=torch.bool, device=device)
            new = torch.zeros_like(old)
            for i, h in enumerate(batch["hour"].tolist()):
                sh = C.nearest_shadow_hour(int(h))
                old[i] = x[i, C.CH_SHADOW[sh]] > 0.5
                new[i] = msk[i, C.SHADOW_HOURS.index(sh)] > 0.5

            canopy = x[:, C.CH_CDSM] > C.CANOPY_M
            finite = torch.isfinite(tgt)
            base = finite & ~canopy

            cell = old.long() * 2 + new.long()          # 0..3 in CELLS order?
            # CELLS order: agree_sun(0,0)=0, missing(0,1)=1, phantom(1,0)=2,
            # agree_shade(1,1)=3  -> old*2 + new matches exactly.

            interior = base & ~boundary_zone(old, INTERIOR_PX) \
                            & ~boundary_zone(new, INTERIOR_PX)

            d_all = pred - tgt
            overall.add(torch.zeros_like(tgt[finite], dtype=torch.long),
                        d_all[finite])
            for vi, view_mask in enumerate((base, interior)):
                idx = vi * nc + cell[view_mask]
                err.add(idx, d_all[view_mask])
                lab.add(idx, tgt[view_mask])

            if bi % 20 == 0:
                print(f"  [{bi * args.batch_size:>5}/{len(test)}]", flush=True)

    n0, _, d20 = overall.cpu()
    N = float(n0.sum())
    base_rmse = math.sqrt(d20.sum() / N)
    gate = "PASS" if abs(base_rmse - V5_EXPECTED_RMSE) <= RMSE_TOL else "FAIL"
    if args.limit:
        gate = "SKIPPED (smoke)"
    print()
    print("=" * 76)
    print(f"GATE : pixel RMSE {base_rmse:.4f} C vs recorded "
          f"{V5_EXPECTED_RMSE} -> {gate}")
    print("=" * 76)

    en, ed, ed2 = err.cpu()
    ln, ldm, _ = lab.cpu()
    j = {"gate": gate, "views": {}}
    for vi, view in enumerate(VIEWS):
        print(f"\n{view.upper()} (non-canopy"
              + (f", >{INTERIOR_PX}px from any shade edge)" if vi else ")"))
        print(f"  {'cell':<15}{'share':>8}{'label':>8}{'bias':>8}{'rmse':>8}")
        vtot = en[vi * nc:(vi + 1) * nc].sum()
        rows = {}
        for ci, cn in enumerate(CELLS):
            k = vi * nc + ci
            if en[k] == 0:
                continue
            r = {"pixels": int(en[k]),
                 "share_pct": 100.0 * float(en[k] / vtot),
                 "mean_label_C": float(ldm[k] / ln[k]),
                 "bias_C": float(ed[k] / en[k]),
                 "rmse_C": math.sqrt(float(ed2[k] / en[k]))}
            rows[cn] = r
            print(f"  {cn:<15}{r['share_pct']:>7.1f}%{r['mean_label_C']:>8.2f}"
                  f"{r['bias_C']:>+8.2f}{r['rmse_C']:>8.2f}")
        j["views"][view] = rows

    # --- retrain ceiling: lying cells brought to their honest counterparts --
    a = j["views"]["all"]
    fixable = 0.0
    for liar, honest in (("missing_shade", "agree_shade"),
                         ("phantom_shade", "agree_sun")):
        if liar in a and honest in a:
            n_l = a[liar]["pixels"]
            fixable += n_l * (a[liar]["rmse_C"] ** 2 -
                              a[honest]["rmse_C"] ** 2)
    fixable = max(fixable, 0.0)
    new_rmse = math.sqrt(max(d20.sum() - fixable, 0.0) / N)
    j["retrain_ceiling_rmse"] = new_rmse
    print()
    print("RETRAIN CEILING (corrected-planes V7, single-variable):")
    print(f"  corpus RMSE if lying cells matched their honest counterparts:")
    print(f"  {base_rmse:.3f} -> {new_rmse:.3f} C   (gain {base_rmse - new_rmse:+.3f})")
    print()
    print("HOW TO READ")
    print("  missing_shade bias >> 0 (esp. INTERIOR)  -> V5 trusts lying")
    print("  planes and renders sun in front of shaded buildings. A V7 with")
    print("  direction-fixed planes on the ORIGINAL dsm_1m surface (single")
    print("  variable, unlike V6) has the ceiling printed above.")
    print("  missing_shade bias ~ 0 and rmse ~ agree_shade -> V5 already")
    print("  overrides the lie from geometry; the planes are not the blocker.")

    out_dir = args.out_dir or args.ckpt.parent
    out = out_dir / f"{args.tag}_shadow_lie.json"
    out.write_text(json.dumps(j, indent=1))
    print(f"\nwrote {out}")
    return 0 if gate != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
