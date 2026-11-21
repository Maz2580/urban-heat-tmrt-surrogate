#!/usr/bin/env python3
r"""SCATTER DIAGNOSIS -- where does V5's remaining error actually live?

The ceiling tests (v5_ceiling.json, v5_maskv6_ceiling.json, v6_ceiling.json)
closed the bias path: every per-bin / per-class bias removed buys only ~0.22 C
of 4.18. What remains is SCATTER, worst in building shade (RMSE 5.40 with the
corrected mask). This script splits that scatter three ways, each pointing at a
different V7 lever:

  1. BY HOUR x CLASS      shadow planes exist at 09/13/16 only; samples run
                          9/11/13/15, so hours 11 and 15 never see their own
                          shadow state. If shade RMSE at 11/15 >> 9/13 the
                          lever is a PER-HOUR shadow input channel.
  2. BY PATCH-EDGE BAND   a shadow cast into the window by a tower OUTSIDE it
                          is unpredictable from the inputs. If error rises
                          toward the border the lever is CONTEXT (bigger
                          patches / overlap-stitch inference).
  3. BY SHADE-BOUNDARY    if error concentrates within a few px of a shade
     DISTANCE             edge, shadows are right-but-misplaced; the lever is
                          localisation (resolution / features), and more data
                          or capacity would buy little.
  4. BY DATE (DOY)        season sanity: leaf-on/off, sun elevation.

Read-only, inference only, test split only. venv_train. Everything stays in
the working folder. Reuses ceiling_test.py's dataset/mask machinery -- keep both
files in the same folder.

    & .\venv_train\Scripts\python.exe stage_d\scatter_diag.py `
        --corpus stage_d\out --ckpt stage_d\models\v5_best.pt `
        --mask-corpus stage_d\out_v6 --tag v5

(--mask-corpus out_v6 so 'building shade' means TRUE shade, as established.)
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
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_v5 as T                      # noqa: E402
import ceiling_test as C                  # noqa: E402  (Sums, MaskSwapDataset)

CLASSES = C.CLASSES
NCLS = len(CLASSES)
HOURS = (9, 11, 13, 15)
# distance-to-patch-border bands, px (512 patch -> centre band is 112+)
EDGE_EDGES = (16, 48, 112)
EDGE_NAMES = ("0-15", "16-47", "48-111", "112+")
# distance-to-shade-boundary bands, px (boundary = edge of the building-shadow
# mask). Built by successive dilation, so bands are "within k px".
BND_KS = (1, 4, 12)
BND_NAMES = ("<=1", "2-4", "5-12", ">12")


def dilate(m: torch.Tensor, k: int) -> torch.Tensor:
    """Binary dilation by k px via max-pool (kernel 2k+1). m: (B,H,W) float."""
    return F.max_pool2d(m.unsqueeze(1), kernel_size=2 * k + 1,
                        stride=1, padding=k).squeeze(1)


def stats(sums: C.Sums):
    n, d, d2 = sums.cpu()
    out = []
    for k in range(len(n)):
        if n[k] == 0:
            out.append(None)
            continue
        out.append({"pixels": int(n[k]), "bias_C": float(d[k] / n[k]),
                    "rmse_C": math.sqrt(float(d2[k] / n[k]))})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--tag", default="v5")
    ap.add_argument("--mask-corpus", type=Path, default=None)
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

    doys = sorted(test["doy"].unique().tolist())
    doy_idx = {d: i for i, d in enumerate(doys)}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = (torch.bfloat16
                 if (device.type == "cuda" and not args.no_amp) else None)
    print(f"corpus  : {args.corpus}   mask: {args.mask_corpus or '(own ch17-19)'}")
    print(f"test    : {len(test):,} samples  device {device} "
          f"amp={'bf16' if amp_dtype else 'off'}  doys {doys}")

    model = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                     in_channels=T.NUM_CHANNELS, classes=len(T.HEAD_NAMES),
                     activation=None)
    ck = torch.load(args.ckpt, map_location=device, weights_only=True)
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()

    ds = (C.MaskSwapDataset(test, args.corpus, args.mask_corpus)
          if args.mask_corpus else T.V5PatchDataset(test, args.corpus))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers,
                        pin_memory=(device.type == "cuda"))

    s_hour = C.Sums(len(HOURS) * NCLS, device)
    s_edge = C.Sums(len(EDGE_NAMES) * NCLS, device)
    s_bnd = C.Sums(len(BND_NAMES), device)
    s_doy = C.Sums(len(doys) * NCLS, device)
    overall = C.Sums(1, device)

    edge_band = None                      # built once per (H, W)

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
            pred = pred[:, 0]

            B, H, W = tgt.shape
            if edge_band is None or edge_band.shape != (H, W):
                yy = torch.arange(H, device=device).view(-1, 1).expand(H, W)
                xx = torch.arange(W, device=device).view(1, -1).expand(H, W)
                dist = torch.minimum(torch.minimum(yy, H - 1 - yy),
                                     torch.minimum(xx, W - 1 - xx))
                edge_band = torch.bucketize(
                    dist.float(),
                    torch.tensor(EDGE_EDGES, dtype=torch.float32,
                                 device=device))          # 0..3

            canopy = x[:, C.CH_CDSM] > C.CANOPY_M
            bshadow = torch.zeros_like(canopy)
            msk = batch.get("mask_shadow")
            if msk is not None:
                msk = msk.to(device, non_blocking=True)
            for i, h in enumerate(batch["hour"].tolist()):
                sh = C.nearest_shadow_hour(int(h))
                if msk is not None:
                    bshadow[i] = msk[i, C.SHADOW_HOURS.index(sh)] > 0.5
                else:
                    bshadow[i] = x[i, C.CH_SHADOW[sh]] > 0.5

            cls = torch.full_like(tgt, 2.0)
            cls[bshadow & ~canopy] = 1.0
            cls[canopy] = 0.0
            cls = cls.long()

            # shade-boundary distance bands (from the building-shadow mask)
            mf = bshadow.float()
            bnd = (dilate(mf, 1) > 0.5) & (dilate(1.0 - mf, 1) > 0.5)
            bf = bnd.float()
            near = [dilate(bf, k) > 0.5 for k in BND_KS]
            bnd_band = torch.full_like(cls, len(BND_NAMES) - 1)   # >12
            bnd_band[near[2]] = 2
            bnd_band[near[1]] = 1
            bnd_band[near[0]] = 0

            hour_i = torch.tensor(
                [HOURS.index(int(h)) for h in batch["hour"].tolist()],
                device=device).view(B, 1, 1).expand(B, H, W)
            doy_i = torch.tensor(
                [doy_idx[int(d)] for d in batch["doy"].tolist()],
                device=device).view(B, 1, 1).expand(B, H, W)
            eb = edge_band.unsqueeze(0).expand(B, H, W)

            finite = torch.isfinite(tgt)
            d = (pred - tgt)[finite]
            c = cls[finite]
            overall.add(torch.zeros_like(c), d)
            s_hour.add(hour_i[finite] * NCLS + c, d)
            s_edge.add(eb[finite] * NCLS + c, d)
            s_bnd.add(bnd_band[finite], d)
            s_doy.add(doy_i[finite] * NCLS + c, d)

            if bi % 20 == 0:
                print(f"  [{bi * args.batch_size:>5}/{len(test)}]", flush=True)

    n0, _, d20 = overall.cpu()
    base = math.sqrt(d20.sum() / n0.sum())
    print(f"\noverall RMSE {base:.4f} C  "
          "(must match the ceiling run on the same corpus/mask)")

    def table(title, names, per, sums):
        st = stats(sums)
        print("\n" + "=" * 72)
        print(title)
        print("=" * 72)
        hdr = f"  {'':<10}" + "".join(f"{c:>22}" for c in CLASSES)
        print(hdr)
        rows = {}
        for r, nm in enumerate(names):
            cells = []
            row = {}
            for k in range(NCLS):
                v = st[r * NCLS + k]
                row[CLASSES[k]] = v
                cells.append("            --        " if v is None else
                             f"  {v['rmse_C']:>6.2f} ({v['bias_C']:+5.2f}) "
                             f"{v['pixels']/1e6:>5.1f}M")
            print(f"  {nm:<10}" + "".join(cells))
            rows[nm] = row
        print("  cell = rmse (bias) pixels")
        return rows

    j = {"corpus": str(args.corpus),
         "mask_corpus": str(args.mask_corpus) if args.mask_corpus else None,
         "checkpoint": str(args.ckpt), "overall_rmse": base}

    j["by_hour"] = table("1. BY HOUR x CLASS   (shadow snapshots exist at "
                         "09/13/16 -> 11 and 15 are stale)",
                         [f"h{h:02d}" for h in HOURS], NCLS, s_hour)
    j["by_edge"] = table("2. BY DISTANCE TO PATCH BORDER (px)",
                         EDGE_NAMES, NCLS, s_edge)

    st = stats(s_bnd)
    print("\n" + "=" * 72)
    print("3. BY DISTANCE TO SHADE BOUNDARY (px from a building-shadow edge)")
    print("=" * 72)
    j["by_boundary"] = {}
    for r, nm in enumerate(BND_NAMES):
        v = st[r]
        if v is None:
            continue
        print(f"  {nm:>5} px   rmse {v['rmse_C']:>6.2f}  bias {v['bias_C']:+6.2f}"
              f"   {v['pixels']/1e6:>7.1f}M px")
        j["by_boundary"][nm] = v

    j["by_doy"] = table("4. BY DATE (DOY) x CLASS",
                        [f"doy{d}" for d in doys], NCLS, s_doy)

    print("\nHOW TO READ")
    print("  1. shade rmse at h11/h15 >> h09/h13    -> per-hour shadow channel")
    print("  2. rmse rising toward the border       -> context / bigger window")
    print("  3. rmse concentrated at <=4 px of edge -> localisation, not data")
    print("  If none dominates, scatter is diffuse  -> capacity / ensemble.")

    out_dir = args.out_dir or args.ckpt.parent
    out = out_dir / f"{args.tag}_scatter_diag.json"
    out.write_text(json.dumps(j, indent=1))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
