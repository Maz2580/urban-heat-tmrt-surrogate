#!/usr/bin/env python3
r"""ERROR SPECTRUM -- at what SPATIAL SCALE does V5's error live?

edge_floor.py showed V5 is at the 1-px rasterisation floor at shadow
boundaries, but 2x above local label structure in FLAT regions (3.62 C where
labels self-shift at 1.67 C). Hypothesis: the flat-region error is a LEVEL
error -- the model misjudges scene-scale heat under some conditions (doy46!)
-- not a texture error.

Test: average the signed error map over k x k tiles and take the RMS of the
tile means. Texture errors cancel under averaging; level errors survive it.

  RMS at 512 (whole scene)  = how wrong the model is about a scene's MEAN
  RMS at 128 / 32 / 8       = intermediate scales
  RMS at 1                  = the ordinary pixel RMSE (must gate at 4.1824)

Reading it:
  512-scale RMS >= ~2 C                -> scene-level conditioning is the
                                          problem; V7 = stronger global/met
                                          conditioning. Real headroom.
  error mostly gone by 32 px           -> diffuse texture; we are near the
                                          floor everywhere; V5 is the model.

Read-only inference, test split, venv_train, everything in the working folder.

    & .\venv_train\Scripts\python.exe stage_d\error_spectrum.py `
        --corpus stage_d\out --ckpt stage_d\models\v5_best.pt --tag v5
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_v5 as T                       # noqa: E402
import ceiling_test as C                   # noqa: E402

SCALES = (512, 128, 32, 8)
MIN_VALID = 0.5           # a tile counts only if >=50% of its pixels are valid
V5_EXPECTED_RMSE = 4.1824
RMSE_TOL = 0.02


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--tag", default="v5")
    ap.add_argument("--expected-rmse", type=float, default=V5_EXPECTED_RMSE)
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
    nd = len(doys)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = (torch.bfloat16
                 if (device.type == "cuda" and not args.no_amp) else None)
    print(f"corpus : {args.corpus}")
    print(f"test   : {len(test):,} samples  device {device} "
          f"amp={'bf16' if amp_dtype else 'off'}  doys {doys}")

    model = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                     in_channels=T.NUM_CHANNELS, classes=len(T.HEAD_NAMES),
                     activation=None)
    ck = torch.load(args.ckpt, map_location=device, weights_only=True)
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()

    loader = DataLoader(T.V5PatchDataset(test, args.corpus),
                        batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers,
                        pin_memory=(device.type == "cuda"))

    # per scale: index = doy_idx (tile means grouped by date) + one global row
    tiles = {k: C.Sums(nd + 1, device) for k in SCALES}
    pix = C.Sums(nd + 1, device)

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
            fin = torch.isfinite(tgt)
            e = torch.where(fin, pred - tgt, torch.zeros_like(pred))
            fm = fin.float()
            di = torch.tensor([doy_idx[int(d)] for d in batch["doy"].tolist()],
                              device=device)

            dpx = (pred - tgt)[fin]
            # pixel level: global row nd, plus per-doy rows
            pix.add(torch.full((dpx.numel(),), nd, dtype=torch.long,
                               device=device), dpx)
            dexp = di.view(B, 1, 1).expand(B, H, W)[fin]
            pix.add(dexp, dpx)

            for k in SCALES:
                if k > H:
                    continue
                se = F.avg_pool2d(e.unsqueeze(1), k).squeeze(1) * (k * k)
                cn = F.avg_pool2d(fm.unsqueeze(1), k).squeeze(1) * (k * k)
                ok = cn >= (k * k * MIN_VALID)
                mean = torch.where(ok, se / cn.clamp(min=1.0),
                                   torch.zeros_like(se))
                m = mean[ok]
                tiles[k].add(torch.full((m.numel(),), nd, dtype=torch.long,
                                        device=device), m)
                dt = di.view(B, 1, 1).expand_as(mean)[ok]
                tiles[k].add(dt, m)

            if bi % 20 == 0:
                print(f"  [{bi * args.batch_size:>5}/{len(test)}]", flush=True)

    def rms(sums: C.Sums, row: int):
        n, _, d2 = sums.cpu()
        return (math.sqrt(float(d2[row] / n[row])), int(n[row])) \
            if n[row] else (None, 0)

    base, npx = rms(pix, nd)
    gate = "PASS" if abs(base - args.expected_rmse) <= RMSE_TOL else "FAIL"
    if args.limit:
        gate = "SKIPPED (smoke)"
    print()
    print("=" * 72)
    print(f"GATE : pixel RMSE {base:.4f} C vs recorded {args.expected_rmse}"
          f" -> {gate}")
    print("=" * 72)

    print("\nRMS OF TILE-MEAN ERROR BY SCALE  (level error survives averaging;")
    print("texture cancels). Columns: scale in px (= metres).")
    hdr = f"  {'':<10}" + "".join(f"{k:>9}" for k in SCALES) + f"{'1 (pixel)':>12}"
    print(hdr)
    j = {"gate": gate, "pixel_rmse": base, "scales": {}}

    def row_str(name, row):
        cells = ""
        vals = {}
        for k in SCALES:
            v, n = rms(tiles[k], row)
            cells += f"{v:>9.2f}" if v is not None else f"{'--':>9}"
            vals[str(k)] = v
        pv, _ = rms(pix, row)
        cells += f"{pv:>12.2f}"
        vals["1"] = pv
        print(f"  {name:<10}" + cells)
        return vals

    j["scales"]["ALL"] = row_str("ALL", nd)
    for d in doys:
        j["scales"][f"doy{d}"] = row_str(f"doy{d}", doy_idx[d])

    print()
    print("HOW TO READ")
    print("  512 column = how wrong the model is about a whole scene's MEAN.")
    print("  >= ~2 C there -> scene-level conditioning problem: the met/date")
    print("  channels are not doing their job. V7 = global conditioning.")
    print("  If the 512/128 columns are small and error only appears at 8/1,")
    print("  the error is diffuse texture -> near the floor -> V5 is the model.")

    out_dir = args.out_dir or args.ckpt.parent
    out = out_dir / f"{args.tag}_error_spectrum.json"
    out.write_text(json.dumps(j, indent=1))
    print(f"\nwrote {out}")
    return 0 if gate != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
