#!/usr/bin/env python3
r"""FIT SHADE CALIBRATION -- per-(hour x trajectory) bias, fitted on VAL.

shade_age_audit.py (test split) showed V5 runs +1 to +2.3 C warm in true,
current shade, growing with shade depth -- small in corpus RMSE, large in
perceived shadow contrast. This fits the correction table on the VAL split
(so the test split remains an honest judge) for the serving-time calibration:

    tmrt_corrected = tmrt - bias[hour][trajectory]     (non-canopy px only)

where trajectory = 3 bits of the CORRECTED planes (s09,s13,s16) and hour is
the model-convention hour, snapped to the nearest of 9/11/13/15. Cells with
fewer than MIN_PIXELS get bias 0 (no confident fit -> no correction).

Writes v5_shade_calib.json next to the checkpoint. Copy it to the server at
urban_heat_data/models/ for serving.

Read-only inference, VAL split, venv_train, everything in the working folder.

    & .\venv_train\Scripts\python.exe stage_d\fit_shade_calib.py `
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
NPAT = 8
MIN_PIXELS = 50_000
CANOPY_M = C.CANOPY_M


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--mask-corpus", type=Path, required=True)
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
    val = manifest[manifest.split == "val"].reset_index(drop=True)
    if args.limit:
        val = val.head(args.limit)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = (torch.bfloat16
                 if (device.type == "cuda" and not args.no_amp) else None)
    print(f"fit split : VAL, {len(val):,} samples (test stays the judge)")

    model = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                     in_channels=T.NUM_CHANNELS, classes=len(T.HEAD_NAMES),
                     activation=None)
    ck = torch.load(args.ckpt, map_location=device, weights_only=True)
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()

    loader = DataLoader(C.MaskSwapDataset(val, args.corpus, args.mask_corpus),
                        batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers,
                        pin_memory=(device.type == "cuda"))

    err = C.Sums(len(HOURS) * NPAT, device)
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
            pat = ((msk[:, 0] > 0.5).long() * 4 + (msk[:, 1] > 0.5).long() * 2
                   + (msk[:, 2] > 0.5).long())
            hour_i = torch.tensor(
                [HOURS.index(int(h)) for h in batch["hour"].tolist()],
                device=device).view(B, 1, 1).expand(B, H, W)
            base = torch.isfinite(tgt) & ~(x[:, C.CH_CDSM] > CANOPY_M)
            err.add((hour_i * NPAT + pat)[base], (pred - tgt)[base])

            if bi % 20 == 0:
                print(f"  [{bi * args.batch_size:>5}/{len(val)}]", flush=True)

    en, ed, ed2 = err.cpu()
    bias = {}
    npx = {}
    print(f"\n  {'hour':>5}{'pat':>5}{'n(px)':>13}{'bias':>8}  applied?")
    for hi, h in enumerate(HOURS):
        bias[str(h)] = {}
        npx[str(h)] = {}
        for p in range(NPAT):
            k = hi * NPAT + p
            n = int(en[k])
            b = float(ed[k] / en[k]) if n else 0.0
            use = n >= MIN_PIXELS
            bias[str(h)][format(p, "03b")] = round(b, 4) if use else 0.0
            npx[str(h)][format(p, "03b")] = n
            print(f"  {h:>5}{format(p, '03b'):>5}{n:>13,}{b:>+8.2f}"
                  f"  {'yes' if use else 'no (thin cell)'}")

    out_dir = args.out_dir or args.ckpt.parent
    out = out_dir / f"{args.tag}_shade_calib.json"
    out.write_text(json.dumps({
        "model": args.tag,
        "fit_split": "val",
        "fit_corpus": str(args.corpus),
        "truth_planes": str(args.mask_corpus),
        "checkpoint": str(args.ckpt),
        "hours": list(HOURS),
        "canopy_threshold_m": CANOPY_M,
        "min_pixels": MIN_PIXELS,
        "semantics": "tmrt_corrected = tmrt - bias[hour][s09s13s16]; "
                     "non-canopy pixels only; hour snapped to nearest of "
                     "9/11/13/15 in model convention; trajectory bits from "
                     "the CORRECTED (v6) shadow planes",
        "bias": bias,
        "n_pixels": npx,
    }, indent=1))
    print(f"\nwrote {out}")
    print("next: rclone this JSON to gdrive:UMAMI/stage_d/ so the server "
          "can pick it up")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
