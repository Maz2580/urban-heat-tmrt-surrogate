#!/usr/bin/env python3
"""
Export the V5 checkpoint to ONNX, and prove the export is faithful.
===================================================================

Inference serves ONNX, training produced a PyTorch checkpoint. An export that
silently changes the numbers would look like a bad model rather than a bad
export, so this compares the two directly on real corpus samples and refuses to
write the final file if they disagree.

Writes ONLY into --out-dir. Nothing here touches urban_heat_data/models/;
promoting the file there is a separate, explicit step.

    python export_v5_onnx.py --ckpt <v5_best.pt> --out-dir <scratch dir>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

DATA = Path("/data") if Path("/data/training").exists() else Path("/app/urban_heat_data")
NUM_CHANNELS = 23
PATCH = 512


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path,
                    default=DATA / "training/v5_calib/models/v5_best.pt")
    ap.add_argument("--out-dir", type=Path,
                    default=DATA / "training/v5_calib/onnx_export")
    ap.add_argument("--samples", type=Path,
                    default=DATA / "training/v5_calib/stage_d_pack_test")
    # Tolerance applies to the tmrt head ONLY, which is the channel inference
    # serves. The tair_sparse head trained on an all-NaN target, so it is inert
    # and its outputs are unconstrained -- holding arbitrary values to a
    # temperature tolerance measures nothing. Measured on the worst sample the
    # tmrt head agrees to 0.0100 C max, median 0.00002 C, which is fp32
    # arithmetic reordering (0.24% of the model's own 4.18 C RMSE).
    ap.add_argument("--tol", type=float, default=0.05,
                    help="max allowed |pytorch - onnx| on the tmrt head, deg C")
    # Default preserves the original V5 behaviour exactly, so this script still
    # does what it did for the model it was written for.
    ap.add_argument("--out-name", default="umami_unet_v5_fp32.onnx",
                    help="output filename, e.g. umami_unet_v6_fp32.onnx")
    args = ap.parse_args()

    import segmentation_models_pytorch as smp

    args.out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = args.out_dir / args.out_name

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    print(f"checkpoint: epoch {ck.get('epoch','?')}, "
          f"val_rmse_tmrt {ck.get('val_rmse_tmrt', float('nan')):.4f}")

    model = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                     in_channels=NUM_CHANNELS, classes=3, activation=None)
    missing, unexpected = model.load_state_dict(ck["state_dict"], strict=False)
    if missing or unexpected:
        print(f"ABORT: state_dict mismatch missing={len(missing)} "
              f"unexpected={len(unexpected)}", file=sys.stderr)
        return 2
    model.eval()
    print("state_dict loaded cleanly")

    dummy = torch.zeros(1, NUM_CHANNELS, PATCH, PATCH)
    print(f"exporting -> {onnx_path}")
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
        dynamo=False,            # legacy exporter: no onnxscript dependency
    )
    size = onnx_path.stat().st_size
    print(f"  wrote {size/1e6:.1f} MB")

    # --- faithfulness check on REAL samples, not the zero dummy ------------
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    print(f"  onnx input: {iname} {sess.get_inputs()[0].shape}")

    files = sorted(args.samples.glob("*.npz"))[:6]
    if not files:
        print(f"no samples in {args.samples}", file=sys.stderr)
        return 2

    print(f"\n{'sample':<28} {'max|diff|':>11} {'mean pt':>9} {'mean onnx':>10}")
    worst = 0.0
    for f in files:
        x = np.load(f)["input"].astype(np.float32)
        with torch.no_grad():
            pt = model(torch.from_numpy(x)[None]).numpy()
        on = sess.run(None, {iname: x[None]})[0]
        d = float(np.abs(pt[0, 0] - on[0, 0]).max())      # tmrt head only
        worst = max(worst, d)
        print(f"{f.stem:<28} {d:>11.6f} {pt[0,0].mean():>9.3f} {on[0,0].mean():>10.3f}")

    print()
    if worst > args.tol:
        print(f"FAILED: max difference {worst:.6f} C exceeds tolerance {args.tol}")
        print("The export is NOT faithful. Do not promote this file.")
        return 1
    print(f"PASS: pytorch and onnx agree to {worst:.6f} C (tolerance {args.tol})")
    print(f"\nfile ready at {onnx_path}")
    print("It is NOT installed. Promoting it into urban_heat_data/models/ is a")
    print("separate, explicit step.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
