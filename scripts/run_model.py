#!/usr/bin/env python3
"""Run one trained model on one corpus sample, and score it if the sample carries a T_mrt label.

    python scripts/run_model.py --version v7 --sample patch.npz            # prediction + RMSE against the label
    python scripts/run_model.py --version v4                                # check the model runs (random input)

A corpus sample is an .npz with `input` (channels x 512 x 512, already scaled as in training) and, optionally, `tmrt`
(the label, degrees C; NaN = no value). This is how the V3/V4 test evaluation (`eval_v3_v4_test.py`) and the V5 to V7
training script read the corpora. V1 and V2 output T_mrt / 63 (`tmrt_range` in `results/dataset_stats.json`), which this
script rescales; from V3 onward the first output channel is T_mrt in degrees C. Each version needs inputs built for it:
V7 changes the meaning of channels 0, 10 and 17 to 19 (paper Section 3.4), so V5-era inputs give V7 a large error.
Requires numpy and onnxruntime; the models come from `scripts/download_models.py`.
"""
import argparse
import os

import numpy as np
import onnxruntime as ort

FILES = {"v1": "umami_unet_fp32.onnx", "v2": "umami_unet_v2_fp32.onnx", "v3": "umami_unet_v3_fp32.onnx",
         "v3.1": "umami_unet_v31_fp32.onnx", "v4": "umami_unet_v4_fp32.onnx", "v4.1": "umami_unet_v41_fp32.onnx",
         "v5": "umami_unet_v5_fp32.onnx", "v6": "umami_unet_v6_fp32.onnx", "v7": "umami_unet_v7_fp32.onnx"}
SCALE = {"v1": 63.0, "v2": 63.0}   # V1/V2 were trained on T_mrt / 63 (results/dataset_stats.json, tmrt_range)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--version", required=True, choices=sorted(FILES))
    ap.add_argument("--models", default="models", help="folder with the downloaded models (default: models)")
    ap.add_argument("--sample", help="corpus sample (.npz with 'input', optional 'tmrt'); omit for a random-input check")
    ap.add_argument("--save", help="write the T_mrt prediction to this .npy file")
    a = ap.parse_args()
    sess = ort.InferenceSession(os.path.join(a.models, FILES[a.version]), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    channels = inp.shape[1]
    if a.sample:
        d = np.load(a.sample)
        x = d["input"].astype(np.float32)[None, ...]
        if x.shape[1] != channels:
            raise SystemExit(f"{a.version} expects {channels} input channels, the sample has {x.shape[1]}")
    else:
        x = np.random.default_rng(0).random((1, channels, 512, 512), dtype=np.float32)
    out = sess.run(None, {inp.name: x})[0]
    pred = out[0, 0] * SCALE.get(a.version, 1.0)
    print(f"{a.version}: input {tuple(x.shape)} -> output {tuple(out.shape)}; T_mrt channel min {np.nanmin(pred):.2f}, "
          f"mean {np.nanmean(pred):.2f}, max {np.nanmax(pred):.2f}")
    if a.sample and "tmrt" in d.files:
        t = d["tmrt"].astype(np.float32)
        ok = np.isfinite(t) & np.isfinite(pred)
        print(f"RMSE against the label: {np.sqrt(np.mean((pred[ok] - t[ok]) ** 2)):.3f} C over {int(ok.sum()):,} pixels")
    if a.save:
        np.save(a.save, pred)
        print(f"prediction saved to {a.save}")


if __name__ == "__main__":
    main()
