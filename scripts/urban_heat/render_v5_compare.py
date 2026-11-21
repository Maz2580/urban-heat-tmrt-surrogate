#!/usr/bin/env python3
"""
Four-panel comparison: old label, new label, V4.1 prediction, V5 prediction.
============================================================================

This is the check six model versions failed. Every metric was green and every
map was wrong, so the numbers coming out of training are NOT the verdict.

ONE SHARED COLOUR SCALE across all panels. Per-panel scaling makes a flat field
and a structured one look equally vivid, and is plausibly how the original maps
passed inspection. The scale is computed once from the reference label.

Stated in advance, so they can fail:
  * canopy should sit ~19 C below sunlit ground   (old engine managed ~7 C)
  * individual tree crowns should read as cool islands, not a smear
  * shadow corridors should line up with the buildings that cast them
  * dark asphalt hot, water cool -- the old engine had this ORDER INVERTED

Read-only except for the PNG it writes.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch

DATA = Path("/data") if Path("/data/training").exists() else Path("/app/urban_heat_data")
V4_DIR = DATA / "training/patches_v4"
STAGE_C = DATA / "training/v5_calib/colab_check"
LIVE_MODELS = DATA / "models"

NUM_CHANNELS = 23


def load_v5(ckpt_path: Path, device):
    import segmentation_models_pytorch as smp
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                     in_channels=NUM_CHANNELS, classes=3, activation=None)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state = ck.get("state_dict", ck)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"  V5 checkpoint: epoch {ck.get('epoch', '?')}, "
          f"val_rmse_tmrt {ck.get('val_rmse_tmrt', float('nan')):.3f}")
    # smp versions differ in key naming; a silent mismatch would render noise
    # that looks like a bad model rather than a bad load.
    if missing or unexpected:
        print(f"  WARNING: missing={len(missing)} unexpected={len(unexpected)}")
        for k in list(missing)[:4]:
            print(f"    missing: {k}")
        for k in list(unexpected)[:4]:
            print(f"    unexpected: {k}")
        if len(missing) > 10:
            print("  ABORT: too many missing keys; this is a version mismatch,")
            print("  not a model problem. Renders would be meaningless.")
            sys.exit(3)
    else:
        print("  V5 state_dict loaded cleanly (0 missing, 0 unexpected)")
    return model.to(device).eval()


def run_v41(inp: np.ndarray) -> np.ndarray | None:
    import onnxruntime as ort
    p = LIVE_MODELS / "umami_unet_v41_fp32.onnx"
    if not p.exists():
        print(f"  V4.1 not found at {p}")
        return None
    sess = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    ishape = sess.get_inputs()[0].shape
    print(f"  V4.1 onnx input {iname} {ishape}")
    if isinstance(ishape[1], int) and ishape[1] != inp.shape[0]:
        print(f"  V4.1 expects {ishape[1]} channels, corpus has {inp.shape[0]} -- skipping")
        return None
    out = sess.run(None, {iname: inp[None].astype(np.float32)})[0]
    return np.asarray(out)[0, 0]


def contrast_stats(field: np.ndarray, shadow: np.ndarray, cdsm: np.ndarray,
                   valid: np.ndarray) -> dict:
    """Sunlit ground vs canopy shade. SOLWEIG shadow: 1 sunlit, 0.03 veg, 0 bldg."""
    canopy = (cdsm > 2.0) & valid
    sunlit = (shadow > 0.9) & valid & ~canopy
    bldg = (shadow < 0.005) & valid
    veg_sh = (shadow > 0.005) & (shadow < 0.2) & valid
    out = {}
    for name, m in (("sunlit", sunlit), ("canopy", canopy),
                    ("bldg_shade", bldg), ("veg_shade", veg_sh)):
        out[name] = float(np.nanmean(field[m])) if m.sum() > 50 else float("nan")
        out[name + "_n"] = int(m.sum())
    out["contrast"] = out["sunlit"] - out["canopy"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", default="patch_0013")
    ap.add_argument("--doy", type=int, default=15)
    ap.add_argument("--hour", type=int, default=13)
    ap.add_argument("--stage-d", type=Path,
                    default=DATA / "training/v5_calib/stage_d_pack_test")
    ap.add_argument("--ckpt", type=Path,
                    default=DATA / "training/v5_calib/models/v5_best.pt")
    ap.add_argument("--out", type=Path,
                    default=DATA / "training/v5_calib/renders")
    args = ap.parse_args()

    stem = f"{args.patch}_d{args.doy:03d}_h{args.hour:02d}"
    print(f"rendering {stem}")

    new_npz = args.stage_d / f"{stem}.npz"
    old_npz = V4_DIR / f"{stem}.npz"
    for p in (new_npz, old_npz, args.ckpt):
        if not p.exists():
            print(f"missing: {p}", file=sys.stderr)
            return 2

    new = np.load(new_npz)
    old = np.load(old_npz)
    inp_v5 = new["input"].astype(np.float32)
    valid = new["valid"].astype(bool)
    label_new = np.where(valid, new["tmrt"], np.nan)
    label_old = np.array(old["tmrt"], dtype=np.float32)

    # SOLWEIG's own shadow for this exact band, for the contrast statistics
    import json
    done = json.loads((STAGE_C / args.patch / "DONE.json").read_text())
    idx = [b["index_in_file"] for b in done["saved_bands"]
           if b["doy"] == args.doy and b["corpus_hour"] == args.hour][0]
    with rasterio.open(STAGE_C / args.patch / "Shadow_0_0.tif") as s:
        shadow = s.read()[idx]
    cdsm = inp_v5[1] * 50.0

    device = torch.device("cpu")
    print("loading models")
    v5 = load_v5(args.ckpt, device)
    with torch.no_grad():
        pred_v5 = v5(torch.from_numpy(inp_v5)[None])[0, 0].numpy()
    pred_v41 = run_v41(old["input"].astype(np.float32))

    panels = [("OLD label (V4 corpus)", label_old),
              ("NEW label (reference SOLWEIG)", label_new)]
    if pred_v41 is not None:
        panels.append(("V4.1 prediction (LIVE)", np.where(valid, pred_v41, np.nan)))
    panels.append(("V5 prediction (candidate)", np.where(valid, pred_v5, np.nan)))

    # ONE shared scale, from the reference label. Everything is judged against
    # the same ruler; per-panel scaling would hide the entire finding.
    lo = float(np.nanpercentile(label_new, 1))
    hi = float(np.nanpercentile(label_new, 99))
    print(f"shared colour scale: {lo:.1f} to {hi:.1f} C (1-99 pct of the reference)")

    print(f"\n{'panel':<32} {'sunlit':>8} {'canopy':>8} {'contrast':>9} "
          f"{'bldg sh':>8}")
    rows = []
    for name, f in panels:
        st = contrast_stats(f, shadow, cdsm, valid)
        rows.append((name, st))
        print(f"{name:<32} {st['sunlit']:>8.2f} {st['canopy']:>8.2f} "
              f"{st['contrast']:>9.2f} {st['bldg_shade']:>8.2f}")
    print(f"\ncanopy px {rows[0][1]['canopy_n']:,}  "
          f"sunlit px {rows[0][1]['sunlit_n']:,}  "
          f"bldg-shade px {rows[0][1]['bldg_shade_n']:,}")

    # TWO rows, because one scale cannot answer both questions.
    #
    #   row 1  ONE SHARED SCALE -> is the MAGNITUDE right?
    #          Panels far below the range go black. That is the magnitude error
    #          being shown honestly, NOT evidence that a panel lacks structure.
    #   row 2  EACH PANEL ON ITS OWN 1-99 PCT -> is there STRUCTURE at all?
    #          This is the fair test of "the old maps are flat", and it is the
    #          view that would EXONERATE the old labels if they were fine.
    #
    # Showing only row 1 would flatter the new model in exactly the way
    # per-panel scaling once flattered the old one.
    n = len(panels)
    fig, axes = plt.subplots(2, n, figsize=(5.2 * n, 11.0))
    for col, ((name, f), (_, st)) in enumerate(zip(panels, rows)):
        ax = axes[0, col]
        im = ax.imshow(f, vmin=lo, vmax=hi, cmap="inferno")
        ax.set_title(f"{name}\ncontrast {st['contrast']:.1f} C", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

        ax2 = axes[1, col]
        p1 = float(np.nanpercentile(f, 1))
        p99 = float(np.nanpercentile(f, 99))
        ax2.imshow(f, vmin=p1, vmax=p99, cmap="inferno")
        ax2.set_title(f"own scale {p1:.1f}-{p99:.1f} C  "
                      f"(span {p99-p1:.1f})", fontsize=9)
        ax2.set_xticks([]); ax2.set_yticks([])

    fig.colorbar(im, ax=axes[0, :].tolist(), fraction=0.020,
                 label="T_mrt (deg C)")
    axes[0, 0].set_ylabel("SHARED SCALE\n(magnitude)", fontsize=10)
    axes[1, 0].set_ylabel("OWN SCALE\n(structure)", fontsize=10)
    fig.suptitle(f"{stem}   (TEST split)\n"
                 f"top: one shared scale {lo:.0f}-{hi:.0f} C  |  "
                 f"bottom: each panel stretched to its own 1-99 pct",
                 fontsize=12)

    args.out.mkdir(parents=True, exist_ok=True)
    png = args.out / f"compare_{stem}.png"
    fig.savefig(png, dpi=110, bbox_inches="tight")
    print(f"\nwrote {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
