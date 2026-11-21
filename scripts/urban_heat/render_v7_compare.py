#!/usr/bin/env python3
r"""V7 vs V5 RENDERS -- the check every metric can pass and the map still fail.

For the test patches with the MOST building shade at the chosen hour (not
hand-picked), draw on ONE shared colour scale:

    label | V5 pred | V7 pred | V5 - label | V7 - label

with the TRUE building-shadow outline (labels-world planes, out_v7) on every
panel and the canopy outline dashed. The question to answer by eye is the
team's: are building shadows rendered where, and as cool as, they should be?

Each model is fed ITS OWN corpus (V5 <- out, V7 <- out_v7); the manifest is
shared so row i is the same patch/doy/hour in both. Pixels where the label is
NaN are blanked in every panel.

Read-only except the PNGs. venv_train (needs torch + smp + matplotlib).

    & .\venv_train\Scripts\python.exe stage_d\render_v7_compare.py `
        --v5-corpus stage_d\out --v5-ckpt stage_d\models\v5_best.pt `
        --v7-corpus stage_d\out_v7 --v7-ckpt stage_d\models\v7_best.pt `
        --hour 13 --n 8 --out-dir stage_d\renders_v7
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_v5 as T                       # noqa: E402
import ceiling_test as C                   # noqa: E402


def load(ckpt, device):
    import segmentation_models_pytorch as smp
    m = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                 in_channels=T.NUM_CHANNELS, classes=len(T.HEAD_NAMES),
                 activation=None)
    ck = torch.load(ckpt, map_location=device, weights_only=True)
    m.load_state_dict(ck["state_dict"])
    return m.to(device).eval()


def predict(model, x, device):
    with torch.no_grad(), torch.autocast(device_type=device.type,
                                         dtype=torch.bfloat16,
                                         enabled=device.type == "cuda"):
        p = model(x[None].to(device))
    return p.float()[0, 0].cpu().numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v5-corpus", type=Path, required=True)
    ap.add_argument("--v5-ckpt", type=Path, required=True)
    ap.add_argument("--v7-corpus", type=Path, required=True)
    ap.add_argument("--v7-ckpt", type=Path, required=True)
    ap.add_argument("--hour", type=int, default=13)
    ap.add_argument("--doy", type=int, default=None,
                    help="restrict to one DOY (default: any)")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_parquet(args.v7_corpus / "manifest.parquet")
    manifest = manifest[manifest["has_tmrt"]].reset_index(drop=True)
    manifest = T.build_splits(manifest)               # SPLIT_SEED = 42
    test = manifest[(manifest.split == "test") & (manifest.hour == args.hour)]
    if args.doy is not None:
        test = test[test.doy == args.doy]
    test = test.reset_index(drop=True)
    print(f"test samples at h{args.hour:02d}: {len(test)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    v5, v7 = load(args.v5_ckpt, device), load(args.v7_ckpt, device)
    ds5 = T.V5PatchDataset(test, args.v5_corpus)
    ds7 = T.V5PatchDataset(test, args.v7_corpus)
    sh_idx = C.CH_SHADOW[C.nearest_shadow_hour(args.hour)]

    # rank by TRUE non-canopy building shade fraction (labels-world planes)
    frac = []
    for i in range(len(ds7)):
        x = ds7[i]["input"]
        shade = (x[sh_idx] > 0.5) & (x[C.CH_CDSM] <= C.CANOPY_M)
        frac.append(float(shade.float().mean()))
    order = np.argsort(frac)[::-1][:args.n]
    print("selected (patch, doy, shade%):",
          [(Path(test.iloc[i].npz_path).stem, int(test.iloc[i].doy),
            round(100 * frac[i], 1)) for i in order])

    summary = []
    for i in order:
        row = test.iloc[i]
        s5, s7 = ds5[i], ds7[i]
        lab = s7["tmrt"].numpy()
        valid = np.isfinite(lab)
        p5, p7 = predict(v5, s5["input"], device), predict(v7, s7["input"], device)
        x7 = s7["input"].numpy()
        shade = (x7[sh_idx] > 0.5) & (x7[C.CH_CDSM] <= C.CANOPY_M)
        canopy = x7[C.CH_CDSM] > C.CANOPY_M
        for a in (p5, p7):
            a[~valid] = np.nan
        lo, hi = np.nanpercentile(lab, 1), np.nanpercentile(lab, 99)
        err = 8.0
        m5 = float(np.nanmean((p5 - lab)[shade & valid]))
        m7 = float(np.nanmean((p7 - lab)[shade & valid]))
        r5 = float(np.sqrt(np.nanmean(((p5 - lab)[shade & valid]) ** 2)))
        r7 = float(np.sqrt(np.nanmean(((p7 - lab)[shade & valid]) ** 2)))
        summary.append((row.npz_path, int(row.doy), m5, m7, r5, r7))

        fig, ax = plt.subplots(1, 5, figsize=(25, 5.4))
        panels = [(lab, "SOLWEIG label", "inferno", lo, hi),
                  (p5, f"V5  (shade bias {m5:+.2f}, rmse {r5:.2f})", "inferno", lo, hi),
                  (p7, f"V7  (shade bias {m7:+.2f}, rmse {r7:.2f})", "inferno", lo, hi),
                  (p5 - lab, "V5 - label", "RdBu_r", -err, err),
                  (p7 - lab, "V7 - label", "RdBu_r", -err, err)]
        for a, (img, title, cmap, vmin, vmax) in zip(ax, panels):
            im = a.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
            a.contour(shade.astype(float), levels=[0.5], colors="cyan", linewidths=0.6)
            a.contour(canopy.astype(float), levels=[0.5], colors="lime",
                      linewidths=0.5, linestyles="dashed")
            a.set_title(title, fontsize=10); a.set_xticks([]); a.set_yticks([])
            plt.colorbar(im, ax=a, fraction=0.046, pad=0.02)
        stem = Path(row.npz_path).stem
        fig.suptitle(f"{stem}  doy {row.doy}  h{args.hour:02d}   "
                     f"cyan = true building shadow (labels world), "
                     f"green dashed = canopy; T_mrt C, shared scale "
                     f"[{lo:.0f}, {hi:.0f}]", fontsize=11)
        fig.tight_layout()
        out = args.out_dir / f"{stem}_h{args.hour:02d}_v5_vs_v7.png"
        fig.savefig(out, dpi=110); plt.close(fig)
        print(f"  wrote {out.name}   V5 shade {m5:+.2f}/{r5:.2f}   "
              f"V7 shade {m7:+.2f}/{r7:.2f}")

    print("\nmean over selected patches (in true building shade):")
    a = np.array([s[2:] for s in summary])
    print(f"  V5 bias {a[:,0].mean():+.2f}  rmse {a[:,2].mean():.2f}"
          f"   V7 bias {a[:,1].mean():+.2f}  rmse {a[:,3].mean():.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
