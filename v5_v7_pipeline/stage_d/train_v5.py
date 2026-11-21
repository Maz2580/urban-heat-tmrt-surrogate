#!/usr/bin/env python3
"""
Train Urban Heat V5 on the Stage D corpus.
==========================================

Converted from notebooks/UMAMI_UrbanHeat_V4_Training.ipynb so it can run
unattended on the Windows box instead of being babysat in a browser.

WHAT IS DELIBERATELY DIFFERENT FROM V4
--------------------------------------
1. NO WARM-START. V4 warm-started from the V3 checkpoint. V5 must not: those
   weights were fitted to labels from an engine with no surface energy balance
   (`Tground = Ta + 8`), so they encode the wrong physics. `--warm-start` exists
   but defaults OFF and prints a warning when used.

2. NEW FILENAME, NOT REGISTERED. Writes `v5_best.pt` into --out-dir. Nothing
   here touches `urban_heat_data/models/`, which is what live inference reads.
   Registration happens only after the user has seen renders and approved.

3. SAME SPLIT AS V4. Split is by `v2_patch_id` with seed 42, exactly as V4 did,
   so the two are comparable. Do not change the seed.

WHAT IS UNCHANGED ON PURPOSE
----------------------------
Architecture (resnet34 U-Net, 23 in, 3 heads), loss weights, LR schedule,
batch size. V5 changes the LABELS. Changing the model at the same time would
make the result unattributable -- the mistake that cost six versions.

NOTES
-----
* `masked_mse` uses `torch.isfinite(target)`, so Stage D's NaN-where-invalid
  labels are excluded from the loss automatically. That is the mechanism that
  keeps the ~3% invented pixels out of training.
* `tair_sparse` is 100% NaN across the corpus yet carries LOSS_WEIGHTS 10.0.
  Its masked loss is therefore exactly 0 and the head is inert. Left as-is for
  comparability with V4; a warning is printed at startup.
* Output is pure ASCII. Windows pipes stdout through Tee-Object, which makes it
  a non-console stream, and Python then falls back to cp1252 and raises
  UnicodeEncodeError on the first non-ASCII character.

USAGE
    python train_v5.py --corpus <stage_d out dir> --out-dir <models dir>
    python train_v5.py --corpus ... --out-dir ... --resume        # continue
    python train_v5.py --corpus ... --out-dir ... --smoke         # 2 epochs, tiny
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

CHANNEL_ORDER = (
    "dsm_corrected", "cdsm", "albedo",
    "sin_alt", "alt_norm",
    "svf", "building_density", "tree_distance",
    "aspect_sin", "aspect_cos", "shadow_fraction",
    "era5_tair_c",
    "era5_rh_pct", "era5_wind_speed_ms", "era5_ssrd_wm2",
    "ndvi", "geoscape_solar_panel_mask",
    "shadow_h09", "shadow_h13", "shadow_h16",
    "roof_type_flat", "roof_type_moderate", "roof_type_steep",
)
NUM_CHANNELS = len(CHANNEL_ORDER)
HEAD_NAMES = ("tmrt", "lst", "tair_sparse")
LOSS_WEIGHTS = {"tmrt": 1.0, "lst": 1.0, "tair_sparse": 10.0}

SPLIT_SEED = 42          # same as V4; changing it destroys comparability


class V5PatchDataset(Dataset):
    def __init__(self, df: pd.DataFrame, patches_dir: Path):
        self.df = df.reset_index(drop=True)
        self.patches_dir = Path(patches_dir)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        with np.load(self.patches_dir / Path(row["npz_path"]).name) as d:
            return {
                "input": torch.from_numpy(d["input"].astype(np.float32)),
                "tmrt": torch.from_numpy(d["tmrt"].astype(np.float32)),
                "lst": torch.from_numpy(d["lst"].astype(np.float32)),
                "tair_sparse": torch.from_numpy(d["tair_sparse"].astype(np.float32)),
                "doy": int(row["doy"]), "hour": int(row["hour"]),
            }


def masked_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """NaN in the target means 'no measurement here'. Stage D writes NaN exactly
    where the input DSM was missing, so this is what keeps invented pixels out."""
    mask = torch.isfinite(target)
    if mask.sum() == 0:
        return torch.zeros((), device=pred.device, requires_grad=True)
    diff = pred[mask] - target[mask]
    return (diff ** 2).mean()


def multi_task_loss(pred: torch.Tensor, batch: dict, device) -> tuple:
    losses = {}
    for i, head in enumerate(HEAD_NAMES):
        losses[head] = masked_mse(pred[:, i], batch[head].to(device, non_blocking=True))
    total = sum(LOSS_WEIGHTS[h] * losses[h] for h in HEAD_NAMES)
    return total, losses


def epoch_pass(model, loader, device, training: bool, opt=None,
               grad_clip: float = 5.0, amp_dtype=None) -> dict:
    model.train(training)
    accum, n_batches = 0.0, 0
    se = {h: 0.0 for h in HEAD_NAMES}
    npx = {h: 0 for h in HEAD_NAMES}

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for batch in loader:
            x = batch["input"].to(device, non_blocking=True)
            if amp_dtype is not None:
                with torch.autocast(device_type=device.type, dtype=amp_dtype):
                    pred = model(x)
                pred = pred.float()
            else:
                pred = model(x)

            total, _ = multi_task_loss(pred, batch, device)
            if training:
                opt.zero_grad(set_to_none=True)
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()

            accum += float(total.detach().cpu())
            n_batches += 1
            for i, h in enumerate(HEAD_NAMES):
                tgt = batch[h].to(device, non_blocking=True)
                m = torch.isfinite(tgt)
                if m.sum() > 0:
                    d = pred[:, i][m] - tgt[m]
                    se[h] += float((d ** 2).sum().detach().cpu())
                    npx[h] += int(m.sum().detach().cpu())

    rmse = {h: math.sqrt(se[h] / npx[h]) if npx[h] else float("nan") for h in HEAD_NAMES}
    return {"loss": accum / max(n_batches, 1), "rmse": rmse, "pixels": npx}


def build_splits(manifest: pd.DataFrame) -> pd.DataFrame:
    """Split by patch, never by sample, so no patch appears in two splits.

    Two hazards handled explicitly:

    * `Series.unique()` returns an Arrow-backed array under pandas 3.x, and
      `rng.shuffle()` on a non-Sequence is NOT guaranteed to be a permutation --
      numpy warns it "may contain duplicates after shuffling". A duplicate here
      would put one patch in two splits, i.e. train/test leakage that inflates
      the test score with nothing downstream to reveal it. Convert to a real
      numpy string array first.

    * `unique()` returns first-seen order, which depends on row order and on the
      pandas version. Sorting first makes the split reproducible across machines
      and versions, which is what lets `v5` and `v5_zeros` be compared at all.
    """
    ids = np.array(sorted(manifest["v2_patch_id"].astype(str).unique()), dtype="<U64")
    rng = np.random.default_rng(seed=SPLIT_SEED)
    rng.shuffle(ids)

    n_test = int(len(ids) * 0.10)
    n_val = int(len(ids) * 0.10)
    test_ids, val_ids = set(ids[:n_test]), set(ids[n_test:n_test + n_val])
    train_ids = set(ids[n_test + n_val:])

    # Checks that can fail. Cheap, and the failure they guard against is silent.
    assert len(set(ids)) == len(ids), "shuffle produced duplicate patch ids"
    assert not (test_ids & val_ids), "patch in both test and val"
    assert not (test_ids & train_ids), "patch in both test and train"
    assert not (val_ids & train_ids), "patch in both val and train"
    assert len(test_ids | val_ids | train_ids) == len(ids), "patches lost in split"

    manifest = manifest.copy()
    manifest["split"] = manifest["v2_patch_id"].astype(str).apply(
        lambda p: "test" if p in test_ids else "val" if p in val_ids else "train")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", type=Path, required=True,
                    help="Stage D output dir (must contain manifest.parquet)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--tag", default="v5", help="checkpoint prefix, e.g. v5 or v5_zeros")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--resume", action="store_true",
                    help="continue from <tag>_last.pt if present")
    ap.add_argument("--warm-start", type=Path, default=None,
                    help="NOT RECOMMENDED: V4/V3 weights encode the old physics")
    ap.add_argument("--no-amp", action="store_true",
                    help="disable bf16 autocast (CUDA only; roughly halves speed)")
    ap.add_argument("--smoke", action="store_true",
                    help="2 epochs on 200 samples, to prove the pipeline runs")
    args = ap.parse_args()

    try:
        import segmentation_models_pytorch as smp
    except ImportError:
        print("segmentation-models-pytorch is not installed.", file=sys.stderr)
        print("  pip install segmentation-models-pytorch", file=sys.stderr)
        return 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 68)
    print(f"Urban Heat V5 training  [tag={args.tag}]")
    print("=" * 68)
    print(f"device       : {device}"
          + (f"  ({torch.cuda.get_device_name()})" if device.type == "cuda" else ""))
    print(f"torch        : {torch.__version__}")
    if device.type != "cuda":
        print("  WARNING: no CUDA. This will take days, not hours.")

    amp_dtype = None
    if device.type == "cuda" and not args.no_amp:
        amp_dtype = torch.bfloat16
        print("precision    : bf16 autocast (no GradScaler needed)")

    manifest_path = args.corpus / "manifest.parquet"
    if not manifest_path.exists():
        print(f"no manifest at {manifest_path}", file=sys.stderr)
        return 2
    manifest = pd.read_parquet(manifest_path)
    print(f"corpus       : {args.corpus}")
    print(f"manifest     : {len(manifest):,} rows, "
          f"{manifest['v2_patch_id'].nunique()} patches")
    print(f"  has_tmrt   : {int(manifest['has_tmrt'].sum()):,}")
    print(f"  has_lst    : {int(manifest['has_lst'].sum()):,}")

    manifest = manifest[manifest["has_tmrt"]].reset_index(drop=True)
    manifest = build_splits(manifest)
    counts = manifest["split"].value_counts().to_dict()
    print(f"split(seed={SPLIT_SEED}) : {counts}")

    # An empty val split makes val RMSE NaN, and `NaN < best` is always False,
    # so NO best checkpoint is ever saved and you silently end up with only the
    # last epoch. Say so loudly rather than discovering it after a long run.
    for s in ("val", "test"):
        if counts.get(s, 0) == 0:
            print(f"  WARNING: {s} split is EMPTY. RMSE will be NaN"
                  + (" and NO best checkpoint will be saved." if s == "val" else "."))

    if args.smoke:
        manifest = manifest.groupby("split", group_keys=False).head(70)
        args.epochs = 2
        print("SMOKE TEST: 2 epochs on a tiny subset")

    dss = {s: V5PatchDataset(manifest[manifest.split == s], args.corpus)
           for s in ("train", "val", "test")}
    loaders = {
        "train": DataLoader(dss["train"], batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, pin_memory=True,
                            drop_last=True, persistent_workers=args.num_workers > 0),
        "val": DataLoader(dss["val"], batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True,
                          persistent_workers=args.num_workers > 0),
        "test": DataLoader(dss["test"], batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, pin_memory=True),
    }

    # Sanity: the corpus should carry live solar channels unless this is the
    # deliberate ablation. Report rather than assume -- silence here is how the
    # V3/V4 dead-channel defect survived four model versions.
    s0 = dss["train"][0]["input"]
    print(f"ch3 sin_alt  : mean {float(s0[3].mean()):.4f}"
          + ("   <-- ZERO (ablation corpus)" if float(s0[3].mean()) == 0 else ""))
    print(f"tair_sparse  : all-NaN target with LOSS_WEIGHT "
          f"{LOSS_WEIGHTS['tair_sparse']} -> head is inert (V4 behaviour, kept)")

    model = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                     in_channels=NUM_CHANNELS, classes=len(HEAD_NAMES),
                     activation=None)
    print(f"parameters   : {sum(p.numel() for p in model.parameters()):,}")

    # weights_only=True everywhere: torch.load() otherwise unpickles arbitrary
    # objects, which is remote code execution if a checkpoint came from anyone
    # else. Our checkpoints hold only tensors and plain dicts, so this is free.
    if args.warm_start and args.warm_start.exists():
        print("  WARNING: warm-starting. V3/V4 weights were fitted to labels from")
        print("  an engine with no surface energy balance. Any gain becomes")
        print("  unattributable. Proceeding because it was explicitly requested.")
        raw = torch.load(args.warm_start, map_location="cpu", weights_only=True)
        state = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"  loaded; missing={len(missing)}, unexpected={len(unexpected)}")

    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.out_dir / f"{args.tag}_best.pt"
    last_path = args.out_dir / f"{args.tag}_last.pt"
    hist_path = args.out_dir / f"{args.tag}_history.json"

    start_epoch, best_rmse = 0, float("inf")
    history = {k: [] for k in ("train_loss", "val_loss", "val_rmse_tmrt",
                               "val_rmse_lst", "val_rmse_tair")}
    if args.resume and last_path.exists():
        ck = torch.load(last_path, map_location=device, weights_only=True)
        model.load_state_dict(ck["state_dict"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        start_epoch = ck["epoch"] + 1
        best_rmse = ck.get("best_rmse", float("inf"))
        history = ck.get("history", history)
        print(f"RESUMED from epoch {start_epoch} (best tmrt RMSE {best_rmse:.3f})")

    print("=" * 68, flush=True)
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        tr = epoch_pass(model, loaders["train"], device, True, opt, amp_dtype=amp_dtype)
        va = epoch_pass(model, loaders["val"], device, False, amp_dtype=amp_dtype)
        sched.step()

        history["train_loss"].append(tr["loss"])
        history["val_loss"].append(va["loss"])
        history["val_rmse_tmrt"].append(va["rmse"]["tmrt"])
        history["val_rmse_lst"].append(va["rmse"]["lst"])
        history["val_rmse_tair"].append(va["rmse"]["tair_sparse"])

        print(f"epoch {epoch+1:3d}/{args.epochs}  train={tr['loss']:.3f}  "
              f"val={va['loss']:.3f}  rmse[tmrt={va['rmse']['tmrt']:.3f} "
              f"lst={va['rmse']['lst']:.3f}]  lr={sched.get_last_lr()[0]:.2e}  "
              f"{time.time()-t0:.0f}s", flush=True)

        ckpt = {"epoch": epoch, "state_dict": model.state_dict(),
                "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                "val_rmse_tmrt": va["rmse"]["tmrt"], "best_rmse": best_rmse,
                "history": history, "tag": args.tag, "corpus": str(args.corpus)}
        torch.save(ckpt, last_path)                       # resume point, every epoch
        if va["rmse"]["tmrt"] < best_rmse:
            best_rmse = va["rmse"]["tmrt"]
            ckpt["best_rmse"] = best_rmse
            torch.save(ckpt, best_path)
            print(f"  -> best so far ({best_rmse:.3f} C); checkpointed", flush=True)
        hist_path.write_text(json.dumps(history, indent=1))

    # --- test -----------------------------------------------------------
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device,
                                         weights_only=True)["state_dict"])
    te = epoch_pass(model, loaders["test"], device, False, amp_dtype=amp_dtype)
    print()
    print("=" * 68)
    print("TEST SET")
    print("=" * 68)
    for h in HEAD_NAMES:
        print(f"  {h:<14} RMSE {te['rmse'][h]:.3f} C   [{te['pixels'][h]:,} px]")
    print()
    print("  NOTE: V4's pass targets (tmrt <= 3.0 C) were set against the OLD")
    print("  labels, whose mean was ~20 C. The V5 labels average ~50 C with a")
    print("  far wider range, so an RMSE compared across the two is not")
    print("  like-for-like. Judge V5 on renders and on held-out validation,")
    print("  not against a threshold calibrated for a different target.")
    (args.out_dir / f"{args.tag}_test.json").write_text(json.dumps(
        {"rmse": te["rmse"], "pixels": te["pixels"], "corpus": str(args.corpus)}, indent=1))
    # Report what is actually on disk. Announcing a filename the run never wrote
    # is how a missing best-checkpoint goes unnoticed until someone tries to
    # load it.
    print("\nfiles written:")
    for p in (best_path, last_path, hist_path, args.out_dir / f"{args.tag}_test.json"):
        if p.exists():
            print(f"  {p.name:<24} {p.stat().st_size/1e6:8.1f} MB")
        else:
            print(f"  {p.name:<24} NOT WRITTEN")
    if not best_path.exists():
        print("  -> no best checkpoint: val RMSE never improved on inf, which"
              "\n     usually means the val split was empty or all-NaN.")
    return 0


if __name__ == "__main__":
    # Required on Windows: DataLoader workers re-import this module.
    raise SystemExit(main())
