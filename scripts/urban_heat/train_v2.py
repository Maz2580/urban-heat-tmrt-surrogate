#!/usr/bin/env python3
"""
UMAMI Urban Heat — U-Net V2 CPU Training Script (Phase 1.6.9)
=============================================================

Standalone training script for the V2 surrogate model with additional
input features. Runs on CPU (no GPU required). Designed to be launched
as a subprocess by the API and polled via progress.json.

Differences from V1 (Colab):
  - CPU-only: BS=4, num_workers=2, no pin_memory
  - Configurable input channels: 5 (base) + N extra features
  - First conv copies pretrained RGB weights, zero-inits extra channels
  - Perceptual loss optional (--no-perceptual to save CPU)
  - CosineAnnealingWarmRestarts scheduler
  - Checkpoint every epoch + best model
  - Progress JSON for API polling
  - STOP file for graceful halt
  - ONNX export at completion

Usage:
  python scripts/urban_heat/train_v2.py \\
    --channels 7 --extra-features svf building_density \\
    --epochs 200 --batch-size 4 --lr 2e-4 --job-id v2_svf_density

  # Monitor:
  cat urban_heat_data/models/v2_training/<job_id>/progress.json

  # Graceful stop:
  touch urban_heat_data/models/v2_training/<job_id>/STOP
"""

import argparse
import json
import math
import os
import re
import signal
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Graceful SIGTERM handling — allows proc.terminate() to stop training
# ---------------------------------------------------------------------------
_stop_requested = False

def _sigterm_handler(signum, frame):
    global _stop_requested
    _stop_requested = True
    print(f"\nSIGTERM received — will stop after current batch.", flush=True)

signal.signal(signal.SIGTERM, _sigterm_handler)

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PATCHES_DIR = BASE_DIR / "urban_heat_data" / "training" / "patches"
STATS_FILE = BASE_DIR / "urban_heat_data" / "models" / "dataset_stats.json"
MODELS_DIR = BASE_DIR / "urban_heat_data" / "models"


# ---------------------------------------------------------------------------
# Solar position (same as inference pipeline)
# ---------------------------------------------------------------------------

def compute_solar_position(lat, lon, doy, hour, utc_offset=10):
    solar_hour = hour - utc_offset + lon / 15.0
    dec = -23.45 * math.cos(math.radians(360 / 365 * (doy + 10)))
    ha = 15 * (solar_hour - 12)
    lat_r, dec_r, ha_r = math.radians(lat), math.radians(dec), math.radians(ha)
    sin_alt = (
        math.sin(lat_r) * math.sin(dec_r)
        + math.cos(lat_r) * math.cos(dec_r) * math.cos(ha_r)
    )
    altitude = math.degrees(math.asin(max(-1, min(1, sin_alt))))
    if math.cos(math.radians(altitude)) < 0.001:
        return altitude, 180.0
    cos_az = (
        math.sin(dec_r) - math.sin(lat_r) * sin_alt
    ) / (math.cos(lat_r) * math.cos(math.radians(altitude)))
    azimuth = math.degrees(math.acos(max(-1, min(1, cos_az))))
    if ha > 0:
        azimuth = 360 - azimuth
    return altitude, azimuth


# Precompute for Melbourne's 12 temporal combos
SOLAR_CACHE = {}
for _doy, _hour in [
    (356, 9), (356, 11), (356, 13), (356, 15),
    (15, 9), (15, 11), (15, 13), (15, 15),
    (46, 9), (46, 13), (306, 9), (306, 13),
]:
    alt, az = compute_solar_position(-37.81, 144.96, _doy, _hour)
    SOLAR_CACHE[(_doy, _hour)] = (alt, az)

TARGET_RE = re.compile(r"_d(\d{3})_h(\d{2})_target\.npy$")


def parse_temporal(path_str):
    m = TARGET_RE.search(path_str)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class UrbanHeatDatasetV2(Dataset):
    """
    Loads 5-channel base input + optional extra feature channels.

    Base: (5, 512, 512) = DSM + CDSM + Albedo + sin(alt) + alt/90
    Extra: (N, 512, 512) from {patch}_{feature}.npy files
    Total: (5+N, 512, 512)
    Target: (1, 512, 512)
    """

    def __init__(self, patches_dir, patch_names, extra_features=None, augment=False):
        self.patches_dir = Path(patches_dir)
        self.extra_features = extra_features or []
        self.augment = augment
        self.pairs = []

        for base_name in patch_names:
            for variant in [base_name, f"{base_name}_green", f"{base_name}_cool"]:
                input_path = self.patches_dir / f"{variant}_input.npy"
                if not input_path.exists():
                    continue

                # Check extra features exist for this patch
                extra_paths = []
                all_exist = True
                for feat in self.extra_features:
                    fp = self.patches_dir / f"{variant}_{feat}.npy"
                    if not fp.exists():
                        all_exist = False
                        break
                    extra_paths.append(fp)

                if not all_exist and self.extra_features:
                    continue

                for tf in sorted(self.patches_dir.glob(f"{variant}_d*_target.npy")):
                    doy, hour = parse_temporal(tf.name)
                    if doy is None:
                        continue
                    alt, az = SOLAR_CACHE.get(
                        (doy, hour),
                        compute_solar_position(-37.81, 144.96, doy, hour),
                    )
                    self.pairs.append((input_path, tf, alt, az, extra_paths))

        print(f"  Dataset: {len(self.pairs)} pairs from {len(patch_names)} base patches "
              f"(+{len(self.extra_features)} extra features)")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        input_path, target_path, sun_alt, sun_az, extra_paths = self.pairs[idx]

        # Base 3-channel input
        inp = torch.from_numpy(np.load(input_path).astype(np.float32))
        tgt = torch.from_numpy(np.load(target_path).astype(np.float32))

        # Add solar channels
        h, w = inp.shape[1], inp.shape[2]
        ch_sin = torch.full((1, h, w), math.sin(math.radians(max(sun_alt, 0))), dtype=torch.float32)
        ch_norm = torch.full((1, h, w), max(sun_alt, 0) / 90.0, dtype=torch.float32)
        inp = torch.cat([inp, ch_sin, ch_norm], dim=0)  # (5, H, W)

        # Add extra feature channels
        for fp in extra_paths:
            feat = torch.from_numpy(np.load(fp).astype(np.float32))
            inp = torch.cat([inp, feat], dim=0)

        # Augmentation (flip only — rotation breaks solar direction)
        if self.augment:
            if torch.rand(1).item() > 0.5:
                inp, tgt = torch.flip(inp, [2]), torch.flip(tgt, [2])
            if torch.rand(1).item() > 0.5:
                inp, tgt = torch.flip(inp, [1]), torch.flip(tgt, [1])

        return inp, tgt


# ---------------------------------------------------------------------------
# Model (same architecture as V1, parameterized input channels)
# ---------------------------------------------------------------------------

class ResNetEncoder(nn.Module):
    def __init__(self, in_ch=5, pretrained=True):
        super().__init__()
        resnet = models.resnet34(
            weights=models.ResNet34_Weights.DEFAULT if pretrained else None
        )
        self.conv1 = nn.Conv2d(in_ch, 64, 7, stride=2, padding=3, bias=False)
        if pretrained:
            with torch.no_grad():
                self.conv1.weight[:, :3] = resnet.conv1.weight.data
                self.conv1.weight[:, 3:] = 0.0
        self.bn1 = nn.GroupNorm(8, 64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = self._replace_bn(resnet.layer1)
        self.layer2 = self._replace_bn(resnet.layer2)
        self.layer3 = self._replace_bn(resnet.layer3)
        self.layer4 = self._replace_bn(resnet.layer4)

    @staticmethod
    def _replace_bn(layer):
        for name, mod in layer.named_modules():
            if isinstance(mod, nn.BatchNorm2d):
                parts = name.split(".")
                parent = layer
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                setattr(parent, parts[-1], nn.GroupNorm(8, mod.num_features))
        return layer

    def forward(self, x):
        e0 = self.relu(self.bn1(self.conv1(x)))
        e1 = self.layer1(self.maxpool(e0))
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        e4 = self.layer4(e3)
        return [e0, e1, e2, e3, e4]


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch // 2 + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape != skip.shape:
            x = nn.functional.interpolate(
                x, size=skip.shape[2:], mode="bilinear", align_corners=False
            )
        return self.conv(torch.cat([x, skip], dim=1))


class UNetResNet34(nn.Module):
    def __init__(self, in_ch=5, out_ch=1, pretrained=True):
        super().__init__()
        self.encoder = ResNetEncoder(in_ch, pretrained)
        self.dec4 = DecoderBlock(512, 256, 256)
        self.dec3 = DecoderBlock(256, 128, 128)
        self.dec2 = DecoderBlock(128, 64, 64)
        self.dec1 = DecoderBlock(64, 64, 64)
        self.final_up = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_ch, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        e0, e1, e2, e3, e4 = self.encoder(x)
        d4 = self.dec4(e4, e3)
        d3 = self.dec3(d4, e2)
        d2 = self.dec2(d3, e1)
        d1 = self.dec1(d2, e0)
        return self.final_conv(self.final_up(d1))


# ---------------------------------------------------------------------------
# Perceptual loss (optional)
# ---------------------------------------------------------------------------

class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        self.slice1 = nn.Sequential(*list(vgg.features[:4]))
        self.slice2 = nn.Sequential(*list(vgg.features[4:9]))
        self.slice3 = nn.Sequential(*list(vgg.features[9:16]))
        for p in self.parameters():
            p.requires_grad = False
        self.weights = [1.0, 0.5, 0.25]

    def forward(self, pred, target):
        p3 = pred.expand(-1, 3, -1, -1)
        t3 = target.expand(-1, 3, -1, -1)
        loss = 0.0
        for i, layer in enumerate([self.slice1, self.slice2, self.slice3]):
            p3, t3 = layer(p3), layer(t3)
            loss += self.weights[i] * nn.functional.l1_loss(p3, t3)
        return loss


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(pred_batch, target_batch, tmrt_max=63.0):
    pred_np = pred_batch.detach().cpu().numpy() * tmrt_max
    tgt_np = target_batch.detach().cpu().numpy() * tmrt_max
    p, t = pred_np.reshape(-1), tgt_np.reshape(-1)
    rmse = float(np.sqrt(np.mean((p - t) ** 2)))
    mae = float(np.mean(np.abs(p - t)))
    ss_res = np.sum((t - p) ** 2)
    ss_tot = np.sum((t - np.mean(t)) ** 2)
    r2 = float(1 - ss_res / max(ss_tot, 1e-8))
    return {"rmse": rmse, "mae": mae, "r2": r2}


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------

def write_progress(job_dir, data):
    path = job_dir / "progress.json"
    # Replace inf/nan with None for JSON compatibility
    clean = {}
    for k, v in data.items():
        if isinstance(v, float) and (math.isinf(v) or math.isnan(v)):
            clean[k] = None
        else:
            clean[k] = v
    with open(path, "w") as f:
        json.dump(clean, f, indent=2)


def check_stop(job_dir):
    return _stop_requested or (job_dir / "STOP").exists()


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

def export_onnx(model, in_channels, job_dir, filename="umami_unet_v2_fp32.onnx"):
    model_cpu = model.cpu()
    model_cpu.train(False)
    dummy = torch.randn(1, in_channels, 512, 512)
    onnx_path = job_dir / filename

    torch.onnx.export(
        model_cpu,
        dummy,
        str(onnx_path),
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )
    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    print(f"ONNX exported: {onnx_path} ({size_mb:.1f} MB)")
    return str(onnx_path)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args):
    # Setup job directory
    job_dir = MODELS_DIR / "v2_training" / args.job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    print(f"=" * 60)
    print(f"UMAMI U-Net V2 Training — Job: {args.job_id}")
    print(f"=" * 60)
    print(f"Channels: {args.channels} (5 base + {args.channels - 5} extra)")
    print(f"Extra features: {args.extra_features}")
    print(f"Epochs: {args.epochs}, BS: {args.batch_size}, LR: {args.lr}")
    print(f"Perceptual loss: {not args.no_perceptual}")
    print(f"Device: CPU")
    print(f"Output: {job_dir}")
    print()

    # Load dataset stats
    with open(STATS_FILE) as f:
        stats = json.load(f)

    splits = stats["splits"]
    tmrt_max = stats["normalization"]["tmrt_range"][1]

    # Create datasets
    extra_feats = args.extra_features if args.extra_features else []
    print("Loading training data...")
    train_ds = UrbanHeatDatasetV2(
        PATCHES_DIR, splits["train_patches"], extra_feats, augment=True
    )
    print("Loading validation data...")
    val_ds = UrbanHeatDatasetV2(PATCHES_DIR, splits["val_patches"], extra_feats)

    if len(train_ds) == 0:
        print("ERROR: No training pairs found. Check that extra features are computed.")
        sys.exit(1)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    print(f"Train: {len(train_ds)} pairs, {len(train_loader)} batches")
    print(f"Val: {len(val_ds)} pairs, {len(val_loader)} batches")

    # Verify input shape
    sample_inp, sample_tgt = train_ds[0]
    print(f"Input shape: {sample_inp.shape} (expected {args.channels}x512x512)")
    assert sample_inp.shape[0] == args.channels, (
        f"Expected {args.channels} channels, got {sample_inp.shape[0]}"
    )

    # Model
    model = UNetResNet34(args.channels, 1, pretrained=True)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {params:,}")

    # Loss
    l1_fn = nn.L1Loss()
    perc_fn = None
    perc_w = 0.0
    if not args.no_perceptual:
        print("Loading VGG16 for perceptual loss...")
        perc_fn = PerceptualLoss()
        perc_w = 0.1

    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-6)

    # Resume from checkpoint
    best_val_rmse = float("inf")
    start_epoch = 0
    latest_ckpt = job_dir / "latest_model.pt"

    if latest_ckpt.exists() and not args.fresh:
        ckpt = torch.load(latest_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val_rmse = ckpt.get("best_val_rmse", float("inf"))
        print(f"Resumed from epoch {start_epoch}, best RMSE: {best_val_rmse:.4f}")

    # Save config
    config = {
        "job_id": args.job_id,
        "channels": args.channels,
        "extra_features": extra_feats,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "perceptual_loss": not args.no_perceptual,
        "train_pairs": len(train_ds),
        "val_pairs": len(val_ds),
        "model_params": params,
        "tmrt_max": tmrt_max,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(job_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Training loop
    print(f"\nStarting training from epoch {start_epoch}...")
    t_train_start = time.time()

    for epoch in range(start_epoch, args.epochs):
        if check_stop(job_dir):
            print(f"\nSTOP signal received at epoch {epoch}. Saving and exiting.")
            break

        t_epoch = time.time()
        model.train()
        train_loss_sum = 0.0
        n_batches = 0

        total_batches = len(train_loader)
        for batch_idx, (inp, tgt) in enumerate(train_loader):
            if batch_idx % 50 == 0 and check_stop(job_dir):
                print(f"\nSTOP signal received at epoch {epoch}, batch {batch_idx}.")
                break
            optimizer.zero_grad()
            pred = model(inp)

            l1 = l1_fn(pred, tgt)
            loss = l1
            if perc_fn is not None:
                loss = l1 + perc_w * perc_fn(pred, tgt)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss_sum += loss.item()
            n_batches += 1

            # Batch-level progress reporting every 50 batches
            if (batch_idx + 1) % 50 == 0 or batch_idx == total_batches - 1:
                avg_loss = train_loss_sum / max(n_batches, 1)
                elapsed = time.time() - t_epoch
                eta_epoch = (elapsed / (batch_idx + 1)) * (total_batches - batch_idx - 1)
                print(
                    f"  batch {batch_idx+1}/{total_batches} "
                    f"loss={avg_loss:.4f} "
                    f"elapsed={elapsed:.0f}s eta={eta_epoch:.0f}s",
                    flush=True,
                )
                write_progress(job_dir, {
                    "status": "training",
                    "epoch": epoch,
                    "epochs_total": args.epochs,
                    "batch": batch_idx + 1,
                    "batches_total": total_batches,
                    "train_loss": avg_loss,
                    "elapsed_epoch_s": round(elapsed, 1),
                    "eta_epoch_s": round(eta_epoch, 1),
                    "elapsed_total_s": round(time.time() - t_train_start, 1),
                    "best_val_rmse": best_val_rmse,
                })

        if check_stop(job_dir):
            break

        scheduler.step()

        # Validation
        model.train(False)
        val_metrics = {"rmse": [], "mae": [], "r2": []}
        with torch.no_grad():
            for inp, tgt in val_loader:
                m = compute_metrics(model(inp), tgt, tmrt_max)
                for k in val_metrics:
                    val_metrics[k].append(m[k])

        vr = float(np.mean(val_metrics["rmse"]))
        vm = float(np.mean(val_metrics["mae"]))
        vr2 = float(np.mean(val_metrics["r2"]))
        train_loss_avg = train_loss_sum / max(n_batches, 1)
        epoch_time = time.time() - t_epoch
        lr_now = optimizer.param_groups[0]["lr"]

        is_best = vr < best_val_rmse
        if is_best:
            best_val_rmse = vr

        print(
            f"E{epoch:03d} ({epoch_time:.0f}s) Loss={train_loss_avg:.4f} "
            f"RMSE={vr:.3f} MAE={vm:.3f} R2={vr2:.4f} LR={lr_now:.2e}"
            f"{' *** BEST' if is_best else ''}"
        )

        # Save checkpoints
        ckpt_data = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val_rmse": best_val_rmse,
            "val_metrics": {"rmse": vr, "mae": vm, "r2": vr2},
            "config": config,
        }
        torch.save(ckpt_data, latest_ckpt)
        if is_best:
            torch.save(ckpt_data, job_dir / "best_model.pt")
        if (epoch + 1) % 25 == 0:
            torch.save(ckpt_data, job_dir / f"model_e{epoch:03d}.pt")

        # Update progress JSON
        elapsed = time.time() - t_train_start
        eta_seconds = (elapsed / max(epoch - start_epoch + 1, 1)) * (args.epochs - epoch - 1)
        write_progress(job_dir, {
            "status": "training",
            "epoch": epoch,
            "total_epochs": args.epochs,
            "train_loss": round(train_loss_avg, 6),
            "val_rmse": round(vr, 4),
            "val_mae": round(vm, 4),
            "val_r2": round(vr2, 4),
            "best_val_rmse": round(best_val_rmse, 4),
            "lr": lr_now,
            "epoch_time_s": round(epoch_time, 1),
            "elapsed_s": round(elapsed, 1),
            "eta_s": round(eta_seconds, 1),
            "is_best": is_best,
        })

    # Final: export ONNX
    print("\nTraining complete. Loading best model for ONNX export...")
    best_ckpt_path = job_dir / "best_model.pt"
    if best_ckpt_path.exists():
        best_ckpt = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(best_ckpt["model"])
        onnx_path = export_onnx(model, args.channels, job_dir)

        # Final progress
        write_progress(job_dir, {
            "status": "completed",
            "epoch": best_ckpt["epoch"],
            "total_epochs": args.epochs,
            "best_val_rmse": round(best_val_rmse, 4),
            "val_metrics": best_ckpt.get("val_metrics", {}),
            "onnx_path": onnx_path,
            "elapsed_s": round(time.time() - t_train_start, 1),
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })
        print(f"\nDone! Best RMSE: {best_val_rmse:.3f} C")
    else:
        write_progress(job_dir, {
            "status": "completed_no_improvement",
            "elapsed_s": round(time.time() - t_train_start, 1),
        })
        print("\nNo best model saved (no improvement over baseline).")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="UMAMI U-Net V2 CPU Training")
    parser.add_argument("--job-id", required=True, help="Unique job identifier")
    parser.add_argument("--channels", type=int, default=5, help="Total input channels (5=base, 7=base+svf+density)")
    parser.add_argument("--extra-features", nargs="*", default=[], help="Extra feature names (e.g. svf building_density)")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--no-perceptual", action="store_true", help="Disable perceptual loss (saves CPU)")
    parser.add_argument("--fresh", action="store_true", help="Start fresh (ignore existing checkpoints)")
    args = parser.parse_args()

    # Validate channel count — some features produce >1 channel (e.g. aspect = 2)
    FEATURE_CHANNELS = {"svf": 1, "building_density": 1, "tree_distance": 1, "aspect": 2, "shadow_fraction": 1}
    expected = 5 + sum(FEATURE_CHANNELS.get(f, 1) for f in args.extra_features)
    if args.channels != expected:
        print(f"NOTE: Adjusting --channels={args.channels} → {expected} (5 base + {expected - 5} feature channels)")
        args.channels = expected

    train(args)


if __name__ == "__main__":
    main()
