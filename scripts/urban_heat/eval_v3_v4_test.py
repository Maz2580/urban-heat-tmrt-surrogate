"""Offline V3/V4 eval: MAE, R², SSIM, RMSE on the held-out test set.

Reproduces the V3-notebook test split (seed=42, by v2_patch_id, 80/10/10) and
runs ONNX inference per patch, comparing against the ground-truth tmrt label.
"""
import json, time, sys
import numpy as np
import pandas as pd
import onnxruntime as ort
from pathlib import Path
from scipy.ndimage import uniform_filter

PATCH_ROOT_V21 = Path("/app/urban_heat_data/training/patches_v21")
PATCH_ROOT_V4  = Path("/app/urban_heat_data/training/patches_v4")
ONNX_V3 = "/app/urban_heat_data/models/umami_unet_v3_fp32.onnx"
ONNX_V4 = "/app/urban_heat_data/models/umami_unet_v4_fp32.onnx"
OUT_PATH = "/app/docs/report/urbanheat_model_improvement/scratch/offline_eval_results.json"

def ssim_2d(a, b, data_range=63.0, win=11, K1=0.01, K2=0.03):
    """SSIM with uniform window. Returns scalar mean SSIM over valid interior."""
    a = a.astype(np.float64); b = b.astype(np.float64)
    C1 = (K1*data_range)**2; C2 = (K2*data_range)**2
    mu_a = uniform_filter(a, win); mu_b = uniform_filter(b, win)
    sig_a2 = uniform_filter(a*a, win) - mu_a*mu_a
    sig_b2 = uniform_filter(b*b, win) - mu_b*mu_b
    sig_ab = uniform_filter(a*b, win) - mu_a*mu_b
    num = (2*mu_a*mu_b + C1) * (2*sig_ab + C2)
    den = (mu_a*mu_a + mu_b*mu_b + C1) * (sig_a2 + sig_b2 + C2)
    pad = win // 2
    s_map = num[pad:-pad, pad:-pad] / den[pad:-pad, pad:-pad]
    return float(s_map.mean())

def reproduce_test_split(manifest):
    rng = np.random.default_rng(seed=42)
    ids = manifest['v2_patch_id'].unique()
    rng.shuffle(ids)
    n = int(len(ids) * 0.10)
    test_ids = set(ids[:n])
    return manifest[manifest['v2_patch_id'].isin(test_ids)].reset_index(drop=True)

def per_patch_metrics(pred, target):
    mask = np.isfinite(target)
    if mask.sum() == 0:
        return None
    p = pred[mask]; t = target[mask]
    rmse = float(np.sqrt(((p - t)**2).mean()))
    mae  = float(np.abs(p - t).mean())
    ss_res = ((p - t)**2).sum(); ss_tot = ((t - t.mean())**2).sum()
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float('nan')
    pred_full = np.where(mask, pred, 0.0)
    targ_full = np.where(mask, target, 0.0)
    s = ssim_2d(pred_full, targ_full, data_range=63.0)
    return rmse, mae, r2, s

def run_eval(version, manifest_path, patches_root, onnx_path, head_idx, label_key='tmrt'):
    print(f"\n=== {version} eval ===", flush=True)
    manifest = pd.read_parquet(manifest_path)
    test = reproduce_test_split(manifest)
    print(f"Test rows: {len(test)} (from {manifest['v2_patch_id'].nunique()} v2_patch_ids)", flush=True)
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    rmses, maes, r2s, ssims = [], [], [], []
    pixel_count = 0
    se_total, ae_total = 0.0, 0.0
    t0 = time.time()
    for i, row in test.iterrows():
        npz_path = patches_root / Path(row['npz_path']).name
        if not npz_path.exists(): continue
        try:
            d = np.load(npz_path)
            x = d['input'].astype(np.float32)[None, ...]
            y = d[label_key].astype(np.float32)
        except Exception:
            continue
        out = sess.run(None, {"input": x})[0]
        pred = out[0, head_idx]
        m = per_patch_metrics(pred, y)
        if m is None: continue
        rmse, mae, r2, s = m
        rmses.append(rmse); maes.append(mae); r2s.append(r2); ssims.append(s)
        mask = np.isfinite(y)
        n = int(mask.sum())
        se_total += float(((pred[mask] - y[mask])**2).sum())
        ae_total += float(np.abs(pred[mask] - y[mask]).sum())
        pixel_count += n
        if (i+1) % 100 == 0:
            print(f"  {i+1}/{len(test)} | {time.time()-t0:.1f}s | running RMSE_pixel={np.sqrt(se_total/pixel_count):.3f}", flush=True)
    elapsed = time.time() - t0
    n_patches = len(rmses)
    pixel_rmse = float(np.sqrt(se_total/pixel_count)) if pixel_count else float('nan')
    pixel_mae  = float(ae_total/pixel_count) if pixel_count else float('nan')
    print(f"\n{version}: {n_patches} patches, {pixel_count:,} px, {elapsed:.0f}s")
    print(f"  RMSE px-wgt:    {pixel_rmse:.3f} °C")
    print(f"  RMSE patch-med: {np.median(rmses):.3f} °C")
    print(f"  MAE  px-wgt:    {pixel_mae:.3f} °C")
    print(f"  MAE  patch-med: {np.median(maes):.3f} °C")
    print(f"  R²   patch-mean:   {np.mean(r2s):.4f}")
    print(f"  R²   patch-median: {np.median(r2s):.4f}")
    print(f"  SSIM patch-mean:   {np.mean(ssims):.4f}")
    print(f"  SSIM patch-median: {np.median(ssims):.4f}")
    return {'version': version, 'n_patches': n_patches, 'pixel_count': pixel_count,
            'rmse_pixel': pixel_rmse, 'rmse_patch_median': float(np.median(rmses)),
            'mae_pixel': pixel_mae, 'mae_patch_median': float(np.median(maes)),
            'r2_patch_mean': float(np.mean(r2s)), 'r2_patch_median': float(np.median(r2s)),
            'ssim_patch_mean': float(np.mean(ssims)), 'ssim_patch_median': float(np.median(ssims)),
            'elapsed_s': elapsed}

results = []
results.append(run_eval('V3', PATCH_ROOT_V21/'manifest.parquet', PATCH_ROOT_V21, ONNX_V3, 0))
results.append(run_eval('V4', PATCH_ROOT_V4/'manifest.parquet', PATCH_ROOT_V4, ONNX_V4, 0))
print("\n=== SUMMARY ===")
print(json.dumps(results, indent=2))
Path(OUT_PATH).parent.mkdir(parents=True, exist_ok=True)
with open(OUT_PATH, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {OUT_PATH}")
