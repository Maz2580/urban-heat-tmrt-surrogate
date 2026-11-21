#!/usr/bin/env python3
"""Paper §3.7 / §B.5 latency protocol, one run over every fp32 ONNX (V1 .. V7) so the published V1-V4 numbers act as
controls: single patch (1 x C x 512 x 512) of np.random.rand on a fixed seed, onnxruntime CPU execution provider,
3 warm-up calls then N timed calls, median / p95 / min / max / mean +- sd. Two thread settings: the served path's
(intra 4 / inter 2, app/services/urban_heat_inference.py:158-160) and onnxruntime's defaults. Sequential, one model
at a time. Read-only on the models; writes only under paper/figures/data/.

    docker exec -w /app umami_test nice -n 19 python scripts/urban_heat/paper_figs/bench_onnx_latency.py
"""
from __future__ import annotations
import csv, json, os, platform, statistics, sys, time
from pathlib import Path
import numpy as np
import onnxruntime as ort

REPO = Path(__file__).resolve().parents[3]
MODELS = REPO / "urban_heat_data/models"
OUT = REPO / "docs/report/urbanheat_model_improvement/paper/figures/data"
FILES = [("V1", "umami_unet_fp32.onnx"), ("V2", "umami_unet_v2_fp32.onnx"), ("V3", "umami_unet_v3_fp32.onnx"),
         ("V3.1", "umami_unet_v31_fp32.onnx"), ("V4", "umami_unet_v4_fp32.onnx"), ("V4.1", "umami_unet_v41_fp32.onnx"),
         ("V5", "umami_unet_v5_fp32.onnx"), ("V6", "umami_unet_v6_fp32.onnx"), ("V7", "umami_unet_v7_fp32.onnx")]
N, WARM, SEED = int(os.environ.get("BENCH_N", 100)), 3, 0
SETTINGS = {"served_intra4_inter2": (4, 2), "ort_default": (None, None)}


def bench(path: Path, intra, inter) -> dict:
    so = ort.SessionOptions()
    if intra: so.intra_op_num_threads = intra
    if inter: so.inter_op_num_threads = inter
    s = ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])
    inp = s.get_inputs()[0]; c = int(inp.shape[1])
    x = np.random.default_rng(SEED).random((1, c, 512, 512), dtype=np.float32)
    for _ in range(WARM): s.run(None, {inp.name: x})
    t = []
    for _ in range(N):
        t0 = time.perf_counter(); s.run(None, {inp.name: x}); t.append((time.perf_counter() - t0) * 1000)
    t = sorted(t)
    return {"channels": c, "n": N, "median_ms": round(statistics.median(t), 1), "p95_ms": round(t[int(0.95 * N) - 1], 1),
            "min_ms": round(t[0], 1), "max_ms": round(t[-1], 1), "mean_ms": round(statistics.fmean(t), 1),
            "sd_ms": round(statistics.pstdev(t), 1), "external_data": (path.with_suffix(".onnx.data").exists()),
            "size_mb": round(path.stat().st_size / 1e6, 1)}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {"onnxruntime": ort.__version__, "numpy": np.__version__, "python": platform.python_version(),
            "host": platform.node(), "cpu_count": os.cpu_count(), "platform": platform.platform(),
            "protocol": f"single 1xCx512x512 np.random.default_rng({SEED}).random float32; {WARM} warm-up; {N} timed runs; CPUExecutionProvider",
            "started": time.strftime("%F %T")}
    print(json.dumps(meta), flush=True)
    res = {}
    for setting, (intra, inter) in SETTINGS.items():
        res[setting] = {}
        for tag, fn in FILES:
            p = MODELS / fn
            if not p.exists():
                res[setting][tag] = {"missing": fn}; print(f"  {setting} {tag}: missing", flush=True); continue
            r = bench(p, intra, inter); res[setting][tag] = r
            print(f"  {setting:<22} {tag:<5} C={r['channels']:>2} median {r['median_ms']:>7.1f} ms  p95 {r['p95_ms']:>7.1f}  min {r['min_ms']:>6.1f}  max {r['max_ms']:>6.1f}  mean {r['mean_ms']:.1f}±{r['sd_ms']:.1f}", flush=True)
    stamp = time.strftime("%Y-%m-%d")
    (OUT / f"latency_v1_v7_{stamp}.json").write_text(json.dumps({"meta": meta, "settings": {k: {"intra_op": v[0], "inter_op": v[1]} for k, v in SETTINGS.items()}, "results": res}, indent=1))
    with open(OUT / f"latency_v1_v7_{stamp}.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["setting", "version", "channels", "median_ms", "p95_ms", "min_ms", "max_ms", "mean_ms", "sd_ms", "external_data", "size_mb"])
        for setting, d in res.items():
            for tag, r in d.items():
                if "missing" not in r: w.writerow([setting, tag, r["channels"], r["median_ms"], r["p95_ms"], r["min_ms"], r["max_ms"], r["mean_ms"], r["sd_ms"], r["external_data"], r["size_mb"]])
    print("wrote", OUT / f"latency_v1_v7_{stamp}.json"); return 0


if __name__ == "__main__":
    sys.exit(main())
