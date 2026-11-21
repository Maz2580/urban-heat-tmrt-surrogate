# UMAMI Urban Heat V2.1 — Training Workflow (B1 + B2)

## Inputs already on Drive

```
gdrive:UMAMI/urban_heat_v3/
├── patches_v21.tar           # 30.4 GB tar of 11,844 .npz + manifest.parquet
├── models/
│   └── best_model_v2.pt      # V2 checkpoint (280 MB) for warm-start
├── UMAMI_UrbanHeat_V21_Training.ipynb
└── README_V21_TRAINING.md    # this file
```

The corpus is shipped as a single tar instead of thousands of loose
files because Drive throttles per-file metadata and would take hours
streaming 11,844 small files. The notebook's cell 1.5 copies the tar
to `/content/patches_v21.tar` and extracts it to `/content/patches_v21/`
on the Colab VM so DataLoader hits local SSD during training (~5-10 min
prep, then training reads at full SSD speed).

## How everything got there (server-side, already done)

```bash
# 1. Build the corpus (server, ~17 min, multi-process)
docker exec -e PYTHONPATH=/app umami_test python /app/scripts/urban_heat/build_v21_corpus.py \
    --workers 8 --no-lst

# 2. Tar the patches (~2-3 min on the umami_test container)
docker exec umami_test tar -cf /app/urban_heat_data/training/patches_v21.tar \
    -C /app/urban_heat_data/training patches_v21/

# 3. rclone the tar to Drive (~37 min at 14 MiB/s)
rclone copyto /mnt/apps/spatial-llm/src/umami_spatial-llm/urban_heat_data/training/patches_v21.tar \
    gdrive:UMAMI/urban_heat_v3/patches_v21.tar --stats 30s

# 4. V2 checkpoint (renamed in flight)
rclone copyto /mnt/apps/spatial-llm/src/umami_spatial-llm/urban_heat_data/models/v2_training/v2_all_features/best_model.pt \
    gdrive:UMAMI/urban_heat_v3/models/best_model_v2.pt

# 5. Notebook + README
rclone copy /mnt/apps/spatial-llm/src/umami_spatial-llm/notebooks/UMAMI_UrbanHeat_V21_Training.ipynb \
    gdrive:UMAMI/urban_heat_v3/
rclone copy /mnt/apps/spatial-llm/src/umami_spatial-llm/notebooks/README_V21_TRAINING.md \
    gdrive:UMAMI/urban_heat_v3/
```

## Running the notebook on Colab (B2)

1. Open `gdrive:UMAMI/urban_heat_v3/UMAMI_UrbanHeat_V21_Training.ipynb` in Colab
2. **Runtime → Change runtime type → Hardware accelerator: A100** (or T4 if A100 unavailable)
3. Run all cells top-to-bottom
4. **Watch for:**
   - Cell 6 (manifest summary) — confirm `has_tmrt` count is non-zero (should be ~9,500+)
   - Cell 12 (warm-start) — confirm V2 checkpoint loaded with `extended encoder.conv1 from (11) to (12)` log line
   - Cell 18 (training loop) — first epoch should complete in ~3-4 min on A100

**Total wall time:**
- A100: ~3-4 hours for 100 epochs
- T4: ~10-12 hours for 100 epochs (consider reducing to 50 epochs for first run)

## Success criteria (cell 21)

The notebook prints PASS/FAIL against four targets:

| Head | Target RMSE | V2 baseline |
|---|---|---|
| `tmrt` | ≤ 3.2 °C | 4.77 °C |
| `lst` | ≤ 3.5 °C | n/a (V2 didn't predict LST) |
| `tair_sparse` | ≤ 2.5 °C | n/a (V2 didn't predict Tair) |

Plus the per-temporal-combo breakdown (cell 22) — d306_h13 must drop from V2's 16.3 °C to ≤ 1.0 °C.

## If the run misses the target

Most likely failure mode: SOLWEIG label bias dominates. Bump these in cell 5:

```python
LOSS_WEIGHTS = {'tmrt': 1.0, 'lst': 0.5, 'tair_sparse': 10.0}
# → try: {'tmrt': 1.0, 'lst': 1.5, 'tair_sparse': 25.0}
```

If LST RMSE stays 0% finite (the corpus was built with `--no-lst`), Phase 1 just relies on T_mrt + sensor heads. To enable LST in the corpus, rerun:

```bash
python /app/scripts/urban_heat/build_v21_corpus.py --workers 8
# (without --no-lst — adds ~12 min for ~12 GEE Landsat scene fetches)
```

## Outputs after a successful run

```
gdrive:UMAMI/urban_heat_v3/models/
├── v21_best.pt                   # PyTorch checkpoint with state_dict + history
└── umami_unet_v21_fp32.onnx      # ONNX export — feed this to Block C
```

## Pull back to server (B3 → C kickoff)

```bash
rclone copy gdrive:UMAMI/urban_heat_v3/models/umami_unet_v21_fp32.onnx \
    /mnt/apps/spatial-llm/src/umami_spatial-llm/urban_heat_data/models/ \
    --progress
```

Then Block C wires V2.1 into the inference service + ModelVersionSelector.
