# UMAMI Urban Heat V4 — Training Workflow (V4-D + V4-E + Colab)

V4 is the next-generation **23-channel multi-task** model that builds on
V3 by extracting more from the existing GEE feeds (full ERA5 4-tuple,
NDVI as a per-pixel raster, multi-time-of-day shadows) plus Geoscape
solar-panel + roof-type masks. Designed to push past V3's RMSE 2.97 °C
plateau and finally close V3's residual `d306_h13` 1.40 °C gap.

## Inputs already on Drive

```
gdrive:UMAMI/urban_heat_v4/
├── patches_v4.tar              # ~28 GB tar of 11,844 .npz + manifest.parquet
├── models/
│   └── v21_best.pt             # V3 checkpoint (~280 MB) for warm-start
├── UMAMI_UrbanHeat_V4_Training.ipynb
└── README_V4_TRAINING.md       # this file
```

Same single-tar pattern as V3: Drive throttles per-file metadata so
~12K small files would take hours streaming. The notebook's cell 4
copies the tar to `/content/patches_v4.tar` and extracts it to
`/content/patches_v4/` on the Colab VM (~5-10 min prep, then training
reads at full local-SSD speed).

## What V4 adds beyond V3

V4's input is V3's 12 channels (slots 0-11) plus 11 new (slots 12-22).
Source of truth for the slot order is `V4_CHANNEL_ORDER` in
`app/services/urban_heat/data_pipeline/patch_builder.py:442` — also
duplicated in notebook cell 7.

| Slot | Channel | Source | Why |
|---|---|---|---|
| 12 | `era5_rh_pct` | ERA5_LAND hourly | Real humidity, V3 only had Tair |
| 13 | `era5_wind_speed_ms` | ERA5_LAND hourly | Wind affects convective cooling |
| 14 | `era5_ssrd_wm2` | ERA5_LAND hourly | Surface solar radiation downward — direct heat input |
| 15 | `ndvi` | Sentinel-2 median (288 MB raster) | Per-pixel vegetation context vs V3's per-AOI |
| 16 | `geoscape_solar_panel_mask` | Geoscape Dec-2025 | Reflective rooftop signal |
| 17 | `shadow_h09` | Local shadow raster | Morning sun geometry — addresses V3's d306_h13 gap |
| 18 | `shadow_h13` | Local shadow raster | Midday |
| 19 | `shadow_h16` | Local shadow raster | Afternoon |
| 20 | `roof_type_flat` | Geoscape one-hot | Flat roofs absorb / re-emit differently |
| 21 | `roof_type_moderate` | Geoscape one-hot | Moderate pitch |
| 22 | `roof_type_steep` | Geoscape one-hot | Steep pitch sheds heat fastest |

## How everything got there (server-side)

```bash
# 1. Build the corpus (server, ~16 min on 8 workers)
docker exec -e PYTHONPATH=/app umami_test python /app/scripts/urban_heat/build_v4_corpus.py \
    --workers 8

# 2. Tar the patches (~3-5 min on the umami_test container)
docker exec umami_test tar -cf /app/urban_heat_data/training/patches_v4.tar \
    -C /app/urban_heat_data/training patches_v4/

# 3. rclone tar to Drive (~33 min at 14 MiB/s for ~28 GB)
rclone copyto /mnt/apps/spatial-llm/src/umami_spatial-llm/urban_heat_data/training/patches_v4.tar \
    gdrive:UMAMI/urban_heat_v4/patches_v4.tar --stats 30s

# 4. V3 checkpoint for warm-start (copy from existing V3 folder)
rclone copyto gdrive:UMAMI/urban_heat_v3/models/v21_best.pt \
    gdrive:UMAMI/urban_heat_v4/models/v21_best.pt

# 5. Notebook + README
rclone copy /mnt/apps/spatial-llm/src/umami_spatial-llm/notebooks/UMAMI_UrbanHeat_V4_Training.ipynb \
    gdrive:UMAMI/urban_heat_v4/
rclone copy /mnt/apps/spatial-llm/src/umami_spatial-llm/notebooks/README_V4_TRAINING.md \
    gdrive:UMAMI/urban_heat_v4/
```

## Running the notebook on Colab

1. Open `gdrive:UMAMI/urban_heat_v4/UMAMI_UrbanHeat_V4_Training.ipynb` in Colab
2. **Runtime → Change runtime type → Hardware accelerator: A100** (T4 will work but slower; reduce EPOCHS to 50 if needed)
3. Run all cells top-to-bottom
4. **Watch for:**
   - Cell 9 (manifest summary) — confirm ~11,844 rows, all 12 combos present
   - Cell 15 (warm-start) — confirm log line *`extended encoder.conv1 from (12) to (23)`*; if it says (3)→(23) the V3 checkpoint isn't loading correctly
   - Cell 21 (training loop) — first epoch should complete in ~3-4 min on A100; loss should drop from ~10 → <2 within first 20 epochs

**Total wall time:**
- A100: ~4-5 hours for 100 epochs (slightly longer than V3's 3-4 h due to wider input + 23 raster reads per patch)
- T4: ~12-15 hours for 100 epochs (consider 50 epochs)

## Success criteria (cell 22-24)

The notebook prints PASS/FAIL against four targets — all are improvements over V3:

| Head | V4 target | V3 actual | V2 actual |
|---|---|---|---|
| `tmrt` (RMSE °C) | **≤ 2.5** | 2.968 | 4.77 |
| `lst` (RMSE °C) | ≤ 3.0 | n/a (V3 trained without LST) | n/a |
| `tair_sparse` (RMSE °C) | ≤ 2.0 | n/a (sparse — sensitive to sample size) | n/a |
| **d306_h13** (RMSE °C) | **≤ 1.0** | 1.403 | 16.353 |

Plus per-temporal-combo breakdown (cell 24) — all 12 combos must land
under 3.0 °C; V3's worst was 3.81 °C at d015_h15. V4's expanded ERA5 +
multi-shadow channels are explicitly designed to reduce the spread.

## If the run misses the target

**Most common failure mode:** RMSE plateaus around V3's ~3.0 °C — the new channels
aren't being learned because they're zero-initialised in the warm-start.
Try in cell 19 (training):

```python
# Boost LR for the new conv1 input slots so they learn faster than the
# warm-started V3 weights
LR_GROUPS = [
    {'params': [p for n, p in model.named_parameters() if 'encoder.conv1.weight' in n], 'lr': 5e-4},
    {'params': [p for n, p in model.named_parameters() if 'encoder.conv1.weight' not in n], 'lr': 1e-4},
]
opt = torch.optim.AdamW(LR_GROUPS, weight_decay=WEIGHT_DECAY)
```

**Alternative:** if `lst` head is dragging tmrt down, lower its weight
in cell 7:

```python
LOSS_WEIGHTS = {'tmrt': 1.0, 'lst': 1.0, 'tair_sparse': 10.0}
# → try: {'tmrt': 1.0, 'lst': 0.5, 'tair_sparse': 10.0}  # back to V3-style
```

**If d306_h13 doesn't drop below 1.0 °C even with full V4 channels:**
multi-shadow rasters are explicitly designed to fix this — verify the
manifest's d306_h13 patches actually have non-zero shadow_h09/h13/h16
channel data (cell 24's per-channel mean stat). If they're all zero,
the shadow rasters didn't burn into the patches correctly — re-run V4-D
with verbose logging to surface the issue.

## Outputs after a successful run

```
gdrive:UMAMI/urban_heat_v4/models/
├── v4_best.pt                    # PyTorch checkpoint with state_dict + history
└── umami_unet_v4_fp32.onnx       # ONNX export — feed this to V4 Block C
```

## Pull back to server (kicks off V4 Block C-equivalent)

```bash
rclone copy gdrive:UMAMI/urban_heat_v4/models/umami_unet_v4_fp32.onnx \
    /mnt/apps/spatial-llm/src/umami_spatial-llm/urban_heat_data/models/ \
    --progress
```

Then V4 backend integration (mirrors V3's `12324e5` Block C + `b3a4913` Path A):
1. Add `v4` branch to `_run_inference` in `app/services/urban_heat_inference.py` —
   build 23-channel input by composing V3's existing 12-channel logic with the
   6 new V4 raster reads + 3 new ERA5 fields per AOI sample
2. Add `v4` branch to the polygon worker in
   `app/services/urban_heat_polygon_analysis.py` (mirror what was done for v3)
3. Extend `ModelVersion` union in `frontend/src/components/UrbanHeat/ModelVersionSelector.tsx`
   from `'v1' | 'v1.1' | 'v2' | 'v3'` to add `'v4'`
4. Update PHASE_INFO['1.6.3'] in `UrbanHeatPanel.tsx` with V4 actual metrics
5. Run regression probes from `scripts/urban_heat/v3_regression_probes/` against V4
   (probes were designed to be V-agnostic; they should pass for V4 too)
