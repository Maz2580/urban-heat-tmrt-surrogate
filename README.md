# Urban heat T_mrt surrogate: code, models and records

This repository accompanies the paper **"Spatial data provenance in simulator-trained surrogates: four failure modes
that accuracy metrics could not detect"** (Yiqun Chen, Mazdak Ghasemi Tootkaboni, Alice Kesminas, Abbas Rajabifard;
The University of Melbourne).

The paper follows seven generations (V1 to V7) of a ResNet-34 U-Net surrogate that predicts mean radiant temperature
(T_mrt) on a 1 m grid over a 9 km × 9 km domain around Melbourne's central business district. V1 to V4.1 were trained
on labels from a custom GPU implementation of SOLWEIG's geometry; V5 to V7 on labels regenerated with the reference
`solweig_gpu` 1.2.21. The repository contains:

- the **trained models** of every version (GitHub release `models-v1.0`);
- the **code** that built the inputs and labels, trained and exported the models, and evaluated them;
- the **training records** (logs, epoch histories, test results, diagnostics) and the **result tables** of the paper.

## Quick start: run the models

Tested with Python 3.11, numpy 2.4 and onnxruntime 1.29 on Linux; any recent Python 3 should work.

**Step 1. Get the repository**

```
git clone https://github.com/Maz2580/urban-heat-tmrt-surrogate
cd urban-heat-tmrt-surrogate
```

**Step 2. Install the two packages needed to run the models**

```
pip install -r requirements.txt        # numpy and onnxruntime
```

**Step 3. Download the models** (all versions, about 0.9 GB, into `models/`; every file is checked against its SHA-256)

```
python scripts/download_models.py
```

While the repository is private, set `GITHUB_TOKEN` to a GitHub token with read access to it before this step.

**Step 4. Check that a model runs** (here V4 on a random input; use any of `v1 v2 v3 v3.1 v4 v4.1 v5 v6 v7`)

```
python scripts/run_model.py --version v4
```

**Step 5. Run a model on a real input and score it**

```
python scripts/run_model.py --version v5 --sample <sample.npz> --save prediction.npy
```

`<sample.npz>` is one corpus sample: `input` (channels × 512 × 512, scaled as in training) and, optionally, `tmrt`
(the T_mrt label in °C; NaN where there is no value). Samples in this format come with the V3 to V7 corpora (on
request, see Data); the V1/V2 corpora store input and label as separate `.npy` arrays, which can be packed into this
format. The script prints the range of the T_mrt prediction and, when
the sample has a label, the RMSE against it; with `--save` it writes the prediction as a 512 × 512 array in °C.

What each version expects and returns:

| Version | Input channels | Output |
|---|---|---|
| V1 | 5 | T_mrt (the model returns T_mrt / 63; the script converts it to °C) |
| V2 | 11 | T_mrt (as V1) |
| V3, V3.1 | 12 | 3 channels in °C: T_mrt (used here), land surface temperature, air temperature |
| V4, V4.1, V5, V6, V7 | 23 | as V3 |

Each version must be given inputs built for it (paper Table 1 and Supplementary Table S1). V7 in particular changes
the meaning of channels 0, 10 and 17 to 19, so V5-era inputs give V7 a large error.

## What can and cannot be reproduced

| Task | Possible from this repository? |
|---|---|
| Download and run every model | Yes: steps 1 to 5 |
| Check the paper's reported numbers | Yes: result tables in `results/`, training logs and test results in `records/` |
| Reproduce the V1 test RMSE and the V2 validation RMSE | Yes: their inputs come from public data; rebuild inputs and labels with the scripts and the label-generator notebook below (the labels need a GPU) |
| Reproduce the test RMSE of V3 to V7 exactly | Needs their test corpora, which contain Geoscape-derived channels: available on request under a Geoscape licence |
| Rebuild the corpora or retrain | Needs a GPU, an Earth Engine account and, for V3 to V7, a Geoscape licence; the scripts are listed below |

## The pipeline in order

| Step | What | Code | Paper |
|---|---|---|---|
| 1 | Input rasters and features: DSM, canopy, land cover, albedo, Geoscape, NDVI, ERA5, solar position | `process_dsm.py`, `generate_cdsm.py`, `rasterize_vectors.py`, `rasterise_geoscape.py`, `generate_albedo.py`, `generate_landcover.py`, `generate_patches.py`, `v4_prep.py`; `app/services/` (features, solar position, Earth Engine client, patch builder) | 3.1, 3.2 |
| 2a | Labels V1 to V4.1: custom GPU SOLWEIG kernel (Colab) | `notebooks/UMAMI_SOLWEIG_GPU_v2.ipynb`, `convert_epw_to_umep.py`, `package_training_data.py` | 3.3 |
| 2b | Labels V5 to V7: reference `solweig_gpu` 1.2.21 | `v5_v7_pipeline/stage_c/stage_c_generate.py` and its launchers | 3.3 |
| 3 | Training corpora: V2.1 and V4; Stage D for V5 to V7; corrected shadow planes for V6 and V7 | `build_v21_corpus.py`, `build_v4_corpus.py`, `stage_d.py`, `build_donor_pack.py`, `build_v5_manifest.py`, `regen_shadow_v6.py`, `build_shadow_v6_pack.py`, `apply_shadow_v6.py`, `regen_shadow_v7.py`, `build_shadow_v7_pack.py`, `apply_shadow_v7.py`, `v5_v7_pipeline/stage_d/verify_stage_d.py` | 3.3, 3.4 |
| 4 | Training: V1 to V4.1 in notebooks (Colab); V5 to V7 with `train_v5.py` (GPU workstation) | `notebooks/`, `train_v2.py`, `v5_v7_pipeline/stage_d/train_v5.py` and its runners | 3.4, 3.5 |
| 5 | Export and evaluation: ONNX export, test-set metrics, signed error, latency (the train/test overlap of the V5 to V7 split is in `results/split_overlap_v5v7.json`) | `export_v5_onnx.py`, `eval_v3_v4_test.py`, `eval_signed_error.py`, `bench_onnx_latency.py`; outputs in `results/` | 3.6, 3.7, 4.2, 4.7 |
| 6 | Diagnostics of input geometry and of V5 to V7 | `world_change_audit.py`, `shadow_lie_audit.py`, `shade_age_audit.py`, `ceiling_test.py`, `scatter_diag.py`, `edge_floor.py`, `error_spectrum.py`, `sensitivity_probe.py`, `fit_shade_calib.py`, `render_v5_compare.py`, `render_v7_compare.py`; outputs in `records/` and `results/world_change_audit.json` | 4.5, 4.6 |

Scripts given without a folder are in `scripts/urban_heat/`.

## Repository layout

| Folder | Contents |
|---|---|
| `scripts/` | `download_models.py`, `run_model.py` (quick start) |
| `scripts/urban_heat/` | the pipeline scripts listed above |
| `scripts/urban_heat/v3_regression_probes/` | probes of the served model through the platform's polygon-analysis path (need the deployed service) |
| `app/services/` | feature engineering, solar position, patch extraction, Earth Engine client and patch builder used by the corpus builders, and the inference service of the original platform |
| `notebooks/` | training notebooks V1, V2, V3 (`V21`), V3.1, V4, V4.1, and the V1 to V4.1 label generator. In the V3 notebook the model (a `segmentation_models_pytorch.Unet` with a ResNet-34 encoder, 12 input channels and 3 outputs, paper Section 3.4) was created in a cell run interactively that is not saved; the saved cells load the warm-start weights into it |
| `v5_v7_pipeline/stage_c/` | label regeneration for V5 to V7, worker launchers, patch lists |
| `v5_v7_pipeline/stage_d/` | training script for V5 to V7 (the version that trained them), corpus check, Windows runners |
| `records/v5/`, `v6/`, `v7/` | training logs, epoch histories, test results and diagnostics |
| `results/` | result tables of Sections 4.2 and 4.3 (`paper_data/`: aggregate metrics, training curves, per-condition RMSE), V3/V4 test-set evaluation, train/test overlap of the V5 to V7 split, world-change audit, latency benchmark, test results, normalisation statistics |

## Notes for re-running the pipeline

- The pipeline scripts expect the original folder layout: one project root holding `scripts/`, `app/` and
  `urban_heat_data/` side by side (in the original container this root was `/app`). Some scripts carry absolute paths
  or container names of the original setup in their defaults or comments; adjust them to your machine.
- Main packages: `numpy pandas scipy rasterio pyproj shapely matplotlib pillow onnxruntime torch torchvision
  segmentation-models-pytorch earthengine-api requests`; label regeneration for V5 to V7 needs `solweig-gpu==1.2.21`.
  Versions recorded in the paper: label engine numpy 2.2.6 and torch 2.13.0; serving and timing onnxruntime 1.21.1,
  numpy 1.26.2, Python 3.11.14. `v5_v7_pipeline/stage_d/setup_train_env.ps1` builds the V5 to V7 training environment.
- The Windows runners read the working folder from the environment variable `URBAN_HEAT_ROOT`.
- `download_data_sources.py` also lists optional AURIN layers, including a June 2022 Geoscape extract; the paper uses
  the December 2025 Geoscape product (`rasterise_geoscape.py`).
- `app/services/urban_heat_inference.py` is the serving path of the original platform. Its full service entry point
  also calls platform modules that are not included (V8 world, report post-processing); the paper's evaluation runs
  the ONNX models directly, as `run_model.py` does.
- Comments that refer to `memory.md` point to the internal development log, which is not part of this repository.

## Data

### Data sources

| Data | Used for | Source |
|---|---|---|
| City of Melbourne 2018 LiDAR, 0.1 m DSM tiles | surface model and bare-earth DEM (`process_dsm.py`) | City of Melbourne |
| Trees with species and dimensions (urban forest) | canopy height model (`generate_cdsm.py`) | City of Melbourne Open Data, `trees-with-species-and-dimensions-urban-forest`: https://data.melbourne.vic.gov.au/explore/dataset/trees-with-species-and-dimensions-urban-forest/ |
| Building footprints 2023 | roof mask, land cover and albedo | City of Melbourne Open Data, `2023-building-footprints`: https://data.melbourne.vic.gov.au/explore/dataset/2023-building-footprints/ |
| Vicmap Hydro, Vicmap Transport | water and road surfaces for land cover and albedo | Vicmap ArcGIS REST services, https://services-ap1.arcgis.com/P744lA0wf4LlBZ84/ArcGIS/rest/services |
| ERA5-Land hourly | air temperature, humidity, wind, shortwave radiation | Google Earth Engine `ECMWF/ERA5_LAND/HOURLY` |
| Landsat 8/9 Collection 2 Level-2 (ST_B10) | auxiliary land surface temperature target | Google Earth Engine `LANDSAT/LC08/C02/T1_L2`, `LANDSAT/LC09/C02/T1_L2` |
| Sentinel-2 surface reflectance | NDVI (V4 onward) | Google Earth Engine `COPERNICUS/S2_SR_HARMONIZED` |
| Melbourne Airport (Tullamarine) typical meteorological year, TMYx 2009–2023 | weather forcing of the SOLWEIG labels (`convert_epw_to_umep.py`) | climate.onebuilding.org, `AUS_VIC_Melbourne.Intl.AP.948660_TMYx.2009-2023.epw` |
| Geoscape Buildings (December 2025) and greenspace (February 2021) | building heights, roof form, solar panels, material albedo (V3 onward; `rasterise_geoscape.py`) | Geoscape Australia, https://geoscape.com.au (licensed, see below) |

- Geoscape Buildings (December 2025) is licensed and cannot be redistributed, nor can rasters derived from it
  (building heights and albedo). Running the released models needs no Geoscape data; rebuilding the V3 to V7 inputs
  does.
- Models: GitHub release `models-v1.0` (MIT). Corpora: the V1/V2 inputs derive from public data; the V3 to V7 corpora
  contain Geoscape-derived channels and are available on request under the Geoscape licence.

## Changes from the versions that ran

The files are copies of the versions that produced the paper's results, with these edits only:

- the workstation folder in the Windows runners is read from `URBAN_HEAT_ROOT` instead of being hard-coded;
- local folder names, a server path and a container name in comments, in the Stage D README and in a few logs and
  records are replaced by placeholders (`<WORK_ROOT>`, `<working folder>`, `<project root>`, "serving container");
- a server address in a printed note of the label-generator notebook is replaced by `<user>@<server>`, and account
  details are removed from the notebooks' metadata;
- the training logs are converted from UTF-16 (as PowerShell wrote them) to UTF-8, text unchanged;
- in `results/paper_data/`, working-note columns are removed and the split of the V2 per-condition rows is corrected
  to validation (the V2 notebook evaluates on the validation split only);
- the V3/V4 test-set evaluation is stored under the name the paper uses, `eval_v3_v4_test.py`;
- `scripts/download_models.py`, `scripts/run_model.py` and `requirements.txt` were added for this release.

## Citation and licence

Please cite the paper; `CITATION.cff` gives the citation for this repository. Code and models: MIT (see `LICENSE`).
Third-party data keep their own licences.
