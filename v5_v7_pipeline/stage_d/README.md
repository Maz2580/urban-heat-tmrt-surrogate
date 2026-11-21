# Stage D — package for the Windows box

Turns Stage C's reference-SOLWEIG labels into the V5 training corpus.

Run this **after** Stage C reports 482 `DONE.json` markers.

---

## Install (2 minutes)

Everything lives under `the working folder (URBAN_HEAT_ROOT)`, same rule as Stage C.

```powershell
cd <working folder>
$env:RCLONE_CONFIG = ".\rclone.conf"
.\rclone.exe copy gdrive:UMAMI/stage_d .\stage_d --progress
```

That pulls ~2.2 GB: the script, this README, and 482 donor files.

No new Python packages. Stage D needs only `numpy` and `rasterio`, both already
in the Stage C venv. **Do not `pip install` anything** — `numpy==2.2.6` is pinned
because NumPy 2.5 breaks `solweig_gpu`, and an accidental upgrade would silently
break any later Stage C rerun.

---

## Run

```powershell
cd <working folder>
.\stage_d\run_stage_d.ps1 -DryRun      # validates everything, writes nothing
.\stage_d\run_stage_d.ps1              # corpus 1: solar channels FIXED
.\stage_d\run_stage_d.ps1 -SolarChannels zeros -OutSuffix _zeros   # corpus 2
```

The dry run is worth the five minutes. It performs every check and every read,
and reports exactly what the real run would write.

Expected output: **~7,712 files, ~38 GB** per corpus (482 patches x 16
date-hours), in `stage_d\out` and `stage_d\out_zeros`. The script refuses to
start without 30% disk headroom, and **refuses to write into a folder that
already holds npz files** — the two corpora use identical filenames, so mixing
them would be silent and unrecoverable.

### Why build both

Stage D is I/O bound and cheap; this box is free now and is shared, so deciding
later means needing it again. 76 GB against 3.4 TB is 2%.

The second corpus is not bookkeeping — it answers something actionable. If the
solar-channel fix moves the model, that argues for also repairing the backwards
shadow function (below), which needs an inference-side change and therefore its
own careful cycle. If it moves nothing, that work drops down the list.

**Train `out` first.** It is the candidate product; `out_zeros` is the control.

---

## What this fixes

Each of these was verified against data on 2026-08-16, not assumed.

| # | Problem | Fix |
|---|---|---|
| 1 | Band mapping was re-derived, causing two separate past errors | Read from `DONE.json.saved_bands[].index_in_file`; a mismatch is a hard error |
| 2 | Reference SOLWEIG writes a plausible value **over nodata** — the label cannot reveal what is missing | Validity mask built from the **input** DSM, cross-checked against V4 |
| 3 | `package_training_data.py:369-374` filled gaps with the patch mean | Invalid pixels are `NaN` plus an explicit `valid` array. No filling |
| 4 | `sin_alt`/`alt_norm` were **identically zero in the whole V3/V4 corpus** while inference fed real values | Recomputed. `sin_alt` 0.0000 -> 0.6953 for d015 10:00 |
| 5 | Timezone: SOLWEIG derives +11 and reads met `it` as local | `utc_offset=+11`, which reproduces SOLWEIG's own 71.43 deg (+10 gives 62.9) |
| 6 | Weather could drift from the labels | `era5_*` re-read from the met file that drove the labels, at local hour |
| 7 | DOY 351 (the 39.2 C heat day) has no coincident satellite scene | `lst` fully `NaN` with `lst_valid=0`, so no head trains on a substituted scene |

Measured effect on the labels, over 36 paired samples:

```
mean +23.21 C   median +23.90 C   range +10.68 to +34.65
```

---

## What this deliberately does NOT fix

`compute_shadow_fraction()` traces its ray **away from the sun**. The array shift
at `urban_heat_feature_engineering.py:190-200` samples `dsm[p-d]` where it needs
`dsm[p+d]`. Measured against SOLWEIG's own shadow raster:

```
as-is  IoU 0.052        azimuth+180  IoU 0.218        azimuth-90  IoU 0.248
```

That function feeds **live inference**. Filling channel 10 from SOLWEIG's correct
shadow while inference keeps computing a backwards one would CREATE a train/serve
mismatch rather than remove one. So the default is `-ShadowChannel zeros`, which
matches the V4 corpus.

`-ShadowChannel solweig` is available and is physically correct — **but it must
not be used for a model that will be deployed until the inference path is fixed
to match.**

---

## The open decision — at Stage E, not here

- `out` (`-SolarChannels computed`) — the fix. The candidate product.
- `out_zeros` — V4's inputs reproduced exactly, so any change is attributable
  purely to the new labels.

Build both (cheap). The real cost is two *training* runs, and that choice can be
made once the first result is in. Note the label quality does **not** depend on
this: the new labels were already validated directly against a real Landsat
scene at rho 0.912, versus 0.406 for the old ones. The ablation is about how
much the input plumbing matters, not about whether the labels are right.

## Before packaging: the three-way cross-check

`patch_0013` has now been computed by Colab, the Linux server and this box. The
Linux and portable paths are already proven byte-identical. Upload the Windows
copy and it can be compared against Colab's:

```powershell
.\rclone.exe copy .\stage_c\out\patch_0013 gdrive:UMAMI/stage_d/xcheck_windows/patch_0013 --progress
```

33 MB, and it settles whether the labels are machine-independent. Worth doing
BEFORE generating 38 GB on top of them — a check that gates expensive work
should run before that work, not after.

---

## Verifying the result

```powershell
.\venv\Scripts\python.exe .\stage_d\verify_stage_d.py --dir .\stage_d\out
```

23 checks. Two of them are written to fail if the clock is wrong: the weather
check asserts the value matches the **local** hour *and* differs from the corpus
hour. A check that cannot fail has certified nothing.

---

## Cross-check worth doing once

`patch_0013` has now been produced by three independent machines — Colab, this
Linux server, and Windows. The Linux and portable paths are already proven
**byte-identical** across 48 files and every array. Comparing Windows against
either settles the labels completely.

---

## Notes

- Stage D is **I/O bound**, so run one process. Parallelism buys nothing here.
- The donor pack exists because shipping the V4 corpus would mean 44 GB. The 16
  static channels are identical across all 12 date-hours of a patch (verified per
  patch at build time), so one donor per patch suffices — ~4.6 MB each.
- Each donor is stamped `mask_verified`. `stage_d.py` **refuses** any patch whose
  donor lacks the stamp, so a mask that was never checked against V4 cannot
  quietly become a training corpus on a machine that had no way to check it.
