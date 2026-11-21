# Stage C — reference-SOLWEIG label regeneration (Colab)

Regenerates the Urban Heat T_mrt labels. The V4 labels came from a custom engine
with **no surface energy balance** and are ~36 °C too cold in sun; every model
version V2–V4.1 was graded against them, so every published metric was scored
against the defect.

---

## Run it

```python
# 1. mount Drive
from google.colab import drive; drive.mount('/content/drive')

# 2. GPU runtime: Runtime -> Change runtime type -> T4 (or better)
!nvidia-smi

# 3. install the reference engine (GPL, Kamath & Sudharsan)
!pip -q install solweig-gpu==1.2.21 rasterio

# 4. SMOKE TEST FIRST — never launch the full run blind
!python /content/drive/MyDrive/UMAMI/stage_c/stage_c_generate.py \
    --patch-list /content/drive/MyDrive/UMAMI/stage_c/patch_list_B_default_lt70.json \
    --limit 3

# 5. full run
!python /content/drive/MyDrive/UMAMI/stage_c/stage_c_generate.py \
    --patch-list /content/drive/MyDrive/UMAMI/stage_c/patch_list_B_default_lt70.json
```

---

## Crash safety (Colab **will** disconnect)

Resume trusts a **completion marker**, never a file's presence.

```
compute in local scratch  ->  copy outputs to Drive  ->  re-open and verify  ->  write DONE.json
```

The marker is written **last**. If the session dies at any point before it, the
next run redoes that patch-day cleanly. Worst case you lose **one** patch-day
(~1–2 min), never the whole job.

⚠️ **Why not just check the file exists:** Colab disconnects mid-write, which is
precisely when resume matters. A **truncated** `TMRT_*.tif` still exists — an
existence check would skip it and carry a corrupt label into training, silently.
That is how every other defect in this project survived.

Verified behaviour:

| state on disk | resumes? | |
|---|---|---|
| empty dir | redo | ✅ |
| **TMRT present but truncated** | **redo** | ✅ the case that matters |
| marker + files present | skip | ✅ |
| marker but file deleted | redo | ✅ |
| corrupt marker | redo | ✅ |

Intermediates go to `--scratch` (default `/content/stage_c_work`, **local disk**).
Only `TMRT_*`, `Lup_*`, `Shadow_*` and the met file reach Drive. SOLWEIG also
writes tiled inputs, walls, aspect and SVF; pushing those through Drive's FUSE
mount is slow and is itself a common cause of the disconnects this protects
against.

Progress: `out/stage_c.log`. Count finished work any time with:

```python
!ls /content/drive/MyDrive/UMAMI/stage_c/out/*/DONE.json | wc -l
```

⚠️ **Training checkpoints are a separate thing and do NOT exist yet.** This script
generates labels. The V5 training notebook (Stage E) is not written; it will need
its own periodic checkpointing of model weights and optimiser state.

---

## The preflight checks halt. That is deliberate.

Every failure in this project has been **silent**: an engine that rendered clean
maps while pinning all surfaces to air temperature; an evaluation sharing its
source with the training target; a validation run an hour before the satellite
passed that *agreed*, because two errors cancelled. Nothing ever errored.

So each check stops the run:

| check | what it catches | has it fired? |
|---|---|---|
| CUDA present, and timing beats the 85 s CPU baseline | silent CPU fallback turning 4 h into days | — |
| ERA5 forcing == input channels 11–14, to 0.00 °C | labels describing different weather from the inputs | **yes — 1.82 °C, found the UTC+10 convention** |
| leaf fix **verified in output** (shadow 0.03, not 0.5) | canopy treated as leafless; −64% tree benefit | **yes — that is how the bug was found** |
| landcover remap total | an unmapped code silently taking default physics | — |
| met = 24 contiguous hours from midnight | OOM (full year = 8760 steps, killed at 38 min **reporting exit 0**) | **yes** |
| output actually written | a "successful" run that produced nothing | **yes — that is how the OOM presented** |

A warning that scrolls past at 2 a.m. while 12,244 labels build on a broken
assumption is worth nothing.

---

## ⚠️ `--patch-list` is required, and it is a judgement call

The script refuses to guess. `generate_landcover.py:100` fills the raster with
`LC_BARE_SOIL` and paints classes over it — line 13: *"5 = Bare soil (default —
everything else)"*. Real bare soil and "nothing matched" are indistinguishable
afterwards, and SOLWEIG takes albedo, emissivity, `TgK`, `Tstart` **from the
class**. Every default pixel is a guess about the surface energy balance that
lands in the label.

Across the 987 corpus patches: **median default fraction 70.3%**; 65.3% ≥50%;
34.9% ≥90%.

**✅ DECIDED 2026-08-15 — use `patch_list_B_default_lt70.json` (482 patches).**

The stratified validation (150 patches, 0 failures) measured ranking skill against
Landsat as landcover quality degrades:

```
band       n   med.fill  SPEARMAN     |  cumulative cut   n   SPEARMAN
low       50     19.4%     0.906      |  <30%            36     0.900
mid       50     56.1%     0.870      |  <50%            69     0.902
high      50     79.1%     0.451      |  <70%           100     0.913  <- best
                                      |  <80%           127     0.880
fine bins: 50-65% 0.909 | 65-80% 0.637 | 80%+ 0.162   |  all    150     0.828
```

**A cliff at ~65%, not a slope.** Above 80% fill there is no usable ability to say
which block is hotter (0.162). `<70%` scores **0.913** — better than restricting to
the cleanest 30%, because sample size is gained without admitting a broken patch.

| list | patches | ρ | |
|---|---|---|---|
| `patch_list_A_all.json` | 653 | 0.828 | dragged down by unrankable patches |
| **`patch_list_B_default_lt70.json`** | **482** | **0.913** | **use this** |
| `patch_list_C_default_lt50.json` | 341 | 0.902 | 30% less data, no better |

Cost of B over C: bias +4.15 °C vs +3.92. That extra warmth is **ours** — the
default→Dark_asphalt mapping paints unclassified ground as hot road, so bias tracks
fill (+3.05 → +4.87 → +5.18 by band). Ranking is the product claim and the bias is
not to be tuned, so the data is worth more than 0.2 °C.

⚠️ The 482 still include patches up to 70% guessed. They rank correctly; their
absolute values lean warm. Fixing landcover properly (Nearmap-derived) would also
rescue the ~170 dropped patches — option D, V6, the highest-value follow-up.

---

## What this run does *not* change

- **No new input channels.** V4.1 is 23 channels, grown from V2.1's 12 — a growth
  that happened while every version was graded against broken labels, and is
  plausibly the same story as the anti-collapse variance loss: inputs recruited to
  explain variance the targets never contained. V5 changes the **labels only**, so
  the result is attributable. Ablate the 11 V4 channels *after*.
- **No tuning against Landsat.** The model runs **+3.56 °C** warm vs ST_B10
  (**+1.83** after harmonising emissivity conventions). That residual cannot be
  separated from the ST product's own ~1–2 K urban bias, so fitting it away would
  be calibrating against a measurement whose error we cannot isolate.
- **No SVF caching.** `solweig_gpu` writes SVF but has **no code path to reuse
  one**, so this would mean patching library internals upstream of every label.
  Revisit only if GPU timings show SVF dominating, and only with a numerical
  equality check against an uncached run.

---

## The 5th date

DOY 351 (16 Dec 2024, **Tmax 39.2 °C**) was added because the corpus tops out at
26.8 °C while heat-vulnerability decisions are about 40 °C days. It sits 5 days
from DOY 356, so solar geometry is near-identical while air temperature differs by
~20 °C — at 11:00/13:00 the two days have near-identical `ssrd` (727/745, 990/958
W m⁻²). A controlled pair: sun constant, weather varied.

⚠️ **DOY 351 has no coincident cloud-free Landsat scene — mask the `lst`
multi-task head for it.** Do not substitute a nearby scene: 18 Dec was 21.4 °C and
would mislabel the one date added for its heat.

A separate held-out scene exists for validation: **4 Jan 2025, Tmax 36.6 °C,
`LC09_092086_20250104`, 0.2% cloud**, same WRS path and overpass time as the
10 Jan validation scene. Keep it **out of training** — validating on a training
date is the original sin.

---

## After the run

1. Copy `out/` back and run `analyse_spatial.py` on a sample. **Confirm Spearman
   is still ~0.91.** The GPU is a different code path; nothing validated on CPU
   transfers for free.
2. Validate on the held-out 4 Jan 2025 scene — the first test of whether the
   physics holds in the 36 °C regime the product is actually for.
3. Stage D: fix `package_training_data.py:369-374` to emit a **validity mask**
   instead of mean-filling, and handle the `-9999` sentinels in the old labels.
4. Stage E: retrain as V5 under a **new filename**, left **unregistered** until
   the renders have been seen and approved.

## Claim boundary (carry this into the product)

| claim | supported? |
|---|---|
| "this block is hotter than that block" | ✅ ρ = 0.912 — **but measured on well-classified patches only** |
| "this side of the street is hotter" | ❌ no dataset available to us can test it |
| "the absolute temperature is X °C" | ⚠️ ±2 °C at best |
