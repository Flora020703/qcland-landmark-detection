# Endpoint-ordering convention analysis (no retraining)

Answers, with real numbers instead of an abstract question, the question
this project's own audit process raised and could not resolve by code
review alone (see `rtmpose_reproduction/PROTOCOL_AUDIT.md`'s "Still not
fully unified" section): EoMT canonicalises endpoints by per-image
x-coordinate sort, HRNet by a training-set-derived frozen direction
vector, and RTMPose (per `PROTOCOL_LOCKED.md`) currently follows HRNet's
convention. How much does this choice actually matter for the numbers
already reported, and does it change any conclusion?

## What this does and does not do

Re-scores EXISTING, already-saved per-image EoMT/HRNet predictions under
two common EXTERNAL conventions, without retraining, without re-running
inference, and without changing any predicted or ground-truth coordinate:

1. **Native**: whatever convention the file's own training/eval code
   already used, recomputed in that file's OWN native coordinate space
   (HRNet: original image pixels; EoMT: its own 512x512 model-input space
   -- see the coordinate-space note below). This is a SANITY CHECK ONLY,
   validating this script's own coordinate parsing reproduces the file's
   already-reported stored NME within floating-point tolerance; it is
   never compared across methods (different spaces, not a fair comparison).
2. **Unified x-sort**: both GT and prediction, after conversion to a
   COMMON original-image pixel space (see below), independently
   re-labelled by ascending x (tie-break by y).
3. **Unified DOD**: same common original-image space, independently
   re-labelled by projecting onto the SAME frozen, training-set-only
   direction vector (`rtmpose_reproduction/dod_vectors.py` -- the exact
   vectors already verified against real HRNet checkpoints and real HRNet
   per-image output in that adapter's own test suite; never re-estimated
   from Test GT here).

**Coordinate-space note (fixed 2026-08-07, a review finding against the
first version of this script)**: EoMT's own per-image dump
(`training/landmark_detection.py`'s `test_nme_dump_path`) writes
coordinates in its SQUARE 512x512 model-input space, not the original
(generally non-square) image's own pixel space. Before any unified
convention is applied, this script inverts EoMT's coordinates back to real
original-image pixel space using each image's REAL width/height (opened
from the actual image file) and the exact inverse of EoMT's own resize
formula (`rtmpose_reproduction/geometry.to_image_space`, the same
UDP-inspired pixel-centre convention EoMT's own `pixel_center_align=True`
code path uses). HRNet's per-image CSV is already in original-image space
and needs no conversion. Feeding EoMT's raw 512-space coordinates directly
into DOD (or computing NME on them without conversion) is invalid whenever
the source image is not square -- effectively always, for these ultrasound
images -- which is exactly the bug this fix corrects.

**Second coordinate-space note (fixed 2026-08-07, same day, a review
finding against the FIRST fix above)**: EoMT's dump is not exactly
512-model-input-space coordinates either -- it round-trips through a
pixel-centre-aligned heatmap encode (`datasets/landmark_dataset.py`,
`pixel_center_align=True`) composed with a naive, non-pixel-centre
scale-back in the dump itself (`training/landmark_detection.py`), leaving
a constant offset (+3.5px for the standard 512-model-input/64-heatmap
setting, verified against both source files directly). This is recovered
exactly (`_heatmap_dump_to_model_input_space()`) before the original-image
inversion above. The offset cancels out of NME/ordering (a shared
translation across all four points in an image), so it did not corrupt
the first fix's own verification -- but it does shift the recovered
ABSOLUTE coordinate, which matters for the cross-method GT consistency
check below.

**Third fix, same day**: the cross-method GT consistency check initially
compared HRNet's and EoMT's GT points index-to-index (`gt0` vs `gt0`).
This is wrong: HRNet's own native channel order follows DOD, EoMT's
follows x-sort, so the two methods may legitimately (and often will)
label the SAME two physical points with swapped channel indices whenever
x-sort and DOD disagree for that image -- exactly the population this
analysis studies. Fixed to compare both possible pairings and take
whichever is closer, so a pure convention difference is no longer
misreported as a coordinate-recovery bug.

**Tightened thresholds (fixed 2026-08-07, round 10)**: the cross-method GT
consistency check originally warned only above a single 5.0px cutoff --
judged too loose to safely gate a supervisor-facing result on. Now two
tiers: `max_gt_coord_diff_px > 0.1px` writes a row to
`cross_method_gt_consistency_warnings.tsv` (a `severe` column marks
whether it also crosses the next tier); `max_gt_coord_diff_px > 1.0px` is
a HARD FAILURE -- the script prints exactly which (dataset, task) cells
are affected and exits non-zero, and those cells' results must not be
used. The actual max error is always printed for every comparison made
(`[cross-method GT check] ...`), whether or not it crosses either
threshold, so a below-threshold result can still be inspected, not just
inferred from the absence of a warning.

**Important limitation, state this to the supervisor alongside any
numbers**: this quantifies how much the EXTERNAL SCORING RULE alone
changes each method's reported NME and whether conclusions (rankings,
which method looks better) flip. It does NOT retrain either method under a
common convention, and it cannot prove a method trained from scratch under
a different convention would perform identically -- a model's own training
labels already used ONE specific convention, and re-labelling its test-time
predictions after the fact does not undo that.

## Requirements

Needs the REAL per-image prediction files this project already produced
during the HRNet-512/fixed 50-run sweep and the EoMT 5-seed final runs.
These do not exist in this local working copy -- they live on the AutoDL
server (or wherever a local mirror/download of `/root/autodl-tmp/...` has
been kept). Run this script FROM a location where those directories are
reachable, pointing `--*-root` at them.

**EoMT-specific requirement, verify before trusting any EoMT-side result**:
the per-image files must have been written with
`training/landmark_detection.py`'s coordinate-dumping feature enabled
(`test_nme_dump_path`, columns `pred_x0,pred_y0,gt_x0,gt_y0,pred_x1,pred_y1,
gt_x1,gt_y1` -- introduced 2026-07-23/24). If a specific task/backbone/seed's
file only has `index,nme` (the older format), this script raises a clear
`LoadError` naming exactly which file and which columns are missing, and
that cell is recorded in `excluded_images.tsv` rather than silently
skipped or approximated. **UCL BPD's EoMT checkpoints/per-image files are
already known, from this project's own prior records, to no longer exist
on the server** -- expect that cell to be excluded, not a bug.

**Also required for EoMT cells, `--ucl-images-root`/`--multicentre-images-root`
(no default, must be passed explicitly)**: a directory containing
`<Head|Abdomen|Femur>/<filename>` with the REAL original images, needed to
recover each image's true width/height for the coordinate-space conversion
described above -- EoMT's 512-space coordinates cannot be correctly
inverted without it. If a specific image file can't be found under this
root, that cell is excluded with a `LoadError` naming the exact missing
path, never silently approximated (e.g. by assuming a square image).

## Usage

```bash
python endpoint_ordering_analysis/rescore_endpoint_conventions.py \
    --ucl-eomt-root /root/autodl-tmp/ucl_eomt_per_image \
    --ucl-hrnet-root /root/autodl-tmp/hrnet_512_fixed_5seed/output/FETAL \
    --ucl-images-root /root/autodl-tmp/images/UCL \
    --multicentre-eomt-root /root/autodl-tmp/saved_checkpoints/multicentre_5seed \
    --multicentre-hrnet-root /root/autodl-tmp/hrnet_512_fixed_5seed/output/FETAL \
    --multicentre-images-root /root/autodl-tmp/images/MULTICENTRE \
    --output-root endpoint_ordering_analysis/results
```

The `--*-eomt-root`/`--*-hrnet-root` defaults already match the real paths
used by this project's own `final_comparison/analyse_ucl_per_image.py` and
`analyse_multicentre_per_image.py` -- override only if your copy of the
data lives somewhere else. `--*-images-root` has no default and must
always be passed explicitly (see Requirements above).

Run the local test suite first (pure Python, no real data needed, exercises
the loading/canonicalisation/bootstrap code against synthetic files
matching the real CSV schemas exactly):

```bash
python endpoint_ordering_analysis/test_rescore_endpoint_conventions.py
```

## Outputs

```
endpoint_ordering_analysis/results/
  endpoint_ordering_summary.tsv               one row per (dataset, task, method): 5-seed
                                               mean+-SD for all 3 conventions, GT disagreement
                                               rate, x-sort-vs-DOD mean difference + bootstrap
                                               95% CI
  endpoint_ordering_seed_summary.tsv          one row per (dataset, task, method, seed, convention)
  ucl_*_per_image.csv                         per-image, 5-seed-averaged NME under all 3
                                               conventions, in unified original-image coordinates
  multicentre_*_per_image.csv                 (same, Multicentre)
  dod_vectors.tsv                             every frozen (dataset, task) direction vector used
  excluded_images.tsv                         every (dataset, task, method) cell that could not
                                               be scored, and exactly why (missing files, missing
                                               coordinate columns, missing original image file) --
                                               never silently dropped
  cross_method_gt_consistency_warnings.tsv    any (dataset, task) where HRNet's and EoMT's own
                                               GT (after EoMT's original-space conversion) disagree
                                               on location or on x-sort-vs-DOD disagreement rate --
                                               a non-empty file signals a coordinate-recovery or
                                               sample-matching bug, not a real dataset property
```

## Reading the summary for the supervisor conversation

Compress `endpoint_ordering_summary.tsv` into:

| Dataset | Task | Method | Native | Unified x-sort | Unified DOD | x-sort - DOD (95% CI) |
|---|---|---|---|---|---|---|

and lead with the GT disagreement rate per task (`gt_xsort_vs_dod_disagreement_rate`)
-- this is the single number that most directly answers "how much do the
two rules actually differ on this task's own ground truth," independent of
either method's prediction quality.

Suggested framing (adapt the specifics to whatever the real numbers show):

> I retrospectively re-evaluated the existing per-image predictions under
> two common external endpoint conventions, without retraining: per-image
> x-coordinate sorting and a training-set-estimated frozen direction
> vector. The attached summary shows how much the choice changes each
> method and whether it changes the conclusions. Based on this evidence,
> which convention would you prefer us to use consistently for RTMPose
> training and the final comparison?
