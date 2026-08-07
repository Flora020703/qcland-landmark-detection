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
                                               rate, raw-channel-vs-prediction-x-sort audit,
                                               prediction reversal rate, and x-sort-vs-DOD
                                               mean difference + bootstrap 95% CI
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
  raw_channel_vs_prediction_xsort_summary.tsv a compact extract of endpoint_ordering_summary.tsv's
                                               raw-channel/prediction-x-sort columns only (see
                                               "For the supervisor's immediate ... question" below
                                               for what these mean and, critically, for which
                                               method they are and are not a meaningful diagnostic)
  correspondence_diagnostic_summary.tsv       DIFFERENT question from the above: does p0/p1 actually
                                               correspond, by distance, to the correct GT side, or
                                               only look reversed in raw x-order? See "Correspondence
                                               diagnostic" below -- prediction_x_reversal_rate alone
                                               cannot answer this, especially for near-vertical
                                               diameters (BPD/TAD)
```

For the supervisor's immediate training/inference correspondence question,
use these columns from `endpoint_ordering_summary.tsv`:

- `raw_channel_original_5seed_mean_pct` / `*_sample_sd_pct`: channel 0 and
  channel 1 retained exactly as decoded, compared with left/right x-sorted
  GT, all in original-image coordinates;
- `xsort_5seed_mean_pct` / `*_sample_sd_pct`: predictions and GT both
  deterministically x-sorted before the same fixed-channel NME;
- `prediction_x_reversal_rate_5seed_mean` / `*_sample_sd`: fraction of test
  predictions for which raw channel 0 lies to the right of raw channel 1;
- `raw_minus_prediction_xsort_5seed_mean_pp` / `*_sample_sd_pp`: paired
  seed-level numerical reduction associated with enforcing the training
  left-to-right convention after decoding.

`native_*` is retained only as a parser sanity check and must not be
subtracted from unified x-sort for EoMT, because the historical native EoMT
score was computed in the anisotropically resized 512x512 coordinate space.

**Interpretation caveat, do not skip this when reading the raw-channel/
reversal columns for HRNet**: `raw_channel_original`/`prediction_x_reversal_rate`
compare each method's UNTOUCHED, as-decoded channel 0/1 against the
x-sorted (left/right) GT. This is a genuine model-quality diagnostic for a
method whose own TRAINING convention already is x-sort (EoMT) -- a nonzero
gap there means the model itself assigned channels inconsistently. HRNet's
own training/native convention is DOD, not x-sort (see `dod_vectors.py`),
so for HRNet these same columns are EXPECTED to be large exactly on the
`gt_xsort_vs_dod_disagreement_rate` fraction of images, REGARDLESS of how
accurate HRNet's predictions are -- a perfectly-trained HRNet model will
still show a high `prediction_x_reversal_rate` on any image where DOD and
x-sort disagree, because its channel 0 is correctly the DOD-first point,
not the x-sort-left point. Read HRNet's numbers here as "the cost of
scoring HRNet under an x-sort assumption it was never trained for," not as
evidence of a channel-assignment defect in HRNet itself.

## Correspondence diagnostic (`correspondence_diagnostic_summary.tsv`)

`prediction_x_reversal_rate` only checks whether the RAW prediction pair
keeps its own left-to-right order (`pred0.x > pred1.x`) -- it does NOT
check whether p0 is actually the point closer to the LEFT GT and p1
actually closer to the RIGHT GT. These are different questions. On a
near-vertical diameter (BPD/TAD, where the two GT points differ by a
fraction of a pixel in x but tens of pixels in y), a tiny prediction x
error can flip `prediction_x_reversed` to `True` while p0/p1 are each
still, unambiguously, closest to their own correct GT point -- reversal
here is x-sort noise, not a correspondence error (see
`test_correspondence_diagnostic_distinguishes_reversal_from_true_swap` in
the test suite for a constructed example, and the real BPD/TAD d_vect
geometry in `dod_vectors.py` for why this is the realistic regime, not an
edge case, for exactly those two tasks).

The direct, convention-agnostic answer is a per-image bipartite-distance
comparison, computed in `rescore_cell()`:

- `raw_channel_original` (`E_intended`): `||p0-gt_left|| + ||p1-gt_right||`
  -- the AS-TRAINED pairing, already described above.
- `cross_pairing` (`E_crossed`): `||p0-gt_right|| + ||p1-gt_left||` -- the
  OPPOSITE pairing.
- `cross_pairing_preferred`: `True` when `E_crossed < E_intended`, i.e.
  p0/p1 look, purely by distance, like they correspond to the opposite GT
  side from what `raw_channel_original` assumes.
- `oracle_min`: `min(E_intended, E_crossed)`. **DIAGNOSTIC ONLY -- never an
  inference-time metric.** It uses GT to pick the better-scoring pairing
  after the fact, which no real deployment could do.
- `raw_minus_oracle_min`: how much of `raw_channel_original`'s own error is
  a "wrong side" (correspondence) problem, versus genuine localisation
  error that would persist under either pairing.

Reading the four combinations of `prediction_x_reversal_rate` vs
`cross_pairing_preferred_rate`:

- **Both low**: no evidence of a correspondence problem; `raw_channel_original`
  is already close to `oracle_min` -- the reported error is genuine
  localisation error.
- **`cross_pairing_preferred_rate` high, `raw_minus_oracle_min` large**:
  `raw_channel_original`/the currently-reported fixed-channel NME IS
  materially inflated by p0/p1 landing on the wrong GT side -- this is the
  scenario that would support "the gap is a correspondence problem, not
  (only) localisation."
- **`prediction_x_reversal_rate` high but `cross_pairing_preferred_rate`
  low**: the raw x-order looks flipped, but distance-wise the predictions
  are still corresponding correctly -- this is the near-vertical-diameter
  noise case above; do not use the reversal rate alone to conclude there is
  a correspondence problem on BPD/TAD.
- **`prediction_x_reversal_rate` low but `cross_pairing_preferred_rate`
  high**: rare, but possible when localisation error is large enough that
  a "correctly-ordered-looking" prediction pair is nonetheless each closer
  to the opposite GT point -- treat as a genuine correspondence problem the
  x-order check alone would have missed entirely.

## Reading the summary for the supervisor conversation

This analysis answers TWO separate questions -- do not compress them into
one table, they use different columns and support different conclusions.

**Question 1 -- is EoMT's already-reported NME inflated by an evaluation
coordinate-space artifact, not genuinely worse localisation?** (this is the
question that motivated adding `raw_channel_original`): compare
`raw_channel_original_5seed_mean_pct` directly against HRNet's own reported
number for the same (dataset, task). This column is the SAME pred-channel-
vs-x-sorted-GT pairing rule `training/landmark_detection.py`'s own
`compute_nme` + `datasets/landmark_dataset.py`'s (unconditional, train-and-
test-alike) load-time x-sort already use for EoMT's historical numbers --
verified by direct code inspection and a 2000-trial randomised property
check, not assumed -- just recomputed in true original-image pixel space
instead of EoMT's own anisotropically-resized 512x512 space. If
`raw_channel_original` comes out much closer to HRNet's number than the
historical `native` figure did, that supports "the gap was mostly a
coordinate-space evaluation issue, no retraining needed." Use
`cross_pairing_preferred_rate`/`raw_minus_oracle_min` (NOT
`prediction_x_reversal_rate` -- see "Correspondence diagnostic" above for
exactly why the reversal rate alone is an unreliable proxy on BPD/TAD's
near-vertical diameters) alongside it to check the OTHER contributor: if
`raw_minus_oracle_min` remains large after the coordinate-space fix, some
of the remaining gap is a genuine channel-correspondence error, not
(only) coordinate space or localisation. Compress into:

| Dataset | Task | Method | HRNet native | EoMT native (512-space, historical) | EoMT raw_channel_original (original-space) | cross_pairing_preferred_rate | raw_minus_oracle_min |
|---|---|---|---|---|---|---|---|

**Question 2 -- which external convention should RTMPose (and the final
three-way comparison) use, given EoMT and HRNet were trained under
different ones?** Compress `endpoint_ordering_summary.tsv` into:

| Dataset | Task | Method | Native | Unified x-sort | Unified DOD | x-sort - DOD (95% CI) |
|---|---|---|---|---|---|---|

and lead with the GT disagreement rate per task (`gt_xsort_vs_dod_disagreement_rate`)
-- this is the single number that most directly answers "how much do the
two rules actually differ on this task's own ground truth," independent of
either method's prediction quality. `xsort`/`dod` deliberately RE-SORT
predictions on both sides -- this is the right tool for "which convention
should we standardise on," but the wrong tool for question 1 above, since
re-sorting a genuinely reversed prediction would mask exactly the effect
question 1 is trying to isolate.

Suggested framing for question 2 (adapt the specifics to whatever the real
numbers show):

> I retrospectively re-evaluated the existing per-image predictions under
> two common external endpoint conventions, without retraining: per-image
> x-coordinate sorting and a training-set-estimated frozen direction
> vector. The attached summary shows how much the choice changes each
> method and whether it changes the conclusions. Based on this evidence,
> which convention would you prefer us to use consistently for RTMPose
> training and the final comparison?
