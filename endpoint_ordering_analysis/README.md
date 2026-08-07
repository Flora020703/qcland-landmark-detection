# Endpoint-ordering convention analysis (no retraining)

Answers, with real numbers instead of an abstract question, the question
this project's own audit process raised and could not resolve by code
review alone (see `rtmpose_reproduction/PROTOCOL_AUDIT.md`'s "Still not
fully unified" section): EoMT canonicalises endpoints by per-image
x-coordinate sort, HRNet by a training-set-derived frozen direction
vector, and RTMPose (per `PROTOCOL_LOCKED.md`) currently follows HRNet's
convention. How much does this choice actually matter for the numbers
already reported, and does it change any conclusion?

## THE OFFICIAL EVALUATION METRIC (supervisor decision, 2026-08-07)

**Read this section first.** Everything below it (Native/Unified x-sort/
Unified DOD, `raw_channel_original`, the correspondence diagnostic,
`prediction_x_reversal_rate`) is now **Appendix-only implementation-audit
material** documenting HOW this decision was reached -- it must not be
mixed into, or presented alongside, the main results table.

After reviewing this investigation's own correspondence-diagnostic
findings (large, reproducible endpoint-assignment sensitivity on
Multicentre BPD/TAD specifically), the supervisor decided that
**permutation-invariant matching is the correct metric definition** for
this task, for EoMT, HRNet, and RTMPose alike, replacing fixed-channel NME
as the primary reported number. Rationale, in the supervisor's own words:
the two fetal-biometry endpoints define the SAME clinical measurement
regardless of which one is labelled "left"/"channel 0" -- an evaluator
that penalises a channel-identity swap is measuring an artificial
convention with no independent clinical meaning, not a real localisation
error. This is a **metric-definition** decision: no prediction coordinate
is read, modified, or selected based on GT; GT is only used, as in any
correspondence-matching evaluation, to decide which GT point counts as
matched to which predicted point for an unordered pair.

**The metric** (`permutation_invariant_nme()` in
`rescore_endpoint_conventions.py`): for each image,

```
E_direct  = ||p0 - g0|| + ||p1 - g1||
E_crossed = ||p0 - g1|| + ||p1 - g0||
E         = min(E_direct, E_crossed)
```

normalised and aggregated exactly as fixed-channel NME always was (divide
by the GT inter-endpoint distance, mean over seeds, report 5-seed
mean ± sample SD). This is mathematically identical to this project's own
`oracle_min` diagnostic (verified exactly, over 500 randomised trials, not
just argued -- see the test suite) and to HRNet's own native
`swap_min_nme` column (cross-checked against it directly on load, see
`_check_permutation_invariant_sanity`) -- the same computation, now
formally adopted as the metric rather than a diagnostic layered on top of
a different one. **`oracle_min` itself is a deprecated historical alias,
retained only for audit compatibility: its numerical operation is
identical to the official permutation-invariant NME, while the earlier
"diagnostic-only" interpretation described further below has been
superseded by this supervisor-approved metric definition.** Do not read
the "diagnostic only, never a valid metric" framing attached to
`oracle_min` further down as still applying to the underlying
computation -- it applies to that OLD interpretation, not to
`permutation_invariant_nme` itself, which is deliberately the same numbers
under a new, officially-adopted meaning.

**The official final table**: `permutation_invariant_nme_final_table.md`/
`.tsv` (written by every run, see `write_final_permutation_invariant_table()`).
UCL BPD's two EoMT cells are marked `Unavailable` -- the retained per-image
predictions and checkpoints are confirmed gone from the server, and this
is NEVER backfilled with a historical fixed-channel number computed under
a different convention.

**Common cross-method image subset, not each method's own full set**
(2026-08-07 review finding, fixed the same day): real data showed EoMT and
HRNet do not always share an identical per-image filename set for the same
(dataset, task) -- e.g. Multicentre BPD had EoMT n=1191 vs HRNet n=1180.
Comparing each method's own full, differently-sized set would silently mix
a sample-composition difference into what is meant to be a pure method
comparison. The official final table is therefore built from
`common_subset/endpoint_ordering_summary.tsv`, in which every method for a
given (dataset, task) has already been re-aggregated on the INTERSECTION
of filenames across all methods loaded for that cell -- `n` is shown
explicitly in every table cell (`mean±sd (n=...)`) and is now IDENTICAL
across methods in the same row by construction. The top-level
`endpoint_ordering_summary.tsv` (each method's own full available set) is
retained unchanged as supplementary material -- compare the two if a
cell's `n` differs between them, which signals exactly this kind of
cross-method sample mismatch.

**Strict missing-cell gate (`EXPECTED_MISSING`)** (2026-08-07 review
finding, fixed the same day): a load failure is only ever expected for UCL
BPD's two EoMT backbones (checkpoints/per-image files confirmed gone from
the server). Any OTHER load failure is, by construction, something new --
a bad path, a truncated file, a server-side regression -- and rendering it
identically as `Unavailable` in the final table would disguise a real run
problem as the one gap everyone already expects. After every run, the SET
of (dataset, task, method) cells that actually failed to load
(`actual_missing`) is compared against `EXPECTED_MISSING`: **the actual
missing set must be a SUBSET of `EXPECTED_MISSING`** -- any additional
missing cell outside the allowlist causes a non-zero exit WITHOUT
generating the official final table, printing exactly which cell(s) are
unexpectedly missing; a previously-missing cell that becomes available
this run is ACCEPTED (not an error) and reported as a non-fatal reminder
to update the allowlist, and the official table is generated using that
cell's real, freshly-recovered numbers. Every other diagnostic output
(`endpoint_ordering_summary.tsv`, `excluded_images.tsv`, etc.) is still
written first, so the reasons remain inspectable even when the run
hard-fails on this gate.

**No retraining, no new inference required**: this re-scores the SAME
already-saved per-image predictions used throughout this analysis.
Applies retrospectively to the retained EoMT and HRNet predictions, and
prospectively to RTMPose once its results exist (same evaluator function,
no method-specific special-casing).

**Independent cross-codebase verification, and its current asymmetry**
(2026-08-07 review finding): HRNet's own per-image CSV already carries an
independently-computed `swap_min_nme` column, written by a completely
different script (`baseline_reproduction/evaluate_hrnet_fixed.py`) --
`load_hrnet_per_image` cross-checks this module's `permutation_invariant_nme`
against it on every load (`_check_permutation_invariant_sanity`), a genuine
cross-codebase check, not just two code paths in this same file agreeing.
Because HRNet's coordinates are already in original-image space (no
conversion needed), this check validates the ACTUAL number that ends up in
the official table.

`load_eomt_per_image` performs the SAME cross-check, best-effort, against
an optional `*_final_swapmin_per_image.csv` companion file if one happens
to exist next to the required fixed-channel dump -- but whether such a
file genuinely exists for any given EoMT run is not currently confirmed.
When it is absent, the run prints
`[permutation-invariant sanity SKIPPED] ...` and proceeds; in that case,
`permutation_invariant_nme` for that EoMT cell is only verified against
this module's own `oracle_min` (same script) and the 500-trial randomised
property test in the test suite.

**When the companion file IS present and the check passes, do not overstate
what it proves** (2026-08-07, third review round): the check runs against
EoMT's RAW dumped coordinates, in its own 512x512 model-input space,
BEFORE the anisotropic-resize inversion to true original-image pixel
space -- the same space the companion file's own value would have been
computed in. It therefore validates CSV alignment (index<->filename join),
coordinate parsing, and the `min(direct, crossed)` assignment LOGIC, not
the final NUMERIC `permutation_invariant_nme` value that actually appears
in the official table. That final value is computed AFTER the anisotropic
inversion (see the coordinate-space notes above), and because that
inversion is generally anisotropic for these (essentially always
non-square) ultrasound images, neither the Euclidean NME magnitude NOR, in
principle, the direct-vs-crossed assignment decision itself is guaranteed
invariant between the two spaces. So: EoMT's official original-space NME
numbers are, today, verified against this module's own math (`oracle_min`
equivalence + 500-trial property test) but NOT against an independent
codebase's number IN THE SAME SPACE the official table reports -- weaker
evidence than HRNet's equivalent check, which does verify the space that
matters. State this precisely if asked how thoroughly EoMT's official
numbers were cross-validated; do not describe the EoMT companion check as
closing the same gap HRNet's does.

**Explicitly out of scope for the main table, per the same decision**
(kept fully computed below for the Appendix/audit trail, not deleted):
prediction-only x-sorting, frozen-DOD canonicalisation, and raw
channel-identity preservation as a primary metric. These were tested as
candidate GT-independent inference-time fixes BEFORE the permutation-
invariant decision; frozen-DOD in particular came very close to the
GT-informed diagnostic value and remains documented below as it may
inform a future redesign of EoMT's own heatmap-channel convention if the
models are ever retrained -- but is not needed to evaluate the current
models under an unordered-pair metric.

## What this does and does not do (Appendix -- historical audit trail below this line)

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
  permutation_invariant_nme_final_table.md    *** THE OFFICIAL RESULT ***. Ready-to-paste
                                               Markdown table, permutation-invariant NME (%)
                                               +-5-seed sample SD, WITH n SHOWN EXPLICITLY per
                                               cell, per (dataset, method, task) -- built from
                                               common_subset/ below, not the top-level
                                               endpoint_ordering_summary.tsv. Missing cells
                                               (UCL BPD EoMT) marked "Unavailable", never
                                               backfilled. Only generated if the actual set of
                                               load failures is a SUBSET of EXPECTED_MISSING
                                               (see the strict missing-cell gate above) --
                                               otherwise the run exits non-zero before this
                                               file is written.
  permutation_invariant_nme_final_table.tsv   same data, machine-readable
  common_subset/                              *** the OFFICIAL common-cross-method-subset
                                               re-aggregation this final table is built from ***
                                               -- same file layout as the top level (below),
                                               but every method's per-image set is first
                                               restricted to the filenames ALL methods loaded
                                               for that (dataset, task) share.
    endpoint_ordering_summary.tsv             one row per (dataset, task, method), on the
                                               common subset -- `n_images` is identical across
                                               methods in the same (dataset, task) by construction
    endpoint_ordering_seed_summary.tsv        (same, per seed)
    *_per_image.csv                           per-image rows restricted to the common subset
  endpoint_ordering_summary.tsv               SUPPLEMENTARY: one row per (dataset, task, method),
                                               on EACH METHOD'S OWN FULL available set (sample
                                               counts may legitimately differ across methods --
                                               compare `n_images` here against common_subset/'s
                                               copy to see by how much). 5-seed mean+-SD for
                                               permutation_invariant_nme (the official metric)
                                               PLUS every Appendix-only convention below, GT
                                               disagreement rate, raw-channel-vs-prediction-
                                               x-sort audit, prediction reversal rate, and
                                               x-sort-vs-DOD mean difference + bootstrap 95% CI
  endpoint_ordering_seed_summary.tsv          one row per (dataset, task, method, seed, convention),
                                               full set (supplementary, see above)
  ucl_*_per_image.csv                         per-image, 5-seed-averaged NME under all 3
                                               conventions, in unified original-image coordinates,
                                               full set (supplementary)
  multicentre_*_per_image.csv                 (same, Multicentre, full set)
  dod_vectors.tsv                             every frozen (dataset, task) direction vector used
  excluded_images.tsv                         every (dataset, task, method) cell that could not
                                               be scored, and exactly why (missing files, missing
                                               coordinate columns, missing original image file) --
                                               never silently dropped. Checked after every run
                                               against EXPECTED_MISSING (see above); any cell here
                                               NOT in that constant hard-fails the run.
  cross_method_gt_consistency_warnings.tsv    any (dataset, task) where HRNet's and EoMT's own
                                               GT (after EoMT's original-space conversion) disagree
                                               on location or on x-sort-vs-DOD disagreement rate --
                                               a non-empty file signals a coordinate-recovery or
                                               sample-matching bug, not a real dataset property
  raw_channel_vs_prediction_xsort_summary.tsv (APPENDIX / historical) a compact extract of
                                               endpoint_ordering_summary.tsv's raw-channel/
                                               prediction-x-sort columns only -- part of the
                                               audit trail that led to the permutation-invariant
                                               decision above, not part of the main result
  correspondence_diagnostic_summary.tsv       (APPENDIX / historical) the analysis that directly
                                               motivated the supervisor's decision above: does
                                               p0/p1 actually correspond, by distance, to the
                                               correct GT side, or only look reversed in raw
                                               x-order? See "Correspondence diagnostic" below
```

**Everything from here to "Reading the summary for the supervisor
conversation" is Appendix / historical audit-trail material** -- how the
permutation-invariant decision above was reached, kept in full for
provenance, not part of the main result.

For the (pre-decision) audit trail's own "immediate training/inference
correspondence question," these columns from `endpoint_ordering_summary.tsv`
were used:

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

## Correspondence diagnostic (`correspondence_diagnostic_summary.tsv`) -- APPENDIX / historical

**This section documents the analysis that led directly to the
permutation-invariant decision at the top of this README.** It is retained
in full as the audit trail; the numbers here (`raw_channel_original`,
`cross_pairing`, `oracle_min`, etc.) are Appendix material, not the main
result -- see `permutation_invariant_nme_final_table.md` for that.

`prediction_x_reversal_rate` only checks whether the RAW prediction pair
keeps its own left-to-right order (`pred0.x > pred1.x`) -- it does NOT
check whether p0 is actually the point closer to the intended-first GT and
p1 actually closer to the intended-second GT. These are different
questions. On a near-vertical diameter (BPD/TAD, where the two GT points
differ by a fraction of a pixel in x but tens of pixels in y), a tiny
prediction x error can flip `prediction_x_reversed` to `True` while p0/p1
are each still, unambiguously, closest to their own correct GT point --
reversal here is x-sort noise, not a correspondence error (see
`test_correspondence_diagnostic_distinguishes_reversal_from_true_swap` in
the test suite for a constructed example, and the real BPD/TAD d_vect
geometry in `dod_vectors.py` for why this is the realistic regime, not an
edge case, for exactly those two tasks).

**"Intended" is METHOD-SPECIFIC, not always x-sort** (fixed 2026-08-07,
review finding against the first version of this diagnostic): EoMT's own
training convention is x-sort, but HRNet's is DOD. `rescore_cell()`'s
`native_convention` parameter ("xsort" for EoMT, "dod" for HRNet, set in
`main()`) picks which GT ordering counts as "intended" for each method --
using x-sort as "intended" for HRNet too would structurally confound
HRNet's own numbers with the already-known DOD-vs-x-sort disagreement
rate, exactly the same confound this diagnostic exists to escape for
`prediction_x_reversal_rate`. Do not interpret HRNet's and EoMT's columns
here as "the same measurement" -- each is relative to that method's own
training-time GT convention.

The direct, convention-agnostic answer is a per-image bipartite-distance
comparison, computed in `rescore_cell()`:

- `raw_channel_original` (`E_intended`): `||p0-g_intended0|| + ||p1-g_intended1||`
  -- the AS-TRAINED pairing, already described above.
- `cross_pairing` (`E_crossed`): `||p0-g_intended1|| + ||p1-g_intended0||` --
  the OPPOSITE pairing.
- `pairing_status` / `cross_pairing_preferred`: per image, one of
  `intended_preferred`, `crossed_preferred`, or `approximately_tied`
  (gated by `PAIRING_TOL`, an absolute tolerance on the NME-fraction scale
  -- fixed 2026-08-07 so a near-exact tie between `E_intended` and
  `E_crossed` isn't arbitrarily assigned to whichever side of zero
  floating-point noise landed on). `cross_pairing_preferred` is `True` iff
  `pairing_status == "crossed_preferred"`.
- `oracle_min`: `min(E_intended, E_crossed)`. At the time this diagnostic
  was built, **DIAGNOSTIC ONLY -- never an inference-time metric**: it uses
  GT to pick the better-scoring pairing after the fact, which no real
  deployment could do. **This interpretation is now superseded** (see "THE
  OFFICIAL EVALUATION METRIC" at the top of this README): the supervisor
  subsequently adopted exactly this computation, under the name
  `permutation_invariant_nme`, as the official metric definition for an
  unordered pair of clinically-equivalent endpoints -- `oracle_min` is kept
  here, unchanged, only as a deprecated historical alias for audit-trail
  continuity; do not read the "diagnostic only" framing below as still
  describing the underlying number today, only the OLD framing this
  diagnostic was originally built and named under.
- `raw_minus_oracle_min`: an **oracle-reassignment reduction / correspondence-
  SENSITIVITY diagnostic** -- how much `raw_channel_original` could shrink
  under GT-informed reassignment. **This is not a causal decomposition**
  (tightened 2026-08-07, review finding: the original wording called this
  "how much error is caused by wrong correspondence," which overclaims what
  a post-hoc, minimum-biased statistic can prove). Report it as: "the
  fixed-channel score is sensitive to endpoint assignment by X pp," never
  as "X pp of error is caused by a channel/query-identity swap" -- a large
  value does not, by itself, rule out (a) a badly localised prediction
  scoring lower under the opposite pairing by chance, or (b) the fact that
  these endpoints have no independent clinical identity to swap in the
  first place.

Reading the four combinations of `prediction_x_reversal_rate` vs
`cross_pairing_preferred_rate` (within one method -- do not compare these
rates ACROSS HRNet and EoMT directly, see the native-convention note above):

- **Both low**: no evidence of endpoint-assignment sensitivity;
  `raw_channel_original` is already close to `oracle_min` -- the reported
  error is not materially reducible by reassignment.
- **`cross_pairing_preferred_rate` high, `raw_minus_oracle_min` large**:
  `raw_channel_original`/the currently-reported fixed-channel NME IS
  materially sensitive to which GT side p0/p1 are scored against -- this is
  the scenario that would support "part of the gap is an endpoint-
  assignment sensitivity issue, not (only) localisation," though see above
  for why this still isn't a proven causal claim.
- **`prediction_x_reversal_rate` high but `cross_pairing_preferred_rate`
  low**: the raw x-order looks flipped, but distance-wise the predictions
  are still corresponding correctly -- this is the near-vertical-diameter
  noise case above; do not use the reversal rate alone to conclude there is
  an assignment-sensitivity issue on BPD/TAD.
- **`prediction_x_reversal_rate` low but `cross_pairing_preferred_rate`
  high**: rare, but possible when localisation error is large enough that
  a "correctly-ordered-looking" prediction pair nonetheless scores lower
  under the opposite pairing -- worth a closer per-image look, but still
  subject to the same non-causal caveat above.

## Reading the summary for the supervisor conversation -- APPENDIX / historical

**Superseded by the permutation-invariant decision at the top of this
README for the MAIN result.** Retained as the record of how "Question 1"
below was actually resolved (see "THE OFFICIAL EVALUATION METRIC" section)
and because "Question 2" (endpoint-ordering convention choice) remains
independently useful context for RTMPose/future redesign discussions, even
though it is no longer needed to score the CURRENT models.

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
`raw_minus_oracle_min` remains large after the coordinate-space fix, the
remaining gap is at least SENSITIVE to endpoint assignment, not purely
coordinate space -- but per the "Correspondence diagnostic" section's own
caveat, this is a sensitivity signal, not proof of what caused it.
Compress into:

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
