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
   already used (reproduces the already-reported numbers, as a sanity
   check on this script's own coordinate handling).
2. **Unified x-sort**: both GT and prediction independently re-labelled by
   ascending x (tie-break by y).
3. **Unified DOD**: both GT and prediction independently re-labelled by
   projecting onto the SAME frozen, training-set-only direction vector
   (`rtmpose_reproduction/dod_vectors.py` -- the exact vectors already
   verified against real HRNet checkpoints and real HRNet per-image output
   in that adapter's own test suite; never re-estimated from Test GT here).

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

## Usage

```bash
python endpoint_ordering_analysis/rescore_endpoint_conventions.py \
    --ucl-eomt-root /root/autodl-tmp/ucl_eomt_per_image \
    --ucl-hrnet-root /root/autodl-tmp/hrnet_512_fixed_5seed/output/FETAL \
    --multicentre-eomt-root /root/autodl-tmp/saved_checkpoints/multicentre_5seed \
    --multicentre-hrnet-root /root/autodl-tmp/hrnet_512_fixed_5seed/output/FETAL \
    --output-root endpoint_ordering_analysis/results
```

The four `--*-root` defaults already match the real paths used by this
project's own `final_comparison/analyse_ucl_per_image.py` and
`analyse_multicentre_per_image.py` -- override only if your copy of the
data lives somewhere else.

Run the local test suite first (pure Python, no real data needed, exercises
the loading/canonicalisation/bootstrap code against synthetic files
matching the real CSV schemas exactly):

```bash
python endpoint_ordering_analysis/test_rescore_endpoint_conventions.py
```

## Outputs

```
endpoint_ordering_analysis/results/
├── endpoint_ordering_summary.tsv       # one row per (dataset, task, method): 5-seed mean+-SD
│                                         for all 3 conventions, GT disagreement rate,
│                                         x-sort-vs-DOD mean difference + bootstrap 95% CI
├── endpoint_ordering_seed_summary.tsv  # one row per (dataset, task, method, seed, convention)
├── ucl_*_per_image.csv                 # per-image, 5-seed-averaged NME under all 3 conventions
├── multicentre_*_per_image.csv         # (same, Multicentre)
├── dod_vectors.tsv                     # every frozen (dataset, task) direction vector used
└── excluded_images.tsv                 # every (dataset, task, method) cell that could not be
                                          # scored, and exactly why (missing files, missing
                                          # coordinate columns) -- never silently dropped
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
