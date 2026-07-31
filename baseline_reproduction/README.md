# HRNet/BiometryNet baseline reproduction (UCL + Multicentre)

This directory runs the supervisors' HRNet-based BiometryNet pipeline on the
official UCL and pooled Multicentre Train/Test CSVs. It provides the locally
reproduced baseline used alongside the published BiometryNet values and EoMT.

## Source and permitted changes

The source is cloned directly on the training server from the official
`surgical-vision/Multicentre-Fetal-Biometry` GitHub repository and detached at
commit `21ee7cd70b9a3cee58d85dfb50b089dda6076867`. The local clone at
`msc/hrnet_new/Multicentre-Fetal-Biometry` is retained only as an audit copy and
is not used to deploy the reproduction. Do not use the sibling
`hrnet_ repo` checkout: it contains the earlier CPU sanity-check edits and is
not evidence for the thesis.

Apply the required runtime compatibility patch to the HRNet checkout:

```bash
git apply /path/to/baseline_reproduction/forward_compat.patch
```

It imports Python's `math` module and replaces the removed `np.math.floor`
alias with `math.floor`; crop arithmetic is unchanged. It also makes
`tools/test.py` explicitly use `weights_only=False` when loading this
reproduction's own trusted local `final_state.pth`. This preserves the
pre-PyTorch-2.6 `torch.load` behaviour required by the checkpoint's NumPy
`d_vect` payload. Neither compatibility edit changes the model, checkpoint
values, training protocol or evaluator. The driver verifies both edits, the
source commit and the pretrained-weight SHA-256 before running.

The ten canonical YAML files copy the source configs for
`UCL/MULTICENTRE x BPD/OFD/APAD/TAD/FL`. Only `DATASET.ROOT`, `TRAINSET` and
`TESTSET` point to server paths. Structural comparison hard-fails if any
other canonical field changes. Generated runtime copies also redirect
`OUTPUT_DIR` and `LOG_DIR` to the data disk; this changes artifact placement,
not the experiment.

The original sigma (1.0), 256x256 input, 64x64 heatmap, MSE loss, HRNet-W18,
augmentation, optimiser, batch size and 200-epoch schedule are retained. The
driver checks CSV hashes, row counts and referenced images against
`expected_data_sha256.tsv`.

## Repeats: not controlled seeds

The upstream code exposes no seed control. The formal experiment therefore
uses **five independent training repeats with uncontrolled random seeds**.
Repeat numbers isolate output directories; they are not seed values and must
not be paired seed-by-seed with EoMT's fixed seeds. Report the mean and
repeat-level sample SD across the five run-level NMEs. Upstream `nme std` is
stored separately as `per_image_nme_sd_pct`: it is the within-test-set spread,
not repeat-level uncertainty.

## Inherited validation/test protocol

The source `train.py` constructs its per-epoch validation loader from the
configured `TESTSET`; it does not carve out a separate validation set. The
official test split is therefore evaluated after every epoch and drives
creation of `model_best.pth`. This reproduction preserves that behaviour
rather than inventing a new split. Formal testing uses upstream
`final_state.pth`, matching `run_all_tests.sh`, and does not select
`model_best.pth`. The test split was nevertheless monitored during training;
this is not a training-unseen held-out evaluation in the strict sense.

The upstream fetal metric is endpoint-order-invariant swap-min NME. Saved
`predictions.pth` files are retained for later fixed-channel re-evaluation.

## Canary, storage and resumption

Canary output has `_CANARY` in its run name and is written to
`hrnet_canary_results.tsv`; it is never reused as a formal repeat.

The upstream trainer writes a full optimiser checkpoint every epoch. The
driver places artifacts under `/root/autodl-tmp/hrnet_reproduction` and,
while training, retains the two newest numbered epoch checkpoints. After
testing it retains the newest numbered checkpoint, upstream best/final
states, predictions, configs and logs. This is storage housekeeping only and
does not alter optimisation. By default the driver stops after one complete
dataset/task configuration (five repeats), allowing archival before the next
invocation.

## Usage

```bash
export HRNET_REPO_ROOT=/root/Multicentre-Fetal-Biometry

# Start with Multicentre. The isolated canary defaults to the first selected
# config (Multicentre BPD) and never enters the formal TSV.
DATASET_FILTER=MULTICENTRE CANARY_ONLY=1 \
  bash /path/to/baseline_reproduction/run_hrnet_reproduction.sh

# Runs the next incomplete Multicentre configuration (five uncontrolled
# repeats), then stops for archival. Reinvoke for each subsequent task.
DATASET_FILTER=MULTICENTRE \
  bash /path/to/baseline_reproduction/run_hrnet_reproduction.sh

# Run UCL later under the same driver.
DATASET_FILTER=UCL \
  bash /path/to/baseline_reproduction/run_hrnet_reproduction.sh
```

`DATASET_FILTER` accepts only `ALL`, `UCL` or `MULTICENTRE`. An optional full
config stem may be supplied through `CANARY_CONFIG`, but it must belong to the
selected dataset. Filtering changes only orchestration order; canonical YAML
content is untouched.

Set `STOP_AFTER_CONFIG=0` only after deliberately provisioning enough disk
and unattended runtime. Every invocation atomically refreshes
`environment_audit.txt` with the current software stack, CUDA/cuDNN, GPU,
source commit, pretrained hash and `pip freeze`; the append-only
`environment_audit_history.txt` preserves canary/formal snapshots and exposes
any dependency change between runs.

## Final matched 512/fixed-channel protocol (2026-07-31)

The legacy driver above reproduces the released 256×256/swap-min protocol and
is retained for audit purposes. It is not the final matched comparison. The
final comparison uses `run_hrnet_512_fixed_5seed.sh`, which performs ten
measurement-specific configurations (UCL and Multicentre × BPD/OFD/APAD/TAD/FL)
at five pre-fixed seeds (`42, 0, 123, 2024, 3407`) in one resumable invocation.

The new driver changes input size to 512×512 and evaluates the final checkpoint
with fixed-channel NME. HRNet-W18 has a native output stride of four, so its
heatmap changes correspondingly from 64×64 at a 256×256 input to 128×128 at a
512×512 input. Sigma remains 1.0, and the released
MSE/augmentation/optimiser/DOD recipe is otherwise retained. The
upstream native swap-min result is kept as a diagnostic. The external evaluator
writes both metrics and the raw prediction/ground-truth coordinates for every
test image.

To make the five runs controlled seeds rather than uncontrolled repeats, the
driver idempotently invokes `apply_controlled_seed_patch.py`. That audited patch
sets Python, NumPy, PyTorch and CUDA RNGs, seeds the DataLoader generator and
workers, and enables deterministic cuDNN behaviour. The already documented
`forward_compat.patch` must also be present in the upstream checkout.

The driver also idempotently invokes `apply_dynamic_decode_size_patch.py`.
The released `lib/core/function.py` passes a literal `[64, 64]` to
`decode_preds` in training, validation, and inference. That literal is valid
for the released 256x256/64x64 protocol but would silently decode native
128x128 predictions in the wrong coordinate system at 512x512. The audited
patch replaces exactly those three literals with
`list(config.MODEL.HEATMAP_SIZE)`. The driver refuses to run unless all three
sites are patched and no hard-coded site remains. For the old 256x256 configs
this evaluates to the same `[64, 64]`, so their historical behaviour is
unchanged.

```bash
cd /root/eomt
export HRNET_REPO_ROOT=/root/Multicentre-Fetal-Biometry
export PY=/root/hrnet_repro_env/bin/python

# Verify all ten configs and data manifests without starting training.
PREFLIGHT_ONLY=1 bash baseline_reproduction/run_hrnet_512_fixed_5seed.sh

nohup bash baseline_reproduction/run_hrnet_512_fixed_5seed.sh \
  > hrnet_512_fixed_5seed.log 2>&1 &
```

The first UCL-BPD/seed-42 run is the automatic integration gate: any training,
test, prediction-shape, finite-value, data-checksum, or metric-invariant failure
stops the sweep. If it passes, the same process continues through the remaining
49 runs without manual group restarts. A partial directory without
`final_state.pth` causes a deliberate hard failure instead of a statistically
non-reproducible resume; inspect and remove only that exact failed run before
restarting the driver.

All output, TensorBoard data, generated configs, temporary files and runtime
caches are rooted under `/root/autodl-tmp/hrnet_512_fixed_5seed`. The driver
exports `TMPDIR`, `TMP`, `TEMP`, `XDG_CACHE_HOME`, `TORCH_HOME`,
`CUDA_CACHE_PATH` and `MPLCONFIGDIR` accordingly, preventing the sweep from
gradually filling the smaller system disk with run-generated cache files.

The first integration attempt exposed why the heatmap grid must scale with the
input: a 512×512 image produced a 128×128 HRNet output while a mistakenly
retained 64×64 target caused MSE to fail before the first optimisation step.
The gate stopped all remaining runs. No result was produced from that failed
attempt, and no architectural resize layer was added to conceal the mismatch.

A subsequent static audit found a second, silent 512x512 integration issue:
the released coordinate decoder hard-coded a 64x64 output grid at three call
sites. Unlike the loss-shape failure, this would not crash; it would corrupt
validation NME, checkpoint selection, and test coordinates. No successful
512x512 result produced before this correction is admissible. The dynamic
decode-size patch above is therefore a mandatory correctness patch rather than
a model or optimisation change.
