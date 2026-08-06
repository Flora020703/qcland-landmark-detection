# Code-level protocol audit, before canary training

Two review rounds on 2026-08-06. The first round covered the two items the
supervisor explicitly flagged (augmentation-strategy parity, endpoint
ordering/flip convention) and found a real bug, but with an INCORRECT root
cause. A second, independent review of that first round's own work caught
the error with a rigorous mathematical argument, checked directly against
the audited upstream HRNet's actual source, and additionally found four
more blocking implementation issues the first round missed entirely. This
document keeps both rounds' history rather than silently overwriting it,
per this project's own established norm of marking superseded findings
`SUPERSEDED`, not deleting them.

## SUPERSEDED: first-round "100% flip-order bug" framing was mathematically wrong

**What the first round claimed**: `audit_flip_order_stability.py` measured
that a static `flip_indices=[0,1]` design (mirror points on flip, never
re-derive channel order) disagreed with a fresh DOD projection on 100% of
UCL OFD/APAD/FL training images, and concluded this was a genuine bug
because it assumed HRNet's own `_transform_pixel_float(d_vect, center,
scale, output_size, rot)` "re-derives the projection after every
augmentation draw" by meaningfully transforming `d_vect` through the
sample's own center/scale/rotation. The fix applied (`fetal_augment.py`'s
first version) transformed `d_vect` through flip AND rotation before
re-projecting, in already-512-space.

**Why this was wrong, verified against the real HRNet source (`lib/utils/
transforms.py`'s `get_transform`, `lib/datasets/fetal.py`'s DOD
reassignment block)**:

`get_transform(center, scale, output_size, rot)` produces an affine map of
the form `new = L(rot, scale) @ (point - center) + output_size/2`, where
`L` (a rotation composed with an isotropic scale) does NOT depend on
`center`. A sample's own keypoints AND the frozen `d_vect` prototype points
are both passed through this SAME map with the SAME (center, scale, rot)
for a given sample. Two direct consequences, both proven algebraically and
confirmed empirically by literally re-implementing HRNet's exact formula
and testing it against 400 randomised (center, scale, rotation) draws with
ZERO mismatches (script: `verify_hrnet_dvect_cancellation.py`, run
2026-08-06, not checked into the repo -- a scratch verification, kept only
in this document as its written record):

1. `center` cancels EXACTLY in any pairwise comparison (`proj_i - proj_j`),
   because it enters as an additive term shared by both points of a pair.
   This means the "mirrored center" HRNet builds when a sample is flipped
   has **zero effect** on the derived projection direction.
2. `L` is a similarity transform (orthogonal times a positive scalar), so
   the dot product of two vectors both passed through the same `L` changes
   only by a positive scalar factor -- meaning `scale` and `rot` ALSO
   cancel for the ordering decision specifically (they still matter for
   the actual heatmap-target coordinates, just not for which point is
   called channel 0).

**The real, verified fact**: HRNet's own per-sample DOD ordering decision
reduces EXACTLY to comparing the sample's raw, ORIGINAL-image-space
keypoints -- flip-mirrored if a flip was drawn that epoch, untouched
otherwise -- against the STATIC, NEVER-TRANSFORMED original `d_vect`. This
is precisely what `audit_flip_order_stability.py` computed as its
"counterfactual" -- meaning the measured 100%/0% pattern IS an accurate
description of HRNet's OWN real training-time behaviour, not a divergence
from it. **HRNet's own flip handling has exactly this instability** for
near-horizontal-`d_vect` tasks (OFD/APAD/FL): a training sample's
channel-0/1 assignment genuinely depends on that epoch's own flip draw.
This is a real, pre-existing property of the audited upstream reference
implementation, not something introduced by this adapter, and not
something to "fix" relative to HRNet's own convention -- it is the
convention.

**Consequence**: the first round's "fix" (transforming `d_vect` through
flip AND rotation, done in already-resized 512-space) was itself a NEW
divergence from HRNet's real behaviour, compounded by a second, separate
error: doing this comparison in already-resized 512-space uses this
project's own ANISOTROPIC pixel-centre resize, which does not preserve the
sign of a dot-product-based order comparison for a non-square source image
the way HRNet's isotropic crop-scale does. Both errors are corrected below.

## Corrected design (second pass, `fetal_augment.py`/`transforms.py` rewritten)

- `resolve_channel_order_after_flip` (`fetal_augment.py`): the ONLY
  function that touches `d_vect`. Runs on ORIGINAL image-pixel-space
  points, mirrors them if a flip is drawn, and compares against the
  STATIC, un-transformed `d_vect` -- exactly reproducing HRNet's real,
  verified behaviour. Must run BEFORE the anisotropic pixel-centre resize
  (new transform `transforms.FetalRandomFlipAndCanonicalize`, now FIRST in
  the train pipeline, before `PixelCentreResize`).
- `sequential_rotate_scale` (`fetal_augment.py`): rotation/scale as pure
  position updates in already-512-space, no `d_vect` involved at all
  (proven to have zero effect on the ordering decision) -- retains EoMT's
  own independent per-stage accept/reject-if-out-of-canvas policy purely
  to keep points on-canvas, not for ordering. Runs in the new
  `transforms.FetalRotateScaleColorJitter`, AFTER `PixelCentreResize`.
- 7 tests in `test_fetal_augment.py`, rewritten to test the corrected
  functions (including a re-derivation-consistent regression test for the
  real UCL OFD case and a 500-case randomised property test of the
  "same original point always wins channel 0" invariant that DOES hold
  under this corrected design, unlike the abandoned flip-transforms-d_vect
  approach). All pass.

**Practical upshot for the adapter's own results (not HRNet's)**: since the
static-direction design is being kept (not "fixed") for BPD/TAD (whose
`d_vect` is near-vertical, ~0% affected) and OFD/APAD/FL (whose `d_vect` is
near-horizontal, ~100% affected), RTMPose's OWN training targets for
OFD/APAD/FL will have the SAME per-epoch channel-identity instability
HRNet's real training data has. This is disclosed, not hidden, and is now
understood to be a property of the shared method (both HRNet and RTMPose,
by faithful replication), not an adapter-specific defect.

## Still not fully unified: EoMT uses a THIRD, different convention

Re-confirmed via source (`datasets/landmark_dataset.py` line 274-277,
`ablation/ensemble_test.py`'s `dod_sort`): EoMT uses a per-sample
**ascending-x-coordinate sort**, recomputed fresh at every access, AFTER
all geometric augmentation -- equivalent to projecting onto a fixed PURELY
HORIZONTAL direction, not a learned direction at all, and NOT subject to
the same "stale frozen order" concern HRNet/RTMPose have (it is always
freshly recomputed, so it cannot go stale) -- but it IS a different rule
from HRNet/RTMPose's learned-direction convention.

Quantified disagreement between EoMT's x-sort and HRNet's DOD, on real UCL/
Multicentre training data:

| Dataset | Task | d_vect axis | EoMT-x-sort vs HRNet-DOD disagreement |
|---|---|---|---:|
| UCL | BPD | near-vertical | 45.5% |
| UCL | OFD | near-horizontal | 100.0% |
| UCL | APAD | near-horizontal | 0.0% |
| UCL | TAD | near-vertical | 42.6% |
| UCL | FL | near-horizontal | 0.0% |
| Multicentre | BPD | near-vertical | 48.3% |
| Multicentre | TAD | near-vertical | 41.5% |

**This is a genuinely unresolved item, not something this session can close
by itself.** The supervisor's instruction was "use the same
endpoint-ordering and horizontal-flip conventions across RTMPose, HRNet and
EoMT." As implemented: RTMPose and HRNet share an identical, code-verified
convention (learned `d_vect` direction). EoMT uses its own, different,
already-locked convention (x-sort) that cannot be changed without
retraining EoMT's already-reported 5-seed results. Two honest paths
forward, neither of which this session can decide unilaterally:

1. **Accept "method-native training ordering + common external evaluation
   ordering"**: each method keeps its own training-time convention (since
   EoMT can't be retrained), but the EXTERNAL fixed-channel/swap-min
   evaluator is what actually gets compared in the thesis, and that
   evaluator is identical code (`evaluate_hrnet_fixed.py`/
   `evaluate_rtmpose_fixed.py`, same formula) applied to each method's own
   already-decided GT channel assignment. This is the status quo (already
   true for EoMT-vs-HRNet before RTMPose existed) and is already partially
   disclosed in the thesis's "correspondence gap"/endpoint-canonicalisation
   limitation discussion.
2. **Re-canonicalise all three methods' raw predictions offline, post hoc,
   under one single shared rule** (e.g. always use HRNet's `d_vect`
   convention as the arbiter for every method's predictions AND ground
   truth, recomputed at evaluation time only): this would make the
   EXTERNAL comparison genuinely apples-to-apples on ordering, at the cost
   of being a new analysis step not yet implemented for any of the three
   methods' historical results, and would still not make EoMT's OWN
   TRAINING targets consistent with this rule (EoMT was trained under
   x-sort supervision; its predictions' channel identity reflects that,
   not a retrained-under-DOD identity).

This must go back to the supervisor as an explicit, named open question --
not silently treated as resolved. Do not write in the thesis or canary
report that all three methods now share one endpoint-ordering convention;
they do not.

## Second-round review: four additional blocking issues, all fixed

### 1. Test set was being used as validation (real data leakage, not a soft violation)

`make_config.py`'s original `val_dataloader` pointed at the released Test
annotation file, with `test_dataloader = val_dataloader`, PCK computed on
it every `val_interval` epochs, and `default_hooks.checkpoint`'s
`save_best="PCK"` selecting a checkpoint from that same Test-derived
metric. This is a genuine leak regardless of the fact that the REPORTED
result was always going to be the final checkpoint -- the official Test
set must never be read during training at all, not just never used for
final selection.

**Fixed**: new `make_internal_val_split.py` builds a Train-only internal
validation split, REUSING EoMT's own exact subject-grouping/shuffle/split
algorithm (`datasets/landmark_dataset.py`'s `_subject_id`/
`_split_by_subject`, ported verbatim: `re.match(r"^(\d+)", filename)`
subject extraction, `np.random.default_rng(val_split_seed=42)` shuffle,
`ceil(N_subjects * val_fraction=0.1)` held out) rather than an
independently invented split. Run against the real UCL BPD Train CSV: 100
internal-train / 10 internal-val (of 110 total Train rows) -- this exact
100/10 figure independently matches a split size already referenced
elsewhere in this project's own records for UCL's internal validation,
which is a strong external check that the ported algorithm is correct, not
just internally self-consistent. `convert_csv_to_coco.py` gained a
`--internal-split-json`/`--internal-split-part` option to materialise this
split as two separate COCO jsons from the SAME Train CSV.
`make_config.py`'s `val_dataloader` now points at the internal-val json;
`test_dataloader` is a SEPARATE dataloader over the real Test set, touched
only once, after training, by `run_inference.py` -- nothing in the
training loop reads it. `default_hooks.checkpoint` no longer uses
`save_best` at all (removed entirely, not just repointed at the
now-legitimate internal metric) -- `save_last=True` guarantees the true
final-epoch checkpoint always exists, matching PROTOCOL_LOCKED.md's
"final/last" primary convention without ambiguity.

### 2. `run_inference.py` bypassed the model's data preprocessor

The original script called `model.extract_feat(inputs)` directly on the
raw tensor from the dataset pipeline (`LoadImage -> PixelCentreResize ->
PackPoseInputs`), never calling `model.data_preprocessor` (mean/std
normalisation, BGR->RGB if configured). The pipeline transforms do not
apply this normalisation themselves -- it happens at model-forward time,
which is how the model was trained. Feeding un-normalised pixel values
would run without error but make every exported coordinate meaningless.

**Fixed**: `run_inference.py` now calls `model.data_preprocessor({"inputs":
[data["inputs"]], "data_samples": [data_sample]}, False)` before
`extract_feat`, while still deliberately stopping short of
`model.test_step()`/`predict()` (which would additionally invoke MMPose's
stock bbox-based inverse transform this project must avoid). The exact
call contract for a single manually-collated sample is flagged in
`ENVIRONMENT.md`'s checklist as needing live confirmation, same tier as
the rest of this file's MMPose-dependent assumptions.

### 3. Final-checkpoint selection was neither deterministic nor guaranteed final

`run_rtmpose_canary.sh` selected the checkpoint via `find ... -name
"best_PCK_epoch_*.pth" -o -name "epoch_*.pth" | sort | tail -1` --
lexicographic sort is wrong for numeric epoch counts (`epoch_95.pth` sorts
after `epoch_200.pth`), and the glob could pick up a "best" checkpoint file
over the true final one. PROTOCOL_LOCKED.md requires the true final/last
checkpoint as the primary result.

**Fixed**: the script now reads MMEngine's own `last_checkpoint` pointer
file (written by the Runner every time `CheckpointHook` saves, standard
MMEngine behaviour -- flagged for live confirmation of the exact
filename/format) and explicitly asserts its filename is `epoch_
{MAX_EPOCHS}.pth`, failing loudly rather than falling back to any other
checkpoint if this doesn't hold. Combined with fix #1's `save_best`
removal, there is no longer any "best" checkpoint file for a bug to
silently prefer.

### 4. `record_run_provenance.py` never actually verified checkpoint loading

The original script only called `MODELS.build(cfg.model)` and assumed the
backbone's `init_cfg=dict(type='Pretrained', ...)` had already taken
effect -- in MMEngine's convention, `init_cfg` only declares HOW to
initialise; actual loading happens when `model.init_weights()` runs, which
building alone does not guarantee. The script also documented recording
"which keys loaded" without ever actually checking, and
`run_rtmpose_canary.sh` never passed `--pretrained-checkpoint-path`, so the
recorded SHA-256 would always have been `null`.

**Fixed**: `model.init_weights()` is now called explicitly;
`--pretrained-checkpoint-path` is a required argument; the checkpoint's own
state dict is independently loaded and diffed by key name against
`model.backbone.state_dict()` (after stripping the configured `prefix`),
AND the loaded keys' actual tensor VALUES are compared before/after
`init_weights()` to confirm they genuinely changed (catching the case where
key names match but loading silently no-ops). The script now raises loudly
if zero keys match or if every matched key's value is unchanged.
`ENVIRONMENT.md` gained an explicit download step for the checkpoint file
and a required `PRETRAINED_CKPT_PATH` environment variable in the canary
driver.

## Augmentation, item by item (first-round finding, re-confirmed still accurate after the pipeline restructuring above)

| Item | EoMT (`datasets/landmark_dataset.py`) | HRNet (`lib/datasets/fetal.py`, `lib/utils/transforms.py`) | RTMPose (this adapter) |
|---|---|---|---|
| Horizontal flip probability | 0.5 | 0.5 | 0.5 (matched) |
| Rotation range | U(-30, 30) deg, p=0.6 | U(-ROT_FACTOR, ROT_FACTOR), ROT_FACTOR=30, p=0.6 (confirmed identical across every `experiments/fetal/fetal_landmark_hrnet_w18_*.yaml`) | U(-30, 30) deg, p=0.6 (matched) |
| Scale range | U(0.75, 1.25), unconditional | U(1-SCALE_FACTOR, 1+SCALE_FACTOR), SCALE_FACTOR=0.25, unconditional (matched exactly) | U(0.75, 1.25), unconditional (matched) |
| Translation | none | none | none (matched) |
| Interpolation | PIL `Image.BILINEAR` | `cv2.INTER_LINEAR` everywhere | `cv2.INTER_LINEAR` |
| Order of operations | resize to 512 FIRST, then flip/rotate/scale in the fixed canvas | ONE combined affine on the ORIGINAL (un-resized) image; flip via `np.fliplr` beforehand | flip decided/applied in ORIGINAL space first (channel-order correctness requirement, see above), THEN resize to 512, THEN rotate/scale in the resized canvas -- matches EoMT's staged structure more than HRNet's single-combined-affine structure, a disclosed, deliberate choice since EoMT's numbers are already locked |
| Out-of-bounds policy | reject (skip) rotation/scale independently if any landmark would leave the canvas | none needed (fixed-size output region by construction) | reject (skip) rotation/scale independently, same policy as EoMT |
| Colour jitter | brightness/contrast/saturation, U(-0.2,0.2)/U(-0.2,0.2)/U(-0.1,0.1), each p=0.5 | **none** (confirmed via grep, zero matches) | matches EoMT's exact torchvision calls/ranges; explicit BGR<->RGB channel-order handling added since MMCV's `LoadImage` conventionally produces BGR and torchvision's jitter assumes RGB semantics -- flagged for live confirmation |
| Random crop | none | the affine warp itself acts as an implicit crop, no additional step | none |
| RandomHalfBody | never had any | never had any | explicitly removed per supervisor instruction |

**Conclusion unchanged from the first round**: matched on probability/range
for every shared parameter; NOT bitwise-identical (staged vs combined
affine, PIL vs cv2 resampling, EoMT/RTMPose-only colour jitter). State this
precisely in the thesis, never as bitwise-identical augmentation.

## Status after this second-round audit

Not yet run against a live install (still no MMPose environment available
this session) -- what IS newly true, verified with real passing local
tests:

- The flip-order finding is now mathematically correct, not just
  internally consistent, verified both algebraically and by direct
  numerical reproduction of HRNet's real formula (400/400 match).
- `fetal_augment.py`/`transforms.py` implement the corrected design; 7/7
  tests pass (rewritten from the first round's now-invalid test
  assertions).
- The EoMT-vs-HRNet/RTMPose convention gap is explicitly named as an open
  decision for the supervisor, not silently treated as resolved.
- Test-set leakage into validation/checkpoint-selection is fixed with a
  genuine Train-only internal split reusing EoMT's own algorithm.
- `run_inference.py` routes through the model's real preprocessing.
- Final-checkpoint selection is deterministic and fails loudly if the
  training run didn't actually complete `MAX_EPOCHS`.
- Pretrained-weight loading is independently verified by value comparison,
  not assumed from `init_cfg` alone.
- The full non-square + SimCC-codec-level round trip remains BLOCKED on a
  live MMPose install (see `ENVIRONMENT.md`'s checklist item 6) -- the
  pure-Python geometry/reorder tests that DO pass are a strict subset of
  the real pipeline and must not be read as covering codec quantisation
  error.

Next step, unchanged in kind, more precisely scoped now: install MMPose on
the server per `ENVIRONMENT.md`'s full checklist (including the now-6
numbered items, the checkpoint download, and the still-blocked codec
round-trip test), then run `run_rtmpose_canary.sh`, then the mandatory
visual-overlay audit, before sharing canary numbers with the supervisor --
and separately, raise the EoMT-vs-HRNet/RTMPose ordering-convention
decision with the supervisor explicitly, since this session cannot resolve
it alone.
