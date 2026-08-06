# Code-level protocol audit, before canary training

Three review rounds on 2026-08-06. The first round covered the two items
the supervisor explicitly flagged (augmentation-strategy parity, endpoint
ordering/flip convention) and found a real bug, but with an INCORRECT root
cause. A second, independent review of that first round's own work caught
the error with a rigorous mathematical argument, checked directly against
the audited upstream HRNet's actual source, and additionally found four
more blocking implementation issues the first round missed entirely. A
third round then re-checked round 2's own fixes and found one of them
(pretrained-checkpoint provenance) was still incomplete, plus a real,
verifiable architecture/recipe-fidelity gap against the official RTMPose-s
config, plus a documentation-precision issue about when the Test set is
first touched. This document keeps all three rounds' history rather than
silently overwriting it,
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

## Third-round review: round 2's own "fix" was incomplete, plus a verified official-config-fidelity gap

### 1. Round 2's pretrained-checkpoint verification had a real closed-loop gap

Round 2 fixed `record_run_provenance.py` to independently load a LOCAL
checkpoint file and diff its keys/values against the built model. But the
GENERATED CONFIG's own `backbone.init_cfg.checkpoint` was still a
hardcoded URL (`OFFICIAL_CSPNEXT_S_BACKBONE_CHECKPOINT`) -- meaning
`model.init_weights()` actually loads from the URL, completely
independent of whichever local file the provenance script was separately
hashing. Two files could differ in content while having identical key
names/shapes (or the URL could silently redirect/update over time), and
round 2's script would still report success, because it only checked "did
the values change from a fresh random init," not "do the values match
THIS specific audited file."

**Fixed**: `make_config()` now REQUIRES a local `pretrained_checkpoint_path`
argument and embeds that path directly as `backbone.init_cfg.checkpoint`
in the generated config text -- the file `model.init_weights()` loads and
the file `record_run_provenance.py` hashes are now the same file by
construction, not by assumption. `record_run_provenance.py` additionally
now asserts `cfg.model["backbone"]["init_cfg"]["checkpoint"] ==
str(pretrained_checkpoint_path)` explicitly (failing loudly if a hand-edited
config or mismatched CLI argument ever breaks this invariant), and performs
an EXACT per-key VALUE comparison (`torch.allclose`, not just "differs from
random init") between every backbone parameter and the checkpoint's own
tensor for that key -- missing keys, unexpected keys, and value mismatches
are all now individually fatal, not just "zero keys matched at all."

### 2. Generated config was missing several real (verified, not assumed) official-recipe settings

**SUPERSEDED by round 4, see below -- kept for the record, not deleted.**
The reviewer's list of missing official settings was checked against
`https://raw.githubusercontent.com/open-mmlab/mmpose/main/configs/
body_2d_keypoint/rtmpose/coco/rtmpose-s_8xb256-420e_coco-256x192.py`, not
accepted at face value -- one item in the reviewer's list ("gradient
clipping") appeared NOT to be present in that fetched file. **This
correction was itself wrong**: that file path is a stale/divergent copy;
MMPose's actual actively-maintained RTMPose project config lives at a
DIFFERENT path (`projects/rtmpose/rtmpose/body_2d_keypoint/
rtmpose-s_8xb256-420e_coco-256x192.py`) and DOES contain
`clip_grad=dict(max_norm=35, norm_type=2)` -- see round 4.

Verified-real gaps, fixed:
- `backbone._scope_="mmdet"` -- was missing; CSPNeXt is registered under
  mmdet's scope, so omitting this risks a real build-time failure, not a
  style gap.
- `backbone.expand_ratio=0.5` -- was missing; a genuine CSPNeXt
  architecture parameter (not cosmetic) that determines internal channel
  widths -- omitting it risked the backbone's shapes not matching the
  pretrained checkpoint's own shapes (this would now be CAUGHT by fix #1's
  shape-mismatch assertion, but is fixed at the source instead of relying
  on that assertion to catch it).
- `backbone.norm_cfg` was `BN`, official is `SyncBN` -- fixed to match;
  flagged in `ENVIRONMENT.md` as needing a live check that `SyncBN` builds
  correctly on a single-GPU/non-distributed process (some MMEngine/MMCV
  versions require `torch.distributed` to be initialised even for one
  process).
- `optim_wrapper.paramwise_cfg` (zero weight decay on norm/bias) -- was
  missing; added, no dataset-scale-dependent reason to omit it.
- Cosine LR schedule was starting at epoch 0 (overlapping the LinearLR
  warmup entirely); official starts cosine at `max_epochs // 2`. Fixed to
  the same proportional shape at whatever `max_epochs` this project uses.

Verified-real gaps, DELIBERATELY left unfixed with reasons recorded (not
silently dropped, and not blindly copied either):
- `EMAHook` -- this project's own EMA investigation for EoMT already found
  "insufficient clean evidence either way" (thesis Ch5); adding EMA to
  RTMPose alone, without it being part of the shared cross-method recipe,
  would be a new, unreviewed asymmetry.
- `PipelineSwitchHook` (stage-2 augmentation cooldown) -- the official
  version cools down the OFFICIAL `RandomBBoxTransform`'s own ranges; this
  project already replaces that whole augmentation with EoMT/HRNet-matched
  values, so there is no equivalent "official range" to cool down between.
- `auto_scale_lr=dict(base_batch_size=1024)` -- only takes effect via an
  explicit `--auto-scale-lr` CLI flag, and this project's `base_lr` was
  never tuned as a linear-scaling assumption from batch=1024.
- `max_epochs=420` -- kept at 200 (configurable); 420 was tuned for COCO's
  ~118k training images, this project's fetal datasets are 2-3 orders of
  magnitude smaller, so a directly-copied epoch count has no principled
  basis either way.

### 3. "Test touched only once, after training" was imprecise

The canary driver converted the Test CSV to COCO json (opening every Test
image to read its dimensions) BEFORE training started, in the same step as
the internal-train/internal-val conversions. This never leaked any Test
LABEL into training or checkpoint selection, but the README/PROTOCOL_AUDIT
wording ("touched only once, after training") was imprecise about *when*
the Test files were first read on disk.

**Fixed**: `run_rtmpose_canary.sh` restructured so Test CSV/image
conversion happens in the FINAL step, strictly after training and
checkpoint verification are both complete -- nothing under `coco/Test*`
exists on disk until then. Also added an explicit note that `tools/train.py`
conventionally runs only `train`/`val` phases, never `test` (that requires
a separate `tools/test.py` invocation this script never makes), flagged in
`ENVIRONMENT.md` as needing direct confirmation against the actual training
log (no "Testing" phase should appear anywhere in it).

### Draft message for the supervisor (endpoint-ordering convention), as suggested by this round's reviewer

Not sent by this session (no email access) -- drafted here for the user to
send verbatim or edit:

> During implementation, I confirmed that the existing EoMT pipeline
> canonicalises endpoints by per-image x-coordinate sorting, whereas the
> released HRNet implementation uses a training-set-derived direction
> vector. These rules disagree for a non-trivial fraction of samples, so
> RTMPose cannot simultaneously reproduce both training conventions. My
> current implementation follows the HRNet direction-vector convention and
> retains a common fixed-channel evaluator. Would you prefer this, or
> should RTMPose instead follow EoMT's x-sorting convention?

Recommended sequencing: environment setup, model construction, and the
still-blocked synthetic/codec round-trip test (`ENVIRONMENT.md` items 1-8)
can all proceed before this reply arrives. Do not treat a training run
started before the reply as the official canary result to report.

## Status after the third round

Not yet run against a live install (still no MMPose environment available
this session). Newly true, verified locally to the extent possible without
one:

- Pretrained-checkpoint provenance is now a real closed loop (config path
  == audited path == loaded path, checked; exact value match required, not
  "changed from random").
- The generated config's backbone/optimizer/scheduler now match the real,
  fetched official recipe wherever this project isn't deliberately
  diverging for a stated, recorded reason -- no more silent, unreviewed
  gaps between "should match official" and "actually does."
- The Test set is not read on disk at all until training and checkpoint
  verification are both complete.
- A concrete draft message for the supervisor exists for the still-open
  endpoint-ordering-convention question.

Still blocked, unchanged: the full non-square + SimCC-codec-level round
trip (`ENVIRONMENT.md` item 6), and confirmation that `SyncBN` builds on a
single-GPU process (`ENVIRONMENT.md` item 3). Do not report canary NME to
the supervisor until: (a) the environment checklist is fully green, (b)
the visual-overlay audit passes, and (c) ideally, the supervisor's reply on
endpoint-ordering has been received -- environment setup and dry
construction can proceed in parallel with waiting for that reply.

## Fourth-round review: round 3's own official-config fetch was from the wrong path, plus a real LR/scheduler risk and provenance-precision gaps

### 1. Round 3's "gradient clipping is not official" correction was itself wrong

A second reviewer pointed out that MMPose's repository contains TWO
different, independently-maintained files both named
`rtmpose-s_8xb256-420e_coco-256x192.py`:
`configs/body_2d_keypoint/rtmpose/coco/...` (round 3's source, last
touched at commit `a910fd4c5684b0480f561efd703635d817944568`, no
`clip_grad` key) and `projects/rtmpose/rtmpose/body_2d_keypoint/...`
(MMPose's own dedicated, actively-maintained RTMPose project directory,
last touched at commit `94e15226a29a7067d9bb0cb7937b86e3c3fd0c8e`). Fetched
the `projects/rtmpose/` version verbatim -- it DOES contain
`clip_grad=dict(max_norm=35, norm_type=2)`. Round 3's correction of the
original reviewer's claim was itself the error, caught by checking a
second, more authoritative source rather than stopping at the first
fetch that seemed to settle the question.

**Fixed**: `optim_wrapper.clip_grad=dict(max_norm=35, norm_type=2)` added
to the generated config. `ENVIRONMENT.md` now explicitly documents that
`projects/rtmpose/` is this project's chosen authoritative source (with
both paths' commit hashes recorded) so a future session doesn't have to
re-derive this distinction, and doesn't accidentally check the other path
again.

### 2. Real methodological risk: official base_lr=4e-3 used unscaled at a 64x-smaller batch size

The official recipe's `base_lr=4e-3` is explicitly paired with
`auto_scale_lr=dict(base_batch_size=1024)` (8 GPUs x 256). This project's
generated config used `lr=4e-3` directly at `batch_size=16` -- a 64x
smaller batch with no corresponding LR reduction, which is not a neutral
scope decision but a real risk of training instability (too-large an
effective step size per gradient update relative to the batch's own noise
level). The previous framing ("this project's base_lr was never tuned as
a linear-scaling assumption from batch=1024") accurately described the
gap but incorrectly treated it as low-risk.

**Fixed**: `base_lr` is now computed as `4e-3 * (batch_size / 1024)` inside
`make_config()`, giving `6.25e-5` at this project's `batch_size=16`,
applied explicitly and once. `auto_scale_lr` is deliberately NOT also
added to the generated config -- if `--auto-scale-lr` were ever passed to
`tools/train.py` on top of an already-scaled `base_lr`, the LR would be
scaled twice. The scaled value is recorded directly in the generated
config's own comment, not left implicit.

### 3. Real scheduler-overlap risk on small datasets

The official recipe's `LinearLR` warmup is fixed at `end=1000` iterations
regardless of dataset size, and `CosineAnnealingLR` begins at
`max_epochs // 2` in EPOCH units (converted to iterations by MMEngine at
runtime using the actual dataloader length). For COCO (~118k train images,
~460 iterations/epoch at official batch_size=256), 1000 iterations is a
small fraction of the ~96,600 iterations at which cosine begins -- no
overlap. For this project's UCL BPD internal-train split (100 images,
`batch_size=16`, ~7 iterations/epoch), cosine begins at epoch 100 = ~700
iterations, LESS than the fixed 1000-iteration warmup -- the two schedulers
would genuinely overlap, which a naive "keep the official recipe's shape
proportionally" fix (round 3's own `max_epochs // 2` change) did not
address, since it only touched the EPOCH-based cosine begin, not the
iteration-based warmup end.

**Fixed**: `make_config()` now reads the ACTUAL internal-train COCO json's
image count (produced by `convert_csv_to_coco.py` before `make_config.py`
runs in `run_rtmpose_canary.sh`), computes the real `iterations_per_epoch`,
and sets `warmup_end_iters = min(1000, cosine_begin_iters // 2)`. The
generated config embeds an explicit `assert warmup_end_iters <
cosine_begin_epoch * iters_per_epoch` that would run at config-load time
(before any GPU work), and `make_config()` itself ALSO raises loudly at
generation time if the computed values would violate this -- two
independent points where an overlapping schedule cannot silently pass
through. Verified by generating a config against a realistic 100-image
internal-train json (`iters_per_epoch=7`, `cosine_begin_iters=700`,
`warmup_end_iters=350`): `make_config()`'s OWN Python-level check (which
runs for real, immediately, every time the config is generated) did not
raise.

**CORRECTION (round 6, self-caught overclaim, not a reviewer finding)**:
this section previously also claimed "the generated config was executed
directly ... and the assertion passed before hitting the expected
cv2/mmcv import boundary" -- this was WRONG. The `import transforms` line
sits NEAR THE TOP of the generated config template (before the
`warmup_end_iters`/`training_recipe_summary` definitions, which come much
later), so every local execution of the generated `.py` file has always
failed at that early import, LONG before reaching the embedded assert --
the assert's line was never actually run in any local test. The embedded
assert is real, correctly placed relative to the values it checks, and
will genuinely execute once MMPose/cv2 are installed (at which point
`live_preflight.py`'s own `check_provenance_fields_present` independently
re-verifies the same relationship using `training_recipe_summary`'s own
values) -- but claiming this session had already "executed and passed" it
was inaccurate, and is corrected here rather than left standing.

### 4. Provenance-verification precision gaps (round 3's own claims didn't fully match its own code)

Three real gaps found by re-reading round 3's `record_run_provenance.py`
against what it actually does, not just what its comments claimed:
- Documented as "EXACT" value comparison, but implemented with
  `torch.allclose(..., atol=1e-6)` -- a numerical-tolerance comparison, not
  a true exact match. Since the config-path assertion guarantees the
  loaded file and the audited file are literally the same file, the
  correct comparison is bit-exact. **Fixed**: switched to `torch.equal()`,
  with the checkpoint tensor cast to the model parameter's own dtype first
  (a lossless upcast for e.g. a stored-fp16 checkpoint into an fp32 model,
  not a lossy rounding).
- `unexpected_in_checkpoint` was computed and included in the output JSON,
  but never actually raised as fatal -- the surrounding documentation's
  claim that "any extra key would be caught" did not match the code.
  **Fixed**: now genuinely fatal.
- The "unchanged from random init" check iterated the ENTIRE
  `state_dict()`, including BN `running_mean`/`running_var`/
  `num_batches_tracked` buffers, some of which can legitimately be
  identical between a fresh init and a real checkpoint (e.g.
  `num_batches_tracked=0` right after construction) -- a false-failure
  risk unrelated to whether loading actually worked. **Fixed**: removed
  entirely, since the exact-value-vs-checkpoint check already fully
  verifies correctness on its own without this weaker, buffer-confounded
  secondary check.
- Also fixed (a smaller, defensive improvement, not a bug report):
  `pretrained_checkpoint_path` is now `.resolve()`d to an absolute path
  before being embedded in the generated config, so the config is not
  sensitive to the working directory it happened to be generated from.

### 5. EMA framing corrected

Round 3 justified not adding `EMAHook` by citing this project's OWN EMA
investigation for EoMT ("insufficient clean evidence either way"). A
reviewer correctly pointed out this is not valid evidence for RTMPose --
EoMT and RTMPose are structurally different models with different training
setups; a finding about EMA's effect on EoMT says nothing about whether
RTMPose specifically benefits from EMA (which IS part of RTMPose's own
official recipe, unlike stage-2 pipeline switching or `auto_scale_lr`,
which have direct substitutes or dependencies already addressed elsewhere
in this project). **Corrected framing**: not using EMA for the canary and
initial runs is a SCOPE decision (fewer moving parts while validating the
adapter end-to-end), not a claim that RTMPose doesn't need EMA. Any writeup
of these results must describe the model as "RTMPose-s architecture
trained under this project's common fetal training protocol," never as
"the official RTMPose-s recipe" or "an RTMPose-s reproduction" -- EMA, the
stage-2 augmentation switch, and the native RTMPose augmentation are all
official-recipe components this project deliberately does not use. Revisit
EMA with an actual seed-42 raw-vs-EMA diagnostic on this project's own data
if the canary's results motivate it; do not decide this from EoMT's
unrelated result.

## Status after the fourth round

Not yet run against a live install (still no MMPose environment available
this session). Newly true:

- The official-recipe source is now pinned to a specific file path AND
  commit (`projects/rtmpose/rtmpose/body_2d_keypoint/
  rtmpose-s_8xb256-420e_coco-256x192.py` @ `94e15226a29a7067d9bb0cb7937b86e3c3fd0c8e`),
  not re-derived from a floating `main` checkout or an unverified second
  path every time the question comes up.
- `clip_grad` matches the real official setting.
- The learning rate is explicitly, correctly scaled for this project's
  actual batch size, not copied unscaled from a 64x-larger official batch.
- The warmup/cosine schedule cannot silently overlap for small datasets --
  computed from the real image count, asserted at both generation time and
  config-load time.
- Provenance verification is now genuinely exact (not tolerance-based) and
  genuinely enforces every claimed check (unexpected keys included).
- EMA's omission is now framed as a scope decision, not (mis)supported by
  an unrelated EoMT finding -- and any results writeup must name the model
  precisely ("architecture under this project's protocol," not "official
  reproduction").

Unchanged, still blocking: install MMPose, confirm `SyncBN` builds
single-GPU, run the full non-square + SimCC-codec round trip, resolve the
EoMT-vs-HRNet/RTMPose endpoint-ordering question with the supervisor.

## Fifth-round review: warmup still too long, a real crash/wrong-number risk in periodic validation, and no enforced live gate

A fifth reviewer confirmed rounds 1-4's fixes were correctly landed, then
found three more real issues: the round-4 warmup fix only avoided overlap,
not underfitting; periodic internal validation could crash or silently
produce a meaningless number; and the checklist in `ENVIRONMENT.md` was
still purely documentation, never actually enforced by the canary script.

### 1. Warmup no longer overlaps cosine, but is now far too long

Round 4's `warmup_end_iters = min(1000, cosine_begin_iters // 2)` gave
`350` iterations of warmup out of BPD's ~1400 total training iterations --
**25% of all training spent ramping up from a near-zero learning rate**,
with no training-methodology basis for that specific fraction (it was
purely "half of whatever avoids overlap"), a real underfitting risk.

**Fixed**: warmup is now a short, fixed number of EPOCHS
(`warmup_epochs = min(5, max(1, max_epochs // 20))`, i.e. at most 5 epochs)
converted to iterations using the real image count. For BPD's numbers this
gives `warmup_epochs=5`, `warmup_end_iters=35` (2.5% of total training),
much closer to the official recipe's own proportion (1000 of ~193,200
total iterations at official scale, ~0.5%) than either the original fixed
1000 or round 4's `cosine_begin_iters // 2` ever were. Still asserts
`warmup_end_iters < cosine_begin_iters`. Also added: `training_recipe_summary`
as a real top-level config value (`n_train_images`, `batch_size`,
`iters_per_epoch`, `effective_lr`, `warmup_end_iters`, `cosine_begin_epoch`,
`max_epochs`), read back verbatim by `record_run_provenance.py` into its
output JSON, so a canary report can state the exact training setup used
without a reader re-deriving it from `make_config.py`'s source.

### 2. Periodic internal validation depended on an unverified, possibly-crashing pathway

The generated config's periodic validation used MMEngine's default val
loop (`model.val_step()` -> `model.predict()`) with `PCKAccuracy` as
`val_evaluator`. For a stock `TopdownPoseEstimator`, `predict()` typically
needs `bbox_center`/`bbox_scale` metadata that `GetBBoxCenterScale`/
`TopdownAffine` would normally populate -- this project's own
`PixelCentreResize` deliberately never sets that metadata (the exact same
reason `run_inference.py` avoids `predict()` for final inference, per that
file's own long-standing docstring). Whether the default val loop would
crash outright, silently use some default bbox and produce a meaningless
number, or something else, was never actually addressed for the TRAINING-
time periodic case -- only for final inference.

**Fixed**: `low_level_decode.py` (new) factors `run_inference.py`'s
verified-safe decode path (data_preprocessor -> extract_feat ->
head.forward -> codec decode -> `geometry.to_image_space()`, bypassing
`predict()` entirely) into a shared module. `internal_val_hook.py` (new)
is a custom MMEngine `Hook` that uses this SAME path to compute the SAME
fixed-channel NME formula as the final, authoritative evaluation, logged
every `val_interval` epochs via `runner.logger`/`runner.message_hub`.
`make_config.py`'s generated config now sets `val_cfg=None` (disabling
MMEngine's automatic val loop and `model.predict()` entirely for training)
and adds `InternalFixedChannelNMEHook` to `custom_hooks`. This is a
genuinely stronger design, not just a risk-avoidance workaround: internal
monitoring numbers are now DIRECTLY comparable to the final authoritative
NME (identical formula, identical code path), which `PCKAccuracy`'s
different OKS/bbox-normalised metric never was anyway.
`run_inference.py` was refactored to call the same shared
`low_level_decode.decode_batch_low_level`/`to_original_image_space`
functions, so there are no longer two independently-maintained copies of
this logic that could silently diverge. `test_low_level_decode.py` (new,
3/3 pass) covers the pure-Python parts (`to_original_image_space`,
`fixed_channel_nme`); `decode_batch_low_level` itself still needs a live
model, same tier as the rest of this code.

### 3. ENVIRONMENT.md's checklist was documentation only, never enforced

Nothing in `run_rtmpose_canary.sh` actually forced the live-verification
checklist to run before training started -- a plain execution of the
script would go straight from provenance recording into 200 epochs of
training, silently skipping every documented gate.

**Fixed**: `live_preflight.py` (new) is a hard-fail gate `run_rtmpose_canary.sh`
now calls immediately before training (step 4c) and treats as fatal on any
failure. It runs, against the real installed MMPose: (1) the full
non-square + SimCC-codec round trip (ENVIRONMENT.md item 6, now automated
rather than a manual instruction); (2) one real train-batch forward + loss
+ backward pass, asserting all loss components are finite and gradients
reach both the backbone and the head; (3) one real call to
`decode_batch_low_level` plus the `to_original_image_space`/
`fixed_channel_nme` chain, asserting shapes and finite values; (4) that
`training_recipe_summary`'s own numbers are internally consistent (no
warmup/cosine overlap). Its own `encode()`/`decode()` dict-key assumptions
(`keypoint_x_labels`/`keypoint_y_labels`, inferred from `SimCCLabel`'s
documented interface) are explicitly flagged as needing live confirmation
-- if this specific step is what fails first on the server, that is this
gate doing its job, not a defect in the gate itself.

### Precision fix: "bit-exact" wording in `record_run_provenance.py`

A smaller fifth-round finding: the file's own comments described the
value check as verifying the checkpoint file's "raw bytes" are identical
to the model's parameters. The actual, correct claim (unchanged code,
corrected wording) is narrower: the checkpoint's tensor, AFTER being cast
to the dtype the model actually loaded it as, is bit-for-bit identical to
the model's own parameter -- not a claim about the original file's raw
on-disk bytes, since a legitimate dtype upcast (e.g. stored fp16 into an
fp32 model) changes the in-memory representation without losing
information. Fixed in both the module docstring and the inline comment.

## Status after the fifth round

Not yet run against a live install (still no MMPose environment available
this session). Newly true:

- The LinearLR warmup is short and methodologically motivated (~2.5% of
  total training for BPD), not just "whatever avoids overlap."
- Periodic internal validation cannot depend on `model.predict()`'s
  unverified bbox-metadata contract at all -- it uses the same low-level,
  verified-safe path as final inference, producing directly comparable
  numbers.
- The full live-verification checklist has an actual enforcement
  mechanism (`live_preflight.py`), not just documentation a user could
  skip by running the canary script normally.
- `record_run_provenance.py`'s own claims about what it verifies are now
  precisely worded.

## Sixth-round review: the preflight script written to catch config bugs had its own real bugs

A sixth reviewer confirmed the warmup fix was reasonable, then found four
real problems IN `live_preflight.py`/`make_config.py`'s round-5 val-loop
design itself -- the tool built to catch mistakes needed the same scrutiny
as everything else.

### 1. `val_cfg=None` alongside populated `val_dataloader`/`val_evaluator` -- VERIFIED against the real MMEngine Runner source to be a genuine, immediate crash

Round 5's design set `val_cfg=None` but left `val_dataloader`/`val_evaluator`
as real dicts (so `InternalFixedChannelNMEHook` could read
`runner.cfg.val_dataloader`). Checked directly against MMEngine's actual
`Runner.__init__` source: it contains an explicit check --
`val_related = [val_dataloader, val_cfg, val_evaluator]`; if these are not
either all `None` or all not-`None`, it raises `ValueError` immediately.
This is not a hypothetical risk; Runner construction itself would have
failed before a single training step ran.

**Fixed**: the Train-only internal validation dataloader now lives under a
non-standard config key, `internal_val_dataloader`, which `Runner.from_cfg()`
never reads at all (confirmed against the same source: `from_cfg` only
ever reads `cfg.get('val_dataloader')`, not arbitrary keys), so it cannot
participate in the all-or-nothing check. `val_dataloader` and
`val_evaluator` are both explicitly `None` alongside `val_cfg=None`.
`internal_val_hook.py` and `live_preflight.py` both updated to read
`cfg.internal_val_dataloader` instead of the now-`None` `cfg.val_dataloader`.
`test_dataloader`/`test_cfg`/`test_evaluator` remain a consistent
non-`None` trio (required both by the same Runner constraint applied to
the test-side triple, and because `run_inference.py` reads
`cfg.test_dataloader["dataset"]` directly) -- with an explicit, loud
`*** DO NOT RUN tools/test.py AGAINST THIS CONFIG ***` comment at
`test_evaluator`'s own definition, since `PCKAccuracy` there has the
identical unverified-bbox-metadata risk that motivated replacing the val
loop, and nothing about keeping the trio consistent makes it safe to
actually invoke.

### 2. `custom_hooks`' registration relied on a bare `import`, no `custom_imports` fallback

The generated config's `import internal_val_hook` statement (matching the
pre-existing `import transforms` pattern used since round 1) is very
likely sufficient -- MMEngine's `Config.fromfile()` genuinely imports a
Python-format config as a real module with real import side effects, not
a restricted AST-only extraction. Re-checking the actual committed file at
the reviewed commit confirmed the import statement WAS already present
(the reviewer's specific claim of a missing import did not match the
committed code) -- but added MMEngine's own officially-documented
`custom_imports = dict(imports=[...], allow_failed_imports=False)`
mechanism as a belt-and-suspenders safeguard regardless, since it is the
more version-robust of the two mechanisms, and because `live_preflight.py`
now genuinely registry-builds the Hook (`HOOKS.build()`, see item 4 below)
rather than assuming registration worked.

### 3. The preflight's own geometric round-trip test used the AUGMENTED train pipeline, not a deterministic one

`check_geometric_round_trip` built its dataset from `cfg.train_dataloader["dataset"]`,
whose pipeline includes `FetalRandomFlipAndCanonicalize`/
`FetalRotateScaleColorJitter` (random flip/rotation/scale/colour jitter) --
then compared the decoded coordinates against the ORIGINAL, un-augmented
synthetic `p0`/`p1`. The function's own comments claimed a "deterministic
val_pipeline," directly contradicting what the code actually did; the test
would fail unpredictably whenever augmentation happened to trigger, and
even when it passed, the result would not actually isolate SimCC
quantisation error the way the module docstring claimed.

**Fixed**: both `check_geometric_round_trip` and `check_decode_path` (renamed
from `check_decode_and_internal_val_path`) now build from
`cfg.internal_val_dataloader["dataset"]` -- `LoadImage -> PixelCentreResize
-> PackPoseInputs` only, no augmentation stage at all, matching what the
comments always claimed.

### 4. The preflight never actually built or invoked the Hook -- and never used pretrained weights

The original `live_preflight.py` only exercised the shared
`low_level_decode` functions the Hook depends on, never `HOOKS.build()`
itself or the Hook's own `after_train_epoch` lifecycle method -- meaning
neither of the two bugs above (missing registration, wrong dataloader
reference) would actually have been CAUGHT by this preflight, only
theoretically avoided by inspection. Separately, `check_train_forward_backward`
built the model without calling `model.init_weights()`, so the "real"
smoke test forward/backward pass actually ran on a randomly-initialised
backbone, not the pretrained one the canary will use.

**Fixed**: new `check_hook_registry_and_lifecycle` genuinely builds
`InternalFixedChannelNMEHook` via `HOOKS.build()`, asserts the returned
class name, constructs a minimal duck-typed fake Runner (exposing exactly
`model`/`cfg`/`epoch`/`logger`/a REAL `mmengine.logging.MessageHub`
instance, not a full `mmengine.Runner` which would need a complete working
optimizer/scheduler just to construct), and calls `after_train_epoch` once
-- asserting the logged NME is finite, the message hub actually received
the scalar, and `model.training` is correctly restored afterward.
`check_train_forward_backward` and `check_decode_path` both now call
`model.init_weights()` before use. Also added `check_bgr_rgb_channel_order`
(previously claimed as covered, never actually implemented): writes a
synthetic image with a KNOWN top-left pixel value and compares what the
pipeline's own array shows against `FetalRotateScaleColorJitter`'s
`assume_bgr=True` default, failing loudly if they disagree. `internal_val_hook.py`'s
hardcoded `device="cuda:0"` default was also changed to `None`, resolved
dynamically via `next(model.parameters()).device` at call time, so it does
not silently break under a future non-`cuda:0` device.

### Self-caught correction (not a reviewer finding): an earlier "verified by execution" claim was wrong

While fixing the above, re-checked this document's own round-4 claim that
"the generated config was executed directly ... and the assertion passed
before hitting the expected cv2/mmcv import boundary" for the warmup/cosine
assert. This was WRONG: `import transforms` sits near the top of the
generated config template, well BEFORE the `warmup_end_iters`/
`training_recipe_summary` definitions -- every local execution of the
generated file has always failed at that early import, long before
reaching the embedded assert. The assert is real and correctly placed, and
`make_config()`'s OWN Python-level check (which runs immediately, for
real, every time a config is generated, independent of the template
string) DID genuinely execute and pass -- but the specific claim about the
EMBEDDED assert having been executed was inaccurate and is corrected in
round 4's own section above rather than left standing. Worth naming
explicitly: this session's own verification claims need the same scrutiny
applied to reviewers' claims, not just when a reviewer happens to catch it.

### Added: `PREFLIGHT_ONLY` safety flag

Per this round's explicit request: `run_rtmpose_canary.sh` now defaults to
`PREFLIGHT_ONLY=1`, stopping immediately after `live_preflight.py` passes
and printing a clear message rather than continuing into the 200-epoch
canary in the same invocation. Set `PREFLIGHT_ONLY=0` explicitly to
actually start training -- intended workflow: run preflight on the server
tonight, review its output, get the supervisor's endpoint-ordering reply,
THEN re-run with `PREFLIGHT_ONLY=0` the next morning.

## Status after the sixth round

Not yet run against a live install (still no MMPose environment available
this session). Newly true:

- The val-loop redesign (round 5) is now actually constructible by
  MMEngine's Runner -- verified against the real constraint in Runner's
  own source, not assumed.
- `live_preflight.py` can actually catch the two structural bugs above
  (it now builds the Hook for real and calls its real lifecycle method),
  not just the shared functions underneath them.
- The geometric round-trip test is deterministic, matching what it always
  claimed to be.
- BGR/RGB channel order has a real, automated check instead of a claimed-
  but-missing one.
- The forward/backward and decode smoke tests use the pretrained backbone,
  not a randomly-initialised one.
- `PREFLIGHT_ONLY=1` is the default, so running this script cannot
  accidentally start the 200-epoch canary before the supervisor's
  endpoint-ordering reply arrives.

Given six review rounds have each found real, distinct issues -- three of
them (rounds 2, 4, 5) in code that was ITSELF written to fix or verify a
previous round's finding -- the honest status remains: still not run
against live MMPose. The next real step is unchanged: run
`live_preflight.py` for real on the server tonight (`PREFLIGHT_ONLY=1`,
the default), fix whatever it finds (very likely at least one of its own
flagged live-only assumptions, e.g. the SimCC codec's exact
`encode()`/`decode()` dict keys), and send the already-drafted
endpoint-ordering question to the supervisor now -- that question cannot
be resolved by further code review at all, and should not wait for a
seventh round.
