# Code-level protocol audit, before canary training (2026-08-06)

Requested explicitly before starting the UCL BPD/seed-42 canary, given the
supervisor's approval email listed two items that needed verifying against
actual code rather than assumed identical by name: (1) "same augmentation
strategy" across EoMT/HRNet/RTMPose, itemised; (2) endpoint ordering and
flip convention, verified through code, not just a `flip_indices` setting.
Both are covered below, against the real source of all three methods (this
repo's `datasets/landmark_dataset.py` and `ablation/ensemble_test.py` for
EoMT; the audited upstream `lib/datasets/fetal.py`/`lib/utils/transforms.py`
for HRNet; this adapter's own code for RTMPose), not from memory or
docstrings alone.

## 1. Augmentation, item by item

| Item | EoMT (`datasets/landmark_dataset.py`) | HRNet (`lib/datasets/fetal.py`, `lib/utils/transforms.py`) | RTMPose (this adapter, after today's fix) |
|---|---|---|---|
| Horizontal flip probability | 0.5 | 0.5 | 0.5 (matched) |
| Rotation range | U(-30, 30) deg, applied at p=0.6 | U(-ROT_FACTOR, ROT_FACTOR), ROT_FACTOR=30, p=0.6 (confirmed identical across every `experiments/fetal/fetal_landmark_hrnet_w18_*.yaml`) | U(-30, 30) deg, p=0.6 (matched) |
| Scale range | U(0.75, 1.25), unconditional (no probability gate) | U(1-SCALE_FACTOR, 1+SCALE_FACTOR), SCALE_FACTOR=0.25 -> U(0.75,1.25), unconditional (matched exactly) | U(0.75, 1.25), unconditional (matched) |
| Translation | none | none | none (matched: none of the three do random translation) |
| Interpolation | PIL `Image.BILINEAR` (resize, rotate, and the scale-zoom's resize+paste) | `cv2.INTER_LINEAR` everywhere (`crop_v2`/`crop`) | `cv2.INTER_LINEAR` (PixelCentreResize + FetalTrainAugment's warpAffine calls) |
| Order of operations | resize to 512 FIRST, then flip, then rotate (in-place, own centre), then scale (in-place zoom), then DOD x-sort, then color jitter | ONE combined affine (`crop()`, built from center/scale/rot) applied to the ORIGINAL (un-resized) image in a single warp; flip happens separately beforehand via `np.fliplr` on the raw array | resize to 512 first (`PixelCentreResize`), then flip, then rotate (separate warpAffine), then scale (separate warpAffine), then DOD reorder, then color jitter -- **matches EoMT's staged order, not HRNet's single-combined-affine order** (deliberate: EoMT's own numbers are already locked/non-retrainable, so RTMPose was built to match EoMT's mechanics where the two diverge from HRNet) |
| Out-of-bounds policy | reject (skip) rotation/scale independently if any landmark would leave the canvas; never clamp | none needed -- HRNet's crop/warp always samples a fixed-size output region by construction, points can't leave the canvas | reject (skip) rotation/scale independently, same policy as EoMT (`fetal_augment.sequential_train_augment`) |
| Colour jitter | brightness/contrast/saturation, U(-0.2,0.2)/U(-0.2,0.2)/U(-0.1,0.1), each p=0.5, via `torchvision.transforms.functional` | **none** (confirmed via `grep -n "bright\|contrast\|satur\|hue\|jitter" lib/datasets/fetal.py` -> zero matches) | matches EoMT exactly (same torchvision calls, same ranges/probabilities) -- **this means RTMPose's colour jitter matches EoMT but NOT HRNet; HRNet has no colour augmentation to match** |
| Random crop | none (whole image, no separate crop step) | the affine warp itself acts as a crop around `center`/`scale`, but there is no ADDITIONAL random-crop step beyond that single warp | none (matches EoMT structurally: no separate crop step) |
| RandomHalfBody / person-specific transforms | never had any (not a person-keypoint task) | never had any | explicitly removed from the RTMPose config per the supervisor's instruction (not present in `make_config.py`'s train_pipeline) |

**Conclusion**: augmentation is now matched on probability/range for every
parameter that exists in more than one method (flip, rotation, scale), and
RTMPose additionally now HAS rotation+scale (previously deferred, see
Section 3). It is **not bitwise-identical** across all three implementations
-- the disclosed, real differences are: (a) HRNet performs one combined
affine warp on the un-resized image where EoMT/RTMPose perform staged
operations on an already-512-resized canvas (a pre-existing EoMT-vs-HRNet
difference, not something introduced today); (b) PIL vs cv2 bilinear
resampling kernels are not guaranteed bit-identical; (c) colour jitter
exists in EoMT and RTMPose but not in HRNet at all. None of this should be
described in the thesis as bitwise-identical augmentation; it should be
described as "the same augmentation family and parameter ranges, with the
disclosed mechanical differences above."

## 2. Endpoint ordering / flip convention -- verified through code, not assumed

### 2a. What each method's channel-0/channel-1 rule actually is

- **EoMT** (`datasets/landmark_dataset.py` line 274-277, `ablation/ensemble_test.py`'s `dod_sort`): a **per-sample ascending-x sort**, recomputed fresh at every access, AFTER all geometric augmentation. This is equivalent to projecting onto a fixed PURELY HORIZONTAL direction. It is not a learned direction and has no memory across samples.
- **HRNet** (`lib/datasets/fetal.py` lines 249-289, confirmed via direct source read): a **learned direction vector `d_vect`** (two prototype points, fit once per (dataset, task) via a Gaussian mixture on the training CSV, `random_state=0`, frozen thereafter), re-projected through that sample's own center/scale/rotation transform every `__getitem__` call, tie-safe (`proj0 <= proj1` keeps order).
- **RTMPose (this adapter)**: reuses HRNet's own frozen `d_vect`, extracted directly from real trained checkpoints (`dod_vectors.py`), verified against HRNet's own real per-image test output (`test_endpoint_order.py`). This is a deliberate choice matching `PROTOCOL_LOCKED.md`'s explicit instruction ("audited against the project's existing training-derived DOD/canonicalisation convention used for the matched HRNet fixed-channel results") -- i.e. **RTMPose is built to match HRNet's convention, not EoMT's.**

### 2b. EoMT and HRNet do NOT use the same rule -- quantified, not assumed

Since EoMT's rule (pure horizontal-axis projection) and HRNet's rule
(learned, task-specific direction) are mathematically different rules, they
necessarily disagree on some images. This was measured directly (not
theorised) using the real released UCL Train CSVs and the real frozen
`d_vect` values:

| Dataset | Task | d_vect dominant axis | EoMT-x-sort vs HRNet-DOD disagreement (pre-flip) |
|---|---|---|---:|
| UCL | BPD | near-vertical | 50/110 = 45.5% |
| UCL | OFD | near-horizontal | 110/110 = 100.0% |
| UCL | APAD | near-horizontal | 0/94 = 0.0% |
| UCL | TAD | near-vertical | 40/94 = 42.6% |
| UCL | FL | near-horizontal | 0/96 = 0.0% |
| Multicentre | BPD | near-vertical | 767/1588 = 48.3% |
| Multicentre | TAD | near-vertical | 275/662 = 41.5% |

(Full script: `audit_flip_order_stability.py`; run against
`annotations/{UCL,MULTICENTRE}/{Head,Abdomen,Femur}_Train.csv`.)

This is a **direct, large, real confirmation of the mechanism already
hypothesised** in the thesis's discussion of the EoMT-vs-HRNet
"correspondence gap" (Chapter 6, `sec:discussion-limitations-endpoint-canon`)
-- e.g. BPD's own `d_vect` is `((545.26, 125.9), (549.10, 562.26))`, i.e.
almost perfectly VERTICAL (x barely changes, y changes by 436px), so a
purely-horizontal x-sort is close to arbitrary/noise-sensitive for BPD
specifically, exactly matching the previously-measured large
correspondence-gap finding for BPD/TAD in `final_comparison/analyse_*.py`.

**Consequence for this project**: RTMPose will be internally consistent
with HRNet's convention (verified byte-for-byte against real HRNet output)
but will inherit the SAME pre-existing disagreement against EoMT's own
convention that HRNet already has. This is not a new problem introduced by
RTMPose and is not fixable without either (a) retraining EoMT under a
different channel convention (out of scope, EoMT's results are already
locked/reported) or (b) reprocessing EoMT's own evaluation to use HRNet's
`d_vect` post hoc (would require re-deriving which of EoMT's two trained
output channels corresponds to which physical endpoint per image, which
EoMT's architecture does not support without retraining). **State this
explicitly in the RTMPose results section**: RTMPose and HRNet share an
identical, code-verified endpoint-ordering convention; EoMT uses a
different, pre-existing convention of its own, already disclosed as a
limitation elsewhere in the thesis.

### 2c. A real bug found and fixed: flip-consistency was NOT "just set flip_indices and assume"

The original adapter design (before today) set MMPose's `flip_indices=[0,
1]` (no channel swap on flip) once, reasoning by analogy to HRNet's own
`_flip_x_only` (which also doesn't swap indices on flip). **This reasoning
was incomplete**: HRNet doesn't need to swap on flip because it RE-DERIVES
the DOD projection fresh after every augmentation draw (including flip);
this adapter's original design instead FROZE the canonical order once at
CSV-conversion time and relied on the static `flip_indices` setting alone
for every subsequent flip during training.

Measuring this directly (`audit_flip_order_stability.py`, comparing the
frozen order against what a fresh DOD projection would say post-flip):

| Dataset | Task | d_vect dominant axis | Fraction of training images where flip breaks the frozen order |
|---|---|---|---:|
| UCL | BPD | near-vertical | 0/110 = 0.0% |
| UCL | OFD | near-horizontal | 110/110 = **100.0%** |
| UCL | APAD | near-horizontal | 94/94 = **100.0%** |
| UCL | TAD | near-vertical | 0/94 = 0.0% |
| UCL | FL | near-horizontal | 96/96 = **100.0%** |
| Multicentre | BPD | near-vertical | 3/1588 = 0.2% |
| Multicentre | TAD | near-vertical | 1/662 = 0.2% |

The pattern is exactly what the geometry predicts: a static "never swap on
flip" rule is correct only when `d_vect` is close to vertical (flipping x
doesn't reverse a near-vertical projection's order) and is wrong for nearly
every sample when `d_vect` is close to horizontal (flipping x reverses a
near-horizontal projection's order almost every time). **This is a
near-total training-label-corruption bug for 3 of 5 tasks (OFD, APAD, FL),
not a rare edge case** -- though it happens to be a non-issue for the BPD
canary specifically (0.0%/0.2% measured).

**Fix applied** (`fetal_augment.py`, `transforms.py`'s new
`FetalTrainAugment`, replacing the stock `RandomFlip` + static
`flip_indices` entirely): the frozen `d_vect` prototype points are now
carried through the EXACT SAME flip/rotation operations as the sample's own
keypoints, and the final channel order is re-derived by projecting onto the
correspondingly-transformed direction -- mirroring HRNet's own
per-sample-transformed-direction architecture (`fetal.py` lines 249-289)
instead of a static rule. This is provable to be correct in general (not
just for the measured cases): a positive uniform scale about a fixed centre
provably cannot change the projection order (proof in `fetal_augment.py`'s
docstring); a consistent flip or rotation applied to BOTH the points and
the direction provably preserves the projection order relative to the
unaugmented baseline (algebraic proof in the same file, verified empirically
by a 500-case randomised property test, `test_fetal_augment.py`). Rotation
and scale retain EoMT's own independent per-stage accept/reject-if-
out-of-canvas policy (a rejected rotation does not discard an already-
accepted flip or a later accepted scale). 7/7 new unit tests pass, including
a direct regression test reproducing the specific UCL OFD failure case
found above and confirming it is now handled correctly.

**Scope note, still disclosed, not fixed today**: `sequential_train_augment`
re-derives the order fresh from ORIGINAL-image-space `d_vect` and the
current sample's own draw every time it runs (i.e. every training epoch,
per-sample) -- this is training-time only. The CSV-conversion-time
`canonical_order()` call (`convert_csv_to_coco.py`, used to write the
COCO-format ground truth and thus also the GT used for the reported
fixed-channel/swap-min NME) still applies the DOD projection exactly ONCE,
on the un-augmented original coordinates -- which is the correct, intended
behaviour (it mirrors HRNet's own TEST-time behaviour exactly, `is_train=
False` never augments, and this is what `test_endpoint_order.py` verifies
against real HRNet test output). Today's fix only concerns TRAINING-time
label consistency; it does not change what ground truth is used for
reported metrics.

## 3. Rotation/scale augmentation: no longer deferred

`make_config.py`'s previous version deliberately shipped RTMPose with flip
only, explicitly deferring rotation/scale as a documented scope decision
"for the canary and initial runs." Given today's audit needed to resolve
the augmentation-parity question precisely anyway, rotation (`ROT_FACTOR=
30`, p=0.6) and scale (`SCALE_FACTOR=0.25`, unconditional) have now been
added (`transforms.FetalTrainAugment`, wired into `make_config.py`'s
`train_pipeline`), matching EoMT/HRNet's actual parameters exactly (Section
1's table). This closes the augmentation-parity gap before any training
happens, rather than leaving it for a later, harder-to-notice fix once
OFD/APAD/FL checkpoints already existed under the old (flip-only, and
separately, flip-broken) design.

## 4. Pretrained-weight provenance and parameter counts: now a required, recorded artifact

`PROTOCOL_LOCKED.md`'s "Required outputs" list did not previously name
parameter counts or a structured provenance record as their own artifact
(only "environment audit"). `record_run_provenance.py` (new) is now wired
into `run_rtmpose_canary.sh` as step 3b, run right after config generation,
and records, as a JSON file saved alongside the canary's other outputs:

- official base config name and the generated config's own path;
- pretrained checkpoint URL, load prefix (`backbone.`), and local SHA-256
  (if the file is available locally to hash);
- source dataset (COCO + AI Challenger, from the official checkpoint's own
  filename) and pretraining task (256x192 17-keypoint human pose --
  explicitly NOT this project's own task);
- backbone vs head parameter counts and a sample of which keys belong to
  each (confirming the head is freshly initialised, not loaded);
- **actual** total/trainable/frozen parameter counts for THIS project's
  real out_channels=2, 512x512 config, measured by building the model, not
  assumed;
- an explicit note not to cite the official RTMPose-s paper's ~5.47M-
  parameter figure (a different config: COCO 256x192, 17 keypoints) as this
  project's own number.

This has not been run against a live install (needs MMPose, same tier as
`run_inference.py`); it is syntax-checked only (`py_compile`) in this
session, same disclosure level as every other MMPose-dependent file.

## Status after this audit

Not yet run: no live MMPose environment exists in this session, so none of
today's fixes (`FetalTrainAugment`, `record_run_provenance.py`) have been
exercised end-to-end. What IS newly true, verified with real passing local
tests (12 new/updated pure-Python tests across `test_fetal_augment.py`,
plus the pre-existing 4 suites, 22 tests total, all passing):

- augmentation parameters are now matched item-by-item against real EoMT/
  HRNet source, with every remaining difference named explicitly (Section 1);
- the flip-order bug is fixed and proven correct by an algebraic invariant
  plus a 500-case randomised property test AND a direct regression test on
  the real measured UCL OFD failure case (Section 2c);
- rotation/scale augmentation is implemented, not deferred (Section 3);
- pretrained-weight provenance and actual parameter counts are a required,
  automatically-recorded artifact, not a manual afterthought (Section 4).

Next step, unchanged from before this audit: install MMPose on the server
per `ENVIRONMENT.md`'s checklist, then run `run_rtmpose_canary.sh` (UCL BPD
seed 42 only), including its new step 3b, then do the mandatory visual-
overlay audit before sharing canary numbers with the supervisor.
