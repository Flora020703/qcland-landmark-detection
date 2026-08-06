# RTMPose-s adapter (UCL + Multicentre, two-endpoint fetal measurements)

Implements `PROTOCOL_LOCKED.md`, confirmed by the supervisor's email (RTMPose-s,
backbone-only pretrained init, `sigma=(8.0,8.0)` extrapolation, direct
non-aspect-preserving 512x512 resize with the pixel-centre convention,
UCL-BPD/seed-42 canary before the five-seed sweep). See `PROTOCOL_AUDIT.md`
for the full pre-canary code-level audit (two review rounds, 2026-08-06) --
it found and fixed one real bug, corrected a mathematically wrong first
diagnosis of that bug, fixed four more blocking issues (test-set leakage,
inference preprocessing bypass, non-deterministic checkpoint selection,
unverified pretrained-weight loading), and named one item (endpoint-order
convention parity with EoMT) as still genuinely open, not resolved.

## Status: core geometry/labelling verified locally; MMPose wiring not yet
## run against a live install

This was built and reviewed without server/GPU/MMPose access in this
session. Everything that could be verified with pure Python has been, with
tests that actually run and pass (not just written):

| File | Verified how |
|---|---|
| `geometry.py` | `test_geometry.py`: round-trip exactness (corners, non-square images, 5 resolutions), pixel-centre-vs-naive-scaling distinction, horizontal-flip commutation. **6/6 pass.** |
| `dod_vectors.py` | Extracted directly from the real, already-trained `checkpoint_backups/hrnet-512-fixed-50runs.tar` (`torch.load` on `final_state.pth`), not re-derived from scratch. Cross-checked UCL-BPD seed 0 vs seed 42 give the bit-identical vector, confirming it depends only on (dataset, task), not the training seed. |
| `endpoint_order.py` | `test_endpoint_order.py`: reproduces HRNet's OWN real per-image canonicalisation exactly for two actual UCL BPD test images pulled from `fixed_channel_per_image.csv` -- including `004_HC.jpeg`, a genuine swap case (raw CSV order differs from canonical order). **4/4 pass.** |
| `convert_csv_to_coco.py` | `test_convert_csv_to_coco.py`: synthetic non-square images, row filtering (missing image / negative landmark), bbox correctness, and that the real swap case round-trips through the actual CSV-parsing code path. **Passes.** |
| `evaluate_rtmpose_fixed.py` | `test_evaluate_rtmpose_fixed.py`: exact-prediction zero-NME, swapped-prediction-only-hurts-fixed-channel, and a deliberately-broken-swap-min case proving the fixed>=swap-min invariant assertion actually fires rather than passing by luck. **3/3 pass.** |
| `fetal_augment.py` | `test_fetal_augment.py`, rewritten in the second audit round after the first round's own design was found mathematically wrong (see `PROTOCOL_AUDIT.md`): tests the CORRECTED functions (`resolve_channel_order_after_flip`, `sequential_rotate_scale`) -- a regression test on the real UCL OFD case, a 500-case randomised property test of the "same original point always wins channel 0" invariant, and per-stage accept/reject (out-of-bounds rotation/scale) tests. **7/7 pass.** |
| `make_internal_val_split.py` (added in the second audit round) | Run against the real UCL BPD Train CSV: 100 internal-train / 10 internal-val (of 110 rows) -- ports EoMT's own exact subject-grouping/shuffle/split algorithm verbatim, not an independently invented split. |
| `audit_flip_order_stability.py` | Not a unit test -- a one-off measurement script, already run against the real UCL/Multicentre Train CSVs; its findings (and their correct interpretation, after the second-round correction) are recorded in `PROTOCOL_AUDIT.md`, not just left as ad hoc console output. |
| `low_level_decode.py` (pure functions only) | `test_low_level_decode.py`: `to_original_image_space` checked against `geometry.py`'s own round-trip; `fixed_channel_nme` checked against a hand-computed value. **3/3 pass.** (`decode_batch_low_level` itself needs a live model, see the live-verification table below.) |

Run all test files directly (`python test_*.py`) -- no MMPose dependency,
works in a plain Python env (this session used the project's local
`comp0081` conda env).

**Not yet verified, because doing so needs a real MMPose install this
session did not have** -- see `ENVIRONMENT.md`'s numbered checklist before
trusting canary output:

| File | Main risk if unverified |
|---|---|
| `transforms.py` (`PixelCentreResize`) | Exact `results` dict keys/shapes MMPose's `PackPoseInputs` expects; import paths (`mmcv.transforms.BaseTransform`, `mmpose.registry.TRANSFORMS`) may differ by installed version. |
| `transforms.py` (`FetalRandomFlipAndCanonicalize`, corrected in the second audit round) | Runs BEFORE `PixelCentreResize` (operates on the ORIGINAL, not-yet-resized image/keypoints) -- deliberate, since channel identity can only be decided correctly in original-image space (see `PROTOCOL_AUDIT.md`). Its own flip/reorder LOGIC is fully unit-tested pure Python (`fetal_augment.py`), only the MMPose glue is unverified. |
| `transforms.py` (`FetalRotateScaleColorJitter`, corrected in the second audit round) | Runs AFTER `PixelCentreResize`, position-only (no channel reordering). Colour-jitter channel-order handling (`assume_bgr=True`) is a documented assumption about `LoadImage`'s default output that needs live confirmation -- see the class's own docstring. |
| `make_config.py` | Field names/nesting cross-checked against the REAL official config (fetched verbatim from GitHub in the third audit round, not just documentation) -- `_scope_="mmdet"`, `expand_ratio=0.5`, `SyncBN`, `paramwise_cfg`, and a proportional cosine-LR start were all found missing and fixed; `EMAHook`/stage-2 pipeline switch/`auto_scale_lr`/420-epoch training are deliberately NOT matched, with reasons recorded in the file's own module docstring. The generated config DOES successfully import `dod_vectors`/`fetal_dataset_info` and fails only at the `cv2`/`mmcv` import boundary when executed locally -- everything before that boundary is real, working Python. `SyncBN` building on a single-GPU/non-distributed process still needs live confirmation (`ENVIRONMENT.md` item 3). |
| `run_inference.py` | **Highest risk file.** Deliberately bypasses MMPose's high-level inference API to avoid its stock bbox-based inverse transform (see the file's own docstring for why), routing through `model.data_preprocessor` directly instead (fixed in the second audit round -- the first version bypassed it entirely, feeding un-normalised pixels to the network). Depends on `model.data_preprocessor`'s exact call contract and `model.head.decode(...)` behaving exactly as assumed -- confirm both against the installed source before trusting any exported coordinate. |
| `record_run_provenance.py` | Corrected in the second audit round to diff state-dict values against a local file; CORRECTED AGAIN in the third round after a review found the local file being hashed was not actually guaranteed to be the file `model.init_weights()` loaded (the config's own `init_cfg.checkpoint` was still a URL) -- `make_config.py` now embeds the local path directly, and this script asserts the two match plus requires an EXACT per-key value match against the checkpoint, not just "changed from random init." Still assumes `model.backbone`/`model.head` attribute names and `cfg.model["backbone"]["init_cfg"]` structure match the installed `TopdownPoseEstimator`/`CSPNeXt`, confirm against the installed source. |
| `run_rtmpose_canary.sh` | Assumes `tools/train.py <config>` is the correct entrypoint, and that MMEngine writes a `last_checkpoint` pointer file in the documented format (used for final-checkpoint verification, corrected in the second audit round from an unreliable glob+sort) -- both standard OpenMMLab/MMEngine conventions, not confirmed against this specific installed checkout. Now calls `live_preflight.py` as a mandatory, hard-fail gate before training (round 5) -- see that file's own row below. |
| `live_preflight.py` (added round 5) | The enforcement mechanism for ENVIRONMENT.md's checklist -- previously documented but never actually forced to run before training. Exercises the full non-square+SimCC-codec round trip, one real train forward/backward pass, and the low-level decode path, all against the real installed MMPose, failing loudly (and refusing to let the canary proceed to training) on any assertion failure. Its own `encode()`/`decode()` dict-key assumptions (`keypoint_x_labels`/`keypoint_y_labels`) are inferred from SimCCLabel's documented interface, not run against a live install -- if this specific step fails, that IS the preflight doing its job, not a bug in the preflight itself. |
| `internal_val_hook.py` (added round 5) | Replaces MMPose's default `model.predict()`-based periodic validation (disabled via `val_cfg=None` in `make_config.py`, round 5) after a review flagged that `predict()` likely needs bbox metadata this project's pipeline never sets. Shares `low_level_decode.py`'s verified-safe decode path with `run_inference.py`. Hook lifecycle method name/signature (`after_train_epoch`) and `runner.model`/`runner.message_hub` access patterns are standard MMEngine conventions, not confirmed against the installed version. |
| `low_level_decode.py` (added round 5) | Refactored out of `run_inference.py` so both that script and `internal_val_hook.py` share one implementation instead of two independently-maintained copies that could silently diverge. `decode_batch_low_level` needs live MMPose (same assumptions as `run_inference.py`'s own docstring); `to_original_image_space`/`fixed_channel_nme` are pure and unit-tested locally (`test_low_level_decode.py`, 3/3 pass). |

## Design decisions

- **DOD/endpoint identity**: reuses HRNet's own frozen `d_vect` values
  verbatim (extracted from its checkpoints, not re-derived), giving
  byte-identical channel-ordering convention across HRNet and RTMPose. See
  `dod_vectors.py`'s docstring for the extraction provenance. **This makes
  RTMPose match HRNet, not EoMT** -- EoMT uses its own, different,
  already-locked convention (per-sample x-coordinate sort); see
  `PROTOCOL_AUDIT.md`'s "Still not fully unified" section for the
  quantified disagreement and the still-open question for the supervisor.
- **Geometry**: the same UDP-inspired pixel-centre convention as EoMT
  (`x'=(x+0.5)(S/W)-0.5`), NOT MMPose's default padded/aspect-preserving
  `GetBBoxCenterScale`+`TopdownAffine`. `PixelCentreResize` replaces both in
  the pipeline; the codec's own `.decode()` only ever sees/returns
  512-space coordinates, and `geometry.to_image_space()` is the only path
  back to original-image coordinates.
- **Channel identity must be decided in ORIGINAL image space, before the
  resize** (`FetalRandomFlipAndCanonicalize`, runs first in the train
  pipeline) -- verified against HRNet's real per-sample transform formula
  that this is what HRNet's own decision reduces to; deciding it after the
  anisotropic 512x512 resize (this project's own first, incorrect design)
  would silently diverge from HRNet's real behaviour for non-square source
  images. See `PROTOCOL_AUDIT.md` for the full derivation.
- **Evaluator**: identical fixed-channel/swap-min formula as
  `baseline_reproduction/evaluate_hrnet_fixed.py`, so RTMPose and HRNet
  numbers are computed by the same code, not parallel reimplementations
  that could quietly diverge.
- **Train-only internal validation split** (`make_internal_val_split.py`),
  reusing EoMT's own exact algorithm -- the released Test set is never read
  during training, only once, after training, by `run_inference.py`.
- **flip_test=False**: this project already investigated and abandoned
  flip-based TTA for two-endpoint fetal tasks (thesis `sec:method-tta`) due
  to endpoint-order instability for near-vertical diameters. Enabling
  RTMPose's own `flip_test` would reintroduce exactly that failure mode.

## Still genuinely open (not resolved by this session, needs the supervisor)

**EoMT does not share RTMPose/HRNet's endpoint-ordering convention.**
RTMPose and HRNet both use a learned, per-(dataset,task) direction vector
(`d_vect`); EoMT uses a per-sample x-coordinate sort. These are
mathematically different rules that disagree on 0-100% of images depending
on the task (quantified in `PROTOCOL_AUDIT.md`). This cannot be resolved
by retraining EoMT (its 5-seed results are already locked/reported) without
a separate decision from the supervisor about which of two paths to take
(method-native training + common external evaluator, vs. a new offline
re-canonicalisation of all three methods' raw predictions). Do not describe
the three methods as sharing one ordering convention in any canary report
or thesis text until this is explicitly settled.

## Usage

```bash
export PY=/root/rtmpose_env/bin/python
export MMPOSE_TRAIN_TOOL=/root/mmpose/tools/train.py
export DATA_ROOT=/root/autodl-tmp
export PRETRAINED_CKPT_PATH=/root/cspnext-s_udp-aic-coco_210e-256x192-92f5a029_20230130.pth
bash rtmpose_reproduction/run_rtmpose_canary.sh
```

Stops after the canary. Do not write a 50-run driver or start the five-seed
sweep until the canary's numbers have been reviewed and shared with the
supervisor, per `PROTOCOL_LOCKED.md` and the supervisor's own email -- and
until the still-open endpoint-ordering-convention question above has been
raised and answered.
