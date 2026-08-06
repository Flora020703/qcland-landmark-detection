# RTMPose-s adapter (UCL + Multicentre, two-endpoint fetal measurements)

Implements `PROTOCOL_LOCKED.md`, confirmed by the supervisor's email (RTMPose-s,
backbone-only pretrained init, `sigma=(8.0,8.0)` extrapolation, direct
non-aspect-preserving 512x512 resize with the pixel-centre convention,
UCL-BPD/seed-42 canary before the five-seed sweep).

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
| `fetal_augment.py` | `test_fetal_augment.py`: algebraic invariant (flip/rotation preserve DOD projection order when the direction vector is transformed consistently) checked via a 500-case randomised property test, a direct regression test on the real UCL OFD flip-bug case found by `audit_flip_order_stability.py` (see `PROTOCOL_AUDIT.md` Section 2c), and per-stage accept/reject (out-of-bounds rotation/scale) tests. **7/7 pass.** |
| `audit_flip_order_stability.py` | Not a unit test -- a one-off measurement script, already run against the real UCL/Multicentre Train CSVs; its findings are recorded in `PROTOCOL_AUDIT.md`, not just left as ad hoc console output. |

Run all four test files directly (`python test_*.py`) -- no MMPose
dependency, works in a plain Python env (this session used the project's
local `comp0081` conda env).

**Not yet verified, because doing so needs a real MMPose install this
session did not have** -- see `ENVIRONMENT.md`'s numbered checklist before
trusting canary output:

| File | Main risk if unverified |
|---|---|
| `transforms.py` (`PixelCentreResize`) | Exact `results` dict keys/shapes MMPose's `PackPoseInputs` expects; import paths (`mmcv.transforms.BaseTransform`, `mmpose.registry.TRANSFORMS`) may differ by installed version. |
| `transforms.py` (`FetalTrainAugment`, added 2026-08-06) | Same `results`-dict-contract risk as `PixelCentreResize`, plus: assumes it runs immediately after `PixelCentreResize` (operates on an already-512x512 image/keypoints, not the original); its own flip/rotate/scale + DOD-reorder LOGIC is fully unit-tested pure Python (`fetal_augment.py`, see above), only the MMPose glue (image dict keys, `GenerateTarget` compatibility) is unverified. |
| `make_config.py` | Field names/nesting for `RTMCCHead`/`CSPNeXt`/`SimCCLabel` were cross-checked against the official config's *documented* structure, not built and run. The generated config DOES successfully import `dod_vectors`/`fetal_dataset_info` and fails only at the `cv2`/`mmcv` import boundary when executed locally (confirmed 2026-08-06) -- everything before that boundary is real, working Python, not just plausible-looking text. |
| `run_inference.py` | **Highest risk file.** Deliberately bypasses MMPose's high-level inference API to avoid its stock bbox-based inverse transform (see the file's own docstring for why) -- but this means it depends on `model.head.decode(...)` behaving exactly as assumed. Confirm against the installed source before trusting any exported coordinate. |
| `record_run_provenance.py` (added 2026-08-06) | Assumes `model.backbone`/`model.head` attribute names and `cfg.model["backbone"]["init_cfg"]` structure match the installed `TopdownPoseEstimator`/`CSPNeXt` -- confirm against the installed source, same tier as the rest of this table. |
| `run_rtmpose_canary.sh` | Assumes `tools/train.py <config>` is the correct entrypoint for the installed MMPose version (standard OpenMMLab convention, not confirmed against this specific checkout). |

## Design decisions carried over from the HRNet-512 adapter

- **DOD/endpoint identity**: reuses HRNet's own frozen `d_vect` values
  verbatim (extracted from its checkpoints, not re-derived), giving
  byte-identical channel-ordering convention across HRNet and RTMPose. See
  `dod_vectors.py`'s docstring for the extraction provenance.
- **Geometry**: the same UDP-inspired pixel-centre convention as EoMT
  (`x'=(x+0.5)(S/W)-0.5`), NOT MMPose's default padded/aspect-preserving
  `GetBBoxCenterScale`+`TopdownAffine`. `PixelCentreResize` replaces both in
  the pipeline; the codec's own `.decode()` only ever sees/returns
  512-space coordinates, and `geometry.to_image_space()` is the only path
  back to original-image coordinates.
- **Evaluator**: identical fixed-channel/swap-min formula as
  `baseline_reproduction/evaluate_hrnet_fixed.py`, so RTMPose and HRNet
  numbers are computed by the same code, not parallel reimplementations
  that could quietly diverge.
- **flip_test=False**: this project already investigated and abandoned
  flip-based TTA for two-endpoint fetal tasks (thesis `sec:method-tta`) due
  to endpoint-order instability for near-vertical diameters. Enabling
  RTMPose's own `flip_test` would reintroduce exactly that failure mode.

## RESOLVED 2026-08-06 (see `PROTOCOL_AUDIT.md` for the full write-up)

The two items previously listed here as deferred scope decisions were
resolved during a pre-canary code-level protocol audit, not left for later:

- **Rotation/scale augmentation** is now implemented (`transforms.
  FetalTrainAugment`), matching EoMT/HRNet's exact parameters
  (ROT_FACTOR=30 p=0.6, SCALE_FACTOR=0.25 unconditional) -- see
  `PROTOCOL_AUDIT.md` Section 3.
- **Post-flip DOD re-sort** is now implemented and was in fact a REAL BUG,
  not just a theoretical limitation: `audit_flip_order_stability.py`,
  run against the real released UCL Train CSVs, measured that the old
  static `flip_indices=[0,1]` design silently mislabelled the channel
  identity on 100% of UCL OFD/APAD/FL training images (0.0%/0.2% for
  BPD/TAD specifically, since those two tasks' DOD direction happens to be
  near-vertical) -- see `PROTOCOL_AUDIT.md` Section 2c for the measured
  numbers, the fix (`fetal_augment.py`'s `sequential_train_augment`,
  matching HRNet's own per-sample re-projection architecture), and the
  algebraic + empirical proof of correctness (`test_fetal_augment.py`,
  7/7 pass).

Still not fully identical to HRNet's own training-time behaviour: this
adapter re-derives DOD order using ORIGINAL-image-space `d_vect`, carried
through EACH sample's own flip/rotate/scale draws independently -- this
mirrors HRNet's architecture but has not been proven bit-identical to
HRNet's exact training-time numerical output (out of scope; the two
implementations' underlying warp mechanics already differ per Section 1's
table). What IS verified is that RTMPose's own training targets are now
internally self-consistent under augmentation, which they provably were not
before this fix for 3 of 5 tasks.

## Usage

```bash
export PY=/root/rtmpose_env/bin/python
export MMPOSE_TRAIN_TOOL=/root/mmpose/tools/train.py
export DATA_ROOT=/root/autodl-tmp
bash rtmpose_reproduction/run_rtmpose_canary.sh
```

Stops after the canary. Do not write a 50-run driver or start the five-seed
sweep until the canary's numbers have been reviewed and shared with the
supervisor, per `PROTOCOL_LOCKED.md` and the supervisor's own email.
