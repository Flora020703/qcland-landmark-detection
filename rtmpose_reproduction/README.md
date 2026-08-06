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

Run all four test files directly (`python test_*.py`) -- no MMPose
dependency, works in a plain Python env (this session used the project's
local `comp0081` conda env).

**Not yet verified, because doing so needs a real MMPose install this
session did not have** -- see `ENVIRONMENT.md`'s numbered checklist before
trusting canary output:

| File | Main risk if unverified |
|---|---|
| `transforms.py` | Exact `results` dict keys/shapes MMPose's `PackPoseInputs` expects; import paths (`mmcv.transforms.BaseTransform`, `mmpose.registry.TRANSFORMS`) may differ by installed version. |
| `make_config.py` | Field names/nesting for `RTMCCHead`/`CSPNeXt`/`SimCCLabel` were cross-checked against the official config's *documented* structure, not built and run. |
| `run_inference.py` | **Highest risk file.** Deliberately bypasses MMPose's high-level inference API to avoid its stock bbox-based inverse transform (see the file's own docstring for why) -- but this means it depends on `model.head.decode(...)` behaving exactly as assumed. Confirm against the installed source before trusting any exported coordinate. |
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

## Explicit scope decisions NOT yet raised with the supervisor (do before the full sweep, not necessarily before the canary)

- **No rotation/scale augmentation in this first version** (`make_config.py`'s
  docstring). The canary only needs to prove basic geometric/architectural
  correctness; adding HRNet-equivalent rotation-aware DOD re-projection is
  extra work this session did not attempt. Flag this explicitly once the
  canary passes -- do not silently add rotation augmentation later without
  re-deriving the endpoint-order handling under it.
- **No post-flip DOD re-sort**: `endpoint_order.canonical_order()` runs once
  at CSV-conversion time, before any augmentation; `RandomFlip` mirrors
  coordinates without swapping channel identity (`flip_indices=[0,1]` in
  `fetal_dataset_info.py`). HRNet's own pipeline re-derives the projection
  sort after every flip/rotation draw; this adapter does not yet. Documented
  in `fetal_dataset_info.py`'s docstring as the same class of limitation
  already disclosed for EoMT's own endpoint canonicalisation, not a new
  silent bug -- but it is a real behavioural difference from HRNet worth
  naming explicitly if it turns out to matter.

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
