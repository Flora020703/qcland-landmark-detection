# RTMPose-s final-comparison protocol

Status: **locked by supervisor correspondence, 3 August 2026**.

## Why RTMPose replaces YOLO-Pose

The released fetal-biometry datasets provide landmark endpoints but no
meaningful object boxes.  YOLO-Pose would learn an artificial detection target
and incur an irrelevant box loss even if every label used a full-image box.
RTMPose is a top-down pose estimator: a supplied region defines the affine input
crop, while the model predicts keypoints and does not predict a bounding box.
Every sample can therefore use its full source image as the deterministic input
region without adding a box-prediction task.

## Locked comparison scope

- Method: OpenMMLab **RTMPose-s**, measurement-specific two-keypoint model.
- Datasets: UCL and Multicentre released Train/Test partitions.
- Tasks: BPD, OFD, APAD, TAD and FL as five independent models per dataset.
- Input: square `512 x 512` full-image input.  To match the existing EoMT and
  HRNet comparison protocol, the source image is resized directly and
  anisotropically to `512 x 512`; aspect ratio is not preserved.
- Region: the complete source image (`[0, 0, width, height]` in COCO-style
  annotation coordinates); the region is an input transform, not a target
  predicted by the network.
- Keypoints: exactly two visible endpoints for the selected measurement.
- Seeds: `42, 0, 123, 2024, 3407` if the implementation passes the canary.
- Primary checkpoint convention: final/last model state, consistent with the
  locked cross-method table.  Any validation metric is diagnostic and must not
  silently change this convention.
- Primary external metric: per-image **fixed-channel NME**, using exactly the
  same endpoint distance normalization and reporting code as EoMT/HRNet.
- Diagnostic metric: swap-min NME from the same predictions.
- Required outputs: original filename, both predicted endpoint coordinates,
  both ground-truth endpoint coordinates, fixed-channel NME, swap-min NME,
  run-level aggregate, configuration, environment audit and checkpoint.

RTMPose keeps its native RTMCC/SimCC keypoint objective.  Matching input size,
split, seeds and external NME does not mean replacing the method's native loss
with an EoMT heatmap loss.

## Backbone, head and codec lock

- Load only the official CSPNeXt-s backbone initialization using the official
  `prefix='backbone.'` convention.  Do not attempt to load a COCO RTMCCHead.
- Initialize the complete two-keypoint RTMCCHead from scratch.  At `512 x 512`,
  set `in_featuremap_size=(16, 16)`, `simcc_split_ratio=2.0`, and therefore use
  1024 x-bins and 1024 y-bins.
- Set the SimCC label sigma to `(8.0, 8.0)`.  This is an explicit extrapolation,
  not a published 512-square RTMPose setting: the official values
  `(4.9, 5.66)` at `(192, 256)` and `(6.0, 6.93)` at `(288, 384)` follow
  `sigma_axis = sqrt(axis_length / 8)`, which gives `sqrt(512/8)=8.0`.
- Record that HRNet's two-dimensional heatmap `sigma=1` and RTMPose's SimCC
  sigma have different meanings and are not numerically matched parameters.

The stock MMPose full-bbox pipeline must not be assumed to reproduce this
geometry. `GetBBoxCenterScale` defaults to 1.25 padding, and `TopdownAffine`
expands the shorter bbox dimension to the target aspect ratio before warping.
For a non-square source image this preserves aspect ratio and pads outside the
shorter dimension, whereas the locked EoMT/HRNet protocol uses a direct
non-aspect-preserving resize.  The adapter must therefore implement the direct
mapping explicitly using the same UDP-inspired pixel-centre convention as the
locked EoMT pipeline:

`x_512=(x_original+0.5)*(512/width)-0.5` and
`y_512=(y_original+0.5)*(512/height)-0.5`.

The exact inverses are
`x_original=(x_512+0.5)/(512/width)-0.5` and the corresponding y expression.
It must not silently use the stock padded full-bbox transform.

The native SimCC codec remains responsible for converting between its 1024-bin
axis distributions and coordinates in the 512 input space.  Only the subsequent
512-space-to-original-image mapping is replaced by the explicit inverse above;
MMPose's bbox/aspect-ratio inverse transform must not be called for final saved
coordinates.

## Endpoint identity gate

The two output channels require a reproducible label identity.  Before any
training, the converter and evaluator must be audited against the project's
existing training-derived DOD/canonicalisation convention used for the matched
HRNet fixed-channel results.  Direction information must be estimated from the
training partition only and reused unchanged for validation/test; test labels
must never determine channel ordering.  Horizontal-flip metadata must swap or
retain endpoint indices exactly as implied by that audited canonicalisation.

A synthetic coordinate test must cover horizontal, vertical, near-horizontal
and near-vertical endpoint pairs and prove that:

1. CSV -> dataset annotation -> 512-space target -> inverse geometric transform
   recovers the original coordinates within a declared numerical tolerance;
2. fixed-channel and swap-min evaluators reproduce direct NumPy calculations;
3. fixed-channel NME is never smaller than swap-min NME for the same sample;
4. flip plus inverse flip preserves the canonical endpoint identities.

**Update, 2026-08-06**: item 4 above was audited with real code and real
data (`audit_flip_order_stability.py`) before the canary, per this
project's own review process, and the FIRST implementation of this gate
failed it -- a static `flip_indices` setting silently mislabelled ~100% of
UCL OFD/APAD/FL training samples under flip (0.0-0.2% for BPD/TAD, whose
DOD direction happens to be near-vertical). Fixed via
`fetal_augment.sequential_train_augment` / `transforms.FetalTrainAugment`,
which re-derives the DOD projection after every accepted flip/rotation
instead of assuming a fixed swap/no-swap rule. Full writeup:
`PROTOCOL_AUDIT.md`. This is the concrete lesson behind this file's own
instruction above ("must be audited... before any training") -- audit
against real code and real data, not by name/analogy alone.

## Mandatory canary

The first and only canary is **UCL BPD, seed 42**.  It must stop after training
and external evaluation; the remaining 49 runs are not released automatically
until all gates below are reviewed:

- exact released Train/Test filenames and counts;
- no Train/Test leakage introduced by conversion;
- two valid visible keypoints per retained row, with exclusions reported;
- full-image region equals each source image's actual dimensions;
- model input and codec input are both `512 x 512`;
- deterministic seed coverage includes Python, NumPy, PyTorch, CUDA, sampler
  and DataLoader workers;
- predicted coordinates are inverse-transformed to original image space;
- fixed-channel and swap-min per-image CSVs pass the evaluator invariants;
- a small set of visual overlays confirms coordinate and endpoint identity;
- final checkpoint and all audit artifacts are readable and archived.

After canary approval, use one resumable fail-fast driver for the remaining
dataset/task/seed cells.  It must preflight disk space, avoid duplicate result
rows, retain one final checkpoint plus per-image predictions per run, and safe
stop on any failed assertion.

## Version and provenance gate

Before server training, record and freeze:

- the exact MMPose Git commit or release tag;
- compatible MMEngine, MMCV and MMDetection versions;
- the exact official RTMPose-s base config;
- the pretrained backbone/checkpoint URL and SHA-256;
- every local config/data-adapter change as an auditable diff.

Do not train against an unpinned moving `main` checkout.  The official
RTMPose-s configuration is the methodological source, but its COCO-specific 17
keypoints, rectangular input, COCO evaluator and person-specific metadata must
be replaced explicitly for this two-endpoint fetal task.
