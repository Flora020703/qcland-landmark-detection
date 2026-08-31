# SUPERSEDED — DO NOT IMPLEMENT

This candidate protocol was superseded by the supervisor's written decision on
3 August 2026. The final additional landmark baseline is RTMPose-s, not
YOLO-Pose. No YOLO training or data conversion should be started from this
file. The active protocol is documented in
`rtmpose_reproduction/PROTOCOL_LOCKED.md`.

The remainder of this file is retained only as decision history.

# YOLO protocol decision required before implementation

No YOLO source, dependency, checkpoint, configuration, or dataset converter is present in this repository, and the chronological project record does not identify a specific YOLO release. The final experiment must therefore not silently choose one.

Lock the following before writing the training driver:

1. official repository/package and immutable version or commit;
2. model family and size (for example, a particular pose checkpoint—not merely “YOLO”);
3. pretrained checkpoint identity and checksum;
4. representation of one biometry measurement as an object with two ordered keypoints, including how its bounding box is constructed;
5. treatment of endpoint visibility and horizontal flips;
6. fixed final checkpoint convention (last/final, not test-selected best);
7. exact conversion from predicted 512×512 coordinates back to original-image coordinates.

Once locked, the automated contract is already fixed: UCL and Multicentre × BPD/OFD/APAD/TAD/FL × seeds 42/0/123/2024/3407; seed-42 UCL-BPD is the canary; successful canary validation automatically releases the remaining 49 runs; final reporting uses fixed-channel NME and retains per-image coordinates plus diagnostic swap-min NME.

## Current recommendation (not yet supervisor-approved)

The current candidate is **Ultralytics YOLO11s-Pose**, used as a single-instance two-keypoint estimator. Each measurement-specific dataset would contain one class, `kpt_shape: [2, 3]`, and one deterministic full-image box per image (`x_center=0.5`, `y_center=0.5`, `width=1.0`, `height=1.0`). This avoids inventing endpoint-derived box margins and avoids degenerate narrow boxes. It must be disclosed that the detector is therefore not being evaluated as a conventional multi-object detector.

Proposed constraints are `imgsz=512`, final/last checkpoint reporting, external fixed-channel NME, per-image coordinate retention, and diagnostic swap-min NME. Mosaic, mixup, and copy-paste should be disabled because they conflict with the one-image/one-measurement/full-image-instance formulation. Rotation, scale, endpoint canonicalisation, horizontal flips, and `flip_idx` still require a code-level coordinate audit.

This recommendation does **not** close the gate. Supervisor agreement on the pose/full-image-box formulation, followed by an immutable Ultralytics version or commit and pretrained-checkpoint checksum, is required before implementation.
