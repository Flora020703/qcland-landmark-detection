"""MMPose dataset metainfo for the two-endpoint fetal measurement task.

`swap=''` on both keypoints (rather than pointing each at the other) is a
deliberate choice, not an oversight: it tells MMPose's flip augmentation to
mirror each keypoint's x-coordinate WITHOUT swapping which channel holds
which point (matching flip_indices=[0, 1], the identity mapping). This
reproduces the audited upstream HRNet behaviour (`_flip_x_only` in
lib/datasets/fetal.py, used whenever DOD/REASSIGN is enabled) rather than
the "swap paired left/right keypoint" convention COCO's own body keypoints
use -- our two endpoints have no persistent left/right anatomical identity
(thesis Chapter 3, sec:method-formulation); the frozen DOD direction vector
(dod_vectors.py), not the flip transform, is what determines channel
identity.

SCOPE NOTE (documented, not hidden): this project's converter applies
endpoint_order.canonical_order() ONCE, before any augmentation, using the
original un-augmented coordinates. The audited upstream HRNet pipeline
re-applies its own equivalent projection-based sort AFTER every
rotation/flip augmentation draw, so a training-time rotated/flipped HRNet
sample can occasionally end up in the opposite channel order from this
project's fixed-at-conversion-time order. Replicating that exactly would
require re-deriving the direction-vector projection in the current sample's
augmented coordinate frame on every __getitem__ call, which this first
implementation does not attempt -- this is the same class of already-
disclosed endpoint-canonicalisation instability documented for EoMT's own
x-sort convention (thesis sec:discussion-limitations-endpoint-canon), not a
new, silent problem. Revisit only if the canary's visual-overlay audit
(PROTOCOL_LOCKED.md's mandatory canary gate) shows this actually matters in
practice.

`sigmas` below are OKS keypoint sigmas, an MMPose/COCO-metric bookkeeping
field unrelated to the SimCC label sigma locked in PROTOCOL_LOCKED.md
(8.0, 8.0) -- this project's external fixed-channel/swap-min evaluator
(evaluate_rtmpose_fixed.py) does not use OKS or these sigmas at all; they
are set to a neutral placeholder only because MMPose's config schema
expects the field to exist.
"""

from __future__ import annotations

FETAL_DATASET_INFO = dict(
    dataset_name="fetal_two_endpoint",
    paper_info=dict(
        author="", title="", container="", year="2026", homepage="",
    ),
    keypoint_info={
        0: dict(name="endpoint_0", id=0, color=[255, 0, 0], type="", swap=""),
        1: dict(name="endpoint_1", id=1, color=[0, 0, 255], type="", swap=""),
    },
    skeleton_info={
        0: dict(link=("endpoint_0", "endpoint_1"), id=0, color=[0, 255, 0]),
    },
    joint_weights=[1.0, 1.0],
    # Placeholder OKS sigmas -- NOT used by this project's own evaluator.
    sigmas=[0.05, 0.05],
)

# Explicit identity mapping: flipping does not swap channel identity.
FLIP_INDICES = [0, 1]
