"""MMPose dataset metainfo for the two-endpoint fetal measurement task.

`swap=''` on both keypoints (rather than pointing each at the other) means
our two endpoints have no persistent left/right anatomical identity (thesis
Chapter 3, sec:method-formulation) -- unlike the "swap paired left/right
keypoint" convention COCO's own body keypoints use. Channel identity is
determined entirely by the frozen DOD direction vector (dod_vectors.py), as
re-applied after every accepted augmentation draw by
transforms.FetalTrainAugment.

SUPERSEDED 2026-08-06 (kept for the historical record, not deleted, per
this project's own norm of not silently rewriting history): this file
previously exported a static `FLIP_INDICES = [0, 1]` and relied on MMPose's
stock RandomFlip to apply it, on the theory that HRNet's own `_flip_x_only`
(no index swap) meant a static "never swap on flip" rule was safe. That
theory was WRONG in general: audit_flip_order_stability.py, run against the
real UCL Train CSVs, measured 0/110 (0.0%) of UCL BPD training images where
a flip would invalidate that static rule (BPD's d_vect happens to be
near-vertical), but 110/110, 94/94 and 96/96 (100.0% each) for UCL OFD,
APAD and FL respectively (all near-horizontal d_vect directions) -- i.e.
for 3 of 5 tasks, a flip_indices=[0,1] static rule silently mislabels
essentially every flipped training sample. FLIP_INDICES is no longer
imported by make_config.py or used by the training pipeline;
transforms.FetalTrainAugment re-derives the DOD-canonical order itself
after every accepted flip/rotation instead (see fetal_augment.py and
PROTOCOL_AUDIT.md). Left defined below only in case some MMPose tooling
(e.g. a visualisation script) expects the dataset metainfo to declare it;
it does not participate in this project's own training or evaluation code
path.

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
