"""Pure-Python (no MMPose dependency) geometric augmentation + DOD
re-canonicalisation, matching datasets/landmark_dataset.py's exact
augmentation recipe and lib/datasets/fetal.py's ACTUAL (not assumed)
re-projection behaviour.

CORRECTED 2026-08-06 (second pass, same day) after a review caught a real
mathematical error in this file's first version. The first version assumed
HRNet "re-derives the DOD projection after every augmentation draw" by
transforming `d_vect` through the sample's own center/scale/rotation before
projecting -- and therefore made this adapter's FetalTrainAugment do the
same (flip AND rotate d_vect before re-projecting). That assumption was
checked directly against HRNet's real `get_transform`/`_transform_pixel_float`
formulas (lib/utils/transforms.py, lib/datasets/fetal.py) and is WRONG:

`get_transform(center, scale, output_size, rot)` produces an affine map of
the form `new = L(rot, scale) @ (point - center) + output_size/2`, where
`L` does not depend on `center`. Both a sample's own keypoints AND the
frozen `d_vect` prototype points are passed through this SAME map with the
SAME (center, scale, rot) for a given sample. Because `center` enters only
as an additive term shared by both points of a pair, it CANCELS EXACTLY in
any pairwise comparison (`proj_i - proj_j`) -- and because `L` is a
similarity transform (an orthogonal rotation times a positive scalar), the
dot product of two vectors that are BOTH passed through the same `L`
changes by only a positive scalar factor, so `scale` and `rot` cancel too.
The result: HRNet's REAL per-sample DOD ordering decision is provably
(verified both algebraically and by direct numerical reproduction of its
exact formula across 400 randomised center/scale/rotation draws, zero
mismatches) equivalent to comparing the sample's raw, ORIGINAL-image-space
keypoints -- flip-mirrored if a flip was drawn, untouched otherwise -- against
the STATIC, NEVER-TRANSFORMED original `d_vect`. Center, scale, and rotation
have NO effect on which point HRNet calls channel 0.

This means the true, audited-and-verified fact is: **HRNet's own flip
handling has exactly the instability audit_flip_order_stability.py
measured** (100% of UCL OFD/APAD/FL training images end up with a
channel-0/1 assignment that depends on that epoch's own flip draw, because
those tasks' d_vect happens to be near-horizontal) -- this is a real,
pre-existing property of the AUDITED UPSTREAM reference implementation
itself, not something introduced by this adapter, and not something to
"fix" relative to HRNet's own convention. The correct target for this
adapter to replicate is therefore: mirror the ORIGINAL-space keypoints if
flip is drawn, then compare against the STATIC original `d_vect` -- exactly
what `audit_flip_order_stability.py` computed. Rotation and scale need NO
special handling for channel identity at all (proven to have zero effect
on the ordering decision); they only reposition already-ordered points, and
retain EoMT's own independent per-stage accept/reject-if-out-of-canvas
policy for that reason alone (keeping points on-canvas), not for ordering.

The functions below reflect this corrected understanding:
- `resolve_channel_order_after_flip`: the ONLY function that touches
  `d_vect`, evaluated once, in ORIGINAL image-pixel space, using the static
  d_vect -- must run BEFORE the anisotropic pixel-centre resize (see
  transforms.py's FetalRandomFlipAndCanonicalize), because an anisotropic
  resize does not preserve dot-product order for a non-square original
  image the way HRNet's isotropic crop-scale does, so evaluating the
  comparison AFTER resize (as this file's first, incorrect version did)
  would introduce a NEW discrepancy from HRNet's real behaviour rather than
  reproducing it.
- `sequential_rotate_scale`: rotation/scale as pure position updates in
  already-512-space, no d_vect involved, EoMT's own per-stage
  accept/reject-if-out-of-canvas policy.
"""

from __future__ import annotations

import math

from endpoint_order import canonical_order

Point = tuple[float, float]


def _hflip(pt: Point, size: float) -> Point:
    """Matches datasets/landmark_dataset.py line 200 / fetal.py's
    `_flip_x_only`: `size - 1.0 - x` (size is the width of whichever
    coordinate space the point currently lives in)."""
    return (size - 1.0 - pt[0], pt[1])


def _rotate(pt: Point, cx: float, cy: float, angle_deg: float) -> Point:
    """Matches datasets/landmark_dataset.py lines 224-235 exactly, including
    its sign convention (PIL Image.rotate() is counter-clockwise-as-viewed
    for positive angle; image y grows downward, so the sin term's sign is
    flipped relative to the standard y-up rotation matrix) and its choice of
    rotation centre (`iw / 2.0, ih / 2.0`, NOT `(iw - 1) / 2.0`)."""
    theta = math.radians(angle_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    dx, dy = pt[0] - cx, pt[1] - cy
    return (cx + dx * cos_t + dy * sin_t, cy - dx * sin_t + dy * cos_t)


def _scale(pt: Point, cx: float, cy: float, s: float) -> Point:
    """Matches datasets/landmark_dataset.py lines 256-259."""
    return (cx + (pt[0] - cx) * s, cy + (pt[1] - cy) * s)


def rotation_center(size: float) -> float:
    """`size / 2.0`, matching datasets/landmark_dataset.py's own convention
    (NOT the pixel-centre `(size - 1) / 2.0` used elsewhere in this
    project) -- kept as its own tiny function so the two conventions are
    never confused at a call site."""
    return size / 2.0


def in_bounds(pt: Point, size: float) -> bool:
    """Matches the closed-interval bound used by datasets/landmark_dataset.py's
    own out-of-canvas check (`>= 0.0` and `<= iw - 1.0`)."""
    return 0.0 <= pt[0] <= size - 1.0 and 0.0 <= pt[1] <= size - 1.0


def resolve_channel_order_after_flip(
    p0: Point,
    p1: Point,
    d_vect: tuple[Point, Point],
    do_flip: bool,
    orig_width: float,
) -> tuple[Point, Point]:
    """The ONLY function in this module that determines channel identity.
    MUST be called on ORIGINAL image-pixel-space points, BEFORE the
    pixel-centre resize (see this file's module docstring for why an
    anisotropic resize would invalidate the comparison for non-square
    images). Mirrors `p0`/`p1` if `do_flip`, then re-derives the canonical
    order by projecting onto the STATIC, un-transformed `d_vect` -- this is
    what HRNet's real per-sample computation reduces to (verified, see
    module docstring), not a design choice made independently here.

    Rotation and scale are NOT parameters here because they provably do not
    affect this decision (see module docstring); call
    `sequential_rotate_scale` afterward, in already-512-space, purely to
    reposition whichever points this function already ordered.
    """
    pts = [p0, p1]
    if do_flip:
        pts = [_hflip(p, orig_width) for p in pts]
    return canonical_order(pts[0], pts[1], d_vect)


def sequential_rotate_scale(
    p0: Point,
    p1: Point,
    input_size: float,
    angle_deg: float | None,
    scale: float | None,
) -> "SequentialAugmentResult":
    """Repositions an ALREADY-CHANNEL-ORDERED pair (channel identity is
    fixed by `resolve_channel_order_after_flip` and never re-examined here)
    through rotation and/or scale, in already-`input_size`-space. Matches
    datasets/landmark_dataset.py's own independent per-stage accept/reject
    structure: rotation is tried and REVERTED if it would push either point
    outside [0, input_size-1] (keeping the pre-rotation state); scale is
    then tried and independently reverted the same way, on top of whatever
    rotation left behind. A rejected rotation does not discard an
    already-accepted... there is no flip here (flip is handled entirely by
    `resolve_channel_order_after_flip`, before this function ever runs).
    """
    pts = [p0, p1]

    rotation_accepted = False
    if angle_deg:
        c = rotation_center(input_size)
        rot_pts = [_rotate(p, c, c, angle_deg) for p in pts]
        if all(in_bounds(p, input_size) for p in rot_pts):
            pts = rot_pts
            rotation_accepted = True
        # else: reject rotation only, keep pre-rotation pts.

    scale_accepted = False
    if scale is not None and scale != 1.0:
        c = rotation_center(input_size)
        scaled_pts = [_scale(p, c, c, scale) for p in pts]
        if all(in_bounds(p, input_size) for p in scaled_pts):
            pts = scaled_pts
            scale_accepted = True
        # else: reject scale only, keep pre-scale pts.

    return SequentialAugmentResult(
        p0=pts[0], p1=pts[1],
        flip_applied=False,  # flip already handled upstream; not this function's concern
        rotation_accepted=rotation_accepted,
        scale_accepted=scale_accepted,
    )


class SequentialAugmentResult(tuple):
    """Acts like a (p0, p1) tuple (`result[0]`, `result[1]`, and
    `p0, p1 = result` unpacking all work) plus named
    `.rotation_accepted` / `.scale_accepted` flags transforms.py needs to
    keep the actual image pixels consistent with which augmentations the
    keypoints actually underwent. `.flip_applied` is kept for interface
    compatibility with earlier callers but is always False here (flip is
    resolved by `resolve_channel_order_after_flip`, a separate stage)."""

    def __new__(cls, p0, p1, flip_applied, rotation_accepted, scale_accepted):
        obj = super().__new__(cls, (p0, p1))
        obj.flip_applied = flip_applied
        obj.rotation_accepted = rotation_accepted
        obj.scale_accepted = scale_accepted
        return obj
