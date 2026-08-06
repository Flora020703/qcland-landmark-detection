"""Pure-Python (no MMPose dependency) geometric augmentation + DOD
re-canonicalisation, matching datasets/landmark_dataset.py's exact
augmentation recipe and lib/datasets/fetal.py's exact re-projection
architecture.

WHY THIS FILE EXISTS (found via audit_flip_order_stability.py, run against
the real UCL Train CSVs, 2026-08-06): the adapter's original design (freeze
canonical_order() once at CSV-conversion time, then apply MMPose's stock
RandomFlip with a static flip_indices=[0, 1] "no swap" rule) is WRONG for
any task whose frozen `d_vect` direction is predominantly horizontal. The
audit measured 0/110 (0.0%) of UCL BPD training images where a flip would
invalidate the frozen order (d_vect is near-vertical for BPD), but 110/110
(100.0%), 94/94 (100.0%) and 96/96 (100.0%) for UCL OFD, APAD and FL
respectively (all near-horizontal d_vect directions) -- i.e. for 3 of 5
tasks, EVERY flipped training sample would silently receive the wrong
channel-to-endpoint assignment under the original design. This is not the
"occasionally" framing the adapter's docstrings previously used; it is a
near-total failure for those three tasks specifically, though it happens to
be a non-issue for the BPD canary itself.

Fix, matching lib/datasets/fetal.py's OWN architecture (fetal.py lines
249-289: the DOD reassignment there operates on `tpts_flt`, the sample's
OWN keypoints already carried through that sample's random rotation/scale/
flip via `transform_pixel`/`crop`, and separately re-projects the frozen
`self.d_vect` prototype points through the SAME per-sample transform before
computing the projection) -- HRNet does not freeze an order once; it
re-derives the projection-based order after every augmentation draw, using
a per-sample-transformed copy of the same frozen direction. This module
does the equivalent: given a sample's keypoints AND the frozen d_vect (both
in ORIGINAL image pixel coordinates), apply the identical sequence of
geometric operations to both, then canonicalise using the transformed
direction.
"""

from __future__ import annotations

import math

from endpoint_order import canonical_order
from geometry import to_model_space

Point = tuple[float, float]


def _hflip(pt: Point, size: int) -> Point:
    """Matches datasets/landmark_dataset.py line 200: `iw - 1.0 - lms[:, 0]`."""
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


def rotation_center(input_size: int) -> float:
    """`iw / 2.0`, matching datasets/landmark_dataset.py's own convention
    (NOT the pixel-centre `(iw - 1) / 2.0` used elsewhere in this project) --
    kept as its own tiny function so the two conventions are never confused
    at a call site."""
    return input_size / 2.0


def in_bounds(pt: Point, size: int) -> bool:
    """Matches the closed-interval bound used by datasets/landmark_dataset.py's
    own out-of-canvas check (`>= 0.0` and `<= iw - 1.0`)."""
    return 0.0 <= pt[0] <= size - 1.0 and 0.0 <= pt[1] <= size - 1.0


def augment_and_canonicalize(
    p0: Point,
    p1: Point,
    orig_width: float,
    orig_height: float,
    d_vect: tuple[Point, Point],
    input_size: int,
    do_flip: bool = False,
    angle_deg: float | None = None,
    scale: float | None = None,
) -> tuple[Point, Point]:
    """Resize (pixel-centre) to `input_size`-space, apply the given flip/
    rotate/scale draws to BOTH the sample keypoints and the frozen d_vect
    prototype points, then return the keypoints re-canonicalised by
    projecting onto the correspondingly-transformed direction.

    Scale is intentionally NOT applied to the d_vect prototype points: a
    positive uniform scale about a fixed centre preserves the sign of any
    pairwise projection difference along any fixed direction (p_i' - p_j' =
    s * (p_i - p_j) for s > 0), so it can never change which of two points
    projects further along a given direction -- confirmed algebraically, not
    just assumed; see test_fetal_augment.py's
    test_scale_never_changes_dod_order for a randomised check. Flip
    (a reflection) and rotation can both change this sign and DO require
    re-projecting the direction itself.

    Caller is responsible for the "skip this augmentation draw if it would
    push a keypoint off-canvas" decision (matching datasets/landmark_dataset.py's
    own philosophy of skipping rather than clamping) -- this function just
    applies whatever parameters it is given. Use `in_bounds()` on this
    function's *keypoint* outputs (not the d_vect outputs, which are not
    real keypoints) to implement that check at the call site.
    """
    def to_model(pt: Point) -> Point:
        return to_model_space(pt[0], pt[1], orig_width, orig_height, input_size)

    pts = [to_model(p0), to_model(p1)]
    dvs = [to_model(d_vect[0]), to_model(d_vect[1])]

    if do_flip:
        pts = [_hflip(p, input_size) for p in pts]
        dvs = [_hflip(p, input_size) for p in dvs]

    if angle_deg:
        c = rotation_center(input_size)
        pts = [_rotate(p, c, c, angle_deg) for p in pts]
        dvs = [_rotate(p, c, c, angle_deg) for p in dvs]

    if scale is not None and scale != 1.0:
        c = rotation_center(input_size)
        pts = [_scale(p, c, c, scale) for p in pts]
        # dvs deliberately not scaled -- see docstring.

    return canonical_order(pts[0], pts[1], (dvs[0], dvs[1]))


def sequential_train_augment(
    p0: Point,
    p1: Point,
    orig_width: float,
    orig_height: float,
    d_vect: tuple[Point, Point],
    input_size: int,
    do_flip: bool,
    angle_deg: float | None,
    scale: float | None,
) -> "SequentialAugmentResult":
    """The actual training-time entry point (transforms.py calls this, not
    `augment_and_canonicalize` directly). Matches
    datasets/landmark_dataset.py's own per-stage accept/reject structure
    EXACTLY, not just its parameter ranges: flip is applied unconditionally
    when drawn (a horizontal mirror of an in-bounds point on a square canvas
    is always still in-bounds, so it needs no bounds check); rotation is
    then tried and REVERTED (keeping the pre-rotation, post-flip state) if
    it would push either keypoint outside the [0, input_size-1] canvas,
    exactly like `datasets/landmark_dataset.py`'s "skip this sample's
    rotation entirely rather than clamp" comment; scale is then tried and
    independently reverted the same way, on top of whatever state rotation
    left behind. This is a sequence of independent accept/reject decisions,
    not one all-or-nothing draw -- e.g. a rejected rotation does not also
    discard an already-accepted flip.

    d_vect is carried through the same accept/reject decisions as the
    keypoints (an accepted stage transforms both; a rejected stage leaves
    both alone), so the final canonical_order() call always projects onto a
    direction consistent with whatever geometric state the keypoints
    actually ended up in.

    Returns a SequentialAugmentResult, not a bare (p0, p1) tuple: the caller
    (transforms.py) must apply the IDENTICAL accept/reject decisions to the
    actual image pixels (rotate/zoom the image only if this function also
    rotated/scaled the keypoints), so `rotation_accepted`/`scale_accepted`
    are part of the return value, not an implementation detail -- silently
    dropping them would let the image and the label disagree about whether
    a given augmentation was actually applied.
    """
    def to_model(pt: Point) -> Point:
        return to_model_space(pt[0], pt[1], orig_width, orig_height, input_size)

    pts = [to_model(p0), to_model(p1)]
    dvs = [to_model(d_vect[0]), to_model(d_vect[1])]

    if do_flip:
        pts = [_hflip(p, input_size) for p in pts]
        dvs = [_hflip(p, input_size) for p in dvs]
        # Flip cannot push an in-bounds point out of bounds on a square
        # canvas -- no accept/reject check needed, matching
        # datasets/landmark_dataset.py (flip has no bounds guard there either).

    return _sequential_augment_in_model_space(pts[0], pts[1], (dvs[0], dvs[1]),
                                               input_size, do_flip=False,
                                               angle_deg=angle_deg, scale=scale,
                                               pre_flip_applied=do_flip)


def _sequential_augment_in_model_space(
    p0: Point,
    p1: Point,
    d_vect_model_space: tuple[Point, Point],
    input_size: int,
    do_flip: bool,
    angle_deg: float | None,
    scale: float | None,
    pre_flip_applied: bool = False,
) -> "SequentialAugmentResult":
    """Core stage logic, operating on points ALREADY in `input_size`-space
    (no to_model_space call here). `sequential_train_augment` (original-space
    inputs) applies flip here as `do_flip` directly; `transforms.py`'s
    `FetalTrainAugment` (which runs AFTER `PixelCentreResize` has already
    resized+flip is not yet applied) also calls this with `do_flip` as the
    real flip draw. `pre_flip_applied` only exists so
    `sequential_train_augment`'s public flip-order-preserving semantics are
    unchanged after this refactor -- see its own body, which folds the flip
    step in before delegating here with `do_flip=False` (already applied)
    but reports the original draw via `pre_flip_applied` for the returned
    `.flip_applied` flag. New callers (transforms.py) should pass their own
    real, not-yet-applied `do_flip` and leave `pre_flip_applied=False`.
    """
    pts = [p0, p1]
    dvs = list(d_vect_model_space)

    if do_flip:
        pts = [_hflip(p, input_size) for p in pts]
        dvs = [_hflip(p, input_size) for p in dvs]
        # Flip cannot push an in-bounds point out of bounds on a square
        # canvas -- no accept/reject check needed, matching
        # datasets/landmark_dataset.py (flip has no bounds guard there either).

    rotation_accepted = False
    if angle_deg:
        c = rotation_center(input_size)
        rot_pts = [_rotate(p, c, c, angle_deg) for p in pts]
        if all(in_bounds(p, input_size) for p in rot_pts):
            pts = rot_pts
            dvs = [_rotate(p, c, c, angle_deg) for p in dvs]
            rotation_accepted = True
        # else: reject rotation only, keep pre-rotation pts/dvs (post-flip).

    scale_accepted = False
    if scale is not None and scale != 1.0:
        c = rotation_center(input_size)
        scaled_pts = [_scale(p, c, c, scale) for p in pts]
        if all(in_bounds(p, input_size) for p in scaled_pts):
            pts = scaled_pts
            scale_accepted = True
            # dvs not scaled -- see augment_and_canonicalize's docstring;
            # scale can never change projection order so dvs's scale-state
            # is irrelevant to the final canonical_order() call below.
        # else: reject scale only, keep pre-scale pts.

    ordered = canonical_order(pts[0], pts[1], (dvs[0], dvs[1]))
    return SequentialAugmentResult(
        p0=ordered[0], p1=ordered[1],
        flip_applied=do_flip or pre_flip_applied,
        rotation_accepted=rotation_accepted,
        scale_accepted=scale_accepted,
    )


class SequentialAugmentResult(tuple):
    """Acts like the (p0, p1) tuple `sequential_train_augment` used to
    return directly (so `result[0]`, `result[1]`, and `p0, p1 = result`
    unpacking all still work, keeping every existing test unchanged), plus
    named `.flip_applied` / `.rotation_accepted` / `.scale_accepted` flags
    transforms.py needs to keep the actual image pixels consistent with
    which augmentations the keypoints/d_vect actually underwent."""

    def __new__(cls, p0, p1, flip_applied, rotation_accepted, scale_accepted):
        obj = super().__new__(cls, (p0, p1))
        obj.flip_applied = flip_applied
        obj.rotation_accepted = rotation_accepted
        obj.scale_accepted = scale_accepted
        return obj
