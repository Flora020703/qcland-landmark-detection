"""Tests fetal_augment.py's CORRECTED design (second audit round,
2026-08-06) -- see fetal_augment.py's and PROTOCOL_AUDIT.md's module
docstrings for why the FIRST version's design (transform d_vect through
flip+rotation before reprojecting, in already-resized 512-space) was
mathematically wrong, verified directly against HRNet's real
get_transform/_transform_pixel_float source: center/scale/rotation all
CANCEL for HRNet's real ordering decision, which reduces to comparing
flip-mirrored (if applicable) ORIGINAL-space points against the STATIC,
untransformed d_vect.

Central correctness claims tested here:
1. `resolve_channel_order_after_flip` reproduces the real, measured UCL OFD
   flip case correctly (mirrors points, compares against static d_vect --
   this is a DIFFERENT claim from the first version's tests, which checked
   that d_vect ALSO gets mirrored; the corrected version does NOT mirror
   d_vect, by design).
2. `sequential_rotate_scale` never needs to touch d_vect at all: rotation
   and scale cannot change which of two already-ordered points is channel
   0 (proven for scale in the first audit round; proven for rotation in
   the second round via the same "similarity transform preserves dot
   product sign" argument used for HRNet's own formula).

Run directly: python rtmpose_reproduction/test_fetal_augment.py
"""

from __future__ import annotations

import random

from dod_vectors import get_d_vect
from endpoint_order import canonical_order
from fetal_augment import resolve_channel_order_after_flip, sequential_rotate_scale

TOL = 1e-6


def _close(a, b, tol=TOL):
    return abs(a[0] - b[0]) < tol and abs(a[1] - b[1]) < tol


def test_no_flip_matches_plain_canonical_order():
    d_vect = get_d_vect("UCL", "BPD")
    p0, p1 = (565.0, 112.0), (587.0, 464.0)  # real 001_HC.jpg, no-swap case
    width = 959.0
    ordered = resolve_channel_order_after_flip(p0, p1, d_vect, do_flip=False, orig_width=width)
    expected = canonical_order(p0, p1, d_vect)
    assert _close(ordered[0], expected[0]) and _close(ordered[1], expected[1])
    print("[PASS] test_no_flip_matches_plain_canonical_order")


def test_flip_uses_static_d_vect_not_transformed_d_vect():
    """Real case from audit_flip_order_stability.py's UCL OFD run:
    stored channel0=(695,422), channel1=(222,343), image width 959 (typical
    UCL Head image). After flip, the CORRECT (HRNet-real-behaviour-matching)
    channel order is determined by mirroring the points and comparing
    against the STATIC (never mirrored) d_vect -- this is a DIFFERENT,
    verified-correct claim from the first audit round's abandoned design,
    which mirrored d_vect too."""
    d_vect = get_d_vect("UCL", "OFD")
    ch0, ch1 = (695.0, 422.0), (222.0, 343.0)
    width = 959.0

    ordered = resolve_channel_order_after_flip(ch0, ch1, d_vect, do_flip=True, orig_width=width)

    mirrored_ch0 = (width - 1.0 - ch0[0], ch0[1])
    mirrored_ch1 = (width - 1.0 - ch1[0], ch1[1])
    expected = canonical_order(mirrored_ch0, mirrored_ch1, d_vect)  # static d_vect, NOT mirrored

    assert _close(ordered[0], expected[0]) and _close(ordered[1], expected[1]), (ordered, expected)
    print("[PASS] test_flip_uses_static_d_vect_not_transformed_d_vect")


def test_flip_matches_the_measured_audit_script_behaviour():
    """This adapter's design is now DELIBERATELY IDENTICAL to what
    audit_flip_order_stability.py measures (that script's "counterfactual"
    turned out to be HRNet's real behaviour, not a strawman) -- for UCL OFD
    specifically, this means flip WILL flip the stored channel order for
    (as measured) 100% of real training images, matching HRNet's own
    real training-time instability for this task, not a bug to avoid."""
    d_vect = get_d_vect("UCL", "OFD")
    ch0, ch1 = (695.0, 422.0), (222.0, 343.0)
    width = 959.0

    no_flip = resolve_channel_order_after_flip(ch0, ch1, d_vect, do_flip=False, orig_width=width)
    flipped = resolve_channel_order_after_flip(ch0, ch1, d_vect, do_flip=True, orig_width=width)

    no_flip_ch0_is_ch0 = _close(no_flip[0], ch0)
    mirrored_ch0 = (width - 1.0 - ch0[0], ch0[1])
    flipped_ch0_is_mirrored_ch0 = _close(flipped[0], mirrored_ch0)
    # For this real near-horizontal-d_vect case, the channel that wins
    # "channel 0" flips identity across the flip draw (matches the 100%
    # measured rate for UCL OFD) -- i.e. these two booleans must DISAGREE.
    assert no_flip_ch0_is_ch0 != flipped_ch0_is_mirrored_ch0, (
        f"expected UCL OFD's real near-horizontal d_vect to flip channel "
        f"identity under flip (matching the measured 100% rate); "
        f"no_flip={no_flip}, flipped={flipped}"
    )
    print("[PASS] test_flip_matches_the_measured_audit_script_behaviour")


def test_reversed_input_order_gives_same_canonical_result():
    d_vect = get_d_vect("MULTICENTRE", "TAD")
    p0, p1 = (100.0, 400.0), (300.0, 50.0)
    width = 800.0
    a = resolve_channel_order_after_flip(p0, p1, d_vect, do_flip=False, orig_width=width)
    b = resolve_channel_order_after_flip(p1, p0, d_vect, do_flip=False, orig_width=width)
    assert _close(a[0], b[0]) and _close(a[1], b[1])
    print("[PASS] test_reversed_input_order_gives_same_canonical_result")


def test_rotate_scale_never_changes_which_point_is_channel_0():
    """Property test: for a pair already correctly ordered (channel 0 vs
    channel 1 fixed), any combination of rotation/scale (accepted, i.e. not
    rejected for being out-of-bounds) must keep the SAME original point as
    channel 0, just relocated -- proven for scale in the first audit round
    (positive uniform scale preserves projection-order sign) and for
    rotation in the second round (a pure rotation is also a similarity
    transform, so it preserves the sign of any fixed pairwise comparison
    the same way HRNet's own formula does for its ordering decision)."""
    rng = random.Random(2024)
    input_size = 512
    n_checked = 0
    for _ in range(300):
        # Keep points comfortably inside the canvas so small rotations/scales
        # never get rejected for going out-of-bounds (that path is tested
        # separately below) -- this test isolates the "never swaps identity
        # when accepted" claim.
        p0 = (rng.uniform(150, 200), rng.uniform(150, 200))
        p1 = (rng.uniform(300, 360), rng.uniform(300, 360))
        angle = rng.uniform(-30.0, 30.0) if rng.random() < 0.6 else None
        scale = rng.uniform(0.9, 1.1)

        result = sequential_rotate_scale(p0, p1, input_size, angle_deg=angle, scale=scale)

        # p0 was channel 0 before; verify it's STILL channel 0 (relocated).
        c = input_size / 2.0
        import math
        x, y = p0
        if angle:
            theta = math.radians(angle)
            cos_t, sin_t = math.cos(theta), math.sin(theta)
            dx, dy = x - c, y - c
            x = c + dx * cos_t + dy * sin_t
            y = c - dx * sin_t + dy * cos_t
        if scale != 1.0:
            x = c + (x - c) * scale
            y = c + (y - c) * scale

        assert _close(result[0], (x, y), tol=1e-3), (
            f"channel 0 identity changed: p0={p0}, angle={angle}, scale={scale}, "
            f"expected={(x, y)}, got result[0]={result[0]}"
        )
        n_checked += 1
    assert n_checked == 300
    print(f"[PASS] test_rotate_scale_never_changes_which_point_is_channel_0 ({n_checked} cases)")


def test_rotation_rejected_independently_of_scale():
    """A point placed in a canvas 'corner' (radius from centre exceeding
    the inscribed circle) CAN be rotated out of [0, input_size-1] -- must
    be rejected (revert to pre-rotation state), while an in-bounds scale
    draw on top must still be accepted independently."""
    p0, p1 = (500.0, 500.0), (50.0, 50.0)  # opposite corners of a 512 canvas
    result = sequential_rotate_scale(p0, p1, 512, angle_deg=45.0, scale=0.95)

    c = 512 / 2.0
    scaled_p0 = (c + (p0[0] - c) * 0.95, c + (p0[1] - c) * 0.95)
    scaled_p1 = (c + (p1[0] - c) * 0.95, c + (p1[1] - c) * 0.95)
    got = {tuple(round(v, 3) for v in pt) for pt in result}
    expected = {tuple(round(v, 3) for v in pt) for pt in (scaled_p0, scaled_p1)}
    assert got == expected, (result, scaled_p0, scaled_p1)
    assert result.rotation_accepted is False, "45-degree rotation from a corner must be rejected"
    assert result.scale_accepted is True, "0.95 scale must be accepted independently of the rejected rotation"
    print("[PASS] test_rotation_rejected_independently_of_scale")


def main():
    test_no_flip_matches_plain_canonical_order()
    test_flip_uses_static_d_vect_not_transformed_d_vect()
    test_flip_matches_the_measured_audit_script_behaviour()
    test_reversed_input_order_gives_same_canonical_result()
    test_rotate_scale_never_changes_which_point_is_channel_0()
    test_rotation_rejected_independently_of_scale()
    print("[ALL FETAL-AUGMENT TESTS PASSED]")


if __name__ == "__main__":
    main()
