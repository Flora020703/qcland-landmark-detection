"""Tests fetal_augment.augment_and_canonicalize -- the fix for the flip-
order bug found by audit_flip_order_stability.py (see fetal_augment.py's
module docstring for the measured failure rates this replaces: 100%
disagreement on UCL OFD/APAD/FL under the old static-flip_indices design).

Central correctness claim, proved algebraically in fetal_augment.py's
review and checked here empirically: transforming the d_vect prototype
points through the IDENTICAL flip/rotate operations as the sample's own
keypoints makes the canonical channel identity invariant to which
augmentation was drawn -- i.e. whichever of the two original points is
"channel 0" without augmentation is STILL "channel 0" (just relocated)
under any flip/rotation/scale combination. This is what makes the training
target well-defined and self-consistent under augmentation, without having
to bit-replicate HRNet's own internal training-time coordinate handling
(out of scope -- HRNet's crop-based single-affine architecture already
differs structurally from this project's separate-transform approach, a
pre-existing, disclosed difference, not something this file re-litigates).

Run directly: python rtmpose_reproduction/test_fetal_augment.py
"""

from __future__ import annotations

import random

from dod_vectors import get_d_vect
from fetal_augment import (
    _sequential_augment_in_model_space,
    augment_and_canonicalize,
    sequential_train_augment,
)
from geometry import to_model_space

TOL = 1e-6


def _close(a, b, tol=TOL):
    return abs(a[0] - b[0]) < tol and abs(a[1] - b[1]) < tol


def test_no_augmentation_matches_plain_canonical_order():
    d_vect = get_d_vect("UCL", "BPD")
    p0, p1 = (565.0, 112.0), (587.0, 464.0)  # real 001_HC.jpg, no-swap case
    width, height = 959, 720
    out = augment_and_canonicalize(p0, p1, width, height, d_vect, input_size=512,
                                    do_flip=False, angle_deg=None, scale=None)
    expected0 = to_model_space(*p0, width, height, 512)
    expected1 = to_model_space(*p1, width, height, 512)
    assert _close(out[0], expected0) and _close(out[1], expected1), out
    print("[PASS] test_no_augmentation_matches_plain_canonical_order")


def test_flip_fixes_the_measured_ofd_bug():
    """Real case from audit_flip_order_stability.py's UCL OFD run:
    002_HC.jpeg, stored channel0=(695,422), channel1=(222,343). Under the
    OLD design (mirror points, keep static d_vect, no reorder), the
    post-flip pair (264,422)/(737,343) was flagged as DOD-inconsistent for
    100% of UCL OFD training images. This test checks the NEW function
    instead: whichever original point is channel 0 without flip must STILL
    be channel 0 (now at its mirrored location) when do_flip=True."""
    d_vect = get_d_vect("UCL", "OFD")
    ch0, ch1 = (695.0, 422.0), (222.0, 343.0)
    width, height = 959, 720

    baseline = augment_and_canonicalize(ch0, ch1, width, height, d_vect, 512,
                                         do_flip=False)
    flipped = augment_and_canonicalize(ch0, ch1, width, height, d_vect, 512,
                                        do_flip=True)

    # The point at baseline's channel-0 slot, mirrored, must be flipped's
    # channel-0 slot -- i.e. flip must not have silently swapped identity.
    baseline_ch0_mirrored = (511.0 - baseline[0][0], baseline[0][1])
    assert _close(flipped[0], baseline_ch0_mirrored), (
        f"flip broke channel identity: baseline={baseline}, flipped={flipped}, "
        f"expected flipped[0]={baseline_ch0_mirrored}"
    )
    print("[PASS] test_flip_fixes_the_measured_ofd_bug")


def test_flip_and_rotation_preserve_channel_identity_property():
    """Property test (not just one fixed example): for random points,
    random real d_vect, and random flip/rotation/scale draws, whichever
    ORIGINAL point wins channel 0 at baseline must still win channel 0
    (i.e. the SAME original point, now relocated) under any augmentation
    combination -- the algebraic invariant this whole fix relies on."""
    rng = random.Random(12345)
    d_vect_options = [get_d_vect("UCL", t) for t in ("BPD", "OFD", "APAD", "TAD", "FL")]
    width, height = 800.0, 600.0
    input_size = 512

    n_checked = 0
    for _ in range(500):
        d_vect = rng.choice(d_vect_options)
        p0 = (rng.uniform(0, width - 1), rng.uniform(0, height - 1))
        p1 = (rng.uniform(0, width - 1), rng.uniform(0, height - 1))
        do_flip = rng.random() < 0.5
        angle = rng.uniform(-30.0, 30.0) if rng.random() < 0.6 else None
        scale = rng.uniform(0.75, 1.25)

        baseline = augment_and_canonicalize(p0, p1, width, height, d_vect, input_size,
                                             do_flip=False, angle_deg=None, scale=None)
        augmented = augment_and_canonicalize(p0, p1, width, height, d_vect, input_size,
                                              do_flip=do_flip, angle_deg=angle, scale=scale)

        # Recompute what channel-0's ORIGINAL (pre-augmentation) point was.
        baseline_p0_is_p0 = _close(baseline[0], to_model_space(*p0, width, height, input_size))
        original_ch0 = p0 if baseline_p0_is_p0 else p1

        # Apply the SAME augmentation directly to that original point by hand
        # (independent re-implementation of the geometry, not calling
        # fetal_augment's own helpers) and check it lands at augmented[0].
        x, y = to_model_space(*original_ch0, width, height, input_size)
        if do_flip:
            x = input_size - 1.0 - x
        if angle:
            import math
            c = input_size / 2.0
            theta = math.radians(angle)
            dx, dy = x - c, y - c
            x = c + dx * math.cos(theta) + dy * math.sin(theta)
            y = c - dx * math.sin(theta) + dy * math.cos(theta)
        if scale is not None and scale != 1.0:
            c = input_size / 2.0
            x = c + (x - c) * scale
            y = c + (y - c) * scale

        assert _close(augmented[0], (x, y), tol=1e-4), (
            f"channel identity changed under augmentation: original_ch0={original_ch0}, "
            f"do_flip={do_flip}, angle={angle}, scale={scale}, "
            f"expected augmented[0]=({x},{y}), got {augmented[0]}"
        )
        n_checked += 1

    assert n_checked == 500
    print(f"[PASS] test_flip_and_rotation_preserve_channel_identity_property ({n_checked} random cases)")


def test_scale_never_changes_dod_order():
    """Isolates the scale-only claim from fetal_augment.py's docstring:
    positive uniform scale about a fixed centre cannot change which of two
    points is channel 0, for any of the 5 real UCL directions."""
    rng = random.Random(999)
    for task in ("BPD", "OFD", "APAD", "TAD", "FL"):
        d_vect = get_d_vect("UCL", task)
        for _ in range(50):
            p0 = (rng.uniform(0, 799), rng.uniform(0, 599))
            p1 = (rng.uniform(0, 799), rng.uniform(0, 599))
            s = rng.uniform(0.5, 2.0)
            base = augment_and_canonicalize(p0, p1, 800, 600, d_vect, 512)
            scaled = augment_and_canonicalize(p0, p1, 800, 600, d_vect, 512, scale=s)
            base_p0_is_p0 = _close(base[0], to_model_space(*p0, 800, 600, 512))
            scaled_p0_is_p0 = _close(scaled[0], to_model_space(*p0, 800, 600, 512),
                                      tol=1e-3) if s == 1.0 else None
            # channel-0 identity (which ORIGINAL point) must match between
            # base and scaled regardless of s -- check via the mirrored
            # baseline-vs-scaled comparison at s's own transformed location.
            c = 512 / 2.0
            expect_x = c + (base[0][0] - c) * s
            expect_y = c + (base[0][1] - c) * s
            assert _close(scaled[0], (expect_x, expect_y), tol=1e-3), (
                f"[{task}] scale changed channel order: base={base}, scaled={scaled}, s={s}"
            )
    print("[PASS] test_scale_never_changes_dod_order")


def test_sequential_matches_monolithic_when_nothing_is_out_of_bounds():
    """For a comfortably-centred pair (far from the canvas edge), rotation
    and scale can never push a keypoint out of [0, 511], so
    sequential_train_augment's per-stage accept/reject must reduce to
    exactly augment_and_canonicalize's all-at-once result."""
    d_vect = get_d_vect("UCL", "TAD")
    p0, p1 = (300.0, 250.0), (500.0, 260.0)  # near image centre, small extent
    width, height = 800.0, 600.0
    for do_flip in (False, True):
        for angle in (None, 10.0, -15.0):
            for scale in (None, 0.9, 1.1):
                mono = augment_and_canonicalize(p0, p1, width, height, d_vect, 512,
                                                 do_flip, angle, scale)
                seq = sequential_train_augment(p0, p1, width, height, d_vect, 512,
                                                do_flip, angle, scale)
                assert _close(mono[0], seq[0]) and _close(mono[1], seq[1]), (
                    f"mismatch: do_flip={do_flip} angle={angle} scale={scale} "
                    f"mono={mono} seq={seq}"
                )
    print("[PASS] test_sequential_matches_monolithic_when_nothing_is_out_of_bounds")


def test_sequential_rejects_out_of_bounds_rotation_independently_of_scale():
    """Rotation preserves distance from the canvas centre, so only a point
    in the square canvas's "corner" region (radius from centre > half the
    canvas side, i.e. outside the inscribed circle) can ever be rotated out
    of bounds -- p0=(500,500) is such a point (radius ~345 from centre 256,
    versus the square's own half-width of 256). A 45-degree rotation must
    be rejected (reverting to the post-flip, pre-rotation state), while an
    in-bounds scale draw on top must still be accepted independently -- i.e.
    one stage's rejection must not discard another accepted stage, matching
    datasets/landmark_dataset.py's independent per-stage skip logic."""
    d_vect = ((0.0, 0.0), (511.0, 0.0))  # purely horizontal synthetic direction
    p0, p1 = (500.0, 500.0), (50.0, 50.0)  # opposite corners
    width = height = 512.0  # identity resize, so model-space == original-space

    seq = sequential_train_augment(p0, p1, width, height, d_vect, 512,
                                    do_flip=False, angle_deg=45.0, scale=0.95)
    # Rotation by 45 degrees around the centre pushes both corner points
    # outside [0, 511] -- must be rejected, leaving pts at their original
    # (post-flip=identity here) positions before scale is tried. The
    # resulting pair (order aside -- canonical_order still picks whichever
    # has the lower horizontal projection as channel 0) must be exactly the
    # 0.95-scaled corners, NOT the 45-degree-rotated ones.
    c = 512 / 2.0
    scaled_p0 = (c + (p0[0] - c) * 0.95, c + (p0[1] - c) * 0.95)
    scaled_p1 = (c + (p1[0] - c) * 0.95, c + (p1[1] - c) * 0.95)
    got = {tuple(round(v, 3) for v in pt) for pt in seq}
    expected = {tuple(round(v, 3) for v in pt) for pt in (scaled_p0, scaled_p1)}
    assert got == expected, (seq, scaled_p0, scaled_p1)
    assert seq.flip_applied is False
    assert seq.rotation_accepted is False, "45-degree rotation from a corner must be rejected"
    assert seq.scale_accepted is True, "0.95 scale must be accepted independently of the rejected rotation"
    print("[PASS] test_sequential_rejects_out_of_bounds_rotation_independently_of_scale")


def test_model_space_core_matches_orig_space_wrapper():
    """transforms.py's FetalTrainAugment (running AFTER PixelCentreResize
    has already resized the sample to 512-space) must call the SAME core
    logic as sequential_train_augment (which takes original-space inputs
    and resizes internally) -- this test pins that the two entry points
    agree, so a future refactor of either one can't silently diverge from
    the other."""
    d_vect = get_d_vect("UCL", "OFD")
    p0, p1 = (695.0, 422.0), (222.0, 343.0)
    width, height = 959.0, 720.0
    input_size = 512
    do_flip, angle, scale = True, 12.0, 1.1

    via_wrapper = sequential_train_augment(p0, p1, width, height, d_vect, input_size,
                                            do_flip, angle, scale)

    p0_model = to_model_space(*p0, width, height, input_size)
    p1_model = to_model_space(*p1, width, height, input_size)
    d0_model = to_model_space(*d_vect[0], width, height, input_size)
    d1_model = to_model_space(*d_vect[1], width, height, input_size)
    via_core = _sequential_augment_in_model_space(
        p0_model, p1_model, (d0_model, d1_model), input_size,
        do_flip=do_flip, angle_deg=angle, scale=scale,
    )

    assert _close(via_wrapper[0], via_core[0]) and _close(via_wrapper[1], via_core[1]), (
        via_wrapper, via_core
    )
    assert via_wrapper.flip_applied == via_core.flip_applied
    assert via_wrapper.rotation_accepted == via_core.rotation_accepted
    assert via_wrapper.scale_accepted == via_core.scale_accepted
    print("[PASS] test_model_space_core_matches_orig_space_wrapper")


def main():
    test_no_augmentation_matches_plain_canonical_order()
    test_flip_fixes_the_measured_ofd_bug()
    test_flip_and_rotation_preserve_channel_identity_property()
    test_scale_never_changes_dod_order()
    test_sequential_matches_monolithic_when_nothing_is_out_of_bounds()
    test_sequential_rejects_out_of_bounds_rotation_independently_of_scale()
    test_model_space_core_matches_orig_space_wrapper()
    print("[ALL FETAL-AUGMENT TESTS PASSED]")


if __name__ == "__main__":
    main()
