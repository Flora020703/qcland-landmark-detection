"""Mandatory pre-canary synthetic coordinate test for the pixel-centre
geometry transform (rtmpose_reproduction/PROTOCOL_LOCKED.md, "Endpoint
identity gate" / "A synthetic coordinate test must cover ...").

Pure Python + this repo's geometry.py only -- no MMPose/MMEngine/MMCV
required, so this can (and must) pass before any server GPU time is spent.

Run directly: python rtmpose_reproduction/test_geometry.py
"""

from __future__ import annotations

import itertools
import math

from geometry import to_image_space, to_model_space

TOLERANCE = 1e-6  # input-space pixels; pure float round-trip, no quantisation here


def _check_round_trip(x, y, width, height, input_size=512, label=""):
    xp, yp = to_model_space(x, y, width, height, input_size)
    x2, y2 = to_image_space(xp, yp, width, height, input_size)
    dx, dy = abs(x2 - x), abs(y2 - y)
    assert dx < TOLERANCE and dy < TOLERANCE, (
        f"[{label}] round trip failed for (x={x}, y={y}, W={width}, H={height}): "
        f"recovered ({x2!r}, {y2!r}), delta=({dx!r}, {dy!r})"
    )
    return xp, yp


def test_known_value_sanity():
    # Square image, no resize (W=H=input_size): a pixel-centre-aligned
    # identity transform should map every coordinate to itself exactly.
    for x, y in [(0, 0), (255.5, 100.25), (511, 511)]:
        xp, yp = to_model_space(x, y, 512, 512, input_size=512)
        assert abs(xp - x) < 1e-9 and abs(yp - y) < 1e-9, (
            f"identity case failed: ({x},{y}) -> ({xp},{yp})"
        )

    # A 2x downsize on a square image: pixel-centre formula, not naive
    # proportional scaling -- x=0 must NOT map to x'=0 in general.
    # x'=(0+0.5)*(512/1024)-0.5 = 0.25-0.5 = -0.25 (naive scaling would give 0).
    xp, yp = to_model_space(0, 0, 1024, 1024, input_size=512)
    assert abs(xp - (-0.25)) < 1e-9 and abs(yp - (-0.25)) < 1e-9, (
        f"pixel-centre sanity check failed: got ({xp}, {yp}), expected (-0.25, -0.25)"
    )
    print("[PASS] test_known_value_sanity")


def test_round_trip_diameter_orientations():
    """Horizontal, vertical, near-horizontal, near-vertical endpoint pairs,
    per PROTOCOL_LOCKED.md's explicit orientation list."""
    width, height = 640, 480
    cases = {
        "horizontal": ((100.0, 240.0), (500.0, 240.0)),
        "vertical": ((320.0, 50.0), (320.0, 430.0)),
        "near_horizontal": ((100.0, 238.0), (500.0, 242.0)),
        "near_vertical": ((318.0, 50.0), (322.0, 430.0)),
    }
    for label, (p0, p1) in cases.items():
        for (x, y) in (p0, p1):
            _check_round_trip(x, y, width, height, label=label)
    print("[PASS] test_round_trip_diameter_orientations")


def test_round_trip_corners_and_borders():
    width, height = 800, 600
    corners = [
        (0.0, 0.0), (width - 1.0, 0.0),
        (0.0, height - 1.0), (width - 1.0, height - 1.0),
    ]
    for (x, y) in corners:
        _check_round_trip(x, y, width, height, label="corner")
    print("[PASS] test_round_trip_corners_and_borders")


def test_round_trip_non_square_and_varied_resolutions():
    resolutions = [(640, 480), (768, 1024), (500, 333), (1920, 1080), (333, 331)]
    xs = [0.0, 1.0, 50.5, 123.25]
    ys = [0.0, 1.0, 77.75, 200.0]
    for (width, height) in resolutions:
        for x, y in itertools.product(xs, ys):
            if x >= width or y >= height:
                continue
            _check_round_trip(x, y, width, height, label=f"res{width}x{height}")
    print("[PASS] test_round_trip_non_square_and_varied_resolutions")


def test_horizontal_flip_consistency():
    """Flipping the ORIGINAL image (x -> W-1-x) and then transforming must
    equal transforming and then flipping in MODEL space (x' -> S-1-x'),
    i.e. the pixel-centre convention must commute with the standard
    (dim-1)-minus flip formula used elsewhere in this project. This is a
    geometry-level consistency check; endpoint SEMANTIC identity (which
    channel swaps under flip) is a dataset-converter-level DOD concern
    tested separately, not here.
    """
    width, height = 640, 480
    input_size = 512
    for x, y in [(0.0, 0.0), (100.3, 200.7), (639.0, 0.0), (320.0, 240.0)]:
        xp, yp = to_model_space(x, y, width, height, input_size)

        x_flipped_orig = (width - 1) - x
        xp_via_orig_flip, _ = to_model_space(x_flipped_orig, y, width, height, input_size)

        xp_via_model_flip = (input_size - 1) - xp

        assert abs(xp_via_orig_flip - xp_via_model_flip) < 1e-6, (
            f"flip commutation failed for x={x}: "
            f"flip-then-transform={xp_via_orig_flip!r}, "
            f"transform-then-flip={xp_via_model_flip!r}"
        )
    print("[PASS] test_horizontal_flip_consistency")


def test_naive_scaling_would_have_been_wrong():
    """Documents *why* this module exists: confirms the pixel-centre formula
    is NOT numerically equivalent to naive proportional scaling x*S/W, so a
    future edit that "simplifies" to naive scaling would be a real regression,
    not a harmless refactor."""
    x, width, input_size = 10.0, 333.0, 512
    naive = x * (input_size / width)
    xp, _ = to_model_space(x, 0.0, width, 1.0, input_size)
    assert abs(xp - naive) > 1e-3, (
        "pixel-centre and naive scaling coincided unexpectedly for this "
        "input -- re-check the test case, this should normally differ"
    )
    print("[PASS] test_naive_scaling_would_have_been_wrong (formulas differ, as expected)")


def main():
    test_known_value_sanity()
    test_round_trip_diameter_orientations()
    test_round_trip_corners_and_borders()
    test_round_trip_non_square_and_varied_resolutions()
    test_horizontal_flip_consistency()
    test_naive_scaling_would_have_been_wrong()
    print("[ALL GEOMETRY TESTS PASSED]")


if __name__ == "__main__":
    main()
