"""Tests the pure-Python parts of low_level_decode.py -- `to_original_image_space`
and `fixed_channel_nme` -- which need no MMPose/torch model to exercise.
`decode_batch_low_level` itself needs a live model and is not testable here
(same tier as run_inference.py/internal_val_hook.py); see ENVIRONMENT.md's
checklist / live_preflight.py for its live verification.

Run directly: python rtmpose_reproduction/test_low_level_decode.py
"""

from __future__ import annotations

import numpy as np

from geometry import to_model_space
from low_level_decode import fixed_channel_nme, to_original_image_space


def test_to_original_image_space_matches_geometry_inverse():
    width, height = 800.0, 600.0
    p0, p1 = (123.0, 456.0), (700.0, 88.0)
    model_p0 = to_model_space(*p0, width, height, 512)
    model_p1 = to_model_space(*p1, width, height, 512)
    model_space_coords = np.array([model_p0, model_p1])

    recovered = to_original_image_space(model_space_coords, width, height)
    assert recovered.shape == (2, 2)
    assert abs(recovered[0, 0] - p0[0]) < 1e-6 and abs(recovered[0, 1] - p0[1]) < 1e-6
    assert abs(recovered[1, 0] - p1[0]) < 1e-6 and abs(recovered[1, 1] - p1[1]) < 1e-6
    print("[PASS] test_to_original_image_space_matches_geometry_inverse")


def test_fixed_channel_nme_zero_for_exact_prediction():
    gt = np.array([[100.0, 100.0], [400.0, 500.0]])
    nme = fixed_channel_nme(gt.copy(), gt.copy())
    assert abs(nme) < 1e-9
    print("[PASS] test_fixed_channel_nme_zero_for_exact_prediction")


def test_fixed_channel_nme_matches_hand_computation():
    gt = np.array([[0.0, 0.0], [100.0, 0.0]])   # reference distance = 100
    pred = np.array([[10.0, 0.0], [100.0, 0.0]])  # channel-0 error = 10
    nme = fixed_channel_nme(pred, gt)
    expected = (10.0 + 0.0) / (2.0 * 100.0)  # = 0.05
    assert abs(nme - expected) < 1e-9, (nme, expected)
    print("[PASS] test_fixed_channel_nme_matches_hand_computation")


def main():
    test_to_original_image_space_matches_geometry_inverse()
    test_fixed_channel_nme_zero_for_exact_prediction()
    test_fixed_channel_nme_matches_hand_computation()
    print("[ALL LOW_LEVEL_DECODE TESTS PASSED]")


if __name__ == "__main__":
    main()
