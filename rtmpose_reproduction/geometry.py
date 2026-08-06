"""Pixel-centre affine transform shared by the RTMPose adapter, its dataset
converter, its evaluator, and its synthetic tests.

Implements exactly the UDP-inspired convention already locked for EoMT
(training/landmark_detection.py's encoding step; thesis Chapter 3
sec:method-udp, Cref{eq:udp-scale}) and reused here per
rtmpose_reproduction/PROTOCOL_LOCKED.md so that RTMPose's input geometry
matches the existing EoMT/HRNet comparison pipeline: a direct,
non-aspect-ratio-preserving resize of the complete source image to a square
INPUT_SIZE, with pixel-centre-aligned coordinate scaling.

This is deliberately NOT MMPose's default GetBBoxCenterScale/TopdownAffine
convention (1.25x bbox padding, then the shorter bbox side expanded to match
the target aspect ratio before warping) -- that convention preserves aspect
ratio and pads outside the image for a non-square source, which is a
different geometry than what EoMT/HRNet use. Do not import or call MMPose's
codec-paired bbox inverse transform for the 512-space -> original-image step;
use to_image_space() below instead.

No MMPose/MMEngine import here: this module is pure Python + no third-party
dependency beyond nothing at all, so it can be (and is) exercised by
test_geometry.py without a working MMPose environment.
"""

from __future__ import annotations

INPUT_SIZE = 512


def to_model_space(x: float, y: float, width: float, height: float,
                    input_size: int = INPUT_SIZE) -> tuple[float, float]:
    """Original-image pixel coordinate -> square model-input-space coordinate.

    x' = (x + 0.5) * (input_size / width)  - 0.5
    y' = (y + 0.5) * (input_size / height) - 0.5

    width/height are the ORIGINAL source image's pixel dimensions (not the
    input_size), matching the direct, non-aspect-ratio-preserving resize
    locked in PROTOCOL_LOCKED.md.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"width/height must be positive, got {width}x{height}")
    sx = input_size / width
    sy = input_size / height
    xp = (x + 0.5) * sx - 0.5
    yp = (y + 0.5) * sy - 0.5
    return xp, yp


def to_image_space(xp: float, yp: float, width: float, height: float,
                    input_size: int = INPUT_SIZE) -> tuple[float, float]:
    """Model-input-space coordinate -> original-image pixel coordinate.

    Exact algebraic inverse of to_model_space (verified by hand and by
    test_geometry.py's round-trip assertions, not merely "close enough"):

        x = (x' + 0.5) / (input_size / width)  - 0.5
        y = (y' + 0.5) / (input_size / height) - 0.5
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"width/height must be positive, got {width}x{height}")
    sx = input_size / width
    sy = input_size / height
    x = (xp + 0.5) / sx - 0.5
    y = (yp + 0.5) / sy - 0.5
    return x, y
