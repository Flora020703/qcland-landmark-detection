"""Custom MMPose pipeline transform implementing the locked, non-aspect-
preserving, pixel-centre geometry (PROTOCOL_LOCKED.md), replacing MMPose's
stock GetBBoxCenterScale + TopdownAffine pair.

MUST be verified against a real MMPose install before the canary is trusted
(see rtmpose_reproduction/README.md "What still needs a live environment").
This file is written to match the official RTMPose-s pipeline's data
contract as closely as this project could confirm without one (verified via
the official rtmpose-s_8xb256-420e_coco-256x192.py config and MMPose's
PackPoseInputs/BaseTransform docs), but the exact `results` dict keys
PackPoseInputs expects (`img`, `keypoints`, `keypoints_visible`,
`img_shape`, plus whatever `input_size`/`input_center`/`input_scale`
bookkeeping downstream visualisation or the stock decode path might read)
must be cross-checked against the installed MMPose version's actual
BaseTransform/PackPoseInputs source before the canary run, not assumed
correct from documentation alone.

Deliberately does NOT set `bbox_center`/`bbox_scale` to any padded/aspect-
fixed value the stock pipeline would use, and does NOT call MMPose's own
bbox-inverse utilities anywhere in this project's inference path -- the
original-image recovery is geometry.to_image_space(), called explicitly by
run_inference.py, never by this transform or by the codec.
"""

from __future__ import annotations

import cv2
import numpy as np

try:
    from mmcv.transforms import BaseTransform
    from mmpose.registry import TRANSFORMS
except ImportError as exc:  # pragma: no cover - exercised only with mmpose installed
    raise ImportError(
        "transforms.py requires mmpose/mmcv to be importable; this module is "
        "not meant to run outside the RTMPose training/inference environment. "
        "geometry.py itself has no such dependency and is unit-testable "
        "without mmpose (see test_geometry.py)."
    ) from exc

import random

from fetal_augment import _sequential_augment_in_model_space
from geometry import to_model_space

INPUT_SIZE = 512


@TRANSFORMS.register_module()
class PixelCentreResize(BaseTransform):
    """Direct, non-aspect-ratio-preserving resize of the COMPLETE source
    image to `input_size x input_size`, with UDP-inspired pixel-centre
    keypoint scaling (geometry.to_model_space), replacing
    GetBBoxCenterScale + TopdownAffine.

    Expects `results['img']` (H, W, 3 uint8, RGB or BGR per the data
    pipeline's own convention -- this transform does not touch colour
    channels) and `results['keypoints']` shaped (1, K, 2) in ORIGINAL image
    pixel coordinates (the standard mmpose convention: one instance per
    sample under the top-down pipeline), consistent with the COCO-format
    keypoints written by convert_csv_to_coco.py.
    """

    def __init__(self, input_size: int = INPUT_SIZE):
        super().__init__()
        self.input_size = input_size

    def transform(self, results: dict) -> dict:
        img = results["img"]
        height, width = img.shape[:2]

        resized = cv2.resize(img, (self.input_size, self.input_size),
                              interpolation=cv2.INTER_LINEAR)

        keypoints = np.asarray(results["keypoints"], dtype=np.float64).copy()
        # keypoints shape: (num_instances, num_keypoints, 2); this project
        # always has exactly one instance (the whole image) and 2 keypoints.
        for i in range(keypoints.shape[0]):
            for k in range(keypoints.shape[1]):
                x, y = keypoints[i, k]
                xp, yp = to_model_space(x, y, width, height, self.input_size)
                keypoints[i, k] = (xp, yp)

        results["img"] = resized
        results["keypoints"] = keypoints
        results["input_size"] = (self.input_size, self.input_size)
        # Record enough of the original geometry for run_inference.py to call
        # geometry.to_image_space() later without re-deriving width/height
        # from anything MMPose itself computed (e.g. its own bbox_scale,
        # which this transform deliberately never sets to a meaningful
        # padded/aspect-fixed value).
        results["ori_shape"] = (height, width)
        return results


@TRANSFORMS.register_module()
class FetalTrainAugment(BaseTransform):
    """Train-only flip/rotate/scale/colour augmentation, run AFTER
    PixelCentreResize (operates entirely in already-512x512 space) and
    BEFORE GenerateTarget. Replaces MMPose's stock RandomFlip +
    RandomBBoxTransform entirely -- do not add either of those to the
    pipeline alongside this transform.

    WHY A CUSTOM TRANSFORM, NOT MMPose's OWN RandomFlip/RandomBBoxTransform
    (audited 2026-08-06, see fetal_augment.py's module docstring and
    audit_flip_order_stability.py's measured numbers): MMPose's stock
    RandomFlip only mirrors coordinates according to a caller-supplied
    static `flip_indices`; it has no notion of this project's DOD direction
    vector, so a static flip_indices setting is CORRECT for tasks whose
    d_vect is predominantly vertical (e.g. UCL BPD, measured 0.0% affected)
    and WRONG for essentially every sample of a task whose d_vect is
    predominantly horizontal (UCL OFD/APAD/FL, each measured 100.0%
    affected under the old design). This transform instead re-derives the
    DOD-canonical channel order after every accepted geometric transform via
    fetal_augment._sequential_augment_in_model_space, exactly mirroring the
    audited upstream HRNet's own per-sample re-projection architecture
    (lib/datasets/fetal.py lines 249-289).

    Parameter ranges (flip p=0.5, rotation angle ~ U(-30, 30) at p=0.6,
    scale ~ U(0.75, 1.25) unconditionally) match BOTH the audited upstream
    HRNet config (ROT_FACTOR=30, SCALE_FACTOR=0.25, confirmed identical
    across every fetal_landmark_hrnet_w18_*.yaml) and this project's own
    EoMT adapter (datasets/landmark_dataset.py's rotate_augment/
    scale_augment, used by the locked *_dinov2_fpn_udp_rotate_scale.yaml /
    *_dinov3_fpn_udp_rotate_scale.yaml configs) -- this is the SAME
    parameterisation used for the final cross-method comparison's common
    augmentation protocol, not an independently chosen RTMPose-specific
    recipe.

    NOT bit-identical to either reference implementation, disclosed
    precisely (see PROTOCOL_AUDIT.md's augmentation item-by-item table):
    HRNet applies rotation+scale as ONE combined cv2.warpAffine crop
    operating on the ORIGINAL (un-resized) image; this project (matching
    EoMT's own already-locked, non-retrainable choice) applies them as
    SEPARATE operations on the ALREADY-512-resized canvas, each
    independently skipped (not clamped) if it would push a keypoint out of
    [0, 511]. Interpolation is bilinear in all three implementations
    (cv2.INTER_LINEAR here and in HRNet; PIL.Image.BILINEAR in EoMT) but the
    exact resampling kernel is not guaranteed bit-identical across
    cv2/PIL. Colour jitter (brightness/contrast/saturation, p=0.5 each,
    matching datasets/landmark_dataset.py's own ranges exactly) exists in
    EoMT but has NO counterpart in the audited upstream HRNet code at all
    (confirmed via grep -- no brightness/contrast/colour augmentation
    anywhere in lib/datasets/fetal.py); this transform includes it to match
    EoMT specifically, since EoMT is this project's other already-locked
    comparison point.

    MUST be verified against a real MMPose install before the canary is
    trusted (same caveat as PixelCentreResize) -- in particular, confirm
    `results['keypoints']` really is (1, 2, 2) float64 in 512-space at this
    point in the pipeline (i.e. that PixelCentreResize ran immediately
    before this transform, and GenerateTarget's own keypoint-visibility/
    weighting expectations are not violated by this transform's output).
    """

    def __init__(self, d_vect: tuple, input_size: int = INPUT_SIZE,
                 flip_prob: float = 0.5, rotate_prob: float = 0.6,
                 rotate_range: float = 30.0,
                 scale_range: tuple = (0.75, 1.25)):
        super().__init__()
        self.d_vect = (tuple(d_vect[0]), tuple(d_vect[1]))
        self.input_size = input_size
        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob
        self.rotate_range = rotate_range
        self.scale_range = scale_range

    def _rotation_matrix(self, angle_deg: float, c: float):
        theta = np.radians(angle_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        # Matches fetal_augment._rotate's forward mapping exactly:
        # new = c + R @ (old - c), with R's sin-sign flipped for the
        # y-grows-downward image convention (see fetal_augment.py docstring).
        return np.array([
            [cos_t, sin_t, c - c * cos_t - c * sin_t],
            [-sin_t, cos_t, c + c * sin_t - c * cos_t],
        ], dtype=np.float64)

    def _scale_matrix(self, s: float, c: float):
        return np.array([
            [s, 0.0, c * (1.0 - s)],
            [0.0, s, c * (1.0 - s)],
        ], dtype=np.float64)

    def _color_jitter(self, img: np.ndarray) -> np.ndarray:
        """Matches datasets/landmark_dataset.py's colour-jitter block
        exactly (brightness/contrast/saturation, U(-0.2,0.2)/U(-0.2,0.2)/
        U(-0.1,0.1), each independently applied at p=0.5), by delegating to
        the SAME torchvision.transforms.functional calls EoMT itself uses,
        rather than a hand-rolled cv2/numpy approximation that could subtly
        diverge in formula (e.g. contrast's grayscale-mean definition)."""
        import torch
        import torchvision.transforms.functional as TF

        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        if random.random() < 0.5:
            img_t = TF.adjust_brightness(img_t, 1.0 + random.uniform(-0.2, 0.2))
        if random.random() < 0.5:
            img_t = TF.adjust_contrast(img_t, 1.0 + random.uniform(-0.2, 0.2))
        if random.random() < 0.5:
            img_t = TF.adjust_saturation(img_t, 1.0 + random.uniform(-0.1, 0.1))
        img_t = img_t.clamp(0.0, 1.0)
        return (img_t.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)

    def transform(self, results: dict) -> dict:
        img = results["img"]
        size = self.input_size
        assert img.shape[0] == size and img.shape[1] == size, (
            f"FetalTrainAugment expects a {size}x{size} image (run "
            f"PixelCentreResize first); got {img.shape}"
        )
        orig_height, orig_width = results["ori_shape"]

        keypoints = np.asarray(results["keypoints"], dtype=np.float64)
        p0 = tuple(keypoints[0, 0])
        p1 = tuple(keypoints[0, 1])
        d0_model = to_model_space(*self.d_vect[0], orig_width, orig_height, size)
        d1_model = to_model_space(*self.d_vect[1], orig_width, orig_height, size)

        do_flip = random.random() < self.flip_prob
        angle = (random.uniform(-self.rotate_range, self.rotate_range)
                  if random.random() < self.rotate_prob else None)
        scale = random.uniform(*self.scale_range)

        result = _sequential_augment_in_model_space(
            p0, p1, (d0_model, d1_model), size,
            do_flip=do_flip, angle_deg=angle, scale=scale,
        )

        c = size / 2.0
        if result.flip_applied:
            img = cv2.flip(img, 1)
        if result.rotation_accepted:
            M = self._rotation_matrix(angle, c)
            img = cv2.warpAffine(img, M, (size, size), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        if result.scale_accepted:
            M = self._scale_matrix(scale, c)
            img = cv2.warpAffine(img, M, (size, size), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        img = self._color_jitter(img)

        keypoints[0, 0] = result[0]
        keypoints[0, 1] = result[1]
        results["img"] = img
        results["keypoints"] = keypoints
        return results
