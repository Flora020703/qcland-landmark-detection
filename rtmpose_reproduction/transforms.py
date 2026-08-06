"""Custom MMPose pipeline transforms implementing the locked, non-aspect-
preserving, pixel-centre geometry (PROTOCOL_LOCKED.md), replacing MMPose's
stock GetBBoxCenterScale + TopdownAffine + RandomFlip + RandomBBoxTransform.

MUST be verified against a real MMPose install before the canary is trusted
(see rtmpose_reproduction/README.md "What still needs a live environment").
This file is written to match the official RTMPose-s pipeline's data
contract as closely as this project could confirm without one, but the
exact `results` dict keys PackPoseInputs expects must be cross-checked
against the installed MMPose version's actual source before the canary run.

Deliberately does NOT set `bbox_center`/`bbox_scale` to any padded/aspect-
fixed value the stock pipeline would use, and does NOT call MMPose's own
bbox-inverse utilities anywhere in this project's inference path -- the
original-image recovery is geometry.to_image_space(), called explicitly by
run_inference.py, never by this transform or by the codec.

Pipeline order (see make_config.py's train_pipeline/val_pipeline):
    LoadImage
 -> FetalRandomFlipAndCanonicalize   (train only, ORIGINAL image size)
 -> PixelCentreResize                (both, resizes to 512x512)
 -> FetalRotateScaleColorJitter      (train only, already-512 space)
 -> GenerateTarget -> PackPoseInputs

The split between the two train-only transforms is NOT arbitrary: channel
identity (which point is "channel 0") can ONLY be decided correctly in
ORIGINAL image-pixel space, using the frozen `d_vect` untouched by any
transform -- see fetal_augment.py's module docstring for the full
derivation (verified against HRNet's actual get_transform/
_transform_pixel_float source, not assumed). Deciding it after
PixelCentreResize's anisotropic resize (this project's own EARLIER,
INCORRECT design) would silently introduce a NEW discrepancy from HRNet's
real behaviour for any non-square source image, rather than reproducing
it. Rotation and scale are proven to have no effect on the ordering
decision, so they run afterward, in already-512-space, purely as position
updates.
"""

from __future__ import annotations

import random

import cv2
import numpy as np

try:
    from mmcv.transforms import BaseTransform
    from mmpose.registry import TRANSFORMS
except ImportError as exc:  # pragma: no cover - exercised only with mmpose installed
    raise ImportError(
        "transforms.py requires mmpose/mmcv to be importable; this module is "
        "not meant to run outside the RTMPose training/inference environment. "
        "geometry.py and fetal_augment.py have no such dependency and are "
        "unit-testable without mmpose (see test_geometry.py, "
        "test_fetal_augment.py)."
    ) from exc

from fetal_augment import resolve_channel_order_after_flip, sequential_rotate_scale
from geometry import to_model_space

INPUT_SIZE = 512


@TRANSFORMS.register_module()
class FetalRandomFlipAndCanonicalize(BaseTransform):
    """Train-only horizontal flip, run on the ORIGINAL (not yet resized)
    image and keypoints, immediately followed by DOD re-canonicalisation
    using the STATIC, un-transformed `d_vect` -- this is what HRNet's own
    per-sample computation reduces to for the channel-identity decision
    (see fetal_augment.py's module docstring for the full derivation and
    empirical verification against HRNet's real transform formula).

    Do NOT move this after PixelCentreResize: an anisotropic resize (this
    project's own non-aspect-preserving 512x512 resize) does not preserve
    the sign of a dot-product-based ordering comparison for a non-square
    source image, so re-deriving the order in already-resized space would
    silently diverge from HRNet's real (original-space) decision.

    Expects `results['img']` (H, W, 3) and `results['keypoints']` shaped
    (1, 2, 2) in ORIGINAL image pixel coordinates.
    """

    def __init__(self, d_vect: tuple, flip_prob: float = 0.5):
        super().__init__()
        self.d_vect = (tuple(d_vect[0]), tuple(d_vect[1]))
        self.flip_prob = flip_prob

    def transform(self, results: dict) -> dict:
        img = results["img"]
        width = img.shape[1]

        keypoints = np.asarray(results["keypoints"], dtype=np.float64)
        p0 = tuple(keypoints[0, 0])
        p1 = tuple(keypoints[0, 1])

        do_flip = random.random() < self.flip_prob
        if do_flip:
            img = cv2.flip(img, 1)

        ordered = resolve_channel_order_after_flip(p0, p1, self.d_vect, do_flip, width)

        keypoints[0, 0] = ordered[0]
        keypoints[0, 1] = ordered[1]
        results["img"] = img
        results["keypoints"] = keypoints
        return results


@TRANSFORMS.register_module()
class PixelCentreResize(BaseTransform):
    """Direct, non-aspect-ratio-preserving resize of the COMPLETE source
    image to `input_size x input_size`, with UDP-inspired pixel-centre
    keypoint scaling (geometry.to_model_space), replacing
    GetBBoxCenterScale + TopdownAffine. Channel identity must already be
    resolved before this runs (see FetalRandomFlipAndCanonicalize) --
    this transform only rescales coordinates, it never reorders them.

    Expects `results['img']` (H, W, 3 uint8) and `results['keypoints']`
    shaped (1, K, 2) in ORIGINAL image pixel coordinates.
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
        # Completed 2026-08-06 (review flagged these as missing): standard
        # mmcv/mmpose metadata fields several downstream consumers (logging,
        # visualisation tools, PackPoseInputs' own metainfo copy) may read.
        # `scale_factor` is (x_ratio, y_ratio), NOT a single scalar, since
        # this resize is deliberately anisotropic for non-square sources.
        results["img_shape"] = (self.input_size, self.input_size)
        results["scale_factor"] = (self.input_size / width, self.input_size / height)
        return results


@TRANSFORMS.register_module()
class FetalRotateScaleColorJitter(BaseTransform):
    """Train-only rotation/scale/colour augmentation, run AFTER
    PixelCentreResize (operates entirely in already-512x512 space) and
    BEFORE GenerateTarget. Channel identity is NOT re-examined here -- it
    was already fixed by FetalRandomFlipAndCanonicalize, and rotation/scale
    are proven to have no effect on which point is channel 0 (see
    fetal_augment.py's module docstring); this transform only repositions
    already-ordered points, with EoMT's own independent per-stage
    accept/reject-if-out-of-canvas policy (a rejected rotation does not
    discard an already-accepted scale, or vice versa).

    Parameter ranges (rotation angle ~ U(-30, 30) at p=0.6, scale ~
    U(0.75, 1.25) unconditionally) match BOTH the audited upstream HRNet
    config (ROT_FACTOR=30, SCALE_FACTOR=0.25, confirmed identical across
    every fetal_landmark_hrnet_w18_*.yaml) and this project's own EoMT
    adapter (datasets/landmark_dataset.py's rotate_augment/scale_augment,
    used by the locked *_dinov2_fpn_udp_rotate_scale.yaml /
    *_dinov3_fpn_udp_rotate_scale.yaml configs) -- the SAME parameters used
    for the final cross-method comparison's common augmentation protocol.

    NOT bit-identical to either reference implementation, disclosed
    precisely (see PROTOCOL_AUDIT.md's augmentation item-by-item table):
    HRNet applies rotation+scale as ONE combined cv2.warpAffine crop on the
    ORIGINAL (un-resized) image; this project (matching EoMT's own
    already-locked, non-retrainable choice) applies them as SEPARATE
    operations on the already-512-resized canvas. Interpolation is
    bilinear in all three implementations (cv2.INTER_LINEAR here and in
    HRNet; PIL.Image.BILINEAR in EoMT) but the exact resampling kernel is
    not guaranteed bit-identical across cv2/PIL. Colour jitter
    (brightness/contrast/saturation, p=0.5 each, matching
    datasets/landmark_dataset.py's own ranges exactly) exists in EoMT but
    has NO counterpart in the audited upstream HRNet code at all (confirmed
    via grep -- no brightness/contrast/colour augmentation anywhere in
    lib/datasets/fetal.py); included here to match EoMT specifically.

    CHANNEL-ORDER CAVEAT for colour jitter (flagged by review, unresolved
    without a live install): MMCV/MMPose's `LoadImage` conventionally
    produces BGR arrays (matching cv2.imread), and `bgr_to_rgb=True` is
    applied by PoseDataPreprocessor at model-forward time, AFTER this
    entire pipeline runs -- meaning `results['img']` here is BGR by
    OpenMMLab convention, not RGB. torchvision's `adjust_saturation`/
    `adjust_contrast` use RGB-specific luminance weights internally,
    so applying them to a BGR array without converting first would compute
    jitter using the WRONG channel-to-luminance mapping (not matching
    EoMT's own RGB-based jitter, which operates on genuinely RGB PIL
    images). This transform explicitly reverses channels before and after
    jitter to compensate -- confirm `results['img']`'s actual channel order
    against the installed LoadImage/pipeline config before trusting this
    (if the pipeline is configured with `color_type='color_ignore_orientation'`
    or a custom RGB-loading LoadImage, this reversal would be WRONG and
    must be removed).

    MUST be verified against a real MMPose install before the canary is
    trusted (same caveat as the other custom transforms).
    """

    def __init__(self, input_size: int = INPUT_SIZE,
                 rotate_prob: float = 0.6, rotate_range: float = 30.0,
                 scale_range: tuple = (0.75, 1.25),
                 assume_bgr: bool = True):
        super().__init__()
        self.input_size = input_size
        self.rotate_prob = rotate_prob
        self.rotate_range = rotate_range
        self.scale_range = scale_range
        self.assume_bgr = assume_bgr

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
        the SAME torchvision.transforms.functional calls EoMT itself uses.
        Converts BGR<->RGB around the jitter if `self.assume_bgr` (see this
        class's own docstring caveat)."""
        import torch
        import torchvision.transforms.functional as TF

        if self.assume_bgr:
            img = img[..., ::-1]

        img_t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
        if random.random() < 0.5:
            img_t = TF.adjust_brightness(img_t, 1.0 + random.uniform(-0.2, 0.2))
        if random.random() < 0.5:
            img_t = TF.adjust_contrast(img_t, 1.0 + random.uniform(-0.2, 0.2))
        if random.random() < 0.5:
            img_t = TF.adjust_saturation(img_t, 1.0 + random.uniform(-0.1, 0.1))
        img_t = img_t.clamp(0.0, 1.0)
        out = (img_t.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)

        if self.assume_bgr:
            out = out[..., ::-1]
        return np.ascontiguousarray(out)

    def transform(self, results: dict) -> dict:
        img = results["img"]
        size = self.input_size
        assert img.shape[0] == size and img.shape[1] == size, (
            f"FetalRotateScaleColorJitter expects a {size}x{size} image "
            f"(run PixelCentreResize first); got {img.shape}"
        )

        keypoints = np.asarray(results["keypoints"], dtype=np.float64)
        p0 = tuple(keypoints[0, 0])
        p1 = tuple(keypoints[0, 1])

        angle = (random.uniform(-self.rotate_range, self.rotate_range)
                  if random.random() < self.rotate_prob else None)
        scale = random.uniform(*self.scale_range)

        result = sequential_rotate_scale(p0, p1, size, angle_deg=angle, scale=scale)

        c = size / 2.0
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
