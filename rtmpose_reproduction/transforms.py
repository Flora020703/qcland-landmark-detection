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
