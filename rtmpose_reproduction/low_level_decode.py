"""Shared low-level (bypass-predict()) decode path, factored out of
run_inference.py so the SAME verified-safe code path can also be used for
periodic internal validation during training (internal_val_hook.py),
instead of depending on MMPose's stock `model.predict()`/`PCKAccuracy`.

WHY THIS MATTERS FOR INTERNAL VALIDATION, NOT JUST FINAL INFERENCE (review
finding, 2026-08-06): the ORIGINAL generated config used MMPose's default
val loop (`model.val_step()` -> `model.predict()`) with `PCKAccuracy` as
the val_evaluator. `model.predict()` -- for a stock TopdownPoseEstimator --
typically maps decoded keypoints back using bbox_center/bbox_scale
metadata that GetBBoxCenterScale/TopdownAffine would normally populate.
This project's own `PixelCentreResize` deliberately does NOT populate that
metadata in the stock format (the same reason `run_inference.py` avoids
`predict()` for final inference, see that file's own docstring). Whether
`model.predict()` would (a) crash outright on missing metadata, (b)
silently use some default/zero bbox and produce meaningless coordinates,
or (c) something else, is UNVERIFIED without a live MMPose install -- and
unlike final inference (which only runs once), a periodic validation
crash could happen mid-training, wasting GPU time, or could pass silently
while logging a meaningless number that looks like real progress.

This module lets internal validation use the EXACT SAME low-level
decode-only path as final inference (bypass `predict()` entirely: forward
pass, codec decode in model-space, explicit `geometry.to_image_space()`
inverse) -- avoiding the ambiguity by construction instead of trusting an
unverified metadata contract, and additionally giving internal validation
numbers that are directly comparable to the final, authoritative
fixed-channel NME (same formula, same code path), not a different metric
(PCK) computed a different way.

NEEDS LIVE MMPOSE (same tier as run_inference.py) to confirm the two
assumptions listed in `decode_batch_low_level`'s own docstring.
"""

from __future__ import annotations

import numpy as np

from geometry import to_image_space


def decode_batch_low_level(model, data: dict, device: str):
    """Runs one sample through `model.data_preprocessor` -> `extract_feat`
    -> `head.forward` -> codec `.decode()`, returning MODEL-SPACE (512x512)
    keypoint coordinates as a (2, 2) numpy array. Does NOT call
    `model.predict()`/`test_step()` and does NOT perform the original-image
    inverse -- callers needing original-image coordinates should follow
    this with `to_original_image_space()` below.

    LIVE-VERIFIED CONTRACT (same as run_inference.py's own path, repeated
    here since this is the shared implementation both files depend on):
      1. `model.data_preprocessor({"inputs": [...], "data_samples": [...]}, False)`
         is the correct call contract for a single manually-collated sample.
      2. In MMPose 1.3.2, `model.head.decode(head_output)` returns one
         prediction InstanceData object per batch element, rather than a
         `(coordinates, scores)` tuple.  Its `keypoints` remain in codec
         input space because this low-level path never calls the estimator's
         bbox-aware `predict()` post-processing.
    """
    data_sample = data["data_samples"]
    batch = model.data_preprocessor(
        {"inputs": [data["inputs"]], "data_samples": [data_sample]},
        False,
    )
    inputs = batch["inputs"].to(device)
    feats = model.extract_feat(inputs)
    head_output = model.head.forward(feats)

    predictions = model.head.decode(head_output)
    if not isinstance(predictions, (list, tuple)) or len(predictions) != 1:
        raise RuntimeError(
            "expected RTMCCHead.decode() to return exactly one prediction "
            f"for the manually constructed one-sample batch, got "
            f"{type(predictions).__name__} with length "
            f"{len(predictions) if hasattr(predictions, '__len__') else 'unknown'}"
        )

    prediction = predictions[0]
    if not hasattr(prediction, "keypoints"):
        raise RuntimeError(
            "RTMCCHead.decode() prediction has no `keypoints` field; the "
            "installed MMPose decode contract differs from the pinned "
            "MMPose 1.3.2 contract"
        )

    model_space_coords = prediction.keypoints
    if hasattr(model_space_coords, "detach"):
        model_space_coords = model_space_coords.detach().cpu().numpy()
    model_space_coords = np.asarray(model_space_coords)
    if model_space_coords.size != 4:
        raise RuntimeError(
            "expected exactly two decoded 2-D keypoints, got array shape "
            f"{model_space_coords.shape}"
        )
    model_space_coords = model_space_coords.reshape(2, 2)
    if not np.isfinite(model_space_coords).all():
        raise RuntimeError("decoded model-space keypoints contain NaN or Inf")
    return model_space_coords


def to_original_image_space(model_space_coords, width: float, height: float,
                             input_size: int = 512):
    """`model_space_coords`: (2, 2) array from `decode_batch_low_level`.
    Returns a (2, 2) array in original-image pixel coordinates via the
    exact inverse pixel-centre transform (geometry.to_image_space), never
    MMPose's own bbox-based inverse."""
    pred0 = to_image_space(model_space_coords[0, 0], model_space_coords[0, 1],
                            width, height, input_size=input_size)
    pred1 = to_image_space(model_space_coords[1, 0], model_space_coords[1, 1],
                            width, height, input_size=input_size)
    return np.asarray([pred0, pred1])


def fixed_channel_nme(pred_orig: np.ndarray, gt_orig: np.ndarray) -> float:
    """Identical formula to evaluate_rtmpose_fixed.py/evaluate_hrnet_fixed.py:
    fixed-channel (no swap-min) NME normalised by the GT endpoint distance.
    `pred_orig`/`gt_orig`: (2, 2) arrays in original-image pixel coordinates."""
    ref = float(np.linalg.norm(gt_orig[0] - gt_orig[1]))
    standard = float(np.linalg.norm(pred_orig[0] - gt_orig[0])
                      + np.linalg.norm(pred_orig[1] - gt_orig[1]))
    return standard / (2.0 * ref)
