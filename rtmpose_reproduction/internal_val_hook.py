"""Custom MMEngine training Hook that replaces MMPose's default periodic
validation (`model.val_step()` -> `model.predict()` -> `PCKAccuracy`) with
a low-level, decode-only check using the SAME code path as run_inference.py
(`low_level_decode.py`), computing the SAME fixed-channel NME formula used
for the final, authoritative evaluation.

WHY NOT JUST USE PCKAccuracy (review finding, 2026-08-06): the generated
config's val/test pipeline (`LoadImage -> PixelCentreResize -> PackPoseInputs`)
deliberately never populates `bbox_center`/`bbox_scale` in the stock
format PROTOCOL_LOCKED.md requires this project to avoid (see
run_inference.py's own docstring for why). MMPose's default val loop goes
through `model.predict()`, which -- for a stock TopdownPoseEstimator --
typically needs that same bbox metadata to map decoded keypoints back to
"original image" coordinates. Whether this crashes outright, silently uses
some default/zero bbox, or does something else, is UNVERIFIED without a
live MMPose install -- and unlike final inference (run once, after
training, already treated as the highest-risk step), a crash or silently
-wrong number here could happen mid-training (wasting GPU time) or produce
a misleading "it's converging" signal computed via wrong coordinates.

This Hook sidesteps the question entirely by construction: it never calls
`model.predict()`/`val_step()` for its own monitoring, using the identical
low-level decode path already used for final inference instead. This is a
genuinely stronger design than "verify PCKAccuracy works," not just a
workaround -- the internal number this produces is DIRECTLY comparable to
the final authoritative fixed-channel NME (same formula, same code path),
which PCKAccuracy's own OKS/bbox-normalised percentage never was anyway
(see make_config.py's own comment: PCKAccuracy was always "an internal
training-time sanity metric ONLY").

NEEDS LIVE MMPOSE (same tier as run_inference.py) -- the Hook lifecycle
method name/signature (`after_train_epoch`), how to access the model via
`runner.model` (vs `runner.model.module` under DDP -- not needed for a
single-GPU canary but relevant later), and `DATASETS.build()`'s exact
contract for a dataset dict all need live confirmation, same tier as the
rest of this project's MMPose-dependent code.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    from mmengine.hooks import Hook
    from mmengine.registry import HOOKS
except ImportError as exc:  # pragma: no cover - exercised only with mmengine installed
    raise ImportError(
        "internal_val_hook.py requires mmengine to be importable; this "
        "module is not meant to run outside the RTMPose training "
        "environment."
    ) from exc

from low_level_decode import decode_batch_low_level, fixed_channel_nme, to_original_image_space


@HOOKS.register_module()
class InternalFixedChannelNMEHook(Hook):
    """Runs every `interval` epochs (matching `val_interval`), over the
    dataset built from `internal_val_ann`/`images_dir`. Logs the mean
    fixed-channel NME (percent) via `runner.logger.info` and
    `runner.message_hub` (so it appears in the standard training log
    alongside loss, without depending on PCKAccuracy/val_evaluator at all).

    `val_dataloader`/`val_evaluator` in the generated config are both
    `None` (round 6 fix -- MMEngine's Runner requires `val_dataloader`,
    `val_cfg`, `val_evaluator` to be either all `None` or all non-`None`,
    verified against the real Runner source; a `val_cfg=None` alongside
    still-populated `val_dataloader`/`val_evaluator` dicts, this class's
    own round-5 design, would have made Runner construction itself raise
    `ValueError` before a single training step ran). The Train-only
    internal validation split instead lives under the NON-standard config
    key `internal_val_dataloader`, which Runner's own `from_cfg()` never
    reads at all, so it cannot participate in that all-or-nothing check.
    This Hook reads `runner.cfg.internal_val_dataloader["dataset"]`, not
    `runner.cfg.val_dataloader`.

    This Hook is the SOLE source of periodic internal monitoring --
    `val_cfg=None` disables MMEngine's own default val loop, and therefore
    `model.predict()`, entirely during training.
    """

    def __init__(self, internal_val_ann: str, images_dir: str,
                 interval: int = 5, device: str | None = None):
        self.internal_val_ann = internal_val_ann
        self.images_dir = images_dir
        self.interval = interval
        # `device=None` resolves to the model's own actual device at call
        # time (round 6 fix, review request) rather than a hardcoded
        # "cuda:0" -- correct on any single device the model happens to be
        # on, and does not silently break under a future DDP/multi-GPU
        # setup the way a hardcoded string would.
        self.device = device
        self._gt_by_id = None
        self._dataset = None

    def _lazy_build(self, runner):
        if self._dataset is not None:
            return
        from mmpose.registry import DATASETS

        gt = json.loads(Path(self.internal_val_ann).read_text(encoding="utf-8"))
        self._gt_by_id = {im["id"]: im for im in gt["images"]}
        ann_by_image = {}
        for ann in gt["annotations"]:
            ann_by_image[ann["image_id"]] = ann
        self._ann_by_image = ann_by_image

        dataset_cfg = dict(runner.cfg.internal_val_dataloader["dataset"])
        dataset_cfg["ann_file"] = self.internal_val_ann
        self._dataset = DATASETS.build(dataset_cfg)

    def after_train_epoch(self, runner) -> None:
        if (runner.epoch + 1) % self.interval != 0:
            return
        self._lazy_build(runner)

        import torch

        model = runner.model
        device = self.device or str(next(model.parameters()).device)
        was_training = model.training
        model.eval()
        nmes = []
        with torch.no_grad():
            for idx in range(len(self._dataset)):
                data = self._dataset[idx]
                data_sample = data["data_samples"]
                img_id = data_sample.get("img_id")
                image_info = self._gt_by_id[img_id]
                width, height = image_info["width"], image_info["height"]
                ann = self._ann_by_image[img_id]
                gt_kpts = np.asarray(ann["keypoints"], dtype=np.float64).reshape(2, 3)[:, :2]

                model_space_coords = decode_batch_low_level(model, data, device)
                pred = to_original_image_space(model_space_coords, width, height)
                nmes.append(fixed_channel_nme(pred, gt_kpts))
        if was_training:
            model.train()

        mean_nme_pct = float(np.mean(nmes)) * 100.0 if nmes else float("nan")
        runner.logger.info(
            f"[InternalFixedChannelNMEHook] epoch={runner.epoch + 1} "
            f"n={len(nmes)} mean_fixed_channel_nme_pct={mean_nme_pct:.4f}"
        )
        runner.message_hub.update_scalar("train/internal_fixed_channel_nme_pct", mean_nme_pct)
