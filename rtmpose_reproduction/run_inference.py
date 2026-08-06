"""Runs a trained RTMPose checkpoint on a COCO-format test set and exports
predictions in ORIGINAL image coordinates, for evaluate_rtmpose_fixed.py.

*** HIGHEST-RISK FILE IN THIS ADAPTATION -- READ BEFORE TRUSTING ANY NUMBER
IT PRODUCES ***

MMPose's high-level convenience APIs (`mmpose.apis.inference_topdown`,
`model.test_step()` via the standard test loop) are DELIBERATELY NOT used
here. Those APIs are written for the stock top-down pipeline and typically
map a model's decoded keypoints back to "original image" coordinates using
the SAME bbox_center/bbox_scale-based inverse transform that
GetBBoxCenterScale/TopdownAffine populated on the way in -- exactly the
padded, aspect-ratio-preserving convention PROTOCOL_LOCKED.md requires this
project to avoid. Because this project's own PixelCentreResize transform
does not populate `bbox_center`/`bbox_scale` in the stock format, it is
unverified (without a live MMPose install) whether the high-level API would
(a) error out loudly, (b) silently fall back to some default region, or
(c) do something else entirely -- none of which is safe to assume.

Instead, this script decodes each sample as low-level as MMPose's public API
allows: run the backbone+head forward pass directly, call the SimCC codec's
own `decode()` method (which operates purely in the codec's own
`input_size` == 512x512 space, with NO bbox/original-image concept at all),
and then perform the ORIGINAL-image recovery exclusively via
geometry.to_image_space() -- the same function test_geometry.py already
verifies exhaustively. This avoids the ambiguity above by construction,
provided the assumptions below hold.

ASSUMPTIONS THAT MUST BE CONFIRMED AGAINST THE ACTUALLY-INSTALLED MMPOSE
VERSION BEFORE THE CANARY IS TRUSTED (this project has no live MMPose
environment to check this from):
  1. `model.head.decode(head_output)` (or the codec's own `.decode()`
     called on the head's raw simcc_x/simcc_y outputs) returns keypoint
     coordinates in the codec's `input_size` space (512x512), not already
     mapped to any other space.
  2. The val/test dataloader built from make_config.py's `val_pipeline`
     (LoadImage -> PixelCentreResize -> PackPoseInputs) feeds the model
     images resized exactly the way PixelCentreResize computed them, with
     `data_sample.gt_instances.keypoints` (if used for any sanity check)
     also already in that same 512-space, not re-transformed by
     PackPoseInputs.
  3. `ori_shape`/`img_id` (or an equivalent identifier) survives being
     packed into `data_sample` so predictions can be matched back to the
     correct filename.
Do not proceed past the canary if any assumption above does not hold; fix
this script (not evaluate_rtmpose_fixed.py, which is generic and already
unit-tested) and re-derive from the codec/model source actually installed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from geometry import to_image_space


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--gt-json", required=True, type=Path,
                         help="the same COCO json used as this config's test ann_file")
    parser.add_argument("--out-predictions-json", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmengine.runner import load_checkpoint
    from mmpose.registry import MODELS, DATASETS
    from mmpose.structures import merge_data_samples  # noqa: F401 (kept for future use)

    init_default_scope("mmpose")
    cfg = Config.fromfile(str(args.config))

    model = MODELS.build(cfg.model)
    load_checkpoint(model, str(args.checkpoint), map_location="cpu")
    model.to(args.device)
    model.eval()

    gt = json.loads(args.gt_json.read_text(encoding="utf-8"))
    images_by_id = {im["id"]: im for im in gt["images"]}

    dataset_cfg = cfg.test_dataloader["dataset"]
    dataset = DATASETS.build(dataset_cfg)

    predictions = []
    with torch.no_grad():
        for idx in range(len(dataset)):
            data = dataset[idx]
            data_sample = data["data_samples"]
            img_id = data_sample.get("img_id")
            image_info = images_by_id[img_id]
            width, height = image_info["width"], image_info["height"]

            inputs = data["inputs"].unsqueeze(0).float().to(args.device)
            feats = model.extract_feat(inputs)
            head_output = model.head.forward(feats)  # (simcc_x, simcc_y) or similar
            # NOTE (must confirm against the installed mmpose version): some
            # RTMCCHead releases expose `.decode(head_output)` directly on
            # the head; others require calling the codec's own `.decode()`
            # with the head's raw outputs. Both return 512-space coordinates
            # ONLY -- neither should be given this sample's bbox metadata,
            # because none was ever set to a meaningful value.
            if hasattr(model.head, "decode"):
                model_space_coords, _scores = model.head.decode(head_output)
            else:
                codec = cfg.codec
                from mmpose.codecs import build_codec  # type: ignore
                model_space_coords, _scores = build_codec(codec).decode(*head_output)

            model_space_coords = np.asarray(model_space_coords).reshape(2, 2)

            pred0 = to_image_space(model_space_coords[0, 0], model_space_coords[0, 1],
                                    width, height, input_size=512)
            pred1 = to_image_space(model_space_coords[1, 0], model_space_coords[1, 1],
                                    width, height, input_size=512)

            predictions.append({
                "file_name": image_info["file_name"],
                "pred": [list(pred0), list(pred1)],
            })

    args.out_predictions_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_predictions_json.write_text(json.dumps(predictions), encoding="utf-8")
    print(f"[OK] wrote {len(predictions)} predictions to {args.out_predictions_json}")


if __name__ == "__main__":
    main()
