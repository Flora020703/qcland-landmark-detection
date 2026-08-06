"""Runs a trained RTMPose checkpoint on a COCO-format test set and exports
predictions in ORIGINAL image coordinates, for evaluate_rtmpose_fixed.py.

*** HIGHEST-RISK FILE IN THIS ADAPTATION -- READ BEFORE TRUSTING ANY NUMBER
IT PRODUCES ***

MMPose's high-level convenience APIs (`mmpose.apis.inference_topdown`,
`model.test_step()` via the standard test loop) are DELIBERATELY NOT used
for the FINAL coordinate mapping. Those APIs are written for the stock
top-down pipeline and typically map a model's decoded keypoints back to
"original image" coordinates using the SAME bbox_center/bbox_scale-based
inverse transform that GetBBoxCenterScale/TopdownAffine populated on the
way in -- exactly the padded, aspect-ratio-preserving convention
PROTOCOL_LOCKED.md requires this project to avoid. Because this project's
own PixelCentreResize transform does not populate `bbox_center`/
`bbox_scale` in the stock format, it is unverified (without a live MMPose
install) whether the high-level API would (a) error out loudly, (b)
silently fall back to some default region, or (c) do something else
entirely -- none of which is safe to assume.

Instead, this script decodes each sample as low-level as MMPose's public API
allows (`low_level_decode.decode_batch_low_level`, shared with
`internal_val_hook.py`'s periodic training-time monitoring so both use the
identical, verified-safe code path): run the backbone+head forward pass
directly, call the SimCC codec's own `decode()` method (which operates
purely in the codec's own `input_size` space, with NO bbox/original-image
concept at all), and then perform the ORIGINAL-image recovery exclusively
via geometry.to_image_space() -- the same function test_geometry.py
already verifies exhaustively. This avoids the ambiguity above by
construction, provided the assumptions below hold.

CORRECTED 2026-08-06 (review finding, must-fix, blocking): the FIRST
version of this script called `model.extract_feat(inputs)` directly on the
raw tensor produced by the dataset's own pipeline, completely bypassing
`model.data_preprocessor`. Fixed by routing through the preprocessor
first, still stopping short of `model.test_step()`/`predict()`.

ASSUMPTIONS THAT MUST BE CONFIRMED AGAINST THE ACTUALLY-INSTALLED MMPOSE
VERSION BEFORE THE CANARY IS TRUSTED (this project has no live MMPose
environment to check this from) -- see low_level_decode.py's own
docstring for the two central assumptions; additionally for this script:
  1. The test dataloader built from make_config.py's `val_pipeline`
     (LoadImage -> PixelCentreResize -> PackPoseInputs) feeds this script
     images resized exactly the way PixelCentreResize computed them.
  2. `ori_shape`/`img_id` (or an equivalent identifier) survives being
     packed into `data_sample` so predictions can be matched back to the
     correct filename.
Do not proceed past the canary if any assumption does not hold; fix this
script (not evaluate_rtmpose_fixed.py, which is generic and already
unit-tested) and re-derive from the codec/model source actually installed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from low_level_decode import decode_batch_low_level, to_original_image_space


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--gt-json", required=True, type=Path,
                         help="the same COCO json used as this config's test ann_file")
    parser.add_argument("--out-predictions-json", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    import torch
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmengine.runner import load_checkpoint
    from mmpose.registry import MODELS, DATASETS

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

            model_space_coords = decode_batch_low_level(model, data, args.device)
            pred = to_original_image_space(model_space_coords, width, height)

            predictions.append({
                "file_name": image_info["file_name"],
                "pred": [list(pred[0]), list(pred[1])],
            })

    args.out_predictions_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_predictions_json.write_text(json.dumps(predictions), encoding="utf-8")
    print(f"[OK] wrote {len(predictions)} predictions to {args.out_predictions_json}")


if __name__ == "__main__":
    main()
