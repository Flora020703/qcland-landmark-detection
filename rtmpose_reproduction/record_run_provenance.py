"""Records the exact provenance/parameter-count facts PROTOCOL_LOCKED.md's
"Required outputs" list did not previously spell out explicitly as their own
artifact (config name, checkpoint URL, SHA-256, which state_dict keys were
actually loaded vs freshly initialised, and ACTUAL total/trainable/frozen
parameter counts for THIS project's 512x512 two-keypoint config) -- run
once per canary/sweep cell, right after building the model and loading the
pretrained backbone, before training starts.

Do not report the official RTMPose-s paper's ~5.47M-parameter figure (the
COCO 256x192, 17-keypoint config) as this project's own parameter count --
that number is for a DIFFERENT head configuration (out_channels=2 vs 17,
different in_featuremap_size) and was never measured for this adapter's
actual config. This script measures the real, as-built number instead.

NEEDS LIVE MMPOSE (same tier as transforms.py/make_config.py) -- writes a
JSON artifact, not just a printed report, so it becomes part of the
canary's saved outputs per PROTOCOL_LOCKED.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--pretrained-checkpoint-path", type=Path, default=None,
                         help="local path to the already-downloaded CSPNeXt-s "
                              "checkpoint file, if available, for a local "
                              "SHA-256 in addition to the URL recorded in the config")
    args = parser.parse_args()

    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmpose.registry import MODELS

    init_default_scope("mmpose")
    cfg = Config.fromfile(str(args.config))

    model = MODELS.build(cfg.model)

    backbone_cfg = cfg.model["backbone"]
    init_cfg = backbone_cfg.get("init_cfg", {})
    checkpoint_url = init_cfg.get("checkpoint")
    prefix = init_cfg.get("prefix")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    backbone_keys = [k for k, _ in model.backbone.named_parameters()]
    head_keys = [k for k, _ in model.head.named_parameters()]

    record = {
        "official_config_name": "rtmpose-s_8xb256-420e_coco-256x192 (adapted, see make_config.py)",
        "generated_config_path": str(args.config),
        "pretrained_checkpoint_url": checkpoint_url,
        "pretrained_checkpoint_load_prefix": prefix,
        "pretrained_checkpoint_source_dataset": "COCO + AI Challenger (per the official RTMPose-s "
                                                 "checkpoint filename: cspnext-s_udp-aic-coco_...)",
        "pretrained_checkpoint_pretraining_task": "256x192 17-keypoint COCO/AIC human pose estimation "
                                                    "(backbone only reused here; this project's head is "
                                                    "freshly initialised for 2 fetal endpoints at 512x512)",
        "pretrained_checkpoint_local_sha256": (
            _sha256(args.pretrained_checkpoint_path)
            if args.pretrained_checkpoint_path and args.pretrained_checkpoint_path.is_file()
            else None
        ),
        "license_note": "OpenMMLab / MMPose model zoo checkpoints are released under the Apache 2.0 "
                         "license (see the mmpose repository's own LICENSE) -- record the exact MMPose "
                         "commit's LICENSE file alongside this record, do not assume it never changes.",
        "backbone_param_count": sum(p.numel() for p in model.backbone.parameters()),
        "backbone_param_names_sample": backbone_keys[:5] + (["..."] if len(backbone_keys) > 5 else []),
        "head_param_count": sum(p.numel() for p in model.head.parameters()),
        "head_param_names_sample": head_keys[:5] + (["..."] if len(head_keys) > 5 else []),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "frozen_params": frozen_params,
        "note_on_official_5_47m_figure": (
            "The official RTMPose-s paper's ~5.47M-parameter figure is for the "
            "COCO 256x192, out_channels=17 config -- NOT this project's "
            "out_channels=2, 512x512 config. Use total_params above, not that "
            "figure, in any thesis table."
        ),
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
