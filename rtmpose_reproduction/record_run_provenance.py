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

CORRECTED 2026-08-06 (review finding): the first version of this script
only called `MODELS.build(cfg.model)` and assumed the backbone's
`init_cfg=dict(type='Pretrained', ...)` had already been applied -- but in
MMEngine's convention, `init_cfg` only DECLARES how to initialise; the
actual weight loading happens when `model.init_weights()` (or the
Runner's own setup, which calls it) actually runs. Simply building the
model does NOT guarantee the pretrained checkpoint was loaded at all, and
the first version never checked for missing/unexpected keys despite
documenting that it does. Fixed below: `model.init_weights()` is called
explicitly, AND the checkpoint's own state dict is independently loaded
and diffed against `model.backbone.state_dict()` by key name (not by
trusting MMEngine's internal logging), so "backbone keys were loaded" is a
verified fact, not an assumption. `--pretrained-checkpoint-path` is now a
REQUIRED argument (previously optional, defaulting to a null SHA-256 in
practice because run_rtmpose_canary.sh never passed it).

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
    parser.add_argument(
        "--pretrained-checkpoint-path", required=True, type=Path,
        help="local path to the already-downloaded CSPNeXt-s checkpoint file "
             "-- REQUIRED (not optional): without it this script cannot verify "
             "which keys actually loaded, only assume they did.",
    )
    args = parser.parse_args()

    if not args.pretrained_checkpoint_path.is_file():
        raise SystemExit(
            f"ERROR: --pretrained-checkpoint-path does not exist: "
            f"{args.pretrained_checkpoint_path}. Download it first (see "
            f"ENVIRONMENT.md) -- refusing to record provenance without the "
            f"actual file to hash and diff against."
        )

    import torch
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

    # Snapshot backbone weights BEFORE init_weights(), so we can also confirm
    # by direct comparison that values actually changed (not just that keys
    # matched by name) -- catches the case where init_weights() silently
    # no-ops but key names still happen to line up.
    pre_init_backbone_state = {k: v.detach().clone()
                                for k, v in model.backbone.state_dict().items()}

    model.init_weights()

    post_init_backbone_state = model.backbone.state_dict()

    # Independently load the checkpoint's own state dict and diff it against
    # the backbone's parameter names (after stripping the configured
    # `prefix`), rather than trusting MMEngine's internal load-time logging.
    raw_ckpt = torch.load(str(args.pretrained_checkpoint_path), map_location="cpu")
    ckpt_state = raw_ckpt.get("state_dict", raw_ckpt)
    if prefix:
        ckpt_backbone_keys = {
            k[len(prefix):] for k in ckpt_state if k.startswith(prefix)
        }
    else:
        ckpt_backbone_keys = set(ckpt_state.keys())

    backbone_param_keys = set(post_init_backbone_state.keys())
    loaded_keys = sorted(backbone_param_keys & ckpt_backbone_keys)
    missing_from_checkpoint = sorted(backbone_param_keys - ckpt_backbone_keys)
    unexpected_in_checkpoint = sorted(ckpt_backbone_keys - backbone_param_keys)

    # Confirm at least the loaded keys' VALUES actually changed from their
    # pre-init_weights() (randomly-initialised) state -- a name-only match
    # with unchanged values would mean init_weights() silently did nothing.
    unchanged_despite_match = [
        k for k in loaded_keys
        if torch.equal(pre_init_backbone_state[k], post_init_backbone_state[k])
    ]

    if not loaded_keys:
        raise SystemExit(
            "ERROR: zero backbone parameter names matched between the model "
            "and the checkpoint (after stripping prefix "
            f"{prefix!r}) -- pretrained loading did not happen. Do not "
            "proceed to training with an unverified/failed backbone init."
        )
    if len(unchanged_despite_match) == len(loaded_keys):
        raise SystemExit(
            "ERROR: every matched backbone key's VALUE is unchanged from its "
            "pre-init_weights() state -- names matched but init_weights() "
            "did not actually load the checkpoint. Do not proceed to "
            "training with an unverified/failed backbone init."
        )

    head_param_keys = {k for k, _ in model.head.named_parameters()}
    head_keys_present_in_checkpoint = sorted(
        (head_param_keys & ckpt_backbone_keys) if not prefix else set()
    )
    # (head keys are never expected to match backbone-prefixed checkpoint
    # keys when a prefix is used; this check is mainly meaningful when
    # prefix is empty/None, kept for completeness.)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    record = {
        "official_config_name": "rtmpose-s_8xb256-420e_coco-256x192 (adapted, see make_config.py)",
        "generated_config_path": str(args.config),
        "pretrained_checkpoint_url": checkpoint_url,
        "pretrained_checkpoint_local_path": str(args.pretrained_checkpoint_path),
        "pretrained_checkpoint_load_prefix": prefix,
        "pretrained_checkpoint_source_dataset": "COCO + AI Challenger (per the official RTMPose-s "
                                                 "checkpoint filename: cspnext-s_udp-aic-coco_...)",
        "pretrained_checkpoint_pretraining_task": "256x192 17-keypoint COCO/AIC human pose estimation "
                                                    "(backbone only reused here; this project's head is "
                                                    "freshly initialised for 2 fetal endpoints at 512x512)",
        "pretrained_checkpoint_local_sha256": _sha256(args.pretrained_checkpoint_path),
        "license_note": "OpenMMLab / MMPose model zoo checkpoints are released under the Apache 2.0 "
                         "license (see the mmpose repository's own LICENSE) -- record the exact MMPose "
                         "commit's LICENSE file alongside this record, do not assume it never changes.",
        "backbone_keys_loaded_from_checkpoint": loaded_keys,
        "backbone_keys_loaded_count": len(loaded_keys),
        "backbone_keys_missing_from_checkpoint": missing_from_checkpoint,
        "backbone_keys_unexpected_in_checkpoint": unexpected_in_checkpoint,
        "backbone_keys_matched_but_value_unchanged": unchanged_despite_match,
        "verified_pretrained_load_actually_happened": (
            len(loaded_keys) > 0 and len(unchanged_despite_match) < len(loaded_keys)
        ),
        "head_keys_unexpectedly_found_in_checkpoint": head_keys_present_in_checkpoint,
        "head_is_freshly_initialised": len(head_keys_present_in_checkpoint) == 0,
        "backbone_param_count": sum(p.numel() for p in model.backbone.parameters()),
        "head_param_count": sum(p.numel() for p in model.head.parameters()),
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
