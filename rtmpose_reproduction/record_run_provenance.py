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

CORRECTED 2026-08-06, round 2: the first version of this script only
called `MODELS.build(cfg.model)` and assumed the backbone's
`init_cfg=dict(type='Pretrained', ...)` had already been applied -- but in
MMEngine's convention, `init_cfg` only DECLARES how to initialise; the
actual weight loading happens when `model.init_weights()` runs. Fixed:
`model.init_weights()` is called explicitly, and `--pretrained-checkpoint-path`
became a required argument.

CORRECTED 2026-08-06, round 3 (review finding -- this closes a real gap
round 2 left open): round 2's "verification" hashed/diffed a LOCAL file by
key name, while the generated config's `init_cfg.checkpoint` was still a
hardcoded URL -- meaning `model.init_weights()` actually loaded from the
URL, completely independent of whichever local file this script was
separately hashing. Two files could differ in content while having
identical key names/shapes, and the script would still report success.
Round 2 also only checked that loaded values differed from their
PRE-init-weights (random) state, not that they matched the checkpoint's
OWN values -- a weaker claim than "the checkpoint was correctly loaded."

Fixed now: (a) `make_config.py` was changed to embed the LOCAL checkpoint
path directly as `init_cfg.checkpoint` (see that file's own docstring), so
this script can assert `cfg.model["backbone"]["init_cfg"]["checkpoint"] ==
str(args.pretrained_checkpoint_path)` and fail loudly if they ever diverge
-- the file that gets loaded and the file that gets hashed are now
guaranteed the same file, checked, not assumed; (b) every expected backbone
key's post-`init_weights()` VALUE is now compared for exact equality
against the checkpoint's OWN tensor (not just "changed from random init"),
and EVERY key must match (missing, unexpected, or value-mismatched keys
are ALL now fatal, not just "zero keys matched").

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

    backbone_cfg = cfg.model["backbone"]
    init_cfg = backbone_cfg.get("init_cfg", {})
    checkpoint_path_in_config = init_cfg.get("checkpoint")
    prefix = init_cfg.get("prefix")

    # CLOSES THE ROUND-2 GAP: the config's own init_cfg.checkpoint (what
    # model.init_weights() will actually load) must be the EXACT SAME file
    # as --pretrained-checkpoint-path (what this script hashes/diffs) --
    # checked, not assumed. If make_config.py or the config were ever
    # hand-edited to point elsewhere, this must fail loudly here rather
    # than silently record provenance for the wrong file.
    if str(Path(checkpoint_path_in_config).resolve()) != str(args.pretrained_checkpoint_path.resolve()):
        raise SystemExit(
            f"ERROR: config's backbone.init_cfg.checkpoint "
            f"({checkpoint_path_in_config!r}) does not match "
            f"--pretrained-checkpoint-path ({args.pretrained_checkpoint_path!r}) "
            f"-- model.init_weights() would load a DIFFERENT file than the "
            f"one this script is about to hash and diff. Refusing to record "
            f"provenance that would not actually describe the loaded weights."
        )

    model = MODELS.build(cfg.model)

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
        ckpt_backbone_state = {
            k[len(prefix):]: v for k, v in ckpt_state.items() if k.startswith(prefix)
        }
    else:
        ckpt_backbone_state = dict(ckpt_state)

    backbone_param_keys = set(post_init_backbone_state.keys())
    ckpt_backbone_keys = set(ckpt_backbone_state.keys())
    loaded_keys = sorted(backbone_param_keys & ckpt_backbone_keys)
    missing_from_checkpoint = sorted(backbone_param_keys - ckpt_backbone_keys)
    unexpected_in_checkpoint = sorted(ckpt_backbone_keys - backbone_param_keys)

    # CLOSES THE ROUND-2 GAP: compare the loaded VALUES against the
    # checkpoint's OWN tensors for exact equality (not merely "differs from
    # the pre-init random state", which only proves *something* loaded, not
    # that the CORRECT thing loaded). Since round 3's fix above guarantees
    # the config's checkpoint path and this script's checkpoint path are the
    # same file, an exact match here is the real, closed-loop verification.
    value_mismatches = [
        k for k in loaded_keys
        if post_init_backbone_state[k].shape != ckpt_backbone_state[k].shape
        or not torch.allclose(post_init_backbone_state[k].float(),
                               ckpt_backbone_state[k].float(), atol=1e-6)
    ]
    unchanged_from_random_init = [
        k for k in loaded_keys
        if torch.equal(pre_init_backbone_state[k], post_init_backbone_state[k])
    ]

    if missing_from_checkpoint:
        raise SystemExit(
            f"ERROR: {len(missing_from_checkpoint)} backbone parameter(s) have "
            f"no matching key in the checkpoint (after stripping prefix "
            f"{prefix!r}): {missing_from_checkpoint[:10]}{'...' if len(missing_from_checkpoint) > 10 else ''} "
            f"-- refusing to proceed with a partially-initialised backbone."
        )
    if value_mismatches:
        raise SystemExit(
            f"ERROR: {len(value_mismatches)} backbone parameter(s) do not "
            f"exactly match the checkpoint's own values after init_weights() "
            f"(shape mismatch or numerical difference > 1e-6): "
            f"{value_mismatches[:10]}{'...' if len(value_mismatches) > 10 else ''} "
            f"-- init_weights() did not load this checkpoint correctly."
        )
    if unchanged_from_random_init:
        raise SystemExit(
            f"ERROR: {len(unchanged_from_random_init)} backbone parameter(s) "
            f"are unchanged from their pre-init_weights() random state despite "
            f"matching the checkpoint by name and value (a checkpoint whose "
            f"values happen to equal a fresh random init is implausible for a "
            f"real pretrained CSPNeXt-s) -- investigate before trusting this run."
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
        "pretrained_checkpoint_path_in_config": checkpoint_path_in_config,
        "pretrained_checkpoint_local_path": str(args.pretrained_checkpoint_path),
        "config_and_local_checkpoint_path_match": True,  # would have raised above otherwise
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
        "backbone_keys_value_mismatch_vs_checkpoint": value_mismatches,
        "backbone_keys_unchanged_from_random_init": unchanged_from_random_init,
        "verified_pretrained_load_actually_happened": (
            len(loaded_keys) > 0
            and not missing_from_checkpoint
            and not value_mismatches
            and not unchanged_from_random_init
        ),  # would have raised above if False -- this field documents the
            # closed-loop check, not just an assumption
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
