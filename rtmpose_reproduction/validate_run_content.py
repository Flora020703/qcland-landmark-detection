#!/usr/bin/env python3
"""Single source of truth for verifying an RTMPose-s run's saved artifacts.

2026-08-10, fourth review round: the previous state (bash heredocs
duplicated across run_rtmpose_full_sweep.sh's `verify_config()` and its
ad hoc "recoverable" re-scoring block) meant "complete" runs were only
ever checked for FILE EXISTENCE, never re-validated for content -- a
cell corrupted by a bad copy, a manual edit, or a partial disk write
after being recorded "complete" would be silently trusted and skipped
forever. This script is now the ONE place both run_rtmpose_full_sweep.sh
(config-only mode before training a fresh config; full mode before
trusting a "recoverable" or "complete" run) and backup_and_clean_cell.sh
(full mode before archiving a cell) call, so there is exactly one
implementation of "is this run's config/predictions/summary/per-image
CSV actually correct", not three copies that can silently drift apart.

Two modes:
  config-only  -- verify the generated mmpose config matches every field
                   make_config.py's own TEMPLATE is supposed to have set
                   (architecture, SimCC codec, optimizer/scheduler,
                   checkpoint policy, augmentation pipeline, dataloader
                   annotation paths, training_recipe_summary). Used BEFORE
                   training starts, when there is no summary/predictions
                   yet.
  full         -- everything config-only does, PLUS: independently
                   re-scores the existing predictions.json against the
                   same GT (evaluate_rtmpose_fixed.evaluate() is a pure,
                   deterministic function of predictions+GT) and requires
                   byte-identical agreement with the existing
                   summary.json/per-image CSV, and checks the existing
                   per-image CSV's own row count equals summary.json's
                   own "n" and contains no duplicate filenames (a
                   dict-keyed read would silently collapse duplicates
                   without this explicit check). Used for "recoverable"
                   (before writing a TSV row), "complete" (before
                   skipping), and cell backup (before archiving).

Run directly, e.g.:
  python validate_run_content.py --mode full \
    --config .../configs/UCL_BPD_seed0_run.py --seed 0 --max-epochs 200 \
    --pretrained-checkpoint-path .../cspnext-s.pth \
    --train-json .../UCL_BPD_internal_train.json \
    --val-json .../UCL_BPD_internal_val.json \
    --test-json .../UCL_BPD_test.json \
    --summary-json .../UCL_BPD_seed0_run_summary.json \
    --per-image-csv .../UCL_BPD_seed0_run_per_image.csv \
    --predictions-json .../UCL_BPD_seed0_run_predictions.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tempfile
from pathlib import Path

from mmengine.config import Config
from mmengine.registry import init_default_scope

from evaluate_rtmpose_fixed import evaluate as rescore_evaluate


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_config_checks(cfg, seed: int, max_epochs: int, pretrained: str,
                         train_json: str, val_json: str, test_json: str, *,
                         mode: str, expected_pretrained_sha256: str | None = None,
                         ) -> list[tuple[str, object, object]]:
    m = cfg.model
    backbone = m["backbone"]
    head = m["head"]
    optim = cfg.optim_wrapper
    ckpt_hook = cfg.default_hooks["checkpoint"]
    scaled_lr = 4e-3 * (cfg.train_dataloader["batch_size"] / 1024.0)

    expected_train_pipeline_types = [
        "LoadImage", "FetalRandomFlipAndCanonicalize", "PixelCentreResize",
        "FetalRotateScaleColorJitter", "GenerateTarget", "PackPoseInputs",
    ]
    expected_val_pipeline_types = ["LoadImage", "PixelCentreResize", "PackPoseInputs"]

    # Independently recompute what make_config.py's own real formula (read
    # directly from its source, not assumed) must have produced for THIS
    # cell's actual internal-train image count, so training_recipe_summary
    # and the param_scheduler list can be checked against real expected
    # numbers rather than just "some list exists".
    n_train_images = len(json.load(open(train_json, encoding="utf-8"))["images"])
    batch_size = cfg.train_dataloader["batch_size"]
    iters_per_epoch = -(-n_train_images // batch_size)  # ceil division
    warmup_epochs = min(5, max(1, max_epochs // 20))     # == 5 for max_epochs=200
    expected_warmup_end_iters = warmup_epochs * iters_per_epoch
    expected_cosine_begin_epoch = max_epochs // 2         # == 100 for max_epochs=200

    # backbone.init_cfg.checkpoint is an ABSOLUTE path baked in at config-
    # generation time. In "config-only" mode (verifying a config JUST
    # generated in this same invocation) it must equal the live --pretrained-
    # checkpoint-path argument exactly -- that's the whole point of the check.
    # In "full" mode (re-verifying an OLD "recoverable"/"complete" run,
    # possibly re-run on a different day / after the server's checkpoint
    # cache was re-provisioned at a different path) demanding textual
    # equality against TODAY's env var would spuriously fail a perfectly
    # good historical run. Check something that actually matters instead:
    # the path the config recorded still points at a real file whose
    # content hashes to the known-correct pretrained weight.
    ckpt_recorded = backbone["init_cfg"]["checkpoint"]
    if mode == "config-only":
        pretrained_checks = [
            ("backbone.init_cfg.checkpoint", ckpt_recorded, pretrained),
        ]
    else:
        ckpt_path = Path(ckpt_recorded) if ckpt_recorded else None
        ckpt_exists = bool(ckpt_path) and ckpt_path.is_file()
        pretrained_checks = [
            ("backbone.init_cfg.checkpoint is a non-empty path string", bool(ckpt_recorded), True),
            ("backbone.init_cfg.checkpoint file still exists on disk", ckpt_exists, True),
        ]
        if ckpt_exists and expected_pretrained_sha256:
            pretrained_checks.append((
                "backbone.init_cfg.checkpoint file sha256 matches expected",
                _sha256_of(ckpt_path), expected_pretrained_sha256,
            ))

    return pretrained_checks + [
        # --- randomness / schedule ---
        ("randomness.seed", cfg.randomness["seed"], seed),
        ("randomness.deterministic", cfg.randomness["deterministic"], True),
        ("train_cfg.by_epoch", cfg.train_cfg["by_epoch"], True),
        ("train_cfg.max_epochs", cfg.train_cfg["max_epochs"], max_epochs),
        # --- SimCC codec ---
        ("codec.type", cfg.codec["type"], "SimCCLabel"),
        ("codec.input_size", tuple(cfg.codec["input_size"]), (512, 512)),
        ("codec.sigma", tuple(cfg.codec["sigma"]), (8.0, 8.0)),
        ("codec.simcc_split_ratio", cfg.codec["simcc_split_ratio"], 2.0),
        ("codec.normalize", cfg.codec["normalize"], False),
        ("codec.use_dark", cfg.codec["use_dark"], False),
        # --- model / preprocessor ---
        ("model.type", m["type"], "TopdownPoseEstimator"),
        ("data_preprocessor.bgr_to_rgb", m["data_preprocessor"]["bgr_to_rgb"], True),
        ("data_preprocessor.mean", list(m["data_preprocessor"]["mean"]), [123.675, 116.28, 103.53]),
        ("data_preprocessor.std", list(m["data_preprocessor"]["std"]), [58.395, 57.12, 57.375]),
        # --- backbone: CSPNeXt-s ---
        ("backbone.type", backbone["type"], "CSPNeXt"),
        ("backbone.arch", backbone["arch"], "P5"),
        ("backbone.expand_ratio", backbone["expand_ratio"], 0.5),
        ("backbone.deepen_factor", backbone["deepen_factor"], 0.33),
        ("backbone.widen_factor", backbone["widen_factor"], 0.5),
        ("backbone.channel_attention", backbone["channel_attention"], True),
        ("backbone.init_cfg.type", backbone["init_cfg"]["type"], "Pretrained"),
        ("backbone.init_cfg.prefix", backbone["init_cfg"]["prefix"], "backbone."),
        # --- head: RTMCCHead ---
        ("head.type", head["type"], "RTMCCHead"),
        ("head.in_channels", head["in_channels"], 512),
        ("head.out_channels", head["out_channels"], 2),
        ("head.input_size", tuple(head["input_size"]), (512, 512)),
        ("head.in_featuremap_size", tuple(head["in_featuremap_size"]), (16, 16)),
        ("head.simcc_split_ratio", head["simcc_split_ratio"], 2.0),
        ("head.final_layer_kernel_size", head["final_layer_kernel_size"], 7),
        ("head.loss.type", head["loss"]["type"], "KLDiscretLoss"),
        ("head.loss.beta", head["loss"]["beta"], 10.0),
        ("head.loss.label_softmax", head["loss"]["label_softmax"], True),
        ("test_cfg.flip_test", m["test_cfg"]["flip_test"], False),
        # --- dataloaders: batch size + exact annotation files wired in ---
        ("train_dataloader.batch_size", cfg.train_dataloader["batch_size"], 16),
        ("train_dataloader.sampler.shuffle", cfg.train_dataloader["sampler"]["shuffle"], True),
        ("train_dataloader.sampler.seed", cfg.train_dataloader["sampler"]["seed"], seed),
        ("train_dataloader.dataset.ann_file", cfg.train_dataloader["dataset"]["ann_file"], train_json),
        ("internal_val_dataloader.batch_size", cfg.internal_val_dataloader["batch_size"], 16),
        ("internal_val_dataloader.sampler.shuffle", cfg.internal_val_dataloader["sampler"]["shuffle"], False),
        ("internal_val_dataloader.dataset.ann_file", cfg.internal_val_dataloader["dataset"]["ann_file"], val_json),
        ("inference_dataloader.dataset.ann_file", cfg.inference_dataloader["dataset"]["ann_file"], test_json),
        # --- val/test loop genuinely disabled (see make_config.py's own long
        #     comment on why: no bbox metadata for a stock predict() path) ---
        ("val_dataloader", cfg.val_dataloader, None),
        ("val_evaluator", cfg.val_evaluator, None),
        ("val_cfg", cfg.val_cfg, None),
        ("test_dataloader", cfg.test_dataloader, None),
        ("test_evaluator", cfg.test_evaluator, None),
        ("test_cfg", cfg.test_cfg, None),
        # --- optimizer / scheduler ---
        ("optim_wrapper.type", optim["type"], "OptimWrapper"),
        ("optim_wrapper.optimizer.type", optim["optimizer"]["type"], "AdamW"),
        ("optim_wrapper.optimizer.lr", optim["optimizer"]["lr"], scaled_lr),
        ("optim_wrapper.optimizer.weight_decay", optim["optimizer"]["weight_decay"], 0.0),
        ("optim_wrapper.clip_grad.max_norm", optim["clip_grad"]["max_norm"], 35),
        ("optim_wrapper.clip_grad.norm_type", optim["clip_grad"]["norm_type"], 2),
        ("optim_wrapper.paramwise_cfg.norm_decay_mult", optim["paramwise_cfg"]["norm_decay_mult"], 0),
        ("optim_wrapper.paramwise_cfg.bias_decay_mult", optim["paramwise_cfg"]["bias_decay_mult"], 0),
        ("optim_wrapper.paramwise_cfg.bypass_duplicate", optim["paramwise_cfg"]["bypass_duplicate"], True),
        # --- checkpoint policy: final/last only, never best ---
        ("default_hooks.checkpoint.type", ckpt_hook["type"], "CheckpointHook"),
        ("default_hooks.checkpoint.interval", ckpt_hook["interval"], 5),
        ("default_hooks.checkpoint.save_last", ckpt_hook["save_last"], True),
        ("default_hooks.checkpoint.max_keep_ckpts", ckpt_hook["max_keep_ckpts"], 1),
        ("default_hooks.checkpoint has no save_best", "save_best" in ckpt_hook, False),
        # --- internal fixed-channel NME monitoring hook ---
        ("custom_hooks[0].type", cfg.custom_hooks[0]["type"], "InternalFixedChannelNMEHook"),
        ("custom_hooks[0].internal_val_ann", cfg.custom_hooks[0]["internal_val_ann"], val_json),
        ("custom_hooks[0].interval", cfg.custom_hooks[0]["interval"], 5),
        # --- augmentation pipeline: exact ordering, nothing extra/missing ---
        ("train_pipeline types", [t["type"] for t in cfg.train_pipeline], expected_train_pipeline_types),
        ("val_pipeline types", [t["type"] for t in cfg.val_pipeline], expected_val_pipeline_types),
        # FetalRandomFlipAndCanonicalize's flip_prob IS a real config-level
        # parameter (unlike FetalRotateScaleColorJitter's rotation/scale/colour
        # ranges, which are hardcoded inside that transform class itself --
        # transforms.py's own __init__ only takes input_size, nothing else is
        # exposed at the config level to assert here).
        ("train_pipeline[1].flip_prob", cfg.train_pipeline[1]["flip_prob"], 0.5),
        # Deliberately NOT set anywhere in make_config.py (see that file's own
        # comment: avoids double-scaling lr if --auto-scale-lr is ever passed
        # to tools/train.py) -- must be absent, not merely False.
        ("auto_scale_lr key absent", "auto_scale_lr" in cfg, False),
        # --- LR schedule: exact scheduler count/types/boundaries, independently
        #     recomputed from make_config.py's own real formula above, not just
        #     "a list exists" ---
        ("param_scheduler count", len(cfg.param_scheduler), 2),
        ("param_scheduler[0].type", cfg.param_scheduler[0]["type"], "LinearLR"),
        ("param_scheduler[0].start_factor", cfg.param_scheduler[0]["start_factor"], 1e-5),
        ("param_scheduler[0].by_epoch", cfg.param_scheduler[0]["by_epoch"], False),
        ("param_scheduler[0].begin", cfg.param_scheduler[0]["begin"], 0),
        ("param_scheduler[0].end", cfg.param_scheduler[0]["end"], expected_warmup_end_iters),
        ("param_scheduler[1].type", cfg.param_scheduler[1]["type"], "CosineAnnealingLR"),
        ("param_scheduler[1].eta_min", cfg.param_scheduler[1]["eta_min"], scaled_lr * 0.05),
        ("param_scheduler[1].begin", cfg.param_scheduler[1]["begin"], expected_cosine_begin_epoch),
        ("param_scheduler[1].end", cfg.param_scheduler[1]["end"], max_epochs),
        ("param_scheduler[1].T_max", cfg.param_scheduler[1]["T_max"], max_epochs - expected_cosine_begin_epoch),
        ("param_scheduler[1].by_epoch", cfg.param_scheduler[1]["by_epoch"], True),
        ("param_scheduler[1].convert_to_iter_based", cfg.param_scheduler[1]["convert_to_iter_based"], True),
        # --- training_recipe_summary: independently cross-checked against the
        #     real internal-train image count and the same formula above, not
        #     just "the key exists" ---
        ("training_recipe_summary.n_train_images", cfg.training_recipe_summary["n_train_images"], n_train_images),
        ("training_recipe_summary.batch_size", cfg.training_recipe_summary["batch_size"], batch_size),
        ("training_recipe_summary.iters_per_epoch", cfg.training_recipe_summary["iters_per_epoch"], iters_per_epoch),
        ("training_recipe_summary.effective_lr", cfg.training_recipe_summary["effective_lr"], scaled_lr),
        ("training_recipe_summary.warmup_end_iters", cfg.training_recipe_summary["warmup_end_iters"], expected_warmup_end_iters),
        ("training_recipe_summary.cosine_begin_epoch", cfg.training_recipe_summary["cosine_begin_epoch"], expected_cosine_begin_epoch),
        ("training_recipe_summary.max_epochs", cfg.training_recipe_summary["max_epochs"], max_epochs),
    ]


def verify_config(config_path: str, seed: int, max_epochs: int, pretrained: str,
                   train_json: str, val_json: str, test_json: str, *,
                   mode: str, expected_pretrained_sha256: str | None = None) -> None:
    init_default_scope("mmpose")
    cfg = Config.fromfile(str(config_path))
    checks = build_config_checks(cfg, seed, max_epochs, pretrained, train_json, val_json, test_json,
                                  mode=mode, expected_pretrained_sha256=expected_pretrained_sha256)
    all_ok = True
    for key, got, expected in checks:
        ok = got == expected
        print(f'  {"[OK]  " if ok else "[FAIL]"} {key}: {got!r}' + ("" if ok else f"  expected={expected!r}"))
        if not ok:
            all_ok = False
    if not all_ok:
        raise SystemExit("ERROR: config verification failed -- aborting")
    print(f"[OK] all {len(checks)} config checks passed")


def verify_content_matches_recorded(test_json: str, predictions_json: str,
                                     summary_json: str, per_image_csv: str) -> None:
    summary_path, csv_path = Path(summary_json), Path(per_image_csv)
    existing_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for key in ("n", "fixed_channel_mean_pct", "swap_min_mean_pct"):
        if key not in existing_summary:
            raise SystemExit(f"ERROR: {summary_path} is missing expected key {key!r}")

    with open(csv_path, newline="", encoding="utf-8") as f:
        existing_rows = list(csv.DictReader(f))

    # Reviewer-flagged gap: reading straight into a {filename: row} dict
    # silently collapses duplicate filenames, so row-count and uniqueness
    # must be checked against the RAW row list first, before any dict keying.
    if len(existing_rows) != existing_summary["n"]:
        raise SystemExit(
            f"ERROR: {csv_path} has {len(existing_rows)} rows but {summary_path} "
            f"reports n={existing_summary['n']} -- refusing to trust"
        )
    filenames = [r["filename"] for r in existing_rows]
    dupes = sorted({fn for fn in filenames if filenames.count(fn) > 1})
    if dupes:
        raise SystemExit(f"ERROR: {csv_path} has duplicate filename(s): {dupes}")
    existing_by_name = {r["filename"]: r for r in existing_rows}

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        fresh_summary = rescore_evaluate(Path(test_json), Path(predictions_json),
                                          tmp / "per_image.csv", tmp / "summary.json")
        for key in ("n", "fixed_channel_mean_pct", "swap_min_mean_pct"):
            if existing_summary[key] != fresh_summary[key]:
                raise SystemExit(
                    f"ERROR: existing summary.json disagrees with an independent fresh re-score of "
                    f"the SAME predictions.json against the SAME GT for key {key!r}: "
                    f"existing={existing_summary[key]!r} fresh={fresh_summary[key]!r} -- refusing to trust"
                )
        with open(tmp / "per_image.csv", newline="", encoding="utf-8") as f:
            fresh_by_name = {r["filename"]: r for r in csv.DictReader(f)}

    if set(existing_by_name) != set(fresh_by_name):
        raise SystemExit(
            "ERROR: existing per-image CSV filenames do not match a fresh re-score's own filenames -- refusing to trust"
        )
    mismatches = []
    for fn, fresh_row in fresh_by_name.items():
        existing_row = existing_by_name[fn]
        for col in ("gt0_x", "gt0_y", "gt1_x", "gt1_y", "pred0_x", "pred0_y", "pred1_x", "pred1_y",
                    "fixed_channel_nme", "swap_min_nme"):
            if existing_row[col] != fresh_row[col]:
                mismatches.append((fn, col))
    if mismatches:
        raise SystemExit(
            f"ERROR: {len(mismatches)} (filename, column) mismatch(es) between the existing per-image "
            f"CSV and an independent fresh re-score, e.g. {mismatches[:5]} -- refusing to trust"
        )
    print(f"[OK] existing summary/per-image CSV (n={existing_summary['n']}) verified byte-identical to "
          f"an independent fresh re-score of the same predictions.json against the same GT; row count "
          f"matches n; no duplicate filenames")


def verify_provenance(provenance_json: str, config_path: str,
                      expected_pretrained_sha256: str) -> None:
    path = Path(provenance_json)
    record = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "pretrained_checkpoint_local_sha256": expected_pretrained_sha256,
        "verified_pretrained_load_actually_happened": True,
        "head_is_freshly_initialised": True,
    }
    for key, expected in required.items():
        if record.get(key) != expected:
            raise SystemExit(
                f"ERROR: {path} has {key}={record.get(key)!r}, expected {expected!r}"
            )
    recorded_config = Path(record.get("generated_config_path", "")).resolve()
    if recorded_config != Path(config_path).resolve():
        raise SystemExit(
            f"ERROR: {path} records config {recorded_config}, but this run uses "
            f"{Path(config_path).resolve()}"
        )
    print(f"[OK] provenance verified: pretrained hash/load, fresh head and config path")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["config-only", "full"], default="full")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--max-epochs", required=True, type=int)
    parser.add_argument("--pretrained-checkpoint-path", required=True)
    parser.add_argument("--train-json", required=True)
    parser.add_argument("--val-json", required=True)
    parser.add_argument("--test-json", required=True)
    parser.add_argument("--summary-json")
    parser.add_argument("--per-image-csv")
    parser.add_argument("--predictions-json")
    parser.add_argument("--provenance-json")
    parser.add_argument("--expected-pretrained-sha256",
                         help="Required for --mode full's pretrained-checkpoint check (config-only mode "
                              "checks exact path equality against --pretrained-checkpoint-path instead).")
    args = parser.parse_args()

    verify_config(args.config, args.seed, args.max_epochs, args.pretrained_checkpoint_path,
                  args.train_json, args.val_json, args.test_json,
                  mode=args.mode, expected_pretrained_sha256=args.expected_pretrained_sha256)

    if args.mode == "full":
        missing = [flag for flag, val in (
            ("--summary-json", args.summary_json),
            ("--per-image-csv", args.per_image_csv),
            ("--predictions-json", args.predictions_json),
            ("--provenance-json", args.provenance_json),
        ) if not val]
        if missing:
            raise SystemExit(f"ERROR: --mode full requires {missing}")
        verify_content_matches_recorded(args.test_json, args.predictions_json,
                                         args.summary_json, args.per_image_csv)
        if not args.expected_pretrained_sha256:
            raise SystemExit("ERROR: --mode full requires --expected-pretrained-sha256")
        verify_provenance(args.provenance_json, args.config,
                          args.expected_pretrained_sha256)

    print("[OK] validate_run_content.py: all checks passed")


if __name__ == "__main__":
    main()
