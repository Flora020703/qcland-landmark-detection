#!/usr/bin/env bash
# UCL BPD / seed 42 canary ONLY, per PROTOCOL_LOCKED.md's "Mandatory canary"
# section. Does not release the remaining 49 runs automatically -- that
# driver does not exist yet (deliberately: PROTOCOL_LOCKED.md and the
# supervisor's email both require sharing canary results before starting
# the full five-seed sweep, not just before the remaining 49 runs of one
# dataset/task group the way the HRNet-512 driver's per-group safe-stop
# works).
#
# CORRECTED 2026-08-06 (review findings, see PROTOCOL_AUDIT.md): this
# script previously (a) never generated an internal validation split, so
# make_config.py pointed straight at the released Test set for periodic
# validation and checkpoint selection -- a real data leak, not just a soft
# violation; (b) selected the "final" checkpoint via
# `find ... -name "best_PCK_epoch_*.pth" -o -name "epoch_*.pth" | sort | tail -1`,
# which is neither guaranteed to be the true final epoch (lexicographic sort
# of "epoch_95.pth" vs "epoch_200.pth" is wrong) nor excludes the "best"
# checkpoint from being picked over the real final one. Both are fixed below.
#
# Prerequisites this script does NOT install for you (see ENVIRONMENT.md):
#   - a pinned MMPose/MMEngine/MMCV install in its own venv, verified importable
#   - the local pure-Python test suite already passing -- this script
#     re-checks that, but do not rely on it alone.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:?set PY to the RTMPose venv python interpreter}"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/root/autodl-tmp/rtmpose_reproduction}"
MAX_EPOCHS="${MAX_EPOCHS:-200}"
WORK_DIR="$ARTIFACT_ROOT/UCL_BPD_seed42_canary"
INTERNAL_SPLIT_JSON="$ARTIFACT_ROOT/coco/UCL_BPD_internal_split.json"
GT_INTERNAL_TRAIN_JSON="$ARTIFACT_ROOT/coco/UCL_BPD_internal_train.json"
GT_INTERNAL_VAL_JSON="$ARTIFACT_ROOT/coco/UCL_BPD_internal_val.json"
GT_TEST_JSON="$ARTIFACT_ROOT/coco/UCL_BPD_test.json"
CONFIG_PATH="$ARTIFACT_ROOT/configs/UCL_BPD_seed42_canary.py"
PRED_JSON="$ARTIFACT_ROOT/UCL_BPD_seed42_canary_predictions.json"

cd "$SCRIPT_DIR"

echo "=== [0/6] local pure-Python test suite (no mmpose required) ==="
"$PY" test_geometry.py
"$PY" test_endpoint_order.py
"$PY" test_convert_csv_to_coco.py
"$PY" test_evaluate_rtmpose_fixed.py
"$PY" test_fetal_augment.py

echo "=== [1/6] verify mmpose/mmengine/mmcv import and record versions ==="
"$PY" - <<'PY'
import mmcv, mmengine, mmpose
print(f"mmcv={mmcv.__version__} mmengine={mmengine.__version__} mmpose={mmpose.__version__}")
PY

echo "=== [2/6] build the Train-only internal validation split (NEVER the Test set) ==="
"$PY" make_internal_val_split.py \
  --csv "$DATA_ROOT/annotations/UCL/Head_Train.csv" \
  --images-dir "$DATA_ROOT/images/UCL/Head" \
  --task BPD \
  --out-json "$INTERNAL_SPLIT_JSON"

echo "=== [3/6] convert UCL BPD internal-train / internal-val / Test CSVs to COCO json ==="
"$PY" convert_csv_to_coco.py \
  --csv "$DATA_ROOT/annotations/UCL/Head_Train.csv" \
  --images-dir "$DATA_ROOT/images/UCL/Head" \
  --dataset UCL --task BPD \
  --out-json "$GT_INTERNAL_TRAIN_JSON" \
  --excluded-log "$ARTIFACT_ROOT/coco/UCL_BPD_internal_train_excluded.json" \
  --internal-split-json "$INTERNAL_SPLIT_JSON" --internal-split-part internal_train

"$PY" convert_csv_to_coco.py \
  --csv "$DATA_ROOT/annotations/UCL/Head_Train.csv" \
  --images-dir "$DATA_ROOT/images/UCL/Head" \
  --dataset UCL --task BPD \
  --out-json "$GT_INTERNAL_VAL_JSON" \
  --excluded-log "$ARTIFACT_ROOT/coco/UCL_BPD_internal_val_excluded.json" \
  --internal-split-json "$INTERNAL_SPLIT_JSON" --internal-split-part internal_val

# The released Test CSV is converted here ONLY so run_inference.py has a
# COCO json to read after training -- nothing above this line, and nothing
# in the generated config's train_dataloader/val_dataloader, ever touches it.
"$PY" convert_csv_to_coco.py \
  --csv "$DATA_ROOT/annotations/UCL/Head_Test.csv" \
  --images-dir "$DATA_ROOT/images/UCL/Head" \
  --dataset UCL --task BPD \
  --out-json "$GT_TEST_JSON" \
  --excluded-log "$ARTIFACT_ROOT/coco/UCL_BPD_test_excluded.json"

echo "=== [4/6] generate the canary config ==="
"$PY" make_config.py \
  --dataset UCL --task BPD --seed 42 \
  --data-root "$DATA_ROOT" \
  --images-dir "$DATA_ROOT/images/UCL/Head" \
  --internal-train-ann "$GT_INTERNAL_TRAIN_JSON" \
  --internal-val-ann "$GT_INTERNAL_VAL_JSON" \
  --test-ann "$GT_TEST_JSON" \
  --work-dir "$WORK_DIR" \
  --out "$CONFIG_PATH"

echo "=== [4b/6] record pretrained-weight provenance + actual parameter counts ==="
PRETRAINED_CKPT_PATH="${PRETRAINED_CKPT_PATH:?set PRETRAINED_CKPT_PATH to the locally-downloaded CSPNeXt-s checkpoint file (see ENVIRONMENT.md)}"
"$PY" record_run_provenance.py \
  --config "$CONFIG_PATH" \
  --pretrained-checkpoint-path "$PRETRAINED_CKPT_PATH" \
  --out-json "$ARTIFACT_ROOT/UCL_BPD_seed42_canary_provenance.json"

echo "=== [5/6] train (this is the canary -- one run, watched, not backgrounded) ==="
MMPOSE_TRAIN_TOOL="${MMPOSE_TRAIN_TOOL:?set MMPOSE_TRAIN_TOOL to the installed mmpose repo tools/train.py path}"
"$PY" "$MMPOSE_TRAIN_TOOL" "$CONFIG_PATH" --work-dir "$WORK_DIR"

# CORRECTED 2026-08-06: read MMEngine's own `last_checkpoint` pointer file
# (written by the Runner every time CheckpointHook saves, standard MMEngine
# behaviour -- confirm this exact filename/format against the installed
# MMEngine version before trusting it) instead of glob+lexicographic-sort,
# and assert its epoch number actually equals max_epochs -- PROTOCOL_LOCKED.md
# requires the true final/last checkpoint, not a lexicographically-last
# filename and not a "best" checkpoint. Fail loudly, do not fall back to
# best/glob, if this file is missing or doesn't point at the real final epoch.
LAST_CKPT_POINTER="$WORK_DIR/last_checkpoint"
[ -f "$LAST_CKPT_POINTER" ] || { echo "ERROR: $LAST_CKPT_POINTER not found -- cannot verify the final checkpoint" >&2; exit 1; }
FINAL_CKPT="$(cat "$LAST_CKPT_POINTER")"
[ -f "$FINAL_CKPT" ] || FINAL_CKPT="$WORK_DIR/$(basename "$FINAL_CKPT")"
[ -f "$FINAL_CKPT" ] || { echo "ERROR: checkpoint path in $LAST_CKPT_POINTER does not exist: $FINAL_CKPT" >&2; exit 1; }
case "$(basename "$FINAL_CKPT")" in
  epoch_${MAX_EPOCHS}.pth) ;;
  *) echo "ERROR: last_checkpoint points at $(basename "$FINAL_CKPT"), not epoch_${MAX_EPOCHS}.pth (MAX_EPOCHS=${MAX_EPOCHS}) -- training may not have completed, or MAX_EPOCHS is set wrong. Refusing to silently fall back to a different checkpoint." >&2; exit 1 ;;
esac
echo "using verified final checkpoint: $FINAL_CKPT"

echo "=== [6/6] export predictions (original-image space) and score ==="
"$PY" run_inference.py \
  --config "$CONFIG_PATH" \
  --checkpoint "$FINAL_CKPT" \
  --gt-json "$GT_TEST_JSON" \
  --out-predictions-json "$PRED_JSON"

"$PY" evaluate_rtmpose_fixed.py \
  --gt-json "$GT_TEST_JSON" \
  --predictions-json "$PRED_JSON" \
  --per-image-csv "$ARTIFACT_ROOT/UCL_BPD_seed42_canary_per_image.csv" \
  --summary-json "$ARTIFACT_ROOT/UCL_BPD_seed42_canary_summary.json"

echo "=== CANARY COMPLETE -- do not start the remaining 49 runs yet ==="
echo "Per PROTOCOL_LOCKED.md, review before proceeding:"
echo "  - exact released Train/Test filenames/counts (see coco/*_excluded.json)"
echo "  - internal validation split subjects (coco/UCL_BPD_internal_split.json) never overlap Test"
echo "  - visual overlay of a handful of predictions vs GT on original images"
echo "  - the printed fixed-channel/swap-min summary above for plausibility"
echo "  - share this canary's numbers with the supervisor before the 5-seed sweep"
