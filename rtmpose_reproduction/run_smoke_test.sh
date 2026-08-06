#!/usr/bin/env bash
# ============================================================================
# ENGINEERING SMOKE TEST ONLY -- NEVER a formal/reportable result.
# ============================================================================
# Added 2026-08-07 per an explicit request to make this a REAL, automated
# entry point: run_rtmpose_canary.sh's own PREFLIGHT_ONLY=1 branch had long
# suggested a manual "MAX_EPOCHS=1 ... WORK_DIR_SUFFIX=_smoketest" command
# that never actually worked -- make_config.py had no --max-epochs CLI flag
# at all until this same change, so that suggested command silently
# generated a 200-epoch config regardless of MAX_EPOCHS. This script is the
# real, working replacement.
#
# Purpose: live_preflight.py's own check_hook_registry_and_lifecycle (round
# 6) builds InternalFixedChannelNMEHook and calls its after_train_epoch()
# directly on a duck-typed fake Runner -- it proves the Hook CAN run, but
# never exercises the REAL mmengine Runner.from_cfg() training loop
# (optimizer stepping, scheduler stepping, the Hook, and CheckpointHook all
# running together for real, via tools/train.py). This script closes that
# gap with exactly 1 real epoch, nothing more.
#
# Hard guarantees, not just conventions:
#   - Uses its OWN work_dir, entirely separate from run_rtmpose_canary.sh's
#     real canary work_dir -- never shares state with a formal run.
#   - NEVER reads the Test set. There is no --test-ann pointing at a real
#     file anywhere in this script; only internal-train/internal-val are
#     used, same split make_internal_val_split.py produces for the real
#     canary.
#   - MAX_EPOCHS and VAL_INTERVAL are fixed at 1 in this script, not
#     env-overridable -- this script's only job is the 1-epoch integration
#     check; a longer run here would just be a worse-isolated duplicate of
#     run_rtmpose_canary.sh itself.
#   - Verifies three concrete things before declaring PASS: (a) a real
#     epoch_1.pth checkpoint was written, (b) InternalFixedChannelNMEHook
#     actually logged something (not just registered), (c) the training
#     log's own loss values are finite (no nan/inf).
#
# Do NOT report any number this script prints as a result, and do not let
# its work_dir get mixed up with a real run's artifacts -- delete it once
# satisfied:
#   rm -rf "$WORK_DIR"   (printed again at the end of a successful run)
#
# Expect make_config.py to print a "WARNING: max_epochs=1 is too short for
# the normal warmup/cosine schedule ... disabling warmup entirely" line
# during step 3 below -- this is EXPECTED and CORRECT for MAX_EPOCHS=1, not
# a failure (see make_config.py's own module docstring table for why this
# degenerate case is handled the way it is, and why it is unreachable for
# any real run in this project).
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:?set PY to the RTMPose venv python interpreter}"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/root/autodl-tmp/rtmpose_reproduction}"
PRETRAINED_CKPT_PATH="${PRETRAINED_CKPT_PATH:?set PRETRAINED_CKPT_PATH to the locally-downloaded CSPNeXt-s checkpoint file, see ENVIRONMENT.md}"
MMPOSE_TRAIN_TOOL="${MMPOSE_TRAIN_TOOL:?set MMPOSE_TRAIN_TOOL to the installed mmpose repo tools/train.py path}"

# Fixed. Not read from the environment -- see header comment.
MAX_EPOCHS=1
VAL_INTERVAL=1

# Deliberately separate from run_rtmpose_canary.sh's own
# $ARTIFACT_ROOT/UCL_BPD_seed42_canary work_dir.
WORK_DIR="$ARTIFACT_ROOT/UCL_BPD_seed42_SMOKETEST"
INTERNAL_SPLIT_JSON="$ARTIFACT_ROOT/coco/UCL_BPD_internal_split.json"
GT_INTERNAL_TRAIN_JSON="$ARTIFACT_ROOT/coco/UCL_BPD_internal_train.json"
GT_INTERNAL_VAL_JSON="$ARTIFACT_ROOT/coco/UCL_BPD_internal_val.json"
CONFIG_PATH="$ARTIFACT_ROOT/configs/UCL_BPD_seed42_SMOKETEST.py"
# Never created, never read by this script -- there is no Test step here at
# all. Only passed because make_config.py's --test-ann is a required CLI
# argument that writes this path STRING into the generated config's
# inference_dataloader entry; per run_rtmpose_canary.sh's own comment, the
# file need not exist unless something later calls run_inference.py, which
# this script never does.
FAKE_UNUSED_TEST_ANN="$WORK_DIR/NEVER_CREATED_test.json"

echo "############################################################"
echo "# ENGINEERING SMOKE TEST -- MAX_EPOCHS=$MAX_EPOCHS, throwaway work_dir"
echo "# NOT a formal result. The Test set is NEVER read by this script."
echo "############################################################"

rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR"

cd "$SCRIPT_DIR"

echo "=== [smoke 1/5] Train-only internal validation split (reused if already built) ==="
if [ ! -f "$INTERNAL_SPLIT_JSON" ]; then
  "$PY" make_internal_val_split.py \
    --csv "$DATA_ROOT/annotations/UCL/Head_Train.csv" \
    --images-dir "$DATA_ROOT/images/UCL/Head" \
    --task BPD \
    --out-json "$INTERNAL_SPLIT_JSON"
else
  echo "  (reusing existing $INTERNAL_SPLIT_JSON)"
fi

echo "=== [smoke 2/5] convert internal-train / internal-val CSVs (reused if already built) -- Test CSV is NEVER touched here ==="
if [ ! -f "$GT_INTERNAL_TRAIN_JSON" ]; then
  "$PY" convert_csv_to_coco.py \
    --csv "$DATA_ROOT/annotations/UCL/Head_Train.csv" \
    --images-dir "$DATA_ROOT/images/UCL/Head" \
    --dataset UCL --task BPD \
    --out-json "$GT_INTERNAL_TRAIN_JSON" \
    --excluded-log "$ARTIFACT_ROOT/coco/UCL_BPD_internal_train_excluded.json" \
    --internal-split-json "$INTERNAL_SPLIT_JSON" --internal-split-part internal_train
else
  echo "  (reusing existing $GT_INTERNAL_TRAIN_JSON)"
fi
if [ ! -f "$GT_INTERNAL_VAL_JSON" ]; then
  "$PY" convert_csv_to_coco.py \
    --csv "$DATA_ROOT/annotations/UCL/Head_Train.csv" \
    --images-dir "$DATA_ROOT/images/UCL/Head" \
    --dataset UCL --task BPD \
    --out-json "$GT_INTERNAL_VAL_JSON" \
    --excluded-log "$ARTIFACT_ROOT/coco/UCL_BPD_internal_val_excluded.json" \
    --internal-split-json "$INTERNAL_SPLIT_JSON" --internal-split-part internal_val
else
  echo "  (reusing existing $GT_INTERNAL_VAL_JSON)"
fi

echo "=== [smoke 3/5] generate a MAX_EPOCHS=$MAX_EPOCHS / VAL_INTERVAL=$VAL_INTERVAL config in its own work_dir ==="
"$PY" make_config.py \
  --dataset UCL --task BPD --seed 42 \
  --data-root "$DATA_ROOT" \
  --images-dir "$DATA_ROOT/images/UCL/Head" \
  --internal-train-ann "$GT_INTERNAL_TRAIN_JSON" \
  --internal-val-ann "$GT_INTERNAL_VAL_JSON" \
  --test-ann "$FAKE_UNUSED_TEST_ANN" \
  --pretrained-checkpoint-path "$PRETRAINED_CKPT_PATH" \
  --work-dir "$WORK_DIR" \
  --max-epochs "$MAX_EPOCHS" \
  --val-interval "$VAL_INTERVAL" \
  --out "$CONFIG_PATH"

echo "=== [smoke 4/5] train for real, through Runner.from_cfg(), for exactly $MAX_EPOCHS epoch ==="
"$PY" "$MMPOSE_TRAIN_TOOL" "$CONFIG_PATH" --work-dir "$WORK_DIR"

echo "=== [smoke 5/5] verify the three things this smoke test exists to check ==="

# (a) a real checkpoint was actually written for epoch $MAX_EPOCHS.
CKPT="$WORK_DIR/epoch_${MAX_EPOCHS}.pth"
if [ ! -f "$CKPT" ]; then
  echo "SMOKE TEST FAILED: $CKPT was never written -- CheckpointHook did not save at" >&2
  echo "epoch $MAX_EPOCHS (confirm --val-interval $VAL_INTERVAL actually reached the generated config's CheckpointHook)." >&2
  exit 1
fi
echo "  [OK] checkpoint saved: $CKPT"

# (b) InternalFixedChannelNMEHook actually ran, not just registered.
LOG_FILE="$(find "$WORK_DIR" -name '*.log' | sort | tail -n1)"
if [ -z "$LOG_FILE" ]; then
  echo "SMOKE TEST FAILED: no training log (*.log) found anywhere under $WORK_DIR" >&2
  exit 1
fi
if ! grep -q "InternalFixedChannelNMEHook" "$LOG_FILE"; then
  echo "SMOKE TEST FAILED: InternalFixedChannelNMEHook never logged anything in $LOG_FILE" >&2
  echo "-- it may be registered in custom_hooks but not actually invoked by the Runner." >&2
  exit 1
fi
echo "  [OK] InternalFixedChannelNMEHook logged at least one line in $LOG_FILE"

# (c) loss is finite -- no nan/inf anywhere in the log's own loss lines, and
# at least one parseable "loss: <value>" line exists to confirm training
# actually stepped forward/backward for real.
if grep -oE "loss[a-zA-Z_]*: [0-9eE.+-]+" "$LOG_FILE" | grep -qiE "nan|inf"; then
  echo "SMOKE TEST FAILED: a non-finite (nan/inf) loss value appears in $LOG_FILE" >&2
  exit 1
fi
LAST_LOSS="$(grep -oE "loss: [0-9eE.+-]+" "$LOG_FILE" | tail -n1)"
if [ -z "$LAST_LOSS" ]; then
  echo "SMOKE TEST FAILED: could not find any 'loss: <value>' line in $LOG_FILE to confirm training actually ran" >&2
  exit 1
fi
echo "  [OK] last logged '$LAST_LOSS' (finite)"

echo ""
echo "############################################################"
echo "# SMOKE TEST PASSED -- ENGINEERING CHECK ONLY, NOT A RESULT."
echo "# Do not report '$LAST_LOSS' or anything else from this run."
echo "# Delete the throwaway work_dir when satisfied:"
echo "#   rm -rf $WORK_DIR"
echo "############################################################"
