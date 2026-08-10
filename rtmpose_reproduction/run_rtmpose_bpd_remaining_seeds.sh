#!/usr/bin/env bash
# UCL BPD / seeds {0,123,2024,3407} -- the remaining four seeds of the
# five-seed sweep, per the supervisor's explicit written approval
# (2026-08-10): "Please proceed with the remaining four UCL BPD seeds
# using exactly the same configuration, so that we can obtain a reliable
# five-seed mean and seed-level standard deviation."
#
# Deliberately does NOT re-run seed 42 (already complete -- see
# run_rtmpose_canary.sh's own output under
# $ARTIFACT_ROOT/UCL_BPD_seed42_canary*) and does NOT regenerate the
# internal train/val split or the Test COCO json: those are FIXED,
# seed-independent artifacts (make_internal_val_split.py's own
# --val-split-seed default is 42, entirely separate from the per-run model
# seed passed to make_config.py -- confirmed by the canary script never
# passing --val-split-seed at all). Reusing the SAME split/COCO files
# across all 5 seeds is required for "exactly the same configuration" to
# actually mean the same released Train/Test partition and internal split,
# not just the same hyperparameters -- regenerating them per seed would
# risk a different split if this script's own logic ever changed between
# runs, silently breaking that guarantee.
#
# "Exactly the same configuration" as the canary means: same
# hyperparameters, same 200-epoch budget, same architecture, same
# preprocessing, same internal-val/Test split -- ONLY the model
# initialisation/DataLoader seed and each run's own checkpoint/prediction
# paths differ, exactly as EoMT's and HRNet's own 5-seed sweeps already do
# elsewhere in this project.
#
# Prerequisites (same as run_rtmpose_canary.sh -- see ENVIRONMENT.md):
#   - a pinned MMPose/MMEngine/MMCV install in its own venv, verified importable
#   - the CSPNeXt-s checkpoint downloaded locally (PRETRAINED_CKPT_PATH)
#   - run_rtmpose_canary.sh's seed-42 run already complete, with its shared
#     coco/ artifacts present under $ARTIFACT_ROOT (checked below, not
#     regenerated)

set -euo pipefail

export CUBLAS_WORKSPACE_CONFIG=:4096:8

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:?set PY to the RTMPose venv python interpreter}"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/root/autodl-tmp/rtmpose_reproduction}"
MAX_EPOCHS="${MAX_EPOCHS:-200}"
PRETRAINED_CKPT_PATH="${PRETRAINED_CKPT_PATH:?set PRETRAINED_CKPT_PATH to the locally-downloaded CSPNeXt-s checkpoint file, see ENVIRONMENT.md}"
MMPOSE_TRAIN_TOOL="${MMPOSE_TRAIN_TOOL:?set MMPOSE_TRAIN_TOOL to the installed mmpose repo tools/train.py path}"

SEEDS=(0 123 2024 3407)

# Shared, seed-independent artifacts -- produced once by the seed-42 canary.
INTERNAL_SPLIT_JSON="$ARTIFACT_ROOT/coco/UCL_BPD_internal_split.json"
GT_INTERNAL_TRAIN_JSON="$ARTIFACT_ROOT/coco/UCL_BPD_internal_train.json"
GT_INTERNAL_VAL_JSON="$ARTIFACT_ROOT/coco/UCL_BPD_internal_val.json"
GT_TEST_JSON="$ARTIFACT_ROOT/coco/UCL_BPD_test.json"

cd "$SCRIPT_DIR"

echo "=== [0/N] local pure-Python test suite (no mmpose required) ==="
"$PY" test_geometry.py
"$PY" test_endpoint_order.py
"$PY" test_convert_csv_to_coco.py
"$PY" test_evaluate_rtmpose_fixed.py
"$PY" test_fetal_augment.py
"$PY" test_low_level_decode.py

echo "=== [0b/N] verify mmpose/mmengine/mmcv import and record versions ==="
"$PY" - <<'PY'
import mmcv, mmengine, mmpose
print(f"mmcv={mmcv.__version__} mmengine={mmengine.__version__} mmpose={mmpose.__version__}")
PY

echo "=== [0c/N] verify the seed-42 canary's shared split/COCO artifacts exist (NOT regenerated here) ==="
for f in "$INTERNAL_SPLIT_JSON" "$GT_INTERNAL_TRAIN_JSON" "$GT_INTERNAL_VAL_JSON" "$GT_TEST_JSON"; do
  if [ ! -f "$f" ]; then
    echo "ERROR: $f not found -- this script requires run_rtmpose_canary.sh's seed-42 run to have already completed (step 6 there writes the Test COCO json; steps 2-3 write the internal split/train/val jsons). Refusing to regenerate them here, since a different regeneration could silently diverge from what seed 42 actually used, breaking 'exactly the same configuration' across all 5 seeds." >&2
    exit 1
  fi
done
echo "[OK] shared split/COCO artifacts present, will be reused unchanged for all 4 seeds"

for SEED in "${SEEDS[@]}"; do

  RUN_NAME="UCL_BPD_seed${SEED}_run"
  WORK_DIR="$ARTIFACT_ROOT/$RUN_NAME"
  CONFIG_PATH="$ARTIFACT_ROOT/configs/UCL_BPD_seed${SEED}_run.py"
  PRED_JSON="$ARTIFACT_ROOT/UCL_BPD_seed${SEED}_run_predictions.json"

  echo ""
  echo "============================================================"
  echo "  START: ${RUN_NAME}  (seed=${SEED})"
  echo "============================================================"

  echo "=== [1/N] generate this seed's config (reusing the shared internal-train/val/Test COCO jsons) ==="
  "$PY" make_config.py \
    --dataset UCL --task BPD --seed "$SEED" \
    --data-root "$DATA_ROOT" \
    --images-dir "$DATA_ROOT/images/UCL/Head" \
    --internal-train-ann "$GT_INTERNAL_TRAIN_JSON" \
    --internal-val-ann "$GT_INTERNAL_VAL_JSON" \
    --test-ann "$GT_TEST_JSON" \
    --pretrained-checkpoint-path "$PRETRAINED_CKPT_PATH" \
    --work-dir "$WORK_DIR" \
    --max-epochs "$MAX_EPOCHS" \
    --out "$CONFIG_PATH"

  echo "=== [1a/N] regression test: generated config avoids MMEngine's lazy-import mode ==="
  "$PY" test_make_config_real_load.py

  echo "=== [1b/N] record + VERIFY pretrained-weight provenance + actual parameter counts ==="
  "$PY" record_run_provenance.py \
    --config "$CONFIG_PATH" \
    --pretrained-checkpoint-path "$PRETRAINED_CKPT_PATH" \
    --out-json "$ARTIFACT_ROOT/${RUN_NAME}_provenance.json"

  echo "=== [1c/N] MANDATORY live preflight gate -- refuses to proceed to training on any failure ==="
  "$PY" live_preflight.py --config "$CONFIG_PATH" \
    || { echo "ERROR: live_preflight.py failed for seed=${SEED} -- refusing to start training." >&2; exit 1; }

  echo "=== [2/N] train ==="
  "$PY" "$MMPOSE_TRAIN_TOOL" "$CONFIG_PATH" --work-dir "$WORK_DIR"

  echo "=== [2a/N] verify the true final checkpoint (last_checkpoint pointer, epoch == MAX_EPOCHS) ==="
  LAST_CKPT_POINTER="$WORK_DIR/last_checkpoint"
  [ -f "$LAST_CKPT_POINTER" ] || { echo "ERROR: $LAST_CKPT_POINTER not found for seed=${SEED}" >&2; exit 1; }
  FINAL_CKPT="$(cat "$LAST_CKPT_POINTER")"
  [ -f "$FINAL_CKPT" ] || FINAL_CKPT="$WORK_DIR/$(basename "$FINAL_CKPT")"
  [ -f "$FINAL_CKPT" ] || { echo "ERROR: checkpoint path in $LAST_CKPT_POINTER does not exist: $FINAL_CKPT" >&2; exit 1; }
  case "$(basename "$FINAL_CKPT")" in
    epoch_${MAX_EPOCHS}.pth) ;;
    *) echo "ERROR: last_checkpoint points at $(basename "$FINAL_CKPT"), not epoch_${MAX_EPOCHS}.pth for seed=${SEED} -- refusing to silently fall back to a different checkpoint." >&2; exit 1 ;;
  esac
  echo "using verified final checkpoint: $FINAL_CKPT"

  echo "=== [3/N] inference on the (already-converted, shared) Test set + score ==="
  "$PY" run_inference.py \
    --config "$CONFIG_PATH" \
    --checkpoint "$FINAL_CKPT" \
    --gt-json "$GT_TEST_JSON" \
    --out-predictions-json "$PRED_JSON"

  "$PY" evaluate_rtmpose_fixed.py \
    --gt-json "$GT_TEST_JSON" \
    --predictions-json "$PRED_JSON" \
    --per-image-csv "$ARTIFACT_ROOT/${RUN_NAME}_per_image.csv" \
    --summary-json "$ARTIFACT_ROOT/${RUN_NAME}_summary.json"

  echo "--- DONE: ${RUN_NAME} ---"

done

echo ""
echo "============================================================"
echo "  All 4 remaining UCL BPD seeds complete"
echo "============================================================"
echo ""
echo "Per-seed outputs: \$ARTIFACT_ROOT/UCL_BPD_seed{0,123,2024,3407}_run_{per_image.csv,summary.json}"
echo "Seed 42's canary outputs (already complete): \$ARTIFACT_ROOT/UCL_BPD_seed42_canary_{per_image.csv,summary.json}"
echo ""
echo "Next: aggregate all 5 seeds into the five-seed mean +/- seed-level sample SD"
echo "(reusing the SAME permutation-invariant evaluator as the EoMT/HRNet comparison --"
echo "evaluate_rtmpose_fixed.py's swap_min_nme column IS permutation_invariant_nme, same"
echo "min(direct,crossed) formula, already verified identical throughout this project)."
echo "Do this from a machine with the analysis repo checked out, NOT necessarily this server."
