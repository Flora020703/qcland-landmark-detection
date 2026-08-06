#!/usr/bin/env bash
# UCL BPD / seed 42 canary ONLY, per PROTOCOL_LOCKED.md's "Mandatory canary"
# section. Does not release the remaining 49 runs automatically -- that
# driver does not exist yet (deliberately: PROTOCOL_LOCKED.md and the
# supervisor's email both require sharing canary results before starting
# the full five-seed sweep, not just before the remaining 49 runs of one
# dataset/task group the way the HRNet-512 driver's per-group safe-stop
# works).
#
# Prerequisites this script does NOT install for you (see ENVIRONMENT.md):
#   - a pinned MMPose/MMEngine/MMCV install in its own venv, verified importable
#   - convert_csv_to_coco.py already run for UCL BPD Train and Test
#   - the local pure-Python test suite (test_geometry.py, test_endpoint_order.py,
#     test_convert_csv_to_coco.py, test_evaluate_rtmpose_fixed.py) already
#     passing -- this script re-checks that, but do not rely on it alone.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:?set PY to the RTMPose venv's python interpreter}"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/root/autodl-tmp/rtmpose_reproduction}"
WORK_DIR="$ARTIFACT_ROOT/UCL_BPD_seed42_canary"
GT_TRAIN_JSON="$ARTIFACT_ROOT/coco/UCL_BPD_train.json"
GT_TEST_JSON="$ARTIFACT_ROOT/coco/UCL_BPD_test.json"
CONFIG_PATH="$ARTIFACT_ROOT/configs/UCL_BPD_seed42_canary.py"
PRED_JSON="$ARTIFACT_ROOT/UCL_BPD_seed42_canary_predictions.json"

cd "$SCRIPT_DIR"

echo "=== [0/5] local pure-Python test suite (no mmpose required) ==="
"$PY" test_geometry.py
"$PY" test_endpoint_order.py
"$PY" test_convert_csv_to_coco.py
"$PY" test_evaluate_rtmpose_fixed.py

echo "=== [1/5] verify mmpose/mmengine/mmcv import and record versions ==="
"$PY" - <<'PY'
import mmcv, mmengine, mmpose
print(f"mmcv={mmcv.__version__} mmengine={mmengine.__version__} mmpose={mmpose.__version__}")
PY

echo "=== [2/5] convert UCL BPD Train/Test CSVs to COCO json ==="
"$PY" convert_csv_to_coco.py \
  --csv "$DATA_ROOT/annotations/UCL/Head_Train.csv" \
  --images-dir "$DATA_ROOT/images/UCL/Head" \
  --dataset UCL --task BPD \
  --out-json "$GT_TRAIN_JSON" \
  --excluded-log "$ARTIFACT_ROOT/coco/UCL_BPD_train_excluded.json"

"$PY" convert_csv_to_coco.py \
  --csv "$DATA_ROOT/annotations/UCL/Head_Test.csv" \
  --images-dir "$DATA_ROOT/images/UCL/Head" \
  --dataset UCL --task BPD \
  --out-json "$GT_TEST_JSON" \
  --excluded-log "$ARTIFACT_ROOT/coco/UCL_BPD_test_excluded.json"

echo "=== [3/5] generate the canary config ==="
"$PY" make_config.py \
  --dataset UCL --task BPD --seed 42 \
  --data-root "$DATA_ROOT" \
  --images-dir "$DATA_ROOT/images/UCL/Head" \
  --train-ann "$GT_TRAIN_JSON" \
  --test-ann "$GT_TEST_JSON" \
  --work-dir "$WORK_DIR" \
  --out "$CONFIG_PATH"

echo "=== [3b/5] record pretrained-weight provenance + actual parameter counts ==="
"$PY" record_run_provenance.py \
  --config "$CONFIG_PATH" \
  --out-json "$ARTIFACT_ROOT/UCL_BPD_seed42_canary_provenance.json"

echo "=== [4/5] train (this is the canary -- one run, watched, not backgrounded) ==="
MMPOSE_TRAIN_TOOL="${MMPOSE_TRAIN_TOOL:?set MMPOSE_TRAIN_TOOL to the installed mmpose repo's tools/train.py path}"
"$PY" "$MMPOSE_TRAIN_TOOL" "$CONFIG_PATH" --work-dir "$WORK_DIR"

FINAL_CKPT=$(find "$WORK_DIR" -maxdepth 1 -name "best_PCK_epoch_*.pth" -o -name "epoch_*.pth" | sort | tail -1)
[ -n "$FINAL_CKPT" ] || { echo "ERROR: no checkpoint found in $WORK_DIR" >&2; exit 1; }
echo "using checkpoint: $FINAL_CKPT"

echo "=== [5/5] export predictions (original-image space) and score ==="
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
echo "  - visual overlay of a handful of predictions vs GT on original images"
echo "  - the printed fixed-channel/swap-min summary above for plausibility"
echo "  - share this canary's numbers with the supervisor before the 5-seed sweep"
