#!/usr/bin/env bash
# UCL BPD / seed 42 canary ONLY, per PROTOCOL_LOCKED.md's "Mandatory canary"
# section. Does not release the remaining 49 runs automatically -- that
# driver does not exist yet (deliberately: PROTOCOL_LOCKED.md and the
# supervisor's email both require sharing canary results before starting
# the full five-seed sweep, not just before the remaining 49 runs of one
# dataset/task group the way the HRNet-512 driver's per-group safe-stop
# works).
#
# CORRECTED 2026-08-06, three review rounds (see PROTOCOL_AUDIT.md):
# round 1 -- never generated an internal validation split, so
# make_config.py pointed straight at the released Test set for periodic
# validation and checkpoint selection (a real leak); selected the "final"
# checkpoint via a lexicographic glob+sort (wrong for numeric epochs, could
# pick a "best" file over the true final). round 2 -- flagged that this
# script converted the Test CSV to COCO json (opening every Test image to
# read its dimensions) BEFORE training started, which is not technically a
# leak (no label ever reaches the training loop) but is imprecise to
# describe as "Test touched only once, after training." Test CSV/image
# conversion is now deferred to AFTER training completes (step 6), so
# nothing under `coco/Test*` exists on disk until training is fully done.
#
# Prerequisites this script does NOT install for you (see ENVIRONMENT.md):
#   - a pinned MMPose/MMEngine/MMCV install in its own venv, verified importable
#   - the local pure-Python test suite already passing -- this script
#     re-checks that, but do not rely on it alone.
#   - the CSPNeXt-s checkpoint downloaded locally (PRETRAINED_CKPT_PATH) --
#     see ENVIRONMENT.md's download step.

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
PRETRAINED_CKPT_PATH="${PRETRAINED_CKPT_PATH:?set PRETRAINED_CKPT_PATH to the locally-downloaded CSPNeXt-s checkpoint file, see ENVIRONMENT.md}"

cd "$SCRIPT_DIR"

echo "=== [0/6] local pure-Python test suite (no mmpose required) ==="
"$PY" test_geometry.py
"$PY" test_endpoint_order.py
"$PY" test_convert_csv_to_coco.py
"$PY" test_evaluate_rtmpose_fixed.py
"$PY" test_fetal_augment.py
"$PY" test_low_level_decode.py

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

echo "=== [3/6] convert UCL BPD internal-train / internal-val CSVs to COCO json (Test CSV NOT touched here) ==="
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

echo "=== [4/6] generate the canary config ==="
# `--test-ann "$GT_TEST_JSON"` below only writes that PATH STRING into the
# generated config's `inference_dataloader` (round 7: NOT `test_dataloader`,
# which is now `None` -- see make_config.py's own comment for why) --
# needed so the config text is complete; the file itself does not exist
# yet and is not required to exist at config-generation time; nothing
# reads it until step 6.
"$PY" make_config.py \
  --dataset UCL --task BPD --seed 42 \
  --data-root "$DATA_ROOT" \
  --images-dir "$DATA_ROOT/images/UCL/Head" \
  --internal-train-ann "$GT_INTERNAL_TRAIN_JSON" \
  --internal-val-ann "$GT_INTERNAL_VAL_JSON" \
  --test-ann "$GT_TEST_JSON" \
  --pretrained-checkpoint-path "$PRETRAINED_CKPT_PATH" \
  --work-dir "$WORK_DIR" \
  --max-epochs "$MAX_EPOCHS" \
  --out "$CONFIG_PATH"
# CORRECTED 2026-08-07: --max-epochs was not previously passed here at all
# -- this script's own $MAX_EPOCHS shell variable was silently ignored by
# make_config.py (which has no CLI flag for it until this fix), so every
# invocation of this script actually generated a 200-epoch config
# regardless of $MAX_EPOCHS. Found while wiring up run_smoke_test.sh, which
# genuinely needs MAX_EPOCHS=1 to take effect.

echo "=== [4b/6] record + VERIFY pretrained-weight provenance (fails loudly on any key/value mismatch) + actual parameter counts ==="
"$PY" record_run_provenance.py \
  --config "$CONFIG_PATH" \
  --pretrained-checkpoint-path "$PRETRAINED_CKPT_PATH" \
  --out-json "$ARTIFACT_ROOT/UCL_BPD_seed42_canary_provenance.json"

echo "=== [4c/6] MANDATORY live preflight gate -- refuses to proceed to training on any failure ==="
# CORRECTED 2026-08-06, round 5 (review finding): ENVIRONMENT.md's
# checklist documented everything that needs live verification (model
# build, full non-square+codec round trip, train forward/backward,
# decode path, BGR/RGB) but nothing in this script actually FORCED those
# checks before training started -- a plain run of this script would have
# gone straight from provenance recording into 200 epochs of training,
# silently skipping every documented gate. live_preflight.py is that
# enforcement: a non-zero exit here is fatal, `set -e` above already stops
# the script, but the explicit check below prints a clear final message
# regardless.
"$PY" live_preflight.py --config "$CONFIG_PATH" \
  || { echo "ERROR: live_preflight.py failed -- refusing to start training. See ENVIRONMENT.md's checklist and PROTOCOL_AUDIT.md for what to fix." >&2; exit 1; }

# PREFLIGHT_ONLY (round 6, review request): lets this script be run on the
# server to install/build/preflight-verify everything WITHOUT also
# auto-starting the 200-epoch canary in the same invocation -- e.g. to run
# preflight the night before, then start the actual canary the next
# morning only after the supervisor's endpoint-ordering reply arrives
# (see README.md "Still genuinely open"). Defaults to 1 (stop after
# preflight) so the safer behaviour is the default; explicitly set
# PREFLIGHT_ONLY=0 to actually proceed to training.
if [ "${PREFLIGHT_ONLY:-1}" = "1" ]; then
  echo "=== PREFLIGHT COMPLETE -- PREFLIGHT_ONLY=1 (default), training NOT started ==="
  echo "Recommended before the real 200-epoch run: run_smoke_test.sh (2026-08-07,"
  echo "genuinely automated -- the manual steps previously suggested here relied on"
  echo "a --max-epochs flag that did not actually exist in make_config.py yet):"
  echo "    PY=\$PY PRETRAINED_CKPT_PATH=\$PRETRAINED_CKPT_PATH MMPOSE_TRAIN_TOOL=\$MMPOSE_TRAIN_TOOL bash run_smoke_test.sh"
  echo "  Exercises the REAL Runner.from_cfg() training loop (optimizer, scheduler,"
  echo "  InternalFixedChannelNMEHook, CheckpointHook) for exactly 1 epoch in its own"
  echo "  throwaway work_dir, NEVER touching the Test set -- something live_preflight.py's"
  echo "  fake-Runner Hook test above does not exercise. It is an engineering check"
  echo "  only; it prints its own result, never a formal one, and cleans up nothing"
  echo "  automatically -- delete its work_dir yourself when satisfied."
  echo "Re-run with PREFLIGHT_ONLY=0 to actually start the seed-42 canary."
  exit 0
fi

echo "=== [5/6] train (this is the canary -- one run, watched, not backgrounded) ==="
MMPOSE_TRAIN_TOOL="${MMPOSE_TRAIN_TOOL:?set MMPOSE_TRAIN_TOOL to the installed mmpose repo tools/train.py path}"
"$PY" "$MMPOSE_TRAIN_TOOL" "$CONFIG_PATH" --work-dir "$WORK_DIR"
# `tools/train.py` runs the training loop plus periodic `val` evaluation
# (against the internal-val split above) at `val_interval` -- it does NOT
# run a `test` loop; that only happens via a separate `tools/test.py`
# invocation, which nothing in this script calls. Confirm this against the
# actual training log (no "Testing" phase should appear) before trusting
# that Test truly was not touched during this run.

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

echo "=== [6/6] NOW convert the released Test CSV, export predictions (original-image space), and score ==="
# This is the FIRST point in this script where the Test CSV/images are
# read at all -- training (step 5) and checkpoint selection are both
# already fully complete by this line.
"$PY" convert_csv_to_coco.py \
  --csv "$DATA_ROOT/annotations/UCL/Head_Test.csv" \
  --images-dir "$DATA_ROOT/images/UCL/Head" \
  --dataset UCL --task BPD \
  --out-json "$GT_TEST_JSON" \
  --excluded-log "$ARTIFACT_ROOT/coco/UCL_BPD_test_excluded.json"

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
echo "  - the training log shows no Testing phase (only Training/Validation)"
echo "  - visual overlay of a handful of predictions vs GT on original images"
echo "  - the printed fixed-channel/swap-min summary above for plausibility"
echo "  - share this canary's numbers with the supervisor before the 5-seed sweep, AND"
echo "    separately raise the EoMT-vs-HRNet/RTMPose endpoint-ordering-convention"
echo "    question (see README.md 'Still genuinely open') before treating any"
echo "    three-method comparison as apples-to-apples on endpoint identity"
