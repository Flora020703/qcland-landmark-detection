#!/usr/bin/env bash
# =============================================================================
# test_sweep_lifecycle_synthetic.sh -- a non-training end-to-end rehearsal of
# the RTMPose backup -> archive-manifest verification -> clean lifecycle.
#
# What this DOES exercise, for real, against a throwaway isolated
# ARTIFACT_ROOT (never the real one -- see ISOLATED_ROOT below):
#   - make_config.py                (real config generation)
#   - validate_run_content.py       (both --mode config-only and --mode full
#                                     -- the exact validator run_rtmpose_full_
#                                     sweep.sh and backup_and_clean_cell.sh
#                                     both call in production)
#   - record_run_provenance.py      (real model build + real backbone-weight
#                                     load against the REAL, already-verified
#                                     pretrained checkpoint)
#   - evaluate_rtmpose_fixed.py     (real scoring)
#   - rtmpose_common.sh             (run_name_for/verify_final_checkpoint/
#                                     shared_json_paths_for/excluded_log_
#                                     paths_for)
#   - backup_and_clean_cell.sh      (verify_cell_complete/do_backup/do_clean,
#                                     the FULL two-phase-verify-then-delete
#                                     lifecycle, run for real)
#
# What this does NOT exercise (skipped deliberately, requires either real
# GPU training or an invasive refactor of run_rtmpose_full_sweep.sh to make
# its train/inference tool paths stubbable -- neither done here):
#   - actual `tools/train.py` / `run_inference.py` calls. Each synthetic seed
#     instead receives a small genuine `torch.save` checkpoint that passes
#     PyTorch-ZIP structure/CRC checks, plus fabricated predictions equal to
#     GT for a deterministic, internally consistent 0% NME. This exercises
#     artifact integrity and scoring, not learned model semantics.
#   - run_rtmpose_full_sweep.sh's own run_artifacts_status() 4-state
#     transitions (fresh/recoverable/complete/inconsistent) and its
#     CELL_DID_WORK pause-then-resume-at-next-cell loop -- these live
#     inside that script's own top-level flow (which unconditionally runs
#     the local test suite + imports mmpose + does real disk/SHA checks the
#     moment it's invoked, so it cannot be safely `source`d as a library
#     the way rtmpose_common.sh can). The lowest-risk way to exercise these
#     for real is simply running run_rtmpose_full_sweep.sh itself for its
#     very first real seed -- it will immediately hit the "recoverable"
#     path recovering the already-approved seed-42 canary, which exercises
#     that exact state without needing a synthetic stand-in.
#
# Uses dataset=UCL task=FL (Femur) with the 5 REAL seed values (42, 0, 123,
# 2024, 3407 -- backup_and_clean_cell.sh's own TSV-completeness check
# requires exactly these 5) purely as valid, whitelisted (dataset,task)
# identifiers; nothing here reads or writes anything under the real
# ARTIFACT_ROOT since ISOLATED_ROOT is always a fresh mktemp -d directory,
# never overridable from the environment.
#
# Usage (on the server, in the RTMPose venv):
#   PY=/path/to/venv/python \
#   PRETRAINED_CKPT_PATH=/path/to/cspnext-s.pth \
#     bash test_sweep_lifecycle_synthetic.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/rtmpose_common.sh"
cd "$SCRIPT_DIR"

PY="${PY:?set PY to the RTMPose venv python interpreter}"
PRETRAINED_CKPT_PATH="${PRETRAINED_CKPT_PATH:?set PRETRAINED_CKPT_PATH to the locally-downloaded CSPNeXt-s checkpoint file}"
[ -f "$PRETRAINED_CKPT_PATH" ] || { echo "ERROR: PRETRAINED_CKPT_PATH does not exist: $PRETRAINED_CKPT_PATH" >&2; exit 1; }

readonly EXPECTED_PRETRAINED_SHA256="aa7d9335bf422ad02a803e36f357dfc6abb807eca42d79e8b3b6e7c5bd1f446b"
GOT_SHA256=$(sha256sum "$PRETRAINED_CKPT_PATH" | awk '{print $1}')
[ "$GOT_SHA256" = "$EXPECTED_PRETRAINED_SHA256" ] || {
  echo "ERROR: PRETRAINED_CKPT_PATH's SHA-256 doesn't match the expected pretrained weight -- this" >&2
  echo "  test reuses the REAL checkpoint file (read-only) so backup_and_clean_cell.sh's own" >&2
  echo "  hardcoded EXPECTED_PRETRAINED_SHA256 check passes; point PRETRAINED_CKPT_PATH at the" >&2
  echo "  real, already-downloaded file, not a substitute." >&2
  exit 1
}

DATASET="UCL"
TASK="FL"
SEEDS=(42 0 123 2024 3407)
MAX_EPOCHS=200

ISOLATED_ROOT="$(mktemp -d)"
cleanup() {
  local status=$?
  echo ""
  echo "=== cleaning up isolated test root: $ISOLATED_ROOT ==="
  rm -rf -- "$ISOLATED_ROOT"
  if [ "$status" -eq 0 ]; then
    echo "[PASS] synthetic backup/clean lifecycle test completed successfully"
  else
    echo "[FAIL] synthetic lifecycle test failed (exit $status) -- see output above" >&2
  fi
  exit "$status"
}
trap cleanup EXIT

RESULTS_TSV="$ISOLATED_ROOT/rtmpose_full_sweep_results.tsv"
printf 'dataset\ttask\tseed\tn\tfixed_channel_mean_pct\tswap_min_mean_pct\n' > "$RESULTS_TSV"
mkdir -p "$ISOLATED_ROOT/coco/manifests" "$ISOLATED_ROOT/configs"

echo "=== [1/4] fabricating tiny but schema-real shared COCO artifacts ==="
shared_paths="$(shared_json_paths_for "$ISOLATED_ROOT" "$DATASET" "$TASK")"
IFS='|' read -r SPLIT_JSON TRAIN_JSON VAL_JSON TEST_JSON MANIFEST <<< "$shared_paths"
excluded_paths="$(excluded_log_paths_for "$ISOLATED_ROOT" "$DATASET" "$TASK")"
IFS='|' read -r TRAIN_EXCLUDED VAL_EXCLUDED TEST_EXCLUDED <<< "$excluded_paths"

"$PY" - "$SPLIT_JSON" "$TRAIN_JSON" "$VAL_JSON" "$TEST_JSON" \
        "$TRAIN_EXCLUDED" "$VAL_EXCLUDED" "$TEST_EXCLUDED" <<'PY'
import json, sys
split_p, train_p, val_p, test_p, train_ex_p, val_ex_p, test_ex_p = sys.argv[1:8]

def coco(names_and_gt):
    images = [{"id": i, "file_name": name, "width": 100, "height": 100}
              for i, (name, _, _) in enumerate(names_and_gt)]
    annotations = [{"id": i, "image_id": i,
                     "keypoints": [g0[0], g0[1], 2, g1[0], g1[1], 2]}
                    for i, (_, g0, g1) in enumerate(names_and_gt)]
    return {"images": images, "annotations": annotations}

train_records = [(f"synthetic_train_{i}.jpg", (10.0 + i, 10.0), (60.0 + i, 60.0)) for i in range(3)]
val_records = [(f"synthetic_val_{i}.jpg", (12.0 + i, 12.0), (62.0 + i, 62.0)) for i in range(2)]
test_records = [(f"synthetic_test_{i}.jpg", (15.0 + i, 15.0), (65.0 + i, 65.0)) for i in range(2)]

json.dump(coco(train_records), open(train_p, "w", encoding="utf-8"))
json.dump(coco(val_records), open(val_p, "w", encoding="utf-8"))
json.dump(coco(test_records), open(test_p, "w", encoding="utf-8"))
json.dump({"internal_train": [n for n, _, _ in train_records],
           "internal_val": [n for n, _, _ in val_records]}, open(split_p, "w", encoding="utf-8"))
for p in (train_ex_p, val_ex_p, test_ex_p):
    json.dump([], open(p, "w", encoding="utf-8"))
print("[OK] fabricated split/train/val/test COCO jsons + 3 empty excluded-logs")
PY

{
  sha256sum "$SPLIT_JSON" "$TRAIN_JSON" "$VAL_JSON" "$TEST_JSON" \
            "$TRAIN_EXCLUDED" "$VAL_EXCLUDED" "$TEST_EXCLUDED"
} > "$MANIFEST"
echo "[OK] wrote shared-artifact manifest: $MANIFEST"

echo ""
echo "=== [2/4] per seed: real config/provenance, valid synthetic checkpoint, fabricated predictions, real scoring/validation ==="
for SEED in "${SEEDS[@]}"; do
  RUN_NAME="$(run_name_for "$DATASET" "$TASK" "$SEED")"
  WORK_DIR="$ISOLATED_ROOT/$RUN_NAME"
  CONFIG_PATH="$ISOLATED_ROOT/configs/${RUN_NAME}.py"
  PROVENANCE_JSON="$ISOLATED_ROOT/${RUN_NAME}_provenance.json"
  PRED_JSON="$ISOLATED_ROOT/${RUN_NAME}_predictions.json"
  PER_IMAGE_CSV="$ISOLATED_ROOT/${RUN_NAME}_per_image.csv"
  SUMMARY_JSON="$ISOLATED_ROOT/${RUN_NAME}_summary.json"

  echo "--- seed=$SEED: real make_config.py ---"
  "$PY" make_config.py \
    --dataset "$DATASET" --task "$TASK" --seed "$SEED" \
    --data-root "$ISOLATED_ROOT" --images-dir "$ISOLATED_ROOT/nonexistent_images" \
    --internal-train-ann "$TRAIN_JSON" --internal-val-ann "$VAL_JSON" --test-ann "$TEST_JSON" \
    --pretrained-checkpoint-path "$PRETRAINED_CKPT_PATH" \
    --work-dir "$WORK_DIR" --max-epochs "$MAX_EPOCHS" --out "$CONFIG_PATH"

  echo "--- seed=$SEED: real test_make_config_real_load.py (lazy-import regression guard) ---"
  "$PY" test_make_config_real_load.py

  echo "--- seed=$SEED: real validate_run_content.py --mode config-only ---"
  "$PY" validate_run_content.py --mode config-only \
    --config "$CONFIG_PATH" --seed "$SEED" --max-epochs "$MAX_EPOCHS" \
    --pretrained-checkpoint-path "$PRETRAINED_CKPT_PATH" \
    --train-json "$TRAIN_JSON" --val-json "$VAL_JSON" --test-json "$TEST_JSON"

  echo "--- seed=$SEED: real record_run_provenance.py (real model build + real weight load) ---"
  "$PY" record_run_provenance.py \
    --config "$CONFIG_PATH" --pretrained-checkpoint-path "$PRETRAINED_CKPT_PATH" \
    --out-json "$PROVENANCE_JSON"

  echo "--- seed=$SEED: creating a small genuine torch.save checkpoint (no real training) ---"
  mkdir -p "$WORK_DIR"
  "$PY" - "$WORK_DIR/epoch_${MAX_EPOCHS}.pth" "$SEED" <<'PY'
import sys
import torch

path, seed = sys.argv[1], int(sys.argv[2])
torch.save(
    {
        "meta": {"synthetic": True, "seed": seed, "epoch": 200},
        "state_dict": {"synthetic.weight": torch.tensor([float(seed)])},
    },
    path,
)
print(f"[OK] wrote structurally valid synthetic PyTorch checkpoint: {path}")
PY
  printf 'epoch_%s.pth\n' "$MAX_EPOCHS" > "$WORK_DIR/last_checkpoint"
  verify_final_checkpoint "$WORK_DIR" "$MAX_EPOCHS" >/dev/null

  echo "--- seed=$SEED: fabricating predictions.json == GT exactly (deterministic 0% NME) ---"
  "$PY" - "$TEST_JSON" "$PRED_JSON" <<'PY'
import json, sys
test_json_path, pred_json_path = sys.argv[1:3]
coco = json.load(open(test_json_path, encoding="utf-8"))
images_by_id = {im["id"]: im["file_name"] for im in coco["images"]}
records = []
for ann in coco["annotations"]:
    fn = images_by_id[ann["image_id"]]
    kp = ann["keypoints"]
    records.append({"file_name": fn, "pred": [[kp[0], kp[1]], [kp[3], kp[4]]]})
json.dump(records, open(pred_json_path, "w", encoding="utf-8"))
print(f"[OK] fabricated {len(records)} exact-match predictions")
PY

  echo "--- seed=$SEED: real evaluate_rtmpose_fixed.py ---"
  "$PY" evaluate_rtmpose_fixed.py \
    --gt-json "$TEST_JSON" --predictions-json "$PRED_JSON" \
    --per-image-csv "$PER_IMAGE_CSV" --summary-json "$SUMMARY_JSON"

  echo "--- seed=$SEED: real validate_run_content.py --mode full ---"
  "$PY" validate_run_content.py --mode full \
    --config "$CONFIG_PATH" --seed "$SEED" --max-epochs "$MAX_EPOCHS" \
    --pretrained-checkpoint-path "$PRETRAINED_CKPT_PATH" \
    --expected-pretrained-sha256 "$EXPECTED_PRETRAINED_SHA256" \
    --train-json "$TRAIN_JSON" --val-json "$VAL_JSON" --test-json "$TEST_JSON" \
    --summary-json "$SUMMARY_JSON" --per-image-csv "$PER_IMAGE_CSV" \
    --predictions-json "$PRED_JSON" --provenance-json "$PROVENANCE_JSON"

  N=$("$PY" -c "import json; print(json.load(open('$SUMMARY_JSON'))['n'])")
  FIXED=$("$PY" -c "import json; print(f\"{json.load(open('$SUMMARY_JSON'))['fixed_channel_mean_pct']:.6f}\")")
  SWAP=$("$PY" -c "import json; print(f\"{json.load(open('$SUMMARY_JSON'))['swap_min_mean_pct']:.6f}\")")
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$DATASET" "$TASK" "$SEED" "$N" "$FIXED" "$SWAP" >> "$RESULTS_TSV"
  echo "[OK] seed=$SEED fully fabricated+validated: n=$N fixed=${FIXED}% swap=${SWAP}% (expect ~0%)"
done

echo ""
echo "=== [3/4] real backup_and_clean_cell.sh backup (full verify_cell_complete + archive build) ==="
ARTIFACT_ROOT="$ISOLATED_ROOT" RESULTS_TSV="$RESULTS_TSV" \
  PY="$PY" PRETRAINED_CKPT_PATH="$PRETRAINED_CKPT_PATH" \
  bash "$SCRIPT_DIR/backup_and_clean_cell.sh" backup "$DATASET" "$TASK"

ARCHIVE_PATH="$(ls -1 "$ISOLATED_ROOT/cell_backups/${DATASET}_${TASK}_5seed_"*.tar)"
[ -f "$ARCHIVE_PATH" ] || { echo "ERROR: backup did not produce an archive" >&2; exit 1; }
N_CKPTS_IN_ARCHIVE=$(tar -tf "$ARCHIVE_PATH" | grep -c 'epoch_200\.pth$')
[ "$N_CKPTS_IN_ARCHIVE" -eq 5 ] || { echo "ERROR: archive has $N_CKPTS_IN_ARCHIVE epoch_200.pth entries, expected 5" >&2; exit 1; }
echo "[OK] archive contains exactly 5 epoch_200.pth checkpoints"

echo "--- local-style extraction + manifest verification (what the human step would do) ---"
EXTRACT_DIR="$(mktemp -d)"
tar -xf "$ARCHIVE_PATH" -C "$EXTRACT_DIR"
[ -f "$EXTRACT_DIR/ARCHIVE_CONTENTS.sha256" ] || { echo "ERROR: ARCHIVE_CONTENTS.sha256 missing from extracted archive" >&2; exit 1; }
( cd "$EXTRACT_DIR" && sha256sum -c ARCHIVE_CONTENTS.sha256 --quiet )
echo "[OK] sha256sum -c ARCHIVE_CONTENTS.sha256 passed on the extracted archive"
rm -rf "$EXTRACT_DIR"

ARCHIVE_SHA256=$(sha256sum "$ARCHIVE_PATH" | awk '{print $1}')

echo ""
echo "=== [4/4] real backup_and_clean_cell.sh clean (two-phase verify-then-delete) ==="
ARTIFACT_ROOT="$ISOLATED_ROOT" RESULTS_TSV="$RESULTS_TSV" \
  PY="$PY" PRETRAINED_CKPT_PATH="$PRETRAINED_CKPT_PATH" \
  bash "$SCRIPT_DIR/backup_and_clean_cell.sh" clean "$DATASET" "$TASK" "$ARCHIVE_SHA256"

echo "--- post-clean state assertions ---"
for SEED in "${SEEDS[@]}"; do
  RUN_NAME="$(run_name_for "$DATASET" "$TASK" "$SEED")"
  [ ! -d "$ISOLATED_ROOT/$RUN_NAME" ] || { echo "ERROR: $ISOLATED_ROOT/$RUN_NAME should have been deleted by clean" >&2; exit 1; }
  for suffix in provenance.json predictions.json per_image.csv summary.json; do
    [ -f "$ISOLATED_ROOT/${RUN_NAME}_${suffix}" ] || { echo "ERROR: ${RUN_NAME}_${suffix} should have survived clean" >&2; exit 1; }
  done
  [ -f "$ISOLATED_ROOT/configs/${RUN_NAME}.py" ] || { echo "ERROR: ${RUN_NAME}.py config should have survived clean" >&2; exit 1; }
done
[ ! -f "$ARCHIVE_PATH" ] || { echo "ERROR: archive should have been deleted by clean" >&2; exit 1; }
echo "[OK] all 5 work_dirs + the archive are gone; all 5x5 lightweight state-machine files survived"
