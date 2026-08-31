#!/usr/bin/env bash
# =============================================================================
# rtmpose_common.sh -- shared helpers sourced by BOTH run_rtmpose_full_sweep.sh
# and backup_and_clean_cell.sh. 2026-08-10, fourth review round: a prior round
# already had to fix one real bug caused by these two standalone scripts each
# keeping their OWN copy of run_name_for() (the UCL/BPD/seed=42 canary naming
# special-case) -- "keep in sync by hand" is exactly the kind of thing that
# silently drifts. This file exists so there is exactly one copy of every
# naming/path convention shared between the two scripts, not two that can
# diverge again.
#
# Not standalone-executable -- `source` this file after `set -euo pipefail`
# in the calling script. Deliberately has no side effects at source time.
# =============================================================================

# --- Canary naming special-case: the seed-42 UCL/BPD canary
#     (run_rtmpose_canary.sh) used the "_canary" suffix throughout
#     (UCL_BPD_seed42_canary_summary.json, work_dir "UCL_BPD_seed42_canary/",
#     etc), NOT the generic "_run" suffix every other (dataset,task,seed)
#     uses. Every place that computes a run's file naming MUST go through
#     this one function. -------------------------------------------------
run_name_for() {
  local dataset="$1" task="$2" seed="$3"
  if [ "$dataset" = "UCL" ] && [ "$task" = "BPD" ] && [ "$seed" = "42" ]; then
    echo "UCL_BPD_seed42_canary"
  else
    echo "${dataset}_${task}_seed${seed}_run"
  fi
}

# --- Per-(dataset,task) shared COCO artifact paths (split/train/val/test +
#     their sha256 manifest). UCL/BPD reuses the canary's own already-
#     archived, already-approved files byte-for-byte instead of generating a
#     "matching" copy. This function only COMPUTES paths -- generating the
#     files (if missing) remains run_rtmpose_full_sweep.sh's own
#     responsibility (ensure_shared_artifacts), since that's the only script
#     that should ever create them. Echoes
#     "split|train|val|test|manifest". -----------------------------------
shared_json_paths_for() {
  local artifact_root="$1" dataset="$2" task="$3"
  local split_json train_json val_json test_json manifest
  if [ "$dataset" = "UCL" ] && [ "$task" = "BPD" ]; then
    split_json="$artifact_root/coco/UCL_BPD_internal_split.json"
    train_json="$artifact_root/coco/UCL_BPD_internal_train.json"
    val_json="$artifact_root/coco/UCL_BPD_internal_val.json"
    test_json="$artifact_root/coco/UCL_BPD_test.json"
  else
    split_json="$artifact_root/coco/${dataset}_${task}_internal_split.json"
    train_json="$artifact_root/coco/${dataset}_${task}_internal_train.json"
    val_json="$artifact_root/coco/${dataset}_${task}_internal_val.json"
    test_json="$artifact_root/coco/${dataset}_${task}_test.json"
  fi
  manifest="$artifact_root/coco/manifests/${dataset}_${task}.sha256.tsv"
  echo "${split_json}|${train_json}|${val_json}|${test_json}|${manifest}"
}

# --- The 3 excluded-image audit logs convert_csv_to_coco.py writes per
#     (dataset,task) -- same "${dataset}_${task}_..._excluded.json" naming
#     convention for every cell, including UCL/BPD's canary-generated ones
#     (run_rtmpose_canary.sh wrote UCL_BPD_internal_train_excluded.json etc,
#     which already matches this pattern -- no special case needed here).
#     Echoes "train_excluded|val_excluded|test_excluded". ------------------
excluded_log_paths_for() {
  local artifact_root="$1" dataset="$2" task="$3"
  echo "$artifact_root/coco/${dataset}_${task}_internal_train_excluded.json|$artifact_root/coco/${dataset}_${task}_internal_val_excluded.json|$artifact_root/coco/${dataset}_${task}_test_excluded.json"
}

# --- Hard verification that a work_dir's final checkpoint is genuinely
#     epoch_<expected_epoch>.pth: 2026-08-10 fourth review round finding --
#     the previous checkpoint-count check (`find ... -name 'epoch_*.pth' |
#     wc -l` == 1) would have passed just as happily for a lone
#     epoch_195.pth as for epoch_200.pth, silently backing up a run that
#     never actually finished its full 200-epoch schedule. Requires:
#       - $work_dir/epoch_<N>.pth exists and is non-empty;
#       - no OTHER epoch_*.pth file exists in $work_dir (exactly one,
#         and it must be the right one, not just "exactly one");
#       - $work_dir/last_checkpoint exists and its content points at that
#         same file (cheap extra corroboration, not a substitute for the
#         two checks above since last_checkpoint could itself be stale).
#       - the checkpoint is a structurally valid PyTorch ZIP archive, every
#         member passes CRC validation, and a data.pkl payload is present.
#     This checks structural integrity, not whether the tensors are
#     semantically correct for a particular architecture; subsequent real
#     inference provides that stronger end-to-end evidence.
#     Prints the verified checkpoint path on success (stdout only -- safe
#     to capture via $(...)); exits 1 with a stderr explanation otherwise. -
verify_final_checkpoint() {
  local work_dir="$1" expected_epoch="$2"
  local expected_name="epoch_${expected_epoch}.pth"
  local expected_path="$work_dir/$expected_name"

  [ -f "$expected_path" ] || {
    echo "ERROR: $expected_path not found -- this run's final checkpoint is missing or was never $expected_name" >&2
    exit 1
  }
  [ -s "$expected_path" ] || {
    echo "ERROR: $expected_path exists but is empty (0 bytes) -- refusing to trust a zero-byte checkpoint" >&2
    exit 1
  }

  local all_epoch_ckpts=()
  while IFS= read -r -d '' f; do all_epoch_ckpts+=("$f"); done \
    < <(find "$work_dir" -maxdepth 1 -name 'epoch_*.pth' -print0)
  if [ "${#all_epoch_ckpts[@]}" -ne 1 ] || [ "$(basename "${all_epoch_ckpts[0]}")" != "$expected_name" ]; then
    echo "ERROR: $work_dir contains epoch checkpoint(s) other than exactly $expected_name: ${all_epoch_ckpts[*]}" >&2
    exit 1
  fi

  local pointer="$work_dir/last_checkpoint"
  [ -s "$pointer" ] || {
    echo "ERROR: required final-checkpoint pointer is missing or empty: $pointer" >&2
    exit 1
  }
  local pointed_name
  pointed_name="$(basename "$(cat "$pointer")")"
  if [ "$pointed_name" != "$expected_name" ]; then
    echo "ERROR: $pointer points at $pointed_name, not $expected_name" >&2
    exit 1
  fi

  local checkpoint_python="${PY:-python3}"
  "$checkpoint_python" - "$expected_path" <<'PY'
from pathlib import Path
import sys
import zipfile

path = Path(sys.argv[1])
if not zipfile.is_zipfile(path):
    raise SystemExit(f"ERROR: {path} is not a structurally valid PyTorch ZIP checkpoint")
with zipfile.ZipFile(path) as archive:
    bad_member = archive.testzip()
    members = archive.namelist()
if bad_member is not None:
    raise SystemExit(f"ERROR: {path} has corrupt ZIP member {bad_member}")
if not any(name == "data.pkl" or name.endswith("/data.pkl") for name in members):
    raise SystemExit(f"ERROR: {path} has no PyTorch data.pkl payload")
PY

  echo "$expected_path"
}
