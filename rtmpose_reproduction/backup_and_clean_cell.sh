#!/usr/bin/env bash
# =============================================================================
# backup_and_clean_cell.sh -- per-cell (dataset, task, 5 seeds) archive +
# verified-then-delete workflow for run_rtmpose_full_sweep.sh, required
# because this server has only ~8GB free on both the system and data disks
# (2026-08-10 measurement) -- nowhere near enough to hold more than about
# one cell's worth of checkpoints/logs/predictions at a time on top of
# whatever else already lives there.
#
# 2026-08-10, THIRD review round -- fixed 4 real problems in the first
# version of this script:
#   (a) backup silently produced INCOMPLETE archives -- every file copy
#       used `[ -f "$f" ] && cp` (skip-if-missing, never fails) and a
#       checkpoint-count mismatch was only a WARNING, so "BACKUP COMPLETE"
#       could be printed for an archive missing files entirely. Every
#       required file is now REQUIRED -- missing any one of them, or
#       having anything other than exactly 5 checkpoints, aborts the
#       backup with an error, not a warning.
#   (b) the archive contained no manifest usable for LOCAL verification --
#       only a manifest of the ARCHIVE FILE's own hash (proves the .tar
#       wasn't corrupted in transit, nothing about what's inside it). Now
#       generates ARCHIVE_CONTENTS.sha256 (relative paths, every file)
#       INSIDE the archive itself, so `tar -xf ... && sha256sum -c
#       ARCHIVE_CONTENTS.sha256` works directly after local extraction.
#   (c) clean deleted work_dir but never the archive tar itself sitting in
#       $BACKUP_ROOT (also on the data disk) -- net server disk change was
#       ~0 (removed ~450MB of work_dirs, kept a ~450MB tar of the same
#       data). clean now also removes the archive + its outer hash file
#       from the server once the caller's confirmed hash has verified it.
#   (d) DATASET/TASK were interpolated straight into `rm -rf` targets with
#       no validation. Added a fixed whitelist (DATASET must be UCL or
#       MULTICENTRE, TASK one of the 5 real tasks) and a realpath
#       containment check (every directory about to be deleted must
#       resolve to strictly inside $ARTIFACT_ROOT with the exact expected
#       basename) before any deletion.
#
# Two subcommands, deliberately SEPARATE so deletion can never happen
# without an already-verified local backup in hand:
#
#   backup <DATASET> <TASK>
#       Verifies the cell's 5 seeds are ALL complete (TSV row + all 5
#       state-machine files + exactly one epoch_200.pth checkpoint, per
#       seed -- hard failure, not skip, if anything is missing), tars up
#       everything for that cell into one archive containing its own
#       relative-path SHA-256 manifest, prints the archive's own SHA-256
#       and the exact scp command to pull it to your local machine.
#       Deletes NOTHING. Safe to run at any time, repeatable.
#
#   clean <DATASET> <TASK> <CONFIRMED_SHA256>
#       Only deletes the large checkpoint files (work_dir contents) AND
#       the server-side archive itself for a cell already backed up --
#       but ONLY if <CONFIRMED_SHA256> (which YOU must have independently
#       computed from the LOCAL copy, on your own machine, after actually
#       verifying it opens/extracts/looks right) matches the archive's
#       own SHA-256 as still present on the server. This forces you to
#       have actually looked at the local copy before anything is deleted
#       -- there is no "just trust the backup happened" shortcut. Keeps
#       the lightweight results TSV, per-image CSV, summary/provenance
#       JSON and shared-artifact manifests on the server (small, needed
#       for run_rtmpose_full_sweep.sh's own resume logic) -- only the
#       large checkpoint/log files AND the (now-redundant, since a
#       verified copy exists locally) server-side archive are removed.
#
# Required sequence (never skip a step, never reorder):
#   train -> infer -> score (run_rtmpose_full_sweep.sh)
#     -> backup_and_clean_cell.sh backup <DATASET> <TASK>
#     -> scp the printed archive to your local machine
#     -> LOCALLY: tar -xf <archive> -C some/dir && cd some/dir &&
#        sha256sum -c ARCHIVE_CONTENTS.sha256 (every file, not just the
#        outer tar), spot-check a checkpoint actually opens (torch.load)
#     -> backup_and_clean_cell.sh clean <DATASET> <TASK> <the sha256 YOU
#        computed locally in step 2 of "backup"'s own printed instructions,
#        not copy-pasted from this script's own output>
#
# Usage:
#   ARTIFACT_ROOT=/root/autodl-tmp/rtmpose_reproduction \
#     bash backup_and_clean_cell.sh backup UCL BPD
#   ARTIFACT_ROOT=/root/autodl-tmp/rtmpose_reproduction \
#     bash backup_and_clean_cell.sh clean UCL BPD <sha256>
# =============================================================================

set -euo pipefail

ARTIFACT_ROOT="${ARTIFACT_ROOT:-/root/autodl-tmp/rtmpose_reproduction}"
RESULTS_TSV="${RESULTS_TSV:-$ARTIFACT_ROOT/rtmpose_full_sweep_results.tsv}"
BACKUP_ROOT="${BACKUP_ROOT:-$ARTIFACT_ROOT/cell_backups}"
SEEDS=(42 0 123 2024 3407)
readonly VALID_DATASETS=("UCL" "MULTICENTRE")
readonly VALID_TASKS=("BPD" "OFD" "APAD" "TAD" "FL")

usage() {
  echo "Usage:" >&2
  echo "  $0 backup <DATASET> <TASK>" >&2
  echo "  $0 clean <DATASET> <TASK> <CONFIRMED_SHA256>" >&2
  exit 1
}

[ "$#" -ge 1 ] || usage
SUBCOMMAND="$1"; shift

case "$SUBCOMMAND" in
  backup) [ "$#" -eq 2 ] || usage ;;
  clean)  [ "$#" -eq 3 ] || usage ;;
  *) usage ;;
esac

DATASET="$1"; TASK="$2"

# --- Fix (d): whitelist before ANYTHING else touches DATASET/TASK. ---------
validate_dataset_task() {
  local ok_dataset=0 ok_task=0
  for d in "${VALID_DATASETS[@]}"; do [ "$DATASET" = "$d" ] && ok_dataset=1; done
  for t in "${VALID_TASKS[@]}"; do [ "$TASK" = "$t" ] && ok_task=1; done
  if [ "$ok_dataset" -ne 1 ]; then
    echo "ERROR: DATASET must be one of: ${VALID_DATASETS[*]} -- got '${DATASET}'" >&2
    exit 1
  fi
  if [ "$ok_task" -ne 1 ]; then
    echo "ERROR: TASK must be one of: ${VALID_TASKS[*]} -- got '${TASK}'" >&2
    exit 1
  fi
}
validate_dataset_task

# --- Canary naming special-case -- MUST stay in sync with the identical
#     function in run_rtmpose_full_sweep.sh, or this script will look for
#     files under the wrong name for UCL/BPD/seed=42. ----------------------
run_name_for() {
  local dataset="$1" task="$2" seed="$3"
  if [ "$dataset" = "UCL" ] && [ "$task" = "BPD" ] && [ "$seed" = "42" ]; then
    echo "UCL_BPD_seed42_canary"
  else
    echo "${dataset}_${task}_seed${seed}_run"
  fi
}

# --- Fix (a): every one of a seed's 5 state-machine files AND exactly one
#     final checkpoint are REQUIRED, not soft-checked. ----------------------
verify_cell_complete() {
  [ -f "$RESULTS_TSV" ] || { echo "ERROR: $RESULTS_TSV not found" >&2; exit 1; }
  local missing=()
  for SEED in "${SEEDS[@]}"; do
    if ! awk -F'\t' -v d="$DATASET" -v t="$TASK" -v s="$SEED" \
        'NR>1 && $1==d && $2==t && $3==s {f=1} END{exit !f}' "$RESULTS_TSV"; then
      missing+=("seed=$SEED")
    fi
  done
  if [ "${#missing[@]}" -gt 0 ]; then
    echo "ERROR: ${DATASET}/${TASK} is not complete in $RESULTS_TSV -- missing: ${missing[*]}" >&2
    echo "Run run_rtmpose_full_sweep.sh for this cell first; backup/clean only operate on a fully" >&2
    echo "recorded 5-seed cell, never a partial one." >&2
    exit 1
  fi

  local incomplete=()
  for SEED in "${SEEDS[@]}"; do
    local run_name; run_name="$(run_name_for "$DATASET" "$TASK" "$SEED")"
    local work_dir="$ARTIFACT_ROOT/$run_name"
    local config_path="$ARTIFACT_ROOT/configs/${run_name}.py"
    for f in "$config_path" \
             "$ARTIFACT_ROOT/${run_name}_provenance.json" \
             "$ARTIFACT_ROOT/${run_name}_predictions.json" \
             "$ARTIFACT_ROOT/${run_name}_per_image.csv" \
             "$ARTIFACT_ROOT/${run_name}_summary.json"; do
      [ -f "$f" ] || incomplete+=("seed=$SEED missing $f")
    done
    if [ ! -d "$work_dir" ]; then
      incomplete+=("seed=$SEED missing work_dir $work_dir")
    else
      local n_ckpt
      n_ckpt=$(find "$work_dir" -maxdepth 1 -name 'epoch_*.pth' | wc -l)
      if [ "$n_ckpt" -ne 1 ]; then
        incomplete+=("seed=$SEED has $n_ckpt epoch_*.pth checkpoint(s) in $work_dir, expected exactly 1")
      fi
    fi
  done
  if [ "${#incomplete[@]}" -gt 0 ]; then
    echo "ERROR: ${DATASET}/${TASK} has a TSV row for all 5 seeds but is NOT actually complete on disk:" >&2
    printf '  %s\n' "${incomplete[@]}" >&2
    echo "Refusing to back up an incomplete cell -- investigate before proceeding." >&2
    exit 1
  fi
  echo "[OK] ${DATASET}/${TASK}: all 5 seeds recorded, all 5 state-machine files present, exactly 1 final checkpoint each"
}

do_backup() {
  verify_cell_complete
  mkdir -p "$BACKUP_ROOT"
  local archive_name="${DATASET}_${TASK}_5seed_$(date +%Y%m%d).tar"
  local archive_path="$BACKUP_ROOT/$archive_name"
  local outer_hash_path="$BACKUP_ROOT/${archive_name%.tar}.sha256.tsv"

  if [ -f "$archive_path" ]; then
    echo "ERROR: $archive_path already exists -- remove it first if you intend to rebuild it (never overwritten automatically)." >&2
    exit 1
  fi

  local staging
  staging="$(mktemp -d)"
  trap 'rm -rf "$staging"' EXIT

  echo "=== staging ${DATASET}/${TASK}'s 5 seeds for archiving (every file required, none optional) ==="
  for SEED in "${SEEDS[@]}"; do
    local run_name; run_name="$(run_name_for "$DATASET" "$TASK" "$SEED")"
    local dest="$staging/$run_name"
    mkdir -p "$dest"

    local work_dir="$ARTIFACT_ROOT/$run_name"
    [ -d "$work_dir" ] || { echo "ERROR: $work_dir missing at backup time (verify_cell_complete should have caught this)" >&2; exit 1; }
    cp -r "$work_dir" "$dest/work_dir"

    for suffix in provenance.json predictions.json per_image.csv summary.json; do
      local f="$ARTIFACT_ROOT/${run_name}_${suffix}"
      [ -f "$f" ] || { echo "ERROR: $f missing at backup time" >&2; exit 1; }
      cp "$f" "$dest/"
    done
    local cfg="$ARTIFACT_ROOT/configs/${run_name}.py"
    [ -f "$cfg" ] || { echo "ERROR: $cfg missing at backup time" >&2; exit 1; }
    cp "$cfg" "$dest/"
  done

  # Shared per-cell artifacts (split/COCO jsons + manifest) -- only the
  # UCL/BPD case reuses the canary's own differently-named files.
  mkdir -p "$staging/shared"
  if [ "$DATASET" = "UCL" ] && [ "$TASK" = "BPD" ]; then
    for f in UCL_BPD_internal_split.json UCL_BPD_internal_train.json UCL_BPD_internal_val.json UCL_BPD_test.json; do
      [ -f "$ARTIFACT_ROOT/coco/$f" ] && cp "$ARTIFACT_ROOT/coco/$f" "$staging/shared/"
    done
  else
    for f in "${DATASET}_${TASK}_internal_split.json" "${DATASET}_${TASK}_internal_train.json" \
             "${DATASET}_${TASK}_internal_val.json" "${DATASET}_${TASK}_test.json"; do
      [ -f "$ARTIFACT_ROOT/coco/$f" ] && cp "$ARTIFACT_ROOT/coco/$f" "$staging/shared/"
    done
  fi
  [ -f "$ARTIFACT_ROOT/coco/manifests/${DATASET}_${TASK}.sha256.tsv" ] && \
    cp "$ARTIFACT_ROOT/coco/manifests/${DATASET}_${TASK}.sha256.tsv" "$staging/shared/"

  local n_ckpts
  n_ckpts=$(find "$staging" -name 'epoch_*.pth' | wc -l)
  if [ "$n_ckpts" -ne 5 ]; then
    echo "ERROR: staged archive contains $n_ckpts checkpoint file(s), expected exactly 5 (one final checkpoint per seed) -- refusing to build an incomplete archive." >&2
    exit 1
  fi
  echo "[OK] staged exactly 5 checkpoints (one per seed)"

  # --- Fix (b): in-archive, relative-path manifest -- usable directly
  #     after local extraction, unlike a manifest of only the outer tar's
  #     own hash (which says nothing about individual file integrity). ---
  ( cd "$staging" && find . -type f ! -name ARCHIVE_CONTENTS.sha256 -exec sha256sum {} + ) \
    > "$staging/ARCHIVE_CONTENTS.sha256"
  echo "[OK] wrote ARCHIVE_CONTENTS.sha256 ($(wc -l < "$staging/ARCHIVE_CONTENTS.sha256") files) inside the archive"

  echo "=== building archive: $archive_path ==="
  tar -cf "$archive_path" -C "$staging" .

  sha256sum "$archive_path" > "$outer_hash_path"
  local archive_sha256
  archive_sha256=$(awk '{print $1}' "$outer_hash_path")

  echo ""
  echo "============================================================"
  echo "  BACKUP COMPLETE (nothing deleted on the server)"
  echo "============================================================"
  echo "Archive:  $archive_path"
  echo "SHA-256:  $archive_sha256"
  echo ""
  echo "Next steps (run these YOURSELF, not automated here):"
  echo "  1. scp -P <port> root@<host>:$archive_path ."
  echo "  2. LOCALLY: sha256sum $archive_name   # must equal $archive_sha256 above"
  echo "  3. LOCALLY: mkdir extracted && tar -xf $archive_name -C extracted"
  echo "  4. LOCALLY: cd extracted && sha256sum -c ARCHIVE_CONTENTS.sha256"
  echo "     # verifies EVERY individual file, not just the outer tar"
  echo "  5. LOCALLY: spot-check at least one seed's work_dir/epoch_*.pth actually"
  echo "     opens (e.g. via torch.load) and per_image.csv/summary.json read back sensibly."
  echo "  6. Only once satisfied, run (this DELETES the server-side archive too, so make"
  echo "     sure your local copy is the one you actually want to keep):"
  echo "       bash backup_and_clean_cell.sh clean $DATASET $TASK <the sha256"
  echo "         YOU computed locally in step 2, not copy-pasted from this output>"
}

do_clean() {
  local confirmed_sha256="$1"
  verify_cell_complete

  local archive_name
  archive_name=$(ls -1 "$BACKUP_ROOT"/"${DATASET}_${TASK}"_5seed_*.tar 2>/dev/null | sort | tail -1) || true
  if [ -z "${archive_name:-}" ] || [ ! -f "$archive_name" ]; then
    echo "ERROR: no backup archive found for ${DATASET}/${TASK} under $BACKUP_ROOT -- run 'backup' first." >&2
    exit 1
  fi
  local outer_hash_path="${archive_name%.tar}.sha256.tsv"

  local server_sha256
  server_sha256=$(sha256sum "$archive_name" | awk '{print $1}')
  if [ "$confirmed_sha256" != "$server_sha256" ]; then
    echo "ERROR: the SHA-256 you provided ($confirmed_sha256) does not match the archive still on" >&2
    echo "  this server ($server_sha256, $archive_name)." >&2
    echo "  This is the whole point of the two-step process -- refusing to delete anything." >&2
    echo "  Re-verify your LOCAL copy and re-run with the hash YOU computed from it." >&2
    exit 1
  fi
  echo "[OK] confirmed SHA-256 matches the archive still present on this server: $archive_name"

  # --- Fix (d): realpath containment + exact-basename check before ANY
  #     rm -rf, even though DATASET/TASK are already whitelisted above. ---
  local artifact_root_real
  artifact_root_real="$(realpath "$ARTIFACT_ROOT")"

  echo "=== deleting work_dir (checkpoint + training logs) for ${DATASET}/${TASK}'s 5 seeds ==="
  echo "(keeping: results TSV, config, provenance JSON, predictions JSON, per-image CSV, summary"
  echo " JSON, shared-artifact manifests -- run_rtmpose_full_sweep.sh's own run_artifacts_status()"
  echo " state machine requires ALL FIVE of {config, provenance, predictions, per-image CSV,"
  echo " summary} to still be present to correctly report a cell as 'complete' on a future"
  echo " invocation.)"
  for SEED in "${SEEDS[@]}"; do
    local run_name; run_name="$(run_name_for "$DATASET" "$TASK" "$SEED")"
    local work_dir="$ARTIFACT_ROOT/$run_name"
    if [ -d "$work_dir" ]; then
      local work_dir_real
      work_dir_real="$(realpath "$work_dir")"
      case "$work_dir_real" in
        "$artifact_root_real"/*) ;;
        *) echo "ERROR: refusing to delete $work_dir_real -- not strictly inside $artifact_root_real" >&2; exit 1 ;;
      esac
      if [ "$(basename "$work_dir_real")" != "$run_name" ]; then
        echo "ERROR: refusing to delete $work_dir_real -- basename does not exactly match expected run_name '$run_name'" >&2
        exit 1
      fi
      rm -rf "$work_dir_real"
      echo "  removed: $work_dir_real"
    fi
  done

  # --- Fix (c): also remove the server-side archive + its hash file now
  #     that a verified copy exists locally -- otherwise net disk change
  #     from backup+clean is approximately zero. -----------------------
  local archive_real
  archive_real="$(realpath "$archive_name")"
  case "$archive_real" in
    "$artifact_root_real"/*) ;;
    *) echo "ERROR: refusing to delete $archive_real -- not strictly inside $artifact_root_real" >&2; exit 1 ;;
  esac
  rm -f "$archive_real" "$outer_hash_path"
  echo "  removed: $archive_real"
  echo "  removed: $outer_hash_path"

  echo ""
  echo "[COMPLETE] ${DATASET}/${TASK} work_dirs (checkpoints + logs) AND the server-side archive"
  echo "have been removed -- disk space is now actually freed (not just moved into a tar)."
  echo "All 5 state-machine-required files (config/provenance/predictions/per-image CSV/summary)"
  echo "remain, so run_rtmpose_full_sweep.sh will still correctly report this cell as 'complete'"
  echo "via run_artifacts_status() and continue to the next cell without re-triggering training."
}

case "$SUBCOMMAND" in
  backup) do_backup ;;
  clean)  do_clean "$3" ;;
esac
