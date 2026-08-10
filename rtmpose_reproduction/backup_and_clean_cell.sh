#!/usr/bin/env bash
# =============================================================================
# backup_and_clean_cell.sh -- per-cell (dataset, task, 5 seeds) archive +
# verified-then-delete workflow for run_rtmpose_full_sweep.sh, required
# because this server has only ~8GB free on both the system and data disks
# (2026-08-10 measurement) -- nowhere near enough to hold more than about
# one cell's worth of checkpoints/logs/predictions at a time on top of
# whatever else already lives there.
#
# Two subcommands, deliberately SEPARATE so deletion can never happen
# without an already-verified local backup in hand:
#
#   backup <DATASET> <TASK>
#       Verifies the cell's 5 seeds are all complete in the results TSV,
#       tars up everything for that cell (work_dir incl. final checkpoint,
#       config, provenance, predictions, per-image CSV, summary, training
#       log) into one archive, prints the archive's own SHA-256 and the
#       exact scp command to pull it to your local machine. Deletes
#       NOTHING. Safe to run at any time, repeatable.
#
#   clean <DATASET> <TASK> <CONFIRMED_SHA256>
#       Only deletes the large checkpoint files (work_dir contents) for a
#       cell already backed up -- but ONLY if <CONFIRMED_SHA256> (which
#       YOU must have independently computed from the LOCAL copy, on your
#       own machine, after actually verifying it opens/extracts/looks
#       right) matches the archive's own SHA-256 as still present on the
#       server. This forces you to have actually looked at the local copy
#       before anything is deleted -- there is no "just trust the backup
#       happened" shortcut. Keeps the lightweight results TSV, per-image
#       CSV, summary/provenance JSON and shared-artifact manifests on the
#       server (small, needed for run_rtmpose_full_sweep.sh's own resume
#       logic) -- only the large checkpoint/log files are removed.
#
# Required sequence (never skip a step, never reorder):
#   train -> infer -> score (run_rtmpose_full_sweep.sh)
#     -> backup_and_clean_cell.sh backup <DATASET> <TASK>
#     -> scp the printed archive to your local machine
#     -> LOCALLY: tar -tzf <archive> | wc -l (sanity), extract, spot-check
#        a checkpoint file opens, sha256sum -c the manifest inside
#     -> backup_and_clean_cell.sh clean <DATASET> <TASK> <the sha256 you
#        just computed LOCALLY, not copy-pasted from the server's own output>
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
  for SEED in "${SEEDS[@]}"; do
    local run_name="${DATASET}_${TASK}_seed${SEED}_run"
    local summary_json="$ARTIFACT_ROOT/${run_name}_summary.json"
    [ -f "$summary_json" ] || { echo "ERROR: $summary_json missing despite a TSV row -- inconsistent state, do not proceed; investigate before backing up." >&2; exit 1; }
  done
  echo "[OK] ${DATASET}/${TASK}: all 5 seeds recorded and their summary.json files present"
}

do_backup() {
  verify_cell_complete
  mkdir -p "$BACKUP_ROOT"
  local archive_name="${DATASET}_${TASK}_5seed_$(date +%Y%m%d).tar"
  local archive_path="$BACKUP_ROOT/$archive_name"
  local manifest_path="$BACKUP_ROOT/${archive_name%.tar}.sha256.tsv"

  if [ -f "$archive_path" ]; then
    echo "ERROR: $archive_path already exists -- remove it first if you intend to rebuild it (never overwritten automatically)." >&2
    exit 1
  fi

  local staging
  staging="$(mktemp -d)"
  trap 'rm -rf "$staging"' EXIT

  echo "=== staging ${DATASET}/${TASK}'s 5 seeds for archiving ==="
  for SEED in "${SEEDS[@]}"; do
    local run_name="${DATASET}_${TASK}_seed${SEED}_run"
    local dest="$staging/$run_name"
    mkdir -p "$dest"
    [ -d "$ARTIFACT_ROOT/$run_name" ] && cp -r "$ARTIFACT_ROOT/$run_name" "$dest/work_dir"
    for suffix in provenance.json predictions.json per_image.csv summary.json; do
      local f="$ARTIFACT_ROOT/${run_name}_${suffix}"
      [ -f "$f" ] && cp "$f" "$dest/"
    done
    local cfg="$ARTIFACT_ROOT/configs/${run_name}.py"
    [ -f "$cfg" ] && cp "$cfg" "$dest/"
  done
  # Shared per-cell artifacts (split/COCO jsons + manifest) -- only the
  # UCL/BPD case reuses the canary's own differently-named files, handled
  # separately below so they are still included in the archive.
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

  echo "=== building archive: $archive_path ==="
  tar -cf "$archive_path" -C "$staging" .

  local n_ckpts
  n_ckpts=$(find "$staging" -name '*.pth' | wc -l)
  echo "[OK] archive contains $n_ckpts checkpoint file(s) (expect 5, one final checkpoint per seed)"
  if [ "$n_ckpts" -ne 5 ]; then
    echo "WARNING: expected exactly 5 checkpoints (one per seed), found $n_ckpts -- inspect before trusting this archive." >&2
  fi

  sha256sum "$archive_path" > "$manifest_path"
  local archive_sha256
  archive_sha256=$(awk '{print $1}' "$manifest_path")

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
  echo "  3. LOCALLY: tar -tf $archive_name | head   # sanity-check contents"
  echo "  4. LOCALLY: tar -xf $archive_name -C /some/dir && spot-check at least"
  echo "     one seed's work_dir/*.pth actually opens (e.g. via torch.load) and"
  echo "     the per_image.csv/summary.json read back sensibly."
  echo "  5. Only once satisfied, run:"
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

  echo "=== deleting ONLY work_dir (checkpoint + training logs) for ${DATASET}/${TASK}'s 5 seeds ==="
  echo "(keeping: results TSV, config, provenance JSON, predictions JSON, per-image CSV, summary"
  echo " JSON, shared-artifact manifests -- run_rtmpose_full_sweep.sh's own run_artifacts_status()"
  echo " state machine requires ALL FIVE of {config, provenance, predictions, per-image CSV,"
  echo " summary} to still be present to correctly report a cell as 'complete' on a future"
  echo " invocation; deleting any of those (not just the checkpoint) would make it see this cell"
  echo " as 'inconsistent' and hard-stop. Only work_dir/*.pth + logs are large enough to matter"
  echo " for this server's disk budget and are also the only files NOT already duplicated by"
  echo " something else already being kept.)"
  for SEED in "${SEEDS[@]}"; do
    local run_name="${DATASET}_${TASK}_seed${SEED}_run"
    local work_dir="$ARTIFACT_ROOT/$run_name"
    if [ -d "$work_dir" ]; then
      rm -rf "$work_dir"
      echo "  removed: $work_dir"
    fi
  done

  echo ""
  echo "[COMPLETE] ${DATASET}/${TASK} work_dirs (checkpoints + logs) removed from the server."
  echo "All 5 state-machine-required files (config/provenance/predictions/per-image CSV/summary)"
  echo "remain, so run_rtmpose_full_sweep.sh will still correctly report this cell as 'complete'"
  echo "via run_artifacts_status() and continue to the next cell without re-triggering training."
}

case "$SUBCOMMAND" in
  backup) do_backup ;;
  clean)  do_clean "$3" ;;
esac
