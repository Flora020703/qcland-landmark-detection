#!/usr/bin/env bash
# =============================================================================
# run_rtmpose_full_sweep.sh -- the full RTMPose-s programme: 2 datasets x 5
# tasks x 5 seeds = 50 runs, structured like
# baseline_reproduction/run_hrnet_512_fixed_5seed.sh (resumable via a
# results TSV, refuses to silently resume a partial/inconsistent run,
# disk-safety preflight) -- NOT a re-run of run_rtmpose_canary.sh's seed-42
# UCL-BPD canary, whose result is pre-seeded into the results TSV below.
#
# 2026-08-10, SECOND hardening round (a relayed review of the FIRST unified
# version found 3 more real gaps; this version closes all of them plus the
# disk-budget reality this server actually has):
#
# Fix A ("完成状态与磁盘产物可能不一致"): `is_recorded()` alone (TSV-only)
# could skip a cell whose on-disk artifacts are actually gone/incomplete,
# or -- worse -- silently RETRAIN a cell whose training/scoring already
# finished but crashed before the TSV append (work_dir + summary.json both
# present, but no TSV row: the old "partial dir" check only fired when
# summary.json was ABSENT, so this exact case fell through to retraining
# and overwriting a good, complete result). Replaced with a proper 4-state
# check (`run_artifacts_status`): fresh / recoverable / complete /
# inconsistent -- "recoverable" (all 5 output files present, no TSV row)
# is recovered directly from the existing summary.json, never retrained;
# "inconsistent" (some but not all expected files present, regardless of
# TSV state) is a hard stop requiring manual inspection.
#
# Fix B ("配置断言远没有达到注释宣称的范围"): the previous version's
# config-verification block checked 5 fields and claimed much more in
# prose. Every field make_config.py's own TEMPLATE actually sets is now
# asserted explicitly -- architecture (CSPNeXt/RTMCCHead exact
# hyperparameters), SimCC codec settings, batch size, optimizer/scheduler,
# checkpoint policy (save_last/max_keep_ckpts/no save_best), augmentation
# pipeline ordering, and the internal-train/val/Test annotation paths
# actually wired into each dataloader -- see the exhaustive `checks` list
# in `verify_config()` below, each one read directly from
# rtmpose_reproduction/make_config.py's real TEMPLATE, not assumed.
#
# Fix C ("UCL BPD manifest仍不是canary时期的原始锁定值"): the seed-42
# canary's own archived deliverables (docs/supervisor_meeting_report_
# 2026-08-08.md section 0.1.2) do NOT include the raw split/COCO json
# files -- only the checkpoint, predictions, per-image CSV, summary,
# provenance, config and overlays were ever archived. A byte-hash
# comparison against an "original" split-json copy is therefore not
# achievable (no such copy exists anywhere). Instead, `verify_ucl_bpd_
# matches_canary()` cross-validates the CURRENT UCL/BPD Test COCO json's
# own GT coordinates, per filename, against the ALREADY-ARCHIVED and
# ALREADY-APPROVED `UCL_BPD_seed42_canary_per_image.csv`'s own gt0/gt1
# columns -- exact match required for all 49 images. This answers the
# real question ("is the Test set currently on disk content-identical to
# what the approved canary actually scored") using an artifact that
# genuinely exists, rather than one that was never saved.
#
# Fix D (disk budget): this server has ~8GB free on BOTH the system and
# data disks (2026-08-10 measurement) -- nowhere near enough to run the
# remaining 45 RTMPose runs back-to-back (~400-500MB per 5-seed cell once
# logs/predictions/provenance are included, on top of whatever else is on
# the disk). MIN_FREE_GB's job is to stop BEFORE a mid-training crash, not
# to be relied on for capacity planning. The sweep now pauses after EVERY
# cell (STOP_AFTER_EACH_CELL=1 by default, matching PREFLIGHT_ONLY's
# existing safe-by-default convention), not just after UCL/BPD, and
# NEVER deletes any checkpoint/log/prediction itself -- see the separate
# backup_and_clean_cell.sh, which enforces backup-verify-THEN-delete and
# is never invoked automatically by this script.
#
# Supervisor's exact instruction (2026-08-10, recorded verbatim in
# docs/supervisor_meeting_report_2026-08-08.md's 阶段 3 section): "Please
# proceed with the remaining four UCL BPD seeds using exactly the same
# configuration, so that we can obtain a reliable five-seed mean and
# seed-level standard deviation. Once we have the five-seed BPD result, we
# can proceed with the remaining measurements using the same RTMPose-s
# configuration." UCL/BPD is still processed first (CELLS order below);
# the per-cell pause additionally satisfies this without a BPD-specific
# special case in the control flow.
#
# Prerequisites (same as run_rtmpose_canary.sh -- see ENVIRONMENT.md):
#   - a pinned MMPose/MMEngine/MMCV install in its own venv, verified importable
#   - the CSPNeXt-s checkpoint downloaded locally (PRETRAINED_CKPT_PATH)
#   - run_rtmpose_canary.sh's seed-42 UCL-BPD canary already complete (its
#     summary.json AND per-image CSV are read directly, never re-produced)
# =============================================================================

set -euo pipefail

export CUBLAS_WORKSPACE_CONFIG=:4096:8

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:?set PY to the RTMPose venv python interpreter}"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/root/autodl-tmp/rtmpose_reproduction}"
PRETRAINED_CKPT_PATH="${PRETRAINED_CKPT_PATH:?set PRETRAINED_CKPT_PATH to the locally-downloaded CSPNeXt-s checkpoint file, see ENVIRONMENT.md}"
MMPOSE_TRAIN_TOOL="${MMPOSE_TRAIN_TOOL:?set MMPOSE_TRAIN_TOOL to the installed mmpose repo tools/train.py path}"

# NOT environment-overridable -- "exactly the same configuration" as the
# approved seed-42 canary means 200 epochs, full stop.
readonly MAX_EPOCHS=200
readonly EXPECTED_PRETRAINED_SHA256="aa7d9335bf422ad02a803e36f357dfc6abb807eca42d79e8b3b6e7c5bd1f446b"

RESULTS_TSV="${RESULTS_TSV:-$ARTIFACT_ROOT/rtmpose_full_sweep_results.tsv}"
CANARY_SUMMARY_JSON="$ARTIFACT_ROOT/UCL_BPD_seed42_canary_summary.json"
CANARY_PER_IMAGE_CSV="$ARTIFACT_ROOT/UCL_BPD_seed42_canary_per_image.csv"
MANIFEST_DIR="$ARTIFACT_ROOT/coco/manifests"
MIN_FREE_GB="${MIN_FREE_GB:-6}"
# Default ON: pause after every (dataset, task) cell's 5 seeds, requiring an
# explicit override to continue -- this server's real ~8GB-free disk budget
# cannot hold more than roughly one cell's worth of artifacts at a time
# alongside everything else already on it (see Fix D above). Set to 1 to
# process every remaining cell back-to-back WITHOUT pausing -- only do this
# if you have independently confirmed enough free disk for the whole run.
STOP_AFTER_EACH_CELL="${STOP_AFTER_EACH_CELL:-1}"

SEEDS=(42 0 123 2024 3407)
# (dataset, task, anatomy) -- UCL BPD FIRST, matching the supervisor's own
# staged instruction (BPD five-seed result reviewed before the rest starts).
CELLS=(
  "UCL:BPD:Head"
  "UCL:OFD:Head"
  "UCL:APAD:Abdomen"
  "UCL:TAD:Abdomen"
  "UCL:FL:Femur"
  "MULTICENTRE:BPD:Head"
  "MULTICENTRE:OFD:Head"
  "MULTICENTRE:APAD:Abdomen"
  "MULTICENTRE:TAD:Abdomen"
  "MULTICENTRE:FL:Femur"
)

cd "$SCRIPT_DIR"
mkdir -p "$ARTIFACT_ROOT" "$MANIFEST_DIR"

echo "=== [setup] local pure-Python test suite (no mmpose required) ==="
"$PY" test_geometry.py
"$PY" test_endpoint_order.py
"$PY" test_convert_csv_to_coco.py
"$PY" test_evaluate_rtmpose_fixed.py
"$PY" test_fetal_augment.py
"$PY" test_low_level_decode.py

echo "=== [setup] verify mmpose/mmengine/mmcv import and record versions ==="
"$PY" - <<'PY'
import mmcv, mmengine, mmpose
print(f"mmcv={mmcv.__version__} mmengine={mmengine.__version__} mmpose={mmpose.__version__}")
PY

echo "=== [setup] verify pretrained checkpoint SHA-256 (fixed, not per-run) ==="
GOT_SHA256=$(sha256sum "$PRETRAINED_CKPT_PATH" | awk '{print $1}')
if [ "$GOT_SHA256" != "$EXPECTED_PRETRAINED_SHA256" ]; then
  echo "ERROR: pretrained checkpoint SHA-256 mismatch." >&2
  echo "  expected (seed-42 canary provenance): $EXPECTED_PRETRAINED_SHA256" >&2
  echo "  got:                                  $GOT_SHA256" >&2
  echo "  This is NOT the same weight file the approved canary used -- refusing to proceed." >&2
  exit 1
fi
echo "[OK] pretrained checkpoint SHA-256 matches the seed-42 canary's own provenance record"

# --- Fix C: UCL/BPD's shared Test set must be content-identical to what the
#     approved canary actually scored (see header comment for why this
#     replaces a raw split-json hash comparison). Runs once, before any
#     other UCL/BPD artifact handling. --------------------------------------
verify_ucl_bpd_matches_canary() {
  local test_json="$ARTIFACT_ROOT/coco/UCL_BPD_test.json"
  [ -f "$test_json" ] || { echo "ERROR: $test_json not found -- required before verifying it against the approved canary." >&2; exit 1; }
  [ -f "$CANARY_PER_IMAGE_CSV" ] || { echo "ERROR: $CANARY_PER_IMAGE_CSV not found -- this is the seed-42 canary's own archived, approved scored output; required to cross-validate the current Test set." >&2; exit 1; }
  "$PY" - "$test_json" "$CANARY_PER_IMAGE_CSV" <<'PY'
import csv, json, sys
test_json_path, canary_csv_path = sys.argv[1], sys.argv[2]

coco = json.load(open(test_json_path, encoding="utf-8"))
images_by_id = {im["id"]: im["file_name"] for im in coco["images"]}
current_gt = {}
for ann in coco["annotations"]:
    fn = images_by_id[ann["image_id"]]
    kp = ann["keypoints"]
    current_gt[fn] = ((kp[0], kp[1]), (kp[3], kp[4]))

canary_gt = {}
with open(canary_csv_path, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        canary_gt[row["filename"]] = (
            (float(row["gt0_x"]), float(row["gt0_y"])),
            (float(row["gt1_x"]), float(row["gt1_y"])),
        )

if set(current_gt) != set(canary_gt):
    missing = sorted(set(canary_gt) - set(current_gt))
    extra = sorted(set(current_gt) - set(canary_gt))
    raise SystemExit(
        f"ERROR: current UCL/BPD Test set filenames do not match the approved "
        f"canary's own scored filenames -- missing={missing[:5]} extra={extra[:5]}"
    )
if len(current_gt) != 49:
    raise SystemExit(f"ERROR: expected 49 UCL/BPD Test images (per the approved canary), got {len(current_gt)}")

mismatches = [fn for fn, (g0, g1) in current_gt.items() if (g0, g1) != canary_gt[fn]]
if mismatches:
    raise SystemExit(
        f"ERROR: {len(mismatches)} image(s) have DIFFERENT GT coordinates in the "
        f"current Test json vs. the approved canary's own scored per-image CSV -- "
        f"the current Test set is NOT content-identical to what the canary actually "
        f"used: {mismatches[:5]}"
    )
print(f"[OK] current UCL/BPD Test set (n={len(current_gt)}) is content-identical, "
      f"per-image, to the approved seed-42 canary's own scored GT")
PY
}
verify_ucl_bpd_matches_canary

EXPECTED_HEADER=$'dataset\ttask\tseed\tn\tfixed_channel_mean_pct\tswap_min_mean_pct'
if [ ! -f "$RESULTS_TSV" ]; then
  printf '%s\n' "$EXPECTED_HEADER" > "$RESULTS_TSV"
elif [ "$(head -n 1 "$RESULTS_TSV")" != "$EXPECTED_HEADER" ]; then
  echo "ERROR: incompatible results header: $RESULTS_TSV" >&2
  exit 1
fi

is_recorded() {
  awk -F'\t' -v d="$1" -v t="$2" -v s="$3" 'NR>1 && $1==d && $2==t && $3==s {f=1} END{exit !f}' "$RESULTS_TSV"
}

append_result_row() {
  local dataset="$1" task="$2" seed="$3" n="$4" fixed="$5" swap="$6"
  if is_recorded "$dataset" "$task" "$seed"; then
    echo "ERROR: refusing to append a duplicate TSV row for ${dataset}/${task}/seed=${seed} (already recorded)." >&2
    exit 1
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$dataset" "$task" "$seed" "$n" "$fixed" "$swap" >> "$RESULTS_TSV"
}

# Pre-seed the already-approved UCL/BPD/seed=42 canary result rather than
# re-running it under a different file-naming scheme.
if ! is_recorded "UCL" "BPD" "42"; then
  if [ ! -f "$CANARY_SUMMARY_JSON" ]; then
    echo "ERROR: $CANARY_SUMMARY_JSON not found -- this script requires the seed-42 canary to already be complete (see run_rtmpose_canary.sh)." >&2
    exit 1
  fi
  read -r CANARY_N CANARY_FIXED CANARY_SWAP <<< "$("$PY" - "$CANARY_SUMMARY_JSON" <<'PY'
import json, sys
s = json.load(open(sys.argv[1], encoding="utf-8"))
print(s['n'], f"{s['fixed_channel_mean_pct']:.6f}", f"{s['swap_min_mean_pct']:.6f}")
PY
)"
  append_result_row "UCL" "BPD" "42" "$CANARY_N" "$CANARY_FIXED" "$CANARY_SWAP"
  echo "[OK] pre-seeded UCL/BPD/seed=42 from the already-approved canary: n=$CANARY_N, PI-NME(swap_min)=${CANARY_SWAP}%"
fi

check_disk() {
  local free_gb
  free_gb=$(df --output=avail -BG "$ARTIFACT_ROOT" | tail -1 | tr -dc '0-9')
  [ "$free_gb" -ge "$MIN_FREE_GB" ] || {
    echo "ERROR: ${free_gb}GB free, require ${MIN_FREE_GB}GB. This server's disk cannot hold many" >&2
    echo "  cells' worth of checkpoints/logs/predictions at once -- run backup_and_clean_cell.sh" >&2
    echo "  for already-completed cells before continuing (see that script's own docstring)." >&2
    exit 1; }
}

# --- generate (once) + verify (every seed) the shared per-cell
#     internal-split/Train/Val/Test COCO artifacts. -------------------------
ensure_shared_artifacts() {
  local dataset="$1" task="$2" anatomy="$3"
  local split_json="$ARTIFACT_ROOT/coco/${dataset}_${task}_internal_split.json"
  local train_json="$ARTIFACT_ROOT/coco/${dataset}_${task}_internal_train.json"
  local val_json="$ARTIFACT_ROOT/coco/${dataset}_${task}_internal_val.json"
  local test_json="$ARTIFACT_ROOT/coco/${dataset}_${task}_test.json"
  local manifest="$MANIFEST_DIR/${dataset}_${task}.sha256.tsv"
  local train_csv="$DATA_ROOT/annotations/${dataset}/${anatomy}_Train.csv"
  local test_csv="$DATA_ROOT/annotations/${dataset}/${anatomy}_Test.csv"
  local images_dir="$DATA_ROOT/images/${dataset}/${anatomy}"

  # NOTE: this function's stdout is captured by the caller via `$(...)` to
  # get its final "return value" line -- every diagnostic/status message
  # below is therefore deliberately sent to stderr (>&2 / 1>&2), never
  # plain stdout, so the capture isn't corrupted with multi-line noise.

  if [ "$dataset" = "UCL" ] && [ "$task" = "BPD" ]; then
    # Reuse the seed-42 canary's own artifacts byte-for-byte -- do not
    # regenerate even a "matching" copy. Content already cross-validated
    # against the canary's own per-image CSV above (Fix C).
    split_json="$ARTIFACT_ROOT/coco/UCL_BPD_internal_split.json"
    train_json="$ARTIFACT_ROOT/coco/UCL_BPD_internal_train.json"
    val_json="$ARTIFACT_ROOT/coco/UCL_BPD_internal_val.json"
    test_json="$ARTIFACT_ROOT/coco/UCL_BPD_test.json"
    for f in "$split_json" "$train_json" "$val_json" "$test_json"; do
      [ -f "$f" ] || { echo "ERROR: $f not found -- the seed-42 canary's own shared artifacts are required for UCL/BPD, not regenerated here." >&2; exit 1; }
    done
  fi

  if [ ! -f "$manifest" ]; then
    if [ ! -f "$split_json" ]; then
      "$PY" make_internal_val_split.py \
        --csv "$train_csv" --images-dir "$images_dir" --task "$task" \
        --out-json "$split_json" 1>&2
      "$PY" convert_csv_to_coco.py \
        --csv "$train_csv" --images-dir "$images_dir" \
        --dataset "$dataset" --task "$task" \
        --out-json "$train_json" \
        --excluded-log "$ARTIFACT_ROOT/coco/${dataset}_${task}_internal_train_excluded.json" \
        --internal-split-json "$split_json" --internal-split-part internal_train 1>&2
      "$PY" convert_csv_to_coco.py \
        --csv "$train_csv" --images-dir "$images_dir" \
        --dataset "$dataset" --task "$task" \
        --out-json "$val_json" \
        --excluded-log "$ARTIFACT_ROOT/coco/${dataset}_${task}_internal_val_excluded.json" \
        --internal-split-json "$split_json" --internal-split-part internal_val 1>&2
      "$PY" convert_csv_to_coco.py \
        --csv "$test_csv" --images-dir "$images_dir" \
        --dataset "$dataset" --task "$task" \
        --out-json "$test_json" \
        --excluded-log "$ARTIFACT_ROOT/coco/${dataset}_${task}_test_excluded.json" 1>&2
    fi

    echo "--- [content sanity] ${dataset}/${task}: internal-train/internal-val/Test disjoint ---" >&2
    "$PY" - "$train_json" "$val_json" "$test_json" 1>&2 <<'PY'
import json, sys
train_p, val_p, test_p = sys.argv[1:4]
def names(p):
    return {im["file_name"] for im in json.load(open(p, encoding="utf-8"))["images"]}
tr, va, te = names(train_p), names(val_p), names(test_p)
overlap_tv = tr & va
overlap_tt = te & (tr | va)
if overlap_tv:
    raise SystemExit(f"ERROR: internal-train/internal-val overlap ({len(overlap_tv)} images): {sorted(overlap_tv)[:5]}")
if overlap_tt:
    raise SystemExit(f"ERROR: Test overlaps internal-train/internal-val ({len(overlap_tt)} images): {sorted(overlap_tt)[:5]}")
print(f"[OK] n_internal_train={len(tr)} n_internal_val={len(va)} n_test={len(te)}, all disjoint")
PY

    {
      sha256sum "$split_json"
      sha256sum "$train_json"
      sha256sum "$val_json"
      sha256sum "$test_json"
    } > "$manifest"
    echo "[OK] recorded shared-artifact manifest: $manifest" >&2
  else
    if ! sha256sum -c "$manifest" --quiet 1>&2 2>&1; then
      echo "ERROR: shared artifact(s) for ${dataset}/${task} changed since first generation (manifest: $manifest) -- refusing to train a seed against possibly-different data than earlier seeds of this same cell used." >&2
      sha256sum -c "$manifest" 1>&2 2>&1 || true
      exit 1
    fi
  fi

  echo "$split_json|$train_json|$val_json|$test_json"
}

# --- Fix A: 4-state artifact/TSV consistency check, replacing the old
#     TSV-only is_recorded() + "work_dir exists but no summary" partial
#     check (which missed the case where work_dir AND summary.json both
#     exist but the TSV append never happened). Sets RUN_STATUS to one of
#     fresh / recoverable / complete / inconsistent. ------------------------
run_artifacts_status() {
  local dataset="$1" task="$2" seed="$3"
  local run_name="${dataset}_${task}_seed${seed}_run"
  local work_dir="$ARTIFACT_ROOT/$run_name"
  local config_path="$ARTIFACT_ROOT/configs/${run_name}.py"
  local provenance_json="$ARTIFACT_ROOT/${run_name}_provenance.json"
  local pred_json="$ARTIFACT_ROOT/${run_name}_predictions.json"
  local per_image_csv="$ARTIFACT_ROOT/${run_name}_per_image.csv"
  local summary_json="$ARTIFACT_ROOT/${run_name}_summary.json"

  local n_present=0
  for f in "$config_path" "$provenance_json" "$pred_json" "$per_image_csv" "$summary_json"; do
    [ -f "$f" ] && n_present=$((n_present + 1))
  done

  if is_recorded "$dataset" "$task" "$seed"; then
    if [ "$n_present" -eq 5 ]; then
      RUN_STATUS="complete"
    else
      RUN_STATUS="inconsistent"
      RUN_STATUS_DETAIL="TSV has a row for ${dataset}/${task}/seed=${seed} but only ${n_present}/5 expected output files are present on disk"
    fi
  else
    if [ "$n_present" -eq 0 ] && [ ! -d "$work_dir" ]; then
      RUN_STATUS="fresh"
    elif [ "$n_present" -eq 5 ]; then
      RUN_STATUS="recoverable"
    else
      RUN_STATUS="inconsistent"
      RUN_STATUS_DETAIL="no TSV row for ${dataset}/${task}/seed=${seed}, but ${n_present}/5 expected output files (or a work directory) already exist -- neither a clean fresh state nor a fully recoverable one"
    fi
  fi
}

run_one() {
  local dataset="$1" task="$2" anatomy="$3" seed="$4"
  local run_name="${dataset}_${task}_seed${seed}_run"
  local work_dir="$ARTIFACT_ROOT/$run_name"
  local config_path="$ARTIFACT_ROOT/configs/${run_name}.py"
  local provenance_json="$ARTIFACT_ROOT/${run_name}_provenance.json"
  local pred_json="$ARTIFACT_ROOT/${run_name}_predictions.json"
  local per_image_csv="$ARTIFACT_ROOT/${run_name}_per_image.csv"
  local summary_json="$ARTIFACT_ROOT/${run_name}_summary.json"

  run_artifacts_status "$dataset" "$task" "$seed"
  case "$RUN_STATUS" in
    complete)
      echo "[SKIP] ${dataset}/${task} seed=${seed} (already in $RESULTS_TSV, all output files present)"
      return
      ;;
    recoverable)
      # "严格恢复" (strict recovery): existence of all 5 files is not enough
      # on its own -- cross-check their CONTENT agrees with each other and
      # with the expected seed before trusting them into the TSV, so a
      # stale/mismatched file from an earlier aborted attempt cannot be
      # silently accepted as this seed's real result.
      local values
      values=$("$PY" - "$summary_json" "$per_image_csv" "$config_path" "$seed" <<'PY'
import csv, json, sys
from mmengine.config import Config
from mmengine.registry import init_default_scope

summary_path, per_image_path, config_path, seed = sys.argv[1:5]
seed = int(seed)

x = json.load(open(summary_path, encoding="utf-8"))
for key in ("n", "fixed_channel_mean_pct", "swap_min_mean_pct"):
    if key not in x:
        raise SystemExit(f"ERROR: {summary_path} is missing expected key {key!r}")

with open(per_image_path, newline="", encoding="utf-8") as f:
    per_image_rows = list(csv.DictReader(f))
if len(per_image_rows) != x["n"]:
    raise SystemExit(
        f"ERROR: {per_image_path} has {len(per_image_rows)} rows but "
        f"{summary_path} reports n={x['n']} -- inconsistent, refusing to recover"
    )

init_default_scope("mmpose")
cfg = Config.fromfile(config_path)
if cfg.randomness["seed"] != seed:
    raise SystemExit(
        f"ERROR: {config_path}'s own randomness.seed={cfg.randomness['seed']} "
        f"does not match the expected seed={seed} for this run -- refusing to recover"
    )

print(f'{x["n"]} {x["fixed_channel_mean_pct"]:.6f} {x["swap_min_mean_pct"]:.6f}')
PY
)
      local n fixed swap
      read -r n fixed swap <<< "$values"
      append_result_row "$dataset" "$task" "$seed" "$n" "$fixed" "$swap"
      echo "[RECOVERED] ${dataset}/${task} seed=${seed}: all 5 output files were already present but missing from $RESULTS_TSV -- content cross-checked (per-image row count matches n, config's own seed matches) and recovered WITHOUT retraining (n=$n fixed=${fixed}% swap_min=${swap}%)"
      return
      ;;
    inconsistent)
      echo "ERROR: inconsistent state for ${dataset}/${task} seed=${seed}: $RUN_STATUS_DETAIL" >&2
      echo "Refusing to auto-resolve -- manually inspect $work_dir and the *_${run_name}_* files, either" >&2
      echo "complete/restore whatever is missing or archive+remove them, then re-run." >&2
      exit 1
      ;;
    fresh) ;;
  esac

  check_disk

  local artifacts train_json val_json test_json
  artifacts="$(ensure_shared_artifacts "$dataset" "$task" "$anatomy")"
  IFS='|' read -r _ train_json val_json test_json <<< "$artifacts"

  echo ""
  echo "============================================================"
  echo "  START: ${run_name}"
  echo "============================================================"

  echo "=== [1/N] generate config ==="
  "$PY" make_config.py \
    --dataset "$dataset" --task "$task" --seed "$seed" \
    --data-root "$DATA_ROOT" \
    --images-dir "$DATA_ROOT/images/${dataset}/${anatomy}" \
    --internal-train-ann "$train_json" \
    --internal-val-ann "$val_json" \
    --test-ann "$test_json" \
    --pretrained-checkpoint-path "$PRETRAINED_CKPT_PATH" \
    --work-dir "$work_dir" \
    --max-epochs "$MAX_EPOCHS" \
    --out "$config_path"

  echo "--- Verifying config for ${run_name} (Fix B: exhaustive field assertions) ---"
  verify_config "$config_path" "$seed" "$train_json" "$val_json" "$test_json"

  echo "=== [1a/N] regression test: generated config avoids MMEngine's lazy-import mode ==="
  "$PY" test_make_config_real_load.py

  echo "=== [1b/N] record + VERIFY pretrained-weight provenance ==="
  "$PY" record_run_provenance.py \
    --config "$config_path" \
    --pretrained-checkpoint-path "$PRETRAINED_CKPT_PATH" \
    --out-json "$provenance_json"
  GOT_RUN_SHA256=$("$PY" -c "import json; print(json.load(open('$provenance_json'))['pretrained_checkpoint_local_sha256'])")
  if [ "$GOT_RUN_SHA256" != "$EXPECTED_PRETRAINED_SHA256" ]; then
    echo "ERROR: ${run_name}'s own provenance record does not match the expected pretrained-checkpoint SHA-256." >&2
    exit 1
  fi

  echo "=== [1c/N] MANDATORY live preflight gate ==="
  "$PY" live_preflight.py --config "$config_path" \
    || { echo "ERROR: live_preflight.py failed for ${run_name} -- refusing to start training." >&2; exit 1; }

  echo "=== [2/N] train ==="
  "$PY" "$MMPOSE_TRAIN_TOOL" "$config_path" --work-dir "$work_dir"

  echo "=== [2a/N] verify the true final checkpoint ==="
  local last_ckpt_pointer="$work_dir/last_checkpoint"
  [ -f "$last_ckpt_pointer" ] || { echo "ERROR: $last_ckpt_pointer not found for ${run_name}" >&2; exit 1; }
  local final_ckpt
  final_ckpt="$(cat "$last_ckpt_pointer")"
  [ -f "$final_ckpt" ] || final_ckpt="$work_dir/$(basename "$final_ckpt")"
  [ -f "$final_ckpt" ] || { echo "ERROR: checkpoint path in $last_ckpt_pointer does not exist: $final_ckpt" >&2; exit 1; }
  case "$(basename "$final_ckpt")" in
    epoch_${MAX_EPOCHS}.pth) ;;
    *) echo "ERROR: last_checkpoint points at $(basename "$final_ckpt"), not epoch_${MAX_EPOCHS}.pth for ${run_name}" >&2; exit 1 ;;
  esac
  echo "using verified final checkpoint: $final_ckpt"

  echo "=== [3/N] inference + score ==="
  "$PY" run_inference.py \
    --config "$config_path" \
    --checkpoint "$final_ckpt" \
    --gt-json "$test_json" \
    --out-predictions-json "$pred_json"

  "$PY" evaluate_rtmpose_fixed.py \
    --gt-json "$test_json" \
    --predictions-json "$pred_json" \
    --per-image-csv "$per_image_csv" \
    --summary-json "$summary_json"

  local values
  values=$("$PY" - "$summary_json" <<'PY'
import json, sys
x = json.load(open(sys.argv[1], encoding="utf-8"))
print(f'{x["n"]} {x["fixed_channel_mean_pct"]:.6f} {x["swap_min_mean_pct"]:.6f}')
PY
)
  local n fixed swap
  read -r n fixed swap <<< "$values"
  append_result_row "$dataset" "$task" "$seed" "$n" "$fixed" "$swap"
  echo "[DONE] ${run_name}: n=$n fixed=${fixed}% swap_min(=PI-NME)=${swap}%"
}

# --- Fix B: exhaustive config-field verification, every value read
#     directly from rtmpose_reproduction/make_config.py's real TEMPLATE. ---
verify_config() {
  local config_path="$1" seed="$2" train_json="$3" val_json="$4" test_json="$5"
  "$PY" - "$config_path" "$seed" "$MAX_EPOCHS" "$PRETRAINED_CKPT_PATH" \
        "$train_json" "$val_json" "$test_json" <<'PYEOF'
import sys
from mmengine.config import Config
from mmengine.registry import init_default_scope
init_default_scope("mmpose")

(cfg_path, seed, max_epochs, pretrained,
 train_json, val_json, test_json) = sys.argv[1:8]
seed, max_epochs = int(seed), int(max_epochs)
cfg = Config.fromfile(cfg_path)

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

checks = [
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
    ("backbone.init_cfg.checkpoint", backbone["init_cfg"]["checkpoint"], pretrained),
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
]

all_ok = True
for key, got, expected in checks:
    ok = got == expected
    print(f'  {"[OK]  " if ok else "[FAIL]"} {key}: {got!r}' + ("" if ok else f"  expected={expected!r}"))
    if not ok:
        all_ok = False
if not all_ok:
    sys.exit("ERROR: config verification failed -- aborting")
print(f"[OK] all {len(checks)} config checks passed")
PYEOF
}

aggregate_and_report() {
  echo ""
  echo "============================================================"
  echo "  Five-seed aggregation (permutation-invariant NME = swap_min column)"
  echo "============================================================"
  "$PY" - "$RESULTS_TSV" <<'PY'
import csv, statistics, sys

EXPECTED_SEEDS = {"42", "0", "123", "2024", "3407"}
rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8"), delimiter="\t"))

by_cell = {}
for r in rows:
    by_cell.setdefault((r["dataset"], r["task"]), []).append(r)

for (dataset, task), cell_rows in sorted(by_cell.items()):
    seeds_present = [r["seed"] for r in cell_rows]
    dupes = {s for s in seeds_present if seeds_present.count(s) > 1}
    if dupes:
        raise SystemExit(f"ERROR: duplicate TSV rows for {dataset}/{task}, seed(s) {sorted(dupes)}")
    seed_set = set(seeds_present)
    unexpected = seed_set - EXPECTED_SEEDS
    if unexpected:
        raise SystemExit(f"ERROR: {dataset}/{task} has unexpected seed(s) {sorted(unexpected)}, not in {sorted(EXPECTED_SEEDS)}")

    n_vals = {r["n"] for r in cell_rows}
    swap_vals = [float(r["swap_min_mean_pct"]) for r in cell_rows]
    status = "COMPLETE" if seed_set == EXPECTED_SEEDS else f"PARTIAL ({len(seed_set)}/5)"
    n_note = f"n={n_vals.pop()}" if len(n_vals) == 1 else f"n MISMATCH ACROSS SEEDS: {sorted(n_vals)}"
    mean = statistics.mean(swap_vals)
    sd = statistics.stdev(swap_vals) if len(swap_vals) > 1 else float("nan")
    print(f"  {dataset:12s} {task:5s} [{status:14s}] {n_note} seeds={sorted(seed_set, key=int)} "
          f"PI-NME={mean:.2f}%" + (f" +/- {sd:.2f}%" if len(swap_vals) > 1 else ""))
PY
}

process_cell() {
  local dataset="$1" task="$2" anatomy="$3"
  for SEED in "${SEEDS[@]}"; do
    run_one "$dataset" "$task" "$anatomy" "$SEED"
  done
  aggregate_and_report
  if [ "$STOP_AFTER_EACH_CELL" = "1" ]; then
    echo ""
    echo "=== ${dataset}/${task} five-seed result above -- STOPPING here (STOP_AFTER_EACH_CELL=1, default) ==="
    echo "Before continuing: back up this cell's checkpoints/logs/predictions to local storage and verify"
    echo "the backup, THEN free the server disk (see backup_and_clean_cell.sh) -- do NOT just lower"
    echo "MIN_FREE_GB or re-run with STOP_AFTER_EACH_CELL=0 to push through; this server's disk genuinely"
    echo "cannot hold many cells' worth of artifacts at once (see this script's own Fix D comment)."
    echo "Re-run this same script (STOP_AFTER_EACH_CELL=1) to process the next remaining cell; already-"
    echo "complete cells are skipped via $RESULTS_TSV, so no progress is lost."
    exit 0
  fi
}

for CELL in "${CELLS[@]}"; do
  IFS=':' read -r DATASET TASK ANATOMY <<< "$CELL"
  process_cell "$DATASET" "$TASK" "$ANATOMY"
done

echo ""
echo "[COMPLETE] all 50 RTMPose-s runs are recorded in $RESULTS_TSV"
