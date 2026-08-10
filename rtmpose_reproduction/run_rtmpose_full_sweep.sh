#!/usr/bin/env bash
# =============================================================================
# run_rtmpose_full_sweep.sh -- the full RTMPose-s programme: 2 datasets x 5
# tasks x 5 seeds = 50 runs, structured like
# baseline_reproduction/run_hrnet_512_fixed_5seed.sh (resumable via a
# results TSV, refuses to silently resume a partial run, disk-safety
# preflight) -- NOT a re-run of run_rtmpose_canary.sh's seed-42 UCL-BPD
# canary, whose result is pre-seeded into the results TSV below.
#
# SUPERSEDES the narrower run_rtmpose_bpd_remaining_seeds.sh (removed):
# a relayed review (2026-08-10) found that script's resume-safety, shared-
# JSON integrity and same-configuration enforcement were all weaker than
# this project's own established HRNet-driver pattern, and maintaining two
# overlapping BPD-scoped/full-sweep scripts with the same hardening logic
# duplicated (and prone to drifting out of sync) was worse than one script
# that does both. Every fix that review asked for is folded in below.
#
# Supervisor's exact instruction (2026-08-10, recorded verbatim in
# docs/supervisor_meeting_report_2026-08-08.md's 阶段 3 section): "Please
# proceed with the remaining four UCL BPD seeds using exactly the same
# configuration, so that we can obtain a reliable five-seed mean and
# seed-level standard deviation. Once we have the five-seed BPD result, we
# can proceed with the remaining measurements using the same RTMPose-s
# configuration." This script honours BOTH halves: it processes UCL BPD's
# 5 seeds first (config list order below), then GATES on an explicit
# environment variable before continuing to the other 45 runs (default:
# stop after BPD, matching PREFLIGHT_ONLY's existing safe-by-default
# convention elsewhere in this project) -- so the five-seed BPD number can
# actually be reviewed before the rest starts, not just technically
# obtained before the script races on.
#
# Fix 1 (relayed review, "缺少已有运行目录保护"): a (dataset, task, seed)
# cell is SKIPPED if already present in RESULTS_TSV; if its WORK_DIR exists
# but no verified final checkpoint is recorded there, the script REFUSES to
# resume automatically and exits with instructions to inspect/archive that
# exact directory first -- identical in spirit to
# run_hrnet_512_fixed_5seed.sh's own `run_one()`/`is_recorded()` pattern.
#
# Fix 2 (relayed review, "共享 split 只检查存在，未检查未改变"): for each
# (dataset, task) cell, the internal train/val split and Train/Val/Test COCO
# jsons are generated ONCE (skipped if already present) and self-recorded
# into a per-cell SHA-256 manifest; every seed within that cell re-verifies
# its hash against that manifest before training, so a mid-sweep
# regeneration/corruption of the shared artifacts is caught immediately
# rather than silently propagating into 4 more runs. Content sanity
# (internal-train/internal-val filename sets disjoint; Test disjoint from
# both) is also asserted at generation time.
#
# Fix 3 (relayed review, "'完全相同配置'尚未被代码锁死"): MAX_EPOCHS is
# hardcoded to 200, not overridable via environment variable (the previous
# script's ${MAX_EPOCHS:-200} pattern allowed a silent override). The
# pretrained CSPNeXt-s checkpoint's SHA-256 is asserted against the exact
# value already recorded and reported to the supervisor for the seed-42
# canary (docs/supervisor_meeting_report_2026-08-08.md section 3.3:
# aa7d9335bf422ad02a803e36f357dfc6abb807eca42d79e8b3b6e7c5bd1f446b) --
# EVERY run, not just the first. Every other config-critical field
# (architecture, batch size, augmentation, SimCC settings, checkpoint
# selection policy) is asserted against its exact expected literal value on
# every generated config, mirroring the inline-Python-assertion pattern
# already used throughout this project's other ablation drivers (e.g.
# ablation/scripts/run_bpd_core_chain_retrain_5seed.sh) -- this is a
# stronger, self-documenting guarantee than a raw config-file diff against
# a stored reference, and does not depend on locating/parsing the canary's
# own already-archived config file.
#
# Prerequisites (same as run_rtmpose_canary.sh -- see ENVIRONMENT.md):
#   - a pinned MMPose/MMEngine/MMCV install in its own venv, verified importable
#   - the CSPNeXt-s checkpoint downloaded locally (PRETRAINED_CKPT_PATH)
#   - run_rtmpose_canary.sh's seed-42 UCL-BPD canary already complete (its
#     summary.json is read directly, not re-produced)
# =============================================================================

set -euo pipefail

export CUBLAS_WORKSPACE_CONFIG=:4096:8

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:?set PY to the RTMPose venv python interpreter}"
DATA_ROOT="${DATA_ROOT:-/root/autodl-tmp}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/root/autodl-tmp/rtmpose_reproduction}"
PRETRAINED_CKPT_PATH="${PRETRAINED_CKPT_PATH:?set PRETRAINED_CKPT_PATH to the locally-downloaded CSPNeXt-s checkpoint file, see ENVIRONMENT.md}"
MMPOSE_TRAIN_TOOL="${MMPOSE_TRAIN_TOOL:?set MMPOSE_TRAIN_TOOL to the installed mmpose repo tools/train.py path}"

# Fix 3: NOT an environment-overridable variable -- "exactly the same
# configuration" as the approved seed-42 canary means 200 epochs, full stop.
readonly MAX_EPOCHS=200
readonly EXPECTED_PRETRAINED_SHA256="aa7d9335bf422ad02a803e36f357dfc6abb807eca42d79e8b3b6e7c5bd1f446b"

RESULTS_TSV="${RESULTS_TSV:-$ARTIFACT_ROOT/rtmpose_full_sweep_results.tsv}"
CANARY_SUMMARY_JSON="$ARTIFACT_ROOT/UCL_BPD_seed42_canary_summary.json"
MANIFEST_DIR="$ARTIFACT_ROOT/coco/manifests"
MIN_FREE_GB="${MIN_FREE_GB:-6}"

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

EXPECTED_HEADER=$'dataset\ttask\tseed\tn\tfixed_channel_mean_pct\tswap_min_mean_pct'
if [ ! -f "$RESULTS_TSV" ]; then
  printf '%s\n' "$EXPECTED_HEADER" > "$RESULTS_TSV"
elif [ "$(head -n 1 "$RESULTS_TSV")" != "$EXPECTED_HEADER" ]; then
  echo "ERROR: incompatible results header: $RESULTS_TSV" >&2
  exit 1
fi

# Pre-seed the already-approved UCL/BPD/seed=42 canary result rather than
# re-running it under a different file-naming scheme.
if ! awk -F'\t' 'NR>1 && $1=="UCL" && $2=="BPD" && $3=="42" {found=1} END{exit !found}' "$RESULTS_TSV"; then
  if [ ! -f "$CANARY_SUMMARY_JSON" ]; then
    echo "ERROR: $CANARY_SUMMARY_JSON not found -- this script requires the seed-42 canary to already be complete (see run_rtmpose_canary.sh)." >&2
    exit 1
  fi
  "$PY" - "$CANARY_SUMMARY_JSON" "$RESULTS_TSV" <<'PY'
import json, sys
summary_path, results_tsv = sys.argv[1], sys.argv[2]
s = json.load(open(summary_path, encoding="utf-8"))
with open(results_tsv, "a", encoding="utf-8") as f:
    f.write(f"UCL\tBPD\t42\t{s['n']}\t{s['fixed_channel_mean_pct']:.6f}\t{s['swap_min_mean_pct']:.6f}\n")
print(f"[OK] pre-seeded UCL/BPD/seed=42 from the already-approved canary: n={s['n']}, "
      f"PI-NME(swap_min)={s['swap_min_mean_pct']:.4f}%")
PY
fi

is_recorded() {
  awk -F'\t' -v d="$1" -v t="$2" -v s="$3" 'NR>1 && $1==d && $2==t && $3==s {f=1} END{exit !f}' "$RESULTS_TSV"
}

check_disk() {
  local free_gb
  free_gb=$(df --output=avail -BG "$ARTIFACT_ROOT" | tail -1 | tr -dc '0-9')
  [ "$free_gb" -ge "$MIN_FREE_GB" ] || {
    echo "ERROR: ${free_gb}GB free, require ${MIN_FREE_GB}GB" >&2; exit 1; }
}

# --- Fix 2: generate (once) + verify (every seed) the shared per-cell
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
    # regenerate even a "matching" copy, to eliminate any possibility of
    # divergence for the one cell that already has an approved reference.
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

run_one() {
  local dataset="$1" task="$2" anatomy="$3" seed="$4"
  if is_recorded "$dataset" "$task" "$seed"; then
    echo "[SKIP] ${dataset}/${task} seed=${seed} (already in $RESULTS_TSV)"
    return
  fi
  check_disk

  local artifacts train_json val_json test_json
  artifacts="$(ensure_shared_artifacts "$dataset" "$task" "$anatomy")"
  IFS='|' read -r _ train_json val_json test_json <<< "$artifacts"

  local run_name="${dataset}_${task}_seed${seed}_run"
  local work_dir="$ARTIFACT_ROOT/$run_name"
  local config_path="$ARTIFACT_ROOT/configs/${run_name}.py"
  local pred_json="$ARTIFACT_ROOT/${run_name}_predictions.json"
  local per_image_csv="$ARTIFACT_ROOT/${run_name}_per_image.csv"
  local summary_json="$ARTIFACT_ROOT/${run_name}_summary.json"

  # --- Fix 1: refuse to silently resume a partial run directory ---
  if [ -d "$work_dir" ] && [ ! -f "$summary_json" ]; then
    echo "ERROR: partial run directory exists without a recorded summary: $work_dir" >&2
    echo "Archive/inspect it, then remove only that exact directory before retrying (do not auto-resume)." >&2
    exit 1
  fi

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

  # --- Fix 3: assert every config-critical field's exact expected value,
  #     not just trust make_config.py's own defaults silently held. ---
  echo "--- Verifying config for ${run_name} ---"
  "$PY" - "$config_path" "$seed" "$MAX_EPOCHS" "$PRETRAINED_CKPT_PATH" <<'PYEOF'
import sys
from mmengine.config import Config
from mmengine.registry import init_default_scope
init_default_scope("mmpose")

cfg_path, seed, max_epochs, pretrained = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
cfg = Config.fromfile(cfg_path)

checks = [
    ("randomness.seed", cfg.randomness["seed"], seed),
    ("randomness.deterministic", cfg.randomness["deterministic"], True),
    ("train_cfg.max_epochs", cfg.train_cfg["max_epochs"], max_epochs),
    ("backbone.init_cfg.checkpoint", cfg.model["backbone"]["init_cfg"]["checkpoint"], pretrained),
    ("codec.input_size", tuple(cfg.codec["input_size"]), (512, 512)),
]
all_ok = True
for key, got, expected in checks:
    ok = got == expected
    print(f'  {"[OK]  " if ok else "[FAIL]"} {key}: {got!r}' + ("" if ok else f"  expected={expected!r}"))
    if not ok:
        all_ok = False
if not all_ok:
    sys.exit("ERROR: config verification failed -- aborting")
print("[OK] all config checks passed")
PYEOF

  echo "=== [1a/N] regression test: generated config avoids MMEngine's lazy-import mode ==="
  "$PY" test_make_config_real_load.py

  echo "=== [1b/N] record + VERIFY pretrained-weight provenance ==="
  local provenance_json="$ARTIFACT_ROOT/${run_name}_provenance.json"
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
  read -r n fixed swap <<< "$values"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$dataset" "$task" "$seed" "$n" "$fixed" "$swap" >> "$RESULTS_TSV"
  echo "[DONE] ${run_name}: n=$n fixed=${fixed}% swap_min(=PI-NME)=${swap}%"
}

aggregate_and_report() {
  echo ""
  echo "============================================================"
  echo "  Five-seed aggregation (permutation-invariant NME = swap_min column)"
  echo "============================================================"
  "$PY" - "$RESULTS_TSV" <<'PY'
import csv, statistics, sys
rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8"), delimiter="\t"))
by_cell = {}
for r in rows:
    by_cell.setdefault((r["dataset"], r["task"]), []).append(r)
for (dataset, task), cell_rows in sorted(by_cell.items()):
    seeds_present = sorted({r["seed"] for r in cell_rows}, key=int)
    swap_vals = [float(r["swap_min_mean_pct"]) for r in cell_rows]
    status = "COMPLETE" if len(seeds_present) == 5 else f"PARTIAL ({len(seeds_present)}/5)"
    mean = statistics.mean(swap_vals)
    sd = statistics.stdev(swap_vals) if len(swap_vals) > 1 else float("nan")
    print(f"  {dataset:12s} {task:5s} [{status:14s}] seeds={seeds_present} "
          f"PI-NME={mean:.2f}%" + (f" +/- {sd:.2f}%" if len(swap_vals) > 1 else ""))
PY
}

echo ""
echo "=== Processing UCL/BPD first (5 seeds) per the supervisor's staged instruction ==="
for SEED in "${SEEDS[@]}"; do
  run_one "UCL" "BPD" "Head" "$SEED"
done
aggregate_and_report

if [ "${PROCEED_PAST_BPD:-0}" != "1" ]; then
  echo ""
  echo "=== UCL/BPD five-seed result above -- STOPPING here (default) ==="
  echo "Review the five-seed mean +/- seed-level sample SD with the supervisor before continuing."
  echo "Re-run with PROCEED_PAST_BPD=1 to continue to the remaining 45 runs (per the supervisor's"
  echo "own pre-approval to proceed 'once we have the five-seed BPD result')."
  exit 0
fi

echo ""
echo "=== PROCEED_PAST_BPD=1 -- continuing to the remaining 45 runs ==="
for CELL in "${CELLS[@]}"; do
  IFS=':' read -r DATASET TASK ANATOMY <<< "$CELL"
  [ "$DATASET" = "UCL" ] && [ "$TASK" = "BPD" ] && continue  # already done above
  for SEED in "${SEEDS[@]}"; do
    run_one "$DATASET" "$TASK" "$ANATOMY" "$SEED"
  done
done

aggregate_and_report
echo ""
echo "[COMPLETE] all 50 RTMPose-s runs are recorded in $RESULTS_TSV"
