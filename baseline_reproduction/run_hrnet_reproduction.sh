#!/usr/bin/env bash
# =============================================================================
# run_hrnet_reproduction.sh -- reproduces the supervisors' HRNet/BiometryNet
# pipeline (surgical-vision/Multicentre-Fetal-Biometry) on UCL and Multicentre
# data, using their tools/train.py + tools/test.py completely unmodified.
# 10 configs (UCL x {BPD,OFD,APAD,TAD,FL}, MULTICENTRE x {BPD,OFD,APAD,TAD,FL})
# x 5 repeats each = 50 runs, matching this project's 5-seed convention. See
# baseline_reproduction/README.md for the full rationale, what was/wasn't
# changed, and why.
#
# NO CODE IN THE HRNET REPO IS EDITED BY THIS SCRIPT. The only per-run
# variation is which config file is passed to --cfg: each repeat uses a copy
# of the canonical config saved under a distinct filename (fetal_landmark_..
# ._repN.yaml), so the repo's own cfg-name-derived output-dir logic
# (lib/utils/utils.py:create_logger, output/FETAL/<cfg_name>/) naturally
# gives each repeat its own directory instead of overwriting the last run.
# The original repo has no seed/manual_seed mechanism anywhere (checked) --
# 5 unmodified repeats already give 5 independently-initialised/shuffled
# runs, which is all the 5-seed convention actually needs.
#
# Prerequisites:
#   - HRNET_REPO_ROOT points at a clean checkout of the base repo (see
#     README.md for the exact commit + the one forward-compat patch), with
#     the modernized environment from ENVIRONMENT.md installed (the pinned
#     environment.yml's torch==1.0.0 cannot run on the AutoDL RTX 4090).
#   - UCL and MULTICENTRE images/annotations already staged at
#     /root/autodl-tmp/images/{UCL,MULTICENTRE}/{Head,Abdomen,Femur}/ and
#     /root/autodl-tmp/annotations/{UCL,MULTICENTRE}/{Head,Abdomen,Femur}_
#     {Train,Test}.csv -- same data this project's EoMT Multicentre pipeline
#     already uses, no new transfer needed. Preflight below verifies this.
#
# Usage:
#   export HRNET_REPO_ROOT=/root/Multicentre-Fetal-Biometry
#   CANARY_ONLY=1 bash run_hrnet_reproduction.sh   # 1 config x 1 rep sanity check
#   bash run_hrnet_reproduction.sh                  # full 10 x 5 sweep
#   # safe to re-invoke: completed (config, rep) rows in the results TSV are
#   # never re-run.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIGS_DIR="${CONFIGS_DIR:-$SCRIPT_DIR/configs}"
HRNET_REPO_ROOT="${HRNET_REPO_ROOT:?set HRNET_REPO_ROOT to the HRNet repo checkout path}"
PY="${PY:-python3}"
REPS="${REPS:-5}"
MIN_FREE_GB="${MIN_FREE_GB:-3}"

RESULTS_TSV="$SCRIPT_DIR/hrnet_reproduction_results.tsv"
LOG_DIR="$SCRIPT_DIR/logs"
REP_CFG_DIR="$HRNET_REPO_ROOT/experiments/fetal_reproduction_reps"
mkdir -p "$LOG_DIR" "$REP_CFG_DIR"

if [ ! -f "$RESULTS_TSV" ]; then
    printf "config\trep\tnme_pct\tnme_mean_pct\tnme_std_pct\n" > "$RESULTS_TSV"
fi

# All 10 configs -- NEVER reduced by CANARY_ONLY (mirrors ALL_SEEDS in
# run_multicentre_5seed_ablation.sh: canary mode must restrict CONFIGS below,
# not this list, so a canary run can never be mistaken for a complete sweep).
ALL_CONFIGS=(
    "fetal_landmark_hrnet_w18_UCL_brain_BPD"
    "fetal_landmark_hrnet_w18_UCL_brain_OFD"
    "fetal_landmark_hrnet_w18_UCL_abdomen_APAD"
    "fetal_landmark_hrnet_w18_UCL_abdomen_TAD"
    "fetal_landmark_hrnet_w18_UCL_femur_FL"
    "fetal_landmark_hrnet_w18_MULTICENTRE_brain_BPD"
    "fetal_landmark_hrnet_w18_MULTICENTRE_brain_OFD"
    "fetal_landmark_hrnet_w18_MULTICENTRE_abdomen_APAD"
    "fetal_landmark_hrnet_w18_MULTICENTRE_abdomen_TAD"
    "fetal_landmark_hrnet_w18_MULTICENTRE_femur_FL"
)
CONFIGS=("${ALL_CONFIGS[@]}")

if [ "${CANARY_ONLY:-0}" = "1" ]; then
    CONFIGS=("fetal_landmark_hrnet_w18_UCL_brain_BPD")
    REPS=1
    echo "[CANARY_ONLY] restricting to ${CONFIGS[*]}, REPS=$REPS"
fi

check_disk() {
    local free_gb
    free_gb=$(df --output=avail -BG "$HRNET_REPO_ROOT" | tail -1 | tr -dc '0-9')
    if [ "$free_gb" -lt "$MIN_FREE_GB" ]; then
        echo "ERROR: only ${free_gb}GB free at $HRNET_REPO_ROOT (need >= ${MIN_FREE_GB}GB). Aborting." >&2
        exit 1
    fi
}

# Verify a config's ROOT/TRAINSET/TESTSET paths actually exist on this
# machine before spending any GPU time on it -- do NOT let train.py fail
# 50 minutes into a run because a path typo or missing transfer went
# unnoticed (same rationale as this project's other preflight checks).
preflight_config() {
    local cfg_file="$1"
    local root trainset testset
    root=$($PY -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['DATASET']['ROOT'])" "$cfg_file")
    trainset=$($PY -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['DATASET']['TRAINSET'])" "$cfg_file")
    testset=$($PY -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['DATASET']['TESTSET'])" "$cfg_file")
    for p in "$root" "$trainset" "$testset"; do
        if [ ! -e "$p" ]; then
            echo "ERROR: preflight failed for $cfg_file -- missing path: $p" >&2
            exit 1
        fi
    done
}

# Confirm the canonical config in CONFIGS_DIR still matches what was
# reviewed (only DATASET.ROOT/TRAINSET/TESTSET may legitimately differ from
# the original repo's experiments/fetal/<name>.yaml) -- guards against this
# folder's configs silently drifting from "their code/config as-is" after
# a manual edit.
verify_config_untouched() {
    local cfg_name="$1"
    local ours="$CONFIGS_DIR/${cfg_name}.yaml"
    local theirs="$HRNET_REPO_ROOT/experiments/fetal/${cfg_name}.yaml"
    if [ ! -f "$theirs" ]; then
        echo "WARNING: original config not found at $theirs -- cannot verify no-drift for $cfg_name" >&2
        return
    fi
    local diff_lines
    diff_lines=$(diff "$theirs" "$ours" | grep -c '^[<>]' || true)
    # 6 changed lines expected: 3 removed (original path lines) + 3 added
    # (server path lines). MULTICENTRE_abdomen_TAD's ROOT line differs only
    # in trailing whitespace after the closing quote, which yaml.safe_load
    # ignores -- still counts as one changed line pair here.
    if [ "$diff_lines" -ne 6 ]; then
        echo "ERROR: $cfg_name has drifted from the official config by more than the expected 3 path lines (diff shows $diff_lines changed lines). Refusing to run -- inspect manually." >&2
        exit 1
    fi
}

nme_seen_for() {
    local cfg="$1" rep="$2"
    awk -F'\t' -v c="$cfg" -v r="$rep" 'NR>1 && $1==c && $2==r {found=1} END{exit !found}' "$RESULTS_TSV"
}

run_one() {
    local cfg="$1" rep="$2"
    if nme_seen_for "$cfg" "$rep"; then
        echo "[skip] $cfg rep$rep already in $RESULTS_TSV"
        return
    fi

    check_disk
    local base_cfg="$CONFIGS_DIR/${cfg}.yaml"
    verify_config_untouched "$cfg"
    preflight_config "$base_cfg"

    local rep_name="${cfg}_rep${rep}"
    local rep_cfg="$REP_CFG_DIR/${rep_name}.yaml"
    cp "$base_cfg" "$rep_cfg"

    echo "=== training $rep_name ==="
    ( cd "$HRNET_REPO_ROOT" && "$PY" tools/train.py --cfg "$rep_cfg" ) \
        > "$LOG_DIR/${rep_name}_train.log" 2>&1

    local model_dir="$HRNET_REPO_ROOT/output/FETAL/${rep_name}"
    local model_file="$model_dir/final_state.pth"
    if [ ! -f "$model_file" ]; then
        echo "ERROR: $rep_name training did not produce $model_file -- check $LOG_DIR/${rep_name}_train.log" >&2
        exit 1
    fi

    echo "=== testing $rep_name ==="
    ( cd "$HRNET_REPO_ROOT" && "$PY" tools/test.py --cfg "$rep_cfg" --model-file "$model_file" ) \
        > "$LOG_DIR/${rep_name}_test.log" 2>&1

    # Parses the exact line lib/core/function.py:inference() logs:
    #   "Test Results time:.. loss:.. nme:X nme mean:X nme std:X [008]:.. [010]:.."
    # nme/nme_mean/nme_std are fractions (compute_nme's own convention) --
    # multiplied by 100 here only for this TSV, to match this project's
    # existing percent-NME reporting convention. No re-computation of the
    # metric itself.
    local line nme nme_mean nme_std
    line=$(grep -o 'nme:[0-9.]* nme mean:[0-9.]* nme std:[0-9.]*' "$LOG_DIR/${rep_name}_test.log" | tail -1)
    if [ -z "$line" ]; then
        echo "ERROR: could not find NME result line in $LOG_DIR/${rep_name}_test.log" >&2
        exit 1
    fi
    nme=$(echo "$line"      | grep -o 'nme:[0-9.]*'      | cut -d: -f2)
    nme_mean=$(echo "$line" | grep -o 'nme mean:[0-9.]*' | cut -d: -f2)
    nme_std=$(echo "$line"  | grep -o 'nme std:[0-9.]*'  | cut -d: -f2)

    read -r pct mean_pct std_pct <<< "$(awk -v a="$nme" -v b="$nme_mean" -v c="$nme_std" \
        'BEGIN{printf "%.4f %.4f %.4f", a*100, b*100, c*100}')"
    printf "%s\t%s\t%s\t%s\t%s\n" "$cfg" "$rep" "$pct" "$mean_pct" "$std_pct" >> "$RESULTS_TSV"

    echo "[done] $rep_name -> nme=$(echo "$line")"
}

for cfg in "${CONFIGS[@]}"; do
    for rep in $(seq 1 "$REPS"); do
        run_one "$cfg" "$rep"
    done
    echo "--- $cfg: all $REPS reps complete, rows in $RESULTS_TSV ---"
    grep "^${cfg}	" "$RESULTS_TSV" || true
done

echo "=== hrnet_reproduction sweep finished (or all remaining rows were already done) ==="
