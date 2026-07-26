#!/usr/bin/env bash
# =============================================================================
# run_multicentre_5seed_ablation.sh — Multicentre (pooled FP+HC18+UCL),
# DINOv2 + DINOv3, all 5 fetal tasks (bpd/ofd/apad/tad/fl), 5 seeds each
# = 50 training runs total. DeconvHeadV2+FPN+UDP+rotate+scale matched-
# protocol recipe -- evaluates the already-fixed pipeline, no component
# re-ablation on Multicentre (see project memory / 04_experimental_setup
# .tex's sec:setup-evidence-scope).
#
# SAFE-STOP DESIGN (2026-07-24, per review): this script processes ONE
# (task, backbone) group fully -- all 5 seeds, then that group's ensemble
# -- and then EXITS, rather than continuing on to the next group
# unattended. Re-invoke the script to process the next incomplete group
# (it resumes correctly; already-complete groups are skipped). This is a
# deliberate choice given: (a) 10 groups x ~3GB each would be ~30GB,
# uncomfortably close to this project's repeated disk-full incidents
# (Rule 16/29 in project pitfalls memory) against a data disk that has
# shown as little as ~14GB free, and (b) under thesis-deadline time
# pressure, a human checking each group's ensemble numbers and disk
# headroom before the next group starts is safer than any automatic
# archive/delete policy touching formal-evidence checkpoints. Checkpoints
# write directly to the data disk (no post-hoc mv step) under BACKUP_ROOT
# below; last.ckpt is renamed (not copied) into *_final.ckpt.
#
# Records BOTH metrics per checkpoint AND per group-ensemble (swap-min
# primary -- also used for validation/early-stopping/checkpoint-selection
# during training; fixed-channel secondary, re-tested post-hoc on the
# SAME per-run config via a CLI override, per Rule 32 -- never a
# different/shared config). Per the locked fetal-NME-metric decision,
# this connects Multicentre to the historical fixed-channel EoMT series
# without retroactively reprocessing it.
#
# Uses MulticentreLandmarkDataModule (datasets/multicentre_dataset.py),
# NOT HeadLandmarkDataModule -- source-aware subject grouping is required
# once FP/HC18/UCL are pooled (see that module's docstring; verified
# against real data, 2026-07-24, via `python3 scripts/test_dataload.py
# multicentre` -- a real HC18/UCL numeric-prefix collision exists in the
# pooled Head data and the naive bare-prefix method would have merged
# it; that self-test also hard-fails if any image referenced by a CSV row
# is missing on disk, rather than silently training on a shrunken
# dataset -- run it before this script, on every fresh data transfer).
#
# Prerequisite: Multicentre data must be on the server first --
#   /root/autodl-tmp/images/MULTICENTRE/{Head,Abdomen,Femur}/
#   /root/autodl-tmp/annotations/MULTICENTRE/{Head,Abdomen,Femur}_{Train,Test}.csv
#   /root/autodl-tmp/annotations/HC18/Head_{Train,Test}.csv  (Head only)
# Run `python3 scripts/test_dataload.py multicentre` first to confirm the
# data is in place and the subject-grouping fix behaves as expected on
# this server's copy, before launching real training.
#
# Usage (on AutoDL server):
#   cd /root/eomt
#   nohup bash ablation/scripts/run_multicentre_5seed_ablation.sh > multicentre_5seed_ablation.log 2>&1 &
#   # (re-invoke after each group finishes to continue to the next one)
# =============================================================================

set -euo pipefail

BASE_CONFIG="configs/landmark/multicentre_fpn_udp_rotate_scale.yaml"

# Canonical 5-seed set -- NEVER overridden by CANARY_ONLY. Group
# completeness (whether to run the group's ensemble) is always checked
# against THIS list, not against whatever SEEDS below was reduced to for
# a given invocation -- see the CANARY_ONLY bug this fixes, noted below.
ALL_SEEDS=(42 0 123 2024 3407)
SEEDS=("${ALL_SEEDS[@]}")

DATA_ROOT="/root/autodl-tmp"
IMAGES_ROOT="${DATA_ROOT}/images/MULTICENTRE"
ANN_ROOT="${DATA_ROOT}/annotations/MULTICENTRE"
HC18_ROOT="${DATA_ROOT}/annotations/HC18"

# anatomy_dir:task:has_hc18(0/1)
TASKS=(
    "Head:bpd:1"
    "Head:ofd:1"
    "Abdomen:apad:0"
    "Abdomen:tad:0"
    "Femur:fl:0"
)
# backbone_tag:backbone_name
BACKBONES=(
    "dinov2:vit_small_patch14_reg4_dinov2"
    "dinov3:facebook/dinov3-vits16-pretrain-lvd1689m"
)

# CANARY MODE -- strongly recommended before the full 50-run sweep, since
# this is the first time MulticentreLandmarkDataModule (new file, new
# source-aware subject-grouping logic) is exercised through the full
# LightningCLI fit/test path rather than just scripts/test_dataload.py's
# direct construction:
#   CANARY_ONLY=1 bash ablation/scripts/run_multicentre_5seed_ablation.sh
#     -> runs ONLY task=bpd, backbone=dinov2, seed=42 (1 of 50 runs).
#     IMPORTANT (fixed 2026-07-24, was broken before): this does NOT mark
#     the bpd_dinov2 group complete and does NOT run its ensemble -- group
#     completeness always checks against ALL_SEEDS (all 5), never against
#     this reduced SEEDS list, so a later full-sweep invocation correctly
#     still trains the remaining 4 seeds for bpd_dinov2 rather than
#     silently skipping the whole group.
#   bash ablation/scripts/run_multicentre_5seed_ablation.sh
#     -> full sweep (resumes past anything CANARY_ONLY already completed,
#        processes exactly one group to completion, then exits).
if [ "${CANARY_ONLY:-0}" = "1" ]; then
    TASKS=("Head:bpd:1")
    BACKBONES=("dinov2:vit_small_patch14_reg4_dinov2")
    SEEDS=(42)
    echo "[CANARY_ONLY] restricting to task=bpd backbone=dinov2 seed=42 only"
    echo "[CANARY_ONLY] this will NOT mark bpd_dinov2 complete or ensemble it -- needs all of: ${ALL_SEEDS[*]}"
fi

RESULTS_TSV="multicentre_5seed_results.tsv"
BACKUP_ROOT="/root/autodl-tmp/saved_checkpoints/multicentre_5seed"

if [ ! -d "${IMAGES_ROOT}/Head" ] || [ ! -f "${ANN_ROOT}/Head_Train.csv" ]; then
    echo "[ERROR] Multicentre data not found under ${IMAGES_ROOT} / ${ANN_ROOT} -- transfer it first"
    exit 1
fi

mkdir -p checkpoints "$BACKUP_ROOT"
[ -f "$RESULTS_TSV" ] || echo -e "task\tbackbone\tseed\tckpt_tag\tswap_min_nme\tfixed_channel_nme" > "$RESULTS_TSV"

if [ ! -f "${BASE_CONFIG}" ]; then
    echo "[ERROR] Base config not found: ${BASE_CONFIG}"
    exit 1
fi
python3 -c "import yaml; yaml.safe_load(open('${BASE_CONFIG}'))" \
    && echo "[OK] Base config valid: ${BASE_CONFIG}" \
    || { echo "[ERROR] Invalid YAML -- aborting"; exit 1; }

# Disk preflight -- checked before every group (not just once), since
# groups run for a long time and disk usage elsewhere on the server can
# change between them.
check_disk() {
    local avail_kb
    avail_kb=$(df -Pk "$BACKUP_ROOT" | awk 'NR==2 {print $4}')
    local min_kb=$((6 * 1024 * 1024))
    if [ "$avail_kb" -lt "$min_kb" ]; then
        echo "[ERROR] Less than 6 GiB free on checkpoint disk: $BACKUP_ROOT"
        df -h "$BACKUP_ROOT"
        exit 1
    fi
}

# --- (task, backbone) group-completeness helpers, computed fresh from
#     the actual TSV/log files every time -- never trusted from a cached
#     "done" marker (a marker can go stale, e.g. the CANARY_ONLY bug this
#     rewrite fixes: writing a marker after only 1/5 seeds ran). ---

seed_pair_done() {
    # true if a given (task, backbone, seed) has both best+final TSV rows
    local task="$1" backbone="$2" seed="$3"
    local rows
    rows=$(awk -F'\t' -v t="$task" -v b="$backbone" -v s="$seed" \
        'NR>1 && $1==t && $2==b && $3==s && $4=="best" {x=1} \
         NR>1 && $1==t && $2==b && $3==s && $4=="final" {y=1} \
         END {print x+y}' \
        "$RESULTS_TSV")
    [ "$rows" -eq 2 ]
}

seeds_all_done() {
    # true if ALL_SEEDS (always the canonical 5, regardless of CANARY_ONLY)
    # each have both best+final TSV rows for this (task, backbone)
    local task="$1" backbone="$2" seed
    for seed in "${ALL_SEEDS[@]}"; do
        seed_pair_done "$task" "$backbone" "$seed" || return 1
    done
    return 0
}

group_is_complete() {
    # True only if all seed results exist AND every ensemble log contains
    # the terminal success line for exactly five models.  A failed command
    # piped through `tee` can leave a non-empty traceback log, so file size
    # alone is not evidence that an ensemble completed successfully.
    local task="$1" backbone="$2" group_dir="$3" tag metric
    seeds_all_done "$task" "$backbone" || return 1
    for tag in best final; do
        for metric in swapmin fixedchannel; do
            local log="${group_dir}/ensemble_${tag}_${metric}_log.txt"
            [ -s "$log" ] || return 1
            grep -qF "[RESULT] Ensemble (5 models) [channel-aligned+DOD-final]" "$log" || return 1
        done
    done
    return 0
}

check_disk
echo "[OK] checkpoint disk preflight"
df -h "$BACKUP_ROOT"

# Set if any group is left partial (e.g. under CANARY_ONLY) so the final
# summary doesn't wrongly claim every group is complete.
ANY_GROUP_PARTIAL=0

for TASK_SPEC in "${TASKS[@]}"; do
IFS=":" read -r ANATOMY_DIR TASK HAS_HC18 <<< "$TASK_SPEC"

for BACKBONE_SPEC in "${BACKBONES[@]}"; do
IFS=":" read -r BACKBONE_TAG BACKBONE_NAME <<< "$BACKBONE_SPEC"

    GROUP_KEY="${TASK}_${BACKBONE_TAG}"
    RUN_GROUP="multicentre-${TASK}-${BACKBONE_TAG}"
    GROUP_DIR="${BACKUP_ROOT}/${RUN_GROUP}"

    if group_is_complete "$TASK" "$BACKBONE_TAG" "$GROUP_DIR"; then
        echo ""
        echo "############################################################"
        echo "  SKIP whole group (verified complete + ensembled): ${GROUP_KEY}"
        echo "############################################################"
        continue
    fi

    check_disk
    echo ""
    echo "############################################################"
    echo "  GROUP: task=${TASK} backbone=${BACKBONE_TAG} (anatomy=${ANATOMY_DIR})"
    echo "############################################################"

    LAST_SEED_CFG=""

    for SEED in "${SEEDS[@]}"; do

        RUN_DIR="${GROUP_DIR}/seed${SEED}"

        if seed_pair_done "$TASK" "$BACKBONE_TAG" "$SEED"; then
            echo ""
            echo "--- SKIP (already completed): task=${TASK} backbone=${BACKBONE_TAG} seed=${SEED} ---"
            LAST_SEED_CFG="${RUN_DIR}/seed${SEED}_config.yaml"
            continue
        fi

        # MODIFIED: purge any stale/partial rows for this (task,backbone,
        # seed) before (re)running it -- a crash between writing this
        # seed's TSV rows and completing the rest could otherwise leave a
        # duplicate row on retry (the rows are about to be regenerated
        # fresh below). Idempotent: no-op if no such rows exist yet.
        TMP_TSV="${RESULTS_TSV}.tmp"
        awk -F'\t' -v t="$TASK" -v b="$BACKBONE_TAG" -v s="$SEED" \
            'NR==1 || !($1==t && $2==b && $3==s)' "$RESULTS_TSV" > "$TMP_TSV"
        mv "$TMP_TSV" "$RESULTS_TSV"

        RUN_NAME="${RUN_GROUP}-seed${SEED}"
        TMP_CFG="/tmp/${RUN_GROUP}_seed${SEED}.yaml"

        echo ""
        echo "============================================================"
        echo "  START: ${RUN_NAME}  (task=${TASK}, backbone=${BACKBONE_TAG}, seed=${SEED})"
        echo "============================================================"

        # --- 1. Generate per-run YAML ---
        python3 -c "
import yaml, sys

task        = sys.argv[1]
anatomy     = sys.argv[2]
has_hc18    = sys.argv[3] == '1'
backbone    = sys.argv[4]
seed        = int(sys.argv[5])
base_cfg    = sys.argv[6]
out_cfg     = sys.argv[7]
run_name    = sys.argv[8]
run_dir     = sys.argv[9]
images_root = sys.argv[10]
ann_root    = sys.argv[11]
hc18_root   = sys.argv[12]

with open(base_cfg) as f:
    cfg = yaml.safe_load(f)

cfg['seed_everything'] = seed
cfg['trainer']['logger']['init_args']['name'] = run_name

ckpt_cb = cfg['trainer']['callbacks'][0]['init_args']
ckpt_cb['dirpath']  = run_dir
ckpt_cb['filename'] = f'seed{seed}_best'

data = cfg['data']['init_args']
data['images_dir']   = f'{images_root}/{anatomy}'
data['ann_train_csv'] = f'{ann_root}/{anatomy}_Train.csv'
data['ann_test_csv']  = f'{ann_root}/{anatomy}_Test.csv'
data['task']          = task
data['loader_seed']   = seed
if has_hc18:
    data['hc18_ann_csvs'] = [f'{hc18_root}/Head_Train.csv', f'{hc18_root}/Head_Test.csv']
else:
    data.pop('hc18_ann_csvs', None)

n = cfg['model']['init_args']['network']['init_args']
n['encoder']['init_args']['backbone_name'] = backbone

with open(out_cfg, 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
" "$TASK" "$ANATOMY_DIR" "$HAS_HC18" "$BACKBONE_NAME" "$SEED" "$BASE_CONFIG" "$TMP_CFG" "$RUN_NAME" "$RUN_DIR" "$IMAGES_ROOT" "$ANN_ROOT" "$HC18_ROOT"

        # --- 2. Verify critical hyperparams ---
        echo "--- Verifying config for task=${TASK} backbone=${BACKBONE_TAG} seed=${SEED} ---"
        python3 - "$TMP_CFG" "$TASK" "$BACKBONE_NAME" "$SEED" "$HAS_HC18" <<'PYEOF'
import yaml, sys

cfg_path = sys.argv[1]
task     = sys.argv[2]
backbone = sys.argv[3]
seed     = int(sys.argv[4])
has_hc18 = sys.argv[5] == '1'

with open(cfg_path) as f:
    cfg = yaml.safe_load(f)

m    = cfg['model']['init_args']
n    = m['network']['init_args']
data = cfg['data']['init_args']
trainer = cfg['trainer']
early_stopping = trainer['callbacks'][1]['init_args']

checks = [
    ('seed_everything',      cfg['seed_everything'],         seed),
    ('loader_seed',          data.get('loader_seed'),        seed),
    ('task',                 data.get('task'),               task),
    ('data.class_path',      cfg['data']['class_path'],
                              'datasets.multicentre_dataset.MulticentreLandmarkDataModule'),
    ('hc18_ann_csvs present', bool(data.get('hc18_ann_csvs')), has_hc18),
    ('endpoint_order_invariant_nme', m.get('endpoint_order_invariant_nme', False), True),
    ('loss_type',            m['loss_type'],                 'hybrid'),
    ('heatmap_head',         n['heatmap_head'],              'deconv_v2'),
    ('freeze_backbone',      n['freeze_backbone'],           False),
    ('num_blocks',           n['num_blocks'],                3),
    ('masked_attn_enabled',  n['masked_attn_enabled'],       True),
    ('use_fpn',              n.get('use_fpn', False),        True),
    ('fpn_layers',           n.get('fpn_layers'),            [4, 8, 12]),
    ('pixel_center_align',   data.get('pixel_center_align', False), True),
    ('rotate_augment',       data.get('rotate_augment', False), True),
    ('scale_augment',        data.get('scale_augment', False), True),
    ('img_size',             data.get('img_size'),             [512, 512]),
    ('heatmap_size (data)',  data.get('heatmap_size'),         [64, 64]),
    ('heatmap_size (net)',   n['heatmap_size'],              [64, 64]),
    ('sigma',                data['sigma'],                  4.0),
    ('val_fraction',         data.get('val_fraction'),       0.1),
    ('val_split_seed',       data.get('val_split_seed'),     42),
    ('batch_size',           data.get('batch_size'),         16),
    ('max_epochs',           trainer.get('max_epochs'),      200),
    ('validation interval',  trainer.get('check_val_every_n_epoch'), 5),
    ('early-stop patience',  early_stopping.get('patience'), 20),
    ('early-stop min_delta', early_stopping.get('min_delta'), 0.005),
    ('backbone_name',        n.get('encoder', {}).get('init_args', {}).get('backbone_name'),
                              backbone),
]

all_ok = True
for key, got, expected in checks:
    ok = got == expected
    print(f'  {"[OK]  " if ok else "[FAIL]"} {key}: {got!r}' + ('' if ok else f'  expected={expected!r}'))
    if not ok:
        all_ok = False

if not all_ok:
    print('\n[ERROR] Config verification failed -- aborting')
    sys.exit(1)
print(f'\n[OK] All checks passed for task={task} backbone={backbone} seed={seed}')
PYEOF

        mkdir -p "${RUN_DIR}"

        # --- 3. Train ---
        echo ""
        echo "--- Training: ${RUN_NAME} ---"
        python3 main_landmark.py fit --config "${TMP_CFG}"

        # --- 4. Collect checkpoints ---
        BEST_CKPT="${RUN_DIR}/seed${SEED}_best.ckpt"
        if [ ! -f "${BEST_CKPT}" ]; then
            FOUND=$(find "${RUN_DIR}" -maxdepth 1 -name "seed${SEED}_best*.ckpt" 2>/dev/null | sort | tail -1)
            [ -n "${FOUND}" ] && cp "${FOUND}" "${BEST_CKPT}" && echo "[OK] best ckpt: $(basename "${FOUND}")"
        fi

        LAST_SRC="${RUN_DIR}/last.ckpt"
        FINAL_CKPT="${RUN_DIR}/seed${SEED}_final.ckpt"
        if [ -f "${LAST_SRC}" ]; then
            # rename, not copy -- avoids one redundant ~checkpoint-sized
            # file per run (see the resolution-screening disk optimisation)
            mv "${LAST_SRC}" "${FINAL_CKPT}"
            echo "[OK] final ckpt saved (last.ckpt renamed, no duplicate copy)"
        else
            echo "[WARN] last.ckpt not found -- skipping final checkpoint"
        fi

        cp "${TMP_CFG}" "${RUN_DIR}/seed${SEED}_config.yaml"
        LAST_SEED_CFG="${RUN_DIR}/seed${SEED}_config.yaml"

        python3 scripts/dump_test_image_order.py \
            --config "${TMP_CFG}" \
            --out "${RUN_DIR}/test_image_order.csv"

        # --- 5. Test best/final checkpoints under BOTH metrics ---
        for CKPT_TAG in best final; do
            CKPT_PATH="${RUN_DIR}/seed${SEED}_${CKPT_TAG}.ckpt"
            if [ ! -f "${CKPT_PATH}" ]; then
                echo "[ERROR] ${CKPT_PATH} not found -- task=${TASK} backbone=${BACKBONE_TAG} seed=${SEED} incomplete, aborting (NOT recording result)"
                exit 1
            fi

            echo ""
            echo "--- Test (${CKPT_TAG}, swap-min): ${CKPT_PATH} ---"
            LOG_SWAPMIN="${RUN_DIR}/seed${SEED}_${CKPT_TAG}_swapmin_test_log.txt"
            DUMP_SWAPMIN="${RUN_DIR}/seed${SEED}_${CKPT_TAG}_swapmin_per_image.csv"
            python3 main_landmark.py test \
                --config "${TMP_CFG}" \
                --ckpt_path "${CKPT_PATH}" \
                --model.init_args.test_nme_dump_path "${DUMP_SWAPMIN}" 2>&1 \
                | tee "${LOG_SWAPMIN}" \
                | grep -E "test_nme|Test NME|Per-sample NME dumped"
            SWAPMIN_NME=$(grep "Test NME:" "${LOG_SWAPMIN}" | grep -oE "[0-9]+\.[0-9]+" | tail -1 || echo "")

            # fixed-channel: CLI override on the SAME per-run config
            # (Rule 32 -- never the shared base config)
            echo ""
            echo "--- Test (${CKPT_TAG}, fixed-channel): ${CKPT_PATH} ---"
            LOG_FIXED="${RUN_DIR}/seed${SEED}_${CKPT_TAG}_fixedchannel_test_log.txt"
            DUMP_FIXED="${RUN_DIR}/seed${SEED}_${CKPT_TAG}_fixedchannel_per_image.csv"
            python3 main_landmark.py test \
                --config "${TMP_CFG}" \
                --ckpt_path "${CKPT_PATH}" \
                --model.init_args.endpoint_order_invariant_nme false \
                --model.init_args.test_nme_dump_path "${DUMP_FIXED}" 2>&1 \
                | tee "${LOG_FIXED}" \
                | grep -E "test_nme|Test NME|Per-sample NME dumped"
            FIXEDCHANNEL_NME=$(grep "Test NME:" "${LOG_FIXED}" | grep -oE "[0-9]+\.[0-9]+" | tail -1 || echo "")

            if [ -z "$SWAPMIN_NME" ] || [ -z "$FIXEDCHANNEL_NME" ]; then
                echo "[ERROR] Failed to parse Test NME (swap-min='${SWAPMIN_NME}' fixed-channel='${FIXEDCHANNEL_NME}') for task=${TASK} backbone=${BACKBONE_TAG} seed=${SEED} ckpt_tag=${CKPT_TAG} -- incomplete, aborting (NOT recording result)"
                exit 1
            fi
            echo -e "${TASK}\t${BACKBONE_TAG}\t${SEED}\t${CKPT_TAG}\t${SWAPMIN_NME}\t${FIXEDCHANNEL_NME}" >> "$RESULTS_TSV"
        done

        echo ""
        echo "--- DONE: ${RUN_NAME} ---"

    done

    # --- Only ensemble + treat the group as complete once ALL 5 canonical
    #     seeds (not just whatever SEEDS was reduced to, e.g. under
    #     CANARY_ONLY) have both checkpoint tags recorded. ---
    if ! seeds_all_done "$TASK" "$BACKBONE_TAG"; then
        ANY_GROUP_PARTIAL=1
        echo ""
        echo "############################################################"
        echo "  GROUP PARTIAL (not all 5 seeds done): ${GROUP_KEY}"
        echo "  Not ensembling, not marking complete. Re-invoke this script"
        echo "  (without CANARY_ONLY) to train the remaining seeds."
        echo "############################################################"
        continue
    fi

    echo ""
    echo "--- Ensembling group: ${GROUP_KEY} ---"
    FIXED_ENSEMBLE_CFG="/tmp/${RUN_GROUP}_ensemble_fixedchannel.yaml"
    python3 -c "
import yaml
with open('${LAST_SEED_CFG}') as f:
    cfg = yaml.safe_load(f)
cfg['model']['init_args']['endpoint_order_invariant_nme'] = False
with open('${FIXED_ENSEMBLE_CFG}', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
"
    for CKPT_TAG in best final; do
        # Build the checkpoint list from the canonical seed set rather than
        # a broad glob.  TSV rows alone are not sufficient evidence that the
        # underlying checkpoints still exist, and ensemble_test.py is valid
        # for arbitrary ensemble sizes, so it would otherwise happily run a
        # four-model ensemble and produce a plausible-looking number.
        CKPTS=()
        for SEED in "${ALL_SEEDS[@]}"; do
            CKPT_PATH="${GROUP_DIR}/seed${SEED}/seed${SEED}_${CKPT_TAG}.ckpt"
            if [ ! -f "$CKPT_PATH" ]; then
                echo "[ERROR] Missing ${CKPT_TAG} checkpoint required for the formal 5-model ensemble:"
                echo "        ${CKPT_PATH}"
                echo "        TSV rows exist for this group, so restore the checkpoint or"
                echo "        remove that seed's stale TSV rows before retraining it."
                exit 1
            fi
            CKPTS+=("$CKPT_PATH")
        done
        if [ "${#CKPTS[@]}" -ne 5 ]; then
            echo "[ERROR] Expected exactly 5 ${CKPT_TAG} checkpoints, found ${#CKPTS[@]}"
            exit 1
        fi

        echo ""
        echo "=== Ensemble (${CKPT_TAG}, swap-min): task=${TASK} backbone=${BACKBONE_TAG} ==="
        python3 ablation/ensemble_test.py \
            --config "${LAST_SEED_CFG}" \
            --align-fetal-channels \
            --ckpts "${CKPTS[@]}" \
            2>&1 | tee "${GROUP_DIR}/ensemble_${CKPT_TAG}_swapmin_log.txt"

        echo ""
        echo "=== Ensemble (${CKPT_TAG}, fixed-channel): task=${TASK} backbone=${BACKBONE_TAG} ==="
        python3 ablation/ensemble_test.py \
            --config "${FIXED_ENSEMBLE_CFG}" \
            --align-fetal-channels \
            --ckpts "${CKPTS[@]}" \
            2>&1 | tee "${GROUP_DIR}/ensemble_${CKPT_TAG}_fixedchannel_log.txt"
    done

    # Re-verify (not just assume) the group is now actually complete --
    # if any ensemble step above failed to produce a log, group_is_complete
    # will correctly report incomplete on the next invocation rather than
    # this script wrongly declaring victory.
    if group_is_complete "$TASK" "$BACKBONE_TAG" "$GROUP_DIR"; then
        echo ""
        echo "############################################################"
        echo "  GROUP DONE: ${GROUP_KEY}"
        echo "############################################################"
    else
        echo ""
        echo "############################################################"
        echo "  [WARN] GROUP ${GROUP_KEY}: seeds finished but one or more"
        echo "  ensemble logs are missing or lack a verified 5-model result"
        echo "  -- re-invoke this script to retry the ensemble step for this"
        echo "  group before trusting it."
        echo "############################################################"
        exit 1
    fi
    df -h "$BACKUP_ROOT"
    echo "NOTE: this group's checkpoints are at ${GROUP_DIR}."
    echo "Once you've confirmed the ensemble numbers above, consider"
    echo "archiving/deleting this group's .ckpt files if disk space is"
    echo "needed before the next group starts -- the per-image swap-min/"
    echo "fixed-channel CSVs under each seed dir (with raw pred/gt"
    echo "coordinates) preserve enough for later re-analysis even if the"
    echo "raw .ckpt files are removed. Do not delete without checking"
    echo "first (Rule 29 in project pitfalls memory)."
    echo ""
    echo "Stopping here by design (safe-stop, see script header) --"
    echo "re-invoke this script to process the next incomplete group."
    exit 0

done
done

echo ""
echo "============================================================"
if [ "$ANY_GROUP_PARTIAL" -eq 1 ]; then
    echo "  Multicentre 5-seed ablation: run finished, but at least one"
    echo "  group is only PARTIALLY done (see GROUP PARTIAL messages above"
    echo "  -- expected under CANARY_ONLY). Re-invoke without CANARY_ONLY"
    echo "  to continue training the remaining seeds."
else
    echo "  Multicentre 5-seed ablation: ALL GROUPS ALREADY COMPLETE"
fi
echo "============================================================"
echo "Per-seed results: ${RESULTS_TSV}"
echo "Per-group ensemble logs: ${BACKUP_ROOT}/multicentre-<task>-<backbone>/ensemble_*_log.txt"
echo "W&B: https://wandb.ai/ucabnx1-ucl/eomt-landmark"
