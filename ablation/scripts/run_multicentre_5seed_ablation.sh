#!/usr/bin/env bash
# =============================================================================
# run_multicentre_5seed_ablation.sh — Multicentre (pooled FP+HC18+UCL),
# DINOv2 + DINOv3, all 5 fetal tasks (bpd/ofd/apad/tad/fl), 5 seeds each
# = 50 training runs total. DeconvHeadV2+FPN+UDP+rotate+scale matched-
# protocol recipe -- evaluates the already-fixed pipeline, no component
# re-ablation on Multicentre (see project memory / 04_experimental_setup
# .tex's sec:setup-evidence-scope).
#
# Processes ONE (task, backbone) group fully -- all 5 seeds, then that
# group's ensemble -- before moving to the next group, so disk usage
# stays bounded to roughly one group at a time (~3GB) rather than all 10
# groups' checkpoints accumulating simultaneously (10 groups x ~3GB would
# be ~30GB, uncomfortably close to this project's repeated disk-full
# incidents -- see Rule 16/29 in project pitfalls memory). Checkpoints
# are written directly to the data disk (no post-hoc mv step, per the
# resolution-screening disk optimisation) under BACKUP_ROOT below.
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
# it).
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
# =============================================================================

set -euo pipefail

BASE_CONFIG="configs/landmark/multicentre_fpn_udp_rotate_scale.yaml"
SEEDS=(42 0 123 2024 3407)

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
#     -> runs ONLY task=bpd, backbone=dinov2, seed=42 (1 of 50 runs),
#        including its group-ensemble step (a 1-model "ensemble", just to
#        exercise that code path too).
#   bash ablation/scripts/run_multicentre_5seed_ablation.sh
#     -> full sweep (resumes past anything CANARY_ONLY already completed).
if [ "${CANARY_ONLY:-0}" = "1" ]; then
    TASKS=("Head:bpd:1")
    BACKBONES=("dinov2:vit_small_patch14_reg4_dinov2")
    SEEDS=(42)
    echo "[CANARY_ONLY] restricting to task=bpd backbone=dinov2 seed=42 only"
fi

RESULTS_TSV="multicentre_5seed_results.tsv"
DONE_MARKER="checkpoints/.multicentre_5seed_completed.txt"
GROUP_DONE_MARKER="checkpoints/.multicentre_5seed_groups_completed.txt"
BACKUP_ROOT="/root/autodl-tmp/saved_checkpoints/multicentre_5seed"

if [ ! -d "${IMAGES_ROOT}/Head" ] || [ ! -f "${ANN_ROOT}/Head_Train.csv" ]; then
    echo "[ERROR] Multicentre data not found under ${IMAGES_ROOT} / ${ANN_ROOT} -- transfer it first"
    exit 1
fi

mkdir -p checkpoints "$(dirname "$DONE_MARKER")" "$BACKUP_ROOT"
touch "$DONE_MARKER" "$GROUP_DONE_MARKER"
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
check_disk
echo "[OK] checkpoint disk preflight"
df -h "$BACKUP_ROOT"

for TASK_SPEC in "${TASKS[@]}"; do
IFS=":" read -r ANATOMY_DIR TASK HAS_HC18 <<< "$TASK_SPEC"

for BACKBONE_SPEC in "${BACKBONES[@]}"; do
IFS=":" read -r BACKBONE_TAG BACKBONE_NAME <<< "$BACKBONE_SPEC"

    GROUP_KEY="${TASK}_${BACKBONE_TAG}"
    RUN_GROUP="multicentre-${TASK}-${BACKBONE_TAG}"
    GROUP_DIR="${BACKUP_ROOT}/${RUN_GROUP}"

    if grep -qxF "${GROUP_KEY}" "$GROUP_DONE_MARKER" 2>/dev/null; then
        echo ""
        echo "############################################################"
        echo "  SKIP whole group (already completed + ensembled): ${GROUP_KEY}"
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

        PAIR_KEY="${GROUP_KEY}_${SEED}"
        RUN_DIR="${GROUP_DIR}/seed${SEED}"
        TSV_PAIR_ROWS=$(awk -F'\t' -v t="$TASK" -v b="$BACKBONE_TAG" -v s="$SEED" \
            'NR>1 && $1==t && $2==b && $3==s && $4=="best" {x=1} \
             NR>1 && $1==t && $2==b && $3==s && $4=="final" {y=1} \
             END {print x+y}' \
            "$RESULTS_TSV")
        if grep -qxF "${PAIR_KEY}" "$DONE_MARKER" 2>/dev/null \
           && [ "$TSV_PAIR_ROWS" -eq 2 ]; then
            echo ""
            echo "--- SKIP (already completed): task=${TASK} backbone=${BACKBONE_TAG} seed=${SEED} ---"
            LAST_SEED_CFG="${RUN_DIR}/seed${SEED}_config.yaml"
            continue
        fi

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
    ('heatmap_size (net)',   n['heatmap_size'],              [64, 64]),
    ('sigma',                data['sigma'],                  4.0),
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
                echo "[ERROR] ${CKPT_PATH} not found -- task=${TASK} backbone=${BACKBONE_TAG} seed=${SEED} incomplete, aborting (NOT marking DONE)"
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
                echo "[ERROR] Failed to parse Test NME (swap-min='${SWAPMIN_NME}' fixed-channel='${FIXEDCHANNEL_NME}') for task=${TASK} backbone=${BACKBONE_TAG} seed=${SEED} ckpt_tag=${CKPT_TAG} -- incomplete, aborting (NOT marking DONE)"
                exit 1
            fi
            echo -e "${TASK}\t${BACKBONE_TAG}\t${SEED}\t${CKPT_TAG}\t${SWAPMIN_NME}\t${FIXEDCHANNEL_NME}" >> "$RESULTS_TSV"
        done

        # Only mark this (task, backbone, seed) DONE now that both
        # checkpoint tags x both metrics are confirmed captured above.
        echo "${PAIR_KEY}" >> "$DONE_MARKER"
        echo ""
        echo "--- DONE: ${RUN_NAME} ---"

    done

    # --- Group finished (all 5 seeds): ensemble across seeds, both
    #     checkpoint tags x both metrics. The swap-min ensemble uses
    #     LAST_SEED_CFG as-is (endpoint_order_invariant_nme: true, same
    #     across every seed's config in this group); the fixed-channel
    #     ensemble uses a copy with that one field flipped. ---
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
        echo ""
        echo "=== Ensemble (${CKPT_TAG}, swap-min): task=${TASK} backbone=${BACKBONE_TAG} ==="
        python3 ablation/ensemble_test.py \
            --config "${LAST_SEED_CFG}" \
            --ckpts "${GROUP_DIR}"/seed*/seed*_${CKPT_TAG}.ckpt \
            2>&1 | tee "${GROUP_DIR}/ensemble_${CKPT_TAG}_swapmin_log.txt"

        echo ""
        echo "=== Ensemble (${CKPT_TAG}, fixed-channel): task=${TASK} backbone=${BACKBONE_TAG} ==="
        python3 ablation/ensemble_test.py \
            --config "${FIXED_ENSEMBLE_CFG}" \
            --ckpts "${GROUP_DIR}"/seed*/seed*_${CKPT_TAG}.ckpt \
            2>&1 | tee "${GROUP_DIR}/ensemble_${CKPT_TAG}_fixedchannel_log.txt"
    done

    echo "${GROUP_KEY}" >> "$GROUP_DONE_MARKER"
    echo ""
    echo "############################################################"
    echo "  GROUP DONE: ${GROUP_KEY}"
    echo "############################################################"
    df -h "$BACKUP_ROOT"
    echo "NOTE: this group's checkpoints are at ${GROUP_DIR}."
    echo "Once you've confirmed the ensemble numbers above, consider"
    echo "archiving/deleting this group's .ckpt files if disk space is"
    echo "needed before the next group starts -- the per-image swap-min/"
    echo "fixed-channel CSVs under each seed dir (with raw pred/gt"
    echo "coordinates) preserve enough for later re-analysis even if the"
    echo "raw .ckpt files are removed. Do not delete without checking"
    echo "first (Rule 29 in project pitfalls memory)."

done
done

echo ""
echo "============================================================"
echo "  Multicentre 5-seed ablation: ALL GROUPS DONE"
echo "============================================================"
echo "Per-seed results: ${RESULTS_TSV}"
echo "Per-group ensemble logs: ${BACKUP_ROOT}/multicentre-<task>-<backbone>/ensemble_*_log.txt"
echo "W&B: https://wandb.ai/ucabnx1-ucl/eomt-landmark"
