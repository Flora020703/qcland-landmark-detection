#!/usr/bin/env bash
# =============================================================================
# run_dinov3_300w_ablation.sh — DINOv2 vs DINOv3 backbone comparison, 300W leg.
#
# Config: face300w_dinov3_fpn_udp.yaml (FPN+UDP — 300W's best/flagship-table
# config, identical to face300w_deconv_v2_fpn_udp.yaml except backbone_name).
# Lowest priority of the three DINOv3 legs (longest per-run time, ~2.5h x 5),
# run this last. No ensemble for 300W (see project convention — report
# 5-seed mean±std only, common/challenging/full subsets).
#
# DINOv2 reference (5-seed, best checkpoint, full subset): 5.93% ± 0.15%
# (all 4 DINOv2 rungs land in 5.93-5.99%, statistically indistinguishable —
# DINOv3's job here is to show whether a stronger backbone can actually move
# this number, since regularization tricks provably cannot at this data scale).
#
# batch_size stays at 4 (same OOM reasoning as the DINOv2 300W config: num_q=68
# x num_blocks=6 is the memory pressure, not the backbone). If DINOv3 OOMs at
# batch_size=4 where DINOv2 didn't, that is the one hyperparameter allowed to
# move (see project convention: don't retune anything except under a real
# memory constraint, and record it if it happens).
#
# Resumable: a completed seed is recorded in DONE_MARKER and skipped on
# rerun; checkpoints move off the system disk after each seed's NME is
# recorded (300W's checkpoint count already caused a disk-full crash once —
# see run_300w_data_efficiency_5seed_ablation.sh header).
#
# Usage (on AutoDL server):
#   cd /root/eomt
#   nohup bash ablation/scripts/run_dinov3_300w_ablation.sh > dinov3_300w_ablation.log 2>&1 &
# =============================================================================

set -euo pipefail

BASE_CONFIG="configs/landmark/face300w_dinov3_fpn_udp.yaml"
SEEDS=(42 0 123 2024 3407)
SUBSETS=(common challenging full)

RUN_GROUP="face300w-dinov3-fpn-udp"
RESULTS_TSV="dinov3_300w_results.tsv"
DONE_MARKER="checkpoints/.dinov3_300w_completed_seeds.txt"
BACKUP_ROOT="/root/autodl-tmp/saved_checkpoints/dinov3_300w"

mkdir -p checkpoints "$(dirname "$DONE_MARKER")"
touch "$DONE_MARKER"
[ -f "$RESULTS_TSV" ] || echo -e "seed\tckpt_tag\tsubset\tnme" > "$RESULTS_TSV"

if [ ! -f "${BASE_CONFIG}" ]; then
    echo "[ERROR] Base config not found: ${BASE_CONFIG}"
    exit 1
fi
python3 -c "import yaml; yaml.safe_load(open('${BASE_CONFIG}'))" \
    && echo "[OK] Base config valid: ${BASE_CONFIG}" \
    || { echo "[ERROR] Invalid YAML — aborting"; exit 1; }

mkdir -p "checkpoints/${RUN_GROUP}"

for SEED in "${SEEDS[@]}"; do

    if grep -qxF "${SEED}" "$DONE_MARKER" 2>/dev/null; then
        echo ""
        echo "--- SKIP (already completed): seed=${SEED} ---"
        continue
    fi

    RUN_NAME="${RUN_GROUP}-seed${SEED}"
    RUN_DIR="checkpoints/${RUN_GROUP}/seed${SEED}"
    TMP_CFG="/tmp/${RUN_GROUP}_seed${SEED}.yaml"

    echo ""
    echo "============================================================"
    echo "  START: ${RUN_NAME}  (seed=${SEED})"
    echo "============================================================"

    # --- 1. Generate per-run YAML ---
    python3 -c "
import yaml, sys

seed     = int(sys.argv[1])
base_cfg = sys.argv[2]
out_cfg  = sys.argv[3]
run_name = sys.argv[4]
run_dir  = sys.argv[5]

with open(base_cfg) as f:
    cfg = yaml.safe_load(f)

cfg['seed_everything'] = seed
cfg['trainer']['logger']['init_args']['name'] = run_name
cfg['data']['init_args']['loader_seed'] = seed

ckpt_cb = cfg['trainer']['callbacks'][0]['init_args']
ckpt_cb['dirpath']  = run_dir
ckpt_cb['filename'] = f'seed{seed}_best'

with open(out_cfg, 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
" "$SEED" "$BASE_CONFIG" "$TMP_CFG" "$RUN_NAME" "$RUN_DIR"

    # --- 2. Verify critical hyperparams ---
    echo "--- Verifying config for seed=${SEED} ---"
    python3 - "$TMP_CFG" "$SEED" <<'PYEOF'
import yaml, sys

cfg_path = sys.argv[1]
seed     = int(sys.argv[2])

with open(cfg_path) as f:
    cfg = yaml.safe_load(f)

m    = cfg['model']['init_args']
n    = m['network']['init_args']
data = cfg['data']['init_args']

checks = [
    ('seed_everything',      cfg['seed_everything'],          seed),
    ('loader_seed',          data.get('loader_seed'),         seed),
    ('num_landmarks',        m['num_landmarks'],               68),
    ('nme_norm_pair',        list(m.get('nme_norm_pair', [])), [36, 45]),
    ('loss_type',            m['loss_type'],                  'hybrid'),
    ('num_q',                n['num_q'],                       68),
    ('num_blocks',           n['num_blocks'],                  6),
    ('heatmap_head',         n['heatmap_head'],               'deconv_v2'),
    ('use_fpn',              n.get('use_fpn', False),         True),
    ('pixel_center_align',   data.get('pixel_center_align', False), True),
    ('img_size',             data['img_size'],                [256, 256]),
    ('heatmap_size',         data['heatmap_size'],            [64, 64]),
    ('sigma',                data['sigma'],                   1.5),
    ('batch_size',           data['batch_size'],              4),
    ('test_subset',          data.get('test_subset'),         'full'),
    ('augment',              data.get('augment', True),       False),
    ('backbone_name',        n.get('encoder', {}).get('init_args', {}).get('backbone_name'),
                              'facebook/dinov3-vits16-pretrain-lvd1689m'),
    ('delta_weights (not True)', m.get('delta_weights', False), False),
]

all_ok = True
for key, got, expected in checks:
    ok = got == expected
    print(f'  {"[OK]  " if ok else "[FAIL]"} {key}: {got!r}' + ('' if ok else f'  expected={expected!r}'))
    if not ok:
        all_ok = False

if not all_ok:
    print('\n[ERROR] Config verification failed — aborting')
    sys.exit(1)
print(f'\n[OK] All checks passed for seed={seed}')
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
        cp "${LAST_SRC}" "${FINAL_CKPT}"
        echo "[OK] final ckpt saved"
    else
        echo "[WARN] last.ckpt not found — skipping final checkpoint"
    fi

    cp "${TMP_CFG}" "${RUN_DIR}/seed${SEED}_config.yaml"

    # --- 5. Test: common / challenging / full, for best and final checkpoints ---
    for CKPT_TAG in best final; do
        CKPT_PATH="${RUN_DIR}/seed${SEED}_${CKPT_TAG}.ckpt"
        if [ ! -f "${CKPT_PATH}" ]; then
            echo "[WARN] ${CKPT_PATH} not found — skipping all subset tests for ${CKPT_TAG}"
            continue
        fi

        for SUBSET in "${SUBSETS[@]}"; do
            TMP_TEST_CFG="/tmp/${RUN_GROUP}_seed${SEED}_${CKPT_TAG}_${SUBSET}.yaml"
            python3 -c "
import yaml, sys

base_cfg = sys.argv[1]
out_cfg  = sys.argv[2]
subset   = sys.argv[3]

with open(base_cfg) as f:
    cfg = yaml.safe_load(f)

cfg['data']['init_args']['test_subset'] = subset

with open(out_cfg, 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
" "$TMP_CFG" "$TMP_TEST_CFG" "$SUBSET"

            echo ""
            echo "--- Test (${RUN_NAME}, ${CKPT_TAG} checkpoint, ${SUBSET} subset) ---"
            LOG_FILE="${RUN_DIR}/seed${SEED}_${CKPT_TAG}_${SUBSET}_test_log.txt"
            python3 main_landmark.py test \
                --config "${TMP_TEST_CFG}" \
                --ckpt_path "${CKPT_PATH}" 2>&1 \
                | tee "${LOG_FILE}" \
                | grep -E "test_nme|Test NME"

            NME=$(grep "Test NME:" "${LOG_FILE}" | grep -oE "[0-9]+\.[0-9]+" | tail -1 || echo "")
            [ -n "$NME" ] && echo -e "${SEED}\t${CKPT_TAG}\t${SUBSET}\t${NME}" >> "$RESULTS_TSV"
        done
    done

    # --- 6. Free system disk: move this seed's checkpoints off to the data disk ---
    mkdir -p "$(dirname "${BACKUP_ROOT}/seed${SEED}")"
    if [ -d "${RUN_DIR}" ]; then
        rm -rf "${BACKUP_ROOT}/seed${SEED}"
        mv "${RUN_DIR}" "${BACKUP_ROOT}/seed${SEED}" \
            && echo "[OK] moved checkpoint to ${BACKUP_ROOT}/seed${SEED} (system disk freed)" \
            || echo "[WARN] could not move checkpoint dir for ${RUN_NAME}"
    fi

    echo "${SEED}" >> "$DONE_MARKER"
    echo ""
    echo "--- DONE: ${RUN_NAME} ---"

done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  300W DINOv3 (FPN+UDP) — 5-seed ablation summary"
echo "============================================================"
python3 - "$RESULTS_TSV" <<'PYEOF'
import sys, statistics
from collections import defaultdict

groups = defaultdict(list)
with open(sys.argv[1]) as f:
    next(f)
    for line in f:
        seed, ckpt_tag, subset, nme = line.rstrip("\n").split("\t")
        groups[(ckpt_tag, subset)].append(float(nme))

ckpt_order = ["best", "final"]
subset_order = ["common", "challenging", "full"]

print(f"\n  {'ckpt':<8}{'subset':<13}{'n':<4}{'mean':<8}{'std':<8}")
for ckpt_tag in ckpt_order:
    for subset in subset_order:
        key = (ckpt_tag, subset)
        if key not in groups:
            continue
        vals = groups[key]
        mean = statistics.mean(vals)
        std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        print(f"  {ckpt_tag:<8}{subset:<13}{len(vals):<4}{mean:<8.2f}{std:<8.2f}")
PYEOF

echo ""
echo "DINOv2 reference (5-seed, best, full): 5.93% ± 0.15%"
echo "HRNetV2-W18 baseline (common/challenging/full): 2.91 / 5.11 / 3.34"
echo ""
echo "Raw per-seed numbers: ${RESULTS_TSV}"
echo "W&B: https://wandb.ai/ucabnx1-ucl/eomt-landmark"
