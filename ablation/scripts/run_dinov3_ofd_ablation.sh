#!/usr/bin/env bash
# =============================================================================
# run_dinov3_ofd_ablation.sh — DINOv2 vs DINOv3 backbone comparison, OFD leg.
#
# Config: ofd_dinov3_fpn_udp.yaml (FPN+UDP, no EMA — OFD's best config,
# identical to ofd_deconv_v2_fpn_udp.yaml except backbone_name). Run this
# AFTER the BPD DINOv3 canary + 5-seed run has confirmed the DINOv2->DINOv3
# backbone swap trains correctly end-to-end (see run_dinov3_bpd_ablation.sh) —
# no separate canary needed here.
#
# DINOv2 reference: OFD FPN+UDP 5-seed last ensemble = 4.46% (already beats
# HRNet baseline 5.00%).
#
# Resumable: a completed seed is recorded in DONE_MARKER and skipped on
# rerun. Checkpoints are moved off the system disk after each seed's NME is
# recorded (same pattern as run_dinov3_bpd_ablation.sh / run_300w_data_
# efficiency_5seed_ablation.sh — a disk-full crash mid-ablation has bitten
# this project's scripts before).
#
# Usage (on AutoDL server):
#   cd /root/eomt
#   nohup bash ablation/scripts/run_dinov3_ofd_ablation.sh > dinov3_ofd_ablation.log 2>&1 &
# =============================================================================

set -euo pipefail

BASE_CONFIG="configs/landmark/ofd_dinov3_fpn_udp.yaml"
SEEDS=(42 0 123 2024 3407)

RUN_GROUP="ofd-dinov3-fpn-udp"
RESULTS_TSV="dinov3_ofd_results.tsv"
DONE_MARKER="checkpoints/.dinov3_ofd_completed_seeds.txt"
BACKUP_ROOT="/root/autodl-tmp/saved_checkpoints/dinov3_ofd"

mkdir -p checkpoints "$(dirname "$DONE_MARKER")"
touch "$DONE_MARKER"
[ -f "$RESULTS_TSV" ] || echo -e "seed\tckpt_tag\tnme" > "$RESULTS_TSV"

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
ckpt = cfg['trainer']['callbacks'][0]['init_args']
data = cfg['data']['init_args']

checks = [
    ('seed_everything',      cfg['seed_everything'],         seed),
    ('loader_seed',          data.get('loader_seed'),        seed),
    ('task',                 data.get('task'),               'ofd'),
    ('loss_type',            m['loss_type'],                 'hybrid'),
    ('heatmap_head',         n['heatmap_head'],              'deconv_v2'),
    ('freeze_backbone',      n['freeze_backbone'],           False),
    ('num_blocks',           n['num_blocks'],                3),
    ('masked_attn_enabled',  n['masked_attn_enabled'],       True),
    ('use_fpn',              n.get('use_fpn', False),        True),
    ('fpn_layers',           n.get('fpn_layers'),            [4, 8, 12]),
    ('use_lora',             n.get('use_lora', False),       False),
    ('pixel_center_align',   data.get('pixel_center_align', False), True),
    ('backbone_name',        n.get('encoder', {}).get('init_args', {}).get('backbone_name'),
                              'facebook/dinov3-vits16-pretrain-lvd1689m'),
    ('delta_weights (not True)', m.get('delta_weights', False), False),
    ('heatmap_size (net)',   n['heatmap_size'],              [64, 64]),
    ('sigma',                data['sigma'],                  4.0),
    ('images_dir',           data['images_dir'],             '/root/autodl-tmp/images/UCL/Head'),
    ('save_last',            ckpt.get('save_last'),          True),
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

    # --- 5. Test val-best / last checkpoints ---
    for CKPT_TAG in best final; do
        CKPT_PATH="${RUN_DIR}/seed${SEED}_${CKPT_TAG}.ckpt"
        if [ ! -f "${CKPT_PATH}" ]; then
            echo "[WARN] ${CKPT_PATH} not found — skipping test"
            continue
        fi
        echo ""
        echo "--- Test (${CKPT_TAG} checkpoint): ${CKPT_PATH} ---"
        LOG_FILE="${RUN_DIR}/seed${SEED}_${CKPT_TAG}_test_log.txt"
        python3 main_landmark.py test \
            --config "${TMP_CFG}" \
            --ckpt_path "${CKPT_PATH}" 2>&1 \
            | tee "${LOG_FILE}" \
            | grep -E "test_nme|Test NME"

        NME=$(grep "Test NME:" "${LOG_FILE}" | grep -oE "[0-9]+\.[0-9]+" | tail -1 || echo "")
        [ -n "$NME" ] && echo -e "${SEED}\t${CKPT_TAG}\t${NME}" >> "$RESULTS_TSV"
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
echo "  OFD DINOv3 (FPN+UDP) — ablation summary"
echo "============================================================"
python3 - "$RESULTS_TSV" <<'PYEOF'
import sys, statistics
from collections import defaultdict

groups = defaultdict(list)
with open(sys.argv[1]) as f:
    next(f)
    for line in f:
        seed, ckpt_tag, nme = line.rstrip("\n").split("\t")
        groups[ckpt_tag].append(float(nme))

print(f"\n  {'ckpt':<8}{'n':<4}{'mean':<8}{'std':<8}")
for tag in ("best", "final"):
    if tag not in groups:
        continue
    vals = groups[tag]
    mean = statistics.mean(vals)
    std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    print(f"  {tag:<8}{len(vals):<4}{mean:<8.2f}{std:<8.2f}")
PYEOF

echo ""
echo "DINOv2 reference: FPN+UDP 5-seed last ensemble = 4.46% (HRNet baseline 5.00%)"
echo ""
echo "Next step once all 5 seeds are done: ensemble on last checkpoints:"
echo "  python3 ablation/ensemble_test.py --config ${BASE_CONFIG} \\"
echo "      --ckpts ${BACKUP_ROOT}/seed*/seed*_final.ckpt"
echo ""
echo "W&B: https://wandb.ai/ucabnx1-ucl/eomt-landmark"
