#!/usr/bin/env bash
# =============================================================================
# run_fpn_ablation.sh — FPN multi-depth fusion (deconv_v2, 64x64) 5-seed ablation
#
# Why 5-seed from the start (not a single spot-check like the original plan):
# deconv_v2 has demonstrated high seed variance (64x64: std 3.00%; hm128: std
# 1.12%/3.01%), AND the hm128 ablation showed the SAME seed (2024) giving
# noticeably different results between a standalone run (12.02%/13.02%) and
# the ablation script's regenerated-config run (15.30%/15.21%) — meaning even
# "same seed" isn't fully reproducible here (DataLoader has no explicit
# generator/worker_init_fn, so exact RNG consumption before dataloader
# creation can drift). A single FPN run could easily be a lucky/unlucky draw
# and mislead the FPN-vs-baseline conclusion — run all 5 before deciding.
#
# Seeds: 42, 0, 123, 2024, 3407 (same set as the other two ablations).
#
# Checkpoint layout:
#   checkpoints/fpn-ablation/seed{N}/
#     seed{N}_best.ckpt   — saved by ModelCheckpoint
#     seed{N}_final.ckpt  — copy of last.ckpt
#     seed{N}_best_test_nme.txt   — test NME from val-best checkpoint
#     seed{N}_final_test_nme.txt  — test NME from last checkpoint
#
# Usage (on AutoDL server):
#   cd /root/eomt
#   bash ablation/scripts/run_fpn_ablation.sh
# =============================================================================

set -euo pipefail

BASE_CONFIG="configs/landmark/bpd_deconv_v2_fpn.yaml"
SEEDS=(42 0 123 2024 3407)

# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------
if [ ! -f "${BASE_CONFIG}" ]; then
    echo "[ERROR] Base config not found: ${BASE_CONFIG}"
    exit 1
fi

python3 -c "import yaml; yaml.safe_load(open('${BASE_CONFIG}'))" \
    && echo "[OK] Base config valid: ${BASE_CONFIG}" \
    || { echo "[ERROR] Invalid YAML — aborting"; exit 1; }

mkdir -p checkpoints/fpn-ablation

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
for SEED in "${SEEDS[@]}"; do

    RUN_NAME="fpn-ablation-seed${SEED}"
    RUN_DIR="checkpoints/fpn-ablation/seed${SEED}"
    TMP_CFG="/tmp/fpn_seed${SEED}.yaml"

    echo ""
    echo "============================================================"
    echo "  START: ${RUN_NAME}  (seed=${SEED})"
    echo "============================================================"

    # --- 1. Generate per-run YAML (seed + W&B name + checkpoint path) ---
    python3 -c "
import yaml, sys

seed     = int(sys.argv[1])
base_cfg = sys.argv[2]
out_cfg  = sys.argv[3]
run_name = f'fpn-ablation-seed{seed}'
run_dir  = f'checkpoints/fpn-ablation/seed{seed}'

with open(base_cfg) as f:
    cfg = yaml.safe_load(f)

cfg['seed_everything'] = seed
cfg['trainer']['logger']['init_args']['name'] = run_name

ckpt_cb = cfg['trainer']['callbacks'][0]['init_args']
ckpt_cb['dirpath']  = run_dir
ckpt_cb['filename'] = f'seed{seed}_best'

with open(out_cfg, 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
" "$SEED" "$BASE_CONFIG" "$TMP_CFG"

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
    ('loss_type',            m['loss_type'],                 'hybrid'),
    ('lambda_coord',         m['lambda_coord'],              0.1),
    ('temperature',          m['temperature'],               10.0),
    ('alpha',                m['alpha'],                     5.0),
    ('heatmap_head',         n['heatmap_head'],              'deconv_v2'),
    ('use_refinement_head',  n['use_refinement_head'],       False),
    ('freeze_backbone',      n['freeze_backbone'],           False),
    ('num_blocks',           n['num_blocks'],                3),
    ('masked_attn_enabled',  n['masked_attn_enabled'],       True),
    ('use_fpn',              n.get('use_fpn', False),        True),
    ('fpn_layers',           n.get('fpn_layers'),            [4, 8, 12]),
    ('use_lora',             n.get('use_lora', False),       False),
    ('backbone_name', n.get('encoder', {}).get('init_args', {}).get('backbone_name', n.get('backbone_name')), 'vit_small_patch14_reg4_dinov2'),
    ('heatmap_size (net)',   n['heatmap_size'],              [64, 64]),
    ('heatmap_size (data)',  data['heatmap_size'],           [64, 64]),
    ('sigma',                data['sigma'],                  4.0),
    ('val_split_seed',       data['val_split_seed'],         42),
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

    # --- 5. Test val-best checkpoint ---
    if [ -f "${BEST_CKPT}" ]; then
        echo ""
        echo "--- Test (val-best checkpoint): ${BEST_CKPT} ---"
        python3 main_landmark.py test \
            --config "${TMP_CFG}" \
            --ckpt_path "${BEST_CKPT}" 2>&1 \
            | tee "${RUN_DIR}/seed${SEED}_best_test_log.txt" \
            | grep -E "test_nme|Test NME"
        # NOTE: must grep for the "Test NME:" line FIRST, then extract the float
        # from just that line (taking the LAST match). Extracting from the raw
        # log directly picks up the tqdm progress bar's it/s number (e.g.
        # "7.89it/s") which precedes "Test NME:" on the same line — that bug
        # silently produced "0.28%" (from an unrelated float) in an earlier run.
        grep "Test NME:" "${RUN_DIR}/seed${SEED}_best_test_log.txt" \
            | grep -oE "[0-9]+\.[0-9]+" | tail -1 > "${RUN_DIR}/seed${SEED}_best_test_nme.txt" || true
    else
        echo "[WARN] Best checkpoint not found — skipping test"
    fi

    # --- 6. Test last checkpoint ---
    if [ -f "${FINAL_CKPT}" ]; then
        echo ""
        echo "--- Test (last checkpoint): ${FINAL_CKPT} ---"
        python3 main_landmark.py test \
            --config "${TMP_CFG}" \
            --ckpt_path "${FINAL_CKPT}" 2>&1 \
            | tee "${RUN_DIR}/seed${SEED}_final_test_log.txt" \
            | grep -E "test_nme|Test NME"
        grep "Test NME:" "${RUN_DIR}/seed${SEED}_final_test_log.txt" \
            | grep -oE "[0-9]+\.[0-9]+" | tail -1 > "${RUN_DIR}/seed${SEED}_final_test_nme.txt" || true
    else
        echo "[WARN] Final checkpoint not found — skipping test"
    fi

    echo ""
    echo "--- DONE: ${RUN_NAME} ---"

done

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  FPN (blocks 4/8/12, 64x64, deconv_v2) — 5-seed ablation complete"
echo "============================================================"
echo ""
printf "  %-8s  %-20s  %-20s\n" "seed" "test NME (val-best)" "test NME (last)"
printf "  %-8s  %-20s  %-20s\n" "--------" "--------------------" "--------------------"

for SEED in "${SEEDS[@]}"; do
    RUN_DIR="checkpoints/fpn-ablation/seed${SEED}"
    BEST_NME=$(cat "${RUN_DIR}/seed${SEED}_best_test_nme.txt"  2>/dev/null || echo "N/A")
    FINAL_NME=$(cat "${RUN_DIR}/seed${SEED}_final_test_nme.txt" 2>/dev/null || echo "N/A")
    printf "  %-8s  %-20s  %-20s\n" "$SEED" "${BEST_NME}%" "${FINAL_NME}%"
done

echo ""
echo "For comparison:"
echo "  einsum baseline (5-seed):        mean 17.48% ± 1.95% (last)"
echo "  deconv_v2 64x64 (5-seed):        mean 15.18% ± 3.00% (last)"
echo "  hm128 (5-seed):                  mean 15.16% ± 1.12% (val-best) / 15.48% ± 3.01% (last)"
echo "  HRNet reference:                 8%"
echo ""
echo "W&B: https://wandb.ai/ucabnx1-ucl/eomt-landmark"
