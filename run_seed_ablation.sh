#!/usr/bin/env bash
# =============================================================================
# run_seed_ablation.sh — Run 18 reproduce + seed ablation (5 seeds)
#
# Seeds: 42 (reproduce), 0, 123, 2024, 3407
# Checkpoints saved to: checkpoints/seed_ablation/seed{N}/
#   seed{N}_best.pth   — best val-NME checkpoint (copy of Lightning .ckpt)
#   seed{N}_final.pth  — final epoch checkpoint  (copy of last.ckpt)
#   seed{N}_config.yaml — exact config used for this run
#
# Usage (on AutoDL server):
#   cd /root/eomt
#   bash run_seed_ablation.sh
#
# Notes:
#   - val_split_seed stays fixed at 42 for all runs (same val set, reproducible comparison)
#   - seed_everything controls torch/numpy/python-random/DataLoader shuffle
#   - Lightning .ckpt files are loadable with torch.load(); .pth copies are identical
# =============================================================================

set -euo pipefail

BASE_CONFIG="configs/landmark/bpd_vit_small.yaml"
SEEDS=(42 0 123 2024 3407)

# ---------------------------------------------------------------------------
# Preflight: verify base config exists and is valid YAML
# ---------------------------------------------------------------------------
if [ ! -f "${BASE_CONFIG}" ]; then
    echo "[ERROR] Base config not found: ${BASE_CONFIG}"
    exit 1
fi

python3 -c "import yaml; yaml.safe_load(open('${BASE_CONFIG}'))" \
    && echo "[OK] Base config is valid YAML: ${BASE_CONFIG}" \
    || { echo "[ERROR] Base config is invalid YAML — aborting"; exit 1; }

mkdir -p checkpoints/seed_ablation

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
for SEED in "${SEEDS[@]}"; do

    RUN_NAME="run18_seed${SEED}"
    RUN_DIR="checkpoints/seed_ablation/seed${SEED}"
    TMP_CFG="/tmp/run18_seed${SEED}.yaml"

    echo ""
    echo "============================================================"
    echo "  START: ${RUN_NAME}  (seed=${SEED})"
    echo "============================================================"

    # --- 1. Generate per-run YAML (seed + W&B name + checkpoint path) ---
    python3 -c "
import yaml, sys

seed        = int(sys.argv[1])
base_cfg    = sys.argv[2]
out_cfg     = sys.argv[3]
run_name    = f'run18_seed{seed}'
run_dir     = f'checkpoints/seed_ablation/seed{seed}'

with open(base_cfg) as f:
    cfg = yaml.safe_load(f)

cfg['seed_everything'] = seed
cfg['trainer']['logger']['init_args']['name'] = run_name

# ModelCheckpoint is always callbacks[0] in this config
ckpt_cb = cfg['trainer']['callbacks'][0]['init_args']
ckpt_cb['dirpath'] = run_dir
ckpt_cb['filename'] = f'seed{seed}_best'

with open(out_cfg, 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
" "$SEED" "$BASE_CONFIG" "$TMP_CFG"

    # --- 2. Print full config for verification ---
    echo ""
    echo "--- Full config for ${RUN_NAME} ---"
    python3 -c "import yaml; print(yaml.dump(yaml.safe_load(open('${TMP_CFG}')), default_flow_style=False, sort_keys=False))"

    # --- 3. Verify all critical hyperparams match Run 18 ---
    echo "--- Verifying critical hyperparams ---"
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
    ('use_refinement_head',  n['use_refinement_head'],       False),
    ('freeze_backbone',      n['freeze_backbone'],           False),
    ('num_blocks',           n['num_blocks'],                3),
    ('masked_attn_enabled',  n['masked_attn_enabled'],       True),
    ('backbone_name', n.get('encoder', {}).get('init_args', {}).get('backbone_name', n.get('backbone_name')), 'vit_small_patch14_reg4_dinov2'),
    ('heatmap_size',         data['heatmap_size'],           [64, 64]),
    ('sigma',                data['sigma'],                  4.0),
    ('val_split_seed',       data['val_split_seed'],         42),
    ('save_last',            ckpt.get('save_last'),          True),
    ('save_top_k',           ckpt.get('save_top_k'),         1),
]

all_ok = True
for key, got, expected in checks:
    ok = got == expected
    status = 'OK' if ok else f'FAIL  got={got!r}  expected={expected!r}'
    print(f'  {"[OK]  " if ok else "[FAIL]"} {key}: {status}')
    if not ok:
        all_ok = False

if not all_ok:
    print('\n[ERROR] Config verification failed — aborting run')
    sys.exit(1)
print('\n[OK] All checks passed for seed=' + str(seed))
PYEOF

    mkdir -p "${RUN_DIR}"

    # --- 4. Run training ---
    echo ""
    echo "--- Training: python main_landmark.py fit --config ${TMP_CFG} ---"
    python3 main_landmark.py fit --config "${TMP_CFG}"

    # --- 5. Save best checkpoint as .pth ---
    BEST_CKPT="${RUN_DIR}/seed${SEED}_best.ckpt"
    if [ -f "${BEST_CKPT}" ]; then
        cp "${BEST_CKPT}" "${RUN_DIR}/seed${SEED}_best.pth"
        echo "[OK] seed${SEED}_best.pth saved"
    else
        # Fallback: ModelCheckpoint may append metric values to filename
        FOUND=$(find "${RUN_DIR}" -maxdepth 1 -name "seed${SEED}_best*.ckpt" 2>/dev/null | sort | tail -1)
        if [ -n "${FOUND}" ]; then
            cp "${FOUND}" "${RUN_DIR}/seed${SEED}_best.pth"
            echo "[OK] seed${SEED}_best.pth saved (from: $(basename "${FOUND}"))"
        else
            echo "[WARN] Best checkpoint not found in ${RUN_DIR} — skipping .pth copy"
        fi
    fi

    # --- 6. Save final checkpoint as .pth (from save_last=true → last.ckpt) ---
    LAST_CKPT="${RUN_DIR}/last.ckpt"
    if [ -f "${LAST_CKPT}" ]; then
        cp "${LAST_CKPT}" "${RUN_DIR}/seed${SEED}_final.pth"
        echo "[OK] seed${SEED}_final.pth saved"
    else
        echo "[WARN] last.ckpt not found in ${RUN_DIR} — save_last may not have triggered"
    fi

    # --- 7. Save exact config alongside checkpoints ---
    cp "${TMP_CFG}" "${RUN_DIR}/seed${SEED}_config.yaml"
    echo "[OK] Config snapshot saved: ${RUN_DIR}/seed${SEED}_config.yaml"

    echo ""
    echo "--- DONE: ${RUN_NAME} ---"
    echo "    best:  ${RUN_DIR}/seed${SEED}_best.pth"
    echo "    final: ${RUN_DIR}/seed${SEED}_final.pth"
    echo "    cfg:   ${RUN_DIR}/seed${SEED}_config.yaml"

done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  Seed ablation complete!"
echo "============================================================"
echo ""
echo "Checkpoint summary:"
for SEED in "${SEEDS[@]}"; do
    RUN_DIR="checkpoints/seed_ablation/seed${SEED}"
    BEST="${RUN_DIR}/seed${SEED}_best.pth"
    FINAL="${RUN_DIR}/seed${SEED}_final.pth"
    BEST_STATUS=$( [ -f "${BEST}"  ] && echo "OK" || echo "MISSING" )
    FINAL_STATUS=$( [ -f "${FINAL}" ] && echo "OK" || echo "MISSING" )
    echo "  seed=${SEED}  best=${BEST_STATUS}  final=${FINAL_STATUS}"
done
echo ""
echo "W&B runs: https://wandb.ai/ucabnx1-ucl/eomt-landmark"
