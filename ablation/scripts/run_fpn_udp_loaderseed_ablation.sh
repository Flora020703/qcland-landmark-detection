#!/usr/bin/env bash
# =============================================================================
# run_fpn_udp_loaderseed_ablation.sh — Exp 7 rerun: FPN + UDP, WITH
# loader_seed enabled, 5-seed ablation.
#
# Fixes a real methodological gap for the publication-grade cross-dataset
# ablation table: the original Exp 7 (run_fpn_udp_ablation.sh,
# bpd_deconv_v2_fpn_udp.yaml) predates the loader_seed determinism fix,
# while Exp 8a (run_fpn_udp_ema_ablation.sh, bpd_deconv_v2_fpn_udp_ema.yaml)
# was the first experiment to use it - so Exp7's "seed=N" and Exp8a's
# "seed=N" don't correspond to the same actual data order, meaning the
# "+FPN+UDP -> +FPN+UDP+EMA" step in the ablation table isn't a clean A/B
# (see docs/experiment_summary_exp4-9_ensemble.md 3.6). This script reruns
# Exp7's exact architecture (FPN+UDP, no EMA) with loader_seed = seed per
# run, same pattern as Exp8a, so the two are now directly comparable.
#
# Reference (original, non-deterministic Exp7 - do not overwrite, kept for
# comparison/history): val-best 13.21%±1.92%, last 13.67%±1.08%.
#
# Seeds: 42, 0, 123, 2024, 3407 (same set as every other ablation).
#
# Checkpoint layout:
#   checkpoints/fpn-udp-loaderseed-ablation/seed{N}/
#     seed{N}_best.ckpt   — saved by ModelCheckpoint
#     seed{N}_final.ckpt  — copy of last.ckpt
#     seed{N}_{best,final}_test_nme.txt
#
# Usage (on AutoDL server):
#   cd /root/eomt
#   bash ablation/scripts/run_fpn_udp_loaderseed_ablation.sh
# =============================================================================

set -euo pipefail

BASE_CONFIG="configs/landmark/bpd_deconv_v2_fpn_udp_loaderseed.yaml"
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

mkdir -p checkpoints/fpn-udp-loaderseed-ablation

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
for SEED in "${SEEDS[@]}"; do

    RUN_NAME="fpn-udp-loaderseed-ablation-seed${SEED}"
    RUN_DIR="checkpoints/fpn-udp-loaderseed-ablation/seed${SEED}"
    TMP_CFG="/tmp/fpn_udp_loaderseed_seed${SEED}.yaml"

    echo ""
    echo "============================================================"
    echo "  START: ${RUN_NAME}  (seed=${SEED})"
    echo "============================================================"

    # --- 1. Generate per-run YAML (seed + loader_seed + W&B name + ckpt path) ---
    python3 -c "
import yaml, sys

seed     = int(sys.argv[1])
base_cfg = sys.argv[2]
out_cfg  = sys.argv[3]
run_name = f'fpn-udp-loaderseed-ablation-seed{seed}'
run_dir  = f'checkpoints/fpn-udp-loaderseed-ablation/seed{seed}'

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
    ('loader_seed',          data.get('loader_seed'),        seed),
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
    ('pixel_center_align',   data.get('pixel_center_align', False), True),
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
echo "  FPN + UDP + loader_seed (deconv_v2, 64x64) — 5-seed ablation complete"
echo "============================================================"
echo ""
printf "  %-8s  %-20s  %-20s\n" "seed" "test NME (val-best)" "test NME (last)"
printf "  %-8s  %-20s  %-20s\n" "--------" "--------------------" "--------------------"

for SEED in "${SEEDS[@]}"; do
    RUN_DIR="checkpoints/fpn-udp-loaderseed-ablation/seed${SEED}"
    BEST_NME=$(cat "${RUN_DIR}/seed${SEED}_best_test_nme.txt"  2>/dev/null || echo "N/A")
    FINAL_NME=$(cat "${RUN_DIR}/seed${SEED}_final_test_nme.txt" 2>/dev/null || echo "N/A")
    printf "  %-8s  %-20s  %-20s\n" "$SEED" "${BEST_NME}%" "${FINAL_NME}%"
done

echo ""
echo "For comparison:"
echo "  Exp7 original (no loader_seed, 5-seed): val-best 13.21% ± 1.92% / last 13.67% ± 1.08%"
echo "  Exp8a (+EMA, WITH loader_seed, 5-seed):  val-best 15.50% ± 1.13% / last 12.09% ± 2.74%"
echo "  This run should now be the clean baseline Exp8a's EMA effect is measured against."
echo "  HRNet reference: 8%"
echo ""
echo "W&B: https://wandb.ai/ucabnx1-ucl/eomt-landmark"
