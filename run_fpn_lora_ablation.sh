#!/usr/bin/env bash
# =============================================================================
# run_fpn_lora_ablation.sh — LoRA + FPN (deconv_v2, 64x64) 5-seed ablation
#
# Run only after run_fpn_ablation.sh (Experiment 4) results are in — LoRA is
# meant to layer on top of FPN, not stand alone. FPN's 5-seed result: val-best
# mean 14.49% ± 2.01% vs no-FPN 15.18% ± 3.00% (both mean and variance improved,
# though the ~0.7pt mean gap is within noise at n=5). This adds LoRA on the
# same FPN backbone to see if backbone adaptation compounds with it.
#
# Requires (run once before this script):
#   pip install peft --break-system-packages
#
# Seeds: 42, 0, 123, 2024, 3407 (same set as the other three ablations).
#
# Checkpoint layout:
#   checkpoints/fpn-lora-ablation/seed{N}/
#     seed{N}_best.ckpt   — saved by ModelCheckpoint
#     seed{N}_final.ckpt  — copy of last.ckpt
#     seed{N}_best_test_nme.txt   — test NME from val-best checkpoint
#     seed{N}_final_test_nme.txt  — test NME from last checkpoint
#
# Usage (on AutoDL server):
#   cd /root/eomt
#   pip install peft --break-system-packages
#   bash run_fpn_lora_ablation.sh
# =============================================================================

set -euo pipefail

BASE_CONFIG="configs/landmark/bpd_deconv_v2_fpn_lora.yaml"
SEEDS=(42 0 123 2024 3407)

# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------
if [ ! -f "${BASE_CONFIG}" ]; then
    echo "[ERROR] Base config not found: ${BASE_CONFIG}"
    exit 1
fi

python3 -c "import peft" 2>/dev/null \
    && echo "[OK] peft installed" \
    || { echo "[ERROR] peft not installed — run: pip install peft --break-system-packages"; exit 1; }

python3 -c "import yaml; yaml.safe_load(open('${BASE_CONFIG}'))" \
    && echo "[OK] Base config valid: ${BASE_CONFIG}" \
    || { echo "[ERROR] Invalid YAML — aborting"; exit 1; }

mkdir -p checkpoints/fpn-lora-ablation

# ---------------------------------------------------------------------------
# One-time sanity check: confirm LoRA params actually show up as trainable
# (target_modules=["qkv"] assumes timm's Attention exposes `self.qkv` — see
# the NOTE in models/eomt.py EoMT.__init__). If this prints 0 lora params,
# stop and fix target_modules before burning 5 training runs on it.
# ---------------------------------------------------------------------------
echo ""
echo "--- Sanity check: LoRA params actually trainable? ---"
python3 - "$BASE_CONFIG" <<'PYEOF'
import sys
import yaml

# Minimal manual instantiation (avoids running the full CLI/trainer)
sys.path.insert(0, ".")
from models.eomt import EoMT
from models.vit import ViT

with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)

n = cfg["model"]["init_args"]["network"]["init_args"]
enc_args = n["encoder"]["init_args"]

encoder = ViT(img_size=cfg["data"]["init_args"]["img_size"], **enc_args)
net = EoMT(
    encoder=encoder,
    num_classes=1,
    num_q=n["num_q"],
    num_blocks=n["num_blocks"],
    masked_attn_enabled=n["masked_attn_enabled"],
    freeze_backbone=n["freeze_backbone"],
    heatmap_head=n["heatmap_head"],
    heatmap_size=tuple(n["heatmap_size"]),
    use_fpn=n["use_fpn"],
    fpn_layers=n["fpn_layers"],
    use_lora=n["use_lora"],
    lora_rank=n["lora_rank"],
    lora_alpha=n["lora_alpha"],
    lora_dropout=n["lora_dropout"],
)

lora_params = sum(p.numel() for name, p in net.named_parameters() if "lora_" in name and p.requires_grad)
total_trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
print(f"LoRA trainable params: {lora_params:,}")
print(f"Total trainable params: {total_trainable:,}")
if lora_params == 0:
    print("[ERROR] 0 LoRA params found — target_modules=['qkv'] didn't match anything.")
    print("        Run: print(net.encoder.backbone) to find the real attention Linear name.")
    sys.exit(1)
print("[OK] LoRA adapters are wired up correctly")
PYEOF

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
for SEED in "${SEEDS[@]}"; do

    RUN_NAME="fpn-lora-ablation-seed${SEED}"
    RUN_DIR="checkpoints/fpn-lora-ablation/seed${SEED}"
    TMP_CFG="/tmp/fpn_lora_seed${SEED}.yaml"

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
run_name = f'fpn-lora-ablation-seed{seed}'
run_dir  = f'checkpoints/fpn-lora-ablation/seed{seed}'

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
    ('use_lora',             n.get('use_lora', False),       True),
    ('lora_rank',            n.get('lora_rank'),             8),
    ('lora_alpha',           n.get('lora_alpha'),             16),
    ('lora_dropout',         n.get('lora_dropout'),          0.1),
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
echo "  LoRA + FPN (deconv_v2, 64x64) — 5-seed ablation complete"
echo "============================================================"
echo ""
printf "  %-8s  %-20s  %-20s\n" "seed" "test NME (val-best)" "test NME (last)"
printf "  %-8s  %-20s  %-20s\n" "--------" "--------------------" "--------------------"

for SEED in "${SEEDS[@]}"; do
    RUN_DIR="checkpoints/fpn-lora-ablation/seed${SEED}"
    BEST_NME=$(cat "${RUN_DIR}/seed${SEED}_best_test_nme.txt"  2>/dev/null || echo "N/A")
    FINAL_NME=$(cat "${RUN_DIR}/seed${SEED}_final_test_nme.txt" 2>/dev/null || echo "N/A")
    printf "  %-8s  %-20s  %-20s\n" "$SEED" "${BEST_NME}%" "${FINAL_NME}%"
done

echo ""
echo "For comparison:"
echo "  einsum baseline (5-seed):        mean 17.48% ± 1.95% (last)"
echo "  deconv_v2 64x64 (5-seed):        mean 15.18% ± 3.00% (last)"
echo "  hm128 (5-seed):                  mean 15.16% ± 1.12% (val-best) / 15.48% ± 3.01% (last)"
echo "  FPN 64x64 (5-seed):              mean 14.49% ± 2.01% (val-best) / 14.91% ± 2.94% (last)"
echo "  HRNet reference:                 8%"
echo ""
echo "W&B: https://wandb.ai/ucabnx1-ucl/eomt-landmark"
