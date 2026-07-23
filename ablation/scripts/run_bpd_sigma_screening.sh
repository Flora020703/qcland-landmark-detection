#!/usr/bin/env bash
# =============================================================================
# run_bpd_sigma_screening.sh — BPD, single-seed (2024) sigma=1.0 vs sigma=4.0
# screening, on the ORIGINAL condition (einsum head, plain MSE loss,
# num_blocks=3, masked_attn_enabled=true, 512x512, no FPN, UDP, or
# additional rotation-scale augmentation) under the current, fixed NME
# implementation. NOTE: HeadLandmarkDataModule.setup() hardcodes
# augment=True on the train split regardless of config, so the
# inherited baseline flip (p=0.5) + colour-jitter pipeline remains
# enabled identically in BOTH rungs -- this is the correct
# reproduction of the original condition, not a gap.
#
# This is a SCREENING, not a full ablation -- single seed only, must be
# reported in the thesis as "screening"/"sensitivity check", never with
# 5-seed-level causal confidence (see project memory's Next-steps queue).
# Answers "does sigma affect training under the ORIGINAL einsum+plain-MSE
# condition", NOT "is sigma=4 still better under the final DeconvHeadV2+
# hybrid-loss pipeline" -- report as a historical-configuration sigma
# screening, not a final-pipeline sigma ablation.
#
# Decision rule for whether to extend to 5 seeds (42/0/123/2024/3407):
# compare final-checkpoint against final-checkpoint (matches this
# project's UCL reporting convention; val-best is diagnostic only).
#   - If sigma=1 clearly collapses (training curve shows no learning)
#     and sigma=4 learns normally: single-seed is sufficient as a
#     mechanistic sensitivity check, keep it single-seed.
#   - If the gap is small or the direction looks unstable, OR if sigma=4
#     is going to be claimed as a formally-selected hyperparameter
#     (not just "why we picked sigma=4 early on"): must extend to the
#     full 5 seeds before drawing a conclusion -- compare the gap
#     against BPD's known seed-to-seed variance (std ~0.5-3.0 depending
#     on rung, see project memory) rather than trusting a single pair.
#
# NOT comparable to the early Run1->Run2 chronological-summary numbers
# (those conflated sigma with num_blocks/masked_attn changes AND used a
# since-fixed, invalid NME implementation).
#
# Prerequisite: data must be on the server first --
#   /root/autodl-tmp/images/UCL/Head/
#   /root/autodl-tmp/annotations/UCL/Head_Train.csv
#   /root/autodl-tmp/annotations/UCL/Head_Test.csv
#
# Usage (on AutoDL server):
#   cd /root/eomt
#   nohup bash ablation/scripts/run_bpd_sigma_screening.sh > bpd_sigma_screening.log 2>&1 &
# =============================================================================

set -euo pipefail

BASE_CONFIG="configs/landmark/bpd_sigma_screening.yaml"
SEED=2024
SIGMAS=(1.0 4.0)

RUN_GROUP="bpd-sigma-screening"
RESULTS_TSV="bpd_sigma_screening_results.tsv"
DONE_MARKER="checkpoints/.bpd_sigma_screening_completed_sigmas.txt"

if [ ! -d /root/autodl-tmp/images/UCL/Head ] || [ ! -f /root/autodl-tmp/annotations/UCL/Head_Train.csv ]; then
    echo "[ERROR] Head data not found on server -- transfer images/UCL/Head/ and annotations/UCL/Head_*.csv first"
    exit 1
fi

mkdir -p checkpoints "$(dirname "$DONE_MARKER")"
touch "$DONE_MARKER"
[ -f "$RESULTS_TSV" ] || echo -e "sigma\tckpt_tag\tnme" > "$RESULTS_TSV"

if [ ! -f "${BASE_CONFIG}" ]; then
    echo "[ERROR] Base config not found: ${BASE_CONFIG}"
    exit 1
fi
python3 -c "import yaml; yaml.safe_load(open('${BASE_CONFIG}'))" \
    && echo "[OK] Base config valid: ${BASE_CONFIG}" \
    || { echo "[ERROR] Invalid YAML -- aborting"; exit 1; }

mkdir -p "checkpoints/${RUN_GROUP}"

for SIGMA in "${SIGMAS[@]}"; do

    if grep -qxF "${SIGMA}" "$DONE_MARKER" 2>/dev/null; then
        echo ""
        echo "--- SKIP (already completed): sigma=${SIGMA} ---"
        continue
    fi

    SIGMA_TAG="sigma$(echo "${SIGMA}" | tr -d '.')"   # 1.0 -> sigma10, 4.0 -> sigma40
    RUN_NAME="${RUN_GROUP}-${SIGMA_TAG}"
    RUN_DIR="checkpoints/${RUN_GROUP}/${SIGMA_TAG}"
    TMP_CFG="/tmp/${RUN_GROUP}_${SIGMA_TAG}.yaml"

    echo ""
    echo "============================================================"
    echo "  START: ${RUN_NAME}  (sigma=${SIGMA}, seed=${SEED})"
    echo "============================================================"

    # --- 1. Generate per-run YAML ---
    python3 -c "
import yaml, sys

sigma    = float(sys.argv[1])
seed     = int(sys.argv[2])
base_cfg = sys.argv[3]
out_cfg  = sys.argv[4]
run_name = sys.argv[5]
run_dir  = sys.argv[6]

with open(base_cfg) as f:
    cfg = yaml.safe_load(f)

cfg['seed_everything'] = seed
cfg['trainer']['logger']['init_args']['name'] = run_name
cfg['data']['init_args']['sigma'] = sigma
cfg['data']['init_args']['loader_seed'] = seed

ckpt_cb = cfg['trainer']['callbacks'][0]['init_args']
ckpt_cb['dirpath']  = run_dir
ckpt_cb['filename'] = f'{run_name}_best'

with open(out_cfg, 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
" "$SIGMA" "$SEED" "$BASE_CONFIG" "$TMP_CFG" "$RUN_NAME" "$RUN_DIR"

    # --- 2. Verify critical hyperparams ---
    echo "--- Verifying config for sigma=${SIGMA} ---"
    python3 - "$TMP_CFG" "$SIGMA" "$SEED" <<'PYEOF'
import yaml, sys

cfg_path = sys.argv[1]
sigma    = float(sys.argv[2])
seed     = int(sys.argv[3])

with open(cfg_path) as f:
    cfg = yaml.safe_load(f)

m         = cfg['model']['init_args']
n         = m['network']['init_args']
data      = cfg['data']['init_args']

checks = [
    ('seed_everything',      cfg['seed_everything'],         seed),
    ('sigma',                data['sigma'],                  sigma),
    ('task',                 data.get('task'),               'bpd'),
    ('loss_type',            m['loss_type'],                 'mse'),
    ('heatmap_head',         n['heatmap_head'],              'einsum'),
    ('freeze_backbone',      n['freeze_backbone'],           False),
    ('num_blocks',           n['num_blocks'],                3),
    ('masked_attn_enabled',  n['masked_attn_enabled'],       True),
    ('use_fpn',              n.get('use_fpn', False),        False),
    ('pixel_center_align',   data.get('pixel_center_align', False), False),
    ('rotate_augment',       data.get('rotate_augment', False), False),
    ('scale_augment',        data.get('scale_augment', False), False),
    ('heatmap_size (net)',   n['heatmap_size'],              [64, 64]),
    ('img_size',             data['img_size'],                [512, 512]),
    ('images_dir',           data['images_dir'],             '/root/autodl-tmp/images/UCL/Head'),
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
print(f'\n[OK] All checks passed for sigma={sigma}')
PYEOF

    mkdir -p "${RUN_DIR}"

    # --- 3. Train ---
    echo ""
    echo "--- Training: ${RUN_NAME} ---"
    python3 main_landmark.py fit --config "${TMP_CFG}"

    # --- 4. Collect checkpoints ---
    BEST_CKPT="${RUN_DIR}/${RUN_NAME}_best.ckpt"
    if [ ! -f "${BEST_CKPT}" ]; then
        FOUND=$(find "${RUN_DIR}" -maxdepth 1 -name "${RUN_NAME}_best*.ckpt" 2>/dev/null | sort | tail -1)
        [ -n "${FOUND}" ] && cp "${FOUND}" "${BEST_CKPT}" && echo "[OK] best ckpt: $(basename "${FOUND}")"
    fi

    LAST_SRC="${RUN_DIR}/last.ckpt"
    FINAL_CKPT="${RUN_DIR}/${RUN_NAME}_final.ckpt"
    if [ -f "${LAST_SRC}" ]; then
        cp "${LAST_SRC}" "${FINAL_CKPT}"
        echo "[OK] final ckpt saved"
    else
        echo "[WARN] last.ckpt not found -- skipping final checkpoint"
    fi

    cp "${TMP_CFG}" "${RUN_DIR}/${RUN_NAME}_config.yaml"

    # --- 5. Test val-best / last checkpoints ---
    for CKPT_TAG in best final; do
        CKPT_PATH="${RUN_DIR}/${RUN_NAME}_${CKPT_TAG}.ckpt"
        if [ ! -f "${CKPT_PATH}" ]; then
            echo "[ERROR] ${CKPT_PATH} not found -- sigma=${SIGMA} incomplete, aborting (NOT marking DONE)"
            exit 1
        fi
        echo ""
        echo "--- Test (${CKPT_TAG} checkpoint): ${CKPT_PATH} ---"
        LOG_FILE="${RUN_DIR}/${RUN_NAME}_${CKPT_TAG}_test_log.txt"
        python3 main_landmark.py test \
            --config "${TMP_CFG}" \
            --ckpt_path "${CKPT_PATH}" 2>&1 \
            | tee "${LOG_FILE}" \
            | grep -E "test_nme|Test NME"

        NME=$(grep "Test NME:" "${LOG_FILE}" | grep -oE "[0-9]+\.[0-9]+" | tail -1 || echo "")
        if [ -z "$NME" ]; then
            echo "[ERROR] Failed to parse Test NME for sigma=${SIGMA} ckpt_tag=${CKPT_TAG} -- incomplete, aborting (NOT marking DONE)"
            exit 1
        fi
        echo -e "${SIGMA}\t${CKPT_TAG}\t${NME}" >> "$RESULTS_TSV"
    done

    # MODIFIED: only mark this sigma DONE now that both best and final NME are
    # confirmed captured above.
    echo "${SIGMA}" >> "$DONE_MARKER"
    echo ""
    echo "--- DONE: ${RUN_NAME} ---"

done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  BPD sigma screening (single-seed=${SEED}) -- summary"
echo "============================================================"
python3 - "$RESULTS_TSV" <<'PYEOF'
import sys
from collections import defaultdict

groups = defaultdict(dict)
with open(sys.argv[1]) as f:
    next(f)
    for line in f:
        sigma, ckpt_tag, nme = line.rstrip("\n").split("\t")
        groups[sigma][ckpt_tag] = float(nme)

print(f"\n  {'sigma':<8}{'best':<8}{'final':<8}")
for sigma in sorted(groups, key=float):
    row = groups[sigma]
    print(f"  {sigma:<8}{row.get('best', float('nan')):<8.2f}{row.get('final', float('nan')):<8.2f}")
PYEOF

echo ""
echo "NOTE: single-seed screening only -- report as a sensitivity check in the"
echo "thesis, not with 5-seed-level causal confidence."
echo "W&B: https://wandb.ai/ucabnx1-ucl/eomt-landmark"
