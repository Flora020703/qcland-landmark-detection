#!/usr/bin/env bash
# =============================================================================
# run_bpd_resolution_screening.sh — BPD, single-seed (2024) img_size=256x256
# vs 512x512 screening, on the DINOv2 matched-protocol reference
# configuration (DeconvHeadV2 + FPN + UDP + rotate+scale, DINOv2 backbone,
# hybrid loss) -- NOT called a "final adopted pipeline"; the thesis does
# not select a single test-set-chosen final pipeline (Chapter 4's locked
# scope). heatmap_size is held fixed at 64x64 for both rungs -- only
# input resolution varies.
#
# This is a SCREENING, not a full ablation -- single seed only. Per Rule 30
# (feedback_training_pitfalls memory): report only what is directly
# manipulated (img_size, one seed) and directly measured (test NME) --
# do not overclaim "optimal resolution" or draw conclusions this single
# run can't support.
#
# Motivation: 512x512 was a one-off design choice ("align with EoMT
# convention") never compared against HRNet's own 256x256 input in a
# controlled experiment -- an unremoved confound in every EoMT-vs-HRNet
# comparison so far.
#
# NOTE: compute_nme() here is still the fixed-channel metric, not Di Vece's
# swap-min (permutation-invariant) metric -- the 256-vs-512 internal
# comparison is still fair (identical metric both rungs), but before
# formally reporting in the thesis, re-test both checkpoints under
# whatever fetal NME definition is finally adopted (no retraining needed --
# per-image dumps below make this cheap).
#
# Each test run also dumps per-image NME (via --model.init_args.
# test_nme_dump_path) and an (index -> filename) mapping (via
# scripts/dump_test_image_order.py), since 256/512 share the same test
# images and this is nearly free -- supports later outlier inspection,
# paired bootstrap, and re-scoring once the final NME metric is settled.
#
# Prerequisite: data must be on the server first --
#   /root/autodl-tmp/images/UCL/Head/
#   /root/autodl-tmp/annotations/UCL/Head_Train.csv
#   /root/autodl-tmp/annotations/UCL/Head_Test.csv
#
# Usage (on AutoDL server):
#   cd /root/eomt
#   nohup bash ablation/scripts/run_bpd_resolution_screening.sh > bpd_resolution_screening.log 2>&1 &
# =============================================================================

set -euo pipefail

BASE_CONFIG="configs/landmark/bpd_resolution_screening.yaml"
SEED=2024
RESOLUTIONS=(256 512)

RUN_GROUP="bpd-resolution-screening"
RESULTS_TSV="bpd_resolution_screening_results.tsv"
DONE_MARKER="checkpoints/.bpd_resolution_screening_completed_resolutions.txt"

if [ ! -d /root/autodl-tmp/images/UCL/Head ] \
   || [ ! -f /root/autodl-tmp/annotations/UCL/Head_Train.csv ] \
   || [ ! -f /root/autodl-tmp/annotations/UCL/Head_Test.csv ]; then
    echo "[ERROR] Head images or Train/Test annotations not found on server -- transfer images/UCL/Head/ and annotations/UCL/Head_*.csv first"
    exit 1
fi

mkdir -p checkpoints "$(dirname "$DONE_MARKER")"
touch "$DONE_MARKER"
[ -f "$RESULTS_TSV" ] || echo -e "resolution\tckpt_tag\tnme" > "$RESULTS_TSV"

if [ ! -f "${BASE_CONFIG}" ]; then
    echo "[ERROR] Base config not found: ${BASE_CONFIG}"
    exit 1
fi
python3 -c "import yaml; yaml.safe_load(open('${BASE_CONFIG}'))" \
    && echo "[OK] Base config valid: ${BASE_CONFIG}" \
    || { echo "[ERROR] Invalid YAML -- aborting"; exit 1; }

mkdir -p "checkpoints/${RUN_GROUP}"

for RES in "${RESOLUTIONS[@]}"; do

    if grep -qxF "${RES}" "$DONE_MARKER" 2>/dev/null; then
        echo ""
        echo "--- SKIP (already completed): resolution=${RES} ---"
        continue
    fi

    RES_TAG="res${RES}"
    RUN_NAME="${RUN_GROUP}-${RES_TAG}"
    RUN_DIR="checkpoints/${RUN_GROUP}/${RES_TAG}"
    TMP_CFG="/tmp/${RUN_GROUP}_${RES_TAG}.yaml"

    echo ""
    echo "============================================================"
    echo "  START: ${RUN_NAME}  (img_size=${RES}x${RES}, seed=${SEED})"
    echo "============================================================"

    # --- 1. Generate per-run YAML ---
    python3 -c "
import yaml, sys

res      = int(sys.argv[1])
seed     = int(sys.argv[2])
base_cfg = sys.argv[3]
out_cfg  = sys.argv[4]
run_name = sys.argv[5]
run_dir  = sys.argv[6]

with open(base_cfg) as f:
    cfg = yaml.safe_load(f)

cfg['seed_everything'] = seed
cfg['trainer']['logger']['init_args']['name'] = run_name
cfg['data']['init_args']['img_size'] = [res, res]
cfg['data']['init_args']['loader_seed'] = seed

ckpt_cb = cfg['trainer']['callbacks'][0]['init_args']
ckpt_cb['dirpath']  = run_dir
ckpt_cb['filename'] = f'{run_name}_best'

with open(out_cfg, 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
" "$RES" "$SEED" "$BASE_CONFIG" "$TMP_CFG" "$RUN_NAME" "$RUN_DIR"

    # --- 2. Verify critical hyperparams ---
    echo "--- Verifying config for resolution=${RES} ---"
    python3 - "$TMP_CFG" "$RES" "$SEED" <<'PYEOF'
import yaml, sys

cfg_path = sys.argv[1]
res      = int(sys.argv[2])
seed     = int(sys.argv[3])

with open(cfg_path) as f:
    cfg = yaml.safe_load(f)

m         = cfg['model']['init_args']
n         = m['network']['init_args']
data      = cfg['data']['init_args']

checks = [
    ('seed_everything',      cfg['seed_everything'],         seed),
    ('img_size',             data['img_size'],               [res, res]),
    ('heatmap_size (data)',  data['heatmap_size'],            [64, 64]),
    ('task',                 data.get('task'),               'bpd'),
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
                              'vit_small_patch14_reg4_dinov2'),
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
print(f'\n[OK] All checks passed for resolution={res}')
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

    # --- 4b. Dump (index -> filename) mapping for this resolution's test
    #     split, so per-image NME below can be joined back to actual images
    #     later (outlier inspection, paired bootstrap, re-scoring under a
    #     different NME definition once one is finalized). 256/512 share the
    #     same underlying test CSV/order, but this is dumped per-resolution
    #     directory to keep each run_dir self-contained.
    python3 scripts/dump_test_image_order.py \
        --config "${TMP_CFG}" \
        --out "${RUN_DIR}/test_image_order.csv"

    # --- 5. Test val-best / last checkpoints ---
    for CKPT_TAG in best final; do
        CKPT_PATH="${RUN_DIR}/${RUN_NAME}_${CKPT_TAG}.ckpt"
        if [ ! -f "${CKPT_PATH}" ]; then
            echo "[ERROR] ${CKPT_PATH} not found -- resolution=${RES} incomplete, aborting (NOT marking DONE)"
            exit 1
        fi
        echo ""
        echo "--- Test (${CKPT_TAG} checkpoint): ${CKPT_PATH} ---"
        LOG_FILE="${RUN_DIR}/${RUN_NAME}_${CKPT_TAG}_test_log.txt"
        PER_IMAGE_CSV="${RUN_DIR}/${RUN_NAME}_${CKPT_TAG}_per_image.csv"
        python3 main_landmark.py test \
            --config "${TMP_CFG}" \
            --ckpt_path "${CKPT_PATH}" \
            --model.init_args.test_nme_dump_path "${PER_IMAGE_CSV}" 2>&1 \
            | tee "${LOG_FILE}" \
            | grep -E "test_nme|Test NME|Per-sample NME dumped"

        NME=$(grep "Test NME:" "${LOG_FILE}" | grep -oE "[0-9]+\.[0-9]+" | tail -1 || echo "")
        if [ -z "$NME" ]; then
            echo "[ERROR] Failed to parse Test NME for resolution=${RES} ckpt_tag=${CKPT_TAG} -- incomplete, aborting (NOT marking DONE)"
            exit 1
        fi
        if [ ! -f "${PER_IMAGE_CSV}" ]; then
            echo "[ERROR] ${PER_IMAGE_CSV} was not written -- per-image dump failed for resolution=${RES} ckpt_tag=${CKPT_TAG}, aborting (NOT marking DONE)"
            exit 1
        fi
        echo -e "${RES}\t${CKPT_TAG}\t${NME}" >> "$RESULTS_TSV"
    done

    # MODIFIED: only mark this resolution DONE now that both best and final
    # NME are confirmed captured above.
    echo "${RES}" >> "$DONE_MARKER"
    echo ""
    echo "--- DONE: ${RUN_NAME} ---"

done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  BPD resolution screening (single-seed=${SEED}) -- summary"
echo "============================================================"
python3 - "$RESULTS_TSV" <<'PYEOF'
import sys
from collections import defaultdict

groups = defaultdict(dict)
with open(sys.argv[1]) as f:
    next(f)
    for line in f:
        res, ckpt_tag, nme = line.rstrip("\n").split("\t")
        groups[res][ckpt_tag] = float(nme)  # dict assignment -- naturally de-dupes repeat rows

print(f"\n  {'res':<8}{'best':<8}{'final':<8}")
for res in sorted(groups, key=int):
    row = groups[res]
    print(f"  {res:<8}{row.get('best', float('nan')):<8.2f}{row.get('final', float('nan')):<8.2f}")
PYEOF

echo ""
echo "NOTE: single-seed screening only -- report as a sensitivity check in the"
echo "thesis (Rule 30: state only what was manipulated/measured), not with"
echo "5-seed-level causal confidence, and not as establishing either"
echo "resolution as formally optimal."
echo "W&B: https://wandb.ai/ucabnx1-ucl/eomt-landmark"
