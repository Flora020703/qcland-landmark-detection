#!/usr/bin/env bash
# =============================================================================
# run_300w_ablation.sh — 300W face landmark detection, augmentation-free
# single run (seed=42, from configs/landmark/face300w_deconv_v2_fpn_udp.yaml).
#
# [2026-07-15] SUPERSEDED: this already ran once and produced rung 3
# (+FPN+UDP) of the 4-rung data-efficiency ablation - see
# ablation/scripts/run_300w_data_efficiency_ablation.sh for the other 3
# rungs (baseline, +FPN, +FPN+UDP+EMA) and the actual thesis rationale.
# The config this script points at was renamed face300w_vit_small.yaml ->
# face300w_deconv_v2_fpn_udp.yaml to match that ladder's naming, but the
# already-completed run's checkpoints/W&B run are still under the OLD name
# (checkpoints/face300w-baseline/, wandb run "face300w-baseline") - not
# retroactively renamed. No need to re-run this script unless reproducing
# rung 3 from scratch.
#
# This was NOT a 5-seed ablation like the fetal scripts (still isn't) - it
# was the first-ever 300W run, following the project's own "confirm the
# pipeline works before optimising" order (see prompt's dev-step
# suggestions).
#
# Reuses BPD/OFD's architecture unchanged (DeconvHead V2, FPN [4,8,12], UDP
# pixel_center_align, hybrid loss) - only task-specific things differ:
# num_q=68, num_blocks=6 (tentative, unablated), heatmap sigma=1.5 (tentative,
# fetal's sigma=4.0 would badly overlap adjacent 68-point landmarks), input
# 256x256 (300W standard, vs fetal's 512x512), and NME normalised by
# nme_norm_pair=[36,45] (outer eye corners - see training/landmark_detection.py
# and datasets/face300w_dataset.py for why this must NOT be an eye-centre
# average).
#
# Reports three test numbers against the SAME trained checkpoint by
# generating three per-subset config overrides (test_subset: common /
# challenging / full) - same "generate temp yaml, override, run test" pattern
# as the fetal ablation scripts use for best-vs-last checkpoints.
#
# Before running for real, do the local sanity steps first:
#   python3 scripts/test_dataload.py face300w
# (visualises a few landmark overlays and checks split sizes/shapes)
#
# Usage:
#   cd /root/eomt   (or repo root locally for a CPU dry run)
#   bash ablation/scripts/run_300w_ablation.sh
# =============================================================================

set -euo pipefail

BASE_CONFIG="configs/landmark/face300w_deconv_v2_fpn_udp.yaml"
RUN_NAME="face300w-deconv-v2-fpn-udp"
RUN_DIR="checkpoints/${RUN_NAME}"

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

echo "--- Verifying critical hyperparams ---"
python3 - "$BASE_CONFIG" <<'PYEOF'
import yaml, sys

with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)

m    = cfg['model']['init_args']
n    = m['network']['init_args']
data = cfg['data']['init_args']

checks = [
    ('num_landmarks',        m['num_landmarks'],              68),
    ('nme_norm_pair',        list(m.get('nme_norm_pair', [])), [36, 45]),
    ('loss_type',            m['loss_type'],                  'hybrid'),
    ('num_q',                n['num_q'],                      68),
    ('heatmap_head',         n['heatmap_head'],               'deconv_v2'),
    ('masked_attn_enabled',  n['masked_attn_enabled'],        True),
    ('use_fpn',              n.get('use_fpn', False),         True),
    ('fpn_layers',           n.get('fpn_layers'),             [4, 8, 12]),
    ('backbone_name', n.get('encoder', {}).get('init_args', {}).get('backbone_name'), 'vit_small_patch14_reg4_dinov2'),
    ('img_size (data)',      data['img_size'],                [256, 256]),
    ('heatmap_size (data)',  data['heatmap_size'],            [64, 64]),
    ('pixel_center_align',   data.get('pixel_center_align', False), True),
    ('test_subset',          data.get('test_subset'),         'full'),
    ('augment',              data.get('augment', True),       False),
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
print('\n[OK] All checks passed')
PYEOF

mkdir -p "${RUN_DIR}"

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
echo ""
echo "--- Training: ${RUN_NAME} ---"
python3 main_landmark.py fit --config "${BASE_CONFIG}"

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
    echo "[WARN] last.ckpt not found — skipping final checkpoint"
fi

# ---------------------------------------------------------------------------
# Test: common / challenging / full, for both val-best and last checkpoints
# ---------------------------------------------------------------------------
SUBSETS=(common challenging full)

declare -A RESULTS

for CKPT_TAG in best final; do
    CKPT_PATH="${RUN_DIR}/${RUN_NAME}_${CKPT_TAG}.ckpt"
    if [ ! -f "${CKPT_PATH}" ]; then
        echo "[WARN] ${CKPT_PATH} not found — skipping all subset tests for ${CKPT_TAG}"
        continue
    fi

    for SUBSET in "${SUBSETS[@]}"; do
        TMP_CFG="/tmp/face300w_${CKPT_TAG}_${SUBSET}.yaml"
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
" "$BASE_CONFIG" "$TMP_CFG" "$SUBSET"

        echo ""
        echo "--- Test (${CKPT_TAG} checkpoint, ${SUBSET} subset) ---"
        LOG_FILE="${RUN_DIR}/${RUN_NAME}_${CKPT_TAG}_${SUBSET}_test_log.txt"
        python3 main_landmark.py test \
            --config "${TMP_CFG}" \
            --ckpt_path "${CKPT_PATH}" 2>&1 \
            | tee "${LOG_FILE}" \
            | grep -E "test_nme|Test NME"

        NME=$(grep "Test NME:" "${LOG_FILE}" | grep -oE "[0-9]+\.[0-9]+" | tail -1 || echo "N/A")
        RESULTS["${CKPT_TAG}_${SUBSET}"]="${NME}"
    done
done

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  300W face landmark baseline (single seed=42) complete"
echo "============================================================"
echo ""
printf "  %-12s  %-12s  %-12s\n" "checkpoint" "subset" "test NME"
printf "  %-12s  %-12s  %-12s\n" "------------" "------------" "------------"
for CKPT_TAG in best final; do
    for SUBSET in "${SUBSETS[@]}"; do
        NME="${RESULTS[${CKPT_TAG}_${SUBSET}]:-N/A}"
        printf "  %-12s  %-12s  %-12s\n" "$CKPT_TAG" "$SUBSET" "${NME}%"
    done
done

echo ""
echo "For comparison (HRNetV2-W18, HRNet-Facial-Landmark-Detection README):"
echo "  Common:      2.91"
echo "  Challenging: 5.11"
echo "  Full:        3.34"
echo "  (official 600-image private test set not reproduced - no local data)"
echo ""
echo "Reminder: num_blocks=6 and sigma=1.5 are first-guess values, not yet"
echo "ablated. If this baseline is far off, those are the first two knobs to"
echo "revisit (one at a time)."
echo ""
echo "W&B: https://wandb.ai/ucabnx1-ucl/eomt-landmark"
