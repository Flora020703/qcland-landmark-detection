#!/usr/bin/env bash
# =============================================================================
# run_300w_data_efficiency_ablation.sh — 300W data-efficiency ablation,
# 3 of 4 rungs (single seed=42 each; the 3rd rung, +FPN+UDP, already ran
# against configs/landmark/face300w_deconv_v2_fpn_udp.yaml - not repeated
# here).
#
# Purpose: this is NOT primarily about closing the gap to HRNet - it tests
# whether the cross-dataset finding from BPD/HC18 ("FPN/UDP/EMA are
# anti-overfitting tools whose benefit shrinks as training-set size grows")
# extends to 300W (3148 train images, natural RGB faces - a third dataset,
# a different visual domain, and bigger than either fetal dataset). If
# FPN/UDP/EMA give ~0 benefit here too, that's a thesis-worthy finding
# spanning two domains and three dataset scales.
#
# Rungs run by this script:
#   1. baseline    - configs/landmark/face300w_deconv_v2.yaml
#   2. +FPN        - configs/landmark/face300w_deconv_v2_fpn.yaml
#   4. +FPN+UDP+EMA - configs/landmark/face300w_deconv_v2_fpn_udp_ema.yaml
# (Rung 3, +FPN+UDP, already has results - see face300w_deconv_v2_fpn_udp.yaml,
# formerly face300w_vit_small.yaml; its completed run is still under the OLD
# name checkpoints/face300w-baseline/ on the server, not retroactively renamed.)
#
# Each rung: train once (seed=42, no augmentation), then test best/final
# (and, for the EMA rung, the materialized EMA checkpoint too) against all
# three 300W test subsets (common/challenging/full) - same "generate temp
# config, override, run test" pattern as the fetal ablation scripts.
#
# Usage (on AutoDL server):
#   cd /root/eomt
#   nohup bash ablation/scripts/run_300w_data_efficiency_ablation.sh > face300w_ablation.log 2>&1 &
# =============================================================================

set -euo pipefail

SUBSETS=(common challenging full)

# name : config path : has_ema (1/0)
RUNS=(
    "face300w-deconv-v2:configs/landmark/face300w_deconv_v2.yaml:0"
    "face300w-deconv-v2-fpn:configs/landmark/face300w_deconv_v2_fpn.yaml:0"
    "face300w-deconv-v2-fpn-udp-ema:configs/landmark/face300w_deconv_v2_fpn_udp_ema.yaml:1"
)

declare -A RESULTS

for ENTRY in "${RUNS[@]}"; do
    IFS=":" read -r RUN_NAME BASE_CONFIG HAS_EMA <<< "$ENTRY"
    RUN_DIR="checkpoints/${RUN_NAME}"

    echo ""
    echo "============================================================"
    echo "  START: ${RUN_NAME}"
    echo "============================================================"

    if [ ! -f "${BASE_CONFIG}" ]; then
        echo "[ERROR] Config not found: ${BASE_CONFIG}"
        exit 1
    fi
    python3 -c "import yaml; yaml.safe_load(open('${BASE_CONFIG}'))" \
        && echo "[OK] Config valid: ${BASE_CONFIG}" \
        || { echo "[ERROR] Invalid YAML — aborting"; exit 1; }

    echo "--- Verifying critical hyperparams ---"
    python3 - "$BASE_CONFIG" "$HAS_EMA" <<'PYEOF'
import yaml, sys

cfg_path = sys.argv[1]
has_ema  = sys.argv[2] == "1"

with open(cfg_path) as f:
    cfg = yaml.safe_load(f)

m    = cfg['model']['init_args']
n    = m['network']['init_args']
data = cfg['data']['init_args']
callbacks = cfg['trainer']['callbacks']
ema_cb = next((c for c in callbacks if c['class_path'] == 'training.ema.EMACallback'), None)

checks = [
    ('num_landmarks',        m['num_landmarks'],              68),
    ('nme_norm_pair',        list(m.get('nme_norm_pair', [])), [36, 45]),
    ('loss_type',            m['loss_type'],                  'hybrid'),
    ('num_q',                n['num_q'],                      68),
    ('num_blocks',           n['num_blocks'],                 6),
    ('heatmap_head',         n['heatmap_head'],               'deconv_v2'),
    ('img_size (data)',      data['img_size'],                [256, 256]),
    ('heatmap_size (data)',  data['heatmap_size'],            [64, 64]),
    ('sigma',                data['sigma'],                   1.5),
    ('batch_size',           data['batch_size'],              4),
    ('test_subset',          data.get('test_subset'),         'full'),
    ('augment',              data.get('augment', True),       False),
    ('ema_callback_present', ema_cb is not None,               has_ema),
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

    # --- Train ---
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

    CKPT_TAGS=(best final)

    # --- Materialize EMA checkpoint, if this rung has one ---
    if [ "$HAS_EMA" = "1" ]; then
        EMA_CKPT="${RUN_DIR}/${RUN_NAME}_ema.ckpt"
        if [ -f "${FINAL_CKPT}" ]; then
            echo ""
            echo "--- Materializing EMA checkpoint: ${EMA_CKPT} ---"
            python3 ablation/apply_ema.py "${FINAL_CKPT}" "${EMA_CKPT}"
            CKPT_TAGS+=(ema)
        else
            echo "[WARN] Final checkpoint not found — skipping EMA materialization"
        fi
    fi

    # --- Test: common / challenging / full, for each available checkpoint ---
    for CKPT_TAG in "${CKPT_TAGS[@]}"; do
        CKPT_PATH="${RUN_DIR}/${RUN_NAME}_${CKPT_TAG}.ckpt"
        if [ ! -f "${CKPT_PATH}" ]; then
            echo "[WARN] ${CKPT_PATH} not found — skipping all subset tests for ${CKPT_TAG}"
            continue
        fi

        for SUBSET in "${SUBSETS[@]}"; do
            TMP_CFG="/tmp/${RUN_NAME}_${CKPT_TAG}_${SUBSET}.yaml"
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
            echo "--- Test (${RUN_NAME}, ${CKPT_TAG} checkpoint, ${SUBSET} subset) ---"
            LOG_FILE="${RUN_DIR}/${RUN_NAME}_${CKPT_TAG}_${SUBSET}_test_log.txt"
            python3 main_landmark.py test \
                --config "${TMP_CFG}" \
                --ckpt_path "${CKPT_PATH}" 2>&1 \
                | tee "${LOG_FILE}" \
                | grep -E "test_nme|Test NME"

            NME=$(grep "Test NME:" "${LOG_FILE}" | grep -oE "[0-9]+\.[0-9]+" | tail -1 || echo "N/A")
            RESULTS["${RUN_NAME}_${CKPT_TAG}_${SUBSET}"]="${NME}"
        done
    done

    echo ""
    echo "--- DONE: ${RUN_NAME} ---"
done

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  300W data-efficiency ablation (rungs 1, 2, 4) complete"
echo "============================================================"
echo ""
printf "  %-28s  %-12s  %-12s  %-12s\n" "rung" "checkpoint" "subset" "test NME"
printf "  %-28s  %-12s  %-12s  %-12s\n" "----------------------------" "------------" "------------" "------------"
for ENTRY in "${RUNS[@]}"; do
    IFS=":" read -r RUN_NAME _ HAS_EMA <<< "$ENTRY"
    CKPT_TAGS=(best final)
    [ "$HAS_EMA" = "1" ] && CKPT_TAGS+=(ema)
    for CKPT_TAG in "${CKPT_TAGS[@]}"; do
        for SUBSET in "${SUBSETS[@]}"; do
            NME="${RESULTS[${RUN_NAME}_${CKPT_TAG}_${SUBSET}]:-N/A}"
            printf "  %-28s  %-12s  %-12s  %-12s\n" "$RUN_NAME" "$CKPT_TAG" "$SUBSET" "${NME}%"
        done
    done
done

echo ""
echo "For reference, rung 3 (+FPN+UDP, already run against face300w_deconv_v2_fpn_udp.yaml):"
echo "  best/final common:      5.22% / 5.25%"
echo "  best/final challenging:  7.75% / 7.85%"
echo "  best/final full:        5.72% / 5.76%"
echo ""
echo "HRNetV2-W18 baseline (common/challenging/full): 2.91 / 5.11 / 3.34"
echo ""
echo "Read this ladder side-by-side with BPD (17.48->15.18->14.49->13.21->12.09,"
echo "each rung a real gain) and HC18 (16.54->16.52, ~0 gain past deconv_v2) to"
echo "see whether 300W's rungs move like BPD's (small data, gains persist) or"
echo "like HC18's (big data, gains vanish)."
echo ""
echo "W&B: https://wandb.ai/ucabnx1-ucl/eomt-landmark"
