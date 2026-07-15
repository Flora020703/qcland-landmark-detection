#!/usr/bin/env bash
# =============================================================================
# run_300w_data_efficiency_5seed_ablation.sh — 300W data-efficiency ablation,
# full 5-seed treatment across all 4 rungs (baseline / +FPN / +FPN+UDP /
# +FPN+UDP+EMA), matching BPD's own 5-seed ladder (deconv_v2 -> +FPN ->
# +FPN+UDP -> +FPN+UDP+EMA, Exp 4/7/8a) for a symmetric, reviewer-proof
# cross-dataset comparison.
#
# IMPORTANT: seed=42 already ran for all 4 rungs (single-seed pass, see
# run_300w_data_efficiency_ablation.sh) - this script only trains the 4
# REMAINING seeds (0, 123, 2024, 3407). The seed=42 numbers are hardcoded
# into RESULTS_TSV below (from the already-completed run) so the final
# mean/std summary is a true 5-seed statistic without retraining seed=42.
#
# Before running this script, reorganize the existing seed=42 checkpoints
# into the seed-subdirectory structure this script expects (see the
# accompanying instructions - each rung needs its seed42 best/final(/ema)
# checkpoint copied from checkpoints/<rung>/ into
# checkpoints/<rung>-ablation/seed42/seed42_{best,final,ema}.ckpt), so that
# a later 5-seed inference ensemble (ablation/ensemble_test.py) can glob
# checkpoints/<rung>-ablation/seed*/seed*_best.ckpt uniformly across all 5
# seeds including 42.
#
# Purpose (unchanged from the single-seed version): this is NOT primarily
# about closing the gap to HRNet - it tests whether the cross-dataset
# finding from BPD/HC18 ("FPN/UDP/EMA are anti-overfitting tools whose
# benefit shrinks as training-set size grows") extends to 300W (3148 train
# images, natural RGB faces). The single-seed pass already showed a
# striking, worth-confirming pattern: FPN's gain persists (~0.3pt across
# all subsets), UDP's marginal gain is exactly 0.00pt (identical to FPN
# alone across all 6 best/final x common/challenging/full measurements),
# and +EMA looks worse (~0.25pt) than +FPN+UDP alone - the EMA result in
# particular needs 5-seed confirmation since EMACallback shouldn't touch
# the raw model's own gradient trajectory at all, so a real, reproducible
# regression would be a genuinely new finding, not just "gain vanishes".
#
# Usage (on AutoDL server, after reorganizing seed=42 checkpoints):
#   cd /root/eomt
#   nohup bash ablation/scripts/run_300w_data_efficiency_5seed_ablation.sh > face300w_5seed_ablation.log 2>&1 &
# =============================================================================

set -euo pipefail

SUBSETS=(common challenging full)
SEEDS=(0 123 2024 3407)   # seed 42 already done - see header

# name : config path : has_ema (1/0) : expect_use_fpn (1/0) : expect_pixel_center_align (1/0)
RUNS=(
    "face300w-deconv-v2:configs/landmark/face300w_deconv_v2.yaml:0:0:0"
    "face300w-deconv-v2-fpn:configs/landmark/face300w_deconv_v2_fpn.yaml:0:1:0"
    "face300w-deconv-v2-fpn-udp:configs/landmark/face300w_deconv_v2_fpn_udp.yaml:0:1:1"
    "face300w-deconv-v2-fpn-udp-ema:configs/landmark/face300w_deconv_v2_fpn_udp_ema.yaml:1:1:1"
)

RESULTS_TSV="face300w_5seed_results.tsv"

# ---------------------------------------------------------------------------
# Seed the results file with the already-completed seed=42 numbers (single-
# seed pass) so the final aggregation is a genuine 5-seed statistic.
# ---------------------------------------------------------------------------
cat > "$RESULTS_TSV" <<'EOF'
rung	seed	ckpt_tag	subset	nme
face300w-deconv-v2	42	best	common	5.54
face300w-deconv-v2	42	best	challenging	8.02
face300w-deconv-v2	42	best	full	6.02
face300w-deconv-v2	42	final	common	5.56
face300w-deconv-v2	42	final	challenging	8.07
face300w-deconv-v2	42	final	full	6.05
face300w-deconv-v2-fpn	42	best	common	5.22
face300w-deconv-v2-fpn	42	best	challenging	7.75
face300w-deconv-v2-fpn	42	best	full	5.72
face300w-deconv-v2-fpn	42	final	common	5.25
face300w-deconv-v2-fpn	42	final	challenging	7.85
face300w-deconv-v2-fpn	42	final	full	5.76
face300w-deconv-v2-fpn-udp	42	best	common	5.22
face300w-deconv-v2-fpn-udp	42	best	challenging	7.75
face300w-deconv-v2-fpn-udp	42	best	full	5.72
face300w-deconv-v2-fpn-udp	42	final	common	5.25
face300w-deconv-v2-fpn-udp	42	final	challenging	7.85
face300w-deconv-v2-fpn-udp	42	final	full	5.76
face300w-deconv-v2-fpn-udp-ema	42	best	common	5.47
face300w-deconv-v2-fpn-udp-ema	42	best	challenging	8.07
face300w-deconv-v2-fpn-udp-ema	42	best	full	5.98
face300w-deconv-v2-fpn-udp-ema	42	final	common	5.49
face300w-deconv-v2-fpn-udp-ema	42	final	challenging	8.09
face300w-deconv-v2-fpn-udp-ema	42	final	full	6.00
face300w-deconv-v2-fpn-udp-ema	42	ema	common	5.48
face300w-deconv-v2-fpn-udp-ema	42	ema	challenging	8.07
face300w-deconv-v2-fpn-udp-ema	42	ema	full	5.99
EOF
echo "[OK] Seeded ${RESULTS_TSV} with the existing seed=42 results (27 rows)"

# ---------------------------------------------------------------------------
# Main loop: 4 rungs x 4 remaining seeds
# ---------------------------------------------------------------------------
for ENTRY in "${RUNS[@]}"; do
    IFS=":" read -r RUN_NAME BASE_CONFIG HAS_EMA EXPECT_FPN EXPECT_UDP <<< "$ENTRY"

    if [ ! -f "${BASE_CONFIG}" ]; then
        echo "[ERROR] Config not found: ${BASE_CONFIG}"
        exit 1
    fi
    python3 -c "import yaml; yaml.safe_load(open('${BASE_CONFIG}'))" \
        && echo "[OK] Config valid: ${BASE_CONFIG}" \
        || { echo "[ERROR] Invalid YAML — aborting"; exit 1; }

    for SEED in "${SEEDS[@]}"; do
        RUN_LABEL="${RUN_NAME}-ablation-seed${SEED}"
        RUN_DIR="checkpoints/${RUN_NAME}-ablation/seed${SEED}"
        TMP_CFG="/tmp/${RUN_NAME}_seed${SEED}.yaml"

        echo ""
        echo "============================================================"
        echo "  START: ${RUN_LABEL}"
        echo "============================================================"

        # --- 1. Generate per-run YAML (seed + loader_seed + W&B name + ckpt path) ---
        python3 -c "
import yaml, sys

seed       = int(sys.argv[1])
base_cfg   = sys.argv[2]
out_cfg    = sys.argv[3]
run_name   = sys.argv[4]
run_dir    = sys.argv[5]
run_label  = sys.argv[6]

with open(base_cfg) as f:
    cfg = yaml.safe_load(f)

cfg['seed_everything'] = seed
cfg['trainer']['logger']['init_args']['name'] = run_label
cfg['data']['init_args']['loader_seed'] = seed

ckpt_cb = cfg['trainer']['callbacks'][0]['init_args']
ckpt_cb['dirpath']  = run_dir
ckpt_cb['filename'] = f'seed{seed}_best'

with open(out_cfg, 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
" "$SEED" "$BASE_CONFIG" "$TMP_CFG" "$RUN_NAME" "$RUN_DIR" "$RUN_LABEL"

        # --- 2. Verify critical hyperparams ---
        echo "--- Verifying config for ${RUN_LABEL} ---"
        python3 - "$TMP_CFG" "$SEED" "$HAS_EMA" "$EXPECT_FPN" "$EXPECT_UDP" <<'PYEOF'
import yaml, sys

cfg_path    = sys.argv[1]
seed        = int(sys.argv[2])
has_ema     = sys.argv[3] == "1"
expect_fpn  = sys.argv[4] == "1"
expect_udp  = sys.argv[5] == "1"

with open(cfg_path) as f:
    cfg = yaml.safe_load(f)

m = cfg['model']['init_args']; n = m['network']['init_args']; data = cfg['data']['init_args']
callbacks = cfg['trainer']['callbacks']
ema_cb = next((c for c in callbacks if c['class_path'] == 'training.ema.EMACallback'), None)

checks = [
    ('seed_everything',      cfg['seed_everything'],          seed),
    ('loader_seed',          data.get('loader_seed'),         seed),
    ('num_landmarks',        m['num_landmarks'],               68),
    ('nme_norm_pair',        list(m.get('nme_norm_pair', [])), [36, 45]),
    ('loss_type',            m['loss_type'],                  'hybrid'),
    ('num_q',                n['num_q'],                       68),
    ('num_blocks',           n['num_blocks'],                  6),
    ('heatmap_head',         n['heatmap_head'],               'deconv_v2'),
    ('use_fpn',              n.get('use_fpn', False),         expect_fpn),
    ('pixel_center_align',   data.get('pixel_center_align', False), expect_udp),
    ('img_size',             data['img_size'],                [256, 256]),
    ('heatmap_size',         data['heatmap_size'],            [64, 64]),
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
print(f'\n[OK] All checks passed for {sys.argv[0]}')
PYEOF

        mkdir -p "${RUN_DIR}"

        # --- 3. Train ---
        echo ""
        echo "--- Training: ${RUN_LABEL} ---"
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

        CKPT_TAGS=(best final)

        # --- 5. Materialize EMA checkpoint, if this rung has one ---
        if [ "$HAS_EMA" = "1" ]; then
            EMA_CKPT="${RUN_DIR}/seed${SEED}_ema.ckpt"
            if [ -f "${FINAL_CKPT}" ]; then
                echo ""
                echo "--- Materializing EMA checkpoint: ${EMA_CKPT} ---"
                python3 ablation/apply_ema.py "${FINAL_CKPT}" "${EMA_CKPT}"
                CKPT_TAGS+=(ema)
            else
                echo "[WARN] Final checkpoint not found — skipping EMA materialization"
            fi
        fi

        # --- 6. Test: common / challenging / full, for each available checkpoint ---
        for CKPT_TAG in "${CKPT_TAGS[@]}"; do
            CKPT_PATH="${RUN_DIR}/seed${SEED}_${CKPT_TAG}.ckpt"
            if [ ! -f "${CKPT_PATH}" ]; then
                echo "[WARN] ${CKPT_PATH} not found — skipping all subset tests for ${CKPT_TAG}"
                continue
            fi

            for SUBSET in "${SUBSETS[@]}"; do
                TMP_TEST_CFG="/tmp/${RUN_NAME}_seed${SEED}_${CKPT_TAG}_${SUBSET}.yaml"
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
                echo "--- Test (${RUN_LABEL}, ${CKPT_TAG} checkpoint, ${SUBSET} subset) ---"
                LOG_FILE="${RUN_DIR}/seed${SEED}_${CKPT_TAG}_${SUBSET}_test_log.txt"
                python3 main_landmark.py test \
                    --config "${TMP_TEST_CFG}" \
                    --ckpt_path "${CKPT_PATH}" 2>&1 \
                    | tee "${LOG_FILE}" \
                    | grep -E "test_nme|Test NME"

                NME=$(grep "Test NME:" "${LOG_FILE}" | grep -oE "[0-9]+\.[0-9]+" | tail -1 || echo "")
                if [ -n "$NME" ]; then
                    echo -e "${RUN_NAME}\t${SEED}\t${CKPT_TAG}\t${SUBSET}\t${NME}" >> "$RESULTS_TSV"
                fi
            done
        done

        echo ""
        echo "--- DONE: ${RUN_LABEL} ---"
    done
done

# ---------------------------------------------------------------------------
# Summary: mean +/- std across all 5 seeds (42 + the 4 just run), per rung /
# checkpoint type / subset
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  300W 4-rung data-efficiency ablation — full 5-seed summary"
echo "============================================================"
python3 - "$RESULTS_TSV" <<'PYEOF'
import sys
from collections import defaultdict
import statistics

path = sys.argv[1]
groups = defaultdict(list)

with open(path) as f:
    next(f)  # header
    for line in f:
        rung, seed, ckpt_tag, subset, nme = line.rstrip("\n").split("\t")
        groups[(rung, ckpt_tag, subset)].append(float(nme))

rung_order = [
    "face300w-deconv-v2",
    "face300w-deconv-v2-fpn",
    "face300w-deconv-v2-fpn-udp",
    "face300w-deconv-v2-fpn-udp-ema",
]
ckpt_order = ["best", "final", "ema"]
subset_order = ["common", "challenging", "full"]

print(f"\n  {'rung':<32}{'ckpt':<8}{'subset':<13}{'n':<4}{'mean':<8}{'std':<8}")
print(f"  {'-'*32}{'-'*8}{'-'*13}{'-'*4}{'-'*8}{'-'*8}")
for rung in rung_order:
    for ckpt_tag in ckpt_order:
        for subset in subset_order:
            key = (rung, ckpt_tag, subset)
            if key not in groups:
                continue
            vals = groups[key]
            mean = statistics.mean(vals)
            std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
            print(f"  {rung:<32}{ckpt_tag:<8}{subset:<13}{len(vals):<4}{mean:<8.2f}{std:<8.2f}")
PYEOF

echo ""
echo "HRNetV2-W18 baseline (common/challenging/full): 2.91 / 5.11 / 3.34"
echo ""
echo "Raw per-seed numbers: ${RESULTS_TSV}"
echo "W&B: https://wandb.ai/ucabnx1-ucl/eomt-landmark"
