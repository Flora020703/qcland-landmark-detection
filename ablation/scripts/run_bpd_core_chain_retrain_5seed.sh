#!/usr/bin/env bash
# =============================================================================
# run_bpd_core_chain_retrain_5seed.sh — retrains the 3 BPD core-architecture
# rungs whose checkpoints/per-image predictions are confirmed lost (original
# einsum head, DeconvHeadV2, +FPN -- see docs/supervisor_meeting_report_
# 2026-08-08.md section 0.6.1), so all 4 core rungs (these 3 plus the
# already-archived +FPN+UDP) can be scored under the SAME permutation-
# invariant evaluator as the official EoMT/HRNet comparison table, replacing
# the old fixed-channel numbers entirely -- no mixing of the two metrics in
# one table (see that doc's Decision 5 for why: BPD is a near-vertical task,
# uniquely vulnerable to a metric-definition confound masquerading as an
# architecture improvement).
#
# 2026-08-09, per explicit user decision: RETRAIN, do not downgrade to a
# text-only description, even if this takes longer than the half-day
# checkpoint search originally proposed.
#
# This is a RETROSPECTIVE CONTROLLED RECONSTRUCTION, not a re-run of the
# original ablation sequence -- state this explicitly in the thesis. Unlike
# the original Run 18/21 -> deconv-v2-ablation -> fpn-ablation sequence
# (which did NOT set an explicit loader_seed, a since-identified confound),
# every one of these 15 new runs explicitly sets BOTH seed_everything AND
# data.init_args.loader_seed to the same seed, eliminating that confound
# retroactively for all 3 rungs at once.
#
# Rungs (each x5 seeds = 15 new training runs):
#   einsum   -- configs/landmark/bpd_einsum_reconstructed.yaml (2026-08-09
#               reconstruction -- the original config file was never
#               committed with correct values, see that file's own header)
#   deconvv2 -- configs/landmark/bpd_deconv_v2.yaml (verified unmodified
#               since commit d5a15b3, used as-is)
#   fpn      -- configs/landmark/bpd_deconv_v2_fpn.yaml (verified unmodified
#               since commit 5b10789, used as-is)
#
# All 3 rungs use pixel_center_align=False (unset, defaults False in
# datasets/landmark_dataset.py) -- this predates the UDP fix. Score with
# landmark_ordering_analysis/rescore_landmark_conventions.py's
# load_eomt_per_image(..., pixel_center_align=False), NOT the module
# default (True), or every recovered coordinate will be silently wrong by
# a constant offset (see that function's own 2026-08-09 docstring finding).
#
# Only the FINAL/last checkpoint is dumped with raw per-image coordinates
# (test_nme_dump_path) -- this project's established "final checkpoint is
# primary" convention for a small UCL Test set. The val-best checkpoint is
# still trained/saved/tested for its own aggregate NME (diagnostic only,
# matching the historical deconv_v2/fpn ablation scripts' own behaviour),
# but its raw coordinates are NOT dumped, since only the final checkpoint's
# numbers feed the official permutation-invariant table.
#
# Prerequisite: data must already be on the server --
#   /root/autodl-tmp/images/UCL/Head/
#   /root/autodl-tmp/annotations/UCL/Head_Train.csv
#   /root/autodl-tmp/annotations/UCL/Head_Test.csv
#
# Usage (on AutoDL server):
#   cd /root/eomt
#   nohup bash ablation/scripts/run_bpd_core_chain_retrain_5seed.sh \
#       > bpd_core_chain_retrain_5seed.log 2>&1 &
# =============================================================================

set -euo pipefail

SEEDS=(42 0 123 2024 3407)
RUNS=(
    "einsum:configs/landmark/bpd_einsum_reconstructed.yaml"
    "deconvv2:configs/landmark/bpd_deconv_v2.yaml"
    "fpn:configs/landmark/bpd_deconv_v2_fpn.yaml"
)

RESULTS_TSV="bpd_core_chain_retrain_5seed_results.tsv"
DONE_MARKER="checkpoints/.bpd_core_chain_retrain_5seed_completed.txt"
BACKUP_ROOT="/root/autodl-tmp/saved_checkpoints/bpd_core_chain_retrain_5seed"

if [ ! -d /root/autodl-tmp/images/UCL/Head ] \
   || [ ! -f /root/autodl-tmp/annotations/UCL/Head_Train.csv ] \
   || [ ! -f /root/autodl-tmp/annotations/UCL/Head_Test.csv ]; then
    echo "[ERROR] Head images or Train/Test annotations not found on server -- transfer images/UCL/Head/ and annotations/UCL/Head_*.csv first"
    exit 1
fi

mkdir -p checkpoints "$(dirname "$DONE_MARKER")"
touch "$DONE_MARKER"
mkdir -p "$BACKUP_ROOT"

# 15 runs, each with best+final checkpoints -- same disk-safety floor as
# run_bpd_resolution_screening_5seed.sh (a safety floor, not a usage prediction).
AVAILABLE_KB=$(df -Pk "$BACKUP_ROOT" | awk 'NR==2 {print $4}')
MIN_AVAILABLE_KB=$((10 * 1024 * 1024))
if [ "$AVAILABLE_KB" -lt "$MIN_AVAILABLE_KB" ]; then
    echo "[ERROR] Less than 10 GiB free on checkpoint disk: $BACKUP_ROOT"
    df -h "$BACKUP_ROOT"
    exit 1
fi
echo "[OK] checkpoint disk preflight"
df -h "$BACKUP_ROOT"

if [ ! -f "$RESULTS_TSV" ]; then
    printf "rung\tseed\tckpt_tag\ttest_nme\n" > "$RESULTS_TSV"
fi

for ENTRY in "${RUNS[@]}"; do
    RUNG="${ENTRY%%:*}"
    BASE_CONFIG="${ENTRY#*:}"

    if [ ! -f "${BASE_CONFIG}" ]; then
        echo "[ERROR] Base config not found: ${BASE_CONFIG}"
        exit 1
    fi
    python3 -c "import yaml; yaml.safe_load(open('${BASE_CONFIG}'))" \
        && echo "[OK] Base config valid: ${BASE_CONFIG}" \
        || { echo "[ERROR] Invalid YAML -- aborting"; exit 1; }

    for SEED in "${SEEDS[@]}"; do

        RUN_KEY="${RUNG}_${SEED}"
        if grep -qxF "${RUN_KEY}" "$DONE_MARKER" 2>/dev/null; then
            echo ""
            echo "--- SKIP (already completed): rung=${RUNG} seed=${SEED} ---"
            continue
        fi

        RUN_NAME="bpd-core-${RUNG}-seed${SEED}"
        RUN_DIR="${BACKUP_ROOT}/bpd_${RUNG}/seed${SEED}"
        TMP_CFG="/tmp/bpd_core_${RUNG}_seed${SEED}.yaml"

        echo ""
        echo "============================================================"
        echo "  START: ${RUN_NAME}  (rung=${RUNG}, seed=${SEED})"
        echo "============================================================"

        # --- 1. Generate per-run YAML (seed + loader_seed + W&B name + checkpoint path) ---
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
# Explicit loader_seed (2026-08-09 fix for the historical loader-seed
# confound -- the original einsum/deconv_v2/fpn ablations never set this).
cfg['data']['init_args']['loader_seed'] = seed
cfg['trainer']['logger']['init_args']['name'] = run_name

ckpt_cb = cfg['trainer']['callbacks'][0]['init_args']
ckpt_cb['dirpath']  = run_dir
ckpt_cb['filename'] = f'seed{seed}_best'

with open(out_cfg, 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
" "$SEED" "$BASE_CONFIG" "$TMP_CFG" "$RUN_NAME" "$RUN_DIR"

        # --- 2. Verify critical hyperparams (rung-specific + held-constant set) ---
        echo "--- Verifying config for rung=${RUNG} seed=${SEED} ---"
        python3 - "$TMP_CFG" "$SEED" "$RUNG" <<'PYEOF'
import yaml, sys

cfg_path = sys.argv[1]
seed     = int(sys.argv[2])
rung     = sys.argv[3]

with open(cfg_path) as f:
    cfg = yaml.safe_load(f)

m    = cfg['model']['init_args']
n    = m['network']['init_args']
data = cfg['data']['init_args']

expected_head = {"einsum": "einsum", "deconvv2": "deconv_v2", "fpn": "deconv_v2"}[rung]
expected_fpn  = rung == "fpn"

checks = [
    ('seed_everything',      cfg['seed_everything'],         seed),
    ('loader_seed',          data.get('loader_seed'),        seed),
    ('loss_type',            m['loss_type'],                 'hybrid'),
    ('lambda_coord',         m['lambda_coord'],               0.1),
    ('temperature',          m['temperature'],                10.0),
    ('alpha',                m['alpha'],                      5.0),
    ('heatmap_head',         n['heatmap_head'],               expected_head),
    ('use_fpn',              n.get('use_fpn', False),         expected_fpn),
    ('pixel_center_align',   data.get('pixel_center_align', False), False),
    ('rotate_augment',       data.get('rotate_augment', False), False),
    ('scale_augment',        data.get('scale_augment', False), False),
    ('use_refinement_head',  n['use_refinement_head'],        False),
    ('freeze_backbone',      n['freeze_backbone'],            False),
    ('num_blocks',           n['num_blocks'],                 3),
    ('masked_attn_enabled',  n['masked_attn_enabled'],        True),
    ('backbone_name', n.get('encoder', {}).get('init_args', {}).get('backbone_name'),
                        'vit_small_patch14_reg4_dinov2'),
    ('heatmap_size (net)',   n['heatmap_size'],               [64, 64]),
    ('heatmap_size (data)',  data['heatmap_size'],            [64, 64]),
    ('sigma',                data['sigma'],                   4.0),
    ('val_split_seed',       data['val_split_seed'],          42),
    ('task',                 data.get('task'),                'bpd'),
]
if rung == "fpn":
    checks.append(('fpn_layers', n.get('fpn_layers'), [4, 8, 12]))

all_ok = True
for key, got, expected in checks:
    ok = got == expected
    print(f'  {"[OK]  " if ok else "[FAIL]"} {key}: {got!r}' + ('' if ok else f'  expected={expected!r}'))
    if not ok:
        all_ok = False

if not all_ok:
    print('\n[ERROR] Config verification failed -- aborting')
    sys.exit(1)
print(f'\n[OK] All checks passed for rung={rung} seed={seed}')
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
            mv "${LAST_SRC}" "${FINAL_CKPT}"
            echo "[OK] final ckpt saved (last.ckpt renamed, no duplicate copy)"
        else
            echo "[ERROR] last.ckpt not found -- rung=${RUNG} seed=${SEED} incomplete, aborting (NOT marking DONE)"
            exit 1
        fi

        cp "${TMP_CFG}" "${RUN_DIR}/seed${SEED}_config.yaml"

        # --- 5. Dump (index -> filename) mapping for this run's test split ---
        python3 scripts/dump_test_image_order.py \
            --config "${TMP_CFG}" \
            --out "${RUN_DIR}/test_image_order.csv"

        # --- 6. Test val-best (diagnostic only, NOT dumped) ---
        if [ -f "${BEST_CKPT}" ]; then
            echo ""
            echo "--- Test (val-best, diagnostic only): ${BEST_CKPT} ---"
            LOG_BEST="${RUN_DIR}/seed${SEED}_best_test_log.txt"
            python3 main_landmark.py test \
                --config "${TMP_CFG}" \
                --ckpt_path "${BEST_CKPT}" 2>&1 \
                | tee "${LOG_BEST}" \
                | grep -E "test_nme|Test NME"
            BEST_NME=$(grep "Test NME:" "${LOG_BEST}" | grep -oE "[0-9]+\.[0-9]+" | tail -1 || echo "")
            [ -n "${BEST_NME}" ] && printf "%s\t%s\tbest\t%s\n" "$RUNG" "$SEED" "$BEST_NME" >> "$RESULTS_TSV"
        else
            echo "[WARN] Best checkpoint not found -- skipping diagnostic test"
        fi

        # --- 7. Test final checkpoint WITH raw-coordinate dump (this is the
        #        row that feeds the official permutation-invariant table) ---
        echo ""
        echo "--- Test (final, WITH per-image coordinate dump): ${FINAL_CKPT} ---"
        LOG_FINAL="${RUN_DIR}/seed${SEED}_final_test_log.txt"
        DUMP_FINAL="${RUN_DIR}/final_fixedchannel_per_image.csv"
        python3 main_landmark.py test \
            --config "${TMP_CFG}" \
            --ckpt_path "${FINAL_CKPT}" \
            --model.init_args.test_nme_dump_path "${DUMP_FINAL}" 2>&1 \
            | tee "${LOG_FINAL}" \
            | grep -E "test_nme|Test NME|Per-sample NME dumped"
        FINAL_NME=$(grep "Test NME:" "${LOG_FINAL}" | grep -oE "[0-9]+\.[0-9]+" | tail -1 || echo "")
        if [ -z "${FINAL_NME}" ] || [ ! -f "${DUMP_FINAL}" ]; then
            echo "[ERROR] Failed to parse Test NME or missing dump for rung=${RUNG} seed=${SEED} -- incomplete, aborting (NOT marking DONE)"
            exit 1
        fi
        printf "%s\t%s\tfinal\t%s\n" "$RUNG" "$SEED" "$FINAL_NME" >> "$RESULTS_TSV"

        echo "${RUN_KEY}" >> "$DONE_MARKER"
        echo ""
        echo "--- DONE: ${RUN_NAME} ---"

    done
done

echo ""
echo "============================================================"
echo "  BPD core-chain retrain (einsum/deconvv2/fpn), 5-seed, complete"
echo "============================================================"
echo ""
echo "Per-run native (fixed-channel, model's own reported) NME in ${RESULTS_TSV}."
echo "This is NOT the number to report -- re-score every rung with the SAME"
echo "permutation-invariant evaluator as the official comparison:"
echo ""
echo "  python3 landmark_ordering_analysis/aggregate_bpd_core_chain.py \\"
echo "      --eomt-root ${BACKUP_ROOT} \\"
echo "      --images-root /root/autodl-tmp/images/UCL \\"
echo "      --output-root landmark_ordering_analysis/results/bpd_core_chain"
echo ""
echo "(script to be added -- reuses load_eomt_per_image(pixel_center_align=False)"
echo "+ rescore_cell()/summarize_and_write() from rescore_landmark_conventions.py,"
echo "the exact same tested pipeline as the main EoMT/HRNet comparison, so no new"
echo "unreviewed scoring logic is introduced for this table)."
echo ""
echo "Checkpoints written to: ${BACKUP_ROOT}"
echo "W&B: https://wandb.ai/ucabnx1-ucl/eomt-landmark"
