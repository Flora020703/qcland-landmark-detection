#!/usr/bin/env bash
# =============================================================================
# run_bpd_resolution_screening_5seed.sh — extends the BPD resolution
# screening (256x256 vs 512x512, DINOv2 matched-protocol reference
# configuration) from the single seed=2024 canary to the project's
# standard 5 seeds (42, 0, 123, 2024, 3407), as a FOLLOW-UP ROBUSTNESS
# ANALYSIS triggered by the seed=2024 result showing a checkpoint- and
# metric-dependent direction (see project memory, 2026-07-24) -- NOT a
# pre-registered experiment, but also not an improper post-hoc fishing
# expedition, AS LONG AS: all 5 seeds are reported regardless of outcome,
# no seeds are added/dropped based on how partial results look, and no
# seed/checkpoint is chosen based on the test result. State the reason
# for expansion explicitly in the thesis (single-seed sensitivity was
# metric/checkpoint-dependent) -- do not present this as if 5 seeds were
# always the plan.
#
# seed=2024 is ALREADY DONE (both resolutions, both checkpoint tags, both
# metrics -- from the original single-seed screening run plus a follow-up
# fixed-channel re-test). This script pre-seeds the results TSV and
# DONE_MARKER with those 4 known rows and only trains the remaining
# 4 seeds x 2 resolutions = 8 new runs. Does NOT touch or reuse the
# original single-seed run's own DONE_MARKER / results TSV / checkpoint
# directory (bpd-resolution-screening, singular -- this script uses
# bpd-resolution-screening-5seed).
#
# Both metrics are recorded for EVERY checkpoint (two separate `test`
# invocations per checkpoint, same per-run config both times):
#   - swap-min (endpoint_order_invariant_nme=true in the config -- matches
#     Di Vece et al.'s published BiometryNet formula). This is also what
#     validation/early-stopping/checkpoint-selection use during training,
#     consistent with seed=2024's original run and the locked fetal-NME-
#     metric decision (see project memory) -- do not disable it for
#     training, only override it for the second, fixed-channel test pass.
#   - fixed-channel (this project's historical metric, re-tested post-hoc
#     via `--model.init_args.endpoint_order_invariant_nme false` on the
#     SAME per-run generated config -- per Rule 32, never the shared base
#     config, which would silently use the wrong img_size for one of the
#     two resolutions).
#
# Reporting convention: "final" checkpoint is primary (this project's
# small-UCL-val-set convention), "best" is diagnostic only.
#
# STATISTICAL FRAMING -- do not deviate when writing this up:
#   - Report per-seed PAIRED differences (delta_s = NME_256,s - NME_512,s),
#     not just two independent per-resolution mean+-SD groups -- same
#     seed/split/init/everything-but-img_size makes the paired structure
#     more informative than an unpaired comparison.
#   - The defensible conclusion, whatever the numbers show, is "no
#     consistent resolution advantage was detected across the five matched
#     seeds" -- NEVER "resolution has no effect" (that would need a
#     pre-specified equivalence margin and an equivalence test, which this
#     screening was not designed or powered for; n=5 is likely underpowered
#     for that anyway).
#
# Prerequisite: data must be on the server first --
#   /root/autodl-tmp/images/UCL/Head/
#   /root/autodl-tmp/annotations/UCL/Head_Train.csv
#   /root/autodl-tmp/annotations/UCL/Head_Test.csv
#
# Usage (on AutoDL server):
#   cd /root/eomt
#   nohup bash ablation/scripts/run_bpd_resolution_screening_5seed.sh > bpd_resolution_screening_5seed.log 2>&1 &
# =============================================================================

set -euo pipefail

BASE_CONFIG="configs/landmark/bpd_resolution_screening.yaml"
SEEDS=(42 0 123 3407)          # 2024 already done, see header comment
RESOLUTIONS=(256 512)

RUN_GROUP="bpd-resolution-screening-5seed"
RESULTS_TSV="bpd_resolution_screening_5seed_results.tsv"
DONE_MARKER="checkpoints/.bpd_resolution_screening_5seed_completed.txt"
BACKUP_ROOT="/root/autodl-tmp/saved_checkpoints/bpd_resolution_screening_5seed"

if [ ! -d /root/autodl-tmp/images/UCL/Head ] \
   || [ ! -f /root/autodl-tmp/annotations/UCL/Head_Train.csv ] \
   || [ ! -f /root/autodl-tmp/annotations/UCL/Head_Test.csv ]; then
    echo "[ERROR] Head images or Train/Test annotations not found on server -- transfer images/UCL/Head/ and annotations/UCL/Head_*.csv first"
    exit 1
fi

mkdir -p checkpoints "$(dirname "$DONE_MARKER")"
touch "$DONE_MARKER"

# --- Pre-seed seed=2024's already-known results (from the original
#     single-seed screening run + its follow-up fixed-channel re-test,
#     2026-07-24) -- only if the TSV doesn't already exist, per Rule 15. ---
if [ ! -f "$RESULTS_TSV" ]; then
    echo -e "seed\tresolution\tckpt_tag\tswap_min_nme\tfixed_channel_nme" > "$RESULTS_TSV"
    echo -e "2024\t256\tbest\t7.44\t11.74"  >> "$RESULTS_TSV"
    echo -e "2024\t256\tfinal\t5.52\t11.68" >> "$RESULTS_TSV"
    echo -e "2024\t512\tbest\t6.01\t8.30"   >> "$RESULTS_TSV"
    echo -e "2024\t512\tfinal\t6.88\t11.11" >> "$RESULTS_TSV"
    echo "[OK] pre-seeded results TSV with seed=2024's known values"
fi
grep -qxF "2024_256" "$DONE_MARKER" || echo "2024_256" >> "$DONE_MARKER"
grep -qxF "2024_512" "$DONE_MARKER" || echo "2024_512" >> "$DONE_MARKER"

if [ ! -f "${BASE_CONFIG}" ]; then
    echo "[ERROR] Base config not found: ${BASE_CONFIG}"
    exit 1
fi
python3 -c "import yaml; yaml.safe_load(open('${BASE_CONFIG}'))" \
    && echo "[OK] Base config valid: ${BASE_CONFIG}" \
    || { echo "[ERROR] Invalid YAML -- aborting"; exit 1; }

mkdir -p "checkpoints/${RUN_GROUP}"

for SEED in "${SEEDS[@]}"; do
for RES in "${RESOLUTIONS[@]}"; do

    PAIR_KEY="${SEED}_${RES}"
    if grep -qxF "${PAIR_KEY}" "$DONE_MARKER" 2>/dev/null; then
        echo ""
        echo "--- SKIP (already completed): seed=${SEED} resolution=${RES} ---"
        continue
    fi

    RUN_NAME="${RUN_GROUP}-res${RES}-seed${SEED}"
    RUN_DIR="checkpoints/${RUN_GROUP}/res${RES}/seed${SEED}"
    TMP_CFG="/tmp/${RUN_GROUP}_res${RES}_seed${SEED}.yaml"

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
    echo "--- Verifying config for seed=${SEED} resolution=${RES} ---"
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
    ('swap-min fetal NME',   m.get('endpoint_order_invariant_nme', False), True),
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
print(f'\n[OK] All checks passed for seed={seed} resolution={res}')
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

    # --- 4b. Dump (index -> filename) mapping for this run's test split ---
    python3 scripts/dump_test_image_order.py \
        --config "${TMP_CFG}" \
        --out "${RUN_DIR}/test_image_order.csv"

    # --- 5. Test best/final checkpoints under BOTH metrics ---
    for CKPT_TAG in best final; do
        CKPT_PATH="${RUN_DIR}/${RUN_NAME}_${CKPT_TAG}.ckpt"
        if [ ! -f "${CKPT_PATH}" ]; then
            echo "[ERROR] ${CKPT_PATH} not found -- seed=${SEED} resolution=${RES} incomplete, aborting (NOT marking DONE)"
            exit 1
        fi

        # -- swap-min (config default: endpoint_order_invariant_nme=true) --
        echo ""
        echo "--- Test (${CKPT_TAG}, swap-min): ${CKPT_PATH} ---"
        LOG_SWAPMIN="${RUN_DIR}/${RUN_NAME}_${CKPT_TAG}_swapmin_test_log.txt"
        DUMP_SWAPMIN="${RUN_DIR}/${RUN_NAME}_${CKPT_TAG}_swapmin_per_image.csv"
        python3 main_landmark.py test \
            --config "${TMP_CFG}" \
            --ckpt_path "${CKPT_PATH}" \
            --model.init_args.test_nme_dump_path "${DUMP_SWAPMIN}" 2>&1 \
            | tee "${LOG_SWAPMIN}" \
            | grep -E "test_nme|Test NME|Per-sample NME dumped"
        SWAPMIN_NME=$(grep "Test NME:" "${LOG_SWAPMIN}" | grep -oE "[0-9]+\.[0-9]+" | tail -1 || echo "")

        # -- fixed-channel: CLI override on the SAME per-run config (Rule 32
        #    -- never the shared base config, wrong img_size for one rung) --
        echo ""
        echo "--- Test (${CKPT_TAG}, fixed-channel): ${CKPT_PATH} ---"
        LOG_FIXED="${RUN_DIR}/${RUN_NAME}_${CKPT_TAG}_fixedchannel_test_log.txt"
        DUMP_FIXED="${RUN_DIR}/${RUN_NAME}_${CKPT_TAG}_fixedchannel_per_image.csv"
        python3 main_landmark.py test \
            --config "${TMP_CFG}" \
            --ckpt_path "${CKPT_PATH}" \
            --model.init_args.endpoint_order_invariant_nme false \
            --model.init_args.test_nme_dump_path "${DUMP_FIXED}" 2>&1 \
            | tee "${LOG_FIXED}" \
            | grep -E "test_nme|Test NME|Per-sample NME dumped"
        FIXEDCHANNEL_NME=$(grep "Test NME:" "${LOG_FIXED}" | grep -oE "[0-9]+\.[0-9]+" | tail -1 || echo "")

        if [ -z "$SWAPMIN_NME" ] || [ -z "$FIXEDCHANNEL_NME" ]; then
            echo "[ERROR] Failed to parse Test NME (swap-min='${SWAPMIN_NME}' fixed-channel='${FIXEDCHANNEL_NME}') for seed=${SEED} resolution=${RES} ckpt_tag=${CKPT_TAG} -- incomplete, aborting (NOT marking DONE)"
            exit 1
        fi
        echo -e "${SEED}\t${RES}\t${CKPT_TAG}\t${SWAPMIN_NME}\t${FIXEDCHANNEL_NME}" >> "$RESULTS_TSV"
    done

    # MODIFIED: only mark this (seed, resolution) pair DONE now that both
    # checkpoint tags x both metrics are confirmed captured above.
    echo "${PAIR_KEY}" >> "$DONE_MARKER"
    echo ""
    echo "--- DONE: ${RUN_NAME} ---"

done
done

# ---------------------------------------------------------------------------
# Summary: paired per-seed analysis, per checkpoint tag x metric
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  BPD resolution screening, 5-seed paired analysis"
echo "============================================================"
python3 - "$RESULTS_TSV" <<'PYEOF'
import sys, statistics

rows = {}
with open(sys.argv[1]) as f:
    next(f)
    for line in f:
        seed, res, ckpt_tag, swap_min, fixed_channel = line.rstrip("\n").split("\t")
        rows[(seed, res, ckpt_tag)] = {
            "swap_min": float(swap_min),
            "fixed_channel": float(fixed_channel),
        }  # dict keyed by (seed,res,tag) -- de-dupes any accidental repeat row

seeds = sorted({s for (s, r, t) in rows}, key=int)
T_975_DF4 = 2.776  # two-sided 95% CI critical value, df=4 (n=5 paired seeds)

for ckpt_tag in ("best", "final"):
    for metric in ("swap_min", "fixed_channel"):
        vals256, vals512, diffs = [], [], []
        for seed in seeds:
            k256, k512 = (seed, "256", ckpt_tag), (seed, "512", ckpt_tag)
            if k256 not in rows or k512 not in rows:
                continue
            v256, v512 = rows[k256][metric], rows[k512][metric]
            vals256.append(v256)
            vals512.append(v512)
            diffs.append(v256 - v512)  # delta_s = NME_256,s - NME_512,s

        print(f"\n=== {ckpt_tag} / {metric} (n={len(diffs)}/{len(seeds)} seeds) ===")
        if len(diffs) < len(seeds):
            print(f"  [WARN] missing data for {len(seeds) - len(diffs)} seed(s) -- incomplete")
        if not diffs:
            continue

        n = len(diffs)
        mean256, mean512 = statistics.mean(vals256), statistics.mean(vals512)
        std256 = statistics.pstdev(vals256) if n > 1 else 0.0
        std512 = statistics.pstdev(vals512) if n > 1 else 0.0
        mean_diff = statistics.mean(diffs)
        # NOTE: sample std (ddof=1) here, NOT this project's usual pstdev --
        # this is a standard-error calculation for a CI on the MEAN of the
        # paired differences (inference), not a population-dispersion
        # description, so it deliberately differs from the pstdev-based
        # "mean+-SD across seeds" convention used elsewhere in this project.
        sd_diff = statistics.stdev(diffs) if n > 1 else 0.0
        se_diff = sd_diff / (n ** 0.5) if n > 1 else 0.0
        ci_half = T_975_DF4 * se_diff if n == 5 else float("nan")
        n_pos = sum(1 for d in diffs if d > 0)   # favours 512 (smaller NME there)
        n_neg = sum(1 for d in diffs if d < 0)   # favours 256

        print(f"  256: mean={mean256:.2f} std={std256:.2f}")
        print(f"  512: mean={mean512:.2f} std={std512:.2f}")
        print(f"  paired diffs (256-512): {[f'{d:+.2f}' for d in diffs]}")
        print(f"  mean paired diff: {mean_diff:+.2f}  95% CI: [{mean_diff - ci_half:+.2f}, {mean_diff + ci_half:+.2f}]")
        print(f"  direction: {n_pos}/{n} favour 512, {n_neg}/{n} favour 256, {n - n_pos - n_neg}/{n} tie")
PYEOF

echo ""
echo "NOTE: report as 'no consistent resolution advantage detected across"
echo "the five matched seeds' -- NOT 'resolution has no effect' (that needs"
echo "a pre-specified equivalence margin + equivalence test, not run here)."
echo "This 5-seed expansion is a follow-up robustness analysis triggered by"
echo "the seed=2024 screening's checkpoint/metric-dependent direction --"
echo "state this explicitly in the thesis, and report all 5 seeds"
echo "regardless of outcome (do not add/drop seeds based on results)."
echo ""
echo "After reviewing results, move checkpoints off the system disk:"
echo "  mkdir -p ${BACKUP_ROOT} && mv checkpoints/${RUN_GROUP}/res*/seed* ${BACKUP_ROOT}/"
echo ""
echo "W&B: https://wandb.ai/ucabnx1-ucl/eomt-landmark"
