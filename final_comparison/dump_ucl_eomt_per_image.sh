#!/usr/bin/env bash
set -euo pipefail

PY="${PY:-python3}"
ROOT="${ROOT:-/root/eomt}"
OUT="${OUT:-/root/autodl-tmp/ucl_eomt_per_image}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
SEEDS=(42 0 123 2024 3407)

# Checkpoints already contain all learned weights.  Keep test-only export local
# and deterministic instead of performing a redundant Hugging Face HEAD request
# and creating 80 disposable W&B runs.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=disabled

mkdir -p "$OUT/driver_logs"
cd "$ROOT"

if [[ -s "scripts/dump_test_image_order.py" ]]; then
  ORDER_DUMPER="scripts/dump_test_image_order.py"
elif [[ -s "ablation/dump_test_image_order.py" ]]; then
  ORDER_DUMPER="ablation/dump_test_image_order.py"
else
  echo "ERROR: dump_test_image_order.py not found under scripts/ or ablation/" >&2
  exit 1
fi

# key|config|checkpoint directory.  Test-set size is discovered from the
# deterministic image-order dump; it is deliberately not hard-coded.
SPECS=(
  "ofd_dinov2|configs/landmark/ofd_dinov2_fpn_udp_rotate_scale.yaml|/root/eomt/checkpoints/ofd-dinov2-fpn-udp-rotate-scale"
  "ofd_dinov3|configs/landmark/ofd_dinov3_fpn_udp_rotate_scale.yaml|/root/autodl-tmp/saved_checkpoints/ofd_dinov3_fpn_udp_rotate_scale"
  "apad_dinov2|configs/landmark/apad_dinov2_fpn_udp_rotate_scale.yaml|/root/autodl-tmp/saved_checkpoints/apad_fpn_udp_rotate_scale"
  "apad_dinov3|configs/landmark/apad_dinov3_fpn_udp_rotate_scale.yaml|/root/autodl-tmp/saved_checkpoints/apad_dinov3_fpn_udp_rotate_scale"
  "tad_dinov2|configs/landmark/tad_dinov2_fpn_udp_rotate_scale.yaml|/root/autodl-tmp/saved_checkpoints/tad_fpn_udp_rotate_scale"
  "tad_dinov3|configs/landmark/tad_dinov3_fpn_udp_rotate_scale.yaml|/root/autodl-tmp/saved_checkpoints/tad_dinov3_fpn_udp_rotate_scale"
  "fl_dinov2|configs/landmark/fl_dinov2_fpn_udp_rotate_scale.yaml|/root/autodl-tmp/saved_checkpoints/fl_fpn_udp_rotate_scale"
  "fl_dinov3|configs/landmark/fl_dinov3_fpn_udp_rotate_scale.yaml|/root/autodl-tmp/saved_checkpoints/fl_dinov3_fpn_udp_rotate_scale"
)

validate_outputs() {
  local order_csv="$1" fixed_csv="$2" swap_csv="$3"
  "$PY" - "$order_csv" "$fixed_csv" "$swap_csv" <<'PY'
import csv
import sys
from pathlib import Path

order_path, fixed_path, swap_path = map(Path, sys.argv[1:4])

def load(path):
    if not path.is_file():
        raise SystemExit(1)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))

order, fixed, swap = load(order_path), load(fixed_path), load(swap_path)
expected = len(order)
if expected == 0:
    raise SystemExit(1)
if not (len(order) == len(fixed) == len(swap) == expected):
    raise SystemExit(1)
if not ({"img_name", "filename"} & set(order[0])) or "index" not in order[0]:
    raise SystemExit(1)
if "nme" not in fixed[0] or "index" not in fixed[0]:
    raise SystemExit(1)
if "nme" not in swap[0] or "index" not in swap[0]:
    raise SystemExit(1)
oi = {int(row["index"]) for row in order}
fi = {int(row["index"]) for row in fixed}
si = {int(row["index"]) for row in swap}
if len(oi) != expected or oi != fi or oi != si:
    raise SystemExit(1)
PY
}

echo -e "task_backbone\tseed\ttest_n\tfixed_nme_pct\tswap_nme_pct" > "$OUT/ucl_eomt_per_image_results.tsv.tmp"

# Fail before starting inference if any of the 40 required inputs is absent.
for spec in "${SPECS[@]}"; do
  IFS='|' read -r key config ckpt_dir <<< "$spec"
  test -s "$config" || { echo "ERROR: missing config $config" >&2; exit 1; }
  for seed in "${SEEDS[@]}"; do
    ckpt="$ckpt_dir/seed${seed}/seed${seed}_final.ckpt"
    test -s "$ckpt" || { echo "ERROR: missing checkpoint $ckpt" >&2; exit 1; }
  done
done
echo "[PREFLIGHT OK] all eight configs and 40 final checkpoints exist"
df -h /root/autodl-tmp

if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
  echo "[PREFLIGHT COMPLETE] no inference was started"
  exit 0
fi

for spec in "${SPECS[@]}"; do
  IFS='|' read -r key config ckpt_dir <<< "$spec"
  for seed in "${SEEDS[@]}"; do
    run_out="$OUT/$key/seed${seed}"
    mkdir -p "$run_out"
    ckpt="$ckpt_dir/seed${seed}/seed${seed}_final.ckpt"
    order_csv="$run_out/test_image_order.csv"
    fixed_csv="$run_out/final_fixedchannel_per_image.csv"
    swap_csv="$run_out/final_swapmin_per_image.csv"

    if validate_outputs "$order_csv" "$fixed_csv" "$swap_csv" 2>/dev/null; then
      echo "[SKIP verified] $key seed=$seed"
    else
      rm -f -- "$order_csv" "$fixed_csv" "$swap_csv"
      echo "[START test-only] $key seed=$seed"

      "$PY" "$ORDER_DUMPER" \
        --config "$config" \
        --out "$order_csv" \
        > "$OUT/driver_logs/${key}_seed${seed}_order.log" 2>&1

      "$PY" main_landmark.py test \
        --config "$config" \
        --ckpt_path "$ckpt" \
        --model.init_args.endpoint_order_invariant_nme false \
        --model.init_args.test_nme_dump_path "$fixed_csv" \
        > "$OUT/driver_logs/${key}_seed${seed}_fixed.log" 2>&1

      "$PY" main_landmark.py test \
        --config "$config" \
        --ckpt_path "$ckpt" \
        --model.init_args.endpoint_order_invariant_nme true \
        --model.init_args.test_nme_dump_path "$swap_csv" \
        > "$OUT/driver_logs/${key}_seed${seed}_swap.log" 2>&1

      validate_outputs "$order_csv" "$fixed_csv" "$swap_csv" \
        || { echo "ERROR: output validation failed for $key seed=$seed" >&2; exit 1; }
      echo "[DONE] $key seed=$seed"
    fi

    "$PY" - "$key" "$seed" "$fixed_csv" "$swap_csv" \
      >> "$OUT/ucl_eomt_per_image_results.tsv.tmp" <<'PY'
import csv
import statistics
import sys

key, seed, fixed_path, swap_path = sys.argv[1:]
def values(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return [float(row["nme"]) for row in csv.DictReader(handle)]
fixed, swap = values(fixed_path), values(swap_path)
if not fixed or len(fixed) != len(swap):
    raise SystemExit("invalid per-image rows")
print(f"{key}\t{seed}\t{len(fixed)}\t{100.0*statistics.fmean(fixed):.8f}\t{100.0*statistics.fmean(swap):.8f}")
PY
  done
done

mv -f -- "$OUT/ucl_eomt_per_image_results.tsv.tmp" "$OUT/ucl_eomt_per_image_results.tsv"
echo "[COMPLETE] 40/40 UCL EoMT final checkpoints exported under $OUT"
