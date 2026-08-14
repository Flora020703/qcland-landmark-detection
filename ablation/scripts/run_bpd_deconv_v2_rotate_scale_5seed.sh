#!/usr/bin/env bash
# Five-seed UCL/BPD follow-up: DeconvHeadV2 + rotation/scale, without FPN/UDP.
# This closes the missing direct comparison needed to determine whether the
# augmented final recipe actually requires FPN+UDP.
set -euo pipefail

cd "$(dirname "$0")/../.."

PYTHON_BIN="${PYTHON_BIN:-python3}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
BASE_CONFIG="configs/landmark/bpd_dinov2_deconv_v2_rotate_scale.yaml"
REFERENCE_CONFIG="configs/landmark/bpd_dinov2_fpn_udp_rotate_scale.yaml"
CONDITION_NAME="deconv_v2_rotate_scale_fixedval"
RUN_ROOT="${RUN_ROOT:-/root/autodl-tmp/saved_checkpoints/bpd_deconv_v2_rotate_scale_fixedval_5seed_v2}"
GROUP_DIR="$RUN_ROOT/bpd_${CONDITION_NAME}"
RESULTS_TSV="$RUN_ROOT/bpd_${CONDITION_NAME}_native_results.tsv"
AGG_ROOT="$RUN_ROOT/original_space_pi_aggregation"
SEEDS=(42 0 123 2024 3407)
if [[ "${CANARY_ONLY:-0}" == "1" ]]; then
  SEEDS=(42)
fi

mkdir -p "$GROUP_DIR"
if [[ ! -f "$RESULTS_TSV" ]]; then
  printf 'seed\tfinal_native_pi_nme_pct\n' > "$RESULTS_TSV"
fi

for SEED in "${SEEDS[@]}"; do
  RUN_DIR="$GROUP_DIR/seed${SEED}"
  CFG="$RUN_DIR/config.yaml"
  FINAL="$RUN_DIR/final.ckpt"
  FIXED="$RUN_DIR/final_fixedchannel_per_image.csv"
  SWAP="$RUN_DIR/final_swapmin_per_image.csv"
  ORDER="$RUN_DIR/test_image_order.csv"
  COMPLETE="$RUN_DIR/COMPLETE"

  if [[ -f "$COMPLETE" ]]; then
    for required in "$CFG" "$FINAL" "$FIXED" "$SWAP" "$ORDER"; do
      [[ -s "$required" ]] || { echo "[ERROR] completion marker exists but file is missing: $required"; exit 1; }
    done
    echo "[SKIP] verified completed seed $SEED"
    continue
  fi
  RECOVER_POST_TRAIN=0
  if [[ -e "$RUN_DIR" ]]; then
    if [[ -s "$CFG" && -s "$FINAL" && ! -e "$COMPLETE" ]]; then
      RECOVER_POST_TRAIN=1
      echo "[RECOVER] final checkpoint and config exist for seed $SEED; skipping training"
    else
      echo "[ERROR] partial run is not safely post-training recoverable: $RUN_DIR"
      exit 1
    fi
  else
    mkdir -p "$RUN_DIR"
  fi

  if [[ "$RECOVER_POST_TRAIN" == "0" ]]; then
  "$PYTHON_BIN" - "$BASE_CONFIG" "$CFG" "$RUN_DIR" "$SEED" <<'PY'
import sys, yaml
src, dst, run_dir, seed = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
with open(src, encoding="utf-8") as f:
    c = yaml.safe_load(f)
c["seed_everything"] = seed
c["data"]["init_args"]["loader_seed"] = seed
c["trainer"]["callbacks"][0]["init_args"]["dirpath"] = run_dir
c["trainer"]["callbacks"][0]["init_args"]["filename"] = f"seed{seed}_best"
c["trainer"]["logger"]["init_args"]["name"] = f"bpd-deconv-v2-rotate-scale-seed{seed}"
with open(dst, "w", encoding="utf-8") as f:
    yaml.safe_dump(c, f, sort_keys=False)
PY

  "$PYTHON_BIN" - "$CFG" "$REFERENCE_CONFIG" "$SEED" <<'PY'
import sys, yaml
with open(sys.argv[1], encoding="utf-8") as f: c=yaml.safe_load(f)
with open(sys.argv[2], encoding="utf-8") as f: ref=yaml.safe_load(f)
s=int(sys.argv[3]); m=c["model"]["init_args"]; n=m["network"]["init_args"]; d=c["data"]["init_args"]
checks = {
 "seed_everything": c["seed_everything"] == s, "loader_seed": d["loader_seed"] == s,
 "max_epochs": c["trainer"]["max_epochs"] == 200, "final checkpoint enabled": c["trainer"]["callbacks"][0]["init_args"]["save_last"] is True,
 "DINOv2": n["encoder"]["init_args"]["backbone_name"] == "vit_small_patch14_reg4_dinov2",
 "joint fine-tuning": n["freeze_backbone"] is False, "LLRD": m["llrd"] == 0.8,
 "DeconvHeadV2": n["heatmap_head"] == "deconv_v2", "FPN disabled": n.get("use_fpn") is False and "fpn_layers" not in n,
 "UDP disabled": d.get("pixel_center_align") is False, "rotation enabled": d.get("rotate_augment") is True,
 "scale enabled": d.get("scale_augment") is True, "hybrid loss": m["loss_type"] == "hybrid",
 "loss constants": (m["alpha"],m["temperature"],m["lambda_coord"]) == (5.0,10.0,0.1),
 "input/heatmap": d["img_size"] == [512,512] and d["heatmap_size"] == [64,64] and n["heatmap_size"] == [64,64],
 "sigma": d["sigma"] == 4.0, "fixed split": d["val_split_seed"] == 42,
 "batch": d["batch_size"] == 16,
 "fixed-channel validation/early stopping": m["endpoint_order_invariant_nme"] is False,
 "checkpoint monitor": c["trainer"]["callbacks"][0]["init_args"]["monitor"] == "metrics/val_nme",
 "early-stopping monitor": c["trainer"]["callbacks"][1]["init_args"]["monitor"] == "metrics/val_nme",
 "early-stopping rule": (
     c["trainer"]["callbacks"][1]["init_args"]["mode"],
     c["trainer"]["callbacks"][1]["init_args"]["patience"],
     c["trainer"]["callbacks"][1]["init_args"]["min_delta"],
 ) == ("min", 20, 0.005),
}
bad=[k for k,v in checks.items() if not v]
for k,v in checks.items(): print(("[OK] " if v else "[FAIL] ")+k)
if bad: raise SystemExit("[ERROR] config checks failed: "+", ".join(bad))

# Recursive equality audit against the historical full configuration.
# Only these intended interventions/run-specific fields may differ.
def flatten(value, prefix=()):
    if isinstance(value, dict):
        out = {}
        for key, child in value.items(): out.update(flatten(child, prefix+(key,)))
        return out
    if isinstance(value, list):
        out = {}
        for idx, child in enumerate(value): out.update(flatten(child, prefix+(idx,)))
        return out
    return {prefix: value}

# Make the reference's implicit default explicit before comparison.
ref["model"]["init_args"]["endpoint_order_invariant_nme"] = False
actual_flat, ref_flat = flatten(c), flatten(ref)
allowed_prefixes = {
    ("seed_everything",),
    ("data","init_args","loader_seed"),
    ("data","init_args","pixel_center_align"),
    ("model","init_args","network","init_args","use_fpn"),
    ("model","init_args","network","init_args","fpn_layers"),
    ("trainer","callbacks",0,"init_args","dirpath"),
    ("trainer","callbacks",0,"init_args","filename"),
    ("trainer","logger","init_args","name"),
}
all_paths=set(actual_flat)|set(ref_flat)
unexpected=[]
for path in sorted(all_paths, key=str):
    if any(path[:len(prefix)] == prefix for prefix in allowed_prefixes): continue
    if actual_flat.get(path, "<MISSING>") != ref_flat.get(path, "<MISSING>"):
        unexpected.append((path, actual_flat.get(path,"<MISSING>"), ref_flat.get(path,"<MISSING>")))
if unexpected:
    for path, got, expected in unexpected:
        print(f"[FAIL] unapproved config drift {path}: got={got!r}, reference={expected!r}")
    raise SystemExit("[ERROR] effective config differs from reference beyond approved interventions")
print("[OK] recursive reference-config audit: no unapproved drift")
PY

  echo "=== training seed $SEED ==="
  "$PYTHON_BIN" main_landmark.py fit --config "$CFG" 2>&1 | tee "$RUN_DIR/train.log"
  [[ -s "$RUN_DIR/last.ckpt" ]] || { echo "[ERROR] missing final checkpoint"; exit 1; }
  mv "$RUN_DIR/last.ckpt" "$FINAL"
  fi

  "$PYTHON_BIN" ablation/dump_test_image_order.py --config "$CFG" --out "$ORDER"

  "$PYTHON_BIN" main_landmark.py test --config "$CFG" --ckpt_path "$FINAL" \
    --model.init_args.endpoint_order_invariant_nme true \
    --model.init_args.test_nme_dump_path "$SWAP" 2>&1 | tee "$RUN_DIR/final_swapmin_test.log"
  "$PYTHON_BIN" main_landmark.py test --config "$CFG" --ckpt_path "$FINAL" \
    --model.init_args.endpoint_order_invariant_nme false \
    --model.init_args.test_nme_dump_path "$FIXED" 2>&1 | tee "$RUN_DIR/final_fixedchannel_test.log"

  "$PYTHON_BIN" - "$ORDER" "$FIXED" "$SWAP" <<'PY'
import csv, sys
def rows(p):
    with open(p, newline="", encoding="utf-8-sig") as f: return list(csv.DictReader(f))
order,fixed,swap=map(rows,sys.argv[1:])
assert len(order)==len(fixed)==len(swap)==49, (len(order),len(fixed),len(swap))
assert [int(x["index"]) for x in fixed] == list(range(49))
assert [int(x["index"]) for x in swap] == list(range(49))
required={"pred_x0","pred_y0","gt_x0","gt_y0","pred_x1","pred_y1","gt_x1","gt_y1"}
assert required <= set(fixed[0]), required-set(fixed[0])
print("[OK] 49-image order and both per-image dumps verified")
PY
  NME=$(grep 'Test NME:' "$RUN_DIR/final_swapmin_test.log" | grep -oE '[0-9]+\.[0-9]+' | tail -1)
  [[ -n "$NME" ]] || { echo "[ERROR] could not parse final PI-NME"; exit 1; }
  EXISTING_NME=$(awk -F'\t' -v seed="$SEED" '$1==seed {print $2}' "$RESULTS_TSV")
  if [[ -n "$EXISTING_NME" ]]; then
    EXISTING_NME_UNIQUE=$(printf '%s\n' "$EXISTING_NME" | sort -u)
    [[ "$(printf '%s\n' "$EXISTING_NME_UNIQUE" | wc -l)" -eq 1 ]] || {
      echo "[ERROR] conflicting duplicate TSV rows already exist for seed $SEED"
      exit 1
    }
    [[ "$EXISTING_NME_UNIQUE" == "$NME" ]] || {
      echo "[ERROR] existing TSV value $EXISTING_NME_UNIQUE disagrees with recovered value $NME for seed $SEED"
      exit 1
    }
    echo "[RECOVER] existing native-results row agrees for seed $SEED"
  else
    printf '%s\t%s\n' "$SEED" "$NME" >> "$RESULTS_TSV"
  fi
  printf 'seed=%s\nfinal_native_pi_nme_pct=%s\n' "$SEED" "$NME" > "$COMPLETE"
done

if [[ "${CANARY_ONLY:-0}" == "1" ]]; then
  echo "[CANARY COMPLETE] seed 42 finished and verified; five-seed aggregation was not run"
  echo "Re-run without CANARY_ONLY=1 to verify/skip seed 42 and train the remaining four seeds."
  exit 0
fi

[[ ! -e "$AGG_ROOT" ]] || { echo "[ERROR] aggregation output already exists: $AGG_ROOT"; exit 1; }
"$PYTHON_BIN" endpoint_ordering_analysis/aggregate_bpd_deconv_v2_rotate_scale.py \
  --run-root "$RUN_ROOT" --images-root /root/autodl-tmp/images/UCL \
  --condition-name "$CONDITION_NAME" --output-root "$AGG_ROOT"

echo "[COMPLETE] five-seed DeconvHeadV2 + rotation/scale follow-up finished"
echo "Artifacts: $RUN_ROOT"
