#!/usr/bin/env bash
# Final matched HRNet comparison: 2 datasets x 5 tasks x 5 controlled seeds.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIGS_DIR="${CONFIGS_DIR:-$SCRIPT_DIR/configs}"
HRNET_REPO_ROOT="${HRNET_REPO_ROOT:?set HRNET_REPO_ROOT to the audited upstream checkout}"
PY="${PY:-python3}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/root/autodl-tmp/hrnet_512_fixed_5seed}"
OUTPUT_ROOT="$ARTIFACT_ROOT/output"
TB_LOG_ROOT="$ARTIFACT_ROOT/tensorboard"
LOG_ROOT="$ARTIFACT_ROOT/driver_logs"
GENERATED_CONFIG_ROOT="$ARTIFACT_ROOT/generated_configs"
RUNTIME_CACHE_ROOT="$ARTIFACT_ROOT/runtime_cache"
RESULTS_TSV="${RESULTS_TSV:-$ARTIFACT_ROOT/hrnet_512_fixed_5seed_results.tsv}"
MIN_FREE_GB="${MIN_FREE_GB:-6}"
PRUNE_INTERVAL_SECONDS="${PRUNE_INTERVAL_SECONDS:-30}"
MANIFEST="$SCRIPT_DIR/expected_data_sha256.tsv"
EVALUATOR="$SCRIPT_DIR/evaluate_hrnet_fixed.py"
SEED_PATCHER="$SCRIPT_DIR/apply_controlled_seed_patch.py"
DECODE_PATCHER="$SCRIPT_DIR/apply_dynamic_decode_size_patch.py"
EXPECTED_COMMIT="21ee7cd70b9a3cee58d85dfb50b089dda6076867"
EXPECTED_PRETRAINED_SHA256="00eb200687c1ed1fc1042767e5b965052f4c7338823ba4d36a790979a078b36b"
PRETRAINED_REL="hrnetv2_pretrained/hrnetv2_w18_imagenet_pretrained.pth"
SEEDS=(42 0 123 2024 3407)
CONFIGS=(
  fetal_landmark_hrnet_w18_UCL_brain_BPD
  fetal_landmark_hrnet_w18_UCL_brain_OFD
  fetal_landmark_hrnet_w18_UCL_abdomen_APAD
  fetal_landmark_hrnet_w18_UCL_abdomen_TAD
  fetal_landmark_hrnet_w18_UCL_femur_FL
  fetal_landmark_hrnet_w18_MULTICENTRE_brain_BPD
  fetal_landmark_hrnet_w18_MULTICENTRE_brain_OFD
  fetal_landmark_hrnet_w18_MULTICENTRE_abdomen_APAD
  fetal_landmark_hrnet_w18_MULTICENTRE_abdomen_TAD
  fetal_landmark_hrnet_w18_MULTICENTRE_femur_FL
)

mkdir -p \
  "$OUTPUT_ROOT" \
  "$TB_LOG_ROOT" \
  "$LOG_ROOT" \
  "$GENERATED_CONFIG_ROOT" \
  "$RUNTIME_CACHE_ROOT/tmp" \
  "$RUNTIME_CACHE_ROOT/xdg" \
  "$RUNTIME_CACHE_ROOT/torch" \
  "$RUNTIME_CACHE_ROOT/cuda" \
  "$RUNTIME_CACHE_ROOT/matplotlib"

# Keep all run-generated caches and temporary files off the 30 GB system disk.
export TMPDIR="$RUNTIME_CACHE_ROOT/tmp"
export TMP="$RUNTIME_CACHE_ROOT/tmp"
export TEMP="$RUNTIME_CACHE_ROOT/tmp"
export XDG_CACHE_HOME="$RUNTIME_CACHE_ROOT/xdg"
export TORCH_HOME="$RUNTIME_CACHE_ROOT/torch"
export CUDA_CACHE_PATH="$RUNTIME_CACHE_ROOT/cuda"
export MPLCONFIGDIR="$RUNTIME_CACHE_ROOT/matplotlib"

EXPECTED_HEADER=$'config\tseed\tfixed_channel_nme_pct\tfixed_channel_per_image_sd_pct\tswap_min_nme_pct\tswap_min_per_image_sd_pct\ttest_n'
if [ ! -f "$RESULTS_TSV" ]; then
  printf '%s\n' "$EXPECTED_HEADER" > "$RESULTS_TSV"
elif [ "$(head -n 1 "$RESULTS_TSV")" != "$EXPECTED_HEADER" ]; then
  echo "ERROR: incompatible results header: $RESULTS_TSV" >&2
  exit 1
fi

check_checkout() {
  [ "$(git -C "$HRNET_REPO_ROOT" rev-parse HEAD)" = "$EXPECTED_COMMIT" ] || {
    echo "ERROR: unexpected HRNet commit" >&2; exit 1; }
  "$PY" - "$HRNET_REPO_ROOT/tools/train.py" "$HRNET_REPO_ROOT/tools/test.py" "$HRNET_REPO_ROOT/lib/utils/transforms.py" "$HRNET_REPO_ROOT/lib/core/function.py" <<'PY'
import pathlib, sys
train = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")
test = pathlib.Path(sys.argv[2]).read_text(encoding="utf-8")
transforms = pathlib.Path(sys.argv[3]).read_text(encoding="utf-8")
function = pathlib.Path(sys.argv[4]).read_text(encoding="utf-8")
required_train = (
    'run_seed = int(os.environ["HRNET_RUN_SEED"])',
    'random.seed(run_seed)', 'np.random.seed(run_seed)',
    'torch.manual_seed(run_seed)', 'torch.cuda.manual_seed_all(run_seed)',
    'worker_init_fn=_seed_worker', 'generator=loader_generator',
    'cudnn.benchmark = False', 'cudnn.deterministic = True',
)
missing = [item for item in required_train if item not in train]
if missing: raise SystemExit("ERROR: controlled_seed.patch incomplete: " + repr(missing))
if 'torch.load(args.model_file, map_location="cpu", weights_only=False)' not in test:
    raise SystemExit("ERROR: forward compatibility patch missing from tools/test.py")
if 'math.floor(max(ht, wd) / sf)' not in transforms:
    raise SystemExit("ERROR: NumPy compatibility patch missing from lib/utils/transforms.py")
dynamic_decode = (
    "decode_preds(score_map, meta['center'], meta['scale'], "
    "list(config.MODEL.HEATMAP_SIZE))"
)
hard_coded_decode = "decode_preds(score_map, meta['center'], meta['scale'], [64, 64])"
if function.count(dynamic_decode) != 3 or hard_coded_decode in function:
    raise SystemExit(
        "ERROR: dynamic heatmap decode patch must replace exactly three "
        "hard-coded 64x64 call sites in lib/core/function.py"
    )
PY
  local got
  got=$(sha256sum "$HRNET_REPO_ROOT/$PRETRAINED_REL" | awk '{print $1}')
  [ "$got" = "$EXPECTED_PRETRAINED_SHA256" ] || {
    echo "ERROR: pretrained weight checksum mismatch: $got" >&2; exit 1; }
}

check_disk() {
  local free_gb
  free_gb=$(df --output=avail -BG "$ARTIFACT_ROOT" | tail -1 | tr -dc '0-9')
  [ "$free_gb" -ge "$MIN_FREE_GB" ] || {
    echo "ERROR: ${free_gb}GB free, require ${MIN_FREE_GB}GB" >&2; exit 1; }
}

is_recorded() {
  awk -F'\t' -v c="$1" -v s="$2" 'NR>1 && $1==c && $2==s {f=1} END{exit !f}' "$RESULTS_TSV"
}

verify_and_generate_config() {
  local name="$1"
  local destination="$2"
  local source="$CONFIGS_DIR/$name.yaml"
  local upstream="$HRNET_REPO_ROOT/experiments/fetal/$name.yaml"
  "$PY" - "$upstream" "$source" "$destination" "$OUTPUT_ROOT" "$TB_LOG_ROOT" "$MANIFEST" <<'PY'
import copy, csv, hashlib, pathlib, sys, yaml
upstream_path, source_path, destination, output_root, tb_root, manifest_path = sys.argv[1:]
upstream = yaml.safe_load(open(upstream_path, encoding="utf-8"))
source = yaml.safe_load(open(source_path, encoding="utf-8"))
for obj in (upstream, source):
    for key in ("ROOT", "TRAINSET", "TESTSET"):
        obj["DATASET"].pop(key, None)
if upstream != source:
    raise SystemExit("ERROR: base config differs from upstream beyond the three audited data paths")
cfg = yaml.safe_load(open(source_path, encoding="utf-8"))
assert cfg["MODEL"]["SIGMA"] == 1.0
assert cfg["MODEL"]["HEATMAP_SIZE"] == [64, 64]
assert cfg["MODEL"]["NUM_JOINTS"] == 2
# CRITERION is inherited from the upstream defaults rather than repeated in
# these experiment YAMLs. Verify both any explicit override and that source
# default, so a future upstream/default drift cannot pass silently.
criterion = cfg["TRAIN"].get("CRITERION")
if criterion is not None and criterion != "MSE":
    raise SystemExit(f"ERROR: unexpected explicit TRAIN.CRITERION={criterion!r}")
repo_root = pathlib.Path(upstream_path).resolve().parents[2]
defaults_text = (repo_root / "lib" / "config" / "defaults.py").read_text(encoding="utf-8")
if "_C.TRAIN.CRITERION = 'MSE'" not in defaults_text and '_C.TRAIN.CRITERION = "MSE"' not in defaults_text:
    raise SystemExit("ERROR: upstream default TRAIN.CRITERION is not verifiably MSE")
cfg["MODEL"]["IMAGE_SIZE"] = [512, 512]
# HRNet-W18 emits a stride-4 heatmap. Doubling the input from 256 to 512
# therefore doubles the native output/target grid from 64 to 128. Keeping a
# 64 grid would require an extra resize operation not present upstream and, as
# the integration gate demonstrated, otherwise causes a 128-vs-64 MSE error.
cfg["MODEL"]["HEATMAP_SIZE"] = [128, 128]
cfg["OUTPUT_DIR"] = output_root
cfg["LOG_DIR"] = tb_root
root = pathlib.Path(cfg["DATASET"]["ROOT"])
if not root.is_dir(): raise SystemExit(f"ERROR: missing image root {root}")
manifest = {}
with open(manifest_path, newline="", encoding="utf-8") as handle:
    for row in csv.DictReader(handle, delimiter="\t"):
        manifest[(row["dataset"], row["file"])] = (int(row["rows"]), row["sha256"])
dataset = "MULTICENTRE" if "MULTICENTRE" in cfg["DATASET"]["TRAINSET"] else "UCL"
for key in ("TRAINSET", "TESTSET"):
    path = pathlib.Path(cfg["DATASET"][key])
    raw = path.read_bytes()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    expected = manifest.get((dataset, path.name))
    actual = (len(rows), hashlib.sha256(raw).hexdigest())
    if expected != actual: raise SystemExit(f"ERROR: data manifest mismatch for {path}: {actual} != {expected}")
    missing = [row["image_name"] for row in rows if not (root / row["image_name"]).is_file()]
    if missing: raise SystemExit(f"ERROR: {len(missing)} missing images for {path}; first={missing[:3]}")
with open(destination, "w", encoding="utf-8") as handle:
    yaml.safe_dump(cfg, handle, sort_keys=False)
PY
}

checkpoint_janitor() {
  local model_dir="$1" train_pid="$2"
  while kill -0 "$train_pid" 2>/dev/null; do
    if [ -d "$model_dir" ]; then
      mapfile -t stale < <(find "$model_dir" -maxdepth 1 -type f -name 'checkpoint_[0-9]*.pth' -printf '%T@ %p\n' | sort -nr | tail -n +3 | cut -d' ' -f2-)
      [ "${#stale[@]}" -eq 0 ] || rm -f -- "${stale[@]}"
    fi
    sleep "$PRUNE_INTERVAL_SECONDS"
  done
}

run_one() {
  local cfg="$1"
  local seed="$2"
  if is_recorded "$cfg" "$seed"; then echo "[SKIP] $cfg seed=$seed"; return; fi
  check_disk
  local run_name="${cfg}_seed${seed}_512fixed"
  local run_cfg="$GENERATED_CONFIG_ROOT/$run_name.yaml"
  local model_dir="$OUTPUT_ROOT/FETAL/$run_name"
  local train_log="$LOG_ROOT/${run_name}_train.log"
  local test_log="$LOG_ROOT/${run_name}_test.log"
  local fixed_csv="$model_dir/fixed_channel_per_image.csv"
  local fixed_json="$model_dir/fixed_channel_summary.json"
  local final_state="$model_dir/final_state.pth"
  verify_and_generate_config "$cfg" "$run_cfg"

  if [ ! -s "$final_state" ]; then
    if [ -d "$model_dir" ]; then
      echo "ERROR: partial run directory exists without final_state: $model_dir" >&2
      echo "Archive/inspect it, then remove only that exact directory before retrying." >&2
      exit 1
    fi
    echo "[START] $cfg seed=$seed"
    (cd "$HRNET_REPO_ROOT" && HRNET_RUN_SEED="$seed" "$PY" tools/train.py --cfg "$run_cfg") > "$train_log" 2>&1 &
    local train_pid=$!
    checkpoint_janitor "$model_dir" "$train_pid" &
    local janitor_pid=$!
    local status=0
    wait "$train_pid" || status=$?
    kill "$janitor_pid" 2>/dev/null || true
    wait "$janitor_pid" 2>/dev/null || true
    [ "$status" -eq 0 ] || { echo "ERROR: training failed: $train_log" >&2; exit "$status"; }
  else
    echo "[RESUME TEST] $cfg seed=$seed has final_state"
  fi

  (cd "$HRNET_REPO_ROOT" && "$PY" tools/test.py --cfg "$run_cfg" --model-file "$final_state") > "$test_log" 2>&1
  [ -s "$model_dir/predictions.pth" ] || { echo "ERROR: predictions missing" >&2; exit 1; }
  "$PY" "$EVALUATOR" --repo "$HRNET_REPO_ROOT" --config "$run_cfg" --checkpoint "$final_state" --predictions "$model_dir/predictions.pth" --per-image-csv "$fixed_csv" --summary-json "$fixed_json"
  local values
  values=$("$PY" - "$fixed_json" <<'PY'
import json, sys
x=json.load(open(sys.argv[1], encoding="utf-8"))
print(f'{x["fixed_channel_mean_pct"]:.6f} {x["fixed_channel_sample_sd_pct"]:.6f} {x["swap_min_mean_pct"]:.6f} {x["swap_min_sample_sd_pct"]:.6f} {x["n"]}')
PY
)
  read -r fixed fixed_sd swap swap_sd count <<< "$values"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$cfg" "$seed" "$fixed" "$fixed_sd" "$swap" "$swap_sd" "$count" >> "$RESULTS_TSV"

  find "$model_dir" -maxdepth 1 -type f \( -name 'checkpoint_[0-9]*.pth' -o -name 'checkpoint_best.pth' -o -name 'model_best.pth' -o -name 'current_pred.pth' \) -delete
  echo "[DONE] $cfg seed=$seed fixed=${fixed}% swap=${swap}% n=$count"
}

"$PY" "$SEED_PATCHER" "$HRNET_REPO_ROOT"
"$PY" "$DECODE_PATCHER" "$HRNET_REPO_ROOT"
check_checkout
if [ "${PREFLIGHT_ONLY:-0}" = 1 ]; then
  for cfg in "${CONFIGS[@]}"; do
    verify_and_generate_config "$cfg" "$GENERATED_CONFIG_ROOT/${cfg}_preflight_512fixed.yaml"
    echo "[PREFLIGHT OK] $cfg: 512x512, native stride-4 heatmap 128x128, sigma=1.0, data manifest verified"
  done
  echo "[PREFLIGHT COMPLETE] no training was started"
  exit 0
fi
for cfg in "${CONFIGS[@]}"; do
  for seed in "${SEEDS[@]}"; do run_one "$cfg" "$seed"; done
done
echo "[COMPLETE] all 50 HRNet 512/fixed runs are recorded in $RESULTS_TSV"
