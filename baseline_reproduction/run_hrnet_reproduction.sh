#!/usr/bin/env bash
# Faithful orchestration of the supervisors' HRNet/BiometryNet code.
# Canonical configs differ from the source repository only in the three data
# paths. Runtime-only output/log paths are prepended to generated config copies.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIGS_DIR="${CONFIGS_DIR:-$SCRIPT_DIR/configs}"
HRNET_REPO_ROOT="${HRNET_REPO_ROOT:?set HRNET_REPO_ROOT to the clean HRNet checkout}"
PY="${PY:-python3}"
REPEATS="${REPEATS:-5}"
DATASET_FILTER="${DATASET_FILTER:-ALL}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/root/autodl-tmp/hrnet_reproduction}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ARTIFACT_ROOT/output}"
TB_LOG_ROOT="${TB_LOG_ROOT:-$ARTIFACT_ROOT/tensorboard}"
DRIVER_LOG_DIR="${DRIVER_LOG_DIR:-$ARTIFACT_ROOT/driver_logs}"
MIN_FREE_GB="${MIN_FREE_GB:-8}"
STOP_AFTER_CONFIG="${STOP_AFTER_CONFIG:-1}"
PRUNE_INTERVAL_SECONDS="${PRUNE_INTERVAL_SECONDS:-30}"
EXPECTED_PRETRAINED_SHA256="00eb200687c1ed1fc1042767e5b965052f4c7338823ba4d36a790979a078b36b"
PRETRAINED_REL="hrnetv2_pretrained/hrnetv2_w18_imagenet_pretrained.pth"
MANIFEST="$SCRIPT_DIR/expected_data_sha256.tsv"
FORMAL_RESULTS="$SCRIPT_DIR/hrnet_reproduction_results.tsv"
CANARY_RESULTS="$SCRIPT_DIR/hrnet_canary_results.tsv"
REP_CFG_DIR="$HRNET_REPO_ROOT/experiments/fetal_reproduction_repeats"

mkdir -p "$OUTPUT_ROOT" "$TB_LOG_ROOT" "$DRIVER_LOG_DIR" "$REP_CFG_DIR"

ALL_CONFIGS=(
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
CONFIGS=("${ALL_CONFIGS[@]}")
RESULTS_TSV="$FORMAL_RESULTS"
RUN_SUFFIX=""

case "$DATASET_FILTER" in
  ALL) ;;
  UCL|MULTICENTRE)
    CONFIGS=()
    for cfg in "${ALL_CONFIGS[@]}"; do
      if [[ "$cfg" == *"_${DATASET_FILTER}_"* ]]; then CONFIGS+=("$cfg"); fi
    done
    ;;
  *)
    echo "ERROR: DATASET_FILTER must be ALL, UCL or MULTICENTRE (got '$DATASET_FILTER')" >&2
    exit 1
    ;;
esac
[ "${#CONFIGS[@]}" -gt 0 ] || { echo "ERROR: dataset filter selected no configs" >&2; exit 1; }

if [ "${CANARY_ONLY:-0}" = 1 ]; then
  CANARY_CONFIG="${CANARY_CONFIG:-${CONFIGS[0]}}"
  valid_canary=0
  for cfg in "${CONFIGS[@]}"; do
    if [ "$cfg" = "$CANARY_CONFIG" ]; then valid_canary=1; break; fi
  done
  [ "$valid_canary" = 1 ] || {
    echo "ERROR: CANARY_CONFIG '$CANARY_CONFIG' is outside DATASET_FILTER='$DATASET_FILTER'" >&2
    exit 1
  }
  CONFIGS=("$CANARY_CONFIG")
  REPEATS=1
  RESULTS_TSV="$CANARY_RESULTS"
  RUN_SUFFIX="_CANARY"
  STOP_AFTER_CONFIG=1
  echo "[CANARY] config=$CANARY_CONFIG; isolated output and TSV; never a formal repeat"
fi

init_results() {
  local file="$1"
  local expected=$'config\trepeat\tnme_pct\tper_image_nme_sd_pct'
  if [ ! -f "$file" ]; then
    printf '%s\n' "$expected" > "$file"
  elif [ "$(head -n 1 "$file")" != "$expected" ]; then
    echo "ERROR: incompatible results header in $file; archive/migrate the old TSV first" >&2
    exit 1
  fi
}
init_results "$RESULTS_TSV"

check_source_checkout() {
  local commit status pretrained_hash
  commit=$(git -C "$HRNET_REPO_ROOT" rev-parse HEAD)
  [ "$commit" = 21ee7cd70b9a3cee58d85dfb50b089dda6076867 ] || {
    echo "ERROR: HRNet checkout is $commit, expected 21ee7cd..." >&2; exit 1; }

  status=$(git -C "$HRNET_REPO_ROOT" status --short --untracked-files=no)
  local allowed=$' M lib/utils/transforms.py\n M tools/test.py'
  [ -z "$status" ] || [ "$status" = "$allowed" ] || {
    echo "ERROR: unexpected tracked changes in HRNet checkout:" >&2
    echo "$status" >&2
    exit 1
  }

  "$PY" - "$HRNET_REPO_ROOT/lib/utils/transforms.py" <<'PY'
import pathlib, sys
p = pathlib.Path(sys.argv[1])
s = p.read_text(encoding="utf-8")
required = ("import math", "math.floor(max(ht, wd) / sf)", "math.floor(ht / sf)", "math.floor(wd / sf)")
missing = [x for x in required if x not in s]
if missing:
    raise SystemExit("ERROR: incomplete NumPy compatibility patch; missing: " + ", ".join(missing))
PY

  "$PY" - "$HRNET_REPO_ROOT/tools/test.py" <<'PY'
import pathlib, sys
s = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")
required = 'torch.load(args.model_file, map_location="cpu", weights_only=False)'
if required not in s:
    raise SystemExit("ERROR: missing trusted-local-checkpoint PyTorch compatibility patch in tools/test.py")
PY

  local pretrained="$HRNET_REPO_ROOT/$PRETRAINED_REL"
  [ -f "$pretrained" ] || { echo "ERROR: missing pretrained weights: $pretrained" >&2; exit 1; }
  pretrained_hash=$(sha256sum "$pretrained" | awk '{print $1}')
  [ "$pretrained_hash" = "$EXPECTED_PRETRAINED_SHA256" ] || {
    echo "ERROR: pretrained-weight SHA-256 mismatch: $pretrained_hash" >&2; exit 1; }
}

check_disk() {
  local free_gb
  free_gb=$(df --output=avail -BG "$ARTIFACT_ROOT" | tail -1 | tr -dc '0-9')
  [ "$free_gb" -ge "$MIN_FREE_GB" ] || {
    echo "ERROR: only ${free_gb}GB free at $ARTIFACT_ROOT; require ${MIN_FREE_GB}GB" >&2
    exit 1
  }
}

verify_config_untouched() {
  local cfg="$1" ours="$CONFIGS_DIR/$1.yaml" theirs="$HRNET_REPO_ROOT/experiments/fetal/$1.yaml"
  [ -f "$theirs" ] || { echo "ERROR: missing source config: $theirs" >&2; exit 1; }
  "$PY" - "$theirs" "$ours" <<'PY'
import copy, sys, yaml
theirs, ours = (yaml.safe_load(open(p, encoding="utf-8")) for p in sys.argv[1:])
for obj in (theirs, ours):
    for key in ("ROOT", "TRAINSET", "TESTSET"):
        obj["DATASET"].pop(key, None)
if theirs != ours:
    raise SystemExit("ERROR: canonical reproduction config differs beyond ROOT/TRAINSET/TESTSET")
PY
}

preflight_config() {
  local cfg_file="$1"
  "$PY" - "$cfg_file" "$MANIFEST" <<'PY'
import csv, hashlib, pathlib, sys, yaml
cfg = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
root = pathlib.Path(cfg["DATASET"]["ROOT"])
csvs = [pathlib.Path(cfg["DATASET"][k]) for k in ("TRAINSET", "TESTSET")]
if not root.is_dir(): raise SystemExit(f"ERROR: missing image root: {root}")
manifest = {}
with open(sys.argv[2], newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        manifest[(r["dataset"], r["file"])] = (int(r["rows"]), r["sha256"])
dataset = "MULTICENTRE" if "MULTICENTRE" in str(csvs[0]) else "UCL"
for p in csvs:
    if not p.is_file(): raise SystemExit(f"ERROR: missing CSV: {p}")
    expected = manifest.get((dataset, p.name))
    if expected is None: raise SystemExit(f"ERROR: no audit manifest entry for {dataset}/{p.name}")
    raw = p.read_bytes(); digest = hashlib.sha256(raw).hexdigest()
    with p.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if (len(rows), digest) != expected:
        raise SystemExit(f"ERROR: CSV mismatch for {p}: got rows={len(rows)}, sha256={digest}")
    missing = [r["image_name"] for r in rows if not (root / r["image_name"]).is_file()]
    if missing:
        raise SystemExit(f"ERROR: {len(missing)} images referenced by {p} are missing; first={missing[:5]}")
PY
}

write_environment_audit() {
  local out="$ARTIFACT_ROOT/environment_audit.txt"
  local history="$ARTIFACT_ROOT/environment_audit_history.txt"
  local tmp
  tmp=$(mktemp "$ARTIFACT_ROOT/.environment_audit.XXXXXX")
  {
    echo "audit_timestamp=$(date -Is)"
    echo "run_mode=$([ "${CANARY_ONLY:-0}" = 1 ] && echo canary || echo formal)"
    echo "hrnet_commit=$(git -C "$HRNET_REPO_ROOT" rev-parse HEAD)"
    echo "pretrained_sha256=$EXPECTED_PRETRAINED_SHA256"
    "$PY" --version
    "$PY" - <<'PY'
import cv2, numpy, scipy, torch, torchvision
print("torch=" + torch.__version__)
print("torchvision=" + torchvision.__version__)
print("numpy=" + numpy.__version__)
print("opencv=" + cv2.__version__)
print("scipy=" + scipy.__version__)
print("cuda=" + str(torch.version.cuda))
print("cudnn=" + str(torch.backends.cudnn.version()))
print("gpu=" + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE"))
PY
    "$PY" -m pip freeze
  } > "$tmp"

  # `out` always describes the environment of the most recent invocation.
  # The append-only history preserves canary/formal transitions and any
  # dependency changes made between invocations.
  {
    printf '\n===== environment audit snapshot =====\n'
    cat "$tmp"
  } >> "$history"
  mv -f "$tmp" "$out"
}

seen_for() {
  awk -F'\t' -v c="$1" -v r="$2" 'NR>1 && $1==c && $2==r {f=1} END{exit !f}' "$RESULTS_TSV"
}

# The upstream code writes a full optimizer checkpoint every epoch. This
# storage-only janitor retains the two newest epoch checkpoints while training;
# latest.pth therefore continues to point to a retained checkpoint for resume.
checkpoint_janitor() {
  local model_dir="$1" train_pid="$2"
  while kill -0 "$train_pid" 2>/dev/null; do
    if [ -d "$model_dir" ]; then
      mapfile -t old < <(find "$model_dir" -maxdepth 1 -type f -name 'checkpoint_[0-9]*.pth' \
        -printf '%T@ %p\n' | sort -nr | tail -n +3 | cut -d' ' -f2-)
      if [ "${#old[@]}" -gt 0 ]; then rm -f -- "${old[@]}"; fi
    fi
    sleep "$PRUNE_INTERVAL_SECONDS"
  done
}

parse_result() {
  "$PY" - "$1" <<'PY'
import math, re, sys
s = open(sys.argv[1], encoding="utf-8", errors="replace").read()
m = re.findall(r"nme:([0-9.eE+-]+) nme mean:([0-9.eE+-]+) nme std:([0-9.eE+-]+)", s)
if not m: raise SystemExit("ERROR: NME result line not found")
nme, mean, sd = map(float, m[-1])
if not all(map(math.isfinite, (nme, mean, sd))) or not (0 <= nme <= 10 and 0 <= sd <= 10):
    raise SystemExit(f"ERROR: invalid NME values: {m[-1]}")
if abs(nme - mean) > 5e-4:
    raise SystemExit(f"ERROR: logged nme and per-image mean disagree: {nme}, {mean}")
print(f"{nme*100:.4f} {sd*100:.4f}")
PY
}

summarise_config() {
  "$PY" - "$RESULTS_TSV" "$1" <<'PY'
import csv, statistics, sys
with open(sys.argv[1], newline="", encoding="utf-8") as f:
    vals=[float(r["nme_pct"]) for r in csv.DictReader(f, delimiter="\t") if r["config"]==sys.argv[2]]
if vals:
    sd=statistics.stdev(vals) if len(vals)>1 else float("nan")
    print(f"[summary] {sys.argv[2]}: repeats={len(vals)}, mean={statistics.mean(vals):.4f}%, repeat-level sample SD={sd:.4f}%")
PY
}

run_one() {
  local cfg="$1" repeat="$2"
  seen_for "$cfg" "$repeat" && { echo "[skip] $cfg repeat$repeat already recorded"; return; }
  DID_RUN_THIS_CONFIG=1
  check_disk
  local base_cfg="$CONFIGS_DIR/$cfg.yaml"
  verify_config_untouched "$cfg"
  preflight_config "$base_cfg"

  local run_name="${cfg}_repeat${repeat}${RUN_SUFFIX}"
  local rep_cfg="$REP_CFG_DIR/$run_name.yaml"
  local model_dir="$OUTPUT_ROOT/FETAL/$run_name"
  local train_log="$DRIVER_LOG_DIR/${run_name}_train.log"
  local test_log="$DRIVER_LOG_DIR/${run_name}_test.log"
  local model_file="$model_dir/final_state.pth"

  { printf "OUTPUT_DIR: '%s'\nLOG_DIR: '%s'\n" "$OUTPUT_ROOT" "$TB_LOG_ROOT"; cat "$base_cfg"; } > "$rep_cfg"

  if [ -s "$model_file" ]; then
    echo "=== reusing completed final_state.pth for $run_name; retrying test only ==="
  else
    echo "=== training $run_name ==="
    (cd "$HRNET_REPO_ROOT" && "$PY" tools/train.py --cfg "$rep_cfg") > "$train_log" 2>&1 &
    local train_pid=$!
    checkpoint_janitor "$model_dir" "$train_pid" &
    local janitor_pid=$!
    local train_status=0
    wait "$train_pid" || train_status=$?
    kill "$janitor_pid" 2>/dev/null || true
    wait "$janitor_pid" 2>/dev/null || true
    [ "$train_status" -eq 0 ] || { echo "ERROR: training failed; see $train_log" >&2; exit "$train_status"; }
  fi

  [ -f "$model_file" ] || { echo "ERROR: missing $model_file" >&2; exit 1; }
  echo "=== testing $run_name using upstream final_state.pth ==="
  (cd "$HRNET_REPO_ROOT" && "$PY" tools/test.py --cfg "$rep_cfg" --model-file "$model_file") > "$test_log" 2>&1
  [ -s "$model_dir/predictions.pth" ] || { echo "ERROR: predictions.pth was not saved" >&2; exit 1; }

  local pct per_image_sd
  read -r pct per_image_sd <<< "$(parse_result "$test_log")"
  printf '%s\t%s\t%s\t%s\n' "$cfg" "$repeat" "$pct" "$per_image_sd" >> "$RESULTS_TSV"

  # Training is complete: retain only the newest numbered checkpoint plus
  # upstream best/final states and predictions. This mirrors the source repo's
  # own post-training cleanup while preserving resume/audit material.
  mapfile -t old < <(find "$model_dir" -maxdepth 1 -type f -name 'checkpoint_[0-9]*.pth' \
    -printf '%T@ %p\n' | sort -nr | tail -n +2 | cut -d' ' -f2-)
  if [ "${#old[@]}" -gt 0 ]; then rm -f -- "${old[@]}"; fi
  echo "[done] $run_name: NME=${pct}%; per-image SD=${per_image_sd}%"
}

check_source_checkout
write_environment_audit

for cfg in "${CONFIGS[@]}"; do
  DID_RUN_THIS_CONFIG=0
  for repeat in $(seq 1 "$REPEATS"); do run_one "$cfg" "$repeat"; done
  summarise_config "$cfg"
  if [ "$STOP_AFTER_CONFIG" = 1 ] && [ "$DID_RUN_THIS_CONFIG" = 1 ]; then
    echo "[SAFE STOP] one complete dataset/task configuration finished; archive before re-invoking"
    exit 0
  fi
done
echo "=== HRNet reproduction sweep finished ==="
