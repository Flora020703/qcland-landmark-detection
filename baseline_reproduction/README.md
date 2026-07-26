# HRNet/BiometryNet baseline reproduction (UCL + Multicentre)

Reproduces the supervisors' own HRNet-based fetal-biometry pipeline
(`surgical-vision/Multicentre-Fetal-Biometry`, Di Vece et al.) on UCL and
Multicentre data, using their code and configs as-is. This is the
"Reproduced HRNet" leg of the three-way comparison (Published BiometryNet /
Reproduced HRNet / EoMT) — see [[project_eomt_landmark]] item 4 in the
experiment queue.

## Source repo

Base: `surgical-vision/Multicentre-Fetal-Biometry`, commit `21ee7cd`
("Fix citation format in README.md"), from the **clean** local clone at
`msc/hrnet_new/Multicentre-Fetal-Biometry` (only 1 forward-compat diff vs.
that commit: `lib/utils/transforms.py`'s `np.math.floor` -> `math.floor`,
since `np.math` was removed in newer numpy — no other change).

**Do NOT use** the sibling clone at `msc/hrnet_ repo/Multicentre-Fetal-Biometry`
as a base — its working tree has uncommitted edits that strip every `.cuda()`
call, disable `DataParallel`, zero out `PRETRAINED`, and cut `END_EPOCH` to 10.
That's the ad-hoc CPU-only sanity harness behind the early
BPD=0.0649/OFD=0.0441 numbers CLAUDE.md already says not to cite — it is not
a real training config and must not be the starting point here.

## What's in this folder

- `configs/` — 10 experiment configs, one per (dataset, anatomy/metric):
  `UCL x {BPD, OFD, APAD, TAD, FL}` and `MULTICENTRE x {BPD, OFD, APAD, TAD, FL}`.
  Each is a byte-for-byte copy of the official
  `experiments/fetal/fetal_landmark_hrnet_w18_<DATASET>_<anatomy>_<METRIC>.yaml`
  from the source repo, with exactly one change: `DATASET.ROOT` /
  `DATASET.TRAINSET` / `DATASET.TESTSET` rewritten from the repo-relative
  `data/images/...` / `data/annotations/...` to the AutoDL server's absolute
  paths (`/root/autodl-tmp/images/...` / `/root/autodl-tmp/annotations/...`),
  which is where this project's EoMT Multicentre pipeline already has the
  same UCL/MULTICENTRE image+annotation data staged — no new data transfer
  needed. `SIGMA`, `IMAGE_SIZE`, `LOSS`, augmentation, optimizer, epochs,
  architecture: all untouched. Verified by `diff` against the official
  files that these 3 path lines are the only difference.
- `run_hrnet_reproduction.sh` — orchestration driver (see below).
- `ENVIRONMENT.md` — why `environment.yml`'s pinned stack can't run on the
  AutoDL RTX 4090, and what to install instead.

## What is intentionally NOT changed

Per CLAUDE.md's baseline-reproduction rule: do not tune sigma/resolution/loss
to match EoMT's own choices. Every hyperparameter in `configs/*.yaml` is the
supervisors' own published default (SIGMA=1.0, IMAGE_SIZE 256x256,
HEATMAP_SIZE 64x64, Adam lr=1e-4, 200 epochs, batch 16, HRNet-W18,
FLIP+SCALE_FACTOR 0.25+ROT_FACTOR 30 augmentation, REASSIGN/DOD enabled).
`lib/core/evaluation.py::compute_nme` already implements the direction-
invariant (swap-min) fetal NME formula natively — no metric patch needed
either; this reproduction and the published BiometryNet numbers are on the
same metric by construction.

## Seeds

The original repo has **no seed-control mechanism anywhere** (checked
`lib/config/`, `tools/train.py`, `tools/test.py` — zero hits for
`seed`/`manual_seed`; also no hardcoded `torch.manual_seed` anywhere that
would make repeats identical). Rather than adding seed-CLI code to their
train.py (which would be a real, if small, code change), the driver gets
5 independent samples per config the same way the project's earlier NME
screening runs would if the code had never had a seed argument: run
`tools/train.py`/`tools/test.py` 5 times unmodified, each pointed at a
distinct copy of the config file (`<name>_rep{1..5}.yaml`, identical body,
different filename only) so each repeat gets its own
`output/FETAL/<cfg_name>_repN/` directory instead of overwriting the last
run. The rep-numbered config copies are generated on the fly by the driver
script on the server, not committed here, to keep git to the 10 canonical
configs.

## Usage (once the AutoDL GPU is free)

```bash
# 1. On the server: clone/scp the clean base repo (see ENVIRONMENT.md for the
#    exact commit + patch), then scp this baseline_reproduction/ folder next
#    to it (sibling directory, NOT inside the HRNet repo).
# 2. Set up the modernized env (ENVIRONMENT.md).
# 3. From inside the HRNet repo root:
CANARY_ONLY=1 bash /path/to/baseline_reproduction/run_hrnet_reproduction.sh   # 1 config x 1 rep sanity check
bash /path/to/baseline_reproduction/run_hrnet_reproduction.sh                  # full 10 configs x 5 reps
```

Results accumulate in `hrnet_reproduction_results.tsv` (resumable — a
completed `config,rep` row is never re-run).
