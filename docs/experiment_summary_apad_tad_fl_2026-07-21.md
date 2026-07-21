# APAD/TAD/FL Cross-Anatomy Extension — Summary (2026-07-21)

## Purpose

Cheap extension of the already-established "best config" (DINOv2 + FPN+UDP +
rotation+scale augmentation — the same recipe validated on BPD/OFD) to two
more UCL anatomical structures: Abdomen (APAD, TAD) and Femur (FL). Goal:
test whether the method generalizes across anatomy, not just across
landmark task within the same Head images (BPD vs OFD).

## Data / methodology verification (done before any training)

- Train/Test split confirmed **byte-identical** to the original teacher's
  project — same unmodified `Abdomen_{Train,Test}.csv` /
  `Femur_{Train,Test}.csv` files, same images.
- Validation-set methodology matches BPD/OFD's own convention exactly:
  `val_fraction: 0.1`, `val_split_seed: 42` (grepped across all 5 configs
  — bpd/ofd/apad/tad/fl — to confirm).
- Config: `configs/landmark/{apad,tad,fl}_dinov2_fpn_udp_rotate_scale.yaml`
- Ablation driver: `ablation/scripts/run_{apad,tad,fl}_fpn_udp_rotate_scale_ablation.sh`
  (resumable — `DONE_MARKER` + checkpoint migration to
  `/root/autodl-tmp/saved_checkpoints/`, same proven template as
  BPD/OFD/300W scripts).
- 5 seeds standard: 42, 0, 123, 2024, 3407.

## Results

HRNet UCL→UCL reference (Di Vece et al., published cross-eval table):
APAD 0.08±0.14, TAD 0.08±0.14, FL 0.02±0.03.

| Dataset | Single-model final (mean±std, 5-seed) | Ensemble best | Ensemble final | HRNet baseline | Verdict |
|---|---|---|---|---|---|
| APAD | 7.53 ± 1.26 | 6.98% | **7.08%** | 8.00% | **beats** |
| TAD  | 12.61 ± 2.53 | 12.60% | **10.03%** | 8.00% | **loses** |
| FL   | 1.83 ± 0.10 | 1.71% | **1.69%** | 2.00% | **beats** |

Combined with OFD (already-established headline: 4.15% ensemble vs HRNet's
5.00%), this extension is **3 wins / 1 loss** across the four non-BPD
datasets tested — a real, mixed cross-anatomy generalization result worth
reporting honestly, not a uniform success story.

### Per-seed TAD detail (the one that lost)

| seed | best | final |
|---|---|---|
| 42 | 10.19 | 10.19 |
| 0 | 9.12 | 15.83 |
| 123 | 19.68 | 15.55 |
| 2024 | 20.04 | 10.50 |
| 3407 | 13.31 | 10.97 |

### Why TAD underperforms — diagnosed, not a bug

seed123's training log shows `val_nme` swinging wildly across epochs
(spiking down to 2.27% and 2.96% against a typical 12–24% elsewhere)
because TAD's validation split is only ~9–10 images — a single
`DataLoader` batch. A lucky batch can transiently crash `val_nme` to a
near-zero fluke that carries no generalization signal.
`ModelCheckpoint(mode="min")` dutifully saved that fluke moment as "best."
On the real 39-image test set, that exact checkpoint scored 19.68% — a
~17-point val/test gap.

This is the same small-val-set unreliability already known from the
project's pitfalls memory (Rule 12: "a 10-image val set is not reliable
for checkpoint selection"), now observed in an unusually extreme,
concretely-documented form. It is good, citable evidence for the thesis
methodology section explaining why "final" (not val-best) is the reporting
convention on these small UCL splits. TAD's overall higher cross-seed
variance (std 2.53 vs APAD's 1.26) is plausibly the same mechanism
recurring to different degrees across seeds, not a config error — TAD's
config was checked line-by-line against APAD's and matches on every
parameter except `task` / `images_dir` (the two things that should
legitimately differ).

## Code changes this session (uncommitted as of this summary — see below)

- `datasets/landmark_dataset.py`: `TASK_COLS` extended with `"apad"`,
  `"tad"`, `"fl"` entries; `task` docstring updated to list all five valid
  keys and their matching data directories.
- `ablation/ensemble_test.py`: added `--ckpt-configs` (per-checkpoint
  config resolution, for cross-architecture ensembles) and `--tta`
  (DOD-resorted flip-averaging — implemented but **not used** for
  APAD/TAD/FL; see the project's pitfalls memory Rule 20 for why TTA was
  abandoned after a negative BPD result). Both are opt-in / off by default
  — the default code path is unchanged from what produced every previously
  reported BPD/OFD/300W number.
- New configs: `configs/landmark/{apad,tad,fl}_dinov2_fpn_udp_rotate_scale.yaml`,
  plus `ofd_deconv_v2.yaml` (OFD baseline-rung config, from earlier in the
  session).
- New ablation scripts: `ablation/scripts/run_{apad,tad,fl}_fpn_udp_rotate_scale_ablation.sh`,
  `run_dinov2_ofd_fpn_udp_rotate_scale_ablation.sh`,
  `run_ofd_deconv_v2_baseline_ablation.sh`.
- New debug script: `scripts/debug_tta_flip.py` (used to root-cause the
  TTA flip-averaging regression, see pitfalls memory Rule 20).

## Next steps

1. **300W DINOv3 ablation** (5-seed, FPN+UDP) — queued, not yet started.
   `ablation/scripts/run_dinov3_300w_ablation.sh` already exists on the
   server.
2. **300W rotate+scale augmentation** — configs/scripts not yet created;
   mirror the BPD/OFD/APAD/TAD/FL rotate_scale pattern when starting this.
   Last item before Conclusion/Future Work.
3. Thesis writing continues in parallel (BPD methodology/ablation chapter
   has all data ready and is startable immediately; OFD/APAD/TAD/FL
   results sections can be written now too).

Full experiment history, negative results, and operational pitfalls are
tracked in this project's Claude memory
(`project_eomt_landmark.md` / `feedback_training_pitfalls.md`), which is
kept current across sessions — this document is a point-in-time snapshot
for handover/reference, not the canonical record.
