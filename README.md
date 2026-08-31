# QCLand: Query-Conditioned Landmark Localisation

Adapting self-supervised vision foundation models (DINOv2 / DINOv3) to precise
anatomical landmark localisation in fetal ultrasound biometry, via learned
landmark-query tokens and a dedicated spatial decoding head, **DeconvHeadV2**.

This repository contains the implementation, training/evaluation configs, and
experiment-reproduction scripts for the MSc thesis *"QCLand: Query-Conditioned
Landmark Localisation with Vision Foundation Models for Fetal Biometry"*
(UCL Department of Computer Science). The full dissertation, including the
complete methodology, experimental protocol, and reproducibility appendix, is
in [`thesis/`](thesis/).

## Overview

Fetal biometry (e.g. biparietal diameter, occipito-frontal diameter, abdominal
diameters, femur length) is measured clinically from a pair of anatomical
landmarks placed by a sonographer on a standard ultrasound plane. Manual
placement is subject to inter-observer variability, motivating automated
landmark localisation. Existing automated approaches are largely built on
task-specific convolutional architectures (e.g. HRNet-based heatmap
regression, SimCC-based coordinate classification), while self-supervised
vision foundation models offer general-purpose visual representations learned
at scale — but adapting them to precise, per-landmark spatial predictions
without losing localisation accuracy is not straightforward.

**QCLand** (Query-Conditioned Landmark Localisation) addresses this by:

1. Inserting a small number of **learned landmark-query tokens** into a
   pretrained ViT backbone's token sequence, so each query forms a
   landmark-specific representation directly through self-attention with the
   image tokens (no Hungarian matching — landmark identity is fixed, unlike
   the variable-cardinality instance segmentation setting this mechanism is
   adapted from).
2. Decoding each query's representation into a landmark heatmap with a new
   spatial decoding head, **DeconvHeadV2** — a FiLM-conditioned convolutional
   decoder that replaces the inherited per-pixel query–feature dot product
   with an explicit local spatial-processing path.
3. Instantiating the framework with two self-supervised backbone generations,
   **DINOv2** and **DINOv3** (`QCLand-DINOv2`, `QCLand-DINOv3`), trained
   end-to-end (no frozen backbone) with layer-wise learning-rate decay.

The query-in-encoder mechanism is adapted from the Encoder-only Mask
Transformer (EoMT — Kerssies et al., CVPR 2025), a semantic/instance/panoptic
segmentation architecture; this repository builds on the
[official EoMT implementation](https://github.com/tue-mps/eomt) and
re-purposes its encoder-query mechanism for fixed-cardinality landmark
detection instead of variable-cardinality segmentation.

## Key results

QCLand is evaluated on five fetal biometric measurements — biparietal
diameter (BPD), occipito-frontal diameter (OFD), transverse and
anteroposterior abdominal diameter (TAD, APAD), and femur length (FL) —
across the UCL and pooled Multicentre datasets, against locally reproduced
HRNet-W18 and RTMPose-s baselines, under a common evaluation protocol
(matched input resolution, five training seeds, common image subsets,
original-image-space permutation-invariant NME, paired bootstrap + Wilcoxon
statistical testing with Holm correction).

| Dataset | Task | QCLand-DINOv2 | QCLand-DINOv3 | HRNet-W18 | RTMPose-s |
|---|---|---|---|---|---|
| UCL | BPD | 5.92 ± 0.66 | **5.14 ± 0.61** | 5.51 ± 0.46 | 11.31 ± 0.86 |
| UCL | OFD | 3.94 ± 0.55 | **3.75 ± 0.09** | 4.59 ± 0.49 | 9.12 ± 1.60 |
| UCL | APAD | 6.50 ± 1.15 | **5.34 ± 0.66** | 7.33 ± 1.30 | 16.56 ± 0.91 |
| UCL | TAD | 9.32 ± 1.33 | **6.47 ± 0.86** | 6.74 ± 0.98 | 18.98 ± 1.76 |
| UCL | FL | 1.61 ± 0.11 | **1.53 ± 0.15** | 1.99 ± 0.41 | 17.83 ± 1.53 |
| Multicentre | BPD | 6.90 ± 0.17 | 6.09 ± 0.19 | **4.68 ± 0.17** | 6.15 ± 0.15 |
| Multicentre | OFD | 6.13 ± 0.45 | 5.73 ± 0.68 | **4.78 ± 0.17** | 5.66 ± 0.10 |
| Multicentre | APAD | **6.99 ± 0.23** | 7.11 ± 0.21 | 8.89 ± 0.19 | 8.89 ± 0.37 |
| Multicentre | TAD | 8.82 ± 0.90 | **8.71 ± 0.54** | 8.75 ± 0.33 | 9.22 ± 0.32 |
| Multicentre | FL | 2.78 ± 0.11 | **2.71 ± 0.13** | 2.93 ± 0.32 | 4.95 ± 0.49 |

*Five-seed mean ± sample SD, permutation-invariant NME (%), lower is better,
bold = lowest mean per row. Full paired significance results (bootstrap CIs,
Wilcoxon, Holm-60) are in
[`final_comparison/four_model_freeze_20260812_v1/`](final_comparison/four_model_freeze_20260812_v1/)
and `thesis/chapters/05_results.tex`.*

A QCLand variant records the lowest mean PI-NME in 8 of the 10
dataset–measurement settings, with QCLand-DINOv3 lowest on all five UCL
measurements. HRNet-W18 retains a clear advantage on the two Multicentre head
measurements (BPD, OFD). A staged ablation on BPD further shows that
replacing the inherited dot-product decoder with DeconvHeadV2 produces the
largest single reduction in localisation error among the evaluated
architectural changes. See `thesis/chapters/05_results.tex` and the
conclusion (`thesis/chapters/07_conclusion.tex`) for the full, appropriately
qualified findings — the advantages above are task- and dataset-dependent,
not universal.

## Repository structure

```
models/                    QCLand / EoMT-derived model definitions (ViT backbone,
                           query-conditioned architecture, DeconvHeadV2, ScaleBlock)
training/                  Lightning modules, losses, LR schedules, EMA
datasets/                  Dataset classes (UCL/Multicentre landmark data,
                           300W, ADE20K/COCO/Cityscapes from the base EoMT repo)
configs/landmark/          YAML configs for every landmark-detection run
                           (per measurement, backbone, and ablation condition)
main_landmark.py           CLI entry point for landmark training/testing
                           (python main_landmark.py fit|test --config ...)
main.py                    Original EoMT segmentation entry point (inherited)

ablation/scripts/          Driver scripts for the BPD development ablation
                           chain, five-seed reruns, and cross-anatomy sweeps
baseline_reproduction/     Locally reproduced HRNet-W18 baseline: patches,
                           configs, and provenance for running the upstream
                           BiometryNet training code on this project's data
rtmpose_reproduction/      Locally reproduced RTMPose-s baseline: locked
                           protocol, generated MMEngine configs, provenance
endpoint_ordering_analysis/  Endpoint-canonicalisation / correspondence checks
final_comparison/          Frozen four-model comparison: paired statistics,
                           per-image results, SHA-256-manifested evidence

thesis/                    Full LaTeX thesis source (chapters, figures,
                           bibliography) and the reproducibility appendix
```

## Setup

```bash
git clone https://github.com/Flora020703/qcland-landmark-detection.git
cd qcland-landmark-detection
pip install -r requirements.txt
```

Training was run on an AutoDL-hosted NVIDIA RTX 4090 (PyTorch 2.7,
CUDA 12.6); local CPU/WSL2 was used only for data-loading and pipeline
debugging. The HRNet-W18 and RTMPose-s baselines each require their own
separate environment — see
[`baseline_reproduction/`](baseline_reproduction/) and
[`rtmpose_reproduction/`](rtmpose_reproduction/) respectively for exact
pinned versions and setup steps.

## Usage

Train a QCLand configuration:

```bash
python main_landmark.py fit --config configs/landmark/bpd_deconv_v2_fpn_udp.yaml
```

Evaluate a trained checkpoint:

```bash
python main_landmark.py test \
  --config configs/landmark/bpd_deconv_v2_fpn_udp.yaml \
  --ckpt_path path/to/checkpoint.ckpt
```

Every reported result in the thesis has a corresponding config under
`configs/landmark/` and, for the five-seed/ablation results, a driver script
under `ablation/scripts/` that trains, evaluates, and aggregates across the
five fixed seeds (42, 0, 123, 2024, 3407) used throughout this thesis.

## Data

This project uses the pooled Multicentre fetal-biometry benchmark (UCL, FP,
and HC18 cohorts) released by Di Vece et al., licensed under
CC BY-NC-SA 4.0. **No dataset images or annotations are redistributed in this
repository** — see `thesis/chapters/04_experimental_setup.tex` (Section on
data sources, ethics, and licensing) for the data access procedure, and
`thesis/chapters/appendix_a_reproducibility.tex` for full licensing and
provenance detail.

## Reproducibility

Every formally reported result in the thesis is backed by an evidence trail
in this repository:

- **[`final_comparison/`](final_comparison/)** — the frozen four-model
  comparison (checksummed TSVs, per-image scores, bootstrap/Wilcoxon/Holm
  statistics) and the BPD development-chain evidence matrix.
- **[`configs/landmark/`](configs/landmark/)** and
  **[`ablation/scripts/`](ablation/scripts/)** — the exact configuration and
  driver script behind every reported checkpoint.
- **[`baseline_reproduction/`](baseline_reproduction/)** and
  **[`rtmpose_reproduction/`](rtmpose_reproduction/)** — audited,
  version-pinned reproductions of the HRNet-W18 and RTMPose-s baselines.
- **`thesis/chapters/appendix_a_reproducibility.tex`** — software/hardware
  environments, seeding methodology, and the authoritative per-run
  configuration convention for all three trained model families.

## Citation

If you use this code, please cite the thesis:

```bibtex
@mastersthesis{xu2026qcland,
  title  = {QCLand: Query-Conditioned Landmark Localisation with Vision Foundation Models for Fetal Biometry},
  author = {Xu, Nan},
  school = {University College London},
  year   = {2026}
}
```

This work adapts the query-in-encoder mechanism from:

```bibtex
@inproceedings{kerssies2025eomt,
  title     = {Your ViT is Secretly an Image Segmentation Model},
  author    = {Kerssies, Tommie and Cavagnero, Niccol{\`o} and Hermans, Alexander and Norouzi, Narges and Averta, Giuseppe and Leibe, Bastian and Dubbelman, Gijs and de Geus, Daan},
  booktitle = {CVPR},
  year      = {2025}
}
```

## Acknowledgements

Supervised by Dr. Zhehua Mao and Dr. Sophia Bano (SRV Group, UCL). Built on
the official [EoMT](https://github.com/tue-mps/eomt) implementation (MIT
licensed, Mobile Perception Systems Lab, TU Eindhoven); the HRNet-W18
baseline is reproduced from the original
[BiometryNet / Multicentre-Fetal-Biometry](https://github.com/surgical-vision/Multicentre-Fetal-Biometry)
release by Di Vece et al.; the RTMPose-s baseline is reproduced from
[OpenMMLab MMPose](https://github.com/open-mmlab/mmpose).

## License

MIT — see [`LICENSE`](LICENSE). This project retains the original license and
copyright notice from the upstream EoMT repository it is built on.
