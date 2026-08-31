# Thesis-facing frozen four-model tables

Primary metric: original-image-space permutation-invariant NME (%). Main values are five-seed mean ± seed-level sample SD; lower is better.

## Main comparison

| Dataset | Measurement | Proposed--DINOv2 | Proposed--DINOv3 | HRNet-W18 | RTMPose-s | n |
|---|---|---:|---:|---:|---:|---:|
| UCL | BPD | 5.92 ± 0.66 | **5.14 ± 0.61** | 5.51 ± 0.46 | 11.31 ± 0.86 | 49 |
| UCL | OFD | 3.94 ± 0.55 | **3.75 ± 0.09** | 4.59 ± 0.49 | 9.12 ± 1.60 | 49 |
| UCL | APAD | 6.50 ± 1.15 | **5.34 ± 0.66** | 7.33 ± 1.30 | 16.56 ± 0.91 | 36 |
| UCL | TAD | 9.32 ± 1.33 | **6.47 ± 0.86** | 6.74 ± 0.98 | 18.98 ± 1.76 | 36 |
| UCL | FL | 1.61 ± 0.11 | **1.53 ± 0.15** | 1.99 ± 0.41 | 17.83 ± 1.53 | 39 |
| Multicentre | BPD | 6.90 ± 0.17 | 6.09 ± 0.19 | **4.68 ± 0.17** | 6.15 ± 0.15 | 1180 |
| Multicentre | OFD | 6.13 ± 0.45 | 5.73 ± 0.68 | **4.78 ± 0.17** | 5.66 ± 0.10 | 1189 |
| Multicentre | APAD | **6.99 ± 0.23** | 7.11 ± 0.21 | 8.89 ± 0.19 | 8.89 ± 0.37 | 161 |
| Multicentre | TAD | 8.82 ± 0.90 | **8.71 ± 0.54** | 8.75 ± 0.33 | 9.22 ± 0.32 | 161 |
| Multicentre | FL | 2.78 ± 0.11 | **2.71 ± 0.13** | 2.93 ± 0.32 | 4.95 ± 0.49 | 362 |

## Pre-specified proposed-versus-baseline paired contrasts

Effects are method A minus method B in NME percentage points; negative values favour the proposed method. CI is a 20,000-replicate paired image-bootstrap 95% interval. The displayed p-value is Wilcoxon signed-rank with global Holm correction across all 60 frozen pairwise contrasts.

| Dataset | Measurement | Contrast (A - B) | Difference, pp [95% CI] | Holm-60 p |
|---|---|---|---:|---:|
| UCL | BPD | Proposed--DINOv2 - HRNet-W18 | 0.41 [-0.98, 1.67] | 0.081 |
| UCL | BPD | Proposed--DINOv2 - RTMPose-s | -5.39 [-7.63, -3.34] | <0.001 |
| UCL | BPD | Proposed--DINOv3 - HRNet-W18 | -0.37 [-1.77, 0.76] | 0.219 |
| UCL | BPD | Proposed--DINOv3 - RTMPose-s | -6.16 [-8.51, -4.00] | <0.001 |
| UCL | OFD | Proposed--DINOv2 - HRNet-W18 | -0.65 [-1.78, 0.16] | 0.861 |
| UCL | OFD | Proposed--DINOv2 - RTMPose-s | -5.18 [-7.25, -3.37] | <0.001 |
| UCL | OFD | Proposed--DINOv3 - HRNet-W18 | -0.84 [-2.19, 0.07] | 1.000 |
| UCL | OFD | Proposed--DINOv3 - RTMPose-s | -5.36 [-7.50, -3.49] | <0.001 |
| UCL | APAD | Proposed--DINOv2 - HRNet-W18 | -0.83 [-3.12, 1.25] | 1.000 |
| UCL | APAD | Proposed--DINOv2 - RTMPose-s | -10.06 [-13.35, -7.23] | <0.001 |
| UCL | APAD | Proposed--DINOv3 - HRNet-W18 | -1.99 [-4.80, 0.51] | 1.000 |
| UCL | APAD | Proposed--DINOv3 - RTMPose-s | -11.22 [-14.94, -8.16] | <0.001 |
| UCL | TAD | Proposed--DINOv2 - HRNet-W18 | 2.58 [0.42, 5.16] | 0.035 |
| UCL | TAD | Proposed--DINOv2 - RTMPose-s | -9.66 [-14.99, -4.93] | 0.005 |
| UCL | TAD | Proposed--DINOv3 - HRNet-W18 | -0.27 [-2.80, 2.20] | 0.474 |
| UCL | TAD | Proposed--DINOv3 - RTMPose-s | -12.51 [-16.51, -8.99] | <0.001 |
| UCL | FL | Proposed--DINOv2 - HRNet-W18 | -0.39 [-2.11, 0.55] | <0.001 |
| UCL | FL | Proposed--DINOv2 - RTMPose-s | -16.22 [-26.67, -8.91] | <0.001 |
| UCL | FL | Proposed--DINOv3 - HRNet-W18 | -0.46 [-2.20, 0.51] | 0.004 |
| UCL | FL | Proposed--DINOv3 - RTMPose-s | -16.29 [-26.65, -9.07] | <0.001 |
| Multicentre | BPD | Proposed--DINOv2 - HRNet-W18 | 2.21 [1.90, 2.52] | <0.001 |
| Multicentre | BPD | Proposed--DINOv2 - RTMPose-s | 0.75 [0.41, 1.08] | <0.001 |
| Multicentre | BPD | Proposed--DINOv3 - HRNet-W18 | 1.40 [1.13, 1.67] | <0.001 |
| Multicentre | BPD | Proposed--DINOv3 - RTMPose-s | -0.06 [-0.39, 0.24] | 0.016 |
| Multicentre | OFD | Proposed--DINOv2 - HRNet-W18 | 1.35 [0.98, 1.75] | <0.001 |
| Multicentre | OFD | Proposed--DINOv2 - RTMPose-s | 0.47 [0.12, 0.82] | <0.001 |
| Multicentre | OFD | Proposed--DINOv3 - HRNet-W18 | 0.96 [0.63, 1.30] | <0.001 |
| Multicentre | OFD | Proposed--DINOv3 - RTMPose-s | 0.07 [-0.23, 0.38] | 1.000 |
| Multicentre | APAD | Proposed--DINOv2 - HRNet-W18 | -1.90 [-2.71, -1.15] | <0.001 |
| Multicentre | APAD | Proposed--DINOv2 - RTMPose-s | -1.89 [-2.91, -1.03] | <0.001 |
| Multicentre | APAD | Proposed--DINOv3 - HRNet-W18 | -1.78 [-2.53, -1.08] | 0.005 |
| Multicentre | APAD | Proposed--DINOv3 - RTMPose-s | -1.78 [-2.62, -1.02] | <0.001 |
| Multicentre | TAD | Proposed--DINOv2 - HRNet-W18 | 0.07 [-0.74, 0.89] | 1.000 |
| Multicentre | TAD | Proposed--DINOv2 - RTMPose-s | -0.40 [-1.75, 0.76] | 1.000 |
| Multicentre | TAD | Proposed--DINOv3 - HRNet-W18 | -0.04 [-0.69, 0.63] | 1.000 |
| Multicentre | TAD | Proposed--DINOv3 - RTMPose-s | -0.50 [-1.96, 0.66] | 1.000 |
| Multicentre | FL | Proposed--DINOv2 - HRNet-W18 | -0.15 [-0.74, 0.30] | <0.001 |
| Multicentre | FL | Proposed--DINOv2 - RTMPose-s | -2.17 [-5.00, -0.59] | <0.001 |
| Multicentre | FL | Proposed--DINOv3 - HRNet-W18 | -0.22 [-0.82, 0.23] | <0.001 |
| Multicentre | FL | Proposed--DINOv3 - RTMPose-s | -2.24 [-5.16, -0.56] | <0.001 |

The full six contrasts per cell, including Proposed--DINOv2 versus Proposed--DINOv3 and HRNet-W18 versus RTMPose-s, remain authoritative in `four_model_paired_statistics.tsv` and should be supplied as supplementary material.
