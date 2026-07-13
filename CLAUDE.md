# Project: EoMT for Fetal Biometry Landmark Detection

## What this is
MSc thesis at UCL. Adapting EoMT (Encoder-only Mask Transformer) from semantic
segmentation to heatmap-based landmark detection for fetal biometry (BPD & OFD).
Supervisors: Dr. Zhehua Mao, Dr. Sophia Bano (SRV group).

## What's been done
- Modified datasets/dataset.py and datasets/ade20k_semantic.py: zip→folder reading
- Successfully ran EOMT segmentation pipeline on ADE20K_mini (50 train/10 val)
  with DINOv2-ViT-S, loss decreasing over 5 epochs, pipeline verified working
- Early HRNet "reproduction" numbers (BPD=0.0649, OFD=0.0441) were only a
  quick pipeline-sanity run on ~10-20 images, NOT a real full-training-set
  reproduction — do not cite or compare against these. The only valid HRNet
  baseline is the published one (Di Vece et al.): BPD 8% (0.08±0.18),
  OFD 5% (0.05±0.11), UCL→UCL.

## Current phase
Phase 2: Convert EOMT from segmentation to landmark detection.
Using UCL Head data (BPD + OFD, 2 landmarks each) with existing Train/Test split
from Multicentre-Fetal-Biometry project.

## Architecture modification plan
1. datasets/landmark_dataset.py — new file: load UCL Head CSV annotations,
   generate Gaussian heatmap targets, output (img, heatmaps) pairs
2. training/landmark_detection.py — new file: MSE loss on heatmaps,
   NME metric, no Hungarian matching needed (1-to-1 query-landmark correspondence)
3. models/eomt.py — modify _predict(): remove class_head usage,
   keep mask_head for heatmap output, adjust upscale to output 64x64 heatmaps
4. configs/landmark/ — new config: num_q=2 (BPD) or num_q=2 (OFD),
   start with DINOv2-ViT-S backbone
5. Multi-layer supervision: keep but use MSE per layer instead of Hungarian matching
6. Attention mask annealing: disable initially, add back as ablation later

## Key differences from segmentation
- num_q = num_landmarks (2 for BPD, 2 for OFD), fixed 1-to-1, no matching
- class_head not needed (no classification, only regression)
- Loss: MSE on Gaussian heatmaps (not BCE+Dice+CE)
- Metric: NME (normalized by inter-landmark distance)
- Heatmap→coordinate: argmax + sub-pixel refinement or soft-argmax
- upscale output should be 64x64 (not ~56x56)

## Data
- UCL Head images: /mnt/d/download/Project coding/msc/Muti/MultiCentre-Fetal-Biometry-2025/images/UCL/Head/
- Annotations CSV: from Multicentre-Fetal-Biometry project (Head_Train.csv, Head_Test.csv)
- Each sample has 2 landmark points (diameter endpoints)
- DOD reassignment may be needed (landmark identity ambiguity)
- Directory convention:
  <dataset_name>/
    train/
      images/
      masks/
    val/
      images/
      masks/
    test/
      images/
      masks/

## Baseline reference code
- HRNet fetal biometry: /mnt/d/download/Project coding/msc/Muti/Multicentre-Fetal-Biometry/
  Key files: lib/datasets/fetal.py (heatmap generation), lib/core/evaluation.py (NME calc),
  lib/utils/transforms.py:generate_target() (Gaussian blob σ=1.0 on 64x64)

## Environment
- Local: Windows + WSL2 (CPU only, for dev/debug)
- Training: AutoDL RTX 4090 (PyTorch 2.8, CUDA 12.8, Python 3.12)

## Conventions
- Comment changes with "# MODIFIED: reason"
- Keep original EOMT code intact where possible, add new files for landmark task
- Test data loading locally on CPU before training on AutoDL