#!/usr/bin/env python3
# ---------------------------------------------------------------
# MODIFIED: new file — end-to-end smoke test for rotate_augment on real
# BPD data (not just the isolated geometry check in
# verify_rotation_augment.py). Loads a few real training samples with
# rotate_augment=True through the full HeadLandmarkDataModule pipeline,
# checks the resulting heatmaps aren't degenerate, and saves a visual
# overlay (image + GT landmark dot) so rotation correctness can be
# eyeballed against real anatomy, not just synthetic geometry.
#
# Run locally on CPU per CLAUDE.md convention ("Test data loading locally
# on CPU before training on AutoDL").
#
# Usage (from repo root, local WSL env):
#   python3 scripts/verify_rotation_augment_pipeline.py
# ---------------------------------------------------------------

import sys
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np
import torch
from PIL import Image, ImageDraw

from datasets.landmark_dataset import HeadLandmarkDataModule

# MODIFIED: local Windows/WSL data paths (see CLAUDE.md) — override if run elsewhere.
IMAGES_DIR = Path("/mnt/d/download/Project coding/msc/Muti/MultiCentre-Fetal-Biometry-2025/images/UCL/Head")
ANN_TRAIN_CSV = Path("/mnt/d/download/Project coding/msc/Muti/MultiCentre-Fetal-Biometry-2025/annotations/UCL/Head_Train.csv")
ANN_TEST_CSV = Path("/mnt/d/download/Project coding/msc/Muti/MultiCentre-Fetal-Biometry-2025/annotations/UCL/Head_Test.csv")
OUT_DIR = Path("docs/static")


def heatmap_to_coord(hm: np.ndarray) -> tuple[float, float]:
    idx = np.unravel_index(np.argmax(hm), hm.shape)
    return float(idx[1]), float(idx[0])  # (x, y)


def main():
    assert IMAGES_DIR.exists(), f"Not found: {IMAGES_DIR}"
    assert ANN_TRAIN_CSV.exists(), f"Not found: {ANN_TRAIN_CSV}"

    dm = HeadLandmarkDataModule(
        images_dir=IMAGES_DIR,
        ann_train_csv=ANN_TRAIN_CSV,
        ann_test_csv=ANN_TEST_CSV,
        task="bpd",
        img_size=(512, 512),
        heatmap_size=(64, 64),
        sigma=4.0,
        rotate_augment=True,
    )
    dm.setup()
    print(f"[OK] train samples: {len(dm.train_dataset)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rotated_count = 0
    degenerate_count = 0
    N_DRAWS = 20  # same sample, drawn repeatedly, to reliably hit the 60% rotation branch a few times

    for i in range(N_DRAWS):
        img_t, heatmaps, lms_hm = dm.train_dataset[0]

        hm = heatmaps.numpy()
        peaks = hm.reshape(hm.shape[0], -1).max(axis=1)
        if np.any(peaks < 0.5):  # a well-formed Gaussian heatmap peak should be close to 1.0
            degenerate_count += 1
            print(f"  [WARN] draw {i}: degenerate heatmap peak(s) {peaks}")

        # Save the first few draws as an image+landmark overlay for visual inspection.
        if i < 4:
            img_np = (img_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            vis = Image.fromarray(img_np).convert("RGB")
            draw = ImageDraw.Draw(vis)
            hm_h, hm_w = hm.shape[1], hm.shape[2]
            iw, ih = vis.size
            for c in range(hm.shape[0]):
                x, y = heatmap_to_coord(hm[c])
                # map heatmap-space peak back to image space for the overlay
                px, py = x * iw / hm_w, y * ih / hm_h
                r = 6
                draw.ellipse([px - r, py - r, px + r, py + r], outline=(255, 0, 0), width=3)
            out_path = OUT_DIR / f"rotate_augment_check_{i}.png"
            vis.save(out_path)
            print(f"  [OK] saved {out_path} (peaks={peaks})")

    print(f"\n[INFO] Out of {N_DRAWS} draws of the same sample, checked for degenerate heatmaps.")
    print(f"[INFO] degenerate draws: {degenerate_count}/{N_DRAWS}")
    if degenerate_count == 0:
        print("[OK] No degenerate heatmaps — rotation's out-of-bounds skip logic is behaving.")
    else:
        print("[WARN] Some draws produced degenerate heatmaps — inspect before trusting rotate_augment.")

    print(f"\nOpen the saved PNGs in {OUT_DIR}/rotate_augment_check_*.png and confirm the red circles")
    print("sit on the same anatomical landmark across draws, including any visibly rotated ones.")


if __name__ == "__main__":
    main()
