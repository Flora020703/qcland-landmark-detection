#!/usr/bin/env python3
# ---------------------------------------------------------------
# MODIFIED: new file — correctness check for scale_augment added to
# datasets/landmark_dataset.py's __getitem__ (paired with rotate_augment,
# see scripts/verify_rotation_augment.py / verify_rotation_augment_pipeline.py
# for the same two-stage approach: synthetic geometry check, then real BPD
# images with a visual overlay).
#
# Part 1: synthetic geometry check — place a marker at a known offset from
# center, apply the scale-jitter image transform (resize + paste-onto-
# same-size-canvas), and confirm the marker's actual pixel position matches
# the coordinate formula used in __getitem__.
#
# Part 2: real pipeline check — load real BPD images through
# HeadLandmarkDataModule with scale_augment=True (and rotate_augment=True,
# matching the actual training config), save a few visual overlays.
#
# Usage (local WSL env, from repo root):
#   python3 scripts/verify_scale_augment.py
# ---------------------------------------------------------------

import os
import sys
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np
import torch
from PIL import Image, ImageDraw

from datasets.landmark_dataset import HeadLandmarkDataModule

# Set QCLAND_UCL_DATA_ROOT to your local copy of the Multicentre Fetal
# Biometry dataset to run this script.
_DATA_ROOT = Path(os.environ.get("QCLAND_UCL_DATA_ROOT", "<LOCAL_DATA_ROOT>/MultiCentre-Fetal-Biometry-2025"))
IMAGES_DIR = _DATA_ROOT / "images" / "UCL" / "Head"
ANN_TRAIN_CSV = _DATA_ROOT / "annotations" / "UCL" / "Head_Train.csv"
ANN_TEST_CSV = _DATA_ROOT / "annotations" / "UCL" / "Head_Test.csv"
OUT_DIR = Path("docs/static")


def check(label, ok, detail=""):
    print(f"  {'[OK]  ' if ok else '[FAIL]'} {label}" + (f"  {detail}" if detail else ""))
    return ok


def part1_geometry():
    print("=" * 60)
    print("  Part 1: synthetic geometry check")
    print("=" * 60)

    W, H = 200, 200
    cx, cy = W / 2.0, H / 2.0
    marker = (150, 100)  # 50px right of center

    all_ok = True
    for s in (1.25, 0.75, 1.0):
        img = Image.new("L", (W, H), color=0)
        px = img.load()
        for ddx in (-1, 0, 1):
            for ddy in (-1, 0, 1):
                px[marker[0] + ddx, marker[1] + ddy] = 255

        new_w, new_h = max(1, round(W * s)), max(1, round(H * s))
        resized = img.resize((new_w, new_h), Image.BILINEAR)
        canvas = Image.new(img.mode, (W, H), color=0)
        offset_x = round((W - new_w) / 2.0)
        offset_y = round((H - new_h) / 2.0)
        canvas.paste(resized, (offset_x, offset_y))

        arr = np.array(canvas)
        print(f"\n--- scale={s} ---")
        if arr.max() == 0:
            check("marker survived scaling (non-empty)", False)
            all_ok = False
            continue
        ys, xs = np.where(arr == arr.max())
        actual = (xs.mean(), ys.mean())

        predicted = (cx + (marker[0] - cx) * s, cy + (marker[1] - cy) * s)

        dist = ((actual[0] - predicted[0]) ** 2 + (actual[1] - predicted[1]) ** 2) ** 0.5
        print(f"  actual marker position (pixel search): ({actual[0]:.1f}, {actual[1]:.1f})")
        print(f"  formula-predicted position:             ({predicted[0]:.1f}, {predicted[1]:.1f})")
        ok = check("match within 2px", dist < 2.0, f"dist={dist:.2f}px")
        all_ok &= ok

    return all_ok


def heatmap_to_coord(hm: np.ndarray) -> tuple[float, float]:
    idx = np.unravel_index(np.argmax(hm), hm.shape)
    return float(idx[1]), float(idx[0])


def part2_pipeline():
    print("\n" + "=" * 60)
    print("  Part 2: real BPD data, full pipeline (rotate+scale together,")
    print("  matching the actual training config)")
    print("=" * 60)

    assert IMAGES_DIR.exists(), f"Not found: {IMAGES_DIR}"

    dm = HeadLandmarkDataModule(
        images_dir=IMAGES_DIR,
        ann_train_csv=ANN_TRAIN_CSV,
        ann_test_csv=ANN_TEST_CSV,
        task="bpd",
        img_size=(512, 512),
        heatmap_size=(64, 64),
        sigma=4.0,
        rotate_augment=True,
        scale_augment=True,
    )
    dm.setup()
    print(f"[OK] train samples: {len(dm.train_dataset)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    degenerate_count = 0
    N_DRAWS = 20

    for i in range(N_DRAWS):
        img_t, heatmaps, lms_hm = dm.train_dataset[0]
        hm = heatmaps.numpy()
        peaks = hm.reshape(hm.shape[0], -1).max(axis=1)
        if np.any(peaks < 0.5):
            degenerate_count += 1
            print(f"  [WARN] draw {i}: degenerate heatmap peak(s) {peaks}")

        if i < 6:
            img_np = (img_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            vis = Image.fromarray(img_np).convert("RGB")
            draw = ImageDraw.Draw(vis)
            hm_h, hm_w = hm.shape[1], hm.shape[2]
            iw, ih = vis.size
            for c in range(hm.shape[0]):
                x, y = heatmap_to_coord(hm[c])
                px, py = x * iw / hm_w, y * ih / hm_h
                r = 6
                draw.ellipse([px - r, py - r, px + r, py + r], outline=(255, 0, 0), width=3)
            out_path = OUT_DIR / f"scale_augment_check_{i}.png"
            vis.save(out_path)
            print(f"  [OK] saved {out_path} (peaks={peaks})")

    print(f"\n[INFO] degenerate draws: {degenerate_count}/{N_DRAWS}")
    return degenerate_count == 0


def main():
    ok1 = part1_geometry()
    ok2 = part2_pipeline()

    print("\n" + "=" * 60)
    if ok1 and ok2:
        print("[OK] Scale-jitter geometry and real-pipeline checks both passed.")
        print(f"Open {OUT_DIR}/scale_augment_check_*.png and confirm the red circles")
        print("sit on the correct anatomy, including any visibly zoomed-in/out draws.")
        sys.exit(0)
    else:
        print("[ERROR] Scale-jitter checks failed — do not use for training yet.")
        sys.exit(1)


if __name__ == "__main__":
    main()
