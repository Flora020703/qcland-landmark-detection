#!/usr/bin/env python3
# ---------------------------------------------------------------
# MODIFIED: new file — geometric correctness check for the rotation
# augmentation added to datasets/landmark_dataset.py's __getitem__.
#
# The risk being checked: PIL's Image.rotate(angle) rotates counter-
# clockwise (as viewed) for positive angle, but image y grows *downward*,
# so the landmark coordinate formula needs a sign flip relative to the
# textbook (y-up) rotation matrix. Getting this wrong silently
# desynchronizes the image and its GT landmark — no exception, just wrong
# training data — so this is verified empirically against PIL's actual
# pixel output, not just re-derived on paper.
#
# Test: place a single bright marker pixel at a known offset from the
# image center, rotate the image 90 degrees with PIL, find where the
# marker actually landed (by pixel search), and compare against where
# the same __getitem__ formula predicts it should land. 90 degrees is
# used here (not the real ±30 training range) purely because it gives an
# unambiguous, easy-to-verify expected position — the same formula is
# used for any angle.
#
# Usage:
#   python3 scripts/verify_rotation_augment.py
# ---------------------------------------------------------------

import math
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, ".")


def rotate_point(x, y, cx, cy, angle_deg):
    """Exact copy of the formula in datasets/landmark_dataset.py's __getitem__."""
    theta = math.radians(angle_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    dx, dy = x - cx, y - cy
    new_x = cx + dx * cos_t + dy * sin_t
    new_y = cy - dx * sin_t + dy * cos_t
    return new_x, new_y


def check(label, ok, detail=""):
    print(f"  {'[OK]  ' if ok else '[FAIL]'} {label}" + (f"  {detail}" if detail else ""))
    return ok


def main():
    W, H = 200, 200
    cx, cy = W / 2.0, H / 2.0
    marker = (150, 100)  # 50px to the right of center, vertically centered

    all_ok = True

    for angle in (90.0, -90.0, 30.0, -30.0, 0.0):
        img = Image.new("L", (W, H), color=0)
        px = img.load()
        px[marker[0], marker[1]] = 255
        # small neighborhood so bilinear resampling doesn't fully erase the marker
        for ddx in (-1, 0, 1):
            for ddy in (-1, 0, 1):
                px[marker[0] + ddx, marker[1] + ddy] = 255

        rotated = img.rotate(angle, resample=Image.BILINEAR, expand=False)
        arr = np.array(rotated)
        if arr.max() == 0:
            print(f"\n--- angle={angle} ---")
            check("marker survived rotation (non-empty)", False)
            all_ok = False
            continue

        ys, xs = np.where(arr == arr.max())
        actual = (xs.mean(), ys.mean())

        predicted = rotate_point(marker[0], marker[1], cx, cy, angle)

        dist = math.hypot(actual[0] - predicted[0], actual[1] - predicted[1])
        print(f"\n--- angle={angle} ---")
        print(f"  PIL actual marker position (pixel search): ({actual[0]:.1f}, {actual[1]:.1f})")
        print(f"  Formula-predicted position:                 ({predicted[0]:.1f}, {predicted[1]:.1f})")
        ok = check(f"match within 2px", dist < 2.0, f"dist={dist:.2f}px")
        all_ok &= ok

    print("\n" + "=" * 60)
    if all_ok:
        print("[OK] Rotation formula matches PIL's actual behavior — safe to use for training.")
        sys.exit(0)
    else:
        print("[ERROR] Rotation formula does NOT match PIL — do not use for training yet.")
        sys.exit(1)


if __name__ == "__main__":
    main()
