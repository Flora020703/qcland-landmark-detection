#!/usr/bin/env python3
# ---------------------------------------------------------------
# MODIFIED: new file — diagnose why --tta in ablation/ensemble_test.py
# made a single model's NME much worse (12.24% -> 17.84%) instead of
# neutral/better. Runs one real test batch through a single checkpoint
# twice (original, then horizontally flipped), prints the raw decoded
# coordinates at each step (before/after unflip, before/after DOD sort),
# and compares against ground truth — to see whether the model's own
# prediction on a flipped image is simply unreliable, or whether the
# coordinate transform math is the bug.
#
# Usage (on the server, same env as ensemble_test.py):
#   python3 scripts/debug_tta_flip.py --config configs/landmark/bpd_dinov3_fpn_udp.yaml \
#       --ckpt /root/autodl-tmp/saved_checkpoints/dinov3_bpd_fpn_udp/seed42/seed42_best.ckpt
# ---------------------------------------------------------------

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
import yaml

from ablation.ensemble_test import build_datamodule, build_model, dod_sort
from training.landmark_detection import heatmap_to_coords


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=3, help="number of test samples to inspect")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    heatmap_size = tuple(cfg["data"]["init_args"]["heatmap_size"])

    model = build_model(cfg)
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict(state, strict=False)
    model.to(device).eval()

    dm = build_datamodule(cfg)
    dm.setup()
    loader = dm.test_dataloader()

    def predict_coords(imgs):
        mask_logits_per_layer, _, _ = model(imgs)
        pred = F.interpolate(mask_logits_per_layer[-1], heatmap_size, mode="bilinear", align_corners=False)
        return heatmap_to_coords(pred)

    seen = 0
    with torch.no_grad():
        for imgs, _gt_heatmaps, gt_coords in loader:
            imgs = imgs.to(device)
            gt_coords = gt_coords.to(device)

            coords_orig_raw = predict_coords(imgs)
            coords_orig = dod_sort(coords_orig_raw)

            imgs_flipped = torch.flip(imgs, dims=[-1])
            coords_flip_raw = predict_coords(imgs_flipped)

            hm_w = heatmap_size[1]
            coords_flip_unflipped = coords_flip_raw.clone()
            coords_flip_unflipped[..., 0] = (hm_w - 1) - coords_flip_raw[..., 0]
            coords_flip_sorted = dod_sort(coords_flip_unflipped)

            for b in range(imgs.shape[0]):
                if seen >= args.n:
                    return
                seen += 1
                print(f"\n=== sample {seen} ===")
                print(f"  GT (DOD order):                  {gt_coords[b].cpu().numpy().tolist()}")
                print(f"  orig pred (raw, model's channel order): {coords_orig_raw[b].cpu().numpy().tolist()}")
                print(f"  orig pred (DOD-sorted):           {coords_orig[b].cpu().numpy().tolist()}")
                print(f"  flip pred (raw, in FLIPPED frame): {coords_flip_raw[b].cpu().numpy().tolist()}")
                print(f"  flip pred (unflipped back, raw order): {coords_flip_unflipped[b].cpu().numpy().tolist()}")
                print(f"  flip pred (unflipped + DOD-sorted): {coords_flip_sorted[b].cpu().numpy().tolist()}")
                diff = (coords_orig[b] - coords_flip_sorted[b]).abs()
                print(f"  |orig_sorted - flip_sorted| per point: {diff.cpu().numpy().tolist()}  (heatmap-space px, small = consistent)")


if __name__ == "__main__":
    main()
