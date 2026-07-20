#!/usr/bin/env python3
# ---------------------------------------------------------------
# MODIFIED: new file — average heatmaps from N trained checkpoints (same
# architecture/config, different seeds) before decoding, then compute test
# NME once. No training involved — pure inference-time ensembling.
#
# Bypasses main_landmark.py's LightningCLI, so it must manually replicate
# the two argument links main_landmark.py sets up (see LandmarkCLI.
# add_arguments_to_parser): data.init_args.img_size -> both
# model.init_args.img_size and the encoder's img_size, and
# data.init_args.heatmap_size -> model.init_args.heatmap_size. Missing these
# would silently make LandmarkDetection.heatmap_size default to (64,64)
# regardless of what the network head actually produces.
#
# Usage (from repo root):
#   python3 ablation/ensemble_test.py --config configs/landmark/bpd_deconv_v2_fpn_udp_ema.yaml \
#       --ckpts checkpoints/fpn-udp-ema-ablation/seed*/seed*_final.ckpt
# ---------------------------------------------------------------

import argparse
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
import yaml

from models.eomt import EoMT
from models.vit import ViT
from training.landmark_detection import LandmarkDetection, compute_nme, heatmap_to_coords


def build_datamodule(cfg: dict):
    """MODIFIED: resolve the data module class from cfg["data"]["class_path"]
    instead of hardcoding HeadLandmarkDataModule — this script was only ever
    exercised against BPD/OFD configs (HeadLandmarkDataModule) until now;
    300W uses a different class (Face300WDataModule) with different
    __init__ kwargs (data_root/test_subset vs images_dir/ann_*_csv/task)."""
    class_path = cfg["data"]["class_path"]
    module_path, class_name = class_path.rsplit(".", 1)
    dm_class = getattr(importlib.import_module(module_path), class_name)
    return dm_class(**cfg["data"]["init_args"])


def build_model(cfg: dict) -> LandmarkDetection:
    m = cfg["model"]["init_args"]
    n = m["network"]["init_args"]
    data_args = cfg["data"]["init_args"]
    img_size = tuple(data_args["img_size"])
    heatmap_size = tuple(data_args["heatmap_size"])

    encoder = ViT(img_size=img_size, **n["encoder"]["init_args"])

    net_kwargs = {k: v for k, v in n.items() if k != "encoder"}
    net_kwargs["heatmap_size"] = tuple(net_kwargs["heatmap_size"])
    network = EoMT(encoder=encoder, **net_kwargs)

    model_kwargs = {k: v for k, v in m.items() if k != "network"}
    model_kwargs["img_size"] = img_size
    model_kwargs["heatmap_size"] = heatmap_size  # replicates main_landmark.py's link_arguments
    return LandmarkDetection(network=network, **model_kwargs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpts", nargs="+", required=True)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dm = build_datamodule(cfg)
    dm.setup()
    loader = dm.test_dataloader()

    models = []
    for ckpt_path in args.ckpts:
        model = build_model(cfg)
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"[WARN] {ckpt_path}: missing={len(missing)} unexpected={len(unexpected)}")
        model.to(device).eval()
        models.append(model)
    print(f"[OK] loaded {len(models)} checkpoints for ensembling")

    heatmap_size = tuple(cfg["data"]["init_args"]["heatmap_size"])
    img_size = tuple(cfg["data"]["init_args"]["img_size"])
    # MODIFIED: read the NME normaliser pair from config instead of relying on
    # compute_nme()'s default (0,1) — that default is correct for BPD/OFD
    # (N=2, normalise by the diameter itself) but silently wrong for 300W
    # (N=68), which needs norm_pair=(36,45) (outer eye corners). Omitting
    # this here previously produced a nonsensical ~23% "NME" on 300W.
    norm_pair = tuple(cfg["model"]["init_args"].get("nme_norm_pair", (0, 1)))

    all_nme = []
    with torch.no_grad():
        for imgs, _gt_heatmaps, gt_coords in loader:
            imgs = imgs.to(device)
            gt_coords = gt_coords.to(device)

            summed = None
            for model in models:
                mask_logits_per_layer, _, _coord_pred = model(imgs)
                pred = F.interpolate(
                    mask_logits_per_layer[-1], heatmap_size,
                    mode="bilinear", align_corners=False,
                )
                summed = pred if summed is None else summed + pred
            avg_pred = summed / len(models)

            pred_coords = heatmap_to_coords(avg_pred)
            nme = compute_nme(pred_coords, gt_coords, heatmap_size, img_size, norm_pair=norm_pair)
            all_nme.extend(nme.cpu().tolist())

    mean_nme = sum(all_nme) / len(all_nme)
    print(f"[RESULT] Ensemble ({len(models)} models) test NME: {mean_nme * 100:.2f}%  (n={len(all_nme)} samples)")


if __name__ == "__main__":
    main()
