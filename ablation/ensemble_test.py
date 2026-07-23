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
#
# MODIFIED: two additions, both opt-in and OFF by default (default path is
# byte-identical to the original heatmap-averaging behavior, so every
# previously reported number is unaffected):
#
#   --ckpt-configs cfg1.yaml cfg2.yaml ...   (same length/order as --ckpts)
#       Build each checkpoint from its OWN config instead of one shared
#       --config. Needed for cross-architecture ensembles (e.g. DINOv2 +
#       DINOv3 checkpoints together) — the previous single-config approach
#       would have silently loaded a mismatched-architecture checkpoint.
#       MODIFIED: state_dict loading is now strict — any missing/unexpected
#       key (from either path) raises RuntimeError instead of warning and
#       continuing, since every checkpoint combined here is expected to
#       match its config's architecture exactly (see code review, 2026-07-24).
#
#   --tta
#       For each model, average its prediction on the image with its
#       prediction on the horizontally-flipped image before ensembling
#       across models. NOT a simple channel-index average: landmark
#       channel identity is DOD-sorted by x-coordinate (see
#       datasets/landmark_dataset.py), so a query's "channel 0" doesn't
#       have a fixed anatomical meaning across a flip — the flipped
#       prediction is unflipped back to original orientation, THEN
#       re-sorted by x, before being combined with the original
#       prediction's (also re-sorted) coordinates. Averaging is done in
#       decoded coordinate space, not heatmap space, since the DOD
#       resort only makes sense post-decode.
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


def dod_sort(coords: torch.Tensor) -> torch.Tensor:
    """Re-canonicalize landmark channel identity by ascending x, matching
    datasets/landmark_dataset.py's DOD sort (channel 0 = lower x). coords:
    (B, Q, 2) with [x, y] in the last dim."""
    order = coords[..., 0].argsort(dim=-1)  # (B, Q)
    return torch.gather(coords, 1, order.unsqueeze(-1).expand(-1, -1, 2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="shared config for all checkpoints")
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--ckpt-configs", nargs="+", default=None,
                     help="per-checkpoint config, same length/order as --ckpts — "
                          "overrides --config, needed for cross-architecture ensembles")
    ap.add_argument("--tta", action="store_true",
                     help="average each model's prediction with its horizontal-flip "
                          "prediction (DOD-resorted) before ensembling across models")
    args = ap.parse_args()

    if args.ckpt_configs:
        assert len(args.ckpt_configs) == len(args.ckpts), \
            f"--ckpt-configs ({len(args.ckpt_configs)}) must match --ckpts ({len(args.ckpts)}) in length"
        config_paths = args.ckpt_configs
    else:
        assert args.config, "either --config or --ckpt-configs is required"
        config_paths = [args.config] * len(args.ckpts)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    models = []
    cfgs = []
    for ckpt_path, cfg_path in zip(args.ckpts, config_paths):
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        model = build_model(cfg)
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            # MODIFIED: was strict=False + print-and-continue (a single missing
            # head/FPN param could silently change the result without ever
            # exceeding some arbitrary "large mismatch" threshold). This script
            # only ever combines same-architecture checkpoints — matched-seed
            # under one --config, or matched-per-checkpoint under
            # --ckpt-configs — so ANY mismatch means something is genuinely
            # wrong (wrong config, wrong checkpoint file, mismatched
            # --ckpt-configs ordering), not something to tolerate and average
            # away. Fail loud instead.
            raise RuntimeError(
                f"[FAIL] {ckpt_path} (config={cfg_path}): missing={len(missing)} unexpected={len(unexpected)} keys — "
                f"checkpoint does not match this config's architecture.\n"
                f"  missing[:10]    = {missing[:10]}\n"
                f"  unexpected[:10] = {unexpected[:10]}"
            )
        model.to(device).eval()
        models.append(model)
        cfgs.append(cfg)
    print(f"[OK] loaded {len(models)} checkpoints for ensembling"
          + (" [cross-config]" if args.ckpt_configs else "")
          + (" [TTA]" if args.tta else ""))

    # All checkpoints being combined must agree on heatmap/img size and NME
    # normaliser — averaging coordinates or heatmaps across mismatched sizes
    # would silently produce garbage. Loud failure instead.
    heatmap_sizes = {tuple(c["data"]["init_args"]["heatmap_size"]) for c in cfgs}
    img_sizes = {tuple(c["data"]["init_args"]["img_size"]) for c in cfgs}
    assert len(heatmap_sizes) == 1, f"mismatched heatmap_size across configs: {heatmap_sizes}"
    assert len(img_sizes) == 1, f"mismatched img_size across configs: {img_sizes}"
    heatmap_size = heatmap_sizes.pop()
    img_size = img_sizes.pop()
    # MODIFIED: read the NME normaliser pair from config instead of relying on
    # compute_nme()'s default (0,1) — that default is correct for BPD/OFD
    # (N=2, normalise by the diameter itself) but silently wrong for 300W
    # (N=68), which needs norm_pair=(36,45) (outer eye corners). Omitting
    # this here previously produced a nonsensical ~23% "NME" on 300W.
    norm_pair = tuple(cfgs[0]["model"]["init_args"].get("nme_norm_pair", (0, 1)))

    dm = build_datamodule(cfgs[0])
    dm.setup()
    loader = dm.test_dataloader()

    def predict_coords(model, imgs):
        mask_logits_per_layer, _, _ = model(imgs)
        pred = F.interpolate(
            mask_logits_per_layer[-1], heatmap_size,
            mode="bilinear", align_corners=False,
        )
        return heatmap_to_coords(pred)  # (B, Q, 2), [x, y] in heatmap space

    all_nme = []
    with torch.no_grad():
        for imgs, _gt_heatmaps, gt_coords in loader:
            imgs = imgs.to(device)
            gt_coords = gt_coords.to(device)

            if args.tta:
                # Coordinate-space ensembling (needed for the DOD resort
                # between the flip and un-flip steps to make sense).
                coord_sum = None
                for model in models:
                    coords_orig = dod_sort(predict_coords(model, imgs))

                    imgs_flipped = torch.flip(imgs, dims=[-1])
                    coords_flip = predict_coords(model, imgs_flipped)
                    hm_w = heatmap_size[1]
                    coords_flip = coords_flip.clone()
                    coords_flip[..., 0] = (hm_w - 1) - coords_flip[..., 0]
                    coords_flip = dod_sort(coords_flip)

                    coords_model = (coords_orig + coords_flip) / 2.0
                    coord_sum = coords_model if coord_sum is None else coord_sum + coords_model
                avg_coords = coord_sum / len(models)
            else:
                # Original heatmap-space ensembling — unchanged from before,
                # byte-identical results to every previously reported number.
                summed = None
                for model in models:
                    mask_logits_per_layer, _, _coord_pred = model(imgs)
                    pred = F.interpolate(
                        mask_logits_per_layer[-1], heatmap_size,
                        mode="bilinear", align_corners=False,
                    )
                    summed = pred if summed is None else summed + pred
                avg_pred = summed / len(models)
                avg_coords = heatmap_to_coords(avg_pred)

            nme = compute_nme(avg_coords, gt_coords, heatmap_size, img_size, norm_pair=norm_pair)
            all_nme.extend(nme.cpu().tolist())

    mean_nme = sum(all_nme) / len(all_nme)
    tags = []
    if args.ckpt_configs:
        tags.append("cross-config")
    if args.tta:
        tags.append("TTA")
    suffix = f" [{'+'.join(tags)}]" if tags else ""
    print(f"[RESULT] Ensemble ({len(models)} models){suffix} test NME: {mean_nme * 100:.2f}%  (n={len(all_nme)} samples)")


if __name__ == "__main__":
    main()
