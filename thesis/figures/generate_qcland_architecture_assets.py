#!/usr/bin/env python3
"""Export four matched QCLand architecture-diagram assets from one real case.

The two heatmaps are direct final-head outputs from the supplied checkpoint.
The endpoint overlay uses the project's heatmap_to_coords decoder and therefore
shows model predictions, not ground-truth landmarks or hand-placed points.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import yaml
from PIL import Image, ImageDraw
from transformers import DINOv3ViTConfig, DINOv3ViTModel

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from models.eomt import EoMT
from models.vit import ViT
from training.landmark_detection import heatmap_to_coords


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_model(config_path: Path) -> tuple[EoMT, dict]:
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    network_args = config["model"]["init_args"]["network"]["init_args"]
    encoder_args = network_args["encoder"]["init_args"]
    data_args = config["data"]["init_args"]

    vit_allowed = set(inspect.signature(ViT.__init__).parameters) - {"self"}
    vit_args = {key: value for key, value in encoder_args.items() if key in vit_allowed}
    vit_args["img_size"] = tuple(data_args["img_size"])
    backbone_name = vit_args.get("backbone_name", "")
    if backbone_name.startswith("facebook/dinov3-vits16"):
        # Construct the exact public ViT-S/16 architecture locally. The
        # checkpoint contains every trained weight, so downloading pretrained
        # Hub weights would be unnecessary and would weaken reproducibility.
        image_height, image_width = vit_args["img_size"]
        assert image_height == image_width
        dinov3_config = DINOv3ViTConfig(
            image_size=image_height,
            patch_size=16,
            hidden_size=384,
            intermediate_size=1536,
            num_hidden_layers=12,
            num_attention_heads=6,
            num_register_tokens=4,
        )
        encoder = ViT.__new__(ViT)
        torch.nn.Module.__init__(encoder)
        encoder.backbone = encoder.transformers_to_timm(
            DINOv3ViTModel(dinov3_config), vit_args["img_size"]
        )
        pixel_mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, -1, 1, 1)
        pixel_std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, -1, 1, 1)
        encoder.register_buffer("pixel_mean", pixel_mean)
        encoder.register_buffer("pixel_std", pixel_std)
    else:
        encoder = ViT(**vit_args)

    eomt_allowed = set(inspect.signature(EoMT.__init__).parameters) - {"self", "encoder"}
    eomt_args = {key: value for key, value in network_args.items() if key in eomt_allowed}
    if "heatmap_size" in eomt_args:
        eomt_args["heatmap_size"] = tuple(eomt_args["heatmap_size"])
    return EoMT(encoder=encoder, **eomt_args), config


def save_heatmap(array: np.ndarray, path: Path) -> None:
    # Preserve the model's full response pattern; normalisation is display-only.
    low, high = float(array.min()), float(array.max())
    shown = (array - low) / max(high - low, 1e-12)
    shown = F.interpolate(
        torch.from_numpy(shown)[None, None].float(),
        size=(256, 256),
        mode="bilinear",
        align_corners=False,
    )[0, 0].numpy()
    plt.imsave(path, shown, cmap="turbo", vmin=0.0, vmax=1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model, config = build_model(args.config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = {
        key.removeprefix("network."): value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith("network.")
    }
    missing, unexpected = model.load_state_dict(state, strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    model.eval()

    image_size = tuple(config["data"]["init_args"]["img_size"])
    height, width = image_size
    input_image = Image.open(args.image).convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
    input_path = args.output_dir / "qcland_input_ultrasound.png"
    input_image.save(input_path)

    tensor = TF.to_tensor(input_image).unsqueeze(0)
    with torch.no_grad():
        heatmaps_per_layer, _, _ = model(tensor)
        heatmaps = heatmaps_per_layer[-1]
        coordinates_hm = heatmap_to_coords(heatmaps)[0]

    heatmaps_np = heatmaps[0].cpu().numpy()
    h1_path = args.output_dir / "qcland_heatmap_h1.png"
    h2_path = args.output_dir / "qcland_heatmap_h2.png"
    save_heatmap(heatmaps_np[0], h1_path)
    save_heatmap(heatmaps_np[1], h2_path)

    hm_height, hm_width = heatmaps_np.shape[-2:]
    scale = coordinates_hm.new_tensor([width / hm_width, height / hm_height])
    coordinates_image = (coordinates_hm * scale).cpu().numpy()

    overlay = input_image.copy()
    draw = ImageDraw.Draw(overlay)
    p1 = tuple(float(value) for value in coordinates_image[0])
    p2 = tuple(float(value) for value in coordinates_image[1])
    yellow = (255, 235, 0)
    draw.line([p1, p2], fill=yellow, width=5)
    radius = 8
    for x, y in (p1, p2):
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=yellow, outline="white", width=2)
    overlay_path = args.output_dir / "qcland_predicted_endpoint_pair.png"
    overlay.save(overlay_path)

    manifest_path = args.output_dir / "qcland_architecture_assets_manifest.tsv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow([
            "source_image", "source_sha256", "checkpoint_sha256", "config_sha256",
            "checkpoint_epoch", "checkpoint_global_step", "heatmap_height", "heatmap_width",
            "p1_x_512", "p1_y_512", "p2_x_512", "p2_y_512",
        ])
        writer.writerow([
            args.image.name, sha256(args.image), sha256(args.checkpoint), sha256(args.config),
            checkpoint.get("epoch"), checkpoint.get("global_step"), hm_height, hm_width,
            f"{p1[0]:.6f}", f"{p1[1]:.6f}", f"{p2[0]:.6f}", f"{p2[1]:.6f}",
        ])


if __name__ == "__main__":
    main()
