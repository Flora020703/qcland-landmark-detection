#!/usr/bin/env python3
"""
scripts/visualize_attention.py

Visualization diagnostic for EoMT fetal biometry landmark detection.
Checks whether the model attends to yellow text annotations vs. anatomy.

Per test image generates two figures:
  {stem}_diagnosis_masked.png   — masked_attn_enabled=True  (training state)
  {stem}_diagnosis_unmasked.png — masked_attn_enabled=False (free attention)

Each figure layout (2 × 3):
  Row 0: [Original + GT/Pred]  [Pred Heatmap Q1]   [Pred Heatmap Q2]
  Row 1: [Decoder Attn Q1]     [Decoder Attn Q2]    [Backbone CLS Self-Attn]

Token layout reference (vit_small_patch14_reg4_dinov2, num_q=2):
  Pre-query blocks : [CLS, reg×4, patch×M]         num_prefix=5, M=grid_h×grid_w
  Last block       : [q0, q1, CLS, reg×4, patch×M] queries prepended at block injection

Hook strategy:
  - Set fused_attn=False on target blocks → explicit qk→softmax→attn_drop path
  - Register forward_hook on attn_drop (nn.Dropout) to capture softmax attention weights
  - In eval mode, attn_drop is a no-op so hook output == softmax attention

Usage:
    python scripts/visualize_attention.py \\
        --config configs/landmark/bpd_vit_small.yaml \\
        --ckpt_path /root/eomt/logs/eomt-landmark/atp44qz2/checkpoints/seed2024_best.ckpt \\
        --output_dir visualizations/gradcam_diagnosis \\
        --num_images 10
"""

import argparse
import os
import sys

# Project root on path so imports work regardless of cwd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

from models.vit import ViT
from models.eomt import EoMT
from training.landmark_detection import (
    LandmarkDetection, heatmap_to_coords, compute_nme,
)
from datasets.landmark_dataset import HeadLandmarkDataModule


# -----------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------

def load_model(config_path: str, ckpt_path: str) -> LandmarkDetection:
    """
    Instantiate LandmarkDetection from the YAML config, then load checkpoint
    weights via state_dict.  Cannot use load_from_checkpoint() because
    'network' is excluded from save_hyperparameters().
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    d = cfg["data"]["init_args"]
    m = cfg["model"]["init_args"]
    n = m["network"]["init_args"]
    e = n["encoder"]["init_args"]

    img_size     = tuple(d["img_size"])
    heatmap_size = tuple(d["heatmap_size"])

    encoder = ViT(img_size=img_size, backbone_name=e["backbone_name"])

    network = EoMT(
        encoder=encoder,
        num_classes=n["num_classes"],
        num_q=n["num_q"],
        num_blocks=n["num_blocks"],
        masked_attn_enabled=n["masked_attn_enabled"],
        freeze_backbone=n.get("freeze_backbone", False),
        upsample_bilinear=n.get("upsample_bilinear", False),
        use_refinement_head=n.get("use_refinement_head", False),
        heatmap_head=n.get("heatmap_head", "einsum"),
        heatmap_size=tuple(n.get("heatmap_size", [64, 64])),
    )

    model = LandmarkDetection(
        network=network,
        img_size=img_size,
        num_landmarks=m["num_landmarks"],
        heatmap_size=heatmap_size,
        loss_type=m.get("loss_type", "hybrid"),
        alpha=m.get("alpha", 5.0),
        temperature=m.get("temperature", 10.0),
        lambda_coord=m.get("lambda_coord", 0.1),
        lambda_awing=m.get("lambda_awing", 0.01),
        awing_omega=m.get("awing_omega", 14.0),
        awing_theta=m.get("awing_theta", 0.5),
        awing_alpha=m.get("awing_alpha", 2.1),
        awing_epsilon=m.get("awing_epsilon", 1.0),
        lr=m.get("lr", 1e-4),
        llrd=m.get("llrd", 0.8),
        warmup_steps=m.get("warmup_steps", [15, 30]),
    )

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


# -----------------------------------------------------------------------
# Attention hooks
# -----------------------------------------------------------------------

def setup_hooks(model: LandmarkDetection):
    """
    Register attn_drop hooks on two backbone blocks.

    Hooks only fire when fused_attn=False (explicit qk path), so we disable
    fused_attn on the two target blocks.  All other blocks are unaffected.

    Returns:
        storage  : dict cleared & populated during each forward call
        handles  : list of hook handles — call h.remove() when done
        meta     : dict with grid_h, grid_w, num_q, num_prefix
    """
    backbone   = model.network.encoder.backbone
    num_blocks = model.network.num_blocks           # EoMT decoder depth (e.g. 3)
    n_bb       = len(backbone.blocks)               # total backbone blocks (e.g. 12)

    pre_q_idx = n_bb - num_blocks - 1              # last block before query injection
    last_idx  = n_bb - 1                           # final block (queries present)

    backbone.blocks[pre_q_idx].attn.fused_attn = False
    backbone.blocks[last_idx].attn.fused_attn  = False

    if hasattr(backbone, "rope_embeddings"):
        print("[WARN] Backbone has RoPE — hooks may not fire for RoPE path.")

    storage = {}

    def make_hook(name):
        def hook(module, inp, out):
            storage[name] = out.detach().cpu()   # (B, heads, N, N)
        return hook

    handles = [
        backbone.blocks[pre_q_idx].attn.attn_drop.register_forward_hook(
            make_hook("backbone_self_attn")
        ),
        backbone.blocks[last_idx].attn.attn_drop.register_forward_hook(
            make_hook("decoder_attn")
        ),
    ]

    num_prefix = getattr(backbone, "num_prefix_tokens", 1)  # CLS + register tokens
    grid_h, grid_w = backbone.patch_embed.grid_size

    print(f"[Hooks] pre-query block={pre_q_idx}, last block={last_idx}, "
          f"num_prefix={num_prefix}, patch grid={grid_h}×{grid_w}")

    meta = dict(
        grid_h=grid_h, grid_w=grid_w,
        num_q=model.network.num_q,
        num_prefix=num_prefix,
    )
    return storage, handles, meta


# -----------------------------------------------------------------------
# Visualisation helpers
# -----------------------------------------------------------------------

def upsample_to(arr_2d: np.ndarray, h: int, w: int) -> np.ndarray:
    """Bilinearly upsample a 2-D [0,1] float array to (h, w)."""
    pil = Image.fromarray((arr_2d * 255).clip(0, 255).astype(np.uint8))
    return np.array(pil.resize((w, h), Image.BILINEAR), dtype=np.float32) / 255.0


def to_spatial(attn_1d: torch.Tensor, gh: int, gw: int) -> np.ndarray:
    """Flatten patch attention (M,) → normalised 2-D map (gh, gw)."""
    arr = attn_1d.float().numpy().reshape(gh, gw)
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-8)


def overlay_ax(ax, img_np: np.ndarray, map_2d: np.ndarray, title: str,
               cmap: str = "jet"):
    """Draw RGB image with a semi-transparent colourmap overlay."""
    h, w = img_np.shape[:2]
    up = upsample_to(map_2d, h, w)
    ax.imshow(img_np)
    ax.imshow(up, cmap=cmap, alpha=0.5, vmin=0, vmax=1)
    ax.set_title(title, fontsize=8)
    ax.axis("off")


# -----------------------------------------------------------------------
# Single forward + 2×3 figure
# -----------------------------------------------------------------------

def run_and_plot(
    img_tensor: torch.Tensor,   # (3, H, W) float [0,1]
    gt_coords: torch.Tensor,    # (N, 2) in heatmap pixel space
    model: LandmarkDetection,
    storage: dict,
    meta: dict,
    device: torch.device,
    mode_label: str,            # "masked" | "unmasked"
) -> plt.Figure:

    hm_size  = model.heatmap_size   # (hm_H, hm_W)
    img_size = model.img_size       # (H, W)
    num_q    = meta["num_q"]
    num_pfx  = meta["num_prefix"]
    gh, gw   = meta["grid_h"], meta["grid_w"]

    storage.clear()

    with torch.no_grad():
        img_b = img_tensor.unsqueeze(0).to(device)
        mask_logits_per_layer, _ = model(img_b)
        pred_hm = F.interpolate(
            mask_logits_per_layer[-1], hm_size,
            mode="bilinear", align_corners=False,
        ).squeeze(0).cpu()   # (N, hm_H, hm_W)

    # NME
    pred_coords = heatmap_to_coords(pred_hm.unsqueeze(0))          # (1, N, 2)
    nme_val     = compute_nme(
        pred_coords, gt_coords.unsqueeze(0), hm_size, img_size
    )[0].item()

    # Landmark positions in image-pixel space
    ih, iw   = img_size
    hm_h, hm_w = hm_size
    sx, sy   = iw / hm_w, ih / hm_h

    gt_xy   = gt_coords.numpy()
    pred_xy = pred_coords.squeeze(0).numpy()
    gt_x,   gt_y   = gt_xy[:, 0] * sx,   gt_xy[:, 1] * sy
    pred_x, pred_y = pred_xy[:, 0] * sx, pred_xy[:, 1] * sy

    # ---- Attention maps ----

    # Backbone self-attention (pre-query block)
    # Token layout: [CLS=0, reg×(num_pfx-1), patch×M]
    # CLS-to-patch: storage[0, :, 0, num_pfx:].mean(heads)
    b_raw = storage.get("backbone_self_attn")
    if b_raw is not None:
        backbone_map = to_spatial(b_raw[0, :, 0, num_pfx:].mean(0), gh, gw)
    else:
        print(f"  [WARN] backbone_self_attn not captured ({mode_label})")
        backbone_map = np.zeros((gh, gw))

    # Decoder query attention (last block)
    # Token layout: [q0, q1, CLS, reg×(num_pfx-1), patch×M]
    # query q → patch: storage[0, :, q, num_q+num_pfx:].mean(heads)
    d_raw = storage.get("decoder_attn")
    q_maps = []
    for q in range(num_q):
        if d_raw is not None:
            q_maps.append(to_spatial(d_raw[0, :, q, num_q + num_pfx:].mean(0), gh, gw))
        else:
            print(f"  [WARN] decoder_attn not captured for Q{q+1} ({mode_label})")
            q_maps.append(np.zeros((gh, gw)))

    # ---- Build 2×3 figure ----
    img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    hm_np  = pred_hm.numpy()   # (N, hm_H, hm_W)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(
        f"[{mode_label}]   NME = {nme_val * 100:.2f}%",
        fontsize=11, fontweight="bold",
    )

    # Row 0, Col 0 — original image + GT and predicted landmark positions
    axes[0, 0].imshow(img_np)
    axes[0, 0].scatter(gt_x, gt_y, c="lime", s=90, marker="o",
                       label="GT", zorder=5, edgecolors="black", linewidths=0.5)
    axes[0, 0].scatter(pred_x, pred_y, c="red", s=90, marker="x",
                       label="Pred", zorder=5, linewidths=2)
    axes[0, 0].legend(fontsize=7, loc="upper right")
    axes[0, 0].set_title("Original + GT (green) / Pred (red)", fontsize=8)
    axes[0, 0].axis("off")

    # Row 0, Col 1-2 — predicted heatmap overlays (per-channel normalised)
    for q in range(num_q):
        hm_q  = hm_np[q]
        hm_n  = (hm_q - hm_q.min()) / (hm_q.max() - hm_q.min() + 1e-8)
        hm_up = upsample_to(hm_n, ih, iw)
        axes[0, q + 1].imshow(img_np)
        axes[0, q + 1].imshow(hm_up, cmap="hot", alpha=0.6, vmin=0, vmax=1)
        axes[0, q + 1].set_title(f"Pred Heatmap Q{q + 1}", fontsize=8)
        axes[0, q + 1].axis("off")

    # Row 1, Col 0-1 — decoder query-to-patch attention
    for q in range(num_q):
        overlay_ax(axes[1, q], img_np, q_maps[q],
                   f"Decoder Attn Q{q + 1}  [{mode_label}]", cmap="jet")

    # Row 1, Col 2 — backbone CLS self-attention
    overlay_ax(axes[1, 2], img_np, backbone_map,
               f"Backbone CLS Self-Attn  [{mode_label}]", cmap="jet")

    plt.tight_layout()
    return fig


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="EoMT attention visualisation — text annotation interference diagnosis"
    )
    parser.add_argument("--config",     required=True)
    parser.add_argument("--ckpt_path",  required=True)
    parser.add_argument("--output_dir", default="visualizations/gradcam_diagnosis")
    parser.add_argument("--num_images", type=int, default=10)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # --- Load model ---
    print(f"\nLoading model: {args.ckpt_path}")
    model = load_model(args.config, args.ckpt_path)
    model = model.to(device)
    model.eval()

    # --- Hooks ---
    print("Setting up attention hooks ...")
    storage, handles, meta = setup_hooks(model)

    # --- Test dataset ---
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    d = cfg["data"]["init_args"]

    dm = HeadLandmarkDataModule(
        images_dir    = d["images_dir"],
        ann_train_csv = d["ann_train_csv"],
        ann_test_csv  = d["ann_test_csv"],
        task          = d["task"],
        img_size      = tuple(d["img_size"]),
        heatmap_size  = tuple(d["heatmap_size"]),
        sigma         = d["sigma"],
        val_fraction  = d.get("val_fraction", 0.1),
        val_split_seed= d.get("val_split_seed", 42),
        batch_size    = 1,
        num_workers   = 0,
        pin_memory    = False,
    )
    dm.setup()
    test_ds      = dm.test_dataset
    test_records = test_ds.records

    n = min(args.num_images, len(test_ds))
    print(f"\nVisualising {n}/{len(test_ds)} test images → {out_dir}/\n")

    for i in range(n):
        img_tensor, gt_heatmaps, gt_coords = test_ds[i]
        img_path = Path(test_records[i]["img_path"])
        stem = img_path.stem
        print(f"[{i+1:02d}/{n}] {img_path.name}")

        # ---- Masked figure (training state) ----
        model.network.masked_attn_enabled = True
        fig_m = run_and_plot(img_tensor, gt_coords, model, storage, meta,
                             device, "masked")
        fig_m.savefig(out_dir / f"{stem}_diagnosis_masked.png",
                      dpi=150, bbox_inches="tight")
        plt.close(fig_m)

        # ---- Unmasked figure (free attention) ----
        model.network.masked_attn_enabled = False
        fig_u = run_and_plot(img_tensor, gt_coords, model, storage, meta,
                             device, "unmasked")
        fig_u.savefig(out_dir / f"{stem}_diagnosis_unmasked.png",
                      dpi=150, bbox_inches="tight")
        plt.close(fig_u)

        model.network.masked_attn_enabled = True   # restore to training state
        print(f"       saved: {stem}_diagnosis_masked.png")
        print(f"       saved: {stem}_diagnosis_unmasked.png")

    for h in handles:
        h.remove()

    print(f"\nDone. {n * 2} figures in {out_dir}/")


if __name__ == "__main__":
    main()
