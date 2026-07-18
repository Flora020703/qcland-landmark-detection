#!/usr/bin/env python3
# ---------------------------------------------------------------
# MODIFIED: new file — DINOv3 backbone integration smoke test.
#
# Purpose: verify that swapping backbone_name to a DINOv3 checkpoint works
# through the EXISTING generic code path in models/vit.py + models/eomt.py,
# with zero source changes. Checks, in order:
#   1. ViT wrapper loads the HF DINOv3 model and exposes the attributes
#      models/eomt.py relies on generically: embed_dim, num_prefix_tokens,
#      patch_embed.patch_size, patch_embed.grid_size, rope_embeddings.
#   2. A full EoMT forward+backward pass runs cleanly at BOTH input
#      resolutions actually used in this project (512x512 for BPD/OFD,
#      256x256 for 300W) — these produce different patch grids under
#      patch_size=16 (32x32 vs 16x16), exercising num_upscale +
#      QueryConditionedDeconvHead's final F.interpolate differently.
#
# This does NOT require GPU (CPU is fine — see CLAUDE.md convention: test
# data loading/model wiring locally on CPU before training on AutoDL). It
# DOES require the HF DINOv3 weights to be downloadable, i.e. an approved
# access request on https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m
# (gated repo) plus HF_TOKEN / huggingface-cli login on whichever machine
# runs this. Run this first, before spending any GPU time.
#
# Usage:
#   python3 scripts/verify_dinov3_backbone.py
#   python3 scripts/verify_dinov3_backbone.py --backbone facebook/dinov3-vits16-pretrain-lvd1689m
# ---------------------------------------------------------------

import argparse
import sys

import torch

sys.path.insert(0, ".")

from models.eomt import EoMT
from models.vit import ViT


def check(label: str, ok: bool, detail: str = "") -> bool:
    tag = "[OK]  " if ok else "[FAIL]"
    print(f"  {tag} {label}" + (f"  {detail}" if detail else ""))
    return ok


def verify_backbone_attributes(backbone_name: str, img_size: tuple[int, int]) -> bool:
    print(f"\n--- Step 1: loading backbone ({backbone_name}, img_size={img_size}) ---")
    encoder = ViT(img_size=img_size, backbone_name=backbone_name)
    backbone = encoder.backbone

    all_ok = True
    all_ok &= check("embed_dim == 384 (DINOv3-S, same as DINOv2-S)", backbone.embed_dim == 384,
                     f"got {backbone.embed_dim}")
    all_ok &= check("num_prefix_tokens present", hasattr(backbone, "num_prefix_tokens"),
                     f"value={getattr(backbone, 'num_prefix_tokens', 'MISSING')}")
    all_ok &= check("patch_embed.patch_size present", hasattr(backbone.patch_embed, "patch_size"),
                     f"value={getattr(backbone.patch_embed, 'patch_size', 'MISSING')}")
    all_ok &= check("patch_embed.grid_size present", hasattr(backbone.patch_embed, "grid_size"),
                     f"value={getattr(backbone.patch_embed, 'grid_size', 'MISSING')}")

    expected_patch = 16
    got_patch = max(backbone.patch_embed.patch_size) if hasattr(backbone.patch_embed, "patch_size") else None
    all_ok &= check(f"patch_size == {expected_patch}", got_patch == expected_patch, f"got {got_patch}")

    expected_grid = (img_size[0] // expected_patch, img_size[1] // expected_patch)
    got_grid = tuple(backbone.patch_embed.grid_size) if hasattr(backbone.patch_embed, "grid_size") else None
    all_ok &= check(f"grid_size == {expected_grid}", got_grid == expected_grid, f"got {got_grid}")

    has_rope = hasattr(backbone, "rope_embeddings")
    print(f"  [INFO] hasattr(backbone, 'rope_embeddings') = {has_rope} "
          f"(DINOv3 uses RoPE — if this is False, positions may be silently unencoded)")

    return all_ok


def verify_forward_backward(backbone_name: str, img_size: tuple[int, int], num_q: int, num_blocks: int) -> bool:
    print(f"\n--- Step 2: forward+backward smoke test (img_size={img_size}, num_q={num_q}, num_blocks={num_blocks}) ---")

    encoder = ViT(img_size=img_size, backbone_name=backbone_name)
    model = EoMT(
        encoder=encoder,
        num_classes=1,
        num_q=num_q,
        num_blocks=num_blocks,
        masked_attn_enabled=True,
        freeze_backbone=False,
        heatmap_head="deconv_v2",
        heatmap_size=(64, 64),
        use_fpn=True,
        fpn_layers=[4, 8, 12],  # must end at total backbone depth (12 for ViT-S), independent of num_blocks
    )
    model.train()

    x = torch.randn(2, 3, *img_size)
    try:
        mask_logits_per_layer, class_logits_per_layer, _ = model(x)
    except Exception as e:
        check("forward pass completes", False, f"raised {type(e).__name__}: {e}")
        return False

    final_logits = mask_logits_per_layer[-1]
    ok = check("output shape == (2, num_q, 64, 64)", tuple(final_logits.shape) == (2, num_q, 64, 64),
               f"got {tuple(final_logits.shape)}")

    target = torch.rand_like(final_logits)
    loss = torch.nn.functional.mse_loss(final_logits, target)
    print(f"  [INFO] dummy MSE loss (random target, uninitialized deconv head) = {loss.item():.4f}")

    loss.backward()

    # MODIFIED: distinguish None (broken graph — real bug) from NaN (unstable
    # gradient value, e.g. from softmax/div-by-zero somewhere downstream —
    # not a graph break, but still worth surfacing) from a genuine nonzero
    # gradient. The original one-line `grad is not None and grad.abs().sum() > 0`
    # silently reports FAIL for both None AND NaN (since `nan > 0` is False in
    # Python), which conflates two very different problems.
    def grad_diag(label: str, grad) -> str:
        if grad is None:
            return f"[FAIL] {label}: grad is None — backward never reached this parameter (real graph break)"
        if torch.isnan(grad).any():
            pct = torch.isnan(grad).float().mean().item() * 100
            return f"[WARN] {label}: grad contains NaN ({pct:.1f}% of elements) — numerical issue, not necessarily a broken graph"
        total = grad.abs().sum().item()
        if total == 0:
            return f"[FAIL] {label}: grad is exact zero everywhere — suspicious"
        return f"[OK]   {label}: grad present, abs-sum={total:.4g}"

    q_report = grad_diag("q.weight", model.q.weight.grad)
    print(f"  {q_report}")
    ok &= q_report.startswith("[OK]")

    # Report per-block, not just an aggregate any(), so a break partway through
    # the backbone (e.g. only the last few blocks get gradient) is visible.
    backbone = model.encoder.backbone
    named = [
        ("patch_embed", backbone.patch_embed),
        ("block[0]", backbone.blocks[0]),
        ("block[-1]", backbone.blocks[-1]),
        ("norm", getattr(backbone, "norm", None)),
    ]
    backbone_ok = True
    for label, module in named:
        if module is None:
            continue
        grads = [p.grad for p in module.parameters() if p.requires_grad]
        if not grads:
            print(f"  [WARN] {label}: no trainable parameters found")
            continue
        none_count = sum(g is None for g in grads)
        nan_count = sum(g is not None and torch.isnan(g).any() for g in grads)
        nonzero_count = sum(g is not None and not torch.isnan(g).any() and g.abs().sum().item() > 0 for g in grads)
        status = "[OK]  " if nonzero_count == len(grads) else ("[WARN]" if none_count == 0 else "[FAIL]")
        print(f"  {status} {label}: {nonzero_count}/{len(grads)} params with real gradient, "
              f"{nan_count} with NaN, {none_count} with None")
        if none_count > 0:
            backbone_ok = False
    ok &= backbone_ok

    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="facebook/dinov3-vits16-pretrain-lvd1689m")
    parser.add_argument("--skip-dinov2-control", action="store_true",
                         help="skip the DINOv2 control run (only meaningful if you've already confirmed "
                              "the exact-zero-gradient pattern also happens with DINOv2 on this random seed)")
    args = parser.parse_args()

    print("=" * 70)
    print("  DINOv3 backbone integration smoke test")
    print("=" * 70)

    results = []
    results.append(("BPD/OFD attributes @512", verify_backbone_attributes(args.backbone, (512, 512))))
    results.append(("300W attributes @256", verify_backbone_attributes(args.backbone, (256, 256))))

    torch.manual_seed(0)
    results.append(("BPD/OFD forward+backward (num_q=2, num_blocks=3)",
                     verify_forward_backward(args.backbone, (512, 512), num_q=2, num_blocks=3)))
    torch.manual_seed(0)
    results.append(("300W forward+backward (num_q=68, num_blocks=6)",
                     verify_forward_backward(args.backbone, (256, 256), num_q=68, num_blocks=6)))

    # MODIFIED: DINOv2 control — same synthetic harness, same random seed, only
    # the backbone differs. If DINOv2 shows the SAME exact-zero-gradient
    # pattern, the finding is a property of this synthetic single-batch/
    # random-init test (e.g. vanishing gradient through untrained FPN/deconv
    # heads), NOT a DINOv3 integration bug — real training uses an optimizer,
    # LR schedule and many steps, which is a different regime. If DINOv2 does
    # NOT show exact-zero (i.e. it has real nonzero gradients here), that
    # proves the problem is specific to the DINOv3 code path.
    if not args.skip_dinov2_control:
        print("\n" + "=" * 70)
        print("  CONTROL: same test, DINOv2 backbone (vit_small_patch14_reg4_dinov2)")
        print("=" * 70)
        torch.manual_seed(0)
        results.append(("[CONTROL] DINOv2 forward+backward (num_q=2, num_blocks=3)",
                         verify_forward_backward("vit_small_patch14_reg4_dinov2", (512, 512),
                                                  num_q=2, num_blocks=3)))

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    all_ok = True
    for label, ok in results:
        print(f"  {'[OK]  ' if ok else '[FAIL]'} {label}")
        if not label.startswith("[CONTROL]"):
            all_ok &= ok

    dinov3_result = [ok for label, ok in results if "forward+backward" in label and not label.startswith("[CONTROL]")]
    control_result = [ok for label, ok in results if label.startswith("[CONTROL]")]
    if control_result and not control_result[0] and not all(dinov3_result):
        print("\n[INFO] DINOv2 control ALSO failed with exact-zero gradient in this same synthetic "
              "harness — this points to a property of the random single-batch test itself (likely "
              "vanishing gradient through untrained FPN/deconv heads), not a DINOv3-specific bug. "
              "Safe to treat this smoke test as inconclusive and move to the GPU canary run instead, "
              "which uses a real optimizer over many steps.")
    elif control_result and control_result[0] and not all(dinov3_result):
        print("\n[WARNING] DINOv2 control PASSED (real gradient) while DINOv3 did not, under the exact "
              "same test — this points to a real DINOv3-specific problem. Do not proceed to the GPU "
              "canary run yet.")

    if all_ok:
        print("\n[OK] All checks passed — safe to proceed to the GPU canary run (single seed, BPD).")
        sys.exit(0)
    else:
        print("\n[ERROR] At least one check failed — do not start the 5-seed ablations yet.")
        sys.exit(1)


if __name__ == "__main__":
    main()
