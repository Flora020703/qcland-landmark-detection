#!/usr/bin/env python3
# ---------------------------------------------------------------
# MODIFIED: new file — narrow diagnostic for the exact-zero-gradient
# finding from verify_dinov3_backbone.py. Isolates the single call site
# in models/eomt.py's _attn() that is unique to DINOv3 (the `rope is not
# None` branch, which calls the HF attention module directly instead of
# manually reimplementing qkv like the DINOv2 path does):
#
#   return module(x, mask, rope)[0]
#
# This script prints the real forward() signature of that module and
# runs an isolated forward+backward on just the attention submodule
# (no EoMT wrapper, no deconv head, no masked-attention logic) to see
# whether gradient reaches block.attention.* parameters at all, and
# whether that depends on the (mask, rope) call convention used above.
#
# Usage (on the server, same env as verify_dinov3_backbone.py):
#   python3 scripts/debug_dinov3_attn.py
# ---------------------------------------------------------------

import inspect
import sys

import torch

sys.path.insert(0, ".")

from models.vit import ViT


def main():
    backbone_name = "facebook/dinov3-vits16-pretrain-lvd1689m"
    img_size = (512, 512)

    print(f"Loading {backbone_name} @ {img_size} ...")
    encoder = ViT(img_size=img_size, backbone_name=backbone_name)
    backbone = encoder.backbone

    block0 = backbone.blocks[0]
    attn_module = block0.attn if hasattr(block0, "attn") else block0.attention
    attr_name = "attn" if hasattr(block0, "attn") else "attention"
    print(f"\nblock[0].{attr_name} is a {type(attn_module).__name__}")

    print("\n--- forward() signature ---")
    try:
        sig = inspect.signature(attn_module.forward)
        print(f"  {sig}")
        for name, param in sig.parameters.items():
            print(f"    {name}: default={param.default!r}")
    except (TypeError, ValueError) as e:
        print(f"  could not introspect signature: {e}")

    print("\n--- module.__call__ signature (what actually gets invoked) ---")
    try:
        print(f"  {inspect.signature(attn_module.__call__)}")
    except (TypeError, ValueError):
        pass

    # ------------------------------------------------------------------
    # Isolated forward+backward: same call convention eomt.py's _attn()
    # uses for the rope branch: module(x, mask, rope)[0]
    # ------------------------------------------------------------------
    print("\n--- Isolated attention forward+backward (mirroring eomt.py's call) ---")
    B, N, C = 2, 5 + 32 * 32, backbone.embed_dim  # 5 prefix tokens + 32x32 patch grid @512
    num_heads = attn_module.num_heads if hasattr(attn_module, "num_heads") else None
    print(f"  num_heads = {num_heads}")

    x = torch.randn(B, N, C, requires_grad=True)

    rope = None
    if hasattr(backbone, "rope_embeddings"):
        dummy_img = torch.randn(B, 3, *img_size)
        rope = backbone.rope_embeddings(dummy_img)
        print(f"  rope type: {type(rope)}"
              + (f", len={len(rope)}" if isinstance(rope, (tuple, list)) else ""))
        if isinstance(rope, (tuple, list)):
            for i, r in enumerate(rope):
                if torch.is_tensor(r):
                    print(f"    rope[{i}]: shape={tuple(r.shape)}, requires_grad={r.requires_grad}")

    mask = torch.ones(B, N, N, dtype=torch.bool)
    mask_expanded = mask[:, None, ...].expand(-1, num_heads, -1, -1) if num_heads else mask

    print("\n  Calling: attn_module(x, mask_expanded, rope) ...")
    try:
        out = attn_module(x, mask_expanded, rope)
        out_tensor = out[0] if isinstance(out, (tuple, list)) else out
        print(f"  output shape: {tuple(out_tensor.shape)}, requires_grad={out_tensor.requires_grad}, "
              f"grad_fn={out_tensor.grad_fn}")

        loss = out_tensor.sum()
        loss.backward()

        x_grad_ok = x.grad is not None and x.grad.abs().sum().item() > 0
        print(f"  d(loss)/d(x): {'present, abs-sum=' + str(x.grad.abs().sum().item()) if x.grad is not None else 'None'}")

        param_report = []
        for name, p in attn_module.named_parameters():
            if p.grad is None:
                param_report.append((name, "None"))
            elif p.grad.abs().sum().item() == 0:
                param_report.append((name, "exact zero"))
            else:
                param_report.append((name, f"OK abs-sum={p.grad.abs().sum().item():.4g}"))
        print("\n  Per-parameter gradient status:")
        for name, status in param_report:
            print(f"    {name}: {status}")

    except Exception as e:
        print(f"  [ERROR] direct call with (x, mask, rope) raised {type(e).__name__}: {e}")
        print("  This confirms a call-signature mismatch — see the forward() signature printed above")
        print("  to find the correct argument names/order, then compare against models/eomt.py's _attn().")

    # ------------------------------------------------------------------
    # Same test but with mask=None, to isolate whether the boolean mask
    # tensor specifically is what's breaking gradient (vs rope, vs
    # something else entirely).
    # ------------------------------------------------------------------
    print("\n--- Same test with mask=None (isolate mask vs rope) ---")
    x2 = torch.randn(B, N, C, requires_grad=True)
    try:
        out2 = attn_module(x2, None, rope)
        out2_tensor = out2[0] if isinstance(out2, (tuple, list)) else out2
        out2_tensor.sum().backward()
        any_grad = any(p.grad is not None and p.grad.abs().sum().item() > 0 for p in attn_module.parameters())
        print(f"  with mask=None: any nonzero param grad = {any_grad}")
    except Exception as e:
        print(f"  [ERROR] raised {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
