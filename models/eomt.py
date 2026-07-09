# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# Portions of this file are adapted from the timm library by Ross Wightman,
# used under the Apache 2.0 License.
# ---------------------------------------------------------------

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from models.scale_block import ScaleBlock


class RefinementHead(nn.Module):
    """
    Lightweight per-heatmap sharpening network (shared weights across all landmarks).
    Input/output: (B, Q, H, W). Internally reshapes to (B*Q, 1, H, W) so a single
    forward pass handles all queries with identical weights, then adds a residual.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, Q, H, W = x.shape
        flat = x.reshape(B * Q, 1, H, W)
        return (self.net(flat) + flat).reshape(B, Q, H, W)


class QueryConditionedDeconvHead(nn.Module):
    """
    V2 heatmap head: FiLM-conditioned patch features → per-landmark heatmaps.

    Replaces the einsum dot-product similarity map with learned, query-specific
    spatial processing.  Patch features carry spatial structure; query tokens
    select which landmark to localise via FiLM modulation.

    Input:
      query_tokens  : (B, Q, C) — raw last-block query tokens (NOT mask_head output)
      patch_features: (B, C, H_f, W_f) — post-upscale patch features from self.upscale
                      e.g. (B, 384, 72, 72) for 512×512 input with patch_size=14

    Output: (B, Q, hm_H, hm_W) — same interface as the einsum path.

    Design notes:
      • GroupNorm on patch features prevents large-magnitude channels from
        overwhelming FiLM modulation (ViT + ScaleBlock outputs are unbounded).
      • All Q queries are processed in a single Conv pass via (B*Q, C, H, W)
        reshape — no per-query loop.
      • Final F.interpolate handles the 72→64 (or any non-power-of-2) resize.
    """

    def __init__(
        self,
        embed_dim: int = 384,
        num_queries: int = 2,
        heatmap_size: tuple = (64, 64),
        hidden_ch: int = 128,
    ):
        super().__init__()
        self.heatmap_size = heatmap_size

        # FiLM generator: query → (γ, β) for channel-wise modulation of patch features
        # MODIFIED: dropout between the two Linears to reduce overfitting (deconv_v2
        # val NME improved on avg but variance rose across seeds — see 5-seed ablation)
        self.film_generator = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(embed_dim, embed_dim * 2),   # γ and β concatenated
        )

        # Normalise patch features before FiLM to stabilise modulation scale
        self.norm = nn.GroupNorm(32, embed_dim)

        # Shared spatial processing — same weights applied to every query's modulated features
        self.conv1 = nn.Conv2d(embed_dim, hidden_ch, kernel_size=3, padding=1)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(hidden_ch, 1, kernel_size=3, padding=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # Override conv2 to produce a small positive constant at init.
        #
        # Why not zero? _attn_mask() uses (interpolated > 0) to build the attention mask.
        # Zero-init → all-False mask → queries can't attend to ANY patch → learning blocked.
        #
        # Why not kaiming? kaiming on conv2 gives output std ≈ 22 (see chain: GroupNorm →
        # FiLM → conv1 → ReLU → conv2). WMSE on pred≈22 vs target∈[0,1] → loss≈266,
        # which distorts the entire gradient trajectory for hundreds of epochs.
        #
        # Small positive constant (0.01): loss starts at ~0.0001 per pixel (vs 266),
        # attention mask sees 0.01 > 0 everywhere → fully open, and gradients from
        # WMSE + coord loss naturally push the head toward correct peaks.
        nn.init.zeros_(self.conv2.weight)
        nn.init.constant_(self.conv2.bias, 0.01)

    def forward(
        self,
        query_tokens: torch.Tensor,    # (B, Q, C)
        patch_features: torch.Tensor,  # (B, C, H_f, W_f)
    ) -> torch.Tensor:
        B, Q, C = query_tokens.shape
        H, W = patch_features.shape[-2:]

        # FiLM params: (B, Q, 2C) → γ and β each (B, Q, C, 1, 1)
        film   = self.film_generator(query_tokens)
        gamma  = film[..., :C].unsqueeze(-1).unsqueeze(-1)
        beta   = film[..., C:].unsqueeze(-1).unsqueeze(-1)

        # GroupNorm then broadcast to Q queries: (B, 1, C, H, W) × (B, Q, C, 1, 1)
        normed    = self.norm(patch_features).unsqueeze(1)   # (B, 1, C, H, W)
        modulated = (gamma * normed + beta).reshape(B * Q, C, H, W)

        # Single Conv pass over all queries: (B*Q, 1, H, W)
        out = self.conv2(self.relu(self.conv1(modulated)))

        # Resize to target heatmap resolution (handles non-integer scale factors)
        out = F.interpolate(out, self.heatmap_size, mode="bilinear", align_corners=False)

        return out.reshape(B, Q, *self.heatmap_size)


class EoMT(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        num_classes,
        num_q,
        num_blocks=4,
        masked_attn_enabled=True,
        freeze_backbone: bool = True,       # MODIFIED: freeze pretrained DINOv2 backbone weights
        upsample_bilinear: bool = False,    # MODIFIED: bilinear resize+conv instead of ConvTranspose2d
        use_refinement_head: bool = False,  # MODIFIED: lightweight Conv refinement after einsum
        heatmap_head: str = "einsum",       # MODIFIED: "einsum" | "deconv_v2"
        heatmap_size: tuple = (64, 64),     # MODIFIED: target heatmap size for deconv heads
    ):
        super().__init__()
        self.encoder = encoder
        self.num_q = num_q
        self.num_blocks = num_blocks
        self.masked_attn_enabled = masked_attn_enabled

        self.register_buffer("attn_mask_probs", torch.ones(num_blocks))

        self.q = nn.Embedding(num_q, self.encoder.backbone.embed_dim)

        self.class_head = nn.Linear(self.encoder.backbone.embed_dim, num_classes + 1)

        self.mask_head = nn.Sequential(
            nn.Linear(self.encoder.backbone.embed_dim, self.encoder.backbone.embed_dim),
            nn.GELU(),
            nn.Linear(self.encoder.backbone.embed_dim, self.encoder.backbone.embed_dim),
            nn.GELU(),
            nn.Linear(self.encoder.backbone.embed_dim, self.encoder.backbone.embed_dim),
        )

        patch_size = encoder.backbone.patch_embed.patch_size
        max_patch_size = max(patch_size[0], patch_size[1])
        num_upscale = max(1, int(math.log2(max_patch_size)) - 2)

        self.upscale = nn.Sequential(
            *[ScaleBlock(self.encoder.backbone.embed_dim, use_bilinear=upsample_bilinear)
              for _ in range(num_upscale)],
        )

        self.refinement_head = RefinementHead() if use_refinement_head else None

        # MODIFIED: learnable deconv heatmap head (replaces einsum when heatmap_head != "einsum")
        self.heatmap_head = heatmap_head
        self.deconv_head = (
            QueryConditionedDeconvHead(
                embed_dim=self.encoder.backbone.embed_dim,
                num_queries=num_q,
                heatmap_size=heatmap_size,
            )
            if heatmap_head == "deconv_v2"
            else None
        )

        # MODIFIED: direct coord regression head (diagnostic — bypasses heatmap entirely).
        # query_tokens (B,Q,C) → Linear → (B,Q,2) in heatmap pixel space.
        # Used when heatmap_head=="coord_direct".  Intermediate layers still use
        # einsum+loss to train the backbone; only the final prediction uses this head.
        self.coord_direct_head = (
            nn.Linear(self.encoder.backbone.embed_dim, 2)
            if heatmap_head == "coord_direct"
            else None
        )
        # Why 2 (not num_q*2): applied to q_final (B,Q,C), nn.Linear acts on last dim
        # → output (B,Q,2); each query independently regresses its own (x,y).

        # MODIFIED: partial freeze — lock first 75% of transformer blocks (early generic features),
        # keep last 25% + norm trainable so high-level features can adapt to landmark task.
        if freeze_backbone:
            backbone = self.encoder.backbone
            n_blocks = len(backbone.blocks)
            freeze_until = int(n_blocks * 0.75)   # e.g. 9 of 12 for ViT-S

            for p in backbone.parameters():        # freeze everything first
                p.requires_grad = False

            for i in range(freeze_until, n_blocks):  # unfreeze last 25% blocks
                for p in backbone.blocks[i].parameters():
                    p.requires_grad = True

            if hasattr(backbone, "norm"):           # unfreeze final norm
                for p in backbone.norm.parameters():
                    p.requires_grad = True

    def _predict_einsum(self, x: torch.Tensor):
        """
        Einsum-based prediction, always used for intermediate decoder layers.

        Regardless of self.heatmap_head, intermediate predictions use the einsum
        dot-product path because:
          (a) Attention masks: intermediate outputs feed `_attn_mask()` via `> 0`.
              As training progresses the DeconvHead pushes background pixels toward
              zero / negative; this can make the mask intermittently block all
              query→patch attention, collapsing intermediate block learning.
          (b) Query quality: block-9 queries are freshly injected (self.q.weight,
              input-independent) — the einsum's mask_head MLP provides better
              calibration than FiLM conditioned on raw embeddings.
          (c) Backbone signal: 3 einsum losses fully train the backbone, so the
              final DeconvHead inherits high-quality spatial features.
        """
        q = x[:, : self.num_q, :]
        class_logits = self.class_head(q)
        patch_tokens = x[:, self.num_q + self.encoder.backbone.num_prefix_tokens :, :]
        patch_features = patch_tokens.transpose(1, 2).reshape(
            patch_tokens.shape[0], -1, *self.encoder.backbone.patch_embed.grid_size
        )
        mask_logits = torch.einsum(
            "bqc, bchw -> bqhw", self.mask_head(q), self.upscale(patch_features)
        )
        return mask_logits, class_logits

    def _predict(self, x: torch.Tensor):
        """Final-layer prediction using the configured heatmap head."""
        q = x[:, : self.num_q, :]   # (B, Q, C) — raw query tokens

        class_logits = self.class_head(q)

        # Reshape patch tokens to spatial grid then upscale
        patch_tokens = x[:, self.num_q + self.encoder.backbone.num_prefix_tokens :, :]
        patch_features = patch_tokens.transpose(1, 2).reshape(
            patch_tokens.shape[0], -1, *self.encoder.backbone.patch_embed.grid_size
        )                                          # (B, C, grid_h, grid_w)
        upscaled = self.upscale(patch_features)    # (B, C, grid_h*2, grid_w*2)

        if self.heatmap_head == "deconv_v2":
            # DeconvHead takes raw q (not mask_head output) — FiLM generator has its own MLP
            mask_logits = self.deconv_head(q, upscaled)
        else:  # "einsum" — original path
            mask_logits = torch.einsum("bqc, bchw -> bqhw", self.mask_head(q), upscaled)
            if self.refinement_head is not None:
                mask_logits = self.refinement_head(mask_logits)

        return mask_logits, class_logits

    @torch.compiler.disable
    def _disable_attn_mask(self, attn_mask, prob):
        if prob < 1:
            random_queries = (
                torch.rand(attn_mask.shape[0], self.num_q, device=attn_mask.device)
                > prob
            )
            attn_mask[
                :, : self.num_q, self.num_q + self.encoder.backbone.num_prefix_tokens :
            ][random_queries] = True

        return attn_mask

    def _attn(
        self,
        module: nn.Module,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        rope: Optional[torch.Tensor],
    ):
        if rope is not None:
            if mask is not None:
                mask = mask[:, None, ...].expand(-1, module.num_heads, -1, -1)
            return module(x, mask, rope)[0]

        B, N, C = x.shape

        qkv = module.qkv(x).reshape(B, N, 3, module.num_heads, module.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        q, k = module.q_norm(q), module.k_norm(k)

        if mask is not None:
            mask = mask[:, None, ...].expand(-1, module.num_heads, -1, -1)

        dropout_p = module.attn_drop.p if self.training else 0.0

        if module.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v, mask, dropout_p)
        else:
            attn = (q @ k.transpose(-2, -1)) * module.scale
            if mask is not None:
                attn = attn.masked_fill(~mask, float("-inf"))
            attn = F.softmax(attn, dim=-1)
            attn = module.attn_drop(attn)
            x = attn @ v

        x = module.proj_drop(module.proj(x.transpose(1, 2).reshape(B, N, C)))

        return x

    def _attn_mask(self, x: torch.Tensor, mask_logits: torch.Tensor, i: int):
        attn_mask = torch.ones(
            x.shape[0],
            x.shape[1],
            x.shape[1],
            dtype=torch.bool,
            device=x.device,
        )
        interpolated = F.interpolate(
            mask_logits,
            self.encoder.backbone.patch_embed.grid_size,
            mode="bilinear",
        )
        interpolated = interpolated.view(interpolated.size(0), interpolated.size(1), -1)
        attn_mask[
            :,
            : self.num_q,
            self.num_q + self.encoder.backbone.num_prefix_tokens :,
        ] = (
            interpolated > 0
        )
        attn_mask = self._disable_attn_mask(
            attn_mask,
            self.attn_mask_probs[
                i - len(self.encoder.backbone.blocks) + self.num_blocks
            ],
        )
        return attn_mask

    def forward(self, x: torch.Tensor):
        x = (x - self.encoder.pixel_mean) / self.encoder.pixel_std

        rope = None
        if hasattr(self.encoder.backbone, "rope_embeddings"):
            rope = self.encoder.backbone.rope_embeddings(x)

        x = self.encoder.backbone.patch_embed(x)

        if hasattr(self.encoder.backbone, "_pos_embed"):
            x = self.encoder.backbone._pos_embed(x)

        attn_mask = None
        mask_logits_per_layer, class_logits_per_layer = [], []

        for i, block in enumerate(self.encoder.backbone.blocks):
            if i == len(self.encoder.backbone.blocks) - self.num_blocks:
                x = torch.cat(
                    (self.q.weight[None, :, :].expand(x.shape[0], -1, -1), x), dim=1
                )

            if (
                self.masked_attn_enabled
                and i >= len(self.encoder.backbone.blocks) - self.num_blocks
            ):
                # Always use einsum for intermediate predictions:
                #   - attention masks must be stable (no risk of all-False from uninitialised heads)
                #   - 3 einsum losses fully train backbone + mask_head + queries
                mask_logits, class_logits = self._predict_einsum(self.encoder.backbone.norm(x))
                mask_logits_per_layer.append(mask_logits)
                class_logits_per_layer.append(class_logits)

                attn_mask = self._attn_mask(x, mask_logits, i)

            if hasattr(block, "attn"):
                attn = block.attn
            else:
                attn = block.attention
            attn_out = self._attn(attn, block.norm1(x), attn_mask, rope=rope)
            if hasattr(block, "ls1"):
                x = x + block.ls1(attn_out)
            elif hasattr(block, "layer_scale1"):
                x = x + block.layer_scale1(attn_out)

            mlp_out = block.mlp(block.norm2(x))
            if hasattr(block, "ls2"):
                x = x + block.ls2(mlp_out)
            elif hasattr(block, "layer_scale2"):
                x = x + block.layer_scale2(mlp_out)

        x_normed = self.encoder.backbone.norm(x)
        mask_logits, class_logits = self._predict(x_normed)
        mask_logits_per_layer.append(mask_logits)
        class_logits_per_layer.append(class_logits)

        # MODIFIED: direct coord regression from final query tokens.
        # Avoids double norm call by reusing x_normed computed above.
        final_coord_pred = None
        if self.coord_direct_head is not None:
            q_final = x_normed[:, : self.num_q, :]
            final_coord_pred = self.coord_direct_head(q_final).reshape(
                x.shape[0], self.num_q, 2
            )

        return (
            mask_logits_per_layer,
            class_logits_per_layer,
            final_coord_pred,   # None unless heatmap_head == "coord_direct"
        )
