# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------


import torch.nn.functional as F
from torch import nn
from timm.layers import LayerNorm2d


class ScaleBlock(nn.Module):
    def __init__(self, embed_dim, conv1_layer=nn.ConvTranspose2d, use_bilinear=False):
        super().__init__()

        # MODIFIED: bilinear resize + conv avoids checkerboard artifacts from ConvTranspose2d.
        # Set use_bilinear=True for tasks requiring precise spatial localisation (landmark detection).
        self._use_bilinear = use_bilinear
        if use_bilinear:
            self.conv1 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)
        else:
            self.conv1 = conv1_layer(embed_dim, embed_dim, kernel_size=2, stride=2)

        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(
            embed_dim,
            embed_dim,
            kernel_size=3,
            padding=1,
            groups=embed_dim,
            bias=False,
        )
        self.norm = LayerNorm2d(embed_dim)

    def forward(self, x):
        if self._use_bilinear:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm(x)

        return x
