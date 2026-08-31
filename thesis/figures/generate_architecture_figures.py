#!/usr/bin/env python3
"""Generate the two Chapter 3 architecture figures with Matplotlib.

Outputs (PDF is the thesis master; PNG is for quick inspection):
  - eomt_landmark_overall_architecture.{pdf,png}
  - deconvheadv2_internal_architecture.{pdf,png}

The diagrams encode the current implementation in models/eomt.py and the
matched fetal-task configuration.  They deliberately distinguish the three
intermediate einsum heads from the final DeconvHeadV2 path, and show that the
same ScaleBlock module is invoked in both paths.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.patheffects as pe


OUT_DIR = Path(__file__).resolve().parent

# Colour-blind-friendly, higher-contrast palette.  Dark outlines identify the
# functional path; moderately saturated fills preserve readable dark text.
NAVY = "#123B5D"
BLUE = "#1677B8"
LIGHT_BLUE = "#CBE6F5"
ORANGE = "#E76F00"
LIGHT_ORANGE = "#FFDDBA"
GREEN = "#14866D"
LIGHT_GREEN = "#C7EBDD"
PURPLE = "#7042A0"
LIGHT_PURPLE = "#E1D2F1"
GREY = "#455966"
LIGHT_GREY = "#DEE7EC"
DARK = "#17232D"
RED = "#C83E4D"
WHITE = "#FFFFFF"


mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10.0,
        "font.weight": "medium",
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
    }
)


def box(
    ax,
    xy,
    width,
    height,
    text,
    *,
    fc=WHITE,
    ec=NAVY,
    lw=1.25,
    fontsize=8.5,
    weight="normal",
    radius=0.08,
    zorder=3,
    text_color=DARK,
):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.025,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        zorder=zorder,
    )
    patch.set_path_effects(
        [pe.SimplePatchShadow(offset=(1.2, -1.2), alpha=0.12), pe.Normal()]
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color=text_color,
        linespacing=1.18,
        zorder=zorder + 1,
    )
    return patch


def arrow(
    ax,
    start,
    end,
    *,
    color=NAVY,
    lw=1.35,
    style="-",
    mutation=10,
    connectionstyle="arc3",
    zorder=2,
):
    a = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=mutation,
        linewidth=lw,
        linestyle=style,
        color=color,
        connectionstyle=connectionstyle,
        shrinkA=2,
        shrinkB=2,
        zorder=zorder,
    )
    ax.add_patch(a)
    return a


def line(ax, start, end, *, color=GREY, lw=1.1, style="--", zorder=1):
    ax.plot(
        [start[0], end[0]],
        [start[1], end[1]],
        color=color,
        linewidth=lw,
        linestyle=style,
        zorder=zorder,
    )


def label(ax, xy, text, *, color=GREY, fontsize=7.8, ha="center", weight="normal"):
    ax.text(
        xy[0],
        xy[1],
        text,
        color=color,
        fontsize=fontsize,
        ha=ha,
        va="center",
        fontweight=weight,
        linespacing=1.15,
    )


def save(fig, stem):
    fig.savefig(OUT_DIR / f"{stem}.pdf", facecolor="white")
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=300, facecolor="white")
    plt.close(fig)


def _retired_draw_overall_architecture():
    raise RuntimeError(
        "Retired draft layout; use draw_overall_architecture_reference_layout()."
    )
    fig, ax = plt.subplots(figsize=(15.2, 8.6))
    ax.set_xlim(0, 15.2)
    ax.set_ylim(0, 8.6)
    ax.axis("off")

    ax.text(
        0.25,
        8.30,
        "EoMT landmark-detection architecture (matched fetal-task configuration)",
        fontsize=15,
        fontweight="bold",
        color=DARK,
        va="top",
    )
    ax.text(
        0.27,
        7.96,
        "Three intermediate einsum outputs supervise and gate the final transformer blocks; FPN and DeconvHeadV2 are used only by the final path.",
        fontsize=9.2,
        color=GREY,
        va="top",
    )

    # Main encoder route.
    box(ax, (0.25, 5.65), 1.25, 1.00, "Input image\n3 × 512 × 512", fc=LIGHT_GREY)
    box(ax, (1.78, 5.65), 1.35, 1.00, "Patch embedding\npatch16", fc=LIGHT_BLUE)
    box(ax, (3.42, 5.65), 1.42, 1.00, "ViT blocks\n1–3", fc=LIGHT_BLUE)
    box(ax, (5.13, 5.65), 1.42, 1.00, "ViT blocks\n4–8", fc=LIGHT_BLUE)
    box(ax, (6.84, 5.65), 1.42, 1.00, "ViT block 9\nthen insert Q queries", fc=LIGHT_ORANGE, ec=ORANGE)
    box(ax, (8.55, 5.65), 1.24, 1.00, "Block 10", fc=LIGHT_BLUE)
    box(ax, (10.08, 5.65), 1.24, 1.00, "Block 11", fc=LIGHT_BLUE)
    box(ax, (11.61, 5.65), 1.24, 1.00, "Block 12", fc=LIGHT_BLUE)
    box(ax, (13.14, 5.65), 1.48, 1.00, "Final tokens\nLayerNorm", fc=LIGHT_GREEN, ec=GREEN)

    main_boxes = [
        (1.50, 6.15, 1.78, 6.15),
        (3.13, 6.15, 3.42, 6.15),
        (4.84, 6.15, 5.13, 6.15),
        (6.55, 6.15, 6.84, 6.15),
        (8.26, 6.15, 8.55, 6.15),
        (9.79, 6.15, 10.08, 6.15),
        (11.32, 6.15, 11.61, 6.15),
        (12.85, 6.15, 13.14, 6.15),
    ]
    for x1, y1, x2, y2 in main_boxes:
        arrow(ax, (x1, y1), (x2, y2))
    label(ax, (2.46, 6.85), "32 × 32 patch grid", color=BLUE)

    # Query insertion.
    box(ax, (6.88, 7.18), 1.34, 0.48, "Learned queries Q", fc=LIGHT_ORANGE, ec=ORANGE, fontsize=8.0)
    arrow(ax, (7.55, 7.18), (7.55, 6.66), color=ORANGE, connectionstyle="arc3,rad=0.0")

    # Auxiliary heads; each prediction is made before its corresponding block.
    aux_x = [8.47, 10.00, 11.53]
    aux_names = ["Aux 1 before block 10", "Aux 2 before block 11", "Aux 3 before block 12"]
    for idx, (x, name) in enumerate(zip(aux_x, aux_names), start=1):
        box(
            ax,
            (x, 3.78),
            1.42,
            1.00,
            f"{name}\nmask-head MLP ⊗\nupscaled patches",
            fc=LIGHT_PURPLE,
            ec=PURPLE,
            fontsize=7.25,
        )
        # Branch from tokens immediately before each block.
        source_x = [8.38, 9.91, 11.44][idx - 1]
        line(ax, (source_x, 5.72), (source_x, 4.68), color=PURPLE, style="--")
        arrow(ax, (source_x, 4.68), (x + 0.71, 4.78), color=PURPLE, style="--", mutation=8)
        # Loss branch.
        box(ax, (x + 0.09, 2.83), 1.24, 0.48, f"Auxiliary loss {idx}", fc=WHITE, ec=PURPLE, fontsize=7.3)
        arrow(ax, (x + 0.71, 3.78), (x + 0.71, 3.31), color=PURPLE, style="--", mutation=8)

    # Feedback masks into the respective transformer block.
    for x_aux, x_block in zip(aux_x, [9.17, 10.70, 12.23]):
        arrow(
            ax,
            (x_aux + 0.71, 4.78),
            (x_block, 5.65),
            color=RED,
            style="--",
            mutation=8,
            connectionstyle="arc3,rad=-0.18",
        )
    label(ax, (10.70, 5.15), "binary query→patch\nattention masks", color=RED, fontsize=7.4)

    # Shared ScaleBlocks note for auxiliary and final paths.
    box(
        ax,
        (6.48, 2.83),
        1.58,
        1.03,
        "Shared ScaleBlocks ×2\nConvTranspose2d ×2\n32×32 → 128×128",
        fc=LIGHT_GREY,
        ec=GREY,
        fontsize=7.5,
    )
    label(ax, (7.27, 2.55), "same module weights are invoked\nby auxiliary and final paths", color=GREY, fontsize=7.2)
    for x in [9.18, 10.71, 12.24]:
        line(ax, (7.97, 3.35), (x, 3.95), color=GREY, style=":")

    # FPN and final route.
    box(
        ax,
        (3.55, 1.28),
        2.05,
        0.96,
        "FPN-style depth fusion\nGN(blocks 4, 8, 12)\nconcat → 1×1 projection",
        fc=LIGHT_GREEN,
        ec=GREEN,
        fontsize=8.0,
    )
    # Captures from depth 4/8 and final block 12.
    arrow(ax, (5.40, 5.65), (4.25, 2.24), color=GREEN, style="--", mutation=8, connectionstyle="arc3,rad=0.08")
    arrow(ax, (6.18, 5.65), (4.63, 2.24), color=GREEN, style="--", mutation=8, connectionstyle="arc3,rad=0.04")
    arrow(ax, (13.82, 5.65), (5.14, 2.24), color=GREEN, style="--", mutation=8, connectionstyle="arc3,rad=-0.10")
    label(ax, (4.30, 2.46), "final path only", color=GREEN, fontsize=7.4, weight="bold")

    box(
        ax,
        (6.35, 1.28),
        1.82,
        0.96,
        "Shared ScaleBlocks ×2\n32×32 → 128×128",
        fc=LIGHT_GREY,
        ec=GREY,
        fontsize=8.0,
    )
    box(
        ax,
        (8.92, 1.28),
        2.05,
        0.96,
        "DeconvHeadV2\nquery-conditioned FiLM\n+ shared local convs",
        fc=LIGHT_ORANGE,
        ec=ORANGE,
        fontsize=8.3,
        weight="bold",
    )
    box(
        ax,
        (11.73, 1.28),
        1.48,
        0.96,
        "Final heatmaps\nN × 64 × 64",
        fc=LIGHT_GREEN,
        ec=GREEN,
        fontsize=8.4,
        weight="bold",
    )
    box(
        ax,
        (13.68, 1.28),
        1.20,
        0.96,
        "Decode\nargmax +\n±0.25 px",
        fc=LIGHT_BLUE,
        ec=BLUE,
        fontsize=8.0,
    )

    arrow(ax, (5.60, 1.76), (6.35, 1.76), color=GREEN)
    arrow(ax, (8.17, 1.76), (8.92, 1.76), color=ORANGE)
    arrow(ax, (10.97, 1.76), (11.73, 1.76), color=ORANGE)
    arrow(ax, (13.21, 1.76), (13.68, 1.76), color=BLUE)

    # Query branch from final tokens to DeconvHeadV2.
    arrow(
        ax,
        (13.88, 5.65),
        (10.23, 2.24),
        color=ORANGE,
        style="--",
        mutation=8,
        connectionstyle="arc3,rad=0.20",
    )
    label(ax, (12.70, 3.38), "final query tokens", color=ORANGE, fontsize=7.6)

    # Supervision and unused class head.
    box(ax, (11.82, 0.30), 1.30, 0.50, "Final hybrid loss", fc=WHITE, ec=GREEN, fontsize=7.6)
    arrow(ax, (12.47, 1.28), (12.47, 0.80), color=GREEN, style="--", mutation=8)
    box(ax, (13.25, 4.18), 1.42, 0.66, "Class head retained\nbut output unused", fc=WHITE, ec=GREY, fontsize=7.4)
    line(ax, (13.88, 5.65), (13.88, 4.84), color=GREY, style=":")

    # Legend.
    label(ax, (0.30, 0.72), "Solid arrows: inference path", color=NAVY, ha="left", fontsize=7.5)
    label(ax, (0.30, 0.43), "Dashed arrows: training-only loss supervision", color=PURPLE, ha="left", fontsize=7.5)
    label(ax, (0.30, 0.14), "Fetal configuration shown: N=2 landmark queries.", color=GREY, ha="left", fontsize=7.5)

    save(fig, "eomt_landmark_overall_architecture")


def _retired_draw_deconvheadv2():
    raise RuntimeError(
        "Retired draft layout; use draw_deconvheadv2_publication()."
    )
    fig, ax = plt.subplots(figsize=(14.6, 7.0))
    ax.set_xlim(0, 14.6)
    ax.set_ylim(0, 7.0)
    ax.axis("off")

    ax.text(
        0.25,
        6.75,
        "DeconvHeadV2: query-conditioned spatial heatmap decoding",
        fontsize=15,
        fontweight="bold",
        color=DARK,
        va="top",
    )
    ax.text(
        0.27,
        6.40,
        "The transposed-convolution upsampling belongs to the shared EoMT ScaleBlocks; the QueryConditionedDeconvHead itself uses FiLM and standard convolutions.",
        fontsize=9.1,
        color=GREY,
        va="top",
    )

    # Inputs and shared upsampling.
    box(ax, (0.35, 3.82), 1.42, 1.02, "FPN-fused patches\nB × 384 × 32 × 32", fc=LIGHT_GREEN, ec=GREEN, fontsize=8.1)
    box(ax, (2.12, 3.82), 1.45, 1.02, "ScaleBlock 1\nConvT 2× + GELU\nDWConv + LN2d", fc=LIGHT_GREY, ec=GREY, fontsize=7.8)
    box(ax, (3.92, 3.82), 1.45, 1.02, "ScaleBlock 2\nConvT 2× + GELU\nDWConv + LN2d", fc=LIGHT_GREY, ec=GREY, fontsize=7.8)
    box(ax, (5.72, 3.82), 1.32, 1.02, "Features F\nB × 384\n× 128 × 128", fc=LIGHT_BLUE, ec=BLUE, fontsize=8.0)
    arrow(ax, (1.77, 4.33), (2.12, 4.33))
    arrow(ax, (3.57, 4.33), (3.92, 4.33))
    arrow(ax, (5.37, 4.33), (5.72, 4.33))
    label(ax, (2.84, 5.05), "same ScaleBlock weights are also used by auxiliary einsum heads", color=GREY, fontsize=7.3)

    # Query FiLM branch.
    box(ax, (0.35, 1.30), 1.42, 0.92, "Final queries q\nB × N × 384", fc=LIGHT_ORANGE, ec=ORANGE, fontsize=8.2)
    box(ax, (2.18, 1.18), 2.02, 1.16, "FiLM generator\nLinear 384→384 + ReLU\nDropout 0.2\nLinear 384→768", fc=LIGHT_ORANGE, ec=ORANGE, fontsize=7.7)
    box(ax, (4.62, 1.30), 1.50, 0.92, "Split channels\nγq, βq\nB × N × 384", fc=LIGHT_ORANGE, ec=ORANGE, fontsize=8.0)
    arrow(ax, (1.77, 1.76), (2.18, 1.76), color=ORANGE)
    arrow(ax, (4.20, 1.76), (4.62, 1.76), color=ORANGE)

    # Normalisation and FiLM merge.
    box(ax, (7.45, 3.82), 1.24, 1.02, "GroupNorm\n32 groups", fc=LIGHT_PURPLE, ec=PURPLE, fontsize=8.2)
    box(ax, (9.10, 3.68), 1.70, 1.30, "Per-query FiLM\nF′q = γq ⊙ GN(F) + βq\nreshape B·N × 384\n× 128 × 128", fc=LIGHT_ORANGE, ec=ORANGE, fontsize=7.8, weight="bold")
    arrow(ax, (7.04, 4.33), (7.45, 4.33), color=PURPLE)
    arrow(ax, (8.69, 4.33), (9.10, 4.33), color=ORANGE)
    arrow(ax, (6.12, 1.76), (9.72, 3.68), color=ORANGE, style="--", connectionstyle="arc3,rad=-0.12")

    # Shared conv decoder.
    box(ax, (11.22, 3.82), 1.18, 1.02, "3×3 Conv\n384 → 128", fc=LIGHT_BLUE, ec=BLUE, fontsize=8.2)
    box(ax, (12.70, 3.82), 0.82, 1.02, "ReLU", fc=LIGHT_BLUE, ec=BLUE, fontsize=8.2)
    box(ax, (11.22, 2.12), 1.18, 1.02, "3×3 Conv\n128 → 1", fc=LIGHT_BLUE, ec=BLUE, fontsize=8.2)
    box(ax, (12.70, 2.12), 1.40, 1.02, "Bilinear resize\n128×128 → 64×64", fc=LIGHT_GREEN, ec=GREEN, fontsize=8.0)
    box(ax, (11.22, 0.43), 2.88, 0.96, "Per-query heatmaps\nB × N × 64 × 64", fc=LIGHT_GREEN, ec=GREEN, fontsize=9.0, weight="bold")

    arrow(ax, (10.80, 4.33), (11.22, 4.33), color=BLUE)
    arrow(ax, (12.40, 4.33), (12.70, 4.33), color=BLUE)
    arrow(ax, (13.11, 3.82), (11.81, 3.14), color=BLUE, connectionstyle="arc3,rad=-0.18")
    arrow(ax, (12.40, 2.63), (12.70, 2.63), color=GREEN)
    arrow(ax, (13.40, 2.12), (13.40, 1.39), color=GREEN)

    # Notes and initialisation.
    box(
        ax,
        (6.70, 0.40),
        3.72,
        1.20,
        "Shared across all N queries\nConv2 weights = 0; bias = 0.01 at initialisation\n(the final head does not generate attention masks)",
        fc=WHITE,
        ec=GREY,
        fontsize=7.8,
    )
    label(ax, (9.95, 5.30), "query-specific modulation", color=ORANGE, fontsize=7.6, weight="bold")
    label(ax, (12.62, 5.30), "shared local spatial processing", color=BLUE, fontsize=7.6, weight="bold")

    # Stage labels.
    label(ax, (3.00, 5.62), "Stage 1 — shared spatial upsampling", color=GREY, fontsize=8.3, weight="bold")
    label(ax, (3.00, 0.72), "Stage 2 — query conditioning", color=ORANGE, fontsize=8.3, weight="bold")
    label(ax, (12.65, 5.62), "Stage 3 — shared decoder", color=BLUE, fontsize=8.3, weight="bold")

    save(fig, "deconvheadv2_internal_architecture")


# ---------------------------------------------------------------------------
# Publication-style versions.  These intentionally use visual abstraction
# (grouped repeated modules and buses) rather than one connector per tensor.
# The detailed implementation remains encoded in the labels and caption-ready
# annotations, while the reader can identify the main argument at a glance.
# ---------------------------------------------------------------------------


def pbox(
    ax,
    xy,
    width,
    height,
    text,
    *,
    fc=WHITE,
    ec=NAVY,
    fontsize=9.0,
    weight="medium",
    lw=1.35,
    radius=0.045,
    align="center",
):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.018,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        zorder=3,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha=align,
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color=DARK,
        linespacing=1.17,
        zorder=4,
    )
    return patch


def panel(ax, xy, width, height, title, letter, *, fc="#F6F9FB", ec="#C8D6DF", title_color=NAVY):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.025,rounding_size=0.06",
        facecolor=fc,
        edgecolor=ec,
        linewidth=1.05,
        zorder=0,
    )
    ax.add_patch(patch)
    ax.text(x + 0.16, y + height - 0.20, letter, fontsize=14, fontweight="bold", color=title_color, va="top")
    ax.text(x + 0.48, y + height - 0.21, title, fontsize=11.5, fontweight="bold", color=title_color, va="top")
    return patch


def dot(ax, xy, color, size=24):
    ax.scatter([xy[0]], [xy[1]], s=size, color=color, edgecolor=WHITE, linewidth=0.6, zorder=5)


def _retired_draw_overall_architecture_publication():
    raise RuntimeError(
        "Retired draft layout; use draw_overall_architecture_reference_layout()."
    )
    fig, ax = plt.subplots(figsize=(13.4, 7.3))
    ax.set_xlim(0, 13.4)
    ax.set_ylim(0, 7.3)
    ax.axis("off")

    # Three quiet visual regions; captions in LaTeX carry the long explanation.
    panel(ax, (0.18, 0.52), 5.05, 6.25, "Encoder and feature extraction", "A")
    panel(ax, (5.42, 3.73), 3.30, 3.04, "Repeated auxiliary supervision", "B", fc="#F9F5FC", ec="#D6C7E5")
    panel(ax, (5.42, 0.52), 7.80, 2.98, "Final landmark-decoding path", "C", fc="#F3FAF7", ec="#C2DED2")

    # Panel A — compact token stream.
    pbox(ax, (0.48, 5.02), 0.92, 0.90, "Image\n512²", fc=LIGHT_GREY, ec=GREY, fontsize=9.3, weight="bold")
    pbox(ax, (1.70, 5.02), 1.04, 0.90, "Patch embed\n16 × 16", fc=LIGHT_BLUE, ec=BLUE, fontsize=8.7)
    pbox(ax, (3.04, 4.70), 1.82, 1.54, "DINO encoder\n12 transformer\nblocks", fc="#E8F0F6", ec=BLUE, fontsize=9.3, weight="bold")
    arrow(ax, (1.40, 5.47), (1.70, 5.47), lw=1.1, mutation=8)
    arrow(ax, (2.74, 5.47), (3.04, 5.47), lw=1.1, mutation=8)
    label(ax, (2.22, 4.78), "32 × 32 tokens", color=BLUE, fontsize=8.5, weight="bold")

    # Encoder depth strip, with feature taps represented as dots and one bus.
    y_strip = 3.70
    strip_specs = [
        (0.55, 0.92, "1–3"),
        (1.58, 0.92, "4–8"),
        (2.61, 0.72, "9"),
        (3.44, 1.34, "10–12"),
    ]
    for x, w, txt in strip_specs:
        pbox(ax, (x, y_strip), w, 0.58, txt, fc=WHITE, ec=BLUE, fontsize=8.4, weight="bold", radius=0.025)
    for x1, x2 in [(1.47, 1.58), (2.50, 2.61), (3.33, 3.44)]:
        arrow(ax, (x1, 3.99), (x2, 3.99), lw=0.9, mutation=6)
    line(ax, (3.95, 4.70), (3.95, 4.42), color="#AAB4BB", lw=0.8, style="-")
    label(ax, (2.63, 4.45), "expanded encoder-depth view", color=GREY, fontsize=8.1, weight="bold")

    # Query injection as a single clean event.
    pbox(ax, (2.43, 2.61), 1.04, 0.58, "Learned Q", fc=LIGHT_ORANGE, ec=ORANGE, fontsize=8.4, weight="bold")
    arrow(ax, (2.95, 3.19), (2.95, 3.70), color=ORANGE, lw=1.0, mutation=7)
    label(ax, (3.07, 3.38), "insert before block 10", color=ORANGE, fontsize=8.0, ha="left", weight="bold")

    # Three FPN feature taps collapse into one bus.
    tap_x = [1.90, 2.33, 4.20]
    for x in tap_x:
        dot(ax, (x, 3.70), GREEN)
        line(ax, (x, 3.66), (x, 1.63), color=GREEN, lw=0.9, style="-")
    line(ax, (1.90, 1.63), (4.20, 1.63), color=GREEN, lw=1.7, style="-")
    pbox(ax, (1.47, 0.83), 3.14, 0.62, "Captured features  ·  blocks 4, 8 and 12", fc=LIGHT_GREEN, ec=GREEN, fontsize=8.6, weight="bold")
    arrow(ax, (3.05, 1.63), (3.05, 1.45), color=GREEN, lw=1.0, mutation=7)
    label(ax, (0.55, 1.90), "multi-depth feature bus", color=GREEN, fontsize=8.2, ha="left", weight="bold")

    # Panel B — one repeated module instead of a web of three copies.
    pbox(ax, (5.75, 5.06), 1.25, 0.76, "Last three\nblocks", fc=LIGHT_BLUE, ec=BLUE, fontsize=8.8, weight="bold")
    pbox(ax, (7.28, 4.82), 1.10, 1.22, "Einsum\nhead\n×3", fc=LIGHT_PURPLE, ec=PURPLE, fontsize=10.0, weight="bold")
    arrow(ax, (7.00, 5.44), (7.28, 5.44), color=PURPLE, lw=1.1, mutation=8)
    arrow(ax, (7.83, 4.82), (7.83, 4.34), color=PURPLE, lw=1.0, style="--", mutation=7)
    label(ax, (7.83, 4.15), "three auxiliary losses", color=PURPLE, fontsize=8.3, weight="bold")
    arrow(ax, (7.28, 5.08), (6.70, 5.06), color=RED, lw=1.0, style="--", mutation=7, connectionstyle="arc3,rad=-0.32")
    label(ax, (6.44, 4.64), "query→patch masks", color=RED, fontsize=8.0, weight="bold")
    label(ax, (6.98, 6.26), "shared mask-head MLP and ScaleBlocks", color=GREY, fontsize=8.1, weight="bold")

    # A single bridge indicates where the repeated pre-block branch attaches.
    arrow(ax, (4.78, 3.99), (5.75, 5.44), color=PURPLE, lw=1.0, style="--", mutation=7, connectionstyle="arc3,rad=-0.18")

    # Panel C — one dominant final route, with a small query conditioning input.
    pbox(ax, (5.72, 1.72), 1.26, 0.74, "FPN\nfusion", fc=LIGHT_GREEN, ec=GREEN, fontsize=9.0, weight="bold")
    pbox(ax, (7.35, 1.72), 1.45, 0.74, "ScaleBlocks ×2\n32² → 128²", fc=LIGHT_GREY, ec=GREY, fontsize=8.3)
    pbox(ax, (9.17, 1.52), 1.67, 1.14, "DeconvHeadV2\nFiLM + local convs", fc=LIGHT_ORANGE, ec=ORANGE, fontsize=9.2, weight="bold")
    pbox(ax, (11.20, 1.72), 1.22, 0.74, "Heatmaps\nN × 64²", fc=LIGHT_GREEN, ec=GREEN, fontsize=9.0, weight="bold")
    arrow(ax, (6.98, 2.09), (7.35, 2.09), color=GREEN, lw=1.2, mutation=8)
    arrow(ax, (8.80, 2.09), (9.17, 2.09), color=ORANGE, lw=1.2, mutation=8)
    arrow(ax, (10.84, 2.09), (11.20, 2.09), color=GREEN, lw=1.2, mutation=8)
    arrow(ax, (12.42, 2.09), (12.82, 2.09), color=BLUE, lw=1.1, mutation=8)
    label(ax, (12.88, 2.09), "decode", color=BLUE, fontsize=8.5, ha="left", weight="bold")

    # FPN input from panel A and query input from final tokens.
    arrow(ax, (4.61, 1.14), (5.72, 2.02), color=GREEN, lw=1.1, mutation=8, connectionstyle="arc3,rad=-0.08")
    pbox(ax, (8.31, 0.75), 1.35, 0.48, "final query tokens", fc=WHITE, ec=ORANGE, fontsize=7.8)
    arrow(ax, (8.99, 1.23), (9.65, 1.52), color=ORANGE, lw=0.95, style="--", mutation=7)

    # A restrained legend.
    line(ax, (5.78, 0.82), (6.20, 0.82), color=NAVY, lw=1.4, style="-")
    label(ax, (6.27, 0.82), "inference", color=GREY, fontsize=7.9, ha="left", weight="bold")
    line(ax, (7.42, 0.82), (7.84, 0.82), color=PURPLE, lw=1.1, style="--")
    label(ax, (7.91, 0.82), "training-only", color=GREY, fontsize=7.9, ha="left", weight="bold")

    save(fig, "eomt_landmark_overall_architecture")


def _retired_draw_overall_architecture_publication_v2():
    """Publication layout with three colour-coded, self-contained stages."""
    raise RuntimeError(
        "Retired draft layout; use draw_overall_architecture_reference_layout()."
    )
    fig, ax = plt.subplots(figsize=(15.2, 7.5))
    ax.set_xlim(0, 15.2)
    ax.set_ylim(0, 7.5)
    ax.axis("off")

    # Panel backgrounds and borders carry the stage colour consistently.
    panel(ax, (0.16, 0.42), 5.72, 6.62, "Encoder and feature extraction", "A", fc="#E6F1F7", ec=BLUE)
    panel(ax, (6.10, 4.14), 8.92, 2.90, "Repeated pre-block prediction and supervision", "B", fc="#F1EAF8", ec=PURPLE)
    panel(ax, (6.10, 0.42), 8.92, 3.46, "Final landmark-decoding path", "C", fc="#E5F4EF", ec=GREEN)
    # Re-colour panel titles to match their stage.
    ax.text(0.32, 6.84, "A", fontsize=14, fontweight="bold", color=BLUE, va="top", zorder=6)
    ax.text(0.64, 6.83, "Encoder and feature extraction", fontsize=11.5, fontweight="bold", color=BLUE, va="top", zorder=6)
    ax.text(6.26, 6.84, "B", fontsize=14, fontweight="bold", color=PURPLE, va="top", zorder=6)
    ax.text(6.58, 6.83, "Repeated pre-block prediction and supervision", fontsize=11.5, fontweight="bold", color=PURPLE, va="top", zorder=6)
    ax.text(6.26, 3.68, "C", fontsize=14, fontweight="bold", color=GREEN, va="top", zorder=6)
    ax.text(6.58, 3.67, "Final landmark-decoding path", fontsize=11.5, fontweight="bold", color=GREEN, va="top", zorder=6)

    label(ax, (10.55, 6.45), "B = batch size  ·  N = landmark queries  ·  C = channels  ·  H×W = spatial size", color=GREY, fontsize=7.8, weight="bold")

    # A — image-to-token pathway.
    pbox(ax, (0.48, 5.12), 1.08, 1.00, "Input image\nH×W\n512×512", fc=LIGHT_BLUE, ec=BLUE, fontsize=8.3, weight="bold")
    pbox(ax, (1.88, 5.02), 1.48, 1.20, "Patch embedding\nkernel/stride: 16×16\ntoken grid: 32×32\nC = 384", fc=LIGHT_BLUE, ec=BLUE, fontsize=7.6)
    pbox(ax, (3.62, 4.90), 1.94, 1.44, "Self-supervised\nDINO encoder\n12 transformer blocks\ntoken width C = 384", fc=LIGHT_BLUE, ec=BLUE, fontsize=8.0, weight="bold")
    arrow(ax, (1.56, 5.62), (1.88, 5.62), color=BLUE, lw=1.2, mutation=8)
    arrow(ax, (3.36, 5.62), (3.62, 5.62), color=BLUE, lw=1.2, mutation=8)

    label(ax, (2.90, 4.52), "expanded view of the 12 encoder blocks", color=BLUE, fontsize=8.1, weight="bold")
    strip_specs = [(0.48, 1.00, "blocks\n1–3"), (1.62, 1.12, "blocks\n4–8"), (2.88, 0.76, "block\n9"), (3.78, 1.70, "blocks 10–12\nquery-aware")]
    for x, w, txt in strip_specs:
        pbox(ax, (x, 3.66), w, 0.70, txt, fc="#D7EAF5", ec=BLUE, fontsize=7.8, weight="bold", radius=0.025)
    for x1, x2 in [(1.48, 1.62), (2.74, 2.88), (3.64, 3.78)]:
        arrow(ax, (x1, 4.01), (x2, 4.01), color=BLUE, lw=1.0, mutation=6)
    pbox(ax, (2.42, 2.64), 1.46, 0.66, "Learned queries q\nN×C = N×384", fc="#D7EAF5", ec=BLUE, fontsize=7.8, weight="bold")
    arrow(ax, (3.15, 3.30), (3.15, 3.66), color=BLUE, lw=1.05, mutation=7)
    label(ax, (4.18, 3.43), "inserted before block 10", color=BLUE, fontsize=7.7, weight="bold")

    # Encoder feature taps used by the FPN-style fusion path.
    for x in [1.90, 2.46, 4.88]:
        dot(ax, (x, 3.66), BLUE)
        line(ax, (x, 3.62), (x, 1.72), color=BLUE, lw=1.0, style="-")
    line(ax, (1.90, 1.72), (4.88, 1.72), color=BLUE, lw=1.8, style="-")
    pbox(ax, (1.10, 0.76), 4.16, 0.72, "Captured encoder features\nblocks 4, 8 and 12; C = 384; H×W = 32×32", fc=LIGHT_BLUE, ec=BLUE, fontsize=8.0, weight="bold")
    arrow(ax, (3.18, 1.72), (3.18, 1.48), color=BLUE, lw=1.05, mutation=7)

    # B — one repeated unit and its two training roles.
    pbox(ax, (6.44, 5.08), 2.06, 1.02, "Transformer blocks\n10, 11 and 12\nwith landmark queries", fc=LIGHT_PURPLE, ec=PURPLE, fontsize=8.1, weight="bold")
    pbox(ax, (8.88, 4.92), 2.22, 1.34, "Einstein-summation\n(einsum) heatmap head\napplied after each block\n×3 outputs", fc=LIGHT_PURPLE, ec=PURPLE, fontsize=7.9, weight="bold")
    pbox(ax, (11.42, 5.14), 1.64, 0.92, "Auxiliary heatmaps\nN maps\n64×64 each", fc=LIGHT_PURPLE, ec=PURPLE, fontsize=7.8, weight="bold")
    pbox(ax, (13.30, 4.96), 1.50, 1.28, "Two training roles\n1  auxiliary loss\n2  attention mask\nfor the next block", fc=LIGHT_PURPLE, ec=PURPLE, fontsize=7.4, weight="bold")
    arrow(ax, (8.50, 5.59), (8.88, 5.59), color=PURPLE, lw=1.2, style="--", mutation=8)
    arrow(ax, (11.10, 5.59), (11.42, 5.59), color=PURPLE, lw=1.2, style="--", mutation=8)
    arrow(ax, (13.06, 5.60), (13.30, 5.60), color=PURPLE, lw=1.2, style="--", mutation=8)
    label(ax, (10.05, 4.53), "shared mask-head multilayer perceptron and shared ScaleBlocks", color=PURPLE, fontsize=7.5, weight="bold")
    arrow(ax, (5.48, 4.01), (6.44, 5.55), color=PURPLE, lw=1.1, style="--", mutation=7, connectionstyle="arc3,rad=-0.15")

    # C — uninterrupted final inference route.
    pbox(ax, (6.36, 1.78), 1.50, 1.14, "Feature Pyramid\nNetwork (FPN)-style\nfusion of blocks 4, 8, 12\nC = 384\nH×W = 32×32", fc=LIGHT_GREEN, ec=GREEN, fontsize=6.65, weight="bold")
    pbox(ax, (8.14, 1.82), 1.50, 1.06, "Shared\nScaleBlocks ×2\nspatial size\n32×32 → 128×128", fc=LIGHT_GREEN, ec=GREEN, fontsize=7.1, weight="bold")
    pbox(ax, (9.94, 1.54), 2.12, 1.62, "DeconvHeadV2\nquery-conditioned\nfeature-wise linear\nmodulation (FiLM)\nand two local 3×3\nconvolutions", fc=LIGHT_GREEN, ec=GREEN, fontsize=7.15, weight="bold")
    pbox(ax, (12.36, 1.82), 1.32, 1.06, "Heatmap output\nN maps\n64×64 each", fc=LIGHT_GREEN, ec=GREEN, fontsize=7.5, weight="bold")
    pbox(ax, (13.96, 1.78), 0.86, 1.14, "Coordinate\ndecoding\nargmax +\n±0.25 pixel", fc=LIGHT_GREEN, ec=GREEN, fontsize=6.8, weight="bold")
    arrow(ax, (7.86, 2.35), (8.14, 2.35), color=GREEN, lw=1.25, mutation=8)
    arrow(ax, (9.64, 2.35), (9.94, 2.35), color=GREEN, lw=1.25, mutation=8)
    arrow(ax, (12.06, 2.35), (12.36, 2.35), color=GREEN, lw=1.25, mutation=8)
    arrow(ax, (13.68, 2.35), (13.96, 2.35), color=GREEN, lw=1.2, mutation=8)

    # Inputs are attached directly to their consumers; no floating labels.
    arrow(ax, (5.26, 1.12), (6.36, 2.22), color=GREEN, lw=1.15, mutation=8, connectionstyle="arc3,rad=-0.08")
    pbox(ax, (9.94, 0.70), 2.30, 0.54, "Final query tokens: N×C = N×384", fc="#D2EAE2", ec=GREEN, fontsize=7.7, weight="bold")
    arrow(ax, (11.09, 1.24), (11.02, 1.54), color=GREEN, lw=1.0, mutation=7)
    label(ax, (7.50, 0.92), "solid arrows: inference", color=GREEN, fontsize=7.6, weight="bold")
    label(ax, (7.50, 0.66), "dashed arrows: training-only losses", color=PURPLE, fontsize=7.3, weight="bold")

    save(fig, "eomt_landmark_overall_architecture")


def _retired_draw_overall_architecture_journal():
    """Journal-style overview: concise main route plus structured annotations."""
    raise RuntimeError(
        "Retired draft layout; use draw_overall_architecture_reference_layout()."
    )
    fig, ax = plt.subplots(figsize=(16.0, 8.0))
    ax.set_xlim(0, 16.0)
    ax.set_ylim(0, 8.0)
    ax.axis("off")

    # Warm journal palette: semantic modules use different pastel families,
    # while fine charcoal connectors keep the complete route visually unified.
    WARM_LINE = "#4B4A48"
    WARM_PINK_BG = "#FAEFEE"
    WARM_YELLOW_BG = "#FFF7DD"
    WARM_BLUE_BG = "#EEF3FA"
    WARM_PEACH = "#F6C9A8"
    WARM_PINK = "#EAB4BF"
    WARM_YELLOW = "#F8DF88"
    WARM_BLUE = "#D9E2F3"
    WARM_GREEN = "#DDECCF"
    WARM_GREY = "#E4E4E2"
    WARM_BORDER = "#8A807A"

    panel(ax, (0.22, 0.48), 5.35, 7.02, "Encoder and feature extraction", "A", fc=WARM_PINK_BG, ec="#D6AAA5", title_color="#8F514A")
    panel(ax, (5.80, 4.32), 9.95, 3.18, "Repeated auxiliary supervision", "B", fc=WARM_YELLOW_BG, ec="#D8BA58", title_color="#806918")
    panel(ax, (5.80, 0.48), 9.95, 3.58, "Final landmark-decoding path", "C", fc=WARM_BLUE_BG, ec="#8FA3CC", title_color="#4D628E")
    pbox(ax, (4.66, 7.06), 0.66, 0.28, "Stage 1", fc="#F4D9D5", ec="#C58F88", fontsize=6.7, weight="bold", lw=0.8, radius=0.02)
    pbox(ax, (14.78, 7.06), 0.66, 0.28, "Stage 2", fc="#FFF0B8", ec="#C7A840", fontsize=6.7, weight="bold", lw=0.8, radius=0.02)
    pbox(ax, (14.78, 3.62), 0.66, 0.28, "Stage 3", fc="#DCE5F5", ec="#8096C4", fontsize=6.7, weight="bold", lw=0.8, radius=0.02)

    card = dict(lw=1.05, radius=0.035)

    # A1. Compact image-to-token route.
    pbox(ax, (0.48, 5.55), 1.00, 1.00, "Input\nimage\n512×512", fc=WARM_GREY, ec=WARM_BORDER, fontsize=8.6, weight="bold", **card)
    pbox(ax, (1.80, 5.45), 1.40, 1.20, "Patch embedding\n16×16 kernel\n32×32 tokens\nC = 384", fc=WARM_PEACH, ec=WARM_BORDER, fontsize=7.8, **card)
    pbox(ax, (3.54, 5.38), 1.70, 1.34, "DINO ViT-S encoder\n12 blocks\nC = 384", fc=WARM_YELLOW, ec=WARM_BORDER, fontsize=8.5, weight="bold", **card)
    arrow(ax, (1.48, 6.05), (1.80, 6.05), color=WARM_LINE, lw=1.05, mutation=7)
    arrow(ax, (3.20, 6.05), (3.54, 6.05), color=WARM_LINE, lw=1.05, mutation=7)

    # A2. Encoder depth and query insertion.
    label(ax, (2.90, 4.88), "encoder depth", color="#8F514A", fontsize=8.3, weight="bold")
    strips = [(0.50, 0.92, "1–3"), (1.55, 1.02, "4–8"), (2.70, 0.70, "9"), (3.53, 1.68, "10–12\nquery-aware")]
    for x, w, txt in strips:
        pbox(ax, (x, 4.08), w, 0.66, txt, fc=WARM_PEACH, ec=WARM_BORDER, fontsize=8.0, weight="bold", **card)
    for x1, x2 in [(1.42, 1.55), (2.57, 2.70), (3.40, 3.53)]:
        arrow(ax, (x1, 4.41), (x2, 4.41), color=WARM_LINE, lw=0.9, mutation=5)
    pbox(ax, (2.55, 3.04), 1.48, 0.62, "N learned queries\nC = 384", fc=WARM_BLUE, ec=WARM_BORDER, fontsize=8.0, weight="bold", **card)
    arrow(ax, (3.29, 3.66), (3.29, 4.08), color=WARM_LINE, lw=1.0, mutation=7)
    label(ax, (4.30, 3.83), "insert before block 10", color="#8F514A", fontsize=7.8, weight="bold")

    # A3. Feature taps are summarised in one card rather than a wiring web.
    for x in [1.98, 2.38, 4.60]:
        dot(ax, (x, 4.08), WARM_LINE, size=22)
        line(ax, (x, 4.04), (x, 1.82), color=WARM_LINE, lw=0.9, style="-")
    line(ax, (1.98, 1.82), (4.60, 1.82), color=WARM_LINE, lw=1.5, style="-")
    pbox(ax, (1.06, 0.82), 4.02, 0.72, "Feature taps F4, F8, F12\nC = 384; H×W = 32×32", fc=WARM_BLUE, ec=WARM_BORDER, fontsize=8.3, weight="bold", **card)
    arrow(ax, (3.29, 1.82), (3.29, 1.54), color=WARM_LINE, lw=1.0, mutation=6)

    # B. Repeated unit and its two distinct training outputs.
    label(ax, (10.85, 6.75), "shared mask-head multilayer perceptron (MLP) and shared ScaleBlocks", color="#806918", fontsize=7.8, weight="bold")
    pbox(ax, (6.18, 5.34), 1.86, 1.10, "Query-aware blocks\n10, 11, 12", fc=WARM_PEACH, ec=WARM_BORDER, fontsize=8.5, weight="bold", **card)
    pbox(ax, (8.42, 5.22), 2.10, 1.34, "Einstein-summation\n(einsum) head ×3", fc=WARM_PINK, ec=WARM_BORDER, fontsize=8.6, weight="bold", **card)
    pbox(ax, (10.90, 5.34), 1.62, 1.10, "Auxiliary heatmaps\nN × 64×64", fc=WARM_BLUE, ec=WARM_BORDER, fontsize=8.1, weight="bold", **card)
    pbox(ax, (12.92, 5.76), 2.16, 0.62, "Auxiliary loss ×3", fc=WARM_YELLOW, ec=WARM_BORDER, fontsize=8.3, weight="bold", **card)
    pbox(ax, (12.92, 4.86), 2.16, 0.62, "Attention-mask feedback", fc=WARM_PINK, ec=WARM_BORDER, fontsize=8.1, weight="bold", **card)
    arrow(ax, (8.04, 5.89), (8.42, 5.89), color=WARM_LINE, lw=1.05, style="--", mutation=7)
    arrow(ax, (10.52, 5.89), (10.90, 5.89), color=WARM_LINE, lw=1.05, style="--", mutation=7)
    arrow(ax, (12.52, 5.98), (12.92, 6.07), color=WARM_LINE, lw=1.0, style="--", mutation=7)
    arrow(ax, (12.52, 5.70), (12.92, 5.17), color=WARM_LINE, lw=1.0, style="--", mutation=7)
    arrow(ax, (5.21, 4.41), (6.18, 5.86), color=WARM_LINE, lw=1.0, style="--", mutation=7, connectionstyle="arc3,rad=-0.14")

    # C. Concise inference route; dimensions sit below the operation name.
    label(ax, (11.05, 3.42), "solid arrows: model path (training + inference)   ·   dashed arrows: training-only losses", color=WARM_LINE, fontsize=7.7, weight="bold")
    label(ax, (11.30, 3.15), "N = landmark queries   ·   C = channels   ·   FPN = Feature Pyramid Network   ·   FiLM = feature-wise linear modulation", color="#4D628E", fontsize=6.75, weight="bold")
    pbox(ax, (6.16, 1.82), 1.62, 1.08, "FPN-style fusion\nF4, F8, F12\n32×32; C = 384", fc=WARM_GREEN, ec=WARM_BORDER, fontsize=7.8, weight="bold", **card)
    pbox(ax, (8.10, 1.82), 1.62, 1.08, "ScaleBlocks ×2\n32×32 → 128×128", fc=WARM_PEACH, ec=WARM_BORDER, fontsize=7.8, weight="bold", **card)
    pbox(ax, (10.04, 1.64), 2.20, 1.44, "DeconvHeadV2\nFiLM conditioning\n+ two 3×3 convolutions", fc=WARM_PINK, ec=WARM_BORDER, fontsize=8.4, weight="bold", **card)
    pbox(ax, (12.58, 1.82), 1.44, 1.08, "Heatmaps\nN × 64×64", fc=WARM_BLUE, ec=WARM_BORDER, fontsize=8.3, weight="bold", **card)
    pbox(ax, (14.34, 1.74), 1.12, 1.24, "Decode\nargmax\n±0.25 pixel", fc=WARM_YELLOW, ec=WARM_BORDER, fontsize=7.8, weight="bold", **card)
    for x1, x2 in [(7.78, 8.10), (9.72, 10.04), (12.24, 12.58), (14.02, 14.34)]:
        arrow(ax, (x1, 2.36), (x2, 2.36), color=WARM_LINE, lw=1.1, mutation=7)
    arrow(ax, (5.08, 1.18), (6.16, 2.25), color=WARM_LINE, lw=1.05, mutation=7, connectionstyle="arc3,rad=-0.08")
    pbox(ax, (10.18, 0.72), 1.92, 0.50, "Final queries: N×384", fc=WARM_BLUE, ec=WARM_BORDER, fontsize=7.9, weight="bold", **card)
    arrow(ax, (11.14, 1.22), (11.14, 1.64), color=WARM_LINE, lw=0.95, mutation=6)

    save(fig, "eomt_landmark_overall_architecture")


def draw_overall_architecture_reference_layout():
    """Reproduce the supplied stepped-panel layout with the warm palette."""
    fig, ax = plt.subplots(figsize=(16.0, 7.7))
    ax.set_xlim(0, 16.0)
    ax.set_ylim(0, 7.7)
    ax.axis("off")

    line_colour = "#55514E"
    bg_a, edge_a = "#FAEFEE", "#C9938D"
    bg_b, edge_b = "#FFF7DD", "#C6A43B"
    bg_c, edge_c = "#EEF3FA", "#8397C3"
    peach, pink = "#F6C9A8", "#EAB4BF"
    yellow, blue = "#F8DF88", "#D9E2F3"
    green, grey = "#DDECCF", "#E4E4E2"

    def region(x, y, w, h, fc, ec):
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.025,rounding_size=0.08",
            facecolor=fc, edgecolor=ec, linewidth=1.2, zorder=0,
        )
        ax.add_patch(patch)

    def card_box(x, y, w, h, text, fc, fontsize=8.4, bold=False):
        pbox(
            ax, (x, y), w, h, text,
            fc=fc, ec=line_colour, fontsize=fontsize,
            weight="bold" if bold else "normal", lw=1.05, radius=0.045,
        )

    # Top notation block, matching the supplied reference hierarchy.
    ax.text(8.0, 7.55, "Notation & Definitions", ha="center", va="center", fontsize=11.5, fontweight="bold", color=DARK)
    ax.text(8.0, 7.30, "N: number of landmark queries  |  C: feature channels  |  H×W: spatial resolution", ha="center", va="center", fontsize=8.3, color=DARK)
    ax.text(8.0, 7.08, "FiLM: feature-wise linear modulation  |  Solid arrows: model path (training + inference)  |  Dashed arrows: training-only losses  |  Fetal configuration shown: B=3", ha="center", va="center", fontsize=8.3, color=DARK)

    # Stepped panel geometry from the reference image.
    region(0.10, 0.18, 6.05, 6.58, bg_a, edge_a)
    region(6.30, 3.76, 9.55, 3.00, bg_b, edge_b)
    region(6.30, 0.18, 9.55, 3.38, bg_c, edge_c)

    ax.text(0.28, 6.58, "A. Encoder and Feature Extraction", fontsize=10.8, fontweight="normal", color=DARK)
    ax.text(6.48, 6.58, "B. Repeated Pre-Block Prediction and Supervision", fontsize=10.8, fontweight="normal", color=DARK)
    ax.text(6.48, 3.34, "C. Final Landmark-Decoding Path", fontsize=10.8, fontweight="normal", color=DARK)

    # Panel A — top image/token/encoder route.
    card_box(0.34, 5.45, 1.25, 0.86, "Input image\n512×512", grey, fontsize=8.5)
    card_box(1.92, 5.38, 1.70, 1.00, "Patch embedding\nruntime 16×16\nC=384", peach, fontsize=7.8)
    card_box(
        3.92,
        5.38,
        1.88,
        1.00,
        "DINO ViT-S encoder\n12 blocks; C=384\njointly fine-tuned (LLRD)",
        yellow,
        fontsize=7.6,
    )
    arrow(ax, (1.59, 5.88), (1.92, 5.88), color=line_colour, lw=1.05, mutation=7)
    arrow(ax, (3.62, 5.88), (3.92, 5.88), color=line_colour, lw=1.05, mutation=7)

    ax.text(2.92, 4.82, "DINO encoder depth (expanded below)", ha="center", fontsize=8.2, color=DARK)
    depth_specs = [
        (0.36, 1.10, "Blocks 1–3"),
        (1.66, 1.24, "Blocks 4–8\n(taps after 4 & 8)"),
        (3.10, 0.76, "Block 9"),
        (4.08, 1.62, "Blocks 10–12\n(query-aware)"),
    ]
    for x, w, text_value in depth_specs:
        card_box(x, 4.06, w, 0.62, text_value, peach, fontsize=7.9)
    for x1, x2 in [(1.46, 1.66), (2.90, 3.10), (3.86, 4.08)]:
        arrow(ax, (x1, 4.37), (x2, 4.37), color=line_colour, lw=0.9, mutation=5)

    card_box(3.18, 2.88, 1.46, 0.94, "N learned queries\nC=384\ninserted before\nblock 10", blue, fontsize=7.0)
    arrow(
        ax, (4.18, 3.82), (4.71, 4.075),
        color=line_colour, lw=0.95, mutation=7,
        connectionstyle="arc3,rad=-0.14", zorder=4,
    )

    # F4/F8 are captured directly after their blocks. F12 is the final,
    # post-LayerNorm representation after block 12.
    for x in [2.08, 2.58]:
        dot(ax, (x, 4.06), "#7089A8", size=34)
        line(ax, (x, 4.02), (x, 2.10), color=line_colour, lw=0.9, style="-")
    dot(ax, (5.70, 4.37), "#7089A8", size=34)
    line(ax, (5.70, 4.33), (5.70, 2.10), color=line_colour, lw=0.9, style="-")
    ax.text(2.00, 3.91, "F4", ha="right", va="top", fontsize=6.4, fontweight="bold", color="#526B89")
    ax.text(2.66, 3.91, "F8", ha="left", va="top", fontsize=6.4, fontweight="bold", color="#526B89")
    ax.text(5.68, 4.00, "F12\n(post-norm)", ha="right", va="top", fontsize=6.1, fontweight="bold", color="#526B89", linespacing=1.0)
    line(ax, (2.08, 2.10), (5.70, 2.10), color=line_colour, lw=1.0, style="-")
    card_box(1.72, 1.18, 2.90, 0.68, "Feature taps F4, F8, F12\nC=384; H×W=32×32", blue, fontsize=8.2)
    arrow(ax, (3.36, 2.10), (3.36, 1.86), color=line_colour, lw=0.95, mutation=6)

    # Panel B — same horizontal structure and fork as the supplied layout.
    ax.text(10.67, 6.34, "Shared mask-head MLP + ScaleBlocks (same module as Panel C, not B-specific)", ha="center", va="center", fontsize=7.1, color=DARK)
    card_box(7.05, 5.10, 1.72, 0.94, "Pre-block states\nfor blocks 10, 11, 12", peach, fontsize=7.7)
    card_box(9.07, 5.10, 1.38, 0.94, "einsum\nhead ×3", pink, fontsize=8.3)
    card_box(10.75, 5.04, 1.26, 1.06, "Auxiliary maps\nN × 128×128\n(model output)", blue, fontsize=7.1)
    card_box(12.35, 5.30, 1.94, 0.72, "Hybrid loss ×3\nweighted MSE + coord L1\n(resized to 64×64)", yellow, fontsize=7.3)
    card_box(
        12.35,
        4.36,
        1.94,
        0.60,
        "Attention mask feeds\ncorresponding block\n(annealing disabled)",
        pink,
        fontsize=6.8,
    )
    arrow(ax, (8.77, 5.57), (9.07, 5.57), color=line_colour, lw=1.0, mutation=7)
    arrow(ax, (10.45, 5.57), (10.75, 5.57), color=line_colour, lw=1.0, mutation=7)
    arrow(ax, (12.01, 5.72), (12.35, 5.66), color=line_colour, lw=0.95, style="--", mutation=6)
    arrow(ax, (12.01, 5.34), (12.35, 4.66), color=line_colour, lw=0.95, mutation=6)
    arrow(ax, (5.70, 4.37), (7.05, 5.55), color=line_colour, lw=0.95, mutation=6, connectionstyle="arc3,rad=-0.10")

    # Panel C — long final route and lower query input.
    card_box(6.62, 1.96, 1.72, 0.82, "Multi-depth fusion\nF4, F8, F12\n32×32; C=384", green, fontsize=7.5)
    card_box(8.60, 1.96, 1.62, 0.82, "Shared ScaleBlocks ×2\n32×32 → 128×128\n(same module as B)", peach, fontsize=7.7)
    card_box(10.50, 1.84, 1.72, 1.06, "DeconvHeadV2\nFiLM conditioning\n3×3 convolutions", pink, fontsize=7.8)
    card_box(12.48, 1.96, 1.44, 0.82, "Final heatmaps\nN × 64×64", blue, fontsize=7.8)
    card_box(
        14.20,
        1.90,
        1.34,
        0.94,
        "Hard argmax\n+ quarter-pixel\nin heatmap space\nthen inverse map",
        yellow,
        fontsize=6.5,
    )
    for x1, x2 in [(8.34, 8.60), (10.22, 10.50), (12.22, 12.48), (13.92, 14.20)]:
        arrow(ax, (x1, 2.37), (x2, 2.37), color=line_colour, lw=1.0, mutation=7)
    arrow(ax, (4.62, 1.34), (6.62, 2.12), color=line_colour, lw=1.0, mutation=7, connectionstyle="arc3,rad=-0.06")
    card_box(10.52, 0.68, 1.70, 0.58, "Final queries\nN×384", blue, fontsize=8.0)
    arrow(ax, (11.37, 1.26), (11.37, 1.84), color=line_colour, lw=0.95, mutation=6)
    card_box(
        12.48,
        0.64,
        1.44,
        0.66,
        "Final hybrid loss\nweighted heatmap MSE\n+ coordinate L1",
        yellow,
        fontsize=6.45,
    )
    arrow(
        ax,
        (13.20, 1.96),
        (13.20, 1.30),
        color=line_colour,
        lw=0.95,
        style="--",
        mutation=6,
    )

    save(fig, "eomt_landmark_overall_architecture")


def _retired_draw_deconvheadv2_publication():
    raise RuntimeError(
        "Retired; the DeconvHeadV2 figure referenced from 03_method.tex is "
        "now figures/query_conditioned_deconvheadv2_architecture.png, "
        "authored directly in PowerPoint (no generator script/source in "
        "this repo). This function's output "
        "(deconvheadv2_internal_architecture.png/.pdf) is no longer "
        "\\includegraphics'd anywhere and also still carries the old "
        "gamma_q/beta_q notation."
    )
    fig, ax = plt.subplots(figsize=(15.6, 8.0))
    ax.set_xlim(0, 15.6)
    ax.set_ylim(0, 8.0)
    ax.axis("off")

    # Notation is deliberately outside the three stages so that it applies to
    # the whole figure and cannot be mistaken for a stage-specific operation.
    ax.text(
        7.80, 7.66,
        r"$B$ = batch size  ·  $N$ = landmark queries  ·  $C$ = channels  ·  $H\!\times\!W$ = spatial size",
        fontsize=12.2, color=DARK, ha="center", va="center",
    )

    # Three equal-height, warm-toned stage panels.
    panel_specs = [
        (0.16, 0.34, 4.66, 6.96, "#EAF2FA", "#5E86AD"),
        (4.98, 0.34, 5.05, 6.96, "#FFF1D9", "#C88A22"),
        (10.19, 0.34, 5.25, 6.96, "#F8EBEF", "#A86B7C"),
    ]
    for x, y, w, h, fc, ec in panel_specs:
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.018,rounding_size=0.11",
            facecolor=fc, edgecolor=ec, linewidth=1.35, zorder=0,
        ))

    ax.text(2.49, 6.94, "Stage 1: Upstream spatial features", ha="center", va="center", fontsize=13.5, fontweight="bold", color="#315F89")
    ax.text(7.50, 6.94, "Stage 2: Query conditioning", ha="center", va="center", fontsize=13.5, fontweight="bold", color="#A86513")
    ax.text(12.82, 6.94, "Stage 3: Shared spatial decoder", ha="center", va="center", fontsize=13.5, fontweight="bold", color="#7F4658")

    # Main feature stream.
    pbox(
        ax,
        (0.42, 4.66),
        1.72,
        1.28,
        "Optional upstream\nFPN-style fusion\n(final configuration)\n$C$ = 384; 32×32",
        fc="#CFE1F2",
        ec="#4778A5",
        fontsize=7.8,
        weight="bold",
    )
    pbox(ax, (2.63, 4.66), 1.76, 1.28, "ScaleBlocks ×2", fc="#DCEAF6", ec="#5E86AD", fontsize=10.0, weight="bold")
    arrow(ax, (2.14, 5.30), (2.63, 5.30), color="#4778A5", lw=1.35, mutation=9)
    label(ax, (3.51, 3.91), "spatial upsampling\n$H\\times W$: 32×32 → 64×64\n→ 128×128", color="#4E6680", fontsize=9.3)
    arrow(ax, (3.51, 4.34), (3.51, 4.66), color="#5E86AD", lw=1.2, mutation=8)

    pbox(ax, (5.23, 4.66), 1.54, 1.28, "Group\nnormalisation\n32 groups", fc="#FFE2B8", ec="#C17A18", fontsize=9.4)
    arrow(ax, (4.39, 5.30), (5.23, 5.30), color="#5E86AD", lw=1.35, mutation=9)

    # Query stream (bottom).
    pbox(ax, (5.22, 1.48), 1.58, 1.08, "Queries $q$\n$N$ queries\n$C$ = 384", fc="#FFE2B8", ec="#C17A18", fontsize=9.5, weight="bold")
    pbox(
        ax, (7.10, 1.22), 2.54, 1.58,
        "FiLM parameter network\nmultilayer perceptron\n384 → 384 → 2×384\nReLU · dropout 0.2\noutputs $γ_q$: scale\nand $β_q$: bias",
        fc="#FFD8B6", ec="#C77618", fontsize=9.0,
    )
    arrow(ax, (6.80, 2.02), (7.10, 2.02), color="#C17A18", lw=1.3, mutation=9)

    pbox(ax, (7.06, 4.31), 2.64, 1.88, "Feature-wise linear\nmodulation (FiLM)\n$F'_q = γ_q \\odot GN(F) + β_q$\n$B\\times N$ feature maps\n$C$ = 384; $H\\times W$ = 128×128", fc="#FFD2B1", ec="#C77618", fontsize=9.0, weight="bold")
    arrow(ax, (6.77, 5.30), (7.06, 5.30), color="#C17A18", lw=1.35, mutation=9)
    arrow(ax, (8.37, 2.80), (8.37, 4.31), color="#C77618", lw=1.25, mutation=9)

    # Shared spatial decoder.
    pbox(ax, (10.48, 4.23), 2.16, 2.02, "Shared decoder\n3×3 convolution\n$C$: 384 → 128\nReLU\n3×3 convolution\n$C$: 128 → 1", fc="#EAB9C5", ec="#91495F", fontsize=9.2, weight="bold")
    arrow(ax, (9.70, 5.30), (10.48, 5.30), color="#91495F", lw=1.4, mutation=9)
    pbox(ax, (12.95, 4.56), 2.13, 1.36, "Bilinear resize\n$H\\times W$\n128×128 → 64×64", fc="#F2CFD8", ec="#A45E72", fontsize=9.4)
    arrow(ax, (12.64, 5.30), (12.95, 5.30), color="#91495F", lw=1.3, mutation=9)
    pbox(ax, (12.95, 2.68), 2.13, 1.04, "Output\n$N$ heatmaps\n64×64 each", fc="#F0C6D1", ec="#A45E72", fontsize=9.6, weight="bold")
    arrow(ax, (14.02, 4.56), (14.02, 3.72), color="#A45E72", lw=1.3, mutation=9)

    pbox(
        ax, (0.62, 1.04), 3.58, 1.34,
        "ScaleBlock internals\n2×2 transposed convolution\nstride 2 → GELU\n3×3 depthwise convolution\n→ 2D layer normalisation",
        fc="#D9E8F5", ec="#5E86AD", fontsize=9.1,
    )
    pbox(
        ax, (10.68, 1.06), 4.20, 0.96,
        "Initialisation and scope\nSecond convolution: weights 0, bias 0.01\nNo final-head attention-mask feedback",
        fc="#EECBD4", ec="#91495F", fontsize=9.1,
    )

    save(fig, "deconvheadv2_internal_architecture")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    draw_overall_architecture_reference_layout()
    print(f"[OK] architecture figures written to {OUT_DIR}")
