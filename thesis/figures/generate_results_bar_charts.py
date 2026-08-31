#!/usr/bin/env python3
"""Generate thesis-ready Results figures from frozen experiment values.

Figure 5.1 reads the unrounded means and seed-level sample SDs directly
from the frozen four-model TSV. Figure 5.2 uses the explicitly frozen
five-stage BPD development summary below. No statistic is recomputed from
rounded prose or Markdown values.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN_TABLE = (
    REPO_ROOT
    / "final_comparison"
    / "four_model_freeze_20260812_v1"
    / "four_model_main_table.tsv"
)
OUTPUT_DIR = Path(__file__).resolve().parent

TASKS = ["BPD", "OFD", "APAD", "TAD", "FL"]
DATASETS = ["UCL", "MULTICENTRE"]
SOURCE_METHODS = [
    "Proposed-DINOv2",
    "Proposed-DINOv3",
    "HRNet-W18",
    "RTMPose-s",
]
DISPLAY_METHODS = [
    "QCLand-DINOv2",
    "QCLand-DINOv3",
    "HRNet-W18",
    "RTMPose-s",
]
COLORS = ["#4477AA", "#EE7733", "#228833", "#777777"]
HATCHES = ["", "", "", "//"]

# Exact frozen summary specified for the staged BPD development sequence.
BPD_STAGE_LABELS = [
    "Einsum\nhead",
    "DeconvHeadV2",
    "+Fusion",
    "+Fusion +\nAlignment",
    "+Rotation/Scale\nAugmentation",
]
BPD_STAGE_MEANS = np.array([11.23, 8.12, 9.00, 8.96, 5.92])
BPD_STAGE_SDS = np.array([2.06, 1.42, 2.20, 2.30, 0.66])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9.5,
            "axes.labelsize": 10,
            "axes.titlesize": 10.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_frozen_table() -> dict[tuple[str, str, str], tuple[float, float]]:
    with FROZEN_TABLE.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len(rows) == 40, f"Expected 40 frozen rows, found {len(rows)}"

    values: dict[tuple[str, str, str], tuple[float, float]] = {}
    for row in rows:
        key = (row["dataset"].upper(), row["task"], row["method"])
        assert key not in values, f"Duplicate frozen row: {key}"
        values[key] = (
            float(row["pi_nme_5seed_mean_pct"]),
            float(row["pi_nme_seed_sample_sd_pct"]),
        )

    expected = {
        (dataset, task, method)
        for dataset in DATASETS
        for task in TASKS
        for method in SOURCE_METHODS
    }
    assert set(values) == expected, (
        f"Frozen-table key mismatch; missing={sorted(expected - set(values))}, "
        f"extra={sorted(set(values) - expected)}"
    )
    return values


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.01)
    fig.savefig(
        OUTPUT_DIR / f"{stem}.png",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.01,
    )
    plt.close(fig)


def make_final_comparison(values: dict[tuple[str, str, str], tuple[float, float]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.05), sharey=False)
    x = np.arange(len(TASKS))
    width = 0.19
    offsets = (np.arange(len(SOURCE_METHODS)) - 1.5) * width

    for panel_index, (ax, dataset) in enumerate(zip(axes, DATASETS)):
        for method_index, method in enumerate(SOURCE_METHODS):
            means = np.array([values[(dataset, task, method)][0] for task in TASKS])
            sds = np.array([values[(dataset, task, method)][1] for task in TASKS])
            ax.bar(
                x + offsets[method_index],
                means,
                width,
                yerr=sds,
                label=DISPLAY_METHODS[method_index],
                color=COLORS[method_index],
                edgecolor="#333333",
                linewidth=0.45,
                hatch=HATCHES[method_index],
                error_kw={"elinewidth": 0.8, "ecolor": "#333333", "capsize": 2.2, "capthick": 0.8},
                zorder=3,
            )

        ax.set_xticks(x, TASKS)
        ax.set_xlim(-0.55, len(TASKS) - 0.45)
        ax.set_ylim(0, 22.5 if dataset == "UCL" else 10.5)
        ax.set_ylabel(r"PI-NME (\%) $\downarrow$", fontsize=9.2)
        ax.set_title(f"({'ab'[panel_index]}) {'UCL' if dataset == 'UCL' else 'Multicentre'}", pad=4)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.75, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", length=0, pad=4)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=4,
        frameon=False,
        handlelength=1.8,
        columnspacing=1.25,
    )
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.16, top=0.76, wspace=0.25)
    save_figure(fig, "final_four_model_comparison")


def make_bpd_progression() -> None:
    fig, ax = plt.subplots(figsize=(7.0, 3.15))
    x = np.arange(len(BPD_STAGE_LABELS))
    # These bars form one staged development chain rather than five unrelated
    # methods. Neutral intermediate stages keep that semantics clear; colour
    # highlights the principal architectural change and the final recipe.
    stage_colors = ["#B8B8B8", "#4477AA", "#D2D7DC", "#BFC7CD", "#EE7733"]
    bars = ax.bar(
        x,
        BPD_STAGE_MEANS,
        width=0.64,
        yerr=BPD_STAGE_SDS,
        color=stage_colors,
        edgecolor="#333333",
        linewidth=0.55,
        error_kw={"elinewidth": 0.9, "ecolor": "#333333", "capsize": 3.0, "capthick": 0.9},
        zorder=3,
    )

    ax.set_xticks(x, BPD_STAGE_LABELS)
    ax.set_ylabel(r"PI-NME (\%) $\downarrow$", fontsize=9.2)
    ax.set_ylim(0, 14.5)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.75, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", length=0, pad=5)
    ax.get_xticklabels()[1].set_fontweight("bold")

    for bar, mean, sd in zip(bars, BPD_STAGE_MEANS, BPD_STAGE_SDS):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            mean + sd + 0.30,
            f"{mean:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.subplots_adjust(left=0.09, right=0.995, bottom=0.22, top=0.98)
    save_figure(fig, "bpd_staged_development")


def write_manifest() -> None:
    manifest = OUTPUT_DIR / "results_bar_charts_manifest.tsv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["figure", "data_source", "source_sha256", "statistics"])
        writer.writerow(
            [
                "final_four_model_comparison",
                str(FROZEN_TABLE.relative_to(REPO_ROOT)).replace("\\", "/"),
                sha256(FROZEN_TABLE),
                "unrounded frozen five-seed means and seed-level sample SDs",
            ]
        )
        writer.writerow(
            [
                "bpd_staged_development",
                "exact frozen values embedded in generate_results_bar_charts.py",
                sha256(Path(__file__)),
                "provided five-seed means and seed-level sample SDs; no recomputation",
            ]
        )


def main() -> None:
    configure_style()
    values = load_frozen_table()
    make_final_comparison(values)
    make_bpd_progression()
    write_manifest()


if __name__ == "__main__":
    main()
