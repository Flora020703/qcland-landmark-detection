#!/usr/bin/env python3
"""Render thesis-facing Markdown and LaTeX from the immutable freeze TSVs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


PROPOSED = ("Proposed-DINOv2", "Proposed-DINOv3")
BASELINES = ("HRNet-W18", "RTMPose-s")
METHOD_LABEL = {
    "Proposed-DINOv2": "Proposed--DINOv2",
    "Proposed-DINOv3": "Proposed--DINOv3",
    "HRNet-W18": "HRNet-W18",
    "RTMPose-s": "RTMPose-s",
}
DATASET_LABEL = {"UCL": "UCL", "MULTICENTRE": "Multicentre"}
TASKS = ("BPD", "OFD", "APAD", "TAD", "FL")


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def p_fmt(value: float) -> str:
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    main_rows = rows(args.freeze_root / "four_model_main_table.tsv")
    stat_rows = rows(args.freeze_root / "four_model_paired_statistics.tsv")
    if len(main_rows) != 40 or len(stat_rows) != 60:
        raise ValueError("freeze row counts are not 40/60")

    main_idx = {(r["dataset"], r["task"], r["method"]): r for r in main_rows}
    stat_idx = {
        (r["dataset"], r["task"], r["method_a"], r["method_b"]): r
        for r in stat_rows
    }

    md = [
        "# Thesis-facing frozen four-model tables",
        "",
        "Primary metric: original-image-space permutation-invariant NME (%). "
        "Main values are five-seed mean ± seed-level sample SD; lower is better.",
        "",
        "## Main comparison",
        "",
        "| Dataset | Measurement | Proposed--DINOv2 | Proposed--DINOv3 | HRNet-W18 | RTMPose-s | n |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    latex = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Original-image-space permutation-invariant NME (\%), reported as mean $\pm$ sample standard deviation across five seeds. Lower is better.}",
        r"\label{tab:four_model_main}",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Dataset & Measurement & Proposed--DINOv2 & Proposed--DINOv3 & HRNet-W18 & RTMPose-s & $n$ \\",
        r"\midrule",
    ]
    for dataset in ("UCL", "MULTICENTRE"):
        for task in TASKS:
            cell = [main_idx[(dataset, task, m)] for m in METHOD_LABEL]
            means = [float(r["pi_nme_5seed_mean_pct"]) for r in cell]
            best = min(range(4), key=means.__getitem__)
            displays = [r["display_mean_sd_pct"] for r in cell]
            md_displays = [f"**{x}**" if i == best else x for i, x in enumerate(displays)]
            md.append(
                f"| {DATASET_LABEL[dataset]} | {task} | "
                + " | ".join(md_displays)
                + f" | {cell[0]['n_images']} |"
            )
            tex_displays = [x.replace("±", r"$\pm$") for x in displays]
            tex_displays = [r"\textbf{" + x + "}" if i == best else x for i, x in enumerate(tex_displays)]
            latex.append(
                f"{DATASET_LABEL[dataset]} & {task} & "
                + " & ".join(tex_displays)
                + f" & {cell[0]['n_images']} " + r"\\"
            )
        if dataset == "UCL":
            latex.append(r"\midrule")
    latex += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]

    md += [
        "",
        "## Pre-specified proposed-versus-baseline paired contrasts",
        "",
        "Effects are method A minus method B in NME percentage points; negative values favour the proposed method. "
        "CI is a 20,000-replicate paired image-bootstrap 95% interval. "
        "The displayed p-value is Wilcoxon signed-rank with global Holm correction across all 60 frozen pairwise contrasts.",
        "",
        "| Dataset | Measurement | Contrast (A - B) | Difference, pp [95% CI] | Holm-60 p |",
        "|---|---|---|---:|---:|",
    ]
    selected = []
    for dataset in ("UCL", "MULTICENTRE"):
        for task in TASKS:
            for proposed in PROPOSED:
                for baseline in BASELINES:
                    row = stat_idx[(dataset, task, proposed, baseline)]
                    selected.append(row)
                    diff = float(row["mean_difference_a_minus_b_pp"])
                    low = float(row["paired_bootstrap_95ci_low_pp"])
                    high = float(row["paired_bootstrap_95ci_high_pp"])
                    p = float(row["wilcoxon_p_holm_global_60"])
                    md.append(
                        f"| {DATASET_LABEL[dataset]} | {task} | "
                        f"{METHOD_LABEL[proposed]} - {METHOD_LABEL[baseline]} | "
                        f"{diff:.2f} [{low:.2f}, {high:.2f}] | {p_fmt(p)} |"
                    )
    if len(selected) != 40:
        raise AssertionError(len(selected))

    md += [
        "",
        "The full six contrasts per cell, including Proposed--DINOv2 versus Proposed--DINOv3 and HRNet-W18 versus RTMPose-s, "
        "remain authoritative in `four_model_paired_statistics.tsv` and should be supplied as supplementary material.",
        "",
    ]
    (args.output_root / "FOUR_MODEL_THESIS_TABLES.md").write_text("\n".join(md), encoding="utf-8")
    (args.output_root / "four_model_main_table.tex").write_text("\n".join(latex) + "\n", encoding="utf-8")
    print("[COMPLETE] wrote Markdown main/statistics tables and LaTeX main table")


if __name__ == "__main__":
    main()
