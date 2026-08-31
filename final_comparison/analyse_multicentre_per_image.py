#!/usr/bin/env python3
"""Paired Multicentre analysis for EoMT-DINOv2/v3 versus HRNet.

The image, rather than the seed/run prediction, is the statistical unit.
Predictions are first averaged across the five pre-specified seeds per image.
The script also quantifies EoMT's fixed-channel minus swap-min identity gap.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


SEEDS = (42, 0, 123, 2024, 3407)
BACKBONES = ("dinov2", "dinov3")
TASKS = {
    # task: (HRNet config tag, raw EoMT test rows, common valid rows)
    "bpd": ("brain_BPD", 1191, 1180),
    "ofd": ("brain_OFD", 1191, 1189),
    "apad": ("abdomen_APAD", 161, 161),
    "tad": ("abdomen_TAD", 161, 161),
    "fl": ("femur_FL", 362, 362),
}


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eomt-root",
        type=Path,
        default=Path("/root/autodl-tmp/saved_checkpoints/multicentre_5seed"),
    )
    parser.add_argument(
        "--hrnet-root",
        type=Path,
        default=Path("/root/autodl-tmp/hrnet_512_fixed_5seed/output/FETAL"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/root/autodl-tmp/hrnet_512_fixed_5seed/paired_analysis"),
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260802)
    return parser.parse_args()


def rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def name(value: str) -> str:
    return value.strip().replace("\\", "/").rsplit("/", 1)[-1]


def unique_map(items: list[dict[str, str]], key: str, path: Path) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for item in items:
        value = name(item[key])
        if value in result:
            raise ValueError(f"duplicate {key}={value} in {path}")
        result[value] = item
    return result


def load_hrnet(root: Path, task_tag: str) -> dict[str, tuple[float, float]]:
    by_seed: list[dict[str, dict]] = []
    for seed in SEEDS:
        run = f"fetal_landmark_hrnet_w18_MULTICENTRE_{task_tag}_seed{seed}_512fixed"
        path = root / run / "fixed_channel_per_image.csv"
        by_seed.append(unique_map(rows(path), "filename", path))
    keys = set(by_seed[0])
    if any(set(item) != keys for item in by_seed[1:]):
        raise ValueError(f"HRNet filename sets differ across seeds for {task_tag}")
    return {
        filename: (
            float(np.mean([float(item[filename]["fixed_channel_nme"]) for item in by_seed])),
            float(np.mean([float(item[filename]["swap_min_nme"]) for item in by_seed])),
        )
        for filename in keys
    }


def load_eomt(
    root: Path, task: str, backbone: str, expected_raw_n: int
) -> dict[str, tuple[float, float]]:
    per_seed: list[dict[str, tuple[float, float]]] = []
    for seed in SEEDS:
        run = root / f"multicentre-{task}-{backbone}" / f"seed{seed}"
        order_path = run / "test_image_order.csv"
        fixed_path = run / f"seed{seed}_final_fixedchannel_per_image.csv"
        swap_path = run / f"seed{seed}_final_swapmin_per_image.csv"
        order = {int(item["index"]): name(item["filename"]) for item in rows(order_path)}
        if len(order) != expected_raw_n:
            raise ValueError(
                f"expected {expected_raw_n} unique order rows in {order_path}; "
                f"got {len(order)}"
            )
        fixed = {int(item["index"]): float(item["nme"]) for item in rows(fixed_path)}
        swap = {int(item["index"]): float(item["nme"]) for item in rows(swap_path)}
        if set(order) != set(fixed) or set(order) != set(swap):
            raise ValueError(f"index mismatch under {run}")
        joined = {order[index]: (fixed[index], swap[index]) for index in order}
        if len(joined) != expected_raw_n:
            raise ValueError(f"duplicate filenames under {run}")
        per_seed.append(joined)
    keys = set(per_seed[0])
    if any(set(item) != keys for item in per_seed[1:]):
        raise ValueError(f"EoMT filename sets differ for {task}/{backbone}")
    return {
        filename: (
            float(np.mean([item[filename][0] for item in per_seed])),
            float(np.mean([item[filename][1] for item in per_seed])),
        )
        for filename in keys
    }


def bootstrap_ci(values: np.ndarray, replicates: int, rng: np.random.Generator) -> tuple[float, float]:
    # Chunking avoids allocating replicates x n all at once for large Head sets.
    means = np.empty(replicates, dtype=np.float64)
    chunk = 1000
    for start in range(0, replicates, chunk):
        stop = min(start + chunk, replicates)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def holm(raw: list[float]) -> list[float]:
    order = np.argsort(raw)
    adjusted = np.empty(len(raw), dtype=np.float64)
    running = 0.0
    m = len(raw)
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * raw[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def main() -> None:
    cfg = args()
    cfg.output_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.bootstrap_seed)
    summaries: list[dict] = []
    per_image_paths: list[Path] = []

    for task, (task_tag, expected_raw_n, expected_n) in TASKS.items():
        hrnet = load_hrnet(cfg.hrnet_root, task_tag)
        if len(hrnet) != expected_n:
            raise ValueError(f"{task}: HRNet n={len(hrnet)} != expected {expected_n}")
        for backbone in BACKBONES:
            eomt = load_eomt(cfg.eomt_root, task, backbone, expected_raw_n)
            common = sorted(set(hrnet) & set(eomt))
            if len(common) != expected_n or set(common) != set(hrnet):
                raise ValueError(
                    f"{task}/{backbone}: common n={len(common)} != expected {expected_n}"
                )
            e_fixed = np.array([eomt[x][0] for x in common]) * 100.0
            e_swap = np.array([eomt[x][1] for x in common]) * 100.0
            h_fixed = np.array([hrnet[x][0] for x in common]) * 100.0
            h_swap = np.array([hrnet[x][1] for x in common]) * 100.0
            if np.any(e_fixed + 1e-10 < e_swap) or np.any(h_fixed + 1e-10 < h_swap):
                raise ValueError(f"fixed >= swap invariant failed for {task}/{backbone}")

            difference = e_fixed - h_fixed
            low, high = bootstrap_ci(difference, cfg.bootstrap_replicates, rng)
            try:
                p_value = float(wilcoxon(difference, alternative="two-sided").pvalue)
            except ValueError:
                p_value = 1.0
            identity_gap = e_fixed - e_swap
            summaries.append({
                "task": task,
                "backbone": backbone,
                "n_images": len(common),
                "eomt_fixed_mean_pct": f"{e_fixed.mean():.8f}",
                "hrnet_fixed_mean_pct": f"{h_fixed.mean():.8f}",
                "eomt_minus_hrnet_mean_pp": f"{difference.mean():.8f}",
                "bootstrap_95ci_low_pp": f"{low:.8f}",
                "bootstrap_95ci_high_pp": f"{high:.8f}",
                "wilcoxon_p_raw": f"{p_value:.12g}",
                "wilcoxon_p_holm": "",
                "eomt_fixed_minus_swap_mean_pp": f"{identity_gap.mean():.8f}",
                "eomt_fixed_minus_swap_median_pp": f"{np.median(identity_gap):.8f}",
                "eomt_swap_preferred_fraction": f"{np.mean(identity_gap > 1e-10):.8f}",
                "hrnet_fixed_minus_swap_mean_pp": f"{(h_fixed-h_swap).mean():.8f}",
            })

            output = cfg.output_root / f"{task}_{backbone}_paired_per_image.csv"
            with output.open("w", newline="", encoding="utf-8") as handle:
                fields = [
                    "filename", "eomt_fixed_nme_pct", "eomt_swap_nme_pct",
                    "hrnet_fixed_nme_pct", "hrnet_swap_nme_pct",
                    "eomt_minus_hrnet_fixed_pp", "eomt_fixed_minus_swap_pp",
                ]
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for index, filename in enumerate(common):
                    writer.writerow({
                        "filename": filename,
                        "eomt_fixed_nme_pct": f"{e_fixed[index]:.8f}",
                        "eomt_swap_nme_pct": f"{e_swap[index]:.8f}",
                        "hrnet_fixed_nme_pct": f"{h_fixed[index]:.8f}",
                        "hrnet_swap_nme_pct": f"{h_swap[index]:.8f}",
                        "eomt_minus_hrnet_fixed_pp": f"{difference[index]:.8f}",
                        "eomt_fixed_minus_swap_pp": f"{identity_gap[index]:.8f}",
                    })
            per_image_paths.append(output)

    adjusted = holm([float(item["wilcoxon_p_raw"]) for item in summaries])
    for item, value in zip(summaries, adjusted):
        item["wilcoxon_p_holm"] = f"{value:.12g}"
    fields = list(summaries[0])
    summary_path = cfg.output_root / "multicentre_paired_summary.tsv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(summaries)
    if len(summaries) != 10 or len(per_image_paths) != 10:
        raise RuntimeError("expected ten task/backbone comparisons")
    print(f"[COMPLETE] wrote {summary_path} and ten paired per-image CSVs")


if __name__ == "__main__":
    main()
