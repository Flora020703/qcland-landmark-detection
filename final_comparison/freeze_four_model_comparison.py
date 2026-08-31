#!/usr/bin/env python3
"""Freeze the four-model fetal landmark comparison from per-image evidence.

Inputs are deliberately lightweight:

* ``--canonical-root`` is the already audited ``common_subset`` directory
  produced by ``rescore_landmark_conventions.py``.  It contains one
  five-seed-mean per-image CSV for Proposed-DINOv2, Proposed-DINOv3 and
  HRNet-W18, plus ``endpoint_ordering_summary.tsv`` with seed-level SDs.
* ``--rtmpose-root`` contains the extracted lightweight RTMPose run files
  (or the full extracted cell archives).  Files are discovered recursively.

The script refuses to reuse the canonical seed summaries if adding RTMPose
changes the canonical filename set.  That guard is important: otherwise a
mean/SD from one subset could silently be printed beside paired statistics
from another subset.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.stats import wilcoxon


DATASETS = ("UCL", "MULTICENTRE")
TASKS = ("BPD", "OFD", "APAD", "TAD", "FL")
SEEDS = (42, 0, 123, 2024, 3407)
METHODS = ("Proposed-DINOv2", "Proposed-DINOv3", "HRNet-W18", "RTMPose-s")
CANONICAL_METHOD = {
    "Proposed-DINOv2": "eomt_dinov2",
    "Proposed-DINOv3": "eomt_dinov3",
    "HRNet-W18": "hrnet",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--canonical-root", type=Path, required=True)
    p.add_argument("--rtmpose-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--bootstrap-replicates", type=int, default=20000)
    p.add_argument("--bootstrap-seed", type=int, default=20260812)
    return p.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def unique_map(rows: Iterable[dict[str, str]], path: Path) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        name = Path(row["filename"]).name
        if name in result:
            raise ValueError(f"duplicate filename {name!r} in {path}")
        result[name] = row
    if not result:
        raise ValueError(f"empty per-image file: {path}")
    return result


def canonical_file(root: Path, dataset: str, task: str, method: str) -> Path:
    stem = f"{dataset.lower()}_{task.lower()}_{CANONICAL_METHOD[method]}_per_image.csv"
    return root / stem


def rtmpose_file(root: Path, dataset: str, task: str, seed: int) -> Path:
    run = f"{dataset}_{task}_seed{seed}_run"
    matches = sorted(root.rglob(f"{run}_per_image.csv"))
    if dataset == "UCL" and task == "BPD" and seed == 42 and not matches:
        matches = sorted(root.rglob("UCL_BPD_seed42_canary_per_image.csv"))
    if len(matches) != 1:
        raise ValueError(f"expected exactly one RTMPose CSV for {run}, found {matches}")
    return matches[0]


def canonical_values(path: Path) -> dict[str, float]:
    rows = unique_map(read_csv(path), path)
    column = "permutation_invariant_nme_pct"
    if column not in next(iter(rows.values())):
        raise ValueError(f"{path} lacks {column}")
    return {name: float(row[column]) for name, row in rows.items()}


def rtmpose_seed_values(path: Path) -> dict[str, float]:
    rows = unique_map(read_csv(path), path)
    if "swap_min_nme" not in next(iter(rows.values())):
        raise ValueError(f"{path} lacks swap_min_nme")
    # RTMPose stores NME as a fraction; canonical files store percent.
    return {name: 100.0 * float(row["swap_min_nme"]) for name, row in rows.items()}


def holm(raw: list[float]) -> list[float]:
    m = len(raw)
    order = sorted(range(m), key=raw.__getitem__)
    adjusted = [0.0] * m
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (m - rank) * raw[index]))
        adjusted[index] = running
    return adjusted


def bootstrap_ci(values: np.ndarray, reps: int, rng: np.random.Generator) -> tuple[float, float]:
    n = len(values)
    # Chunking avoids allocating reps*n for the large Multicentre cells.
    means: list[np.ndarray] = []
    remaining = reps
    while remaining:
        count = min(1000, remaining)
        indices = rng.integers(0, n, size=(count, n))
        means.append(values[indices].mean(axis=1))
        remaining -= count
    samples = np.concatenate(means)
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def safe_wilcoxon(values: np.ndarray) -> float:
    if np.allclose(values, 0.0):
        return 1.0
    return float(wilcoxon(values, zero_method="pratt", alternative="two-sided").pvalue)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_rows(path: Path, rows: list[dict[str, object]], delimiter: str = "\t") -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    a = parse_args()
    a.output_root.mkdir(parents=True, exist_ok=False)
    per_image_root = a.output_root / "per_image"
    per_image_root.mkdir()

    summary_path = a.canonical_root / "endpoint_ordering_summary.tsv"
    summary_rows = read_tsv(summary_path)
    summary_index = {
        (r["dataset"].upper(), r["task"].upper(), r["method"]): r
        for r in summary_rows
    }

    audit: list[dict[str, object]] = []
    main_table: list[dict[str, object]] = []
    comparisons: list[dict[str, object]] = []
    rng = np.random.default_rng(a.bootstrap_seed)

    for dataset, task in itertools.product(DATASETS, TASKS):
        canonical: dict[str, dict[str, float]] = {}
        canonical_sets: dict[str, set[str]] = {}
        for method in CANONICAL_METHOD:
            path = canonical_file(a.canonical_root, dataset, task, method)
            values = canonical_values(path)
            canonical[method] = values
            canonical_sets[method] = set(values)

        rtm_seeds: dict[int, dict[str, float]] = {}
        rtm_paths: list[Path] = []
        for seed in SEEDS:
            path = rtmpose_file(a.rtmpose_root, dataset, task, seed)
            rtm_paths.append(path)
            rtm_seeds[seed] = rtmpose_seed_values(path)

        sets = list(canonical_sets.values()) + [set(x) for x in rtm_seeds.values()]
        common = set.intersection(*sets)
        union = set.union(*sets)
        canonical_reference = next(iter(canonical_sets.values()))
        all_canonical_equal = all(s == canonical_reference for s in canonical_sets.values())
        rtm_all_equal = all(set(v) == set(next(iter(rtm_seeds.values()))) for v in rtm_seeds.values())

        # Seed SDs in endpoint_ordering_summary.tsv are valid only for its
        # exact canonical common subset.  Do not silently reuse them after a
        # four-model intersection changes that subset.
        if not all_canonical_equal:
            raise ValueError(f"canonical three-method filename sets differ for {dataset}/{task}")
        if common != canonical_reference:
            missing = sorted(canonical_reference - common)[:10]
            raise ValueError(
                f"RTMPose changes canonical common subset for {dataset}/{task}; "
                f"seed-level proposed/HRNet reaggregation required; examples={missing}"
            )

        rtm_per_image = {
            name: float(np.mean([rtm_seeds[s][name] for s in SEEDS])) for name in sorted(common)
        }
        values_by_method = {**canonical, "RTMPose-s": rtm_per_image}

        audit.append({
            "dataset": dataset,
            "task": task,
            "n_union": len(union),
            "n_common": len(common),
            "canonical_sets_equal": all_canonical_equal,
            "rtmpose_seed_sets_equal": rtm_all_equal,
            "four_model_sets_exactly_equal": all(s == common for s in sets),
            "rtmpose_seed_count": len(rtm_seeds),
            "status": "PASS",
        })

        paired_rows = []
        for name in sorted(common):
            row: dict[str, object] = {"filename": name}
            for method in METHODS:
                row[method] = f"{values_by_method[method][name]:.8f}"
            paired_rows.append(row)
        write_rows(per_image_root / f"{dataset.lower()}_{task.lower()}_four_model.csv", paired_rows, ",")

        for method in METHODS:
            image_values = np.array([values_by_method[method][x] for x in sorted(common)])
            if method == "RTMPose-s":
                seed_means = np.array([
                    np.mean([rtm_seeds[s][x] for x in common]) for s in SEEDS
                ])
                mean = float(seed_means.mean())
                sd = float(seed_means.std(ddof=1))
            else:
                key = (dataset, task, CANONICAL_METHOD[method])
                if key not in summary_index:
                    raise ValueError(f"missing canonical summary row {key}")
                source = summary_index[key]
                mean = float(source["permutation_invariant_nme_5seed_mean_pct"])
                sd = float(source["permutation_invariant_nme_5seed_sample_sd_pct"])
            # The per-image five-seed mean and mean of five seed means must agree.
            if not math.isclose(mean, float(image_values.mean()), abs_tol=1e-6):
                raise ValueError(
                    f"aggregate/per-image mismatch {dataset}/{task}/{method}: "
                    f"{mean} vs {image_values.mean()}"
                )
            main_table.append({
                "dataset": dataset,
                "task": task,
                "method": method,
                "n_images": len(common),
                "pi_nme_5seed_mean_pct": f"{mean:.8f}",
                "pi_nme_seed_sample_sd_pct": f"{sd:.8f}",
                "display_mean_sd_pct": f"{mean:.2f} ± {sd:.2f}",
            })

        for method_a, method_b in itertools.combinations(METHODS, 2):
            diff = np.array([
                values_by_method[method_a][x] - values_by_method[method_b][x]
                for x in sorted(common)
            ])
            lo, hi = bootstrap_ci(diff, a.bootstrap_replicates, rng)
            comparisons.append({
                "dataset": dataset,
                "task": task,
                "method_a": method_a,
                "method_b": method_b,
                "n_images": len(common),
                "mean_difference_a_minus_b_pp": f"{diff.mean():.8f}",
                "paired_bootstrap_95ci_low_pp": f"{lo:.8f}",
                "paired_bootstrap_95ci_high_pp": f"{hi:.8f}",
                "wilcoxon_p_raw": f"{safe_wilcoxon(diff):.12g}",
                "wilcoxon_p_holm_within_cell": "",
                "wilcoxon_p_holm_global_60": "",
            })

    if len(audit) != 10 or len(main_table) != 40 or len(comparisons) != 60:
        raise AssertionError((len(audit), len(main_table), len(comparisons)))

    for start in range(0, len(comparisons), 6):
        cell = comparisons[start:start + 6]
        adjusted = holm([float(r["wilcoxon_p_raw"]) for r in cell])
        for row, value in zip(cell, adjusted):
            row["wilcoxon_p_holm_within_cell"] = f"{value:.12g}"
    global_adjusted = holm([float(r["wilcoxon_p_raw"]) for r in comparisons])
    for row, value in zip(comparisons, global_adjusted):
        row["wilcoxon_p_holm_global_60"] = f"{value:.12g}"

    write_rows(a.output_root / "four_model_asset_audit.tsv", audit)
    write_rows(a.output_root / "four_model_main_table.tsv", main_table)
    write_rows(a.output_root / "four_model_paired_statistics.tsv", comparisons)

    metadata = {
        "metric": "original-image-space permutation-invariant NME (%)",
        "seeds": list(SEEDS),
        "methods": list(METHODS),
        "bootstrap_replicates": a.bootstrap_replicates,
        "bootstrap_seed": a.bootstrap_seed,
        "holm_families": ["six comparisons within each dataset-task cell", "all 60 comparisons"],
        "n_cells": 10,
        "n_main_rows": 40,
        "n_pairwise_rows": 60,
    }
    (a.output_root / "FREEZE_METADATA.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    manifest_files = sorted(p for p in a.output_root.rglob("*") if p.is_file())
    manifest = [{"sha256": sha256(p), "relative_path": p.relative_to(a.output_root).as_posix()}
                for p in manifest_files]
    write_rows(a.output_root / "FREEZE_MANIFEST.sha256.tsv", manifest)
    print(f"[COMPLETE] froze 10 cells, 40 main-table rows and 60 paired comparisons under {a.output_root}")


if __name__ == "__main__":
    main()
