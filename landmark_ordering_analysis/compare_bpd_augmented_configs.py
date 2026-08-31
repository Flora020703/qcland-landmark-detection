#!/usr/bin/env python3
"""Paired per-image comparison of the two augmented UCL BPD configurations.

The inferential observation is an image after first averaging PI-NME over
the same five training seeds.  Positive differences mean that the simpler
DeconvHeadV2+augmentation condition has higher error, hence favour the
FPN+UDP+augmentation condition.

The DeconvHeadV2-only input must come from the fixed-validation rerun in
which training-time validation and early stopping used fixed-channel NME,
matching the historical FPN+UDP+augmentation runs.  The mean assertion below
deliberately rejects the superseded 6.98779690% run.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_column(path: Path, column: str) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "filename" not in rows[0] or column not in rows[0]:
        raise SystemExit(f"[ERROR] {path} lacks filename/{column}")
    result: dict[str, float] = {}
    for row in rows:
        name = row["filename"].strip().replace("\\", "/").rsplit("/", 1)[-1]
        if name in result:
            raise SystemExit(f"[ERROR] duplicate filename in {path}: {name}")
        result[name] = float(row[column])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deconv-aug-csv", type=Path, required=True)
    parser.add_argument("--fpn-udp-aug-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260812)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise SystemExit(f"[ERROR] output directory already exists: {args.output_dir}")

    simple = read_column(args.deconv_aug_csv, "permutation_invariant_nme_pct")
    full = read_column(args.fpn_udp_aug_csv, "Proposed-DINOv2")
    if set(simple) != set(full):
        only_simple = sorted(set(simple) - set(full))
        only_full = sorted(set(full) - set(simple))
        raise SystemExit(
            "[ERROR] filename sets differ; "
            f"only_deconv_aug={only_simple}, only_fpn_udp_aug={only_full}"
        )
    names = sorted(simple)
    if len(names) != 49:
        raise SystemExit(f"[ERROR] expected 49 matched UCL BPD images, found {len(names)}")

    a = np.asarray([simple[name] for name in names], dtype=np.float64)
    b = np.asarray([full[name] for name in names], dtype=np.float64)
    diff = a - b
    mean_a, mean_b, mean_diff = map(float, (a.mean(), b.mean(), diff.mean()))
    if not np.isclose(mean_a, 6.30202067, rtol=0.0, atol=5e-7):
        raise SystemExit(
            "[ERROR] fixed-validation DeconvHeadV2+augmentation mean mismatch: "
            f"{mean_a:.8f}"
        )
    if not np.isclose(mean_b, 5.92075938, rtol=0.0, atol=5e-7):
        raise SystemExit(f"[ERROR] frozen full-condition mean mismatch: {mean_b:.8f}")

    rng = np.random.default_rng(args.bootstrap_seed)
    boot = np.empty(args.bootstrap_replicates, dtype=np.float64)
    for start in range(0, args.bootstrap_replicates, 1000):
        stop = min(start + 1000, args.bootstrap_replicates)
        indices = rng.integers(0, len(diff), size=(stop - start, len(diff)))
        boot[start:stop] = diff[indices].mean(axis=1)
    ci_low, ci_high = np.percentile(boot, [2.5, 97.5]).tolist()

    if np.allclose(diff, 0.0):
        statistic, p_raw = 0.0, 1.0
    else:
        result = wilcoxon(diff, zero_method="pratt", alternative="two-sided")
        statistic, p_raw = float(result.statistic), float(result.pvalue)

    args.output_dir.mkdir(parents=True)
    per_image_path = args.output_dir / "bpd_augmented_configs_paired_per_image.csv"
    with per_image_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "filename", "deconv_v2_aug_pi_nme_pct",
            "fpn_udp_aug_pi_nme_pct", "difference_deconv_minus_fpn_udp_pp",
        ])
        for name, va, vb in zip(names, a, b, strict=True):
            writer.writerow([name, f"{va:.8f}", f"{vb:.8f}", f"{va-vb:.8f}"])

    summary = {
        "n_images": len(names),
        "deconv_v2_aug_mean_pi_nme_pct": mean_a,
        "fpn_udp_aug_mean_pi_nme_pct": mean_b,
        "mean_difference_deconv_minus_fpn_udp_pp": mean_diff,
        "paired_bootstrap_95_ci_low_pp": float(ci_low),
        "paired_bootstrap_95_ci_high_pp": float(ci_high),
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": args.bootstrap_seed,
        "wilcoxon_zero_method": "pratt",
        "wilcoxon_statistic": statistic,
        "wilcoxon_two_sided_p_raw": p_raw,
        "wilcoxon_holm_adjusted_p_single_comparison_family": p_raw,
        "difference_direction": "positive favours FPN+UDP+augmentation",
    }
    summary_path = args.output_dir / "bpd_augmented_configs_paired_statistics.tsv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary), delimiter="\t")
        writer.writeheader()
        writer.writerow(summary)

    metadata = {
        "analysis": "paired image-level comparison after five-seed averaging",
        "deconv_v2_augmentation_training_validation_metric": "fixed-channel NME",
        "deconv_v2_augmentation_condition": "deconv_v2_rotate_scale_fixedval",
        "superseded_deconv_v2_augmentation_mean_rejected_pct": 6.98779690,
        "input_sha256": {
            "deconv_v2_augmentation_per_image_csv": sha256(args.deconv_aug_csv),
            "frozen_fpn_udp_augmentation_per_image_csv": sha256(args.fpn_udp_aug_csv),
        },
        "input_filenames": {
            "deconv_v2_augmentation": args.deconv_aug_csv.name,
            "fpn_udp_augmentation": args.fpn_udp_aug_csv.name,
        },
        "outputs": [per_image_path.name, summary_path.name],
    }
    metadata_path = args.output_dir / "ANALYSIS_METADATA.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    manifest_path = args.output_dir / "ANALYSIS_MANIFEST.sha256.tsv"
    with manifest_path.open("w", encoding="utf-8") as handle:
        # Keep the descriptive header as a comment so that the manifest is
        # directly consumable by `sha256sum -c` without a format warning.
        handle.write("# sha256\trelative_path\n")
        for path in (metadata_path, per_image_path, summary_path):
            handle.write(f"{sha256(path)}\t{path.name}\n")

    print(f"[OK] matched images: {len(names)}")
    print(f"[OK] DeconvHeadV2+augmentation: {mean_a:.8f}%")
    print(f"[OK] FPN+UDP+augmentation: {mean_b:.8f}%")
    print(f"[RESULT] difference (Deconv - FPN+UDP): {mean_diff:.8f} pp")
    print(f"[RESULT] paired bootstrap 95% CI: [{ci_low:.8f}, {ci_high:.8f}] pp")
    print(f"[RESULT] Pratt-Wilcoxon two-sided p={p_raw:.12g}")
    print(f"[COMPLETE] outputs frozen under {args.output_dir}")


if __name__ == "__main__":
    main()
