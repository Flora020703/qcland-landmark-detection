#!/usr/bin/env python3
"""Aggregate the three five-seed UCL BPD loss ablations in image space.

All three runs use the original einsum heatmap head and pre-UDP coordinate
encoding (pixel_center_align=False).  Scoring reuses the same loader,
coordinate recovery, and permutation-invariant NME implementation as the
official cross-method comparison.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "rtmpose_reproduction"))
from dod_vectors import get_d_vect  # noqa: E402
from rescore_endpoint_conventions import (  # noqa: E402
    LoadError,
    _ImageSizeCache,
    load_eomt_per_image,
    rescore_cell,
    summarize_and_write,
)


LOSS_RUNGS = [
    ("plain_mse", "plain_mse"),
    ("weighted_mse", "weighted_mse"),
    ("hybrid", "hybrid"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--loss-root",
        type=Path,
        required=True,
        help="Root containing bpd_<loss>/seed{seed}/ directories.",
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        required=True,
        help="UCL image root containing Head/<filename>.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260809)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.bootstrap_seed)
    image_cache = _ImageSizeCache(args.images_root, "Head")
    d_vect = get_d_vect("UCL", "BPD")

    all_summary_rows: list[dict] = []
    excluded: list[dict] = []

    for label, directory_name in LOSS_RUNGS:
        try:
            data = load_eomt_per_image(
                args.loss_root,
                "UCL",
                "bpd",
                directory_name,
                image_cache,
                pixel_center_align=False,
            )
        except LoadError as exc:
            excluded.append({"loss": label, "reason": str(exc)})
            print(f"[EXCLUDED] {label}: {exc}")
            continue

        rescored = rescore_cell(data, d_vect, native_convention="xsort")
        _, summary_rows = summarize_and_write(
            "UCL",
            "bpd",
            label,
            rescored,
            args.output_root,
            args.bootstrap_replicates,
            rng,
        )
        all_summary_rows.extend(summary_rows)
        row = summary_rows[0]
        print(
            f"[OK] {label}: n={rescored['n_images']}, "
            "permutation_invariant_nme="
            f"{row['permutation_invariant_nme_5seed_mean_pct']}"
            " +/- "
            f"{row['permutation_invariant_nme_5seed_sample_sd_pct']}%"
        )

    if not all_summary_rows:
        raise SystemExit("[ERROR] zero loss rungs scored")

    summary_path = args.output_root / "bpd_loss_chain_summary.tsv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        fields = list(all_summary_rows[0])
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(all_summary_rows)

    excluded_path = args.output_root / "bpd_loss_chain_excluded.tsv"
    with excluded_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["loss", "reason"], delimiter="\t")
        writer.writeheader()
        writer.writerows(excluded)

    print(
        f"\n[COMPLETE] {len(all_summary_rows)}/{len(LOSS_RUNGS)} loss rungs "
        f"scored, {len(excluded)}/{len(LOSS_RUNGS)} excluded."
    )
    print(f"Wrote: {summary_path} and {excluded_path}")
    print("\nOriginal-image-space permutation-invariant NME (%):")
    for row in all_summary_rows:
        mean = float(row["permutation_invariant_nme_5seed_mean_pct"])
        sd = float(row["permutation_invariant_nme_5seed_sample_sd_pct"])
        print(f"  {row['method']:14s} n={row['n_images']:>3}  {mean:.2f} +/- {sd:.2f}%")

    if excluded:
        raise SystemExit(
            f"[ERROR] {len(excluded)} required loss rung(s) excluded; "
            f"see {excluded_path}"
        )


if __name__ == "__main__":
    main()
