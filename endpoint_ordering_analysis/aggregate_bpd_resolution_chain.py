#!/usr/bin/env python3
"""Aggregate the matched five-seed UCL BPD 256/512 input-resolution study.

The two conditions share a 64x64 heatmap and UDP pixel-centre encoding, but
their dumped coordinates live in different model-input spaces.  This script
therefore passes the input size explicitly and refuses incomplete or
non-matched data rather than silently using the 512 default for both.
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


CONDITIONS = (("res256", 256.0), ("res512", 512.0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resolution-root",
        type=Path,
        required=True,
        help="Staging root containing bpd_res256/seed*/ and bpd_res512/seed*/.",
    )
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260810)
    args = parser.parse_args()

    if args.output_root.exists():
        raise SystemExit(f"[ERROR] output root already exists: {args.output_root}")
    args.output_root.mkdir(parents=True)

    rng = np.random.default_rng(args.bootstrap_seed)
    image_cache = _ImageSizeCache(args.images_root, "Head")
    d_vect = get_d_vect("UCL", "BPD")
    loaded: dict[str, dict] = {}

    for label, input_size in CONDITIONS:
        try:
            loaded[label] = load_eomt_per_image(
                args.resolution_root,
                "UCL",
                "bpd",
                label,
                image_cache,
                pixel_center_align=True,
                model_input_size=input_size,
                heatmap_size=64.0,
            )
        except LoadError as exc:
            raise SystemExit(f"[ERROR] required condition {label} could not be loaded: {exc}") from exc

    reference = loaded["res256"]["filenames"]
    if loaded["res512"]["filenames"] != reference:
        raise SystemExit("[ERROR] res256 and res512 filename sets differ")
    if not reference:
        raise SystemExit("[ERROR] matched resolution subset is empty")

    summary_rows: list[dict] = []
    for label, _ in CONDITIONS:
        rescored = rescore_cell(loaded[label], d_vect, native_convention="xsort")
        _, rows = summarize_and_write(
            "UCL",
            "bpd",
            label,
            rescored,
            args.output_root,
            args.bootstrap_replicates,
            rng,
        )
        summary_rows.extend(rows)
        row = rows[0]
        print(
            f"[OK] {label}: n={rescored['n_images']}, original-space PI-NME="
            f"{row['permutation_invariant_nme_5seed_mean_pct']} +/- "
            f"{row['permutation_invariant_nme_5seed_sample_sd_pct']}%"
        )

    summary_path = args.output_root / "bpd_resolution_summary.tsv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\n[COMPLETE] 2/2 matched resolution conditions scored; n={len(reference)} each")
    print(f"Wrote: {summary_path} and per-image CSVs under {args.output_root}")


if __name__ == "__main__":
    main()
