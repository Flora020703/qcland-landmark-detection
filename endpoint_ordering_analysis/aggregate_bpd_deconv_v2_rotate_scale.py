#!/usr/bin/env python3
"""Freeze original-image-space PI-NME for the five-seed follow-up ablation."""
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260814)
    args = parser.parse_args()

    if args.output_root.exists():
        raise SystemExit(f"[ERROR] output root already exists: {args.output_root}")
    args.output_root.mkdir(parents=True)

    try:
        loaded = load_eomt_per_image(
            args.run_root,
            "UCL",
            "bpd",
            "deconv_v2_rotate_scale",
            _ImageSizeCache(args.images_root, "Head"),
            pixel_center_align=False,
            model_input_size=512.0,
            heatmap_size=64.0,
        )
    except LoadError as exc:
        raise SystemExit(f"[ERROR] follow-up condition could not be loaded: {exc}") from exc

    rescored = rescore_cell(loaded, get_d_vect("UCL", "BPD"), native_convention="xsort")
    _, rows = summarize_and_write(
        "UCL", "bpd", "deconv_v2_rotate_scale", rescored, args.output_root,
        args.bootstrap_replicates, np.random.default_rng(args.bootstrap_seed),
    )
    path = args.output_root / "bpd_deconv_v2_rotate_scale_summary.tsv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    row = rows[0]
    print(
        "[COMPLETE] original-space PI-NME="
        f"{row['permutation_invariant_nme_5seed_mean_pct']} +/- "
        f"{row['permutation_invariant_nme_5seed_sample_sd_pct']}%"
    )


if __name__ == "__main__":
    main()
