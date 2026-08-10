#!/usr/bin/env python3
"""Score the matched single-run UCL BPD sigma screen in original space.

This is deliberately a single-seed exploratory diagnostic.  It must not
emit or imply a seed-level standard deviation.  Both runs use the original
einsum head, plain MSE, and pre-UDP encoding (pixel_center_align=False).
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
import rescore_endpoint_conventions as scoring  # noqa: E402


CASES = [
    ("sigma1", "sigma10", 1.0),
    ("sigma4", "sigma40", 4.0),
]
SEED = 2024


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sigma-root",
        type=Path,
        required=True,
        help="Root containing bpd_sigma10/seed2024 and bpd_sigma40/seed2024.",
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        required=True,
        help="UCL image root containing Head/<filename>.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    image_cache = scoring._ImageSizeCache(args.images_root, "Head")
    d_vect = get_d_vect("UCL", "BPD")

    # The reviewed loader uses its module-level SEEDS tuple. Override it for
    # this process only; never duplicate one run under five seed names.
    scoring.SEEDS = (SEED,)

    summary_rows: list[dict] = []
    exclusions: list[dict] = []

    for label, directory_name, sigma in CASES:
        try:
            data = scoring.load_eomt_per_image(
                args.sigma_root,
                "UCL",
                "bpd",
                directory_name,
                image_cache,
                pixel_center_align=False,
            )
        except scoring.LoadError as exc:
            exclusions.append({"condition": label, "reason": str(exc)})
            print(f"[EXCLUDED] {label}: {exc}")
            continue

        rescored = scoring.rescore_cell(data, d_vect, native_convention="xsort")
        by_name = rescored["per_seed_per_image"][SEED]
        filenames = sorted(by_name)
        pi_values = np.array(
            [by_name[name]["permutation_invariant_nme"] for name in filenames]
        ) * 100.0
        fixed_values = np.array(
            [by_name[name]["raw_channel_original"] for name in filenames]
        ) * 100.0

        per_image_path = args.output_root / f"ucl_bpd_{label}_per_image.csv"
        with per_image_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "filename",
                    "original_space_fixed_channel_nme_pct",
                    "original_space_permutation_invariant_nme_pct",
                    "selected_assignment",
                ],
            )
            writer.writeheader()
            for name, fixed, pi in zip(filenames, fixed_values, pi_values):
                writer.writerow(
                    {
                        "filename": name,
                        "original_space_fixed_channel_nme_pct": f"{fixed:.8f}",
                        "original_space_permutation_invariant_nme_pct": f"{pi:.8f}",
                        "selected_assignment": by_name[name][
                            "permutation_invariant_assignment"
                        ],
                    }
                )

        summary_rows.append(
            {
                "condition": label,
                "sigma": f"{sigma:.1f}",
                "seed": str(SEED),
                "n_images": str(len(filenames)),
                "original_space_fixed_channel_mean_pct": f"{fixed_values.mean():.8f}",
                "original_space_permutation_invariant_mean_pct": f"{pi_values.mean():.8f}",
            }
        )
        print(
            f"[OK] {label}: seed={SEED}, n={len(filenames)}, "
            f"original-space PI-NME={pi_values.mean():.2f}%"
        )

    summary_path = args.output_root / "bpd_sigma_single_run_summary.tsv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "condition",
            "sigma",
            "seed",
            "n_images",
            "original_space_fixed_channel_mean_pct",
            "original_space_permutation_invariant_mean_pct",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(summary_rows)

    excluded_path = args.output_root / "bpd_sigma_single_run_excluded.tsv"
    with excluded_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["condition", "reason"], delimiter="\t"
        )
        writer.writeheader()
        writer.writerows(exclusions)

    if exclusions or len(summary_rows) != len(CASES):
        raise SystemExit(
            f"[ERROR] scored {len(summary_rows)}/{len(CASES)} sigma conditions; "
            f"see {excluded_path}"
        )

    print(f"[COMPLETE] scored {len(summary_rows)}/{len(CASES)} conditions")
    print(f"Wrote: {summary_path} and {excluded_path}")
    print("Single-run exploratory evidence only; no seed-level SD is reported.")


if __name__ == "__main__":
    main()
