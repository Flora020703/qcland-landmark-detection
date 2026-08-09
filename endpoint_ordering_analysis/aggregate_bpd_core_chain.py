#!/usr/bin/env python3
"""Scores the 3 retrained BPD core-architecture rungs (original einsum
head, DeconvHeadV2, +FPN -- see docs/supervisor_meeting_report_2026-08-08.md
section 0.6.1 and ablation/scripts/run_bpd_core_chain_retrain_5seed.sh)
under the SAME permutation-invariant evaluator as the official EoMT/HRNet
comparison table, by directly reusing load_eomt_per_image()/rescore_cell()/
summarize_and_write() from rescore_endpoint_conventions.py -- no new,
unreviewed scoring logic for this table; it goes through the exact same
coordinate-space-conversion, native-sanity-check, and permutation-invariant
machinery already used (and repeatedly reviewed) for the main comparison.

CRITICAL, do not remove: all 3 of these rungs used pixel_center_align=False
(they predate the UDP fix) -- pixel_center_align=False is hardcoded below
for exactly that reason, NOT the module's own default (True, calibrated for
the final +FPN+UDP model). Passing True here would silently apply the wrong
offset correction to every recovered coordinate (see
_heatmap_dump_to_model_input_space's own 2026-08-09 docstring finding).

Does NOT include the already-archived +FPN+UDP rung (different archive
layout, not yet confirmed by this script) -- that rung's official numbers
already exist under endpoint_ordering_analysis/results/ from the main
comparison; combine the two tables by hand (or re-run this script with
--fpn-udp-root once that archive's layout is confirmed).

Usage:
    python endpoint_ordering_analysis/aggregate_bpd_core_chain.py \
        --eomt-root /root/autodl-tmp/saved_checkpoints/bpd_core_chain_retrain_5seed \
        --images-root /root/autodl-tmp/images/UCL \
        --output-root endpoint_ordering_analysis/results/bpd_core_chain
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

# rung -> (backbone-dir-name used by run_bpd_core_chain_retrain_5seed.sh's
# BACKUP_ROOT/bpd_<rung>/seed{seed}/ layout, pixel_center_align this rung's
# config actually used). Order matches the intended thesis table row order.
RUNGS = [
    ("einsum", "einsum", False),
    ("deconvv2", "deconvv2", False),
    ("fpn", "fpn", False),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eomt-root", type=Path, required=True,
                         help="BACKUP_ROOT from run_bpd_core_chain_retrain_5seed.sh, "
                              "containing bpd_<rung>/seed{seed}/ subdirectories")
    parser.add_argument("--images-root", type=Path, required=True,
                         help="directory containing UCL/{Head,Abdomen,Femur}/<filename>")
    parser.add_argument("--output-root", type=Path,
                         default=Path("endpoint_ordering_analysis/results/bpd_core_chain"))
    parser.add_argument("--bootstrap-replicates", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260809)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.bootstrap_seed)
    image_cache = _ImageSizeCache(args.images_root, "Head")
    d_vect = get_d_vect("UCL", "BPD")

    all_summary_rows: list[dict] = []
    excluded: list[dict] = []

    for rung_label, backbone_dir, pixel_center_align in RUNGS:
        try:
            data = load_eomt_per_image(
                args.eomt_root, "UCL", "bpd", backbone_dir, image_cache,
                pixel_center_align=pixel_center_align,
            )
        except LoadError as exc:
            excluded.append({"rung": rung_label, "reason": str(exc)})
            print(f"[EXCLUDED] {rung_label}: {exc}")
            continue

        # EoMT's own native training convention is x-sort (unconditional
        # np.argsort(lms[:,0]) in datasets/landmark_dataset.py) for every
        # one of these rungs -- unaffected by pixel_center_align.
        rescored = rescore_cell(data, d_vect, native_convention="xsort")
        seed_rows, summary_rows = summarize_and_write(
            "UCL", "bpd", rung_label, rescored, args.output_root,
            args.bootstrap_replicates, rng,
        )
        all_summary_rows.extend(summary_rows)
        print(f"[OK] {rung_label}: n={rescored['n_images']}, "
              f"permutation_invariant_nme={summary_rows[0]['permutation_invariant_nme_5seed_mean_pct']}"
              f"±{summary_rows[0]['permutation_invariant_nme_5seed_sample_sd_pct']}%")

    if not all_summary_rows:
        raise SystemExit("[ERROR] zero rungs scored -- see excluded reasons above")

    summary_path = args.output_root / "bpd_core_chain_summary.tsv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        fields = list(all_summary_rows[0])
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(all_summary_rows)

    excluded_path = args.output_root / "bpd_core_chain_excluded.tsv"
    with excluded_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["rung", "reason"], delimiter="\t")
        writer.writeheader()
        writer.writerows(excluded)

    print(f"\n[COMPLETE] {len(all_summary_rows)}/{len(RUNGS)} rungs scored, "
          f"{len(excluded)}/{len(RUNGS)} excluded.")
    print(f"Wrote: {summary_path}, {excluded_path}, and per-image CSVs under {args.output_root}")
    print("\nPermutation-invariant NME (%) ± 5-seed sample SD, single-model final checkpoint:")
    for row in all_summary_rows:
        mean = row["permutation_invariant_nme_5seed_mean_pct"]
        sd = row["permutation_invariant_nme_5seed_sample_sd_pct"]
        print(f"  {row['method']:12s} n={row['n_images']:>3}  {float(mean):.2f}±{float(sd):.2f}%")
    print("\nCombine with the already-archived +FPN+UDP rung's official number "
          "(endpoint_ordering_analysis/results/, method=eomt_dinov2, task=bpd, dataset=UCL) "
          "by hand for the full 4-row core architecture table.")
    if excluded:
        print(f"\n{len(excluded)} rung(s) could not be scored -- see {excluded_path}.")
        sys.exit(1)


if __name__ == "__main__":
    main()
