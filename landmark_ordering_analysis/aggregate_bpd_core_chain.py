#!/usr/bin/env python3
"""Scores the 3 retrained BPD core-architecture rungs (original einsum
head, DeconvHeadV2, +FPN -- see docs/supervisor_meeting_report_2026-08-08.md
section 0.6.1 and ablation/scripts/run_bpd_core_chain_retrain_5seed.sh)
under the SAME permutation-invariant evaluator as the official EoMT/HRNet
comparison table, by directly reusing load_eomt_per_image()/rescore_cell()/
summarize_and_write() from rescore_landmark_conventions.py -- no new,
unreviewed scoring logic for this table; it goes through the exact same
coordinate-space-conversion, native-sanity-check, and permutation-invariant
machinery already used (and repeatedly reviewed) for the main comparison.

CRITICAL, do not remove: the 3 REQUIRED rungs (einsum/deconvv2/fpn) used
pixel_center_align=False (they predate the UDP fix) -- hardcoded below for
exactly that reason, NOT the module's own default (True, calibrated for
the final +FPN+UDP model). Passing True here would silently apply the wrong
offset correction to every recovered coordinate (see
_heatmap_dump_to_model_input_space's own 2026-08-09 docstring finding).

The already-archived +FPN+UDP rung is OPTIONAL and off by default (only
attempted if --fpn-udp-root is passed) -- it lives in a SEPARATE, already-
existing archive (fpn_udp_loaderseed_ablation.tar) produced by a different
script (run_fpn_udp_loaderseed_ablation.sh) with its OWN directory naming
convention, which this script does NOT assume matches --eomt-root's
bpd_<rung>/seed{seed}/ layout the NEW retrain script produces -- pass
--fpn-udp-root/--fpn-udp-backbone-dir pointing at wherever that archive
actually is once its real layout is confirmed. A missing/wrong
--fpn-udp-root does NOT fail the run: the 3 required rungs are the ones
this run exists to produce, and are reported regardless. Its official
number, if not supplied here, already exists under
landmark_ordering_analysis/results/ from the main comparison and can be
combined with this script's 3-row output by hand.

Usage (3 required rungs only):
    python landmark_ordering_analysis/aggregate_bpd_core_chain.py \
        --eomt-root /root/autodl-tmp/saved_checkpoints/bpd_core_chain_retrain_5seed \
        --images-root /root/autodl-tmp/images/UCL \
        --output-root landmark_ordering_analysis/results/bpd_core_chain

Usage (also including the already-archived +FPN+UDP rung, once its real
extracted path is known):
    python landmark_ordering_analysis/aggregate_bpd_core_chain.py \
        --eomt-root /root/autodl-tmp/saved_checkpoints/bpd_core_chain_retrain_5seed \
        --images-root /root/autodl-tmp/images/UCL \
        --fpn-udp-root /root/autodl-tmp/saved_checkpoints/fpn_udp_loaderseed_ablation \
        --fpn-udp-backbone-dir fpn-udp-loaderseed \
        --output-root landmark_ordering_analysis/results/bpd_core_chain
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
from rescore_landmark_conventions import (  # noqa: E402
    LoadError,
    _ImageSizeCache,
    load_eomt_per_image,
    rescore_cell,
    summarize_and_write,
)

# REQUIRED rungs, read from --eomt-root (the new retrain script's own
# output layout, bpd_<rung>/seed{seed}/): (rung label, backbone-dir name,
# pixel_center_align actually used). All 3 predate UDP -> False. Passing
# the wrong value silently applies an invalid coordinate-offset correction.
# This run's exit code depends ONLY on these -- see the optional
# +FPN+UDP handling below for why.
REQUIRED_RUNGS = [
    ("einsum", "einsum", False),
    ("deconvv2", "deconvv2", False),
    ("fpn", "fpn", False),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eomt-root", type=Path, required=True,
                         help="BACKUP_ROOT from run_bpd_core_chain_retrain_5seed.sh, "
                              "containing bpd_<rung>/seed{seed}/ subdirectories for the "
                              "3 REQUIRED rungs (einsum/deconvv2/fpn)")
    parser.add_argument("--images-root", type=Path, required=True,
                         help="directory containing UCL/{Head,Abdomen,Femur}/<filename>")
    parser.add_argument("--fpn-udp-root", type=Path, default=None,
                         help="OPTIONAL: root of the already-archived +FPN+UDP rung "
                              "(fpn_udp_loaderseed_ablation.tar once extracted) -- a "
                              "SEPARATE root from --eomt-root, since that archive was "
                              "produced by a different script with its own directory "
                              "naming, not confirmed to match bpd_<rung>/seed{seed}/. "
                              "If omitted, this rung is skipped entirely (not attempted, "
                              "not counted as a failure) -- its official number already "
                              "exists under landmark_ordering_analysis/results/.")
    parser.add_argument("--fpn-udp-backbone-dir", default="fpn_udp",
                         help="backbone-dir name under --fpn-udp-root, i.e. "
                              "<fpn-udp-root>/bpd_<this>/seed{seed}/ -- adjust to match "
                              "the archive's REAL extracted layout once confirmed")
    parser.add_argument("--output-root", type=Path,
                         default=Path("landmark_ordering_analysis/results/bpd_core_chain"))
    parser.add_argument("--bootstrap-replicates", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260809)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.bootstrap_seed)
    image_cache = _ImageSizeCache(args.images_root, "Head")
    d_vect = get_d_vect("UCL", "BPD")

    all_summary_rows: list[dict] = []
    required_excluded: list[dict] = []
    optional_excluded: list[dict] = []

    def _score_rung(rung_label: str, root: Path, backbone_dir: str, pixel_center_align: bool,
                     excluded_bucket: list[dict]) -> None:
        try:
            data = load_eomt_per_image(
                root, "UCL", "bpd", backbone_dir, image_cache,
                pixel_center_align=pixel_center_align,
            )
        except LoadError as exc:
            excluded_bucket.append({"rung": rung_label, "reason": str(exc)})
            print(f"[EXCLUDED] {rung_label}: {exc}")
            return

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

    for rung_label, backbone_dir, pixel_center_align in REQUIRED_RUNGS:
        _score_rung(rung_label, args.eomt_root, backbone_dir, pixel_center_align, required_excluded)

    # Optional 4th rung -- attempted only if the user explicitly points at
    # it. A failure here is reported but does NOT fail the run: the 3
    # required rungs above are this script's actual purpose.
    if args.fpn_udp_root is not None:
        _score_rung("fpn_udp", args.fpn_udp_root, args.fpn_udp_backbone_dir, True, optional_excluded)

    if not all_summary_rows:
        raise SystemExit("[ERROR] zero rungs scored -- see excluded reasons above")

    summary_path = args.output_root / "bpd_core_chain_summary.tsv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        fields = list(all_summary_rows[0])
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(all_summary_rows)

    all_excluded = required_excluded + optional_excluded
    excluded_path = args.output_root / "bpd_core_chain_excluded.tsv"
    with excluded_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["rung", "reason"], delimiter="\t")
        writer.writeheader()
        writer.writerows(all_excluded)

    n_attempted = len(REQUIRED_RUNGS) + (1 if args.fpn_udp_root is not None else 0)
    print(f"\n[COMPLETE] {len(all_summary_rows)}/{n_attempted} rungs scored, "
          f"{len(all_excluded)}/{n_attempted} excluded.")
    print(f"Wrote: {summary_path}, {excluded_path}, and per-image CSVs under {args.output_root}")
    print("\nPermutation-invariant NME (%) ± 5-seed sample SD, single-model final checkpoint:")
    for row in all_summary_rows:
        mean = row["permutation_invariant_nme_5seed_mean_pct"]
        sd = row["permutation_invariant_nme_5seed_sample_sd_pct"]
        print(f"  {row['method']:12s} n={row['n_images']:>3}  {float(mean):.2f}±{float(sd):.2f}%")
    if args.fpn_udp_root is None:
        print("\n(Optional +FPN+UDP rung not attempted -- pass --fpn-udp-root to include it. "
              "Its official number already exists under landmark_ordering_analysis/results/ "
              "and can be combined with the table above by hand.)")
    elif optional_excluded:
        print(f"\n[NOTE] optional +FPN+UDP rung could not be scored ({optional_excluded[0]['reason']}) "
              "-- this does NOT fail the run; the 3 required rungs above are unaffected.")

    # Exit code depends ONLY on the 3 REQUIRED rungs -- a missing/misconfigured
    # optional +FPN+UDP archive must never make an otherwise-successful
    # required-rung run look like a failure.
    if required_excluded:
        print(f"\n{len(required_excluded)} REQUIRED rung(s) could not be scored -- see {excluded_path}.")
        sys.exit(1)


if __name__ == "__main__":
    main()
