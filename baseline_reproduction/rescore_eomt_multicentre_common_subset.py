#!/usr/bin/env python3
"""Rescore Multicentre EoMT BPD/OFD on HRNet's task-valid test subset.

This is a post-hoc aggregation of already saved per-image fixed-channel NME
values. It does not load a checkpoint or rerun inference. Each NME row is
joined to a filename through the seed-specific test_image_order.csv before
filtering, so row order is never assumed implicitly.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path


TASKS = ("bpd", "ofd")
BACKBONES = ("dinov2", "dinov3")
SEEDS = (42, 0, 123, 2024, 3407)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eomt-root",
        type=Path,
        default=Path("/root/autodl-tmp/saved_checkpoints/multicentre_5seed"),
    )
    parser.add_argument(
        "--manifest-root",
        type=Path,
        default=Path(
            "/root/autodl-tmp/hrnet_512_fixed_5seed/common_eval_manifests"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "/root/autodl-tmp/hrnet_512_fixed_5seed/eomt_common_subset"
        ),
    )
    return parser.parse_args()


def read_dict_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def canonical_filename(value: str) -> str:
    # CSVs currently contain basenames. Normalising separators also makes the
    # audit robust if a future dump contains a relative path.
    return value.strip().replace("\\", "/").rsplit("/", 1)[-1]


def read_manifest(path: Path) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    names = [
        canonical_filename(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate filenames in manifest: {path}")
    return set(names)


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    detail_rows: list[dict] = []
    filtered_paths: list[Path] = []

    for task in TASKS:
        manifest = read_manifest(
            args.manifest_root / f"multicentre_{task}_valid_filenames.txt"
        )
        expected_n = 1180 if task == "bpd" else 1189
        if len(manifest) != expected_n:
            raise ValueError(
                f"{task}: expected {expected_n} valid filenames, got {len(manifest)}"
            )

        for backbone in BACKBONES:
            for seed in SEEDS:
                run_dir = (
                    args.eomt_root
                    / f"multicentre-{task}-{backbone}"
                    / f"seed{seed}"
                )
                nme_path = run_dir / f"seed{seed}_final_fixedchannel_per_image.csv"
                order_path = run_dir / "test_image_order.csv"
                nme_rows = read_dict_rows(nme_path)
                order_rows = read_dict_rows(order_path)

                if len(nme_rows) != 1191 or len(order_rows) != 1191:
                    raise ValueError(
                        f"{task}/{backbone}/seed{seed}: expected 1191 rows; "
                        f"NME={len(nme_rows)}, order={len(order_rows)}"
                    )
                if not {"index", "nme"}.issubset(nme_rows[0]):
                    raise ValueError(f"unexpected NME columns in {nme_path}")
                if not {"index", "filename"}.issubset(order_rows[0]):
                    raise ValueError(f"unexpected order columns in {order_path}")

                order_by_index: dict[int, str] = {}
                for row in order_rows:
                    index = int(row["index"])
                    if index in order_by_index:
                        raise ValueError(f"duplicate index {index} in {order_path}")
                    order_by_index[index] = canonical_filename(row["filename"])

                joined: list[dict[str, str]] = []
                seen_indices: set[int] = set()
                for row in nme_rows:
                    index = int(row["index"])
                    if index in seen_indices:
                        raise ValueError(f"duplicate index {index} in {nme_path}")
                    seen_indices.add(index)
                    if index not in order_by_index:
                        raise ValueError(f"index {index} absent from {order_path}")
                    joined.append({**row, "filename": order_by_index[index]})

                if seen_indices != set(order_by_index):
                    raise ValueError(f"index sets differ: {nme_path} vs {order_path}")
                if len({row["filename"] for row in joined}) != len(joined):
                    raise ValueError(f"duplicate joined filenames in {run_dir}")

                filtered = [row for row in joined if row["filename"] in manifest]
                found = {row["filename"] for row in filtered}
                if found != manifest or len(filtered) != expected_n:
                    missing = sorted(manifest - found)[:5]
                    raise ValueError(
                        f"{task}/{backbone}/seed{seed}: subset mismatch; "
                        f"matched={len(filtered)}, missing examples={missing}"
                    )

                original_values = [float(row["nme"]) for row in joined]
                filtered_values = [float(row["nme"]) for row in filtered]
                original_pct = 100.0 * statistics.fmean(original_values)
                filtered_pct = 100.0 * statistics.fmean(filtered_values)

                filtered_path = args.output_root / (
                    f"multicentre-{task}-{backbone}-seed{seed}_"
                    "final_fixedchannel_common_subset.csv"
                )
                fields = ["filename", *nme_rows[0].keys()]
                with filtered_path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(filtered)
                filtered_paths.append(filtered_path)

                detail_rows.append(
                    {
                        "task": task,
                        "backbone": backbone,
                        "seed": seed,
                        "original_n": len(joined),
                        "common_n": len(filtered),
                        "original_fixed_nme_pct": f"{original_pct:.8f}",
                        "common_fixed_nme_pct": f"{filtered_pct:.8f}",
                        "change_pp": f"{filtered_pct - original_pct:.8f}",
                    }
                )
                print(
                    f"[OK] {task}/{backbone}/seed{seed}: "
                    f"n={len(filtered)}, {original_pct:.4f}% -> {filtered_pct:.4f}%"
                )

    if len(detail_rows) != 20 or len(filtered_paths) != 20:
        raise RuntimeError("expected exactly 20 rescored runs")

    detail_fields = [
        "task",
        "backbone",
        "seed",
        "original_n",
        "common_n",
        "original_fixed_nme_pct",
        "common_fixed_nme_pct",
        "change_pp",
    ]
    write_tsv(args.output_root / "eomt_common_subset_per_seed.tsv", detail_fields, detail_rows)

    summary_rows: list[dict] = []
    for task in TASKS:
        for backbone in BACKBONES:
            values = [
                float(row["common_fixed_nme_pct"])
                for row in detail_rows
                if row["task"] == task and row["backbone"] == backbone
            ]
            if len(values) != 5:
                raise RuntimeError(f"expected five values for {task}/{backbone}")
            summary_rows.append(
                {
                    "task": task,
                    "backbone": backbone,
                    "n_per_seed": 1180 if task == "bpd" else 1189,
                    "mean_fixed_nme_pct": f"{statistics.fmean(values):.8f}",
                    "seed_sample_sd_pct": f"{statistics.stdev(values):.8f}",
                }
            )

    summary_fields = [
        "task",
        "backbone",
        "n_per_seed",
        "mean_fixed_nme_pct",
        "seed_sample_sd_pct",
    ]
    write_tsv(args.output_root / "eomt_common_subset_summary.tsv", summary_fields, summary_rows)
    print(f"[COMPLETE] outputs written under {args.output_root}")


if __name__ == "__main__":
    main()
