"""Builds an internal Train-only validation split for RTMPose, REUSING
EoMT's own exact subject-grouping/shuffle/split algorithm
(HeadLandmarkDataModule._split_by_subject in datasets/landmark_dataset.py),
so RTMPose's internal validation uses the SAME held-out subjects as EoMT's
own training, rather than an independently invented random split.

WHY THIS EXISTS (review finding, 2026-08-06): the adapter's original
make_config.py pointed `val_dataloader` (and therefore `test_dataloader`,
set equal to it) directly at the released Test annotation file, with
`save_best="PCK"` selecting the checkpoint from that same Test-derived
metric every `val_interval` epochs. That is a genuine data leak: the
officially released Test partition must never be read during training,
even only for periodic monitoring, not just never used for final
checkpoint SELECTION. This script produces a Train-only internal
validation split so `val_dataloader`/checkpoint monitoring can be pointed
at genuinely held-out Train data instead; the released Test set is then
only ever touched once, after training, by run_inference.py.

Algorithm (verbatim port of datasets/landmark_dataset.py's
`_subject_id`/`_split_by_subject`, NOT a re-derived approximation):
    subjects = sorted({re.match(r"^(\\d+)", filename).group(1) for filename in kept_filenames})
    rng = np.random.default_rng(val_split_seed)   # default 42, matching EoMT's own default
    rng.shuffle(subjects)
    n_val = max(1, ceil(len(subjects) * val_fraction))   # default 0.1, matching EoMT's own default
    val_subjects = subjects[-n_val:]
    internal_val = [f for f in kept_filenames if subject(f) in val_subjects]
    internal_train = [f for f in kept_filenames if subject(f) not in val_subjects]

`kept_filenames` is exactly the set of filenames convert_csv_to_coco.py
would keep from the SAME Train CSV (same column-resolution, same
missing/negative-landmark and missing-image-file exclusions) -- reusing
those functions directly, not a re-implementation, so the split operates
on the identical row set the actual COCO conversion will use.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

from convert_csv_to_coco import _drop_leading_index_column, _resolve_columns, _to_float_or_none

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None


def _subject_id(img_name: str) -> str:
    """Verbatim port of HeadLandmarkDataModule._subject_id."""
    m = re.match(r"^(\d+)", img_name)
    return m.group(1) if m else img_name


def kept_filenames(csv_path: Path, images_dir: Path, task: str) -> list[str]:
    """Reproduces exactly which filenames convert_csv_to_coco.convert() would
    keep from this CSV, without writing a COCO json (avoids requiring
    Pillow just to check dimensions -- only checks file existence here,
    since dimensions aren't needed for the split decision)."""
    with open(csv_path, newline="", encoding="utf-8-sig") as handle:
        import csv
        reader = csv.reader(handle)
        header = next(reader)
        header = _drop_leading_index_column(header)
        rows = [row[len(row) - len(header):] if len(row) != len(header) else row
                for row in reader]

    start_col, end_col = _resolve_columns(header, task)
    kept = []
    for row in rows:
        image_name = row[0]
        if not image_name or not image_name.strip():
            continue
        landmark_vals = [_to_float_or_none(v) for v in row[start_col:end_col]]
        if len(landmark_vals) != 4 or any(v is None or v < 0 for v in landmark_vals):
            continue
        if not (images_dir / image_name).is_file():
            continue
        kept.append(image_name)
    return kept


def split(filenames: list[str], val_fraction: float = 0.1,
          val_split_seed: int = 42) -> tuple[list[str], list[str]]:
    subjects = sorted({_subject_id(f) for f in filenames})
    rng = np.random.default_rng(val_split_seed)
    subjects_arr = np.array(subjects, dtype=object)
    rng.shuffle(subjects_arr)
    n_val = max(1, math.ceil(len(subjects_arr) * val_fraction))
    val_subjects = set(subjects_arr[-n_val:].tolist())

    internal_train = [f for f in filenames if _subject_id(f) not in val_subjects]
    internal_val = [f for f in filenames if _subject_id(f) in val_subjects]
    return internal_train, internal_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, type=Path, help="the released Train CSV, NEVER the Test CSV")
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--task", required=True, choices=["BPD", "OFD", "APAD", "TAD", "FL"])
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--val-split-seed", type=int, default=42)
    parser.add_argument("--out-json", required=True, type=Path)
    args = parser.parse_args()

    filenames = kept_filenames(args.csv, args.images_dir, args.task)
    internal_train, internal_val = split(filenames, args.val_fraction, args.val_split_seed)

    overlap = set(internal_train) & set(internal_val)
    assert not overlap, f"internal train/val overlap (should be impossible): {overlap}"
    assert len(internal_train) + len(internal_val) == len(filenames)

    manifest = {
        "source_csv": str(args.csv),
        "task": args.task,
        "val_fraction": args.val_fraction,
        "val_split_seed": args.val_split_seed,
        "n_total": len(filenames),
        "n_internal_train": len(internal_train),
        "n_internal_val": len(internal_val),
        "internal_train_filenames": internal_train,
        "internal_val_filenames": internal_val,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[OK] {args.task}: {len(internal_train)} internal-train / "
          f"{len(internal_val)} internal-val (of {len(filenames)} total Train rows) "
          f"-> {args.out_json}")


if __name__ == "__main__":
    main()
