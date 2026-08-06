"""Quantifies (does not just theorize about) how often a horizontal flip
would change the DOD-canonical channel identity for a real dataset/task,
given this adapter's design choice to freeze channel order ONCE at
CSV-conversion time and apply flip via FLIP_INDICES=[0,1] (no re-swap) --
see fetal_dataset_info.py's "SCOPE NOTE" docstring for the disclosed
limitation this measures.

This does not fix anything; it turns a previously qualitative caveat
("occasionally end up in the opposite channel order") into a measured rate
on the actual canary dataset, for the protocol audit requested before
starting training. Run directly (needs Pillow + the real CSV/images, not a
unit test -- no assertions, only a printed report):

    python rtmpose_reproduction/audit_flip_order_stability.py \
        --csv <path>/Head_Train.csv --images-dir <path>/UCL/Head \
        --dataset UCL --task BPD
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from convert_csv_to_coco import _resolve_columns, _drop_leading_index_column, _to_float_or_none
from dod_vectors import get_d_vect
from endpoint_order import canonical_order

from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--dataset", required=True, choices=["UCL", "MULTICENTRE"])
    parser.add_argument("--task", required=True, choices=["BPD", "OFD", "APAD", "TAD", "FL"])
    args = parser.parse_args()

    with open(args.csv, newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        header = _drop_leading_index_column(header)
        rows = [row[len(row) - len(header):] if len(row) != len(header) else row
                for row in reader]

    start_col, end_col = _resolve_columns(header, args.task)
    d_vect = get_d_vect(args.dataset, args.task)

    n_total = 0
    n_flip_disagrees = 0
    n_xsort_vs_dod_disagrees_preflip = 0
    n_xsort_vs_dod_disagrees_postflip = 0
    examples = []

    for row in rows:
        image_name = row[0]
        if not image_name:
            continue
        vals = [_to_float_or_none(v) for v in row[start_col:end_col]]
        if len(vals) != 4 or any(v is None or v < 0 for v in vals):
            continue
        image_path = args.images_dir / image_name
        if not image_path.is_file():
            continue
        with Image.open(image_path) as im:
            width, _ = im.size

        raw_p0, raw_p1 = (vals[0], vals[1]), (vals[2], vals[3])

        # RTMPose/HRNet convention: DOD projection, frozen at conversion time.
        dod_p0, dod_p1 = canonical_order(raw_p0, raw_p1, d_vect)

        # EoMT convention: ascending-x sort (recomputed fresh every access).
        xsort_pre = tuple(sorted((raw_p0, raw_p1), key=lambda p: p[0]))

        n_total += 1
        if dod_p0 != xsort_pre[0]:
            n_xsort_vs_dod_disagrees_preflip += 1

        # Simulate: dataset stores (dod_p0, dod_p1) as channel 0/1. MMPose's
        # RandomFlip with flip_indices=[0,1] mirrors x WITHOUT swapping which
        # channel holds which point.
        flipped_ch0 = (width - 1 - dod_p0[0], dod_p0[1])
        flipped_ch1 = (width - 1 - dod_p1[0], dod_p1[1])

        # Does re-running the SAME frozen d_vect projection on the flipped
        # pair still agree the stored order (channel 0, channel 1) is
        # DOD-ascending? If not, this sample's post-flip stored order
        # disagrees with what the DOD rule itself would say post-flip.
        reordered = canonical_order(flipped_ch0, flipped_ch1, d_vect)
        if reordered != (flipped_ch0, flipped_ch1):
            n_flip_disagrees += 1
            if len(examples) < 5:
                examples.append((image_name, dod_p0, dod_p1, flipped_ch0, flipped_ch1))

        xsort_post = tuple(sorted((flipped_ch0, flipped_ch1), key=lambda p: p[0]))
        if xsort_post[0] != flipped_ch0:
            n_xsort_vs_dod_disagrees_postflip += 1

    print(f"dataset={args.dataset} task={args.task} n={n_total}")
    print(f"DOD-vs-xsort disagreement, pre-flip:  {n_xsort_vs_dod_disagrees_preflip}/{n_total} "
          f"({100.0*n_xsort_vs_dod_disagrees_preflip/n_total:.1f}%)")
    print(f"DOD-vs-xsort disagreement, post-flip: {n_xsort_vs_dod_disagrees_postflip}/{n_total} "
          f"({100.0*n_xsort_vs_dod_disagrees_postflip/n_total:.1f}%)")
    print(f"flip breaks this adapter's frozen DOD order (RTMPose-specific risk): "
          f"{n_flip_disagrees}/{n_total} ({100.0*n_flip_disagrees/n_total:.1f}%)")
    if examples:
        print("example flip-disagreement cases (file, ch0_before, ch1_before, ch0_after_flip, ch1_after_flip):")
        for ex in examples:
            print(f"  {ex}")


if __name__ == "__main__":
    main()
