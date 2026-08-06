"""Convert a released HRNet-style fetal-biometry annotation CSV into a
COCO-keypoints-format JSON for the RTMPose adapter.

Deliberately mirrors the audited upstream `lib/datasets/fetal.py`'s own
column-resolution and row-validity logic byte-for-byte (see
_resolve_columns/_valid_row_mask below) rather than guessing column names,
so this converter selects exactly the same rows and exactly the same four
landmark columns HRNet itself uses for a given (anatomy, metric) -- this is
what let dod_vectors.py's extracted d_vect values be reused unchanged
(Sec. "Endpoint identity gate" in PROTOCOL_LOCKED.md).

Keypoints and bbox are written in ORIGINAL image pixel coordinates (COCO
convention) -- NOT pre-transformed into 512-space. The 512-space resize is
applied later, at data-loading time, by this project's own MMPose transform
(transforms.py), which calls geometry.to_model_space() -- this converter has
no dependency on geometry.py's INPUT_SIZE at all, so changing the model
input resolution never requires re-running this converter.

bbox is always the complete source image: [0, 0, width, height], per the
locked "full image as the deterministic input region" design (no object
detection target is being learned; RTMPose's own architecture never predicts
this box, see PROTOCOL_LOCKED.md's opening rationale).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from dod_vectors import get_d_vect
from endpoint_order import canonical_order

try:
    from PIL import Image
except ImportError:  # pragma: no cover - only needed for the real conversion, not test_convert.py's fakes
    Image = None

_ANATOMY_BY_TASK = {
    "BPD": "brain", "OFD": "brain",
    "APAD": "abdomen", "TAD": "abdomen",
    "FL": "femur",
}


def _resolve_columns(fieldnames: list[str], task: str) -> tuple[int, int]:
    """Mirrors fetal.py's start_col/end_col table exactly (0-indexed, after
    dropping a leading 'index'/'Unnamed' column exactly as fetal.py does)."""
    anatomy = _ANATOMY_BY_TASK[task]
    if anatomy == "brain":
        return (4, 8) if task == "OFD" else (8, 12)
    if anatomy == "abdomen":
        return (4, 8) if task == "TAD" else (8, 12)
    if anatomy == "femur":
        return (4, 8)
    raise ValueError(f"unknown task {task!r}")


def _drop_leading_index_column(fieldnames: list[str]) -> list[str]:
    if not fieldnames:
        return fieldnames
    first = str(fieldnames[0])
    if first == "index" or first.lower() == "index" or first.startswith("Unnamed"):
        return fieldnames[1:]
    return fieldnames


def _to_float_or_none(value: str):
    if value is None or value == "":
        return None
    try:
        f = float(value)
    except ValueError:
        return None
    return f


def convert(csv_path: Path, images_dir: Path, dataset: str, task: str,
            out_json: Path, excluded_log: Path,
            filename_subset: set[str] | None = None) -> dict:
    """`filename_subset`, if given, restricts the output to rows whose
    image_name is in this set (added 2026-08-06 so make_internal_val_split.py's
    Train-only internal validation split can be materialised as its own COCO
    json from the SAME Train CSV, without a second, differently-filtered
    converter implementation -- rows outside the subset are recorded in
    `excluded_log` with reason "not in internal split", not silently
    dropped)."""
    with open(csv_path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        header = _drop_leading_index_column(header)
        rows = [row[len(row) - len(header):] if len(row) != len(header) else row
                for row in reader]

    start_col, end_col = _resolve_columns(header, task)
    d_vect = get_d_vect(dataset, task)

    images, annotations, excluded = [], [], []
    image_id = 0
    ann_id = 0
    seen_filenames: set[str] = set()

    for row in rows:
        image_name = row[0]
        if image_name is None or image_name.strip() == "":
            excluded.append({"image_name": image_name, "reason": "missing image_name"})
            continue

        if filename_subset is not None and image_name not in filename_subset:
            excluded.append({"image_name": image_name, "reason": "not in internal split"})
            continue

        landmark_vals = [_to_float_or_none(v) for v in row[start_col:end_col]]
        if len(landmark_vals) != 4 or any(v is None or v < 0 for v in landmark_vals):
            excluded.append({"image_name": image_name, "reason": "missing/negative landmark"})
            continue

        image_path = images_dir / image_name
        if not image_path.is_file():
            excluded.append({"image_name": image_name, "reason": "image file not found"})
            continue

        with Image.open(image_path) as im:
            width, height = im.size

        p0 = (landmark_vals[0], landmark_vals[1])
        p1 = (landmark_vals[2], landmark_vals[3])
        p0, p1 = canonical_order(p0, p1, d_vect)

        if image_name not in seen_filenames:
            seen_filenames.add(image_name)
            images.append({
                "id": image_id, "file_name": image_name,
                "width": width, "height": height,
            })
            current_image_id = image_id
            image_id += 1
        else:
            # Should not happen for these single-row-per-image CSVs, but
            # fail loudly rather than silently duplicating an image entry.
            raise SystemExit(f"ERROR: duplicate image_name in CSV: {image_name}")

        annotations.append({
            "id": ann_id,
            "image_id": current_image_id,
            "category_id": 1,
            "iscrowd": 0,
            "bbox": [0, 0, width, height],
            "area": float(width * height),
            "num_keypoints": 2,
            "keypoints": [p0[0], p0[1], 2, p1[0], p1[1], 2],
        })
        ann_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{
            "id": 1,
            "name": f"{dataset.lower()}_{task.lower()}",
            "keypoints": ["endpoint_0", "endpoint_1"],
            "skeleton": [[1, 2]],
        }],
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as handle:
        json.dump(coco, handle)

    excluded_log.parent.mkdir(parents=True, exist_ok=True)
    with open(excluded_log, "w", encoding="utf-8") as handle:
        json.dump(excluded, handle, indent=2)

    summary = {
        "csv_rows": len(rows), "kept": len(images), "excluded": len(excluded),
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--dataset", required=True, choices=["UCL", "MULTICENTRE"])
    parser.add_argument("--task", required=True, choices=["BPD", "OFD", "APAD", "TAD", "FL"])
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--excluded-log", required=True, type=Path)
    parser.add_argument("--internal-split-json", type=Path, default=None,
                         help="make_internal_val_split.py's output; if given, "
                              "--internal-split-part selects which half to convert")
    parser.add_argument("--internal-split-part", choices=["internal_train", "internal_val"],
                         default=None)
    args = parser.parse_args()

    if Image is None:
        raise SystemExit("ERROR: Pillow (PIL) is required for real conversion; pip install pillow")

    filename_subset = None
    if args.internal_split_json is not None:
        if args.internal_split_part is None:
            raise SystemExit("ERROR: --internal-split-part is required when "
                              "--internal-split-json is given")
        manifest = json.loads(args.internal_split_json.read_text(encoding="utf-8"))
        filename_subset = set(manifest[f"{args.internal_split_part}_filenames"])

    summary = convert(args.csv, args.images_dir, args.dataset, args.task,
                       args.out_json, args.excluded_log, filename_subset=filename_subset)
    print(f"[OK] {args.dataset} {args.task}: {summary}")


if __name__ == "__main__":
    main()
