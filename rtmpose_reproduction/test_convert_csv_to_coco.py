"""End-to-end test of convert_csv_to_coco.py against synthetic images + a
synthetic CSV -- checks row filtering (missing image, negative landmark),
bbox correctness for non-square images, and that canonical_order() is
actually applied (using the real UCL/BPD frozen d_vect and the same
004_HC.jpeg-style swap case already verified in test_endpoint_order.py).

Run directly: python rtmpose_reproduction/test_convert_csv_to_coco.py
"""

from __future__ import annotations

import csv
import json
import shutil
import tempfile
from pathlib import Path

from PIL import Image

from convert_csv_to_coco import convert

HEADER = [
    "image_name", "scale", "center_w", "center_h",
    "ofd_1_x", "ofd_1_y", "ofd_2_x", "ofd_2_y",
    "bpd_1_x", "bpd_1_y", "bpd_2_x", "bpd_2_y",
    "SubjectID", "px_to_mm_rate", "Algo", "Split",
]


def _write_image(path: Path, width: int, height: int):
    Image.new("RGB", (width, height), color=(0, 0, 0)).save(path)


def main():
    tmp = Path(tempfile.mkdtemp(prefix="rtmpose_convert_test_"))
    try:
        images_dir = tmp / "images"
        images_dir.mkdir()

        # Non-square (matches the geometry test's "non-square" concern) and
        # deliberately different from every other image's size.
        _write_image(images_dir / "keep_a.png", 640, 480)
        # Same swap case already verified against real HRNet output.
        _write_image(images_dir / "keep_swap.png", 800, 600)
        # "missing_row.png" is intentionally NOT created -> excluded.

        rows = [
            # kept, no swap expected: any two points with proj0<=proj1 under
            # UCL/BPD's real d_vect will do; reuse the already-verified
            # 001_HC.jpg case's raw coordinates.
            ["keep_a.png", "1.0", "0", "0", "0", "0", "0", "0",
             "565", "112", "587.0", "464.0", "", "", "", "Test"],
            # kept, swap expected: reuse the already-verified 004_HC.jpeg case.
            ["keep_swap.png", "1.0", "0", "0", "0", "0", "0", "0",
             "403", "532", "399.0", "133.0", "", "", "", "Test"],
            # excluded: negative landmark value
            ["negative_landmark.png", "1.0", "0", "0", "0", "0", "0", "0",
             "-1", "112", "587.0", "464.0", "", "", "", "Test"],
            # excluded: image file does not exist on disk
            ["missing_row.png", "1.0", "0", "0", "0", "0", "0", "0",
             "565", "112", "587.0", "464.0", "", "", "", "Test"],
        ]
        csv_path = tmp / "Head_Test.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(HEADER)
            writer.writerows(rows)

        out_json = tmp / "out.json"
        excluded_log = tmp / "excluded.json"
        summary = convert(csv_path, images_dir, "UCL", "BPD", out_json, excluded_log)

        assert summary == {"csv_rows": 4, "kept": 2, "excluded": 2}, summary

        coco = json.loads(out_json.read_text(encoding="utf-8"))
        assert len(coco["images"]) == 2
        assert len(coco["annotations"]) == 2

        by_name = {im["file_name"]: im for im in coco["images"]}
        assert by_name["keep_a.png"]["width"] == 640
        assert by_name["keep_a.png"]["height"] == 480
        assert by_name["keep_swap.png"]["width"] == 800
        assert by_name["keep_swap.png"]["height"] == 600

        ann_by_image_id = {a["image_id"]: a for a in coco["annotations"]}
        ann_a = ann_by_image_id[by_name["keep_a.png"]["id"]]
        assert ann_a["bbox"] == [0, 0, 640, 480], ann_a["bbox"]
        # No swap expected: keypoints should equal raw (565,112),(587,464).
        kp = ann_a["keypoints"]
        assert abs(kp[0] - 565.0) < 1e-6 and abs(kp[1] - 112.0) < 1e-6
        assert abs(kp[3] - 587.0) < 1e-6 and abs(kp[4] - 464.0) < 1e-6

        ann_swap = ann_by_image_id[by_name["keep_swap.png"]["id"]]
        assert ann_swap["bbox"] == [0, 0, 800, 600], ann_swap["bbox"]
        # Swap expected: raw (403,532),(399,133) -> canonical (399,133),(403,532).
        kp = ann_swap["keypoints"]
        assert abs(kp[0] - 399.0) < 1e-6 and abs(kp[1] - 133.0) < 1e-6
        assert abs(kp[3] - 403.0) < 1e-6 and abs(kp[4] - 532.0) < 1e-6

        excluded = json.loads(excluded_log.read_text(encoding="utf-8"))
        reasons = {e["image_name"]: e["reason"] for e in excluded}
        assert reasons["negative_landmark.png"] == "missing/negative landmark"
        assert reasons["missing_row.png"] == "image file not found"

        print("[PASS] test_convert_csv_to_coco end-to-end (filtering, bbox, canonical order)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
