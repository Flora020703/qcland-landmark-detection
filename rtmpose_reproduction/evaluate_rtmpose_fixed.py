"""External fixed-channel / swap-min NME evaluator for RTMPose predictions,
using the identical formula as baseline_reproduction/evaluate_hrnet_fixed.py
so RTMPose's numbers are computed by the same evaluator as HRNet's, not a
method-specific reimplementation that could quietly diverge.

Inputs are both already in ORIGINAL image pixel coordinates:
  - --gt-json:          the COCO json produced by convert_csv_to_coco.py
                         (ground truth, already DOD-canonicalised).
  - --predictions-json: a list of {"file_name": ..., "pred": [[x0,y0],[x1,y1]]}
                         records, produced by the inference driver AFTER
                         applying geometry.to_image_space() -- this script
                         does not do any coordinate transform itself, only
                         scores already-original-space coordinates. It is
                         the driver's job to ensure MMPose's own
                         bbox/aspect-ratio inverse transform was never
                         invoked; this evaluator cannot detect that mistake
                         from the numbers alone.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def _load_gt(gt_json_path: Path) -> dict[str, tuple[tuple[float, float], tuple[float, float]]]:
    coco = json.loads(gt_json_path.read_text(encoding="utf-8"))
    images_by_id = {im["id"]: im["file_name"] for im in coco["images"]}
    gt = {}
    for ann in coco["annotations"]:
        file_name = images_by_id[ann["image_id"]]
        kp = ann["keypoints"]
        gt[file_name] = ((kp[0], kp[1]), (kp[3], kp[4]))
    return gt


def _load_predictions(pred_json_path: Path) -> dict[str, tuple[tuple[float, float], tuple[float, float]]]:
    records = json.loads(pred_json_path.read_text(encoding="utf-8"))
    preds = {}
    for rec in records:
        (x0, y0), (x1, y1) = rec["pred"]
        preds[rec["file_name"]] = ((x0, y0), (x1, y1))
    return preds


def evaluate(gt_json_path: Path, pred_json_path: Path,
             per_image_csv: Path, summary_json: Path) -> dict:
    gt = _load_gt(gt_json_path)
    preds = _load_predictions(pred_json_path)

    missing = sorted(set(gt) - set(preds))
    if missing:
        raise SystemExit(f"ERROR: {len(missing)} GT images have no prediction; first={missing[:5]}")
    extra = sorted(set(preds) - set(gt))
    if extra:
        raise SystemExit(f"ERROR: {len(extra)} predictions have no matching GT image; first={extra[:5]}")

    rows = []
    fixed_values, swap_values = [], []
    for file_name in sorted(gt):
        gt0, gt1 = gt[file_name]
        pred0, pred1 = preds[file_name]

        ref = math.hypot(gt0[0] - gt1[0], gt0[1] - gt1[1])
        if not math.isfinite(ref) or ref <= 1e-12:
            raise SystemExit(f"ERROR: invalid reference distance for {file_name}: {ref}")

        standard = math.hypot(pred0[0] - gt0[0], pred0[1] - gt0[1]) + \
                   math.hypot(pred1[0] - gt1[0], pred1[1] - gt1[1])
        swapped = math.hypot(pred0[0] - gt1[0], pred0[1] - gt1[1]) + \
                  math.hypot(pred1[0] - gt0[0], pred1[1] - gt0[1])

        fixed = standard / (2.0 * ref)
        swap = min(standard, swapped) / (2.0 * ref)
        fixed_values.append(fixed)
        swap_values.append(swap)

        rows.append({
            "filename": file_name,
            "pred0_x": pred0[0], "pred0_y": pred0[1],
            "pred1_x": pred1[0], "pred1_y": pred1[1],
            "gt0_x": gt0[0], "gt0_y": gt0[1],
            "gt1_x": gt1[0], "gt1_y": gt1[1],
            "reference_distance": ref,
            "fixed_channel_nme": fixed,
            "swap_min_nme": swap,
        })

    per_image_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(per_image_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fixed_arr = np.asarray(fixed_values)
    swap_arr = np.asarray(swap_values)
    summary = {
        "n": len(rows),
        "fixed_channel_mean_pct": float(fixed_arr.mean() * 100.0),
        "fixed_channel_sample_sd_pct": float(fixed_arr.std(ddof=1) * 100.0) if len(rows) > 1 else float("nan"),
        "swap_min_mean_pct": float(swap_arr.mean() * 100.0),
        "swap_min_sample_sd_pct": float(swap_arr.std(ddof=1) * 100.0) if len(rows) > 1 else float("nan"),
    }
    if summary["fixed_channel_mean_pct"] + 1e-10 < summary["swap_min_mean_pct"]:
        raise SystemExit("ERROR: metric invariant failed: fixed-channel mean is below swap-min mean")

    summary_json.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-json", required=True, type=Path)
    parser.add_argument("--predictions-json", required=True, type=Path)
    parser.add_argument("--per-image-csv", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    args = parser.parse_args()
    summary = evaluate(args.gt_json, args.predictions_json, args.per_image_csv, args.summary_json)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
