#!/usr/bin/env python3
"""Audit HRNet predictions under fixed-channel and swap-min endpoint NME."""

import argparse
import csv
import json
import math
import os
import sys

import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--per-image-csv", required=True)
    parser.add_argument("--summary-json", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    repo = os.path.abspath(args.repo)
    sys.path.insert(0, repo)

    from lib.config import config, update_config
    from lib.datasets import get_dataset

    class ConfigArgs:
        cfg = os.path.abspath(args.config)

    update_config(config, ConfigArgs())
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("d_vect") is None:
        raise SystemExit("checkpoint has no training-derived d_vect")
    d_vect = payload["d_vect"]
    if isinstance(d_vect, torch.Tensor):
        d_vect = d_vect.detach().cpu().numpy()

    predictions = torch.load(args.predictions, map_location="cpu", weights_only=False)
    predictions = np.asarray(predictions, dtype=np.float64)
    dataset = get_dataset(config)(config, is_train=False, d_vect=d_vect)
    expected_shape = (len(dataset), 2, 2)
    if predictions.shape != expected_shape:
        raise SystemExit(f"prediction shape {predictions.shape} != {expected_shape}")
    if not np.isfinite(predictions).all():
        raise SystemExit("predictions contain NaN/Inf")

    rows = []
    fixed_values, swap_values = [], []
    for index in range(len(dataset)):
        _, _, meta = dataset[index]
        gt = np.asarray(meta["pts"], dtype=np.float64)
        pred = predictions[index]
        ref = float(np.linalg.norm(gt[0] - gt[1]))
        if not math.isfinite(ref) or ref <= 1e-12:
            raise SystemExit(f"invalid endpoint reference distance at index {index}: {ref}")
        standard = float(np.linalg.norm(pred[0] - gt[0]) + np.linalg.norm(pred[1] - gt[1]))
        swapped = float(np.linalg.norm(pred[0] - gt[1]) + np.linalg.norm(pred[1] - gt[0]))
        fixed = standard / (2.0 * ref)
        swap = min(standard, swapped) / (2.0 * ref)
        fixed_values.append(fixed)
        swap_values.append(swap)
        rows.append({
            "index": index,
            "filename": str(dataset.landmarks_frame.iloc[index, 0]),
            "pred0_x": pred[0, 0], "pred0_y": pred[0, 1],
            "pred1_x": pred[1, 0], "pred1_y": pred[1, 1],
            "gt0_x": gt[0, 0], "gt0_y": gt[0, 1],
            "gt1_x": gt[1, 0], "gt1_y": gt[1, 1],
            "reference_distance": ref,
            "fixed_channel_nme": fixed,
            "swap_min_nme": swap,
        })

    fieldnames = list(rows[0])
    os.makedirs(os.path.dirname(os.path.abspath(args.per_image_csv)), exist_ok=True)
    with open(args.per_image_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    fixed_arr = np.asarray(fixed_values)
    swap_arr = np.asarray(swap_values)
    summary = {
        "n": len(rows),
        "fixed_channel_mean_pct": float(fixed_arr.mean() * 100.0),
        "fixed_channel_sample_sd_pct": float(fixed_arr.std(ddof=1) * 100.0),
        "swap_min_mean_pct": float(swap_arr.mean() * 100.0),
        "swap_min_sample_sd_pct": float(swap_arr.std(ddof=1) * 100.0),
        "fixed_minus_swap_mean_pt": float((fixed_arr - swap_arr).mean() * 100.0),
    }
    if summary["fixed_channel_mean_pct"] + 1e-10 < summary["swap_min_mean_pct"]:
        raise SystemExit("metric invariant failed: fixed-channel mean is below swap-min mean")
    with open(args.summary_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
