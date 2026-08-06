"""End-to-end test of rescore_endpoint_conventions.py against SYNTHETIC
per-image files matching the real HRNet/EoMT CSV schemas exactly (HRNet:
baseline_reproduction/evaluate_hrnet_fixed.py's own writer; EoMT:
training/landmark_detection.py's test_nme_dump_path writer) -- since no
real per-image files exist locally (confirmed: they live only on the
server), this is the only local verification possible, but it exercises
the REAL loading/canonicalisation/bootstrap code paths, not a separate
reimplementation.

Run directly: python endpoint_ordering_analysis/test_rescore_endpoint_conventions.py
"""

from __future__ import annotations

import csv
import shutil
import tempfile
from pathlib import Path

import numpy as np

from rescore_endpoint_conventions import (
    dod_sort,
    fixed_channel_nme,
    load_eomt_per_image,
    load_hrnet_per_image,
    rescore_cell,
    x_sort,
)

SEEDS = (42, 0, 123, 2024, 3407)


def _write_hrnet_synthetic(root: Path, dataset: str, task_tag: str, filenames_and_points):
    for seed in SEEDS:
        run_dir = root / f"fetal_landmark_hrnet_w18_{dataset}_{task_tag}_seed{seed}_512fixed"
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / "fixed_channel_per_image.csv"
        fields = ["index", "filename", "pred0_x", "pred0_y", "pred1_x", "pred1_y",
                  "gt0_x", "gt0_y", "gt1_x", "gt1_y", "reference_distance",
                  "fixed_channel_nme", "swap_min_nme"]
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for i, (fn, gt0, gt1, pred0, pred1) in enumerate(filenames_and_points):
                ref = float(np.hypot(gt0[0] - gt1[0], gt0[1] - gt1[1]))
                fixed = fixed_channel_nme(pred0, pred1, gt0, gt1)
                writer.writerow({
                    "index": i, "filename": fn,
                    "pred0_x": pred0[0], "pred0_y": pred0[1],
                    "pred1_x": pred1[0], "pred1_y": pred1[1],
                    "gt0_x": gt0[0], "gt0_y": gt0[1],
                    "gt1_x": gt1[0], "gt1_y": gt1[1],
                    "reference_distance": ref,
                    "fixed_channel_nme": fixed,
                    "swap_min_nme": min(fixed, fixed_channel_nme(pred1, pred0, gt0, gt1)),
                })


def _write_eomt_synthetic(root: Path, task: str, backbone: str, filenames_and_points,
                           is_multicentre: bool = False):
    for seed in SEEDS:
        if is_multicentre:
            run_dir = root / f"multicentre-{task}-{backbone}" / f"seed{seed}"
            nme_name = f"seed{seed}_final_fixedchannel_per_image.csv"
        else:
            run_dir = root / f"{task}_{backbone}" / f"seed{seed}"
            nme_name = "final_fixedchannel_per_image.csv"
        run_dir.mkdir(parents=True, exist_ok=True)

        order_path = run_dir / "test_image_order.csv"
        with order_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["index", "img_name"])
            for i, (fn, *_rest) in enumerate(filenames_and_points):
                writer.writerow([i, fn])

        nme_path = run_dir / nme_name
        fields = ["index", "nme", "pixel_error", "pred_x0", "pred_y0", "gt_x0", "gt_y0",
                  "pred_x1", "pred_y1", "gt_x1", "gt_y1"]
        with nme_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for i, (fn, gt0, gt1, pred0, pred1) in enumerate(filenames_and_points):
                nme = fixed_channel_nme(pred0, pred1, gt0, gt1)
                writer.writerow({
                    "index": i, "nme": nme, "pixel_error": "",
                    "pred_x0": pred0[0], "pred_y0": pred0[1],
                    "gt_x0": gt0[0], "gt_y0": gt0[1],
                    "pred_x1": pred1[0], "pred_y1": pred1[1],
                    "gt_x1": gt1[0], "gt_y1": gt1[1],
                })


def test_hrnet_loader_and_native_nme_matches_stored_value():
    tmp = Path(tempfile.mkdtemp())
    try:
        samples = [
            ("img1.jpg", (100.0, 100.0), (200.0, 100.0), (105.0, 102.0), (198.0, 99.0)),
            ("img2.jpg", (50.0, 300.0), (50.0, 100.0), (52.0, 298.0), (48.0, 103.0)),
        ]
        _write_hrnet_synthetic(tmp, "UCL", "brain_BPD", samples)
        data = load_hrnet_per_image(tmp, "UCL", "bpd")
        assert data["filenames"] == ["img1.jpg", "img2.jpg"]
        for seed in SEEDS:
            row = data["per_seed"][seed]["img1.jpg"]
            expected = fixed_channel_nme(row["pred0"], row["pred1"], row["gt0"], row["gt1"])
            assert abs(row["native_fixed_nme"] - expected) < 1e-9
        print("[PASS] test_hrnet_loader_and_native_nme_matches_stored_value")
    finally:
        shutil.rmtree(tmp)


def test_eomt_loader_joins_order_and_coords_correctly():
    tmp = Path(tempfile.mkdtemp())
    try:
        samples = [
            ("a.jpg", (10.0, 10.0), (90.0, 10.0), (12.0, 11.0), (88.0, 9.0)),
            ("b.jpg", (5.0, 5.0), (5.0, 95.0), (6.0, 6.0), (4.0, 94.0)),
        ]
        _write_eomt_synthetic(tmp, "bpd", "dinov2", samples, is_multicentre=False)
        data = load_eomt_per_image(tmp, "UCL", "bpd", "dinov2")
        assert data["filenames"] == ["a.jpg", "b.jpg"]
        row = data["per_seed"][42]["a.jpg"]
        assert row["gt0"] == (10.0, 10.0) and row["gt1"] == (90.0, 10.0)
        print("[PASS] test_eomt_loader_joins_order_and_coords_correctly")
    finally:
        shutil.rmtree(tmp)


def test_rescore_cell_recovers_x_sort_and_dod_correctly():
    """Construct a case where native (x-sort, simulating EoMT), unified
    x-sort, and unified DOD are all DIFFERENT, and check each is computed
    correctly by hand."""
    tmp = Path(tempfile.mkdtemp())
    try:
        # GT diameter is near-vertical (small dx, large dy) -- matches the
        # real UCL BPD d_vect's own near-vertical character. Raw CSV order
        # is deliberately "backwards" relative to x-sort so the difference
        # between conventions is visible.
        gt0_raw, gt1_raw = (301.0, 500.0), (300.0, 100.0)  # raw order: larger-x point first
        pred0_raw, pred1_raw = (299.0, 498.0), (302.0, 102.0)
        samples = [("only.jpg", gt0_raw, gt1_raw, pred0_raw, pred1_raw)]
        _write_hrnet_synthetic(tmp, "UCL", "brain_BPD", samples)
        data = load_hrnet_per_image(tmp, "UCL", "bpd")

        d_vect = ((0.0, 0.0), (0.0, 1.0))  # purely vertical synthetic direction
        rescored = rescore_cell(data, d_vect)

        row = rescored["per_seed_per_image"][42]["only.jpg"]

        # Hand-computed expectations:
        gt_x0, gt_x1 = x_sort(gt0_raw, gt1_raw)  # ascending x -> gt1_raw first (x=300 < 301)
        assert gt_x0 == gt1_raw and gt_x1 == gt0_raw
        pred_x0, pred_x1 = x_sort(pred0_raw, pred1_raw)
        expected_xsort_nme = fixed_channel_nme(pred_x0, pred_x1, gt_x0, gt_x1)
        assert abs(row["xsort"] - expected_xsort_nme) < 1e-9

        gt_d0, gt_d1 = dod_sort(gt0_raw, gt1_raw, d_vect)  # ascending y -> gt1_raw (y=100) first
        assert gt_d0 == gt1_raw and gt_d1 == gt0_raw
        pred_d0, pred_d1 = dod_sort(pred0_raw, pred1_raw, d_vect)
        expected_dod_nme = fixed_channel_nme(pred_d0, pred_d1, gt_d0, gt_d1)
        assert abs(row["dod"] - expected_dod_nme) < 1e-9

        # For this near-vertical case, x-sort and DOD happen to agree on GT
        # order here (both pick gt1_raw as channel 0) -- disagreement rate
        # should be 0 for this specific synthetic sample.
        assert rescored["gt_disagreement_rate"] == 0.0
        print("[PASS] test_rescore_cell_recovers_x_sort_and_dod_correctly")
    finally:
        shutil.rmtree(tmp)


def test_gt_disagreement_rate_detects_real_disagreement():
    """A horizontal-diameter GT pair with a vertical d_vect: x-sort and DOD
    projection should DISAGREE on which point is channel 0."""
    tmp = Path(tempfile.mkdtemp())
    try:
        # Horizontal GT: x differs a lot, y differs a little -- x-sort order
        # is clear (by x); vertical d_vect's projection depends only on y,
        # which barely differs here -- construct y so the two rules pick
        # OPPOSITE points.
        gt0_raw, gt1_raw = (100.0, 205.0), (500.0, 195.0)  # x-sort: gt0 first; y-proj (vertical d): gt1 first (195<205)
        pred0_raw, pred1_raw = (102.0, 204.0), (498.0, 196.0)
        samples = [("v.jpg", gt0_raw, gt1_raw, pred0_raw, pred1_raw)]
        _write_hrnet_synthetic(tmp, "UCL", "brain_BPD", samples)
        data = load_hrnet_per_image(tmp, "UCL", "bpd")

        d_vect = ((0.0, 0.0), (0.0, 1.0))  # purely vertical
        rescored = rescore_cell(data, d_vect)

        gt_x0, _ = x_sort(gt0_raw, gt1_raw)
        gt_d0, _ = dod_sort(gt0_raw, gt1_raw, d_vect)
        assert gt_x0 != gt_d0, "test construction error: expected the two rules to disagree"
        assert rescored["gt_disagreement_rate"] == 1.0
        print("[PASS] test_gt_disagreement_rate_detects_real_disagreement")
    finally:
        shutil.rmtree(tmp)


def main():
    test_hrnet_loader_and_native_nme_matches_stored_value()
    test_eomt_loader_joins_order_and_coords_correctly()
    test_rescore_cell_recovers_x_sort_and_dod_correctly()
    test_gt_disagreement_rate_detects_real_disagreement()
    print("[ALL ENDPOINT-ORDERING-ANALYSIS TESTS PASSED]")


if __name__ == "__main__":
    main()
