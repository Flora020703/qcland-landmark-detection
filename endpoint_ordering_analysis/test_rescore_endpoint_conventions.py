"""End-to-end test of rescore_endpoint_conventions.py against SYNTHETIC
per-image files matching the real HRNet/EoMT CSV schemas exactly (HRNet:
baseline_reproduction/evaluate_hrnet_fixed.py's own writer; EoMT:
training/landmark_detection.py's test_nme_dump_path writer) -- since no
real per-image files exist locally (confirmed: they live only on the
server), this is the only local verification possible, but it exercises
the REAL loading/canonicalisation/bootstrap code paths, not a separate
reimplementation.

Includes coverage (added 2026-08-07, fixing a bug a review round found in
the first version of this script) for the EoMT coordinate-SPACE inversion:
EoMT's own dump is in 512x512 model-input space, and load_eomt_per_image
must convert it to true original-image pixel space using each image's REAL
width/height (opened from an actual file on disk, via
rtmpose_reproduction/geometry.py's to_image_space -- the exact inverse of
EoMT's own resize). The synthetic images written here are deliberately
NON-SQUARE (e.g. 300x600) specifically so an un-fixed version of this
script (feeding raw 512-space coordinates straight into dod_sort next to
an original-space d_vect, or computing NME directly in 512-space) would
fail these assertions -- a square synthetic image cannot expose an
anisotropic-scaling bug.

Also includes coverage (added 2026-08-07, second same-day round) for a
SECOND, subtler discrepancy in the same conversion: EoMT's real dump is
NOT simply `to_model_space()` of the original point -- it round-trips
through a pixel-centre-aligned heatmap encode (in
datasets/landmark_dataset.py) composed with a naive (non-pixel-centre)
heatmap-to-"img_size" scale-back (in training/landmark_detection.py),
leaving a constant +/-3.5px (for 512/64) offset relative to true
model-input-space coordinates. `_simulate_real_eomt_dump()` below
reproduces this EXACT real chain (verified against both source files line
by line, not assumed) so these tests fail if `load_eomt_per_image` ever
used a naive `to_model_space()`-only (no heatmap round-trip) recovery
instead of `_heatmap_dump_to_model_input_space()` + `to_image_space()`.

Run directly: python endpoint_ordering_analysis/test_rescore_endpoint_conventions.py
"""

from __future__ import annotations

import csv
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "rtmpose_reproduction"))
from geometry import to_image_space, to_model_space  # noqa: E402

from rescore_endpoint_conventions import (
    EOMT_HEATMAP_SIZE,
    EOMT_MODEL_INPUT_SIZE,
    LoadError,
    _ImageSizeCache,
    _heatmap_dump_to_model_input_space,
    _min_paired_max_abs_diff,
    cross_method_gt_consistency_check,
    dod_sort,
    fixed_channel_nme,
    load_eomt_per_image,
    load_hrnet_per_image,
    rescore_cell,
    x_sort,
)

SEEDS = (42, 0, 123, 2024, 3407)
EOMT_INPUT_SIZE = EOMT_MODEL_INPUT_SIZE


def _simulate_real_eomt_dump(orig_point: tuple[float, float], width: float, height: float) -> tuple[float, float]:
    """Reproduces EXACTLY what training/landmark_detection.py's real
    per-image dump writes for an original-image-space point, under a
    pixel_center_align=True, heatmap_size=64, img_size=512 config (the
    matched-protocol setting every analyzed EoMT run uses) -- verified line
    by line against datasets/landmark_dataset.py (orig->input, then
    pixel-centre input->heatmap encode) and training/landmark_detection.py
    (naive heatmap->"img_size" scale-back, `coord_scale = img_size/heatmap_size`).
    NOT a shortcut via to_model_space() alone -- that would skip the
    heatmap round-trip and fail to reproduce the real +3.5px discrepancy
    this fix specifically addresses."""
    input_x, input_y = to_model_space(*orig_point, width, height, EOMT_MODEL_INPUT_SIZE)
    hm_scale = EOMT_HEATMAP_SIZE / EOMT_MODEL_INPUT_SIZE
    heatmap_x = (input_x + 0.5) * hm_scale - 0.5
    heatmap_y = (input_y + 0.5) * hm_scale - 0.5
    dump_scale = EOMT_MODEL_INPUT_SIZE / EOMT_HEATMAP_SIZE
    return heatmap_x * dump_scale, heatmap_y * dump_scale


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


def _write_synthetic_image(images_root: Path, anatomy: str, filename: str, width: int, height: int):
    """Writes a REAL image file of the given (possibly non-square)
    dimensions -- load_eomt_per_image's _ImageSizeCache opens the actual
    file via PIL, it does not accept a pre-supplied size."""
    directory = images_root / anatomy
    directory.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (width, height), color=(128, 128, 128)).save(directory / filename)


def _write_eomt_synthetic(root: Path, task: str, backbone: str,
                           filenames_points_and_sizes, is_multicentre: bool = False):
    """filenames_points_and_sizes: list of (filename, gt0, gt1, pred0, pred1,
    orig_width, orig_height), all points in TRUE ORIGINAL image space. This
    helper runs each point through `_simulate_real_eomt_dump()` (the real
    orig->input->heatmap->dump chain, including the pixel-centre-vs-naive
    discrepancy) before writing the CSV, exactly mirroring what
    training/landmark_detection.py's real dump actually contains, so that a
    correct load_eomt_per_image() has to invert BOTH steps to recover the
    original-space points passed in here."""
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
            for i, (fn, *_rest) in enumerate(filenames_points_and_sizes):
                writer.writerow([i, fn])

        nme_path = run_dir / nme_name
        fields = ["index", "nme", "pixel_error", "pred_x0", "pred_y0", "gt_x0", "gt_y0",
                  "pred_x1", "pred_y1", "gt_x1", "gt_y1"]
        with nme_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for i, (fn, gt0, gt1, pred0, pred1, width, height) in enumerate(filenames_points_and_sizes):
                # EoMT's real dump is NOT plain 512-model-input-space
                # coordinates -- it round-trips through a pixel-centre
                # heatmap encode + naive scale-back, leaving a constant
                # offset relative to true model-input-space (see this
                # module's docstring and _simulate_real_eomt_dump). Its
                # "nme" column is computed FROM these same dumped points,
                # in this same (slightly-off) space, exactly as the real
                # training code does -- so the native-reproduction sanity
                # check must still pass on these values as-is.
                gt0_dump = _simulate_real_eomt_dump(gt0, width, height)
                gt1_dump = _simulate_real_eomt_dump(gt1, width, height)
                pred0_dump = _simulate_real_eomt_dump(pred0, width, height)
                pred1_dump = _simulate_real_eomt_dump(pred1, width, height)
                nme = fixed_channel_nme(pred0_dump, pred1_dump, gt0_dump, gt1_dump)
                writer.writerow({
                    "index": i, "nme": nme, "pixel_error": "",
                    "pred_x0": pred0_dump[0], "pred_y0": pred0_dump[1],
                    "gt_x0": gt0_dump[0], "gt_y0": gt0_dump[1],
                    "pred_x1": pred1_dump[0], "pred_y1": pred1_dump[1],
                    "gt_x1": gt1_dump[0], "gt_y1": gt1_dump[1],
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
    """Uses a NON-SQUARE synthetic original image (300x600) specifically so
    this test would fail if load_eomt_per_image ever stopped converting
    512-model-space back to original-image space (or converted using the
    wrong width/height) -- a square image can't expose that bug since its
    resize-to-512 is isotropic."""
    tmp = Path(tempfile.mkdtemp())
    images_root = Path(tempfile.mkdtemp())
    try:
        width, height = 300.0, 600.0
        samples = [
            ("a.jpg", (10.0, 10.0), (90.0, 10.0), (12.0, 11.0), (88.0, 9.0), width, height),
            ("b.jpg", (5.0, 5.0), (5.0, 95.0), (6.0, 6.0), (4.0, 94.0), width, height),
        ]
        _write_eomt_synthetic(tmp, "bpd", "dinov2", samples, is_multicentre=False)
        _write_synthetic_image(images_root, "Head", "a.jpg", int(width), int(height))
        _write_synthetic_image(images_root, "Head", "b.jpg", int(width), int(height))
        cache = _ImageSizeCache(images_root, "Head")

        data = load_eomt_per_image(tmp, "UCL", "bpd", "dinov2", cache)
        assert data["filenames"] == ["a.jpg", "b.jpg"]
        row = data["per_seed"][42]["a.jpg"]
        # After correct round-trip inversion, recovered original-space
        # coordinates should match the ORIGINAL points passed to the
        # synthetic writer (small float tolerance only), NOT the raw
        # 512-space numbers actually stored in the CSV.
        assert abs(row["gt0"][0] - 10.0) < 1e-6 and abs(row["gt0"][1] - 10.0) < 1e-6
        assert abs(row["gt1"][0] - 90.0) < 1e-6 and abs(row["gt1"][1] - 10.0) < 1e-6
        assert abs(row["pred0"][0] - 12.0) < 1e-6 and abs(row["pred0"][1] - 11.0) < 1e-6
        print("[PASS] test_eomt_loader_joins_order_and_coords_correctly")
    finally:
        shutil.rmtree(tmp)
        shutil.rmtree(images_root)


def test_eomt_loader_inverts_anisotropic_coordinate_space_correctly():
    """The core regression test for the coordinate-space bug: constructs a
    case where the GT diameter is HORIZONTAL in original-image space (so a
    vertical d_vect and x-sort would DISAGREE the same way DOD-vs-x-sort
    normally disagrees), on a strongly non-square (150x600, 1:4) original
    image. If load_eomt_per_image ever fed raw, un-inverted 512-space
    points into dod_sort alongside an original-space d_vect (the original
    bug), or computed NME directly on 512-space points, this test's
    hand-computed original-space expectations would not match."""
    tmp = Path(tempfile.mkdtemp())
    images_root = Path(tempfile.mkdtemp())
    try:
        width, height = 150.0, 600.0
        gt0, gt1 = (20.0, 305.0), (130.0, 295.0)  # horizontal-ish, small dy
        pred0, pred1 = (22.0, 303.0), (128.0, 296.0)
        samples = [("v.jpg", gt0, gt1, pred0, pred1, width, height)]
        _write_eomt_synthetic(tmp, "bpd", "dinov2", samples, is_multicentre=False)
        _write_synthetic_image(images_root, "Head", "v.jpg", int(width), int(height))
        cache = _ImageSizeCache(images_root, "Head")

        data = load_eomt_per_image(tmp, "UCL", "bpd", "dinov2", cache)
        row = data["per_seed"][42]["v.jpg"]
        assert abs(row["gt0"][0] - gt0[0]) < 1e-6 and abs(row["gt0"][1] - gt0[1]) < 1e-6
        assert abs(row["gt1"][0] - gt1[0]) < 1e-6 and abs(row["gt1"][1] - gt1[1]) < 1e-6

        d_vect = ((0.0, 0.0), (0.0, 1.0))  # purely vertical, like real near-vertical BPD d_vect
        rescored = rescore_cell(data, d_vect)
        out = rescored["per_seed_per_image"][42]["v.jpg"]

        gt_x0, gt_x1 = x_sort(gt0, gt1)  # ascending x -> gt0 first (20 < 130)
        pred_x0, pred_x1 = x_sort(pred0, pred1)
        expected_xsort_nme = fixed_channel_nme(pred_x0, pred_x1, gt_x0, gt_x1)
        assert abs(out["xsort"] - expected_xsort_nme) < 1e-6

        gt_d0, gt_d1 = dod_sort(gt0, gt1, d_vect)  # ascending y -> gt1 first (295 < 305)
        pred_d0, pred_d1 = dod_sort(pred0, pred1, d_vect)
        expected_dod_nme = fixed_channel_nme(pred_d0, pred_d1, gt_d0, gt_d1)
        assert abs(out["dod"] - expected_dod_nme) < 1e-6
        assert gt_x0 != gt_d0, "test construction error: x-sort and DOD should disagree here"

        # A naive, UN-FIXED loader would instead compute to_model_space of
        # these same original points and hand them straight to dod_sort
        # unconverted -- demonstrate that doing so gives a DIFFERENT (wrong)
        # NME, confirming this test actually exercises the fix rather than
        # passing vacuously regardless of whether inversion happened.
        gt0_512 = to_model_space(*gt0, width, height, EOMT_INPUT_SIZE)
        gt1_512 = to_model_space(*gt1, width, height, EOMT_INPUT_SIZE)
        pred0_512 = to_model_space(*pred0, width, height, EOMT_INPUT_SIZE)
        pred1_512 = to_model_space(*pred1, width, height, EOMT_INPUT_SIZE)
        naive_gt_d0, naive_gt_d1 = dod_sort(gt0_512, gt1_512, d_vect)
        naive_pred_d0, naive_pred_d1 = dod_sort(pred0_512, pred1_512, d_vect)
        naive_dod_nme = fixed_channel_nme(naive_pred_d0, naive_pred_d1, naive_gt_d0, naive_gt_d1)
        assert abs(naive_dod_nme - out["dod"]) > 1e-6, (
            "test construction error: naive (un-inverted) and fixed DOD NME "
            "should differ for this anisotropic case, or this test can't "
            "distinguish a correct fix from the original bug"
        )
        print("[PASS] test_eomt_loader_inverts_anisotropic_coordinate_space_correctly")
    finally:
        shutil.rmtree(tmp)
        shutil.rmtree(images_root)


def test_heatmap_dump_offset_recovery_matches_real_encode_chain():
    """Regression test for the SECOND coordinate-space bug (2026-08-07):
    EoMT's real dump is NOT plain to_model_space() of the original point --
    it round-trips through a pixel-centre heatmap encode + a naive
    (non-pixel-centre) scale-back, leaving a constant +3.5px offset (for
    512/64) relative to true model-input-space coordinates. A loader that
    used to_model_space()/to_image_space() alone (skipping
    _heatmap_dump_to_model_input_space) would recover a point shifted by a
    few pixels in ORIGINAL-image space -- small enough to not obviously
    break NME/ordering (a shared per-image translation, mostly cancels in
    pairwise quantities) but large enough to cause spurious cross-method GT
    mismatches (this exact discrepancy on every single image, in the
    consistency check the previous round added)."""
    tmp = Path(tempfile.mkdtemp())
    images_root = Path(tempfile.mkdtemp())
    try:
        width, height = 300.0, 640.0
        gt0, gt1 = (140.0, 600.0), (145.0, 30.0)
        pred0, pred1 = (142.0, 598.0), (143.0, 32.0)
        samples = [("h.jpg", gt0, gt1, pred0, pred1, width, height)]
        _write_eomt_synthetic(tmp, "bpd", "dinov2", samples, is_multicentre=False)
        _write_synthetic_image(images_root, "Head", "h.jpg", int(width), int(height))
        cache = _ImageSizeCache(images_root, "Head")

        data = load_eomt_per_image(tmp, "UCL", "bpd", "dinov2", cache)
        row = data["per_seed"][42]["h.jpg"]
        # Correct fix: exact round-trip recovery (sub-micro-pixel tolerance).
        assert abs(row["gt0"][0] - gt0[0]) < 1e-6 and abs(row["gt0"][1] - gt0[1]) < 1e-6
        assert abs(row["gt1"][0] - gt1[0]) < 1e-6 and abs(row["gt1"][1] - gt1[1]) < 1e-6

        # A loader missing the heatmap-offset recovery (to_image_space()
        # applied directly to the raw dump, as the FIRST fix round did)
        # would NOT recover the true original point -- demonstrate the
        # discrepancy is large enough to matter (not sub-pixel noise).
        with open(tmp / "bpd_dinov2" / "seed42" / "final_fixedchannel_per_image.csv", newline="") as f:
            raw_row = next(csv.DictReader(f))
        naive_recovered = to_image_space(float(raw_row["gt_x0"]), float(raw_row["gt_y0"]),
                                          width, height, EOMT_INPUT_SIZE)
        naive_err = max(abs(naive_recovered[0] - gt0[0]), abs(naive_recovered[1] - gt0[1]))
        assert naive_err > 0.5, (
            "test construction error: expected the heatmap-offset recovery "
            "to matter by more than trivial floating-point noise here"
        )
        print("[PASS] test_heatmap_dump_offset_recovery_matches_real_encode_chain "
              f"(naive-vs-fixed recovery error: {naive_err:.3f}px)")
    finally:
        shutil.rmtree(tmp)
        shutil.rmtree(images_root)


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


def test_eomt_loader_rejects_missing_images_root_file():
    """If the actual image file can't be found, load_eomt_per_image must
    raise LoadError naming the exact missing path -- NOT fall back to
    assuming a square image or any other guessed size, since that would
    silently reintroduce the coordinate-space bug this fix addresses."""
    tmp = Path(tempfile.mkdtemp())
    images_root = Path(tempfile.mkdtemp())
    try:
        samples = [("missing.jpg", (10.0, 10.0), (90.0, 10.0), (12.0, 11.0), (88.0, 9.0), 300.0, 600.0)]
        _write_eomt_synthetic(tmp, "bpd", "dinov2", samples, is_multicentre=False)
        cache = _ImageSizeCache(images_root, "Head")  # deliberately: image file never written
        try:
            load_eomt_per_image(tmp, "UCL", "bpd", "dinov2", cache)
            raise AssertionError("expected LoadError for missing image file")
        except LoadError as exc:
            assert "missing.jpg" in str(exc)
        print("[PASS] test_eomt_loader_rejects_missing_images_root_file")
    finally:
        shutil.rmtree(tmp)
        shutil.rmtree(images_root)


def test_native_sanity_check_detects_corrupted_stored_nme():
    """If a file's own stored native NME doesn't match what this script
    recomputes from that same file's own raw coordinates, load_*_per_image
    must raise LoadError rather than silently proceeding -- this is the
    sanity check that would have caught the original coordinate-space bug
    immediately (feeding 512-space points into an original-space-only NME
    computation would fail to reproduce EoMT's own stored 512-space native
    NME, since native reproduction must stay in native space)."""
    tmp = Path(tempfile.mkdtemp())
    try:
        run_dir = tmp / "fetal_landmark_hrnet_w18_UCL_brain_BPD_seed42_512fixed"
        for seed in SEEDS:
            run_dir = tmp / f"fetal_landmark_hrnet_w18_UCL_brain_BPD_seed{seed}_512fixed"
            run_dir.mkdir(parents=True, exist_ok=True)
            path = run_dir / "fixed_channel_per_image.csv"
            fields = ["index", "filename", "pred0_x", "pred0_y", "pred1_x", "pred1_y",
                      "gt0_x", "gt0_y", "gt1_x", "gt1_y", "reference_distance",
                      "fixed_channel_nme", "swap_min_nme"]
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerow({
                    "index": 0, "filename": "corrupt.jpg",
                    "pred0_x": 100.0, "pred0_y": 100.0, "pred1_x": 200.0, "pred1_y": 100.0,
                    "gt0_x": 100.0, "gt0_y": 100.0, "gt1_x": 200.0, "gt1_y": 100.0,
                    "reference_distance": 100.0,
                    "fixed_channel_nme": 999.0,  # deliberately wrong (true value is 0.0)
                    "swap_min_nme": 999.0,
                })
        try:
            load_hrnet_per_image(tmp, "UCL", "bpd")
            raise AssertionError("expected LoadError for native-sanity-check mismatch")
        except LoadError as exc:
            assert "sanity check FAILED" in str(exc)
        print("[PASS] test_native_sanity_check_detects_corrupted_stored_nme")
    finally:
        shutil.rmtree(tmp)


def test_cross_method_gt_consistency_check_detects_mismatch():
    """Two methods reporting a different GT location for the same filename
    (simulating a coordinate-recovery or sample-matching bug) must be
    flagged, not silently averaged away."""
    method_a = {
        "gt_by_filename": {"x.jpg": ((10.0, 10.0), (90.0, 10.0))},
        "disagree_by_filename": {"x.jpg": False},
    }
    method_b = {
        "gt_by_filename": {"x.jpg": ((10.0, 10.0), (200.0, 10.0))},  # gt1 badly wrong
        "disagree_by_filename": {"x.jpg": True},
    }
    warnings = cross_method_gt_consistency_check("UCL", "bpd", {"hrnet": method_a, "eomt_dinov2": method_b})
    assert len(warnings) == 1
    assert warnings[0]["per_image_disagreement_mismatches"] == 1
    print("[PASS] test_cross_method_gt_consistency_check_detects_mismatch")


def test_cross_method_gt_consistency_check_tolerates_channel_order_convention_difference():
    """Regression test for the THIRD bug (2026-08-07): HRNet's own native
    gt0/gt1 channel order follows DOD, EoMT's follows x-sort -- for the SAME
    two physical points, the two methods may legitimately disagree on which
    one is called 'channel 0' whenever x-sort and DOD disagree for that
    image (exactly the population this whole analysis studies). Method A's
    GT is (p0, p1); Method B's GT is the SAME two physical points with
    channels SWAPPED, (p1, p0). The consistency check must not flag this as
    a coordinate-recovery bug -- the warning list must be empty."""
    p0, p1 = (10.0, 10.0), (90.0, 10.0)
    method_a = {
        "gt_by_filename": {"x.jpg": (p0, p1)},
        "disagree_by_filename": {"x.jpg": False},
    }
    method_b = {
        "gt_by_filename": {"x.jpg": (p1, p0)},  # same physical points, swapped channel order
        "disagree_by_filename": {"x.jpg": False},
    }
    warnings = cross_method_gt_consistency_check("UCL", "bpd", {"hrnet": method_a, "eomt_dinov2": method_b})
    assert warnings == [], f"expected no warning for a pure channel-order convention difference, got {warnings}"
    print("[PASS] test_cross_method_gt_consistency_check_tolerates_channel_order_convention_difference")


def test_min_paired_max_abs_diff_picks_closer_pairing():
    """Direct unit test of the pairing-robust comparison helper itself."""
    bg0, bg1 = (10.0, 10.0), (90.0, 10.0)
    # Same physical points, swapped -> distance should be ~0 (min over both pairings).
    assert _min_paired_max_abs_diff(bg0, bg1, bg1, bg0) < 1e-9
    # Genuinely different second point -> distance should reflect the real gap.
    assert abs(_min_paired_max_abs_diff(bg0, bg1, bg1, (200.0, 10.0)) - 110.0) < 1e-9
    print("[PASS] test_min_paired_max_abs_diff_picks_closer_pairing")


def main():
    test_hrnet_loader_and_native_nme_matches_stored_value()
    test_eomt_loader_joins_order_and_coords_correctly()
    test_eomt_loader_inverts_anisotropic_coordinate_space_correctly()
    test_heatmap_dump_offset_recovery_matches_real_encode_chain()
    test_eomt_loader_rejects_missing_images_root_file()
    test_native_sanity_check_detects_corrupted_stored_nme()
    test_cross_method_gt_consistency_check_detects_mismatch()
    test_cross_method_gt_consistency_check_tolerates_channel_order_convention_difference()
    test_min_paired_max_abs_diff_picks_closer_pairing()
    test_rescore_cell_recovers_x_sort_and_dod_correctly()
    test_gt_disagreement_rate_detects_real_disagreement()
    print("[ALL ENDPOINT-ORDERING-ANALYSIS TESTS PASSED]")


if __name__ == "__main__":
    main()
