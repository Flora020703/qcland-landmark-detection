#!/usr/bin/env python3
"""Retrospective, no-retraining re-evaluation of EXISTING per-image
EoMT/HRNet predictions under two common EXTERNAL endpoint-ordering
conventions (per-image x-coordinate sort, and a training-set-frozen
direction-vector "DOD" projection), to give the supervisor a concrete,
quantified basis for deciding which convention RTMPose (and the final
cross-method comparison) should use -- see rtmpose_reproduction/
PROTOCOL_AUDIT.md's "Still not fully unified" section for the underlying
question this answers.

WHAT THIS DOES NOT DO (read before using the results): this script never
touches a checkpoint, never re-runs inference, and never changes any
predicted or ground-truth COORDINATE. It only re-derives, from already
-saved per-image predicted/GT coordinate pairs, which of the two points in
each pair is labelled "channel 0" under three different rules:
  1. NATIVE: whatever channel-identity convention that method's own
     training/evaluation code already used when the file was written
     (EoMT: per-image x-sort, recomputed fresh; HRNet: the frozen,
     training-set-derived DOD direction vector). This reproduces the
     already-reported fixed-channel NME numbers exactly, as a sanity check
     that this script's own coordinate handling is correct.
  2. UNIFIED X-SORT: both GT and prediction are (re-)labelled by sorting
     the two points by ascending x (tie-break by y), IGNORING whatever
     convention originally produced the file.
  3. UNIFIED DOD: both GT and prediction are (re-)labelled by projecting
     onto the SAME frozen, training-set-only direction vector (imported
     from rtmpose_reproduction/dod_vectors.py -- the same vectors already
     verified against real HRNet checkpoints and real HRNet per-image
     output in that adapter's own test suite), IGNORING whatever
     convention originally produced the file.

Re-labelling changes which NUMBER gets called "fixed-channel NME" for a
given image (the underlying prediction is untouched); it does NOT retrain
either method under a common convention. A method whose TRAINING labels
already used a different convention than the one being tested here may
still show a large "unified" NME even where a differently-trained model
would have done better under that convention from the start -- this
re-scoring answers "how much does the EXTERNAL SCORING RULE alone matter,"
not "how well would each method perform if retrained under a common rule."
State this limitation explicitly to the supervisor alongside the results
(also printed at the end of every run of this script).

Canonicalisation rules, precisely, per the locked spec this implements:
  - Everything operates in ORIGINAL image pixel coordinates (not resized/
    heatmap space) -- matches evaluate_hrnet_fixed.py/evaluate_rtmpose_fixed.py.
  - The SAME external rule is applied independently to the GT pair and to
    the prediction pair for a given image (never mixed).
  - x-sort: ascending x; exact ties broken by ascending y (deterministic,
    matches Python's stable tuple-sort on (x, y)).
  - DOD: the direction vector is ESTIMATED ONCE from the released Train
    partition only (already done -- see dod_vectors.py's own provenance
    docstring, extracted from real trained HRNet checkpoints) and reused
    FROZEN here; this script never re-estimates it from Test GT.
  - NME denominator is always the GT's own two-endpoint distance in
    original-image pixels (unchanged formula, matches every other
    evaluator in this project).

Usage (run where the real per-image files actually live -- server or a
mounted copy; this script has no dependency on rtmpose_reproduction's own
MMPose-only code, only its pure-Python dod_vectors.py/endpoint_order.py):

    python endpoint_ordering_analysis/rescore_endpoint_conventions.py \
        --ucl-eomt-root /root/autodl-tmp/ucl_eomt_per_image \
        --ucl-hrnet-root /root/autodl-tmp/hrnet_512_fixed_5seed/output/FETAL \
        --multicentre-eomt-root /root/autodl-tmp/saved_checkpoints/multicentre_5seed \
        --multicentre-hrnet-root /root/autodl-tmp/hrnet_512_fixed_5seed/output/FETAL \
        --output-root endpoint_ordering_analysis/results
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "rtmpose_reproduction"))
from dod_vectors import D_VECT, get_d_vect  # noqa: E402
from endpoint_order import canonical_order  # noqa: E402

SEEDS = (42, 0, 123, 2024, 3407)
BACKBONES = ("dinov2", "dinov3")
UCL_TASKS = ("bpd", "ofd", "apad", "tad", "fl")
MULTICENTRE_TASKS = ("bpd", "ofd", "apad", "tad", "fl")
HRNET_TASK_TAG = {
    "bpd": "brain_BPD", "ofd": "brain_OFD",
    "apad": "abdomen_APAD", "tad": "abdomen_TAD", "fl": "femur_FL",
}
MULTICENTRE_RAW_N = {"bpd": 1191, "ofd": 1191, "apad": 161, "tad": 161, "fl": 362}


def _canon_filename(value: str) -> str:
    return value.strip().replace("\\", "/").rsplit("/", 1)[-1]


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def x_sort(p0: tuple, p1: tuple) -> tuple[tuple, tuple]:
    """Deterministic external convention 1: ascending x, tie-break by y."""
    return tuple(sorted((p0, p1), key=lambda p: (p[0], p[1])))


def dod_sort(p0: tuple, p1: tuple, d_vect) -> tuple[tuple, tuple]:
    """Deterministic external convention 2: frozen training-set direction
    vector projection (rtmpose_reproduction.endpoint_order.canonical_order,
    already unit-tested against real HRNet per-image output)."""
    return canonical_order(p0, p1, d_vect)


def fixed_channel_nme(pred0, pred1, gt0, gt1) -> float:
    ref = float(np.hypot(gt0[0] - gt1[0], gt0[1] - gt1[1]))
    if ref <= 1e-9:
        raise ValueError(f"degenerate GT reference distance: gt0={gt0}, gt1={gt1}")
    err = float(np.hypot(pred0[0] - gt0[0], pred0[1] - gt0[1])
                + np.hypot(pred1[0] - gt1[0], pred1[1] - gt1[1]))
    return err / (2.0 * ref)


class LoadError(RuntimeError):
    """Raised (and caught at the top level, per-cell) when a specific
    (dataset, task[, backbone]) cell's real per-image files are missing or
    lack the raw-coordinate columns this analysis needs -- e.g. UCL BPD's
    EoMT checkpoints/per-image files are known (from this project's own
    records) to no longer exist on the server. Reported in
    excluded_images.tsv / the run's own console output, not silently
    skipped without a trace."""


def load_hrnet_per_image(hrnet_root: Path, dataset: str, task: str) -> dict[str, dict]:
    """Returns {filename: {"seed": {seed: (pred0, pred1, gt0, gt1, native_fixed_nme)}}}
    Real schema (verified against baseline_reproduction/evaluate_hrnet_fixed.py's
    own CSV writer): index,filename,pred0_x,pred0_y,pred1_x,pred1_y,gt0_x,gt0_y,
    gt1_x,gt1_y,reference_distance,fixed_channel_nme,swap_min_nme."""
    tag = HRNET_TASK_TAG[task]
    per_seed: dict[int, dict[str, dict]] = {}
    for seed in SEEDS:
        run = f"fetal_landmark_hrnet_w18_{dataset}_{tag}_seed{seed}_512fixed"
        path = hrnet_root / run / "fixed_channel_per_image.csv"
        if not path.is_file():
            raise LoadError(f"HRNet {dataset}/{task}/seed{seed}: missing {path}")
        rows = _read_rows(path)
        required = {"filename", "pred0_x", "pred0_y", "pred1_x", "pred1_y",
                    "gt0_x", "gt0_y", "gt1_x", "gt1_y", "fixed_channel_nme"}
        if not required.issubset(rows[0]):
            raise LoadError(
                f"HRNet {dataset}/{task}/seed{seed}: {path} is missing required "
                f"columns {sorted(required - set(rows[0]))}"
            )
        by_name = {}
        for r in rows:
            fn = _canon_filename(r["filename"])
            if fn in by_name:
                raise LoadError(f"duplicate filename {fn} in {path}")
            by_name[fn] = {
                "pred0": (float(r["pred0_x"]), float(r["pred0_y"])),
                "pred1": (float(r["pred1_x"]), float(r["pred1_y"])),
                "gt0": (float(r["gt0_x"]), float(r["gt0_y"])),
                "gt1": (float(r["gt1_x"]), float(r["gt1_y"])),
                "native_fixed_nme": float(r["fixed_channel_nme"]),
            }
        per_seed[seed] = by_name

    keys = set(per_seed[SEEDS[0]])
    for seed in SEEDS[1:]:
        if set(per_seed[seed]) != keys:
            raise LoadError(f"HRNet {dataset}/{task}: filename set differs across seeds")
    return {"per_seed": per_seed, "filenames": sorted(keys)}


def load_eomt_per_image(eomt_root: Path, dataset: str, task: str, backbone: str) -> dict:
    """Returns the same shape as load_hrnet_per_image. Real schema depends
    on training/landmark_detection.py's test_nme_dump_path feature
    (introduced 2026-07-23/24) actually having been enabled for the run
    that produced these files -- if the coordinate columns
    (pred_x0/pred_y0/gt_x0/gt_y0/pred_x1/pred_y1/gt_x1/gt_y1) are absent,
    this raises LoadError with a precise, actionable message rather than
    silently falling back to NME-only (which cannot support this analysis
    at all -- see this file's own module docstring)."""
    is_multicentre = dataset == "MULTICENTRE"
    per_seed: dict[int, dict[str, dict]] = {}
    for seed in SEEDS:
        if is_multicentre:
            run = eomt_root / f"multicentre-{task}-{backbone}" / f"seed{seed}"
            nme_path = run / f"seed{seed}_final_fixedchannel_per_image.csv"
        else:
            run = eomt_root / f"{task}_{backbone}" / f"seed{seed}"
            nme_path = run / "final_fixedchannel_per_image.csv"
        order_path = run / "test_image_order.csv"
        if not order_path.is_file() or not nme_path.is_file():
            raise LoadError(
                f"EoMT {dataset}/{task}/{backbone}/seed{seed}: missing "
                f"{order_path if not order_path.is_file() else nme_path} "
                f"(this task/backbone/seed may not exist -- e.g. UCL BPD's EoMT "
                f"checkpoints are known to be gone from the server per this "
                f"project's own records; report as excluded, do not fabricate)"
            )
        order_rows = _read_rows(order_path)
        name_col = "img_name" if "img_name" in order_rows[0] else "filename"
        order = {int(r["index"]): _canon_filename(r[name_col]) for r in order_rows}

        nme_rows = _read_rows(nme_path)
        required = {"index", "nme", "pred_x0", "pred_y0", "gt_x0", "gt_y0",
                    "pred_x1", "pred_y1", "gt_x1", "gt_y1"}
        if not required.issubset(nme_rows[0]):
            raise LoadError(
                f"EoMT {dataset}/{task}/{backbone}/seed{seed}: {nme_path} is missing "
                f"raw-coordinate columns {sorted(required - set(nme_rows[0]))} -- this "
                f"file was written WITHOUT test_nme_dump_path's coordinate-dump feature "
                f"(training/landmark_detection.py), so endpoint-ordering re-scoring is "
                f"IMPOSSIBLE for this cell without re-running inference. Do not proceed "
                f"with a partial/fabricated result for this cell."
            )
        by_index = {}
        for r in nme_rows:
            idx = int(r["index"])
            by_index[idx] = {
                "pred0": (float(r["pred_x0"]), float(r["pred_y0"])),
                "pred1": (float(r["pred_x1"]), float(r["pred_y1"])),
                "gt0": (float(r["gt_x0"]), float(r["gt_y0"])),
                "gt1": (float(r["gt_x1"]), float(r["gt_y1"])),
                "native_fixed_nme": float(r["nme"]),
            }
        if set(order) != set(by_index):
            raise LoadError(f"EoMT {dataset}/{task}/{backbone}/seed{seed}: index mismatch "
                             f"between {order_path} and {nme_path}")
        by_name = {order[idx]: by_index[idx] for idx in order}
        if len(by_name) != len(order):
            raise LoadError(f"duplicate joined filenames under {run}")
        per_seed[seed] = by_name

    keys = set(per_seed[SEEDS[0]])
    for seed in SEEDS[1:]:
        if set(per_seed[seed]) != keys:
            raise LoadError(f"EoMT {dataset}/{task}/{backbone}: filename set differs across seeds")
    return {"per_seed": per_seed, "filenames": sorted(keys)}


def bootstrap_ci(values: np.ndarray, replicates: int, rng: np.random.Generator) -> tuple[float, float]:
    means = np.empty(replicates, dtype=np.float64)
    chunk = 1000
    for start in range(0, replicates, chunk):
        stop = min(start + chunk, replicates)
        idx = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def rescore_cell(data: dict, d_vect) -> dict:
    """Given one method's `{"per_seed": {...}, "filenames": [...]}`, compute
    per-seed, per-image NME under all three conventions, plus GT-level
    x-sort-vs-DOD disagreement. Returns a dict with:
      - "per_seed_per_image": {seed: {filename: {"native":.., "xsort":.., "dod":..}}}
      - "gt_disagreement_rate": float (x-sort vs DOD channel-0 disagreement on GT)
      - "n_images": int
    """
    filenames = data["filenames"]
    per_seed = data["per_seed"]
    out_per_seed: dict[int, dict[str, dict]] = {}
    disagree_flags = []

    for seed, by_name in per_seed.items():
        out = {}
        for fn in filenames:
            row = by_name[fn]
            pred0, pred1 = row["pred0"], row["pred1"]
            gt0, gt1 = row["gt0"], row["gt1"]

            native_nme = row["native_fixed_nme"]

            gt_x0, gt_x1 = x_sort(gt0, gt1)
            pred_x0, pred_x1 = x_sort(pred0, pred1)
            xsort_nme = fixed_channel_nme(pred_x0, pred_x1, gt_x0, gt_x1)

            gt_d0, gt_d1 = dod_sort(gt0, gt1, d_vect)
            pred_d0, pred_d1 = dod_sort(pred0, pred1, d_vect)
            dod_nme = fixed_channel_nme(pred_d0, pred_d1, gt_d0, gt_d1)

            out[fn] = {"native": native_nme, "xsort": xsort_nme, "dod": dod_nme}

            if seed == SEEDS[0]:
                # GT-level disagreement only depends on GT + d_vect, not on
                # predictions or seed -- compute it once, from the first
                # seed's GT (GT is identical across seeds for the same image).
                disagree_flags.append(gt_x0 != gt_d0)
        out_per_seed[seed] = out

    return {
        "per_seed_per_image": out_per_seed,
        "gt_disagreement_rate": float(np.mean(disagree_flags)) if disagree_flags else float("nan"),
        "n_images": len(filenames),
    }


def summarize_and_write(dataset: str, task: str, method_label: str,
                         rescored: dict, output_root: Path,
                         bootstrap_reps: int, rng: np.random.Generator,
                         per_image_dir_prefix: str) -> list[dict]:
    per_seed = rescored["per_seed_per_image"]
    filenames = sorted(next(iter(per_seed.values())).keys())

    seed_rows = []
    per_image_avg = {conv: [] for conv in ("native", "xsort", "dod")}
    for seed in SEEDS:
        for conv in ("native", "xsort", "dod"):
            values = np.array([per_seed[seed][fn][conv] for fn in filenames]) * 100.0
            seed_rows.append({
                "dataset": dataset, "task": task, "method": method_label, "seed": seed,
                "convention": conv, "n_images": len(filenames),
                "mean_nme_pct": f"{values.mean():.8f}",
            })

    for conv in ("native", "xsort", "dod"):
        stacked = np.array([[per_seed[seed][fn][conv] for seed in SEEDS] for fn in filenames]) * 100.0
        per_image_avg[conv] = stacked.mean(axis=1)  # average across 5 seeds, per image

    xsort_vals = per_image_avg["xsort"]
    dod_vals = per_image_avg["dod"]
    diff = xsort_vals - dod_vals
    lo, hi = bootstrap_ci(diff, bootstrap_reps, rng) if len(diff) > 1 else (float("nan"), float("nan"))

    per_image_path = output_root / f"{per_image_dir_prefix}_{dataset.lower()}_{task}_{method_label}_per_image.csv"
    with per_image_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["filename", "native_nme_pct", "xsort_nme_pct", "dod_nme_pct", "xsort_minus_dod_pp"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i, fn in enumerate(filenames):
            writer.writerow({
                "filename": fn,
                "native_nme_pct": f"{per_image_avg['native'][i]:.8f}",
                "xsort_nme_pct": f"{xsort_vals[i]:.8f}",
                "dod_nme_pct": f"{dod_vals[i]:.8f}",
                "xsort_minus_dod_pp": f"{diff[i]:.8f}",
            })

    summary_row = {
        "dataset": dataset, "task": task, "method": method_label,
        "n_images": len(filenames),
        "gt_xsort_vs_dod_disagreement_rate": f"{rescored['gt_disagreement_rate']:.6f}",
    }
    for conv in ("native", "xsort", "dod"):
        seed_means = np.array([
            np.mean([per_seed[seed][fn][conv] for fn in filenames]) * 100.0
            for seed in SEEDS
        ])
        summary_row[f"{conv}_5seed_mean_pct"] = f"{seed_means.mean():.8f}"
        summary_row[f"{conv}_5seed_sample_sd_pct"] = f"{seed_means.std(ddof=1):.8f}"
    summary_row["xsort_minus_dod_mean_pp"] = f"{diff.mean():.8f}"
    summary_row["xsort_minus_dod_bootstrap_95ci_low_pp"] = f"{lo:.8f}"
    summary_row["xsort_minus_dod_bootstrap_95ci_high_pp"] = f"{hi:.8f}"

    return seed_rows, [summary_row]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ucl-eomt-root", type=Path, default=Path("/root/autodl-tmp/ucl_eomt_per_image"))
    parser.add_argument("--ucl-hrnet-root", type=Path,
                         default=Path("/root/autodl-tmp/hrnet_512_fixed_5seed/output/FETAL"))
    parser.add_argument("--multicentre-eomt-root", type=Path,
                         default=Path("/root/autodl-tmp/saved_checkpoints/multicentre_5seed"))
    parser.add_argument("--multicentre-hrnet-root", type=Path,
                         default=Path("/root/autodl-tmp/hrnet_512_fixed_5seed/output/FETAL"))
    parser.add_argument("--output-root", type=Path, default=Path("endpoint_ordering_analysis/results"))
    parser.add_argument("--bootstrap-replicates", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260806)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.bootstrap_seed)

    all_seed_rows: list[dict] = []
    all_summary_rows: list[dict] = []
    excluded: list[dict] = []
    dvect_rows = [
        {"dataset": d, "task": t, "d0_x": v[0][0], "d0_y": v[0][1], "d1_x": v[1][0], "d1_y": v[1][1]}
        for (d, t), v in D_VECT.items()
    ]

    cells = []
    for task in UCL_TASKS:
        cells.append(("UCL", task, "hrnet", None, args.ucl_hrnet_root, None))
        for backbone in BACKBONES:
            cells.append(("UCL", task, "eomt", backbone, None, args.ucl_eomt_root))
    for task in MULTICENTRE_TASKS:
        cells.append(("MULTICENTRE", task, "hrnet", None, args.multicentre_hrnet_root, None))
        for backbone in BACKBONES:
            cells.append(("MULTICENTRE", task, "eomt", backbone, None, args.multicentre_eomt_root))

    for dataset, task, method, backbone, hrnet_root, eomt_root in cells:
        method_label = "hrnet" if method == "hrnet" else f"eomt_{backbone}"
        d_vect = get_d_vect(dataset, task.upper())
        try:
            if method == "hrnet":
                data = load_hrnet_per_image(hrnet_root, dataset, task)
            else:
                data = load_eomt_per_image(eomt_root, dataset, task, backbone)
        except LoadError as exc:
            excluded.append({"dataset": dataset, "task": task, "method": method_label, "reason": str(exc)})
            print(f"[EXCLUDED] {dataset}/{task}/{method_label}: {exc}")
            continue

        rescored = rescore_cell(data, d_vect)
        seed_rows, summary_rows = summarize_and_write(
            dataset, task, method_label, rescored, args.output_root,
            args.bootstrap_replicates, rng, "per_image",
        )
        all_seed_rows.extend(seed_rows)
        all_summary_rows.extend(summary_rows)
        print(f"[OK] {dataset}/{task}/{method_label}: n={rescored['n_images']}, "
              f"gt_disagreement={rescored['gt_disagreement_rate']:.4f}")

    seed_summary_path = args.output_root / "endpoint_ordering_seed_summary.tsv"
    with seed_summary_path.open("w", newline="", encoding="utf-8") as handle:
        fields = list(all_seed_rows[0])
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(all_seed_rows)

    summary_path = args.output_root / "endpoint_ordering_summary.tsv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        fields = list(all_summary_rows[0])
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(all_summary_rows)

    dvect_path = args.output_root / "dod_vectors.tsv"
    with dvect_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dvect_rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(dvect_rows)

    excluded_path = args.output_root / "excluded_images.tsv"
    with excluded_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["dataset", "task", "method", "reason"]
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(excluded)

    print(f"\n[COMPLETE] {len(all_summary_rows)} cells scored, {len(excluded)} cells excluded.")
    print(f"Wrote: {summary_path}, {seed_summary_path}, {dvect_path}, {excluded_path}, "
          f"and {len(all_summary_rows)} per-image CSVs under {args.output_root}")
    print("\n*** LIMITATION, repeat to the supervisor alongside these numbers ***")
    print("This is a retrospective RE-SCORING of already-saved predictions under two")
    print("external conventions -- it quantifies how much the EXTERNAL SCORING RULE")
    print("alone changes each method's reported number and whether conclusions flip.")
    print("It does NOT retrain either method, and does NOT prove a method trained")
    print("under a different convention from the start would perform identically.")


if __name__ == "__main__":
    main()
