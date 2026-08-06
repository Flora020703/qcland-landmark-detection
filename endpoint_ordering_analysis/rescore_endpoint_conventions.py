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
predicted or ground-truth COORDINATE (beyond the coordinate-SPACE
conversion described below, which recovers the same physical location a
different way, not a different location). It only re-derives, from
already-saved per-image predicted/GT coordinate pairs, which of the two
points in each pair is labelled "channel 0" under three different rules --
see the three-rule list further down.

*** CRITICAL FIX (2026-08-06, review finding, corrects the FIRST version of
this script) ***: EoMT's own per-image dump
(training/landmark_detection.py's test_nme_dump_path) writes coordinates
as `pred_coords * [img_size_w / heatmap_w, img_size_h / heatmap_h]` -- i.e.
in EoMT's SQUARE 512x512 MODEL-INPUT space, not the original (generally
non-square) image's own pixel space, despite that code's own comment
calling it "image-pixel space." HRNet's per-image CSV, by contrast, really
is in original image pixel space. The first version of this script fed
EoMT's raw 512-space coordinates directly into `dod_sort()` together with
`d_vect` (frozen in ORIGINAL image space) -- a coordinate-frame mismatch.

Why this specifically breaks DOD but not (the ordering decision of) x-sort:
x-sort only compares the two points' x-coordinates, and EoMT's own
original-image -> 512-model-input resize scales x by a single positive
per-image constant (512/original_width) applied identically to both
points -- a positive scalar multiple preserves ordering, so x-sort's
CHANNEL DECISION is accidentally invariant to this bug. The resulting NME
MAGNITUDE is NOT invariant, though: Euclidean distance mixes x and y, and
EoMT's original resize is generally ANISOTROPIC (different x/y scale
factors when original width != height, which is essentially always true
for these ultrasound images) -- so an NME computed on 512-space
coordinates does not equal the NME that would be computed on the same
points in real original-image space. DOD is broken on BOTH counts: the
projection direction mixes x and y, so the CHANNEL DECISION itself can
differ under anisotropic distortion, in addition to the same NME-magnitude
distortion x-sort also has.

Fixed: EoMT's dumped coordinates are now explicitly inverted back to real
original-image pixel space before ANY unified-convention canonicalisation
or NME computation, using the EXACT inverse of EoMT's own resize formula
(rtmpose_reproduction/geometry.to_image_space -- the identical UDP-inspired
pixel-centre convention EoMT's own `pixel_center_align=True` code path
uses, already unit-tested in that adapter's own test_geometry.py). This
requires each EoMT image's REAL original width/height, obtained by opening
the actual image file (matching the pattern
rtmpose_reproduction/convert_csv_to_coco.py already uses) -- see
`--ucl-images-root`/`--multicentre-images-root` below.

*** SECOND CRITICAL FIX (2026-08-07, review finding against the FIRST fix
above) ***: the "512-model-input space" description above is itself only
approximately true. `training/landmark_detection.py`'s dump does not invert
its own heatmap encoding exactly -- it converts heatmap-space coordinates
to "img_size space" via a plain multiply
(`coord_scale = img_size / heatmap_size`), but
`datasets/landmark_dataset.py` encodes model-input-space coordinates INTO
heatmap space using the PIXEL-CENTRE-ALIGNED formula (confirmed via
`grep` that every matched-protocol landmark config this analysis targets --
every `{bpd,ofd,apad,tad,fl}_{dinov2,dinov3}_fpn_udp_rotate_scale.yaml` plus
`multicentre_fpn_udp_rotate_scale.yaml` -- sets `pixel_center_align: true`
and `heatmap_size: [64, 64]` with `img_size: [512, 512]`):
`heatmap = (input + 0.5) * (heatmap_size/input_size) - 0.5`. Composing the
encode with the dump's naive (non-pixel-centre) inverse leaves a residual,
purely additive offset that is IDENTICAL for every point in every image
(a function only of `input_size`/`heatmap_size`, not of image content):
`dumped = input - 0.5 * (input_size/heatmap_size - 1) = input - 3.5` for
512/64. This offset cancels out of any within-image pairwise quantity
(distances, NME, x-sort/DOD ordering decisions all unaffected -- it is a
shared translation applied identically to pred0/pred1/gt0/gt1), so it did
NOT corrupt anything the first fix's own verification checked. It DOES
shift the recovered ABSOLUTE original-space coordinate, which matters for
the cross-method GT consistency check below (comparing an EoMT-recovered
GT location against HRNet's own GT location for the same filename) --
without this correction that check would report a spurious ~3.5-pixel
(in model-input-space, scaled further by the original/512 resize factor)
mismatch on every single image, even though nothing was actually wrong
with the ordering-convention numbers themselves. See
`_heatmap_dump_to_model_input_space()` below for the exact recovery.

*** THIRD FIX, same round ***: the cross-method GT consistency check
initially compared `gt0<->gt0`, `gt1<->gt1` directly across methods.
This is wrong on its own terms: HRNet's own native gt0/gt1 channel order
follows the DOD convention (its training-time channel assignment), EoMT's
follows x-sort -- for the SAME two physical points, the two methods may
legitimately (and often will) disagree on which one is called "channel 0"
whenever x-sort and DOD disagree for that image, which is exactly the
population this whole analysis is studying. Comparing channel-index to
channel-index directly conflates this EXPECTED convention difference with
an actual coordinate-recovery bug. Fixed to compare the two POSSIBLE
pairings (standard and swapped) and take whichever is closer -- see
`_min_paired_max_abs_diff()` below.

The three canonicalisation rules, everything now genuinely in ORIGINAL
image pixel coordinates for BOTH methods:
  1. NATIVE: recomputed directly from each file's OWN raw dumped
     coordinates in whatever space that file actually stores (EoMT: 512
     model-input space, using its own per-sample channel order as-is;
     HRNet: already original space). This is a SANITY CHECK ONLY -- it must
     reproduce the file's own stored NME value within floating-point
     tolerance, or this script's own coordinate parsing has a bug. It is
     NOT compared to the other method's "native" value (different spaces,
     not a fair comparison) and is not used for any cross-method claim.
  2. UNIFIED X-SORT: after converting to a COMMON original-image
     coordinate space, both GT and prediction independently re-labelled by
     ascending x (tie-break by y).
  3. UNIFIED DOD: after the same conversion, both GT and prediction
     independently re-labelled by projecting onto the SAME frozen,
     training-set-only direction vector (rtmpose_reproduction/dod_vectors.py).

Re-labelling changes which NUMBER gets called "fixed-channel NME" for a
given image; it does NOT retrain either method under a common convention.
A method whose TRAINING labels already used a different convention than
the one being tested here may still show a large "unified" NME even where
a differently-trained model would have done better under that convention
from the start -- this re-scoring answers "how much does the EXTERNAL
SCORING RULE alone matter," not "how well would each method perform if
retrained under a common rule." State this limitation explicitly to the
supervisor alongside the results (also printed at the end of every run).

Cross-checks this script performs and reports (do not trust any "unified"
number if these fail):
  - Native-reproduction sanity check: each file's own recomputed NME
    (using its own raw coordinates, no space conversion) vs its stored
    `nme`/`fixed_channel_nme` column, per method, per cell -- using a
    COMBINED absolute+relative tolerance (see NATIVE_SANITY_ATOL/RTOL),
    since a pure relative tolerance is unstable for near-zero NME values.
  - Cross-method GT consistency: for a given (dataset, task), HRNet's own
    GT and EoMT's (inverted-to-original-space) GT should describe the SAME
    physical annotation for the SAME filename -- checked by comparing
    coordinates under the CLOSER of the two possible channel pairings
    (`_min_paired_max_abs_diff`), not index-to-index, since HRNet's own
    native channel order follows DOD while EoMT's follows x-sort and the
    two conventions legitimately disagree on which physical point is
    "channel 0" for some images -- comparing index-to-index directly would
    conflate that expected convention difference with an actual
    coordinate-recovery bug.
  - Cross-method GT disagreement-rate consistency: the x-sort-vs-DOD
    disagreement rate on GT alone should be (near-)IDENTICAL whether
    computed from HRNet's GT or EoMT's (inverted) GT for the same
    (dataset, task) -- if it is not, that is a strong signal of a
    coordinate-recovery or filename-matching bug, not a real dataset
    property, and is reported explicitly rather than silently averaged over.

Usage (run where the real per-image files AND the real images actually
live -- server or a mounted copy; this script has no dependency on
rtmpose_reproduction's own MMPose-only code, only its pure-Python
dod_vectors.py/endpoint_order.py/geometry.py):

    python endpoint_ordering_analysis/rescore_endpoint_conventions.py \
        --ucl-eomt-root /root/autodl-tmp/ucl_eomt_per_image \
        --ucl-hrnet-root /root/autodl-tmp/hrnet_512_fixed_5seed/output/FETAL \
        --ucl-images-root /root/autodl-tmp/images/UCL \
        --multicentre-eomt-root /root/autodl-tmp/saved_checkpoints/multicentre_5seed \
        --multicentre-hrnet-root /root/autodl-tmp/hrnet_512_fixed_5seed/output/FETAL \
        --multicentre-images-root /root/autodl-tmp/images/MULTICENTRE \
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
from geometry import to_image_space  # noqa: E402

SEEDS = (42, 0, 123, 2024, 3407)
BACKBONES = ("dinov2", "dinov3")
UCL_TASKS = ("bpd", "ofd", "apad", "tad", "fl")
MULTICENTRE_TASKS = ("bpd", "ofd", "apad", "tad", "fl")
HRNET_TASK_TAG = {
    "bpd": "brain_BPD", "ofd": "brain_OFD",
    "apad": "abdomen_APAD", "tad": "abdomen_TAD", "fl": "femur_FL",
}
ANATOMY_BY_TASK = {"bpd": "Head", "ofd": "Head", "apad": "Abdomen", "tad": "Abdomen", "fl": "Femur"}
EOMT_MODEL_INPUT_SIZE = 512  # matches every matched-protocol EoMT config's img_size
EOMT_HEATMAP_SIZE = 64  # verified via grep: every matched-protocol landmark config
                        # (bpd/ofd/apad/tad/fl x dinov2/dinov3 x _fpn_udp_rotate_scale.yaml,
                        # plus multicentre_fpn_udp_rotate_scale.yaml) sets
                        # heatmap_size: [64, 64] and pixel_center_align: true
# Combined absolute+relative tolerance for the native-reproduction sanity
# check -- a pure relative tolerance is unstable for near-zero stored NME
# values, where the CSV's own 8-decimal text formatting alone can produce
# a large relative (but tiny absolute) discrepancy.
NATIVE_SANITY_RTOL = 1e-4
NATIVE_SANITY_ATOL = 5e-8


def _heatmap_dump_to_model_input_space(dumped_x: float, dumped_y: float) -> tuple[float, float]:
    """Recovers true 512x512 model-input-space coordinates from EoMT's raw
    per-image dump -- see this module's docstring ("SECOND CRITICAL FIX")
    for the full derivation. `training/landmark_detection.py` dumps
    heatmap-space coordinates via a naive scale multiply
    (`coord_scale = img_size/heatmap_size`), which is NOT the exact inverse
    of how `datasets/landmark_dataset.py` encoded them (pixel-centre-aligned:
    `heatmap = (input + 0.5) * (heatmap_size/input_size) - 0.5`). The
    residual is a constant offset, identical for every point in every
    image: `dumped = input - 0.5 * (input_size/heatmap_size - 1)`."""
    scale = EOMT_MODEL_INPUT_SIZE / EOMT_HEATMAP_SIZE
    offset = 0.5 * scale - 0.5
    return dumped_x + offset, dumped_y + offset


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
    (dataset, task[, backbone]) cell's real per-image files are missing,
    lack the raw-coordinate columns this analysis needs, or the actual
    image file needed to recover EoMT's original-space size cannot be
    found -- e.g. UCL BPD's EoMT checkpoints are known (from this
    project's own records) to no longer exist on the server. Reported in
    excluded_images.tsv / the run's own console output, not silently
    skipped without a trace."""


class _ImageSizeCache:
    """Opens each real image file at most once (per images_root), via PIL,
    matching rtmpose_reproduction/convert_csv_to_coco.py's own pattern.
    Raises LoadError with the exact missing path if an image can't be found
    -- EoMT's coordinate-space fix depends entirely on getting a REAL width/
    height per image, not an assumption."""

    def __init__(self, images_root: Path, anatomy: str):
        self.dir = images_root / anatomy
        self._cache: dict[str, tuple[float, float]] = {}

    def size(self, filename: str) -> tuple[float, float]:
        if filename not in self._cache:
            path = self.dir / filename
            if not path.is_file():
                raise LoadError(
                    f"cannot recover EoMT's original image size: {path} not found "
                    f"-- EoMT's dumped coordinates are in 512x512 model-input space "
                    f"and MUST be inverted using the real original width/height; "
                    f"without the actual image file this is impossible, not "
                    f"approximable. Check --*-images-root."
                )
            from PIL import Image
            with Image.open(path) as im:
                self._cache[filename] = (float(im.size[0]), float(im.size[1]))
        return self._cache[filename]


def load_hrnet_per_image(hrnet_root: Path, dataset: str, task: str) -> dict[str, dict]:
    """Returns {"per_seed": {seed: {filename: {...}}}, "filenames": [...]}.
    Real schema (verified against baseline_reproduction/evaluate_hrnet_fixed.py's
    own CSV writer): index,filename,pred0_x,pred0_y,pred1_x,pred1_y,gt0_x,gt0_y,
    gt1_x,gt1_y,reference_distance,fixed_channel_nme,swap_min_nme -- already in
    ORIGINAL image pixel space, no conversion needed."""
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

    _check_native_sanity("HRNet", dataset, task, per_seed)
    return {"per_seed": per_seed, "filenames": sorted(keys)}


def load_eomt_per_image(eomt_root: Path, dataset: str, task: str, backbone: str,
                         image_size_cache: _ImageSizeCache) -> dict:
    """Returns the same shape as load_hrnet_per_image, but with
    coordinates ALREADY CONVERTED to real original-image pixel space
    (see this module's own docstring for why this conversion is required
    and how it is done) -- callers never see EoMT's raw 512-space numbers.

    Real schema depends on training/landmark_detection.py's
    test_nme_dump_path feature (introduced 2026-07-23/24) actually having
    been enabled for the run that produced these files -- if the
    coordinate columns are absent, this raises LoadError with a precise,
    actionable message rather than silently falling back to NME-only."""
    is_multicentre = dataset == "MULTICENTRE"
    per_seed_raw: dict[int, dict[str, dict]] = {}
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
                # RAW, still in EoMT's 512x512 model-input space -- NOT
                # converted yet at this point in the function.
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
        per_seed_raw[seed] = by_name

    keys = set(per_seed_raw[SEEDS[0]])
    for seed in SEEDS[1:]:
        if set(per_seed_raw[seed]) != keys:
            raise LoadError(f"EoMT {dataset}/{task}/{backbone}: filename set differs across seeds")

    # Native-reproduction sanity check BEFORE any space conversion -- this
    # validates that this script's own parsing of the raw dumped 512-space
    # numbers reproduces the file's own stored `nme` value exactly (EoMT's
    # native convention is computed in ITS OWN space, with no re-sort
    # needed since predicted/GT channels are already aligned 1:1 as trained).
    _check_native_sanity(f"EoMT({backbone})", dataset, task, per_seed_raw)

    # *** THE FIX ***: convert every coordinate to real original-image
    # pixel space, using each image's REAL width/height (opened once per
    # filename, reused across all 5 seeds). Two steps, composed: (1) recover
    # true 512x512 model-input-space coordinates from the raw dump (undoes
    # the dump's own naive-vs-pixel-centre-encode mismatch, a constant
    # +3.5 offset for 512/64 -- see _heatmap_dump_to_model_input_space's
    # docstring and this module's "SECOND CRITICAL FIX" note); (2) invert
    # the original-image -> 512-model-input resize via to_image_space.
    filenames = sorted(keys)
    sizes = {fn: image_size_cache.size(fn) for fn in filenames}
    per_seed_converted: dict[int, dict[str, dict]] = {}
    for seed, by_name in per_seed_raw.items():
        converted = {}
        for fn in filenames:
            row = by_name[fn]
            width, height = sizes[fn]

            def _to_orig(point: tuple[float, float]) -> tuple[float, float]:
                model_space = _heatmap_dump_to_model_input_space(*point)
                return to_image_space(*model_space, width, height, EOMT_MODEL_INPUT_SIZE)

            converted[fn] = {
                "pred0": _to_orig(row["pred0"]),
                "pred1": _to_orig(row["pred1"]),
                "gt0": _to_orig(row["gt0"]),
                "gt1": _to_orig(row["gt1"]),
                "native_fixed_nme": row["native_fixed_nme"],
            }
        per_seed_converted[seed] = converted

    return {"per_seed": per_seed_converted, "filenames": filenames}


def _check_native_sanity(method_label: str, dataset: str, task: str, per_seed: dict) -> None:
    """Recomputes fixed_channel_nme directly from each row's OWN raw
    coordinates (no unified re-sort, no space conversion) and asserts it
    matches the file's own stored value -- validates this script's parsing
    is correct, independent of the separate original-space conversion
    (EoMT) or lack thereof (HRNet, already original space).

    Uses a COMBINED absolute+relative tolerance (`abs_err <= atol +
    rtol*|stored|`), not a pure relative tolerance: for images with a
    near-zero true NME, the stored value's own 8-decimal text formatting
    alone can produce a large relative error despite a tiny, meaningless
    absolute discrepancy -- a pure-relative check would wrongly exclude
    such cells."""
    worst_abs_err = 0.0
    worst_rel_err = 0.0
    all_within_tolerance = True
    n_checked = 0
    for seed, by_name in per_seed.items():
        for fn, row in by_name.items():
            recomputed = fixed_channel_nme(row["pred0"], row["pred1"], row["gt0"], row["gt1"])
            stored = row["native_fixed_nme"]
            abs_err = abs(recomputed - stored)
            rel_err = abs_err / max(abs(stored), 1e-12)
            worst_abs_err = max(worst_abs_err, abs_err)
            worst_rel_err = max(worst_rel_err, rel_err)
            if abs_err > NATIVE_SANITY_ATOL + NATIVE_SANITY_RTOL * abs(stored):
                all_within_tolerance = False
            n_checked += 1
    if not all_within_tolerance:
        raise LoadError(
            f"{method_label} {dataset}/{task}: native-reproduction sanity check FAILED "
            f"(worst absolute error {worst_abs_err:.3e}, worst relative error "
            f"{worst_rel_err:.6g}, across {n_checked} (seed, image) pairs -- exceeds "
            f"combined tolerance atol={NATIVE_SANITY_ATOL}, rtol={NATIVE_SANITY_RTOL}) -- "
            f"this script's own coordinate parsing does not reproduce the file's own "
            f"stored NME; do not trust any unified-convention number for this cell "
            f"until this is resolved."
        )
    print(f"  [native sanity OK] {method_label} {dataset}/{task}: "
          f"worst absolute error {worst_abs_err:.2e}, worst relative error "
          f"{worst_rel_err:.2e}, over {n_checked} (seed,image) pairs")


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
    """Given one method's `{"per_seed": {...}, "filenames": [...]}` -- ALL
    coordinates already in a common ORIGINAL-image-space representation --
    compute per-seed, per-image NME under the two unified conventions, plus
    GT-level x-sort-vs-DOD disagreement (and the per-filename disagreement
    flags themselves, so the caller can cross-check against another
    method's GT for the same dataset/task)."""
    filenames = data["filenames"]
    per_seed = data["per_seed"]
    out_per_seed: dict[int, dict[str, dict]] = {}
    disagree_by_filename: dict[str, bool] = {}

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
                disagree_by_filename[fn] = (gt_x0 != gt_d0)
        out_per_seed[seed] = out

    return {
        "per_seed_per_image": out_per_seed,
        "disagree_by_filename": disagree_by_filename,
        "gt_disagreement_rate": float(np.mean(list(disagree_by_filename.values())))
                                  if disagree_by_filename else float("nan"),
        "n_images": len(filenames),
        "gt_by_filename": {fn: (per_seed[SEEDS[0]][fn]["gt0"], per_seed[SEEDS[0]][fn]["gt1"])
                            for fn in filenames},
    }


def summarize_and_write(dataset: str, task: str, method_label: str,
                         rescored: dict, output_root: Path,
                         bootstrap_reps: int, rng: np.random.Generator) -> tuple[list, list]:
    per_seed = rescored["per_seed_per_image"]
    filenames = sorted(next(iter(per_seed.values())).keys())

    seed_rows = []
    for seed in SEEDS:
        for conv in ("native", "xsort", "dod"):
            values = np.array([per_seed[seed][fn][conv] for fn in filenames]) * 100.0
            seed_rows.append({
                "dataset": dataset, "task": task, "method": method_label, "seed": seed,
                "convention": conv, "n_images": len(filenames),
                "mean_nme_pct": f"{values.mean():.8f}",
            })

    per_image_avg = {}
    for conv in ("native", "xsort", "dod"):
        stacked = np.array([[per_seed[seed][fn][conv] for seed in SEEDS] for fn in filenames]) * 100.0
        per_image_avg[conv] = stacked.mean(axis=1)  # average across 5 seeds, per image

    xsort_vals = per_image_avg["xsort"]
    dod_vals = per_image_avg["dod"]
    diff = xsort_vals - dod_vals
    lo, hi = bootstrap_ci(diff, bootstrap_reps, rng) if len(diff) > 1 else (float("nan"), float("nan"))

    per_image_path = output_root / f"{dataset.lower()}_{task}_{method_label}_per_image.csv"
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


def _min_paired_max_abs_diff(bg0: tuple, bg1: tuple, og0: tuple, og1: tuple) -> float:
    """Compares two methods' GT pairs for the SAME two physical points
    WITHOUT assuming they use the same channel-order convention: HRNet's
    own native gt0/gt1 order follows DOD, EoMT's follows x-sort, so for any
    image where the two rules disagree, `base_gt0` and `other_gt0` may
    legitimately refer to DIFFERENT physical points even when both methods
    recovered the location correctly. Comparing channel-index to
    channel-index directly (as the first version of this check did) would
    conflate this EXPECTED convention difference with an actual
    coordinate-recovery bug. Tries both possible pairings and returns
    whichever is closer -- robust even at an exact x-sort tie, unlike
    re-canonicalising both sides with x_sort first (which could pick
    different channels between methods right at a tie under floating-point
    rounding)."""
    standard = max(abs(bg0[0] - og0[0]), abs(bg0[1] - og0[1]),
                   abs(bg1[0] - og1[0]), abs(bg1[1] - og1[1]))
    swapped = max(abs(bg0[0] - og1[0]), abs(bg0[1] - og1[1]),
                  abs(bg1[0] - og0[0]), abs(bg1[1] - og0[1]))
    return min(standard, swapped)


def cross_method_gt_consistency_check(dataset: str, task: str, rescored_by_method: dict) -> list[dict]:
    """Per this file's own module docstring: HRNet's GT and EoMT's
    (inverted-to-original-space) GT should describe the SAME physical
    annotation for the same filename, and the x-sort-vs-DOD disagreement
    rate on GT alone should therefore be (near-)identical across methods
    for the same (dataset, task). Returns a list of warning rows (empty if
    everything is consistent) -- printed AND written to
    excluded_images.tsv's sibling, not silently absorbed into an average."""
    warnings = []
    methods = list(rescored_by_method)
    if len(methods) < 2:
        return warnings
    base_method = methods[0]
    base = rescored_by_method[base_method]
    for other_method in methods[1:]:
        other = rescored_by_method[other_method]
        common = sorted(set(base["gt_by_filename"]) & set(other["gt_by_filename"]))
        if not common:
            continue
        max_gt_coord_diff = 0.0
        disagreement_mismatches = 0
        for fn in common:
            (bg0, bg1) = base["gt_by_filename"][fn]
            (og0, og1) = other["gt_by_filename"][fn]
            d = _min_paired_max_abs_diff(bg0, bg1, og0, og1)
            max_gt_coord_diff = max(max_gt_coord_diff, d)
            if base["disagree_by_filename"][fn] != other["disagree_by_filename"][fn]:
                disagreement_mismatches += 1

        base_rate = np.mean([base["disagree_by_filename"][fn] for fn in common])
        other_rate = np.mean([other["disagree_by_filename"][fn] for fn in common])
        print(f"  [cross-method GT check] {dataset}/{task}: {base_method} vs {other_method}, "
              f"n_common={len(common)}, max_gt_coord_diff_px={max_gt_coord_diff:.3f}, "
              f"disagreement_rate {base_method}={base_rate:.4f} vs {other_method}={other_rate:.4f}, "
              f"per-image mismatches={disagreement_mismatches}")
        if max_gt_coord_diff > 5.0 or disagreement_mismatches > 0:
            warnings.append({
                "dataset": dataset, "task": task,
                "method_a": base_method, "method_b": other_method,
                "n_common": len(common),
                "max_gt_coord_diff_px": f"{max_gt_coord_diff:.4f}",
                "disagreement_rate_a": f"{base_rate:.6f}",
                "disagreement_rate_b": f"{other_rate:.6f}",
                "per_image_disagreement_mismatches": disagreement_mismatches,
            })
    return warnings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ucl-eomt-root", type=Path, default=Path("/root/autodl-tmp/ucl_eomt_per_image"))
    parser.add_argument("--ucl-hrnet-root", type=Path,
                         default=Path("/root/autodl-tmp/hrnet_512_fixed_5seed/output/FETAL"))
    parser.add_argument("--ucl-images-root", type=Path, required=True,
                         help="directory containing UCL/{Head,Abdomen,Femur}/<filename> -- "
                              "REQUIRED, EoMT's coordinate conversion cannot proceed without "
                              "the real original image dimensions")
    parser.add_argument("--multicentre-eomt-root", type=Path,
                         default=Path("/root/autodl-tmp/saved_checkpoints/multicentre_5seed"))
    parser.add_argument("--multicentre-hrnet-root", type=Path,
                         default=Path("/root/autodl-tmp/hrnet_512_fixed_5seed/output/FETAL"))
    parser.add_argument("--multicentre-images-root", type=Path, required=True,
                         help="directory containing MULTICENTRE/{Head,Abdomen,Femur}/<filename>")
    parser.add_argument("--output-root", type=Path, default=Path("endpoint_ordering_analysis/results"))
    parser.add_argument("--bootstrap-replicates", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260806)
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.bootstrap_seed)

    all_seed_rows: list[dict] = []
    all_summary_rows: list[dict] = []
    excluded: list[dict] = []
    gt_consistency_warnings: list[dict] = []
    dvect_rows = [
        {"dataset": d, "task": t, "d0_x": v[0][0], "d0_y": v[0][1], "d1_x": v[1][0], "d1_y": v[1][1]}
        for (d, t), v in D_VECT.items()
    ]

    task_groups = [
        ("UCL", UCL_TASKS, args.ucl_hrnet_root, args.ucl_eomt_root, args.ucl_images_root),
        ("MULTICENTRE", MULTICENTRE_TASKS, args.multicentre_hrnet_root,
         args.multicentre_eomt_root, args.multicentre_images_root),
    ]

    for dataset, tasks, hrnet_root, eomt_root, images_root in task_groups:
        for task in tasks:
            anatomy = ANATOMY_BY_TASK[task]
            image_cache = _ImageSizeCache(images_root, anatomy)
            d_vect = get_d_vect(dataset, task.upper())
            rescored_by_method: dict[str, dict] = {}

            cell_specs = [("hrnet", None)] + [("eomt", b) for b in BACKBONES]
            for method, backbone in cell_specs:
                method_label = "hrnet" if method == "hrnet" else f"eomt_{backbone}"
                try:
                    if method == "hrnet":
                        data = load_hrnet_per_image(hrnet_root, dataset, task)
                    else:
                        data = load_eomt_per_image(eomt_root, dataset, task, backbone, image_cache)
                except LoadError as exc:
                    excluded.append({"dataset": dataset, "task": task, "method": method_label, "reason": str(exc)})
                    print(f"[EXCLUDED] {dataset}/{task}/{method_label}: {exc}")
                    continue

                rescored = rescore_cell(data, d_vect)
                rescored_by_method[method_label] = rescored
                seed_rows, summary_rows = summarize_and_write(
                    dataset, task, method_label, rescored, args.output_root,
                    args.bootstrap_replicates, rng,
                )
                all_seed_rows.extend(seed_rows)
                all_summary_rows.extend(summary_rows)
                print(f"[OK] {dataset}/{task}/{method_label}: n={rescored['n_images']}, "
                      f"gt_disagreement={rescored['gt_disagreement_rate']:.4f}")

            gt_consistency_warnings.extend(
                cross_method_gt_consistency_check(dataset, task, rescored_by_method)
            )

    n_task_cells = sum(len(tasks) for _, tasks, *_ in task_groups) * (1 + len(BACKBONES))
    if not all_summary_rows:
        raise SystemExit(
            f"ERROR: ZERO cells could be scored out of {n_task_cells} attempted -- every "
            f"cell was excluded (see excluded_images.tsv reasons above). This is not a "
            f"result to report to the supervisor; check --*-root/--*-images-root paths."
        )

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

    consistency_path = args.output_root / "cross_method_gt_consistency_warnings.tsv"
    with consistency_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["dataset", "task", "method_a", "method_b", "n_common",
                   "max_gt_coord_diff_px", "disagreement_rate_a", "disagreement_rate_b",
                   "per_image_disagreement_mismatches"]
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(gt_consistency_warnings)

    n_scored = len(all_summary_rows)
    n_excluded = len(excluded)
    print(f"\n[COMPLETE] {n_scored}/{n_task_cells} cells scored, {n_excluded}/{n_task_cells} excluded.")
    if gt_consistency_warnings:
        print(f"[WARNING] {len(gt_consistency_warnings)} cross-method GT consistency check(s) "
              f"FAILED -- see {consistency_path}. Do not treat the affected (dataset, task) "
              f"cells' 'unified' numbers as trustworthy until resolved.")
    for row in excluded:
        print(f"  excluded: {row['dataset']}/{row['task']}/{row['method']}")
    print(f"Wrote: {summary_path}, {seed_summary_path}, {dvect_path}, {excluded_path}, "
          f"{consistency_path}, and {n_scored} per-image CSVs under {args.output_root}")
    print("\n*** LIMITATION, repeat to the supervisor alongside these numbers ***")
    print("This is a retrospective RE-SCORING of already-saved predictions under two")
    print("external conventions -- it quantifies how much the EXTERNAL SCORING RULE")
    print("alone changes each method's reported number and whether conclusions flip.")
    print("It does NOT retrain either method, and does NOT prove a method trained")
    print("under a different convention from the start would perform identically.")
    print(f"\n{n_excluded} of {n_task_cells} cells could NOT be scored (see excluded_images.tsv) --")
    print("for those specific (dataset, task) combinations, this analysis provides NO evidence")
    print("about endpoint-ordering sensitivity; state this explicitly, do not extrapolate from")
    print("other tasks. UCL BPD/EoMT is expected to be among these (checkpoints already gone).")


if __name__ == "__main__":
    main()
