# ---------------------------------------------------------------
# MODIFIED: new file — pooled multi-centre fetal biometry data (the
# Multicentre release's FP + HC18 + UCL union, verified 2026-07-24 by
# exact row-count arithmetic: Head 757+737+110=1604 train / 880+262+49=1191
# test; Abdomen 568+94=662 train / 125+36=161 test (no HC18); Femur
# 437+96=533 train / 323+39=362 test (no HC18)). CSV schema (image_name,
# scale, center_w, center_h, <task>_i_x/y, px_to_mm_rate, ...) is byte-
# identical to the existing UCL-only convention already handled by
# HeadLandmarkDataModule/LandmarkDataset/TASK_COLS in landmark_dataset.py,
# and images/MULTICENTRE/<Anatomy>/ is a flat pool matching every row's
# image_name directly (no per-source subdirectory) -- so this file reuses
# that entire pipeline and only overrides subject-id extraction.
#
# WHY subject-id extraction needs to change for pooled data: the parent's
# _subject_id() takes the bare leading numeric filename prefix (e.g.
# "022_3HC.jpeg" -> "022"). That's safe for a single-source CSV, but three
# problems appear once FP/HC18/UCL are pooled together:
#   1. FP filenames don't start with a digit at all
#      ("Patient00683_Plane3_1_of_2.png") -- the bare-prefix regex fails to
#      match and falls back to the full filename, so FP patients with
#      multiple frames (multiple Planes) never get grouped together for
#      the internal validation split.
#   2. HC18 and UCL share the exact same "[ID]_HC.*"/"[ID]_[N]HC.*"
#      filename convention with NO source column in the pooled CSV to
#      tell them apart, so a bare numeric prefix can silently merge an
#      HC18 patient and an unrelated UCL patient into one validation-split
#      group. This is not hypothetical: images/MULTICENTRE/Head/ actually
#      contains both "001_HC.jpg" and "001_HC.png" side by side (verified
#      2026-07-24), consistent with two different real patients from two
#      different sources happening to share the numeric ID "001".
#   3. Even where prefixes don't collide, mixing three sources under one
#      undifferentiated numeric ID space is fragile going forward as more
#      data is added.
#
# Fix: prefix the subject key with an inferred source tag.
#   - FP is reliably distinguishable via its own "Patient<digits>_Plane"
#     pattern (per-anatomy verification, 2026-07-24) -- no lookup needed.
#   - HC18 vs UCL (only relevant for Head/bpd/ofd -- HC18 contributes no
#     Abdomen/Femur data at all) is disambiguated by cross-referencing the
#     filename against HC18's own separate annotation CSVs (passed in via
#     hc18_ann_csvs). Anything not found there defaults to UCL, which is
#     automatically correct for Abdomen/Femur (hc18_ann_csvs simply isn't
#     passed for those tasks, so the lookup set stays empty).
# ---------------------------------------------------------------

import csv
import re
from pathlib import Path
from typing import Optional, Union

from datasets.landmark_dataset import HeadLandmarkDataModule

# Matches e.g. "Patient00683_Plane3_1_of_2.png" -> group(1) = "00683".
_FP_PATTERN = re.compile(r"^Patient(\d+)_Plane")


class MulticentreLandmarkDataModule(HeadLandmarkDataModule):
    """
    Pooled multi-centre counterpart of HeadLandmarkDataModule. Same
    constructor signature plus one addition:

        hc18_ann_csvs: paths to HC18's own annotation CSV(s) (e.g. both
            its Train and Test files) for this anatomy, used only to
            disambiguate HC18 vs UCL filenames that would otherwise share
            an ambiguous numeric prefix. Omit (default None/empty) for
            Abdomen/Femur, where HC18 contributes no data and every
            non-FP filename is therefore unambiguously UCL.

    Everything else (CSV column layout via TASK_COLS, heatmap generation,
    augmentation, train/val/test split machinery, dataloaders) is
    inherited unchanged from HeadLandmarkDataModule -- only
    _subject_id() is overridden.
    """

    def __init__(
        self,
        *args,
        hc18_ann_csvs: Optional[list[Union[str, Path]]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._hc18_filenames: set[str] = set()
        for p in (hc18_ann_csvs or []):
            with open(p, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    self._hc18_filenames.add(row["image_name"])

    def _subject_id(self, img_name: str) -> str:
        """
        MODIFIED: overrides the parent's @staticmethod bare-numeric-prefix
        extraction with a source-tagged version -- see module docstring
        for why this is necessary once FP/HC18/UCL are pooled together.
        """
        fp_match = _FP_PATTERN.match(img_name)
        if fp_match:
            return f"FP_{fp_match.group(1)}"

        prefix_match = re.match(r"^(\d+)", img_name)
        prefix = prefix_match.group(1) if prefix_match else img_name
        source = "HC18" if img_name in self._hc18_filenames else "UCL"
        return f"{source}_{prefix}"
