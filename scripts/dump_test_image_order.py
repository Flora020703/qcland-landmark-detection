#!/usr/bin/env python3
# ---------------------------------------------------------------
# MODIFIED: new file — dump the (index -> filename) mapping for a config's
# test split, in the exact order test_dataloader() iterates (shuffle=False,
# so this order is deterministic and stable across runs of the same config).
#
# training/landmark_detection.py's test_nme_dump_path records metrics and
# coordinates keyed by integer iteration index, but not the image name.
# This script provides the index-to-image mapping so those rows can be
# joined back to actual images for outlier inspection and paired analysis.
#
# Usage:
#   python3 scripts/dump_test_image_order.py --config configs/landmark/bpd_resolution_screening.yaml \
#       --out checkpoints/bpd-resolution-screening/res512/test_image_order.csv
# ---------------------------------------------------------------

import argparse
import csv
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml


def build_datamodule(cfg: dict):
    class_path = cfg["data"]["class_path"]
    module_path, class_name = class_path.rsplit(".", 1)
    dm_class = getattr(importlib.import_module(module_path), class_name)
    return dm_class(**cfg["data"]["init_args"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    dm = build_datamodule(cfg)
    dm.setup()

    records = dm.test_dataset.records
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "filename"])
        for i, rec in enumerate(records):
            writer.writerow([i, Path(rec["img_path"]).name])

    print(f"[OK] wrote {len(records)} (index,filename) rows to {out_path}")


if __name__ == "__main__":
    main()
