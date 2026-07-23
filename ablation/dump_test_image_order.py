#!/usr/bin/env python3
# ---------------------------------------------------------------
# MODIFIED: new file — dump the test-set image order for a config, so
# per-sample NME files (written by training/landmark_detection.py's
# test_nme_dump_path, indexed 0..n-1 in test_dataloader iteration order)
# can be joined back to actual image filenames / subject prefixes.
#
# Written once per config (not per checkpoint/seed) — every checkpoint
# trained from the same config shares the same deterministic test set
# order (test_dataloader is always shuffle=False, drop_last=False), so
# this file is reusable across all 5 seeds x {best, final} for that config.
#
# Usage:
#   python3 ablation/dump_test_image_order.py --config configs/landmark/bpd_dinov2_fpn_udp_rotate_scale.yaml \
#       --out checkpoints/bpd-dinov2-fpn-udp-rotate-scale/test_image_order.csv
# ---------------------------------------------------------------

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from ablation.ensemble_test import build_datamodule


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
    with open(args.out, "w") as f:
        f.write("index,img_name,filename_prefix\n")
        for i, rec in enumerate(records):
            name = rec.get("img_name") or rec.get("image_name") or "?"
            m = re.match(r"^(\d+)", name)
            prefix = m.group(1) if m else ""
            f.write(f"{i},{name},{prefix}\n")

    print(f"[OK] wrote {len(records)} rows to {args.out}")


if __name__ == "__main__":
    main()
