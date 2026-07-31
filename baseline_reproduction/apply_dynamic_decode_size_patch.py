#!/usr/bin/env python3
"""Apply the audited HRNet heatmap-coordinate decode fix idempotently.

The released implementation hard-codes a 64x64 heatmap in train, validation,
and inference coordinate decoding. That is correct for its released 256x256
input, but wrong when the native stride-4 HRNet output is 128x128 at 512x512.
Use the active experiment configuration in all three call sites instead.
"""

import argparse
from pathlib import Path


OLD = "decode_preds(score_map, meta['center'], meta['scale'], [64, 64])"
NEW = (
    "decode_preds(score_map, meta['center'], meta['scale'], "
    "list(config.MODEL.HEATMAP_SIZE))"
)
EXPECTED_CALL_SITES = 3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo")
    args = parser.parse_args()

    path = Path(args.repo).resolve() / "lib" / "core" / "function.py"
    text = path.read_text(encoding="utf-8")
    old_count = text.count(OLD)
    new_count = text.count(NEW)

    if old_count == EXPECTED_CALL_SITES and new_count == 0:
        path.write_text(text.replace(OLD, NEW), encoding="utf-8")
        print(f"[OK] applied dynamic decode-size patch to {path}")
        return

    if old_count == 0 and new_count == EXPECTED_CALL_SITES:
        print(f"[OK] dynamic decode-size patch already present in {path}")
        return

    raise SystemExit(
        f"refusing to patch {path}: expected either "
        f"{EXPECTED_CALL_SITES} old and 0 new call sites, or 0 old and "
        f"{EXPECTED_CALL_SITES} new call sites; found old={old_count}, "
        f"new={new_count}"
    )


if __name__ == "__main__":
    main()
