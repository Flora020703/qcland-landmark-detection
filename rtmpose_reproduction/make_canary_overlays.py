#!/usr/bin/env python3
"""Render deterministic RTMPose canary prediction/GT overlays.

The input CSV is produced by ``evaluate_rtmpose_fixed.py`` and already uses
original-image coordinates.  This tool is audit-only: it never changes or
re-scores predictions.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"empty per-image CSV: {path}")
    return rows


def _point(row: dict[str, str], prefix: str) -> tuple[float, float]:
    return float(row[f"{prefix}_x"]), float(row[f"{prefix}_y"])


def _selected(rows: list[dict[str, str]], edge_count: int, median_count: int):
    ranked = sorted(rows, key=lambda r: float(r["swap_min_nme"]))
    mid = len(ranked) // 2
    half = median_count // 2
    candidates = (
        [("best", row) for row in ranked[:edge_count]]
        + [("median", row) for row in ranked[mid - half: mid - half + median_count]]
        + [("worst", row) for row in ranked[-edge_count:][::-1]]
    )
    seen: set[str] = set()
    for group, row in candidates:
        name = row["filename"]
        if name not in seen:
            seen.add(name)
            yield group, row


def _render(image_path: Path, row: dict[str, str], output_path: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    gt0, gt1 = _point(row, "gt0"), _point(row, "gt1")
    pred0, pred1 = _point(row, "pred0"), _point(row, "pred1")
    radius = max(4, round(min(image.size) / 120))
    width = max(2, round(radius / 2))

    # Ground truth: green; prediction: red.  Connecting each pair makes a
    # global offset, wrong scale, or anatomically implausible diameter clear.
    draw.line([gt0, gt1], fill=(0, 255, 0), width=width)
    draw.line([pred0, pred1], fill=(255, 48, 48), width=width)
    for point, colour in ((gt0, (0, 255, 0)), (gt1, (0, 255, 0)),
                          (pred0, (255, 48, 48)), (pred1, (255, 48, 48))):
        x, y = point
        draw.ellipse((x-radius, y-radius, x+radius, y+radius),
                     outline=colour, width=width)

    nme = 100.0 * float(row["swap_min_nme"])
    label = f"green=GT red=prediction  PI-NME={nme:.2f}%"
    bbox = draw.textbbox((0, 0), label, font=font)
    pad = 4
    draw.rectangle((0, 0, bbox[2] + 2*pad, bbox[3] + 2*pad), fill=(0, 0, 0))
    draw.text((pad, pad), label, fill=(255, 255, 255), font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-image-csv", required=True, type=Path)
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--edge-count", type=int, default=10)
    parser.add_argument("--median-count", type=int, default=3)
    args = parser.parse_args()

    rows = _read_rows(args.per_image_csv)
    manifest = []
    for rank, (group, row) in enumerate(
            _selected(rows, args.edge_count, args.median_count), start=1):
        source = args.images_dir / row["filename"]
        if not source.is_file():
            raise SystemExit(f"missing source image: {source}")
        nme_pct = 100.0 * float(row["swap_min_nme"])
        target = args.output_dir / group / f"{rank:02d}_{nme_pct:07.3f}_{source.stem}.png"
        _render(source, row, target)
        manifest.append((group, row["filename"], nme_pct, str(target)))

    manifest_path = args.output_dir / "overlay_manifest.tsv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["group", "filename", "permutation_invariant_nme_pct", "overlay"])
        writer.writerows(manifest)
    print(f"[OK] wrote {len(manifest)} overlays and {manifest_path}")


if __name__ == "__main__":
    main()
