"""Generate the Chapter 1 fetal-biometry task illustration from UCL GT.

The selected images are public release samples. Coordinates are read directly
from the released UCL Test CSVs; no landmark positions are drawn by hand.
Run with ``--contact-sheet`` to review high-quality candidates, or without it
to create the final three-panel thesis figure using SELECTED_IMAGES.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


MEASUREMENTS = {
    "Head": (
        ("BPD", "bpd_1_x", "bpd_1_y", "bpd_2_x", "bpd_2_y", "#ff3b30"),
        ("OFD", "ofd_1_x", "ofd_1_y", "ofd_2_x", "ofd_2_y", "#00d8ff"),
    ),
    "Abdomen": (
        ("TAD", "tad_1_x", "tad_1_y", "tad_2_x", "tad_2_y", "#ff3b30"),
        ("APAD", "apad_1_x", "apad_1_y", "apad_2_x", "apad_2_y", "#00d8ff"),
    ),
    "Femur": (
        ("FL", "fl_1_x", "fl_1_y", "fl_2_x", "fl_2_y", "#ff3b30"),
    ),
}

# Fixed only after visual review of the automatically annotated contact sheet.
SELECTED_IMAGES = {
    "Head": "007_13HC.jpeg",
    "Abdomen": "006_3AC.jpeg",
    "Femur": "005_11FL.jpeg",
}

LABEL_OFFSETS = {
    "BPD": (8, -15),
    "OFD": (8, 6),
    "TAD": (9, -16),
    "APAD": (9, 7),
    "FL": (8, -12),
}

LABEL_POSITIONS = {
    "BPD": 0.65,
    "OFD": 0.65,
    "TAD": 0.38,
    "APAD": 0.70,
    "FL": 0.65,
}


def read_rows(data_root: Path, anatomy: str) -> list[dict[str, str]]:
    csv_path = data_root / "annotations" / "UCL" / f"{anatomy}_Test.csv"
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def coords(row: dict[str, str], spec: tuple[str, ...]) -> tuple[float, ...]:
    return tuple(float(row[key]) for key in spec[1:5])


def candidate_score(row: dict[str, str], anatomy: str, size: tuple[int, int]) -> float:
    width, height = size
    diag = math.hypot(width, height)
    score = 0.0
    for spec in MEASUREMENTS[anatomy]:
        try:
            x1, y1, x2, y2 = coords(row, spec)
        except (KeyError, TypeError, ValueError):
            return -1.0
        margin = min(x1, x2, width - x1, width - x2, y1, y2, height - y1, height - y2)
        length = math.hypot(x2 - x1, y2 - y1)
        score += min(margin / max(min(width, height), 1), 0.25) + length / diag
    return score


def annotate(ax, image: Image.Image, row: dict[str, str], anatomy: str, title: str) -> None:
    ax.imshow(image, cmap="gray")
    for label, x1k, y1k, x2k, y2k, colour in MEASUREMENTS[anatomy]:
        x1, y1, x2, y2 = (float(row[k]) for k in (x1k, y1k, x2k, y2k))
        ax.plot([x1, x2], [y1, y2], color=colour, linewidth=2.2, solid_capstyle="round")
        ax.scatter([x1, x2], [y1, y2], s=38, c=colour, edgecolors="white", linewidths=0.8, zorder=3)
        # Place labels beyond the exact intersection of paired diameters so
        # BPD/OFD and TAD/APAD remain separately readable at thesis scale.
        fraction = LABEL_POSITIONS[label]
        mx, my = (1.0 - fraction) * x1 + fraction * x2, (1.0 - fraction) * y1 + fraction * y2
        ax.annotate(
            label,
            (mx, my),
            xytext=LABEL_OFFSETS[label],
            textcoords="offset points",
            color="white",
            fontsize=11,
            weight="bold",
            bbox={"boxstyle": "round,pad=0.18", "facecolor": colour, "edgecolor": "white", "alpha": 0.88},
        )
    ax.set_title(title, fontsize=11, weight="bold", pad=5)
    ax.set_axis_off()


def find_row(rows: list[dict[str, str]], image_name: str) -> dict[str, str]:
    for row in rows:
        if row.get("image_name") == image_name:
            return row
    raise KeyError(f"{image_name!r} is absent from the released CSV")


def make_contact_sheet(data_root: Path, output: Path) -> None:
    fig, axes = plt.subplots(3, 6, figsize=(15, 8.2), constrained_layout=True)
    for row_idx, anatomy in enumerate(MEASUREMENTS):
        rows = read_rows(data_root, anatomy)
        image_dir = data_root / "images" / "UCL" / anatomy
        ranked = []
        for row in rows:
            path = image_dir / row["image_name"]
            if not path.is_file():
                continue
            with Image.open(path) as image:
                score = candidate_score(row, anatomy, image.size)
            if score >= 0:
                ranked.append((score, row, path))
        for ax, (_, row, path) in zip(axes[row_idx], sorted(ranked, reverse=True, key=lambda item: item[0])[:6]):
            with Image.open(path) as image:
                annotate(ax, image.convert("L"), row, anatomy, f"{anatomy}: {row['image_name']}")
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_final(data_root: Path, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.15), constrained_layout=True)
    panel_letters = ("a", "b", "c")
    for ax, anatomy, letter in zip(axes, MEASUREMENTS, panel_letters):
        rows = read_rows(data_root, anatomy)
        image_name = SELECTED_IMAGES[anatomy]
        row = find_row(rows, image_name)
        path = data_root / "images" / "UCL" / anatomy / image_name
        with Image.open(path) as image:
            annotate(ax, image.convert("L"), row, anatomy, f"({letter}) {anatomy} plane")
    fig.savefig(output, dpi=300, bbox_inches="tight", pad_inches=0.03, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contact-sheet", action="store_true")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.contact_sheet:
        make_contact_sheet(args.data_root, args.output)
    else:
        make_final(args.data_root, args.output)


if __name__ == "__main__":
    main()
