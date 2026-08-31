#!/usr/bin/env python3
"""Generate the BPD einsum-vs-DeconvHeadV2 qualitative heatmap comparison
figure (thesis Chapter 5, tab:bpd-progression's first two rungs).

This script does NOT bundle or hardcode any private absolute path or the
underlying ~550MB checkpoint archives. It takes local paths as arguments
and expects the caller to have already extracted the two relevant seed42
run directories from this project's own checkpoint backups
(bpd-loss-hybrid-einsum-5seed-pi-20260809.tar and
bpd-arch-deconv-v2-5seed-pi-20260809.tar), plus a local copy of the UCL
Head test images.

Reproduces exactly:
  - datasets/landmark_dataset.py's test-time preprocessing (PIL bilinear
    resize to 512x512, [0,1] tensor, no client-side normalisation).
  - models/eomt.py's EoMT + models/vit.py's ViT architecture. Available
    model parameters and the input size are read from each run's archived
    config; ViT's patch-size default is used only when that config omits it.
  - The exact case-selection rule described in Chapter 5: the 49 BPD test
    images are ranked by (DeconvHeadV2 - einsum) per-image PI-NME
    difference (from each run's own final_swapmin_per_image.csv), and the
    images at the 50th / 75th / 100th percentile ranks of that ordering
    are selected (median / upper-quartile / largest-positive-difference).
    This is a deterministic, outcome-stratified selection, not a random
    or representative sample -- see the Chapter 5 body text for the
    methodological caveat.

Outputs:
  - bpd_einsum_deconvheadv2_heatmaps.{png,pdf} (this figure)
  - bpd_heatmap_comparison_manifest.tsv (one row per selected case,
    recording every number quoted in the thesis caption/body text, so the
    figure can be audited without re-running inference)

Usage:
  python generate_bpd_heatmap_comparison.py \\
      --einsum-run-dir /path/to/extracted/bpd_loss_hybrid_einsum/seed42 \\
      --deconv-run-dir /path/to/extracted/bpd_arch_deconv_v2/seed42 \\
      --images-dir /path/to/UCL/Head

The two run directories are the seed42/ subfolders extracted from this
project's own checkpoint_backups/bpd-loss-hybrid-einsum-5seed-pi-20260809.tar
and checkpoint_backups/bpd-arch-deconv-v2-5seed-pi-20260809.tar; each must
contain seed42_final.ckpt, seed42_config.yaml, test_image_order.csv, and
final_swapmin_per_image.csv exactly as archived. --images-dir must point
at a local copy of the UCL Head image set (not included in this
repository; see the thesis's Data Ethics and Provenance section).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_per_image_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def select_cases(einsum_rows, deconv_rows, order_rows):
    n = len(order_rows)
    assert len(einsum_rows) == n == len(deconv_rows), (
        f"per-image CSV row counts disagree: einsum={len(einsum_rows)} "
        f"deconv={len(deconv_rows)} order={n}"
    )
    expected_indices = set(range(n))
    for label, rows in [
        ("einsum per-image CSV", einsum_rows),
        ("DeconvHeadV2 per-image CSV", deconv_rows),
        ("test-image order CSV", order_rows),
    ]:
        indices = [int(r["index"]) for r in rows]
        assert len(indices) == len(set(indices)), f"{label}: duplicate indices"
        assert set(indices) == expected_indices, (
            f"{label}: indices are not exactly 0..{n - 1}"
        )

    filenames = {int(r["index"]): r["filename"] for r in order_rows}
    for label, rows in [
        ("einsum per-image CSV", einsum_rows),
        ("DeconvHeadV2 per-image CSV", deconv_rows),
    ]:
        if rows and "filename" in rows[0]:
            row_filenames = {int(r["index"]): r["filename"] for r in rows}
            assert row_filenames == filenames, (
                f"{label}: filenames disagree with test-image order CSV"
            )
    e_nme = {int(r["index"]): float(r["nme"]) for r in einsum_rows}
    d_nme = {int(r["index"]): float(r["nme"]) for r in deconv_rows}

    diffs = sorted(
        ((i, d_nme[i] - e_nme[i], e_nme[i], d_nme[i], filenames[i]) for i in range(n)),
        key=lambda t: t[1],
    )

    def pick(pct):
        rank = round(pct * (n - 1))
        return rank, diffs[rank]

    cases = {}
    for name, pct in [("median", 0.50), ("p75", 0.75), ("worst", 1.00)]:
        rank, (idx, diff, e, d, fn) = pick(pct)
        cases[name] = dict(rank=rank, test_idx=idx, diff=diff, einsum_nme=e,
                            deconv_nme=d, filename=fn)
    return cases, n


def build_model(config_path: Path, heatmap_head_override: str | None = None):
    sys.path.insert(0, str(REPO_ROOT))
    from models.eomt import EoMT
    from models.vit import ViT

    cfg = yaml.safe_load(open(config_path))
    net_args = cfg["model"]["init_args"]["network"]["init_args"]
    enc_args = net_args["encoder"]["init_args"]

    data_args = cfg["data"]["init_args"]
    vit_allowed = set(inspect.signature(ViT.__init__).parameters) - {"self"}
    vit_args = {k: v for k, v in enc_args.items() if k in vit_allowed}
    vit_args["img_size"] = tuple(data_args["img_size"])
    encoder = ViT(**vit_args)

    eomt_allowed = set(inspect.signature(EoMT.__init__).parameters) - {
        "self", "encoder"
    }
    eomt_args = {k: v for k, v in net_args.items() if k in eomt_allowed}
    if "heatmap_size" in eomt_args:
        eomt_args["heatmap_size"] = tuple(eomt_args["heatmap_size"])
    if heatmap_head_override is not None:
        eomt_args["heatmap_head"] = heatmap_head_override
    net = EoMT(encoder=encoder, **eomt_args)
    return net


def load_checkpoint_into(net, ckpt_path: Path) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = {k[len("network."):]: v for k, v in ckpt["state_dict"].items()
          if k.startswith("network.")}
    missing, unexpected = net.load_state_dict(sd, strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    return {"epoch": ckpt.get("epoch"), "global_step": ckpt.get("global_step")}


def run_inference(net, img_path: Path) -> np.ndarray:
    img = Image.open(img_path).convert("RGB").resize((512, 512), Image.BILINEAR)
    x = TF.to_tensor(img).unsqueeze(0)
    net.eval()
    with torch.no_grad():
        mask_logits_per_layer, _, _ = net(x)
    return mask_logits_per_layer[-1][0].numpy()  # (2, H, W)


def heatmap_rgb(hm: np.ndarray) -> np.ndarray:
    t = torch.from_numpy(hm).unsqueeze(0).float()
    t = F.interpolate(t, size=(512, 512), mode="bilinear", align_corners=False)[0].numpy()
    t = np.clip(t, 0.0, 1.0)
    rgb = np.zeros((512, 512, 3), dtype=np.float32)
    rgb[..., 0] = t[0]
    rgb[..., 2] = t[1]
    return rgb


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--einsum-run-dir", required=True, type=Path)
    ap.add_argument("--deconv-run-dir", required=True, type=Path)
    ap.add_argument("--images-dir", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).parent)
    args = ap.parse_args()

    einsum_ckpt = args.einsum_run_dir / "seed42_final.ckpt"
    deconv_ckpt = args.deconv_run_dir / "seed42_final.ckpt"
    einsum_cfg = args.einsum_run_dir / "seed42_config.yaml"
    deconv_cfg = args.deconv_run_dir / "seed42_config.yaml"

    einsum_rows = load_per_image_csv(args.einsum_run_dir / "final_swapmin_per_image.csv")
    deconv_rows = load_per_image_csv(args.deconv_run_dir / "final_swapmin_per_image.csv")
    order_rows = load_per_image_csv(args.einsum_run_dir / "test_image_order.csv")
    order_rows_d = load_per_image_csv(args.deconv_run_dir / "test_image_order.csv")
    assert order_rows == order_rows_d, "einsum and DeconvHeadV2 test-image orderings differ"

    cases, n = select_cases(einsum_rows, deconv_rows, order_rows)
    print(f"n = {n} test images; selected cases:")
    for name, c in cases.items():
        print(f"  {name}: rank={c['rank']} idx={c['test_idx']} file={c['filename']} "
              f"einsum={c['einsum_nme']*100:.3f}% deconv={c['deconv_nme']*100:.3f}% "
              f"diff={c['diff']*100:+.3f}pp")

    print("Building models and loading checkpoints...")
    nets, ckpt_meta, ckpt_sha, config_sha = {}, {}, {}, {}
    for head, cfg_path, ckpt_path in [
        ("einsum", einsum_cfg, einsum_ckpt),
        ("deconv_v2", deconv_cfg, deconv_ckpt),
    ]:
        net = build_model(cfg_path)
        ckpt_meta[head] = load_checkpoint_into(net, ckpt_path)
        ckpt_sha[head] = sha256_of(ckpt_path)
        config_sha[head] = sha256_of(cfg_path)
        nets[head] = net
        print(f"  {head}: sha256={ckpt_sha[head]} epoch={ckpt_meta[head]['epoch']} "
              f"step={ckpt_meta[head]['global_step']}")

    heatmaps = {}
    manifest_rows = []
    for name, c in cases.items():
        img_path = args.images_dir / c["filename"]
        assert img_path.exists(), img_path
        for head in ["einsum", "deconv_v2"]:
            hm = run_inference(nets[head], img_path)
            heatmaps[f"{name}__{head}"] = hm

        re = einsum_rows[c["test_idx"]]
        rd = deconv_rows[c["test_idx"]]
        e_hm = heatmaps[f"{name}__einsum"]
        d_hm = heatmaps[f"{name}__deconv_v2"]
        e_peak = tuple(int(v) for v in np.unravel_index(np.argmax(e_hm[0]), e_hm[0].shape))
        d_peak0 = np.unravel_index(np.argmax(d_hm[0]), d_hm[0].shape)
        d_peak1 = np.unravel_index(np.argmax(d_hm[1]), d_hm[1].shape)
        same_pixel_collapse = d_peak0 == d_peak1

        manifest_rows.append(dict(
            case=name, test_idx=c["test_idx"], filename=c["filename"],
            selection_rank=c["rank"], selection_rule=(
                "ascending sort by (deconv_v2_nme - einsum_nme) per-image PI-NME; "
                "rank = round(percentile * (n-1)), n=49"
            ),
            einsum_ckpt_sha256=ckpt_sha["einsum"], deconv_ckpt_sha256=ckpt_sha["deconv_v2"],
            einsum_config=str(einsum_cfg.name), deconv_config=str(deconv_cfg.name),
            einsum_config_sha256=config_sha["einsum"],
            deconv_config_sha256=config_sha["deconv_v2"],
            einsum_per_image_csv_sha256=sha256_of(
                args.einsum_run_dir / "final_swapmin_per_image.csv"
            ),
            deconv_per_image_csv_sha256=sha256_of(
                args.deconv_run_dir / "final_swapmin_per_image.csv"
            ),
            test_image_order_csv_sha256=sha256_of(
                args.einsum_run_dir / "test_image_order.csv"
            ),
            source_image_sha256=sha256_of(img_path),
            einsum_checkpoint_epoch=ckpt_meta["einsum"]["epoch"],
            einsum_checkpoint_global_step=ckpt_meta["einsum"]["global_step"],
            deconv_checkpoint_epoch=ckpt_meta["deconv_v2"]["epoch"],
            deconv_checkpoint_global_step=ckpt_meta["deconv_v2"]["global_step"],
            seed=42,
            einsum_pi_nme_pct=float(re["nme"]) * 100, deconv_pi_nme_pct=float(rd["nme"]) * 100,
            diff_pp=c["diff"] * 100,
            einsum_pred_x0=re["pred_x0"], einsum_pred_y0=re["pred_y0"],
            einsum_pred_x1=re["pred_x1"], einsum_pred_y1=re["pred_y1"],
            deconv_pred_x0=rd["pred_x0"], deconv_pred_y0=rd["pred_y0"],
            deconv_pred_x1=rd["pred_x1"], deconv_pred_y1=rd["pred_y1"],
            einsum_raw_min=float(e_hm.min()), einsum_raw_max=float(e_hm.max()),
            deconv_raw_min=float(d_hm.min()), deconv_raw_max=float(d_hm.max()),
            deconv_channel_collapse_same_pixel=same_pixel_collapse,
            display_resize="bilinear to 512x512, clipped to [0,1]",
        ))

    manifest_path = args.out_dir / "bpd_heatmap_comparison_manifest.tsv"
    fieldnames = list(manifest_rows[0].keys())
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        w.writerows(manifest_rows)
    print(f"Wrote {manifest_path}")

    # --- render figure ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(3, 4, figsize=(11.5, 9.0))
    for row, name in enumerate(["median", "p75", "worst"]):
        c = cases[name]
        img = np.asarray(Image.open(args.images_dir / c["filename"]).convert("RGB")
                          .resize((512, 512), Image.BILINEAR))
        re, rd = einsum_rows[c["test_idx"]], deconv_rows[c["test_idx"]]
        gt = [(float(re["gt_x0"]), float(re["gt_y0"])), (float(re["gt_x1"]), float(re["gt_y1"]))]
        pred_e = [(float(re["pred_x0"]), float(re["pred_y0"])),
                  (float(re["pred_x1"]), float(re["pred_y1"]))]
        pred_d = [(float(rd["pred_x0"]), float(rd["pred_y0"])),
                  (float(rd["pred_x1"]), float(rd["pred_y1"]))]

        ax = axes[row, 0]
        ax.imshow(img)
        for x, y in gt:
            ax.plot(x, y, marker="o", markerfacecolor="none", markeredgecolor="lime",
                    markeredgewidth=1.8, markersize=11)
        case_label = {
            "median": "Median case",
            "p75": "Upper-quartile case",
            "worst": "Largest-difference case",
        }[name]
        ax.set_ylabel(case_label, fontsize=8.5)
        ax.set_xticks([]); ax.set_yticks([])

        for col, head in [(1, "einsum"), (2, "deconv_v2")]:
            ax = axes[row, col]
            ax.imshow(heatmap_rgb(heatmaps[f"{name}__{head}"]))
            for x, y in gt:
                ax.plot(x, y, marker="+", color="lime", markersize=10, markeredgewidth=1.5)
            ax.set_xticks([]); ax.set_yticks([])
            nme_val = (re["nme"] if head == "einsum" else rd["nme"])
            ax.set_title(f"PI-NME={float(nme_val)*100:.2f}%", fontsize=7.8, pad=3)

        ax = axes[row, 3]
        ax.imshow(img)
        for x, y in gt:
            ax.plot(x, y, marker="o", markerfacecolor="none", markeredgecolor="lime",
                    markeredgewidth=1.8, markersize=11)
        for x, y in pred_e:
            ax.plot(x, y, marker="x", color="orange", markersize=9, markeredgewidth=2.0)
        for x, y in pred_d:
            ax.plot(x, y, marker="^", markerfacecolor="none", markeredgecolor="magenta",
                    markeredgewidth=1.8, markersize=8)
        ax.set_xticks([]); ax.set_yticks([])

    axes[0, 0].set_title("Input + GT", fontsize=9.2)
    axes[0, 3].set_title("Prediction overlay", fontsize=9.2)
    for col, label in [(1, "einsum heatmaps\n(raw ch0=red, ch1=blue)"),
                       (2, "DeconvHeadV2 heatmaps\n(raw ch0=red, ch1=blue)")]:
        axes[0, col].set_title(f"{label}\n{axes[0, col].get_title()}", fontsize=8.0)

    legend_elems = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
               markeredgecolor="lime", markeredgewidth=1.8, markersize=10, label="Ground truth"),
        Line2D([0], [0], marker="x", color="orange", markeredgewidth=2.0, markersize=9,
               linestyle="none", label="einsum prediction"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor="none",
               markeredgecolor="magenta", markeredgewidth=1.8, markersize=8,
               linestyle="none", label="DeconvHeadV2 prediction"),
    ]
    fig.legend(handles=legend_elems, loc="lower center", ncol=3, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.005), frameon=False)
    fig.suptitle(
        "BPD seed-42 einsum vs. DeconvHeadV2: final-layer response maps for three\n"
        "deterministically selected test cases (median / 75th-percentile / worst\n"
        "DeconvHeadV2-minus-einsum per-image PI-NME difference)",
        fontsize=9.5, y=1.01,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.99])

    out_png = args.out_dir / "bpd_einsum_deconvheadv2_heatmaps.png"
    out_pdf = args.out_dir / "bpd_einsum_deconvheadv2_heatmaps.pdf"
    fig.savefig(out_png, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    print(f"Wrote {out_png}")
    print(f"Wrote {out_pdf}")


if __name__ == "__main__":
    main()
