#!/usr/bin/env python3
"""Image-level UCL analysis for the eight recoverable EoMT task/backbone cells."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

SEEDS = (42, 0, 123, 2024, 3407)
TASKS = {"ofd": "brain_OFD", "apad": "abdomen_APAD",
         "tad": "abdomen_TAD", "fl": "femur_FL"}
BACKBONES = ("dinov2", "dinov3")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--eomt-root", type=Path, default=Path("/root/autodl-tmp/ucl_eomt_per_image"))
    p.add_argument("--hrnet-root", type=Path, default=Path("/root/autodl-tmp/hrnet_512_fixed_5seed/output/FETAL"))
    p.add_argument("--output-root", type=Path, default=Path("/root/autodl-tmp/hrnet_512_fixed_5seed/ucl_paired_analysis"))
    p.add_argument("--bootstrap-replicates", type=int, default=20000)
    p.add_argument("--bootstrap-seed", type=int, default=20260802)
    return p.parse_args()


def load(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def canon(s):
    return s.strip().replace("\\", "/").rsplit("/", 1)[-1]


def hrnet(root, tag):
    seeds = []
    for seed in SEEDS:
        run = f"fetal_landmark_hrnet_w18_UCL_{tag}_seed{seed}_512fixed"
        path = root / run / "fixed_channel_per_image.csv"
        rows = load(path)
        data = {canon(r["filename"]): (float(r["fixed_channel_nme"]), float(r["swap_min_nme"])) for r in rows}
        if len(data) != len(rows):
            raise ValueError(f"duplicate HRNet filenames in {path}")
        seeds.append(data)
    keys = set(seeds[0])
    if any(set(x) != keys for x in seeds[1:]):
        raise ValueError(f"HRNet seed filename mismatch: {tag}")
    return {k: (np.mean([x[k][0] for x in seeds]), np.mean([x[k][1] for x in seeds])) for k in keys}


def eomt(root, task, backbone):
    seeds = []
    for seed in SEEDS:
        run = root / f"{task}_{backbone}" / f"seed{seed}"
        order_rows = load(run / "test_image_order.csv")
        name_column = "img_name" if "img_name" in order_rows[0] else "filename"
        order = {int(r["index"]): canon(r[name_column]) for r in order_rows}
        fixed_rows = load(run / "final_fixedchannel_per_image.csv")
        swap_rows = load(run / "final_swapmin_per_image.csv")
        fixed = {int(r["index"]): float(r["nme"]) for r in fixed_rows}
        swap = {int(r["index"]): float(r["nme"]) for r in swap_rows}
        if set(order) != set(fixed) or set(order) != set(swap):
            raise ValueError(f"EoMT index mismatch: {run}")
        joined = {order[i]: (fixed[i], swap[i]) for i in order}
        if len(joined) != len(order):
            raise ValueError(f"duplicate EoMT filenames: {run}")
        seeds.append(joined)
    keys = set(seeds[0])
    if any(set(x) != keys for x in seeds[1:]):
        raise ValueError(f"EoMT seed filename mismatch: {task}/{backbone}")
    return {k: (np.mean([x[k][0] for x in seeds]), np.mean([x[k][1] for x in seeds])) for k in keys}


def boot(values, reps, rng):
    means = np.empty(reps)
    for start in range(0, reps, 1000):
        stop = min(start + 1000, reps)
        idx = rng.integers(0, len(values), size=(stop-start, len(values)))
        means[start:stop] = values[idx].mean(axis=1)
    return np.quantile(means, [0.025, 0.975])


def holm(ps):
    order = np.argsort(ps); out = np.empty(len(ps)); running = 0.0; m = len(ps)
    for rank, idx in enumerate(order):
        running = max(running, (m-rank)*ps[idx]); out[idx] = min(1.0, running)
    return out


def main():
    a = parse_args(); a.output_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.bootstrap_seed); summary = []
    for task, tag in TASKS.items():
        h = hrnet(a.hrnet_root, tag)
        for backbone in BACKBONES:
            e = eomt(a.eomt_root, task, backbone)
            common = sorted(set(e) & set(h))
            if not common or set(common) != set(h) or set(common) != set(e):
                raise ValueError(f"{task}/{backbone}: filename mismatch, common={len(common)}")
            ef=np.array([e[x][0] for x in common])*100; es=np.array([e[x][1] for x in common])*100
            hf=np.array([h[x][0] for x in common])*100; hs=np.array([h[x][1] for x in common])*100
            if np.any(ef+1e-10<es) or np.any(hf+1e-10<hs): raise ValueError("fixed/swap invariant failed")
            diff=ef-hf; lo,hi=boot(diff,a.bootstrap_replicates,rng)
            try: p=float(wilcoxon(diff).pvalue)
            except ValueError: p=1.0
            gap=ef-es
            summary.append({"task":task,"backbone":backbone,"n_images":len(common),
              "eomt_fixed_mean_pct":f"{ef.mean():.8f}","hrnet_fixed_mean_pct":f"{hf.mean():.8f}",
              "eomt_minus_hrnet_mean_pp":f"{diff.mean():.8f}","bootstrap_95ci_low_pp":f"{lo:.8f}",
              "bootstrap_95ci_high_pp":f"{hi:.8f}","wilcoxon_p_raw":f"{p:.12g}","wilcoxon_p_holm":"",
              "eomt_fixed_minus_swap_mean_pp":f"{gap.mean():.8f}",
              "eomt_fixed_minus_swap_median_pp":f"{np.median(gap):.8f}",
              "eomt_swap_preferred_fraction":f"{np.mean(gap>1e-10):.8f}",
              "hrnet_fixed_minus_swap_mean_pp":f"{(hf-hs).mean():.8f}"})
            path=a.output_root/f"{task}_{backbone}_paired_per_image.csv"
            fields=["filename","eomt_fixed_nme_pct","eomt_swap_nme_pct","hrnet_fixed_nme_pct","hrnet_swap_nme_pct","eomt_minus_hrnet_fixed_pp","eomt_fixed_minus_swap_pp"]
            with path.open("w",newline="",encoding="utf-8") as f:
                w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
                for i,n in enumerate(common): w.writerow(dict(zip(fields,[n,f"{ef[i]:.8f}",f"{es[i]:.8f}",f"{hf[i]:.8f}",f"{hs[i]:.8f}",f"{diff[i]:.8f}",f"{gap[i]:.8f}"])))
    for row,p in zip(summary,holm([float(x["wilcoxon_p_raw"]) for x in summary])): row["wilcoxon_p_holm"]=f"{p:.12g}"
    path=a.output_root/"ucl_paired_summary.tsv"; fields=list(summary[0])
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields,delimiter="\t"); w.writeheader(); w.writerows(summary)
    if len(summary)!=8: raise RuntimeError("expected eight comparisons")
    print(f"[COMPLETE] wrote {path} and eight paired per-image CSVs")


if __name__ == "__main__": main()
