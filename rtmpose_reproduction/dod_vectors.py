"""Frozen endpoint-direction ("DOD") prototype vectors, reused verbatim from
the already-trained, already-locked HRNet-512/fixed reproduction so that
RTMPose shares an IDENTICAL endpoint-ordering convention with HRNet, rather
than an independently re-derived (and therefore only similar, not identical)
one.

Per rtmpose_reproduction/PROTOCOL_LOCKED.md's "Endpoint identity gate":
direction information must be estimated from the training partition only and
frozen for validation/test. HRNet already does exactly this (see the
`d_vect`/`determine_direction` logic in lib/datasets/fetal.py of the audited
upstream checkout) via a Gaussian-Mixture fit with a FIXED
`random_state=0` -- deterministic, and a function of the training CSV only,
not of the model's training seed.

**Verified empirically, not assumed**: extracted `d_vect` from two different
seeds of the same task (UCL brain/BPD, seed 42 vs seed 0) out of the
archived `checkpoint_backups/hrnet-512-fixed-50runs.tar` final_state.pth
checkpoints -- both seeds produced the bit-identical
[[545.2607421875, 125.89927673339844], [549.0957641601562, 562.2645874023438]],
confirming d_vect does not depend on the training seed, only on the
(dataset, task) pair. One seed per (dataset, task) was therefore sufficient
to extract all 10 vectors below (seed 42 used throughout for consistency).

**Also verified against real per-image output, not just re-derived from
theory**: using these vectors with endpoint_order.canonical_order() exactly
reproduces the gt0/gt1 order in HRNet's own
`fixed_channel_per_image.csv` for UCL brain/BPD seed 42, including a case
where the raw CSV order is swapped (004_HC.jpeg: raw (bpd_1=403,532),
(bpd_2=399,133) -> canonical gt0=(399,133), gt1=(403,532), matching this
project's own extracted HRNet output exactly). See test_endpoint_order.py.

Provenance command (re-run if any of the 10 tasks is retrained and this file
ever needs regenerating -- do not hand-edit the numbers below without
re-running this):

    tar --force-local -xf checkpoint_backups/hrnet-512-fixed-50runs.tar \\
        -C <scratch> <path>/final_state.pth   # one seed42 run per task
    python -c "
        import torch
        p = torch.load('<final_state.pth>', map_location='cpu', weights_only=False)
        print(p['d_vect'].tolist())
    "

Extracted 2026-08-0x from the locked hrnet-512-fixed-50runs.tar archive
(SHA-256 d3c5229597205e0f9457c0237be791af7af927ee902f8949cede630ee2d9fe65
per the chronological summary's Sec.31.18), seed 42 run for every task.
"""

from __future__ import annotations

# (dataset, task) -> ((x0, y0), (x1, y1)) prototype endpoint pair, in
# ORIGINAL image pixel coordinates (not model-input-space, not heatmap-space
# -- this matches HRNet's own self.d_vect, which is computed directly from
# the raw training CSV before any crop/warp/resize).
D_VECT: dict[tuple[str, str], tuple[tuple[float, float], tuple[float, float]]] = {
    ("UCL", "BPD"): ((545.2607421875, 125.89927673339844), (549.0957641601562, 562.2645874023438)),
    ("UCL", "OFD"): ((815.1532592773438, 336.1002197265625), (257.09954833984375, 347.5360107421875)),
    ("UCL", "APAD"): ((300.5794372558594, 336.6518859863281), (670.0562744140625, 358.6502990722656)),
    ("UCL", "TAD"): ((501.6695861816406, 139.31483459472656), (562.228515625, 517.4895629882812)),
    ("UCL", "FL"): ((357.1091003417969, 250.12591552734375), (672.7321166992188, 272.8202819824219)),
    ("MULTICENTRE", "BPD"): ((414.9869079589844, 440.8841247558594), (423.97113037109375, 76.86175537109375)),
    ("MULTICENTRE", "OFD"): ((637.6757202148438, 263.7032165527344), (184.76670837402344, 268.9605407714844)),
    ("MULTICENTRE", "APAD"): ((663.3988647460938, 291.9216613769531), (251.21719360351562, 314.1796875)),
    ("MULTICENTRE", "TAD"): ((466.22216796875, 88.43079376220703), (476.3613586425781, 491.39727783203125)),
    ("MULTICENTRE", "FL"): ((228.73423767089844, 202.2417755126953), (563.6278686523438, 192.7150115966797)),
}


def get_d_vect(dataset: str, task: str) -> tuple[tuple[float, float], tuple[float, float]]:
    key = (dataset.upper(), task.upper())
    if key not in D_VECT:
        raise KeyError(f"no frozen d_vect for {key}; expected one of {sorted(D_VECT)}")
    return D_VECT[key]
