"""Generates an MMPose config .py file for one (dataset, task, seed) cell,
adapted from the official rtmpose-s_8xb256-420e_coco-256x192.py per every
lock in PROTOCOL_LOCKED.md.

NEEDS LIVE-ENVIRONMENT VERIFICATION (see README.md "What still needs a live
environment"): the exact field names/shapes below were cross-checked against
the official config's documented structure (backbone init_cfg/prefix, codec
args, head args, pipeline transform list) but this project has not yet
imported mmpose to confirm the generated config actually builds a model and
runs a forward pass. Run PREFLIGHT_ONLY-style config validation (build the
model, print structure, run one dummy batch) before spending GPU time.

Deliberate deviations from the official recipe, each one because
PROTOCOL_LOCKED.md or the supervisor's email requires it, not because the
official recipe was copied carelessly:
  - out_channels=2 (not 17), in_featuremap_size=(16,16) (not (6,8)),
    input_size=(512,512) (not (192,256)): two-endpoint task, 512x512 lock.
  - sigma=(8.0,8.0): extrapolated per PROTOCOL_LOCKED.md, not an official
    512x512 value -- said explicitly in a config comment, not just the
    surrounding docs, so it cannot be missed by someone reading the .py file
    in isolation.
  - GetBBoxCenterScale + TopdownAffine replaced by PixelCentreResize in both
    train and val pipelines.
  - RandomHalfBody removed entirely (supervisor's explicit instruction --
    a person-specific augmentation with no meaning for a two-endpoint
    fetal measurement).
  - test_cfg flip_test=False: this project already investigated and
    abandoned flip-based TTA for two-endpoint fetal tasks (thesis
    sec:method-tta) because endpoint-order canonicalisation is unstable
    under flip for near-vertical diameters; enabling RTMPose's own
    flip-test would silently reintroduce the exact failure mode already
    documented and avoided for EoMT.
  - Rotation/scale/flip augmentation (RandomBBoxTransform/RandomFlip) is
    NOT delegated to MMPose's stock transforms; two custom transforms
    (transforms.FetalRandomFlipAndCanonicalize, transforms.
    FetalRotateScaleColorJitter) replace them. Two review passes on
    2026-08-06: the first found that a static flip_indices setting (this
    file's original version) is silently wrong whenever a task's DOD
    direction is near-horizontal; the SECOND pass then found that the
    first pass's own "fix" (transforming d_vect through flip+rotation
    before reprojecting, done in already-resized 512-space) was ITSELF
    mathematically wrong -- verified directly against HRNet's real
    get_transform/_transform_pixel_float formulas that HRNet's actual
    per-sample decision is invariant to center/scale/rotation and depends
    ONLY on flip, evaluated via the STATIC original d_vect in ORIGINAL
    (pre-resize) image space. See fetal_augment.py's module docstring for
    the full derivation (verified both algebraically and by direct
    numerical reproduction of HRNet's exact formula) and PROTOCOL_AUDIT.md
    for the complete history of both passes. Rotation (ROT_FACTOR=30,
    p=0.6) and scale (SCALE_FACTOR=0.25, unconditional) augmentation
    (previously deferred entirely) are now implemented as pure position
    updates with no channel-order involvement, since they are proven to
    have no effect on the ordering decision.
  - val_dataloader points at a Train-only internal validation split
    (make_internal_val_split.py, reusing EoMT's own exact subject-based
    split algorithm), NOT the released Test set -- CORRECTED 2026-08-06,
    review finding: the original version pointed val_dataloader directly
    at Test with test_dataloader=val_dataloader and save_best="PCK",
    meaning the official Test set was read every val_interval epochs
    during training and used for checkpoint selection, a genuine leak.
    test_dataloader is now a separate dataloader over the real Test set,
    touched only once, after training, by run_inference.py.
  - Backbone-only pretrained init via prefix='backbone.'; RTMCCHead fully
    randomly initialised (matches the OFFICIAL recipe's own convention,
    confirmed against the real config -- not a compromise specific to this
    adaptation).

THIRD REVIEW ROUND (2026-08-06): the official
rtmpose-s_8xb256-420e_coco-256x192.py was fetched and read VERBATIM
(https://raw.githubusercontent.com/open-mmlab/mmpose/main/configs/
body_2d_keypoint/rtmpose/coco/rtmpose-s_8xb256-420e_coco-256x192.py) to
check a reviewer's claim that this generator had silently dropped several
official settings -- confirmed true for some, NOT confirmed for one
(the reviewer's claim of "gradient clipping" in the official optim_wrapper
does not match the actual fetched file, which has no `clip_grad` key at
all; not added here since it was never actually there to match). Full
item-by-item table:

| Official setting | Present here? | Disposition |
|---|---|---|
| `backbone._scope_='mmdet'` | was MISSING | **FIXED** -- CSPNeXt is registered under mmdet's scope; omitting this could fail to resolve the builder in a real install, a construction-correctness bug, not a style gap |
| `backbone.expand_ratio=0.5` | was MISSING | **FIXED** -- an actual CSPNeXt architecture parameter, not cosmetic; omitting it risks the backbone's internal channel widths not matching the pretrained checkpoint's own shapes, which could silently corrupt or outright fail the weight load |
| `backbone.norm_cfg=SyncBN` | was `BN` | **FIXED to SyncBN** to match the checkpoint's own training norm type; flagged in ENVIRONMENT.md that SyncBN may need `torch.distributed` initialised even for a single-GPU run depending on the installed MMEngine/MMCV version -- confirm live, fall back to BN with a recorded justification if SyncBN errors out single-process |
| `backbone.act_cfg` extra `inplace=True` | present, official has none | **FIXED** -- removed for exact fidelity (does not change weight shapes, low risk, but no reason to diverge) |
| `optim_wrapper.paramwise_cfg` (zero weight decay on norm/bias) | was MISSING | **FIXED** -- matched, no dataset-scale-dependent reason to omit it |
| Cosine LR starting at `max_epochs // 2` | was starting at epoch 0 | **FIXED** -- now starts at `max_epochs // 2` proportionally, preserving the official recipe's warmup-then-half-cosine shape at whatever `max_epochs` this project uses |
| `auto_scale_lr=dict(base_batch_size=1024)` | MISSING | **DELIBERATELY NOT ADDED** -- only takes effect via an explicit `--auto-scale-lr` CLI flag (not automatic from the config alone), and this project's `base_lr` was never tuned as a linear-scaling assumption from batch=1024; our batch_size (16, dataset-scale-appropriate) differs from official's 256 by design, not by oversight |
| `custom_hooks: EMAHook` | MISSING | **DELIBERATELY NOT ADDED** -- this project's own EMA investigation for EoMT found "insufficient clean evidence either way" (thesis Ch5 ablation), so adding EMA to RTMPose alone, without it being part of the shared cross-method recipe, would be a new, unreviewed asymmetry, not a neutral fidelity fix |
| `custom_hooks: PipelineSwitchHook` (stage-2 augmentation cooldown) | MISSING | **DELIBERATELY NOT ADDED** -- the official stage-2 switch reduces the OFFICIAL RandomBBoxTransform's own scale/rotate ranges; this project already replaces that whole augmentation with EoMT/HRNet-matched values (PROTOCOL_LOCKED.md), so there is no equivalent "official range" to cool down between two stages |
| `max_epochs=420` | is 200 (configurable) | **KEPT at 200, documented, not silently accidental** -- 420 epochs was tuned for COCO's ~118k training images; this project's fetal datasets are 2-3 orders of magnitude smaller (Train ~100-1600 images per task), so a directly-copied epoch count has no principled basis either way. 200 is this project's own choice, adjustable via `--max-epochs`, not a claimed replication of the official schedule. |
| Pretrained checkpoint's own `checkpoint=` field | was a hardcoded URL | **FIXED (blocking issue)**: `model.init_weights()` would have loaded from this URL, completely independent of whatever local file `record_run_provenance.py` was separately hashing -- meaning the recorded SHA-256 was never guaranteed to correspond to the actual weights loaded into the model. `make_config()` now REQUIRES a local `pretrained_checkpoint_path` and embeds THAT path as `init_cfg.checkpoint`, so the file that gets loaded and the file that gets hashed/diffed are, by construction, the same file. |
"""

from __future__ import annotations

from pathlib import Path

OFFICIAL_CSPNEXT_S_BACKBONE_CHECKPOINT_URL = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
    "cspnext-s_udp-aic-coco_210e-256x192-92f5a029_20230130.pth"
)
# Download-source documentation ONLY (see ENVIRONMENT.md's download step) --
# NOT used as the generated config's own init_cfg.checkpoint value anymore
# (see the provenance fix above); make_config() requires a local file path
# instead, so the loaded weights and the audited/hashed weights are
# guaranteed to be the same file, not just assumed to match by URL.

TEMPLATE = '''\
# AUTO-GENERATED by rtmpose_reproduction/make_config.py -- do not hand-edit.
# dataset={dataset} task={task} seed={seed}
# See rtmpose_reproduction/PROTOCOL_LOCKED.md and make_config.py's own
# module docstring for what was locked/changed and why.

import sys
sys.path.insert(0, {repo_root!r})  # for transforms.py / fetal_dataset_info.py

default_scope = "mmpose"
randomness = dict(seed={seed}, deterministic=True)

# --- import the custom transform so @TRANSFORMS.register_module() runs ---
import transforms  # noqa: F401,E402
from fetal_dataset_info import FETAL_DATASET_INFO  # noqa: E402
from dod_vectors import get_d_vect  # noqa: E402

# Frozen DOD prototype vector for this (dataset, task), reused verbatim from
# the audited upstream HRNet reproduction (dod_vectors.py) -- FetalRandomFlipAndCanonicalize
# uses this to re-derive the canonical channel order after every flip draw,
# in ORIGINAL image space, replacing the old static-flip_indices design (see
# fetal_augment.py's module docstring and PROTOCOL_AUDIT.md).
d_vect = get_d_vect({dataset!r}, {task!r})

input_size = (512, 512)

codec = dict(
    type="SimCCLabel",
    input_size=input_size,
    sigma=(8.0, 8.0),  # EXTRAPOLATED value, not an official 512x512 setting
                        # -- see PROTOCOL_LOCKED.md for the sigma_axis =
                        # sqrt(axis_length / 8) derivation from the two real
                        # official (192,256)/(288,384) configs.
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False,
)

model = dict(
    type="TopdownPoseEstimator",
    data_preprocessor=dict(
        type="PoseDataPreprocessor",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
    ),
    backbone=dict(
        _scope_="mmdet",  # CSPNeXt is registered under mmdet's scope --
                          # was missing; likely a real build-time failure
                          # risk in a live install, not cosmetic.
        type="CSPNeXt",
        arch="P5",
        expand_ratio=0.5,  # matches the official recipe exactly -- omitting
                           # this real architecture parameter risked a
                           # channel-width mismatch against the pretrained
                           # checkpoint's own shapes.
        deepen_factor=0.33,
        widen_factor=0.5,
        out_indices=(4,),
        channel_attention=True,
        norm_cfg=dict(type="SyncBN"),  # matches official; CONFIRM this
                                        # builds single-GPU/non-distributed
                                        # on the installed MMEngine/MMCV
                                        # before trusting it (see
                                        # ENVIRONMENT.md) -- fall back to BN
                                        # only with a recorded justification.
        act_cfg=dict(type="SiLU"),
        init_cfg=dict(
            type="Pretrained",
            prefix="backbone.",
            checkpoint={backbone_checkpoint!r},  # LOCAL FILE PATH, not a
                                                  # URL -- see make_config()'s
                                                  # own docstring: this
                                                  # guarantees the weights
                                                  # loaded here are the same
                                                  # file record_run_provenance.py
                                                  # hashes and diffs.
        ),
    ),
    head=dict(
        type="RTMCCHead",
        in_channels=512,
        out_channels=2,          # LOCKED: two endpoints, not COCO's 17
        input_size=input_size,   # LOCKED: 512x512, not (192,256)
        in_featuremap_size=(16, 16),  # = input_size / 32 (stride-32 backbone stage)
        simcc_split_ratio=2.0,
        final_layer_kernel_size=7,
        gau_cfg=dict(
            hidden_dims=256, s=128, expansion_factor=2,
            dropout_rate=0.0, drop_path=0.0, act_fn="SiLU",
            use_rel_bias=False, pos_enc=False,
        ),
        loss=dict(
            type="KLDiscretLoss", use_target_weight=True,
            beta=10.0, label_softmax=True,
        ),
        decoder=codec,
    ),
    test_cfg=dict(
        flip_test=False,  # LOCKED off: see module docstring (endpoint-order
                           # instability under flip for near-vertical
                           # diameters, already documented for EoMT's TTA).
    ),
)

# --- data ---
dataset_type = "CocoDataset"
data_mode = "topdown"

train_pipeline = [
    dict(type="LoadImage"),
    # Channel identity MUST be decided in ORIGINAL image space, before the
    # anisotropic 512x512 resize -- see transforms.FetalRandomFlipAndCanonicalize's
    # own docstring and fetal_augment.py's module docstring for why (verified
    # against HRNet's real per-sample transform formula, not assumed).
    dict(type="FetalRandomFlipAndCanonicalize", d_vect=d_vect, flip_prob=0.5),
    # RandomHalfBody deliberately removed (supervisor instruction).
    dict(type="PixelCentreResize", input_size=512),
    # Rotation (+-30 deg, p=0.6), scale (0.75-1.25, unconditional), colour
    # jitter -- position-only, no re-canonicalisation needed here (proven to
    # have no effect on channel order). See PROTOCOL_AUDIT.md's augmentation
    # item-by-item table for what remains disclosed as non-bit-identical to
    # EoMT/HRNet.
    dict(type="FetalRotateScaleColorJitter", input_size=512),
    dict(type="GenerateTarget", encoder=codec),
    dict(type="PackPoseInputs"),
]

val_pipeline = [
    dict(type="LoadImage"),
    dict(type="PixelCentreResize", input_size=512),
    dict(type="PackPoseInputs"),
]

train_dataloader = dict(
    batch_size={batch_size},
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True, seed={seed}),
    dataset=dict(
        type=dataset_type,
        data_root={data_root!r},
        data_mode=data_mode,
        ann_file={internal_train_ann!r},
        data_prefix=dict(img={images_dir!r}),
        metainfo=FETAL_DATASET_INFO,
        pipeline=train_pipeline,
    ),
)

# CORRECTED 2026-08-06 (review finding): this used to point at the released
# Test annotation file, with test_dataloader = val_dataloader, so the
# official Test set was read every val_interval epochs during training and
# fed into save_best="PCK" checkpoint selection -- a genuine data leak, not
# just a soft protocol violation, regardless of the fact that the REPORTED
# result was always going to be the final checkpoint. val_dataloader now
# points at a Train-only internal validation split
# (make_internal_val_split.py, reusing EoMT's own exact subject-grouping/
# shuffle/split algorithm from datasets/landmark_dataset.py so the held-out
# subjects match EoMT's own internal validation). test_dataloader is now a
# SEPARATE dataloader pointing at the real released Test set, and nothing
# in this config ever runs it automatically -- run_inference.py is the only
# thing that reads it, once, after training is completely finished.
val_dataloader = dict(
    batch_size={batch_size},
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root={data_root!r},
        data_mode=data_mode,
        ann_file={internal_val_ann!r},
        data_prefix=dict(img={images_dir!r}),
        metainfo=FETAL_DATASET_INFO,
        pipeline=val_pipeline,
        test_mode=True,
    ),
)
test_dataloader = dict(
    batch_size={batch_size},
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root={data_root!r},
        data_mode=data_mode,
        ann_file={test_ann!r},
        data_prefix=dict(img={images_dir!r}),
        metainfo=FETAL_DATASET_INFO,
        pipeline=val_pipeline,
        test_mode=True,
    ),
)

# Internal training-time sanity metric ONLY, computed on the Train-only
# internal-val split above -- NOT the reported number, and no longer
# computed from the released Test set at all. The authoritative
# fixed-channel/swap-min NME comes from evaluate_rtmpose_fixed.py run on
# run_inference.py's exported, original-image-space TEST predictions
# (produced exactly once, after training), using the identical formula as
# HRNet's evaluator.
val_evaluator = dict(type="PCKAccuracy", thr=0.05)
test_evaluator = val_evaluator

train_cfg = dict(by_epoch=True, max_epochs={max_epochs}, val_interval={val_interval})
val_cfg = dict()
test_cfg = dict()

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=4e-3, weight_decay=0.0),
    # Matches the official recipe's paramwise_cfg exactly -- was missing;
    # no dataset-scale-dependent reason to omit this.
    paramwise_cfg=dict(norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True),
)
# Cosine annealing now starts at max_epochs // 2, matching the official
# recipe's own proportional shape (there: begin=210 of max_epochs=420) --
# was previously starting at epoch 0, overlapping the LinearLR warmup
# entirely and not matching the official schedule's shape at all.
_cosine_begin = {max_epochs} // 2
param_scheduler = [
    dict(type="LinearLR", start_factor=1e-5, by_epoch=False, begin=0, end=1000),
    dict(type="CosineAnnealingLR", eta_min=0.0002, begin=_cosine_begin, end={max_epochs},
         T_max={max_epochs} - _cosine_begin, by_epoch=True, convert_to_iter_based=True),
]

# CORRECTED 2026-08-06 (review finding): save_best="PCK" removed entirely --
# PROTOCOL_LOCKED.md's primary checkpoint convention is final/last, and
# selecting-by-best (even on the now-legitimate Train-only internal val
# metric, not the Test set) risks the canary driver silently picking up a
# "best" checkpoint file instead of the true final one. save_last=True
# guarantees the true final-epoch checkpoint is always kept regardless of
# max_keep_ckpts, and run_rtmpose_canary.sh now reads MMEngine's own
# last_checkpoint pointer file rather than glob-sorting filenames.
default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook", interval={val_interval},
        save_last=True, max_keep_ckpts=1,
    ),
    logger=dict(type="LoggerHook", interval=10),
)

work_dir = {work_dir!r}
'''


def make_config(dataset: str, task: str, seed: int, data_root: str,
                 images_dir: str, internal_train_ann: str, internal_val_ann: str,
                 test_ann: str, pretrained_checkpoint_path: str,
                 work_dir: str, out_path: Path,
                 batch_size: int = 16, max_epochs: int = 200,
                 val_interval: int = 5, repo_root: str | None = None) -> Path:
    """`internal_train_ann`/`internal_val_ann` are the two COCO jsons
    produced by converting the Train CSV twice, restricted to
    make_internal_val_split.py's two filename lists (see that script's own
    docstring) -- NEVER pass the released Test CSV/json as either of these.
    `test_ann` is the real released Test set, used only by run_inference.py
    after training; nothing in the generated config's training loop reads it.

    `pretrained_checkpoint_path` MUST be a local file path (downloaded per
    ENVIRONMENT.md's instructions), NOT the download URL -- this is what
    gets embedded as the backbone's `init_cfg.checkpoint`, so the weights
    `model.init_weights()` actually loads and the file
    `record_run_provenance.py` hashes/diffs are, by construction, the same
    file (see this file's own module docstring for the leak this closes)."""
    if not Path(pretrained_checkpoint_path).is_file():
        raise SystemExit(
            f"ERROR: --pretrained-checkpoint-path does not exist: "
            f"{pretrained_checkpoint_path}. Download it first (see "
            f"ENVIRONMENT.md) -- refusing to embed a checkpoint path into "
            f"the config that init_weights() cannot actually load from."
        )
    text = TEMPLATE.format(
        dataset=dataset, task=task, seed=seed,
        repo_root=repo_root or str(Path(__file__).resolve().parent),
        backbone_checkpoint=str(pretrained_checkpoint_path),
        data_root=data_root, images_dir=images_dir,
        internal_train_ann=internal_train_ann, internal_val_ann=internal_val_ann,
        test_ann=test_ann,
        batch_size=batch_size, max_epochs=max_epochs, val_interval=val_interval,
        work_dir=work_dir,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    return out_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["UCL", "MULTICENTRE"])
    parser.add_argument("--task", required=True, choices=["BPD", "OFD", "APAD", "TAD", "FL"])
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--internal-train-ann", required=True,
                         help="COCO json converted from Train CSV, internal_train filenames only "
                              "(make_internal_val_split.py + convert_csv_to_coco.py --internal-split-part internal_train)")
    parser.add_argument("--internal-val-ann", required=True,
                         help="COCO json converted from Train CSV, internal_val filenames only "
                              "(make_internal_val_split.py + convert_csv_to_coco.py --internal-split-part internal_val)")
    parser.add_argument("--test-ann", required=True,
                         help="COCO json converted from the REAL released Test CSV -- "
                              "never read during training, only by run_inference.py afterward")
    parser.add_argument("--pretrained-checkpoint-path", required=True,
                         help="LOCAL file path (not a URL) to the downloaded CSPNeXt-s "
                              "checkpoint -- see ENVIRONMENT.md's download step. Embedded "
                              "directly as backbone.init_cfg.checkpoint so the weights "
                              "actually loaded and the file record_run_provenance.py "
                              "hashes are guaranteed to be the same file.")
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    path = make_config(args.dataset, args.task, args.seed, args.data_root,
                        args.images_dir, args.internal_train_ann, args.internal_val_ann,
                        args.test_ann, args.pretrained_checkpoint_path,
                        args.work_dir, args.out)
    print(f"[OK] wrote {path}")
