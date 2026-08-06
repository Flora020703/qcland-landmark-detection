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
  - `val_cfg=None`, `PCKAccuracy` NOT used for periodic monitoring --
    ROUND 5 fix (review finding, real crash/silent-wrong-number risk):
    MMEngine's default val loop calls `model.predict()`, which for a stock
    TopdownPoseEstimator typically needs bbox_center/bbox_scale metadata
    this project's PixelCentreResize never populates; whether this crashes
    or silently produces a meaningless number was unverified. Periodic
    internal monitoring is now `InternalFixedChannelNMEHook`
    (internal_val_hook.py), reusing run_inference.py's own verified-safe
    low-level decode path and the SAME fixed-channel NME formula as the
    final evaluation -- directly comparable numbers, not a different metric.
  - Backbone-only pretrained init via prefix='backbone.'; RTMCCHead fully
    randomly initialised (matches the OFFICIAL recipe's own convention,
    confirmed against the real config -- not a compromise specific to this
    adaptation).

THIRD REVIEW ROUND (2026-08-06): fetched
configs/body_2d_keypoint/rtmpose/coco/rtmpose-s_8xb256-420e_coco-256x192.py
verbatim to check a reviewer's claim that this generator had silently
dropped several official settings. That specific file has no `clip_grad`
key, so "gradient clipping" was (wrongly, see round 4) recorded here as
not a real official setting.

FOURTH REVIEW ROUND (2026-08-06, corrects round 3's own mistake): a
second reviewer pointed out that MMPose's actively-maintained RTMPose
project config lives at a DIFFERENT path,
projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-s_8xb256-420e_coco-256x192.py
(confirmed to exist via the GitHub contents API, last modified at commit
`94e15226a29a7067d9bb0cb7937b86e3c3fd0c8e`), not the
`configs/body_2d_keypoint/rtmpose/coco/` path round 3 fetched (last
modified at a DIFFERENT commit, `a910fd4c5684b0480f561efd703635d817944568`
-- these are two independently-maintained, diverged files with the same
filename, not one file at two mirror locations). Fetched the
`projects/rtmpose/` version verbatim: it DOES contain
`clip_grad=dict(max_norm=35, norm_type=2)`. Round 3's "gradient clipping
is not a real official setting" claim was WRONG -- it checked a stale/
divergent copy. Fixed below. This project now treats
`projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-s_8xb256-420e_coco-256x192.py`
at commit `94e15226a29a7067d9bb0cb7937b86e3c3fd0c8e` as the SOLE
authoritative source for "what does the official recipe do" -- per a
reviewer's explicit suggestion, do not re-derive this from a floating
`main` checkout in future sessions; if MMPose is later pinned to a
different commit for the actual training environment (see
ENVIRONMENT.md), diff this specific file at that commit against the
frozen table below before trusting either source blindly.

Full item-by-item table (all rows re-verified against the CORRECT
`projects/rtmpose/` source in round 4):

| Official setting | Present here? | Disposition |
|---|---|---|
| `backbone._scope_='mmdet'` | was MISSING | **FIXED** -- CSPNeXt is registered under mmdet's scope; omitting this could fail to resolve the builder in a real install, a construction-correctness bug, not a style gap |
| `backbone.expand_ratio=0.5` | was MISSING | **FIXED** -- an actual CSPNeXt architecture parameter, not cosmetic; omitting it risks the backbone's internal channel widths not matching the pretrained checkpoint's own shapes, which could silently corrupt or outright fail the weight load |
| `backbone.norm_cfg=SyncBN` | was `BN` | **FIXED to SyncBN** to match the checkpoint's own training norm type; flagged in ENVIRONMENT.md that SyncBN may need `torch.distributed` initialised even for a single-GPU run depending on the installed MMEngine/MMCV version -- confirm live, fall back to BN with a recorded justification if SyncBN errors out single-process |
| `backbone.act_cfg` extra `inplace=True` | present, official has none | **FIXED** -- removed for exact fidelity (does not change weight shapes, low risk, but no reason to diverge) |
| `optim_wrapper.clip_grad=dict(max_norm=35, norm_type=2)` | was MISSING | **FIXED in round 4** -- round 3 wrongly recorded this as not a real official setting after checking the wrong file path (see above); this is a real official optimizer setting, now matched |
| `optim_wrapper.paramwise_cfg` (zero weight decay on norm/bias) | was MISSING | **FIXED** -- matched, no dataset-scale-dependent reason to omit it |
| `base_lr=4e-3` used directly at `batch_size=16` | official's `4e-3` is paired with `train_batch_size=256` and `auto_scale_lr=dict(base_batch_size=1024)` (8 GPUs x 256) | **FIXED in round 4 (real methodological risk, not just a fidelity gap)**: using the official LR unscaled at a batch size 64x smaller than official's base_batch_size risked severe training instability. `base_lr` is now computed as `4e-3 * (batch_size / 1024)` -- for this project's `batch_size=16`, that is `6.25e-5`. `auto_scale_lr` is deliberately NOT also added to the generated config to avoid double-scaling if `--auto-scale-lr` is ever passed to `tools/train.py`; the scaling is applied once, explicitly, in this file, and the resulting `base_lr` is recorded in the generated config's own comment. |
| Cosine LR starting at `max_epochs // 2` | was starting at epoch 0, AND (round 4 finding) overlapped the 1000-iteration LinearLR warmup for this project's tiny datasets | **FIXED in round 4, warmup formula corrected again in round 5**: for a ~100-image internal-train split at `batch_size=16` (~7 iterations/epoch), the official recipe's fixed `end=1000` warmup would still be running past iteration 700 (`epoch_100 * 7`), overlapping a naive proportional `max_epochs // 2` cosine-begin. Round 4's fix (`warmup_end_iters = min(1000, cosine_begin_iters // 2)`) only guaranteed no overlap with no training-methodology basis -- for BPD's real numbers it gave 350 warmup iterations out of ~1400 total (25% of ALL training), a real underfitting risk a reviewer caught. Round 5: warmup is now a short, fixed number of EPOCHS (`min(5, max(1, max_epochs // 20))`, i.e. <=5 epochs) converted to iterations using the real image count -- ~35 iterations (2.5% of BPD's total training) instead of 350, much closer to the official recipe's own proportion (1000 of ~193,200 total iterations ~= 0.5%). Still asserts `warmup_end_iters < cosine_begin_iters`. |
| `custom_hooks: EMAHook` | MISSING | **KEPT NOT ADDED, reframed in round 4**: round 3 justified this by citing EoMT's own EMA findings, which a reviewer correctly pointed out is not valid evidence for RTMPose (a structurally different model/training setup) -- EoMT's EMA result says nothing about whether RTMPose specifically benefits from EMA. The real, honest framing is a SCOPE decision: the canary and initial runs use the raw final checkpoint deliberately, to keep one fewer moving part while validating the adapter end-to-end; this must be described in any writeup as "RTMPose-s architecture trained under this project's common fetal protocol," NOT as "the official RTMPose-s recipe" or "an RTMPose-s reproduction," precisely because EMA (and the stage-2 switch, and the native augmentation) are official-recipe components this project does not use. Revisit EMA with a real seed-42 raw-vs-EMA diagnostic if the canary's own numbers motivate it -- do not decide this from EoMT's unrelated result. |
| `custom_hooks: PipelineSwitchHook` (stage-2 augmentation cooldown) | MISSING | **DELIBERATELY NOT ADDED** -- the official stage-2 switch reduces the OFFICIAL RandomBBoxTransform's own scale/rotate ranges; this project already replaces that whole augmentation with EoMT/HRNet-matched values (PROTOCOL_LOCKED.md), so there is no equivalent "official range" to cool down between two stages |
| `max_epochs=420` | is 200 (configurable) | **KEPT at 200, documented, not silently accidental** -- 420 epochs was tuned for COCO's ~118k training images; this project's fetal datasets are 2-3 orders of magnitude smaller (Train ~100-1600 images per task), so a directly-copied epoch count has no principled basis either way. 200 is this project's own choice, adjustable via `--max-epochs` -- MUST be justified by the canary's own train/val convergence curve, not asserted a priori, per a reviewer's explicit requirement. |
| Pretrained checkpoint's own `checkpoint=` field | was a hardcoded URL | **FIXED (blocking issue)**: `model.init_weights()` would have loaded from this URL, completely independent of whatever local file `record_run_provenance.py` was separately hashing. `make_config()` now REQUIRES a local `pretrained_checkpoint_path` (resolved to an absolute path before being embedded, so the generated config is not sensitive to the working directory it was generated from) and embeds THAT path as `init_cfg.checkpoint`, so the file that gets loaded and the file that gets hashed/diffed are, by construction, the same file. |
| `--max-epochs`/`--val-interval` CLI flags | claimed to exist (the `max_epochs=420` row above already said "adjustable via `--max-epochs`"), but the argparse block never actually defined them | **FIXED (real, self-caught discrepancy, 2026-08-07)**: `make_config()` always accepted `max_epochs`/`val_interval` as Python keyword arguments with defaults, but the CLI (`if __name__ == "__main__":` block) never exposed a flag for either -- meaning `run_rtmpose_canary.sh`'s own `$MAX_EPOCHS` shell variable was silently ignored, and every invocation actually generated a 200-epoch config regardless of that variable's value. Found while wiring up `run_smoke_test.sh`, which genuinely needs `--max-epochs 1 --val-interval 1` to take effect. Both flags now exist and are wired through. |
| Warmup/cosine overlap for very short `max_epochs` | the round-5 warmup formula (`min(5, max(1, max_epochs//20))` epochs) always warms up for >=1 epoch, but `cosine_begin_epoch = max_epochs//2` is 0 or 1 for `max_epochs<4` -- mathematically guaranteed to overlap and trip the existing `SystemExit` guard | **HANDLED, not "fixed" (2026-08-07)**: unreachable for any real config in this project (every real run uses `max_epochs>=20`, comfortably satisfying `warmup_epochs<=5 < max_epochs//2`), so this is not a methodology bug in any existing result. Only reachable when a run genuinely wants `max_epochs<5` (`run_smoke_test.sh`'s `MAX_EPOCHS=1`) -- for that case only, warmup is disabled entirely (`warmup_end_iters=0`, with a loud `WARNING:` printed) rather than erroring, since LR-schedule shape is irrelevant to a smoke test's actual purpose. The generated config's own embedded assert was updated to tolerate `warmup_end_iters==0` as this deliberate degenerate case, not silently pass over a real overlap. |
"""

from __future__ import annotations

import json
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

# --- import custom transforms/hooks so their @*.register_module() run ---
# BOTH mechanisms used deliberately (round 6, review request for a more
# robust registration path than a bare top-level `import`): the plain
# imports below already work for this project's existing custom transforms
# (this is the same pattern `transforms.py`'s own TRANSFORMS registration
# has used since round 1 -- MMEngine's Config.fromfile() genuinely imports
# a Python-format config as a real module, executing top-level imports with
# real side effects, not a restricted AST-only variable extraction).
# `custom_imports` is ADDITIONALLY declared as MMEngine's own officially
# documented mechanism for this, in case some installed version's config-
# loading path is more restrictive than assumed -- belt and suspenders,
# not a sign the plain imports were known to be broken.
custom_imports = dict(
    imports=["transforms", "internal_val_hook"],
    allow_failed_imports=False,
)
import transforms  # noqa: F401,E402
import internal_val_hook  # noqa: F401,E402
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
# result was always going to be the final checkpoint.
#
# CORRECTED AGAIN, round 6 (review finding, VERIFIED against MMEngine's real
# Runner source -- this is a genuine, confirmed crash, not a hypothetical):
# MMEngine's Runner.__init__ enforces `val_dataloader, val_cfg, val_evaluator
# should be either all None or not None`, raising ValueError otherwise. The
# round-5 design (val_cfg=None while val_dataloader/val_evaluator were still
# populated dicts, so InternalFixedChannelNMEHook could read
# `runner.cfg.val_dataloader`) would have made Runner construction itself
# fail immediately, before a single training step ever ran. Fixed: the
# Train-only internal validation split now lives under a NON-standard name,
# `internal_val_dataloader`, that Runner's own from_cfg() never reads at
# all (so it cannot participate in the all-or-nothing check); `val_dataloader`
# and `val_evaluator` are both explicitly `None` alongside `val_cfg=None`,
# satisfying Runner's constraint by being genuinely, consistently absent.
# InternalFixedChannelNMEHook and live_preflight.py both read
# `runner.cfg.internal_val_dataloader` / `cfg.internal_val_dataloader`
# instead of `val_dataloader`.
internal_val_dataloader = dict(
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
val_dataloader = None
val_evaluator = None

# CORRECTED round 7 (review finding): round 6 kept test_dataloader/
# test_cfg/test_evaluator as a consistent non-None trio (required by
# Runner's own all-or-nothing constraint, verified against the same
# Runner source that governs val_dataloader/val_cfg/val_evaluator -- see
# val_cfg's own comment below) purely so run_inference.py could read
# `cfg.test_dataloader["dataset"]` directly, relying on a loud comment
# ("DO NOT RUN tools/test.py") to prevent PCKAccuracy from ever actually
# being invoked against a pipeline it was never verified safe for. A
# comment is not a structural guarantee. Fixed: the dataset this project's
# own inference step actually needs now lives under a non-standard key,
# `inference_dataloader`, that Runner.from_cfg() never reads at all (same
# technique as `internal_val_dataloader` above) -- test_dataloader,
# test_cfg, and test_evaluator are all now genuinely `None`, a legitimate
# all-None trio Runner accepts without complaint, and there is no longer
# any config-level entry point through which `tools/test.py`/`runner.test()`
# could invoke PCKAccuracy against this pipeline at all -- eliminated
# structurally, not just discouraged by comment. run_inference.py updated
# to read `cfg.inference_dataloader["dataset"]`.
inference_dataloader = dict(
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
test_dataloader = None
test_evaluator = None

train_cfg = dict(by_epoch=True, max_epochs={max_epochs})
# CORRECTED round 5 (review finding, real crash/silent-wrong-number risk,
# not just a style gap): val_cfg=None disables MMEngine's own automatic
# periodic validation loop entirely -- that loop calls model.val_step() ->
# model.predict(), which for a stock TopdownPoseEstimator typically needs
# bbox_center/bbox_scale metadata that this project's own PixelCentreResize
# deliberately never populates (same reason run_inference.py avoids
# predict() for final inference). Whether this would crash mid-training or
# silently produce a meaningless PCKAccuracy number was unverified without
# a live MMPose install -- rather than gambling on it, periodic internal
# monitoring is now done entirely by InternalFixedChannelNMEHook
# (internal_val_hook.py), which reuses run_inference.py's own verified-safe
# low-level decode path and computes the SAME fixed-channel NME formula as
# the final, authoritative evaluation -- directly comparable numbers, not
# PCKAccuracy's different metric computed a different way.
# test_cfg=None alongside test_dataloader=None/test_evaluator=None above --
# see the inference_dataloader comment for why (round 7 fix).
val_cfg = None
test_cfg = None

optim_wrapper = dict(
    type="OptimWrapper",
    # base_lr is ALREADY linearly scaled from the official recipe's
    # base_lr=4e-3 at base_batch_size=1024 (8 GPUs x train_batch_size=256)
    # down to this project's actual batch_size={batch_size}: 4e-3 *
    # ({batch_size}/1024) = {scaled_lr!r}. Fixed round 4 (real methodological
    # risk, not just fidelity): using the official 4e-3 unscaled at a batch
    # size 64x smaller than official's base_batch_size risked severe
    # training instability. Deliberately NOT also adding auto_scale_lr to
    # this config -- if `--auto-scale-lr` were ever passed to
    # tools/train.py on top of an already-scaled base_lr, it would scale
    # TWICE. The scaling is applied exactly once, here, explicitly.
    optimizer=dict(type="AdamW", lr={scaled_lr!r}, weight_decay=0.0),
    # Matches the official recipe exactly -- both were missing. clip_grad
    # was wrongly recorded as "not a real official setting" in round 3
    # (checked a stale/divergent config path by mistake, see module
    # docstring) -- fixed in round 4 after a reviewer caught this using the
    # correct projects/rtmpose/ source.
    clip_grad=dict(max_norm=35, norm_type=2),
    paramwise_cfg=dict(norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True),
)
# Cosine annealing begins at max_epochs // 2 in EPOCH units (MMEngine
# converts this to iterations at runtime using the actual dataloader
# length) -- matches the official recipe's own proportional shape (there:
# begin=210 of max_epochs=420). The LinearLR warmup below is fixed in
# ITERATION units by the official recipe (end=1000) regardless of dataset
# size; for this project's much smaller datasets that fixed value can
# still be RUNNING past the point cosine annealing has already started
# (round 4 finding, verified against the actual internal-train image count
# below, not assumed) -- e.g. ~100 images at batch_size=16 gives ~7
# iterations/epoch, so cosine's epoch-100 begin converts to ~700 iterations,
# LESS than the official recipe's fixed 1000-iteration warmup end.
# CORRECTED round 5: the first fix (warmup_end_iters = min(1000,
# cosine_begin_iters // 2)) only guaranteed no overlap, with no training-
# methodology basis -- for BPD's real numbers it gave 350 iterations of
# warmup out of ~1400 total (25% of all training), a real underfitting
# risk. Now a short, fixed-EPOCH warmup (<= 5 epochs) converted to
# iterations using the real image count -- ~35 iterations (2.5% of BPD's
# total training) instead of 350, much closer to the official recipe's own
# proportion (1000 of ~193,200 total iterations ~= 0.5%).
warmup_end_iters = {warmup_end_iters}
cosine_begin_epoch = {max_epochs} // 2
# warmup_end_iters == 0 means make_config.py itself already determined no
# warmup fits before cosine begins (only possible for a very short
# max_epochs, e.g. an engineering smoke test) and disabled it -- that is a
# deliberate, already-validated degenerate case, not a schedule overlap to
# re-check here.
assert warmup_end_iters == 0 or warmup_end_iters < cosine_begin_epoch * {iters_per_epoch}, (
    "LinearLR warmup would still be running when CosineAnnealingLR begins -- "
    "recompute warmup_end_iters/cosine_begin_epoch for this dataset size."
)
param_scheduler = [
    dict(type="LinearLR", start_factor=1e-5, by_epoch=False, begin=0, end=warmup_end_iters),
    dict(type="CosineAnnealingLR", eta_min={scaled_lr!r} * 0.05, begin=cosine_begin_epoch,
         end={max_epochs}, T_max={max_epochs} - cosine_begin_epoch,
         by_epoch=True, convert_to_iter_based=True),
]

# Round 5 fix: a reviewer's own recorded training-recipe fields, as a real
# top-level config value (not just local Python variables computed at
# generation time) so record_run_provenance.py can read them straight back
# via Config.fromfile() -- lets the canary report exactly what training
# setup was used, not just what make_config.py's own source claims it did.
training_recipe_summary = dict(
    n_train_images={n_train_images},
    batch_size={batch_size},
    iters_per_epoch={iters_per_epoch},
    effective_lr={scaled_lr!r},
    warmup_end_iters=warmup_end_iters,
    cosine_begin_epoch=cosine_begin_epoch,
    max_epochs={max_epochs},
)

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

# Sole source of periodic internal monitoring now that val_cfg=None
# disables MMEngine's own model.predict()-based val loop -- see val_cfg's
# own comment above and internal_val_hook.py's module docstring.
custom_hooks = [
    dict(type="InternalFixedChannelNMEHook",
         internal_val_ann={internal_val_ann!r},
         images_dir={images_dir!r},
         interval={val_interval}),
]

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
    file (see this file's own module docstring for the leak this closes).
    Resolved to an absolute path (round 4 fix) so the generated config is
    not sensitive to whatever working directory it happened to be
    generated from."""
    ckpt_path = Path(pretrained_checkpoint_path).resolve()
    if not ckpt_path.is_file():
        raise SystemExit(
            f"ERROR: --pretrained-checkpoint-path does not exist: "
            f"{pretrained_checkpoint_path}. Download it first (see "
            f"ENVIRONMENT.md) -- refusing to embed a checkpoint path into "
            f"the config that init_weights() cannot actually load from."
        )

    # Real LR scaling (round 4 fix): official base_lr=4e-3 is paired with
    # a base_batch_size of 1024 (auto_scale_lr in the official recipe);
    # applying it unscaled at this project's much smaller batch_size risks
    # severe instability. Scaled once, explicitly, here.
    scaled_lr = 4e-3 * (batch_size / 1024.0)

    # Real warmup/cosine-overlap fix (round 4): read the ACTUAL internal-train
    # COCO json's image count (already converted by the time make_config.py
    # runs in run_rtmpose_canary.sh) to compute real iterations/epoch, rather
    # than assuming the official recipe's fixed 1000-iteration warmup is
    # safe at any dataset size.
    internal_train_path = Path(internal_train_ann)
    if not internal_train_path.is_file():
        raise SystemExit(
            f"ERROR: --internal-train-ann does not exist yet: {internal_train_ann}. "
            f"Run convert_csv_to_coco.py for the internal-train split BEFORE "
            f"make_config.py -- the warmup/cosine schedule below needs the "
            f"real image count, not an assumption."
        )
    n_train_images = len(json.loads(internal_train_path.read_text(encoding="utf-8"))["images"])
    iters_per_epoch = max(1, -(-n_train_images // batch_size))  # ceil division
    cosine_begin_iters = (max_epochs // 2) * iters_per_epoch
    # CORRECTED round 5 (review finding): `cosine_begin_iters // 2` only
    # guaranteed no overlap, it had no training-methodology basis -- for
    # UCL BPD's real numbers (100 images, batch 16, ~7 iters/epoch, 200
    # epochs = ~1400 total iterations) it gave warmup_end_iters=350, i.e.
    # 25% of ALL training spent in warmup, a real underfitting risk, not
    # just a cosmetic schedule-shape mismatch. Replaced with a short,
    # fixed-EPOCH warmup (at most 5 epochs, or max_epochs//20 for shorter
    # runs) converted to iterations -- for BPD's numbers this gives
    # warmup_epochs=5, warmup_end_iters=35 (2.5% of training), much closer
    # to the official recipe's own proportion (1000 of ~193,200 total
    # iterations ~= 0.5%) than the previous formula ever was.
    warmup_epochs = min(5, max(1, max_epochs // 20))
    warmup_end_iters = warmup_epochs * iters_per_epoch
    if warmup_end_iters >= cosine_begin_iters:
        # Every REAL config this project generates uses max_epochs >= 20
        # (this project's own established minimum -- see make_config.py's
        # module docstring table), for which this branch is mathematically
        # unreachable: warmup_epochs <= 5 < max_epochs // 2 always holds.
        # This branch is only reachable for a deliberately very short run
        # (max_epochs < ~5, e.g. run_smoke_test.sh's MAX_EPOCHS=1), where
        # there is no room for both a warmup phase and a cosine phase --
        # disable warmup entirely rather than error, since LR-schedule
        # SHAPE is irrelevant to a smoke test's actual purpose (verifying
        # the Runner/Hook/checkpoint machinery integrates, not
        # convergence). NEVER silently rely on this for a real result --
        # max_epochs is always this project's own explicit choice, and a
        # loud warning is printed so a real short-schedule experiment
        # would not go unnoticed.
        if max_epochs < 5:
            print(
                f"WARNING: max_epochs={max_epochs} is too short for the normal "
                f"warmup(<=5 epochs)/cosine(begins at max_epochs//2) schedule "
                f"(cosine would begin at iteration {cosine_begin_iters}) -- "
                f"disabling warmup entirely (warmup_end_iters=0) for this run. "
                f"Only appropriate for a short engineering smoke test "
                f"(run_smoke_test.sh); NEVER for a real ablation/formal result."
            )
            warmup_epochs = 0
            warmup_end_iters = 0
        else:
            raise SystemExit(
                f"ERROR: computed warmup_end_iters ({warmup_end_iters}) >= "
                f"cosine_begin_iters ({cosine_begin_iters}) for n_train_images="
                f"{n_train_images}, batch_size={batch_size}, max_epochs={max_epochs} "
                f"-- the LinearLR warmup would still be running when "
                f"CosineAnnealingLR begins. Adjust batch_size/max_epochs or "
                f"revisit this formula, do not generate an overlapping schedule."
            )

    text = TEMPLATE.format(
        dataset=dataset, task=task, seed=seed,
        repo_root=repo_root or str(Path(__file__).resolve().parent),
        backbone_checkpoint=str(ckpt_path),
        data_root=data_root, images_dir=images_dir,
        internal_train_ann=internal_train_ann, internal_val_ann=internal_val_ann,
        test_ann=test_ann,
        batch_size=batch_size, max_epochs=max_epochs, val_interval=val_interval,
        scaled_lr=scaled_lr, iters_per_epoch=iters_per_epoch,
        warmup_end_iters=warmup_end_iters, n_train_images=n_train_images,
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
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-epochs", type=int, default=200,
                         help="NOTE: run_rtmpose_canary.sh's own $MAX_EPOCHS shell variable "
                              "was NOT actually wired to this flag before this was added -- "
                              "the generated config silently used the default (200) regardless "
                              "of that env var. Always pass --max-epochs explicitly if you need "
                              "anything other than the default; do not rely on the shell "
                              "variable alone to control it.")
    parser.add_argument("--val-interval", type=int, default=5,
                         help="CheckpointHook and InternalFixedChannelNMEHook both save/log at "
                              "this interval (epochs). Must be <= --max-epochs or no checkpoint "
                              "will ever be written -- e.g. a 1-epoch smoke run REQUIRES "
                              "--val-interval 1, not the default 5.")
    args = parser.parse_args()
    path = make_config(args.dataset, args.task, args.seed, args.data_root,
                        args.images_dir, args.internal_train_ann, args.internal_val_ann,
                        args.test_ann, args.pretrained_checkpoint_path,
                        args.work_dir, args.out,
                        batch_size=args.batch_size, max_epochs=args.max_epochs,
                        val_interval=args.val_interval)
    print(f"[OK] wrote {path}")
