# RTMPose-s environment

Per PROTOCOL_LOCKED.md's "Version and provenance gate": pin and record
everything below before spending server GPU time, in a dedicated venv
(mirroring `baseline_reproduction/ENVIRONMENT.md`'s "don't pollute the EoMT
env" convention).

## Install (server)

```bash
python3 -m venv /root/rtmpose_env --system-site-packages
source /root/rtmpose_env/bin/activate

pip install -U pip
pip install torch torchvision  # match the CUDA 12.8 build already used for EoMT/HRNet
pip install -U openmim
mim install "mmengine>=0.10.0"
mim install "mmcv>=2.1.0,<2.2.0"
pip install "mmdet>=3.2.0"   # only if a chosen augmentation transform imports mmdet
git clone --branch main https://github.com/open-mmlab/mmpose.git /root/mmpose
cd /root/mmpose
git rev-parse HEAD > /root/mmpose_commit.txt   # RECORD this before training a single sample
pip install -e .
```

**Do not train against a floating `main` checkout** (PROTOCOL_LOCKED.md).
Once the exact commit above is confirmed to build the canary successfully,
freeze it: `git checkout <that commit hash>` in future invocations, and
record the hash next to every result (mirrors the HRNet driver's own
`EXPECTED_COMMIT` hard-check pattern in `run_hrnet_512_fixed_5seed.sh`).

**Pin the "official recipe" source file, not just the MMPose commit**
(round 4 finding, real not hypothetical): MMPose's repo contains TWO
different, independently-maintained files both named
`rtmpose-s_8xb256-420e_coco-256x192.py` --
`configs/body_2d_keypoint/rtmpose/coco/rtmpose-s_8xb256-420e_coco-256x192.py`
(no `clip_grad`, last touched at commit `a910fd4c5684b0480f561efd703635d817944568`)
and `projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-s_8xb256-420e_coco-256x192.py`
(HAS `clip_grad=dict(max_norm=35, norm_type=2)`, last touched at commit
`94e15226a29a7067d9bb0cb7937b86e3c3fd0c8e`) -- a round-3 review checked the
former and wrongly concluded gradient clipping wasn't a real official
setting; a round-4 review caught this. **This project treats the
`projects/rtmpose/` path as the sole authoritative "official recipe"
source** (it is MMPose's own actively-maintained, dedicated RTMPose
project directory). If re-deriving "what does official do" in a future
session, fetch that exact path, at the exact commit you have installed
(`git log -1 --format=%H -- projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-s_8xb256-420e_coco-256x192.py`
inside the installed checkout), and do not assume the `configs/` path is
equivalent or up to date.

## Download the pretrained backbone checkpoint locally (required)

`record_run_provenance.py` now REQUIRES a local file path (not just the
URL baked into the generated config) to hash and diff against, since a
2026-08-06 review found the original version of that script never actually
verified the checkpoint loaded at all -- it only assumed `init_cfg` had
taken effect. Download it explicitly, set `PRETRAINED_CKPT_PATH` before
running the canary:

```bash
curl -L -o /root/cspnext-s_udp-aic-coco_210e-256x192-92f5a029_20230130.pth \
  https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/cspnext-s_udp-aic-coco_210e-256x192-92f5a029_20230130.pth
export PRETRAINED_CKPT_PATH=/root/cspnext-s_udp-aic-coco_210e-256x192-92f5a029_20230130.pth
```

## What must be verified once this is installed, before the canary

This project has no live MMPose environment to check these from; they were
inferred from MMPose's public documentation and the official RTMPose-s
config, not from running the actual installed package. Do not skip this
step.

**Round 5 fix (review finding): this checklist is now ENFORCED, not just
documented.** `run_rtmpose_canary.sh` calls `live_preflight.py` as a
mandatory, hard-fail gate right before training starts -- it exercises
items 6 (full non-square + codec round trip) and 3/4 (model build, decode
path) directly against the real installed MMPose, refusing to let the
canary proceed to training on any failure. Items 1, 2, 5, 7, and 8 below
still need a one-time human confirmation (import paths, channel order,
provenance JSON review, training-log inspection, architecture fidelity)
that `live_preflight.py` does not automate.

1. `python -c "import mmcv, mmengine, mmpose; print(mmcv.__version__, mmengine.__version__, mmpose.__version__)"` succeeds.
2. `rtmpose_reproduction/transforms.py` imports cleanly (`from mmcv.transforms import BaseTransform`, `from mmpose.registry import TRANSFORMS` — these exact import paths may have moved between mmpose/mmcv versions; fix imports here, not by downgrading to whatever version happens to match a stale example). Also confirm `results["img"]`'s actual channel order as produced by the configured `LoadImage` (BGR is the OpenMMLab default, but `FetalRotateScaleColorJitter._color_jitter`'s `assume_bgr=True` default must match reality or colour jitter silently uses the wrong channel-to-luminance mapping — see that class's own docstring caveat).
3. `make_config.py`'s generated config actually builds via `Config.fromfile()` + `MODELS.build(cfg.model)` without error. Specifically confirm `backbone.norm_cfg=dict(type="SyncBN")` (matching the official recipe, added in round 3) actually constructs on a single-GPU/non-distributed process — `SyncBatchNorm` sometimes requires `torch.distributed` to be initialised even for one process, depending on the installed MMEngine/MMCV version; if it errors, fall back to `BN` with the reason recorded in this file, not silently. Do NOT rely on log inspection alone for "was the backbone checkpoint actually loaded" — `record_run_provenance.py` (item 5 below) now performs a closed-loop check (see item 5), not just an eyeballed log. Separately (round 5 finding): `val_cfg=None` in the generated config means MMEngine's default `model.predict()`-based val loop never runs at all during training — confirm the training log shows NO validation phase and instead shows `InternalFixedChannelNMEHook`'s own `[InternalFixedChannelNMEHook] epoch=... mean_fixed_channel_nme_pct=...` log lines every `val_interval` epochs (`internal_val_hook.py`); if `PCKAccuracy`/`model.predict()` were ever re-enabled instead, first confirm live whether it crashes or silently produces a meaningless number given `PixelCentreResize` never sets bbox_center/bbox_scale.
4. `run_inference.py`'s two central assumptions must BOTH be confirmed against the installed source, not assumed: (a) `model.data_preprocessor({"inputs": [...], "data_samples": [...]}, False)` is the correct call contract for a single manually-collated sample (read the installed `PoseDataPreprocessor`/`BaseDataPreprocessor` source — the exact dict keys and whether `inputs` should be a list of per-sample tensors or an already-stacked batch may differ by version); (b) `model.head.decode(...)` or the codec's own `.decode()` returns 512-space coordinates, not already bbox-inverse-transformed — read `mmpose/models/heads/coord_cls_heads/rtmcc_head.py` (or wherever RTMCCHead lives in the installed version) directly.
5. Pretrained backbone checksum, load-scope, and actual (not published) parameter counts: run `record_run_provenance.py` right after `make_config.py` generates the config (wired into `run_rtmpose_canary.sh`'s step 4b, `PRETRAINED_CKPT_PATH` set per the download step above). CORRECTED round 3 (a real gap in round 2's own verification): `make_config.py` now embeds the LOCAL checkpoint path directly as `backbone.init_cfg.checkpoint` (not a URL), so the weights `model.init_weights()` actually loads and the file this script hashes/diffs are, by construction, the same file — `record_run_provenance.py` asserts this equality explicitly and fails loudly if it ever doesn't hold. It also now requires an EXACT value match (not just "changed from random init") between every backbone parameter and the checkpoint's own tensor for that key, failing loudly on any missing, unexpected, or value-mismatched key. Treat the JSON's `verified_pretrained_load_actually_happened` field as the authority (it would not be `true` unless the script's own asserts already passed), not eyeballed logs — and record the REAL total/trainable/frozen parameter counts for this project's actual out_channels=2, 512x512 config (NOT the official RTMPose-s paper's ~5.47M figure, which is for the COCO 256x192, 17-keypoint config and must never be cited as this project's own number).
6. **Full non-square, codec-level round trip -- NOW AUTOMATED by `live_preflight.py` (round 5), still needs its first live run to actually pass**: `test_geometry.py`/`test_fetal_augment.py` only test the pure-Python geometry/reorder logic; they do NOT exercise MMPose's own SimCC codec `encode()`/`decode()` round trip. `live_preflight.py`'s first check does exactly this end-to-end (synthetic non-square image -> `convert_csv_to_coco.py`-equivalent COCO annotation -> the real `train_pipeline` -> real codec `encode()`/`decode()` -> `geometry.to_image_space()` -> compare to the original coordinates), printing the max absolute pixel error split into the PURE geometric contribution (already ~0, proven by `test_geometry.py`) and the SimCC 1024-bin quantisation contribution (previously entirely unmeasured). `live_preflight.py`'s own `encode()`/`decode()` dict-key names (`keypoint_x_labels`/`keypoint_y_labels`) are inferred from SimCCLabel's documented interface, not confirmed live -- if this specific step is what fails first, fix the key names against the installed `mmpose/codecs/simcc_label.py` source, that is exactly what this gate exists to catch.
7. **No test loop during training**: `make_config.py`'s generated config includes a `test_dataloader`/`test_evaluator` (needed for `run_inference.py`'s own `DATASETS.build(cfg.test_dataloader["dataset"])` call after training), but `tools/train.py` conventionally only runs `train`/`val` phases, never `test` — confirm this directly against the actual training log for the installed MMPose version (no "Testing" phase should appear anywhere in it) rather than assuming the convention holds for every release.
8. **CSPNeXt architecture/recipe fidelity** (round 3 review, verified against the real fetched `rtmpose-s_8xb256-420e_coco-256x192.py`): confirm the generated config's `backbone` section (`_scope_="mmdet"`, `expand_ratio=0.5`, `norm_cfg=SyncBN`) actually resolves and builds a CSPNeXt-s whose channel widths match the pretrained checkpoint's own shapes — a mismatch here would show up as `record_run_provenance.py`'s shape-mismatch check failing (item 5), not as a separate silent failure. Deliberately NOT matched to the official recipe, with reasons recorded in `make_config.py`'s own module docstring: `EMAHook`, the stage-2 `PipelineSwitchHook` augmentation cooldown, `auto_scale_lr`, and 420-epoch training — these are dataset-scale/methodology decisions, not accidental omissions, and should not be "fixed" to match official without a fresh methodological review.

## Seed control

Set via `randomness=dict(seed=X, deterministic=True)` in the generated
config (mmengine's `Runner` reads this directly) plus
`sampler=dict(..., seed=X)` on the train dataloader. This is a built-in
mmengine mechanism (unlike HRNet's upstream code, which needed an external
patch, `baseline_reproduction/apply_controlled_seed_patch.py`, because it
predates any such interface) -- confirm the installed mmengine version
actually seeds CUDA/DataLoader-worker RNGs from this single field before
trusting "seed 42" to mean the same thing it does for EoMT/HRNet; if it
does not, an equivalent explicit patch will be needed here too.
