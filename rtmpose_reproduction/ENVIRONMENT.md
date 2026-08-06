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

## What must be verified once this is installed, before the canary

This project has no live MMPose environment to check these from; they were
inferred from MMPose's public documentation and the official RTMPose-s
config, not from running the actual installed package. Do not skip this
step:

1. `python -c "import mmcv, mmengine, mmpose; print(mmcv.__version__, mmengine.__version__, mmpose.__version__)"` succeeds.
2. `rtmpose_reproduction/transforms.py` imports cleanly (`from mmcv.transforms import BaseTransform`, `from mmpose.registry import TRANSFORMS` — these exact import paths may have moved between mmpose/mmcv versions; fix imports here, not by downgrading to whatever version happens to match a stale example).
3. `make_config.py`'s generated config actually builds via `Config.fromfile()` + `MODELS.build(cfg.model)` without error, and the backbone log shows the CSPNeXt-s checkpoint's `backbone.`-prefixed keys were loaded (not silently skipped) — mirrors how the HRNet driver's `check_checkout()` confirms a patch actually applied rather than assuming it.
4. `run_inference.py`'s central assumption (`model.head.decode(...)` or the codec's own `.decode()` returns 512-space coordinates, not already bbox-inverse-transformed) — read this project's own installed `mmpose/models/heads/coord_cls_heads/rtmcc_head.py` (or wherever RTMCCHead lives in the installed version) and confirm directly, do not assume from this repo's comments alone.
5. Pretrained backbone checksum: `sha256sum` the downloaded `cspnext-s_udp-aic-coco_...-256x192-....pth` and record it next to the MMPose commit (same rigor as `baseline_reproduction`'s `EXPECTED_PRETRAINED_SHA256` gate).

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
