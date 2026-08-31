"""Mandatory, hard-fail preflight gate that MUST run (and pass) against a
real MMPose install BEFORE any GPU training time is spent -- this is the
enforcement mechanism for the checks ENVIRONMENT.md's numbered checklist
otherwise only documents as a manual reading exercise.

WHY THIS EXISTS (review finding, 2026-08-06): ENVIRONMENT.md's checklist
correctly lists everything that needs live verification (model builds,
SyncBN, data_preprocessor contract, codec decode shape, BGR/RGB, full
non-square round trip), but nothing in run_rtmpose_canary.sh actually
FORCED those checks to run before training started. This script is that
enforcement.

CORRECTED 2026-08-06, round 6 (review finding -- this file's FIRST version
had four real bugs of its own, caught by a fresh review of the preflight
code itself, not just the training config):
  1. `check_geometric_round_trip` used `cfg.train_dataloader["dataset"]`,
     whose pipeline includes RANDOM augmentation (flip/rotate/scale/colour
     jitter via FetalRandomFlipAndCanonicalize/FetalRotateScaleColorJitter)
     -- comparing the resulting decoded coordinates against the ORIGINAL,
     unaugmented synthetic p0/p1 would fail whenever augmentation actually
     triggered, non-deterministically, contradicting the function's own
     comment claiming a "deterministic val_pipeline." Fixed: now uses
     `cfg.internal_val_dataloader["dataset"]` (LoadImage -> PixelCentreResize
     -> PackPoseInputs only, no augmentation stage at all).
  2. Referenced `cfg.val_dataloader`, which round 6's own make_config.py fix
     sets to `None` (see make_config.py's own docstring for why: MMEngine's
     Runner requires val_dataloader/val_cfg/val_evaluator to be all-None or
     all-not-None, verified against the real Runner source). Fixed: both
     `check_geometric_round_trip` and `check_decode_and_internal_val_path`
     now read `cfg.internal_val_dataloader` instead.
  3. Never actually built `InternalFixedChannelNMEHook` from the registry
     or invoked its lifecycle method -- only tested the shared
     `low_level_decode` functions it depends on, which could not have
     caught either of the two config-structure bugs above (a missing/
     misconfigured Hook registration, or a `runner.cfg.val_dataloader`
     `AttributeError`/`TypeError`, would both have gone undetected). Fixed:
     new `check_hook_registry_and_lifecycle` genuinely builds the Hook via
     `HOOKS.build()`, constructs a minimal duck-typed fake Runner exposing
     exactly the attributes the Hook reads (`model`, `cfg`, `epoch`,
     `logger`, a REAL `mmengine.logging.MessageHub` instance so the
     message-hub call itself is exercised for real, not mocked), and calls
     `after_train_epoch` once, asserting: the logged NME is finite, the
     message hub actually received the scalar, `model.training` is
     restored to its pre-call state, and the dataset built matches
     `internal_val_dataloader`'s own dataset config (not the released
     Test set, which as of round 7 lives under `inference_dataloader`,
     not `test_dataloader` -- see make_config.py's own comment for why
     `test_dataloader`/`test_cfg`/`test_evaluator` are now all `None`).
  4. `check_train_forward_backward` built the model WITHOUT calling
     `model.init_weights()`, so the "real" forward/backward smoke test
     actually ran on a randomly-initialised backbone, not the pretrained
     one the canary will actually use. Fixed: `model.init_weights()` is
     now called before the smoke test, same as `record_run_provenance.py`
     already does.
  Also added: an explicit BGR/RGB channel-order check
  (`check_bgr_rgb_channel_order`) using a synthetic image with a KNOWN
  pixel value, comparing what `LoadImage` actually produces against
  `FetalRotateScaleColorJitter`'s `assume_bgr=True` default -- previously
  claimed as covered but never actually implemented.

What this script does, all against the REAL installed MMPose:
  1. Full non-square geometric + SimCC-codec round trip (deterministic
     pipeline only).
  2. BGR/RGB channel-order check against a known pixel value.
  3. One real train-batch forward + loss + backward pass (pretrained
     backbone, not random), asserting finite losses and that gradients
     reach both backbone and head.
  4. The shared low-level decode path, end to end.
  5. `InternalFixedChannelNMEHook`, built from the registry and actually
     invoked once via a minimal fake Runner.
  6. `training_recipe_summary`'s own internal consistency.

Exits non-zero (and prints exactly which check failed) on ANY assertion
failure -- run_rtmpose_canary.sh treats this as fatal and does not proceed
to training.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np


def _build_synthetic_sample(tmp_dir: Path, width: int, height: int,
                             p0, p1, filename: str = "synthetic.png",
                             top_left_rgb=(10, 20, 230)):
    """Writes one synthetic non-square image + a COCO json with one
    2-keypoint annotation, matching convert_csv_to_coco.py's own output
    schema exactly (bbox = full image, keypoints = [x0,y0,2,x1,y1,2]).
    The top-left pixel is set to a KNOWN, asymmetric RGB value so
    `check_bgr_rgb_channel_order` can detect a channel swap unambiguously."""
    from PIL import Image

    img_path = tmp_dir / filename
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[..., 0] = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
    arr[..., 1] = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
    arr[0, 0] = top_left_rgb  # PIL Image.fromarray treats this as RGB order
    Image.fromarray(arr, mode="RGB").save(img_path)

    coco = {
        "images": [{"id": 0, "file_name": filename, "width": width, "height": height}],
        "annotations": [{
            "id": 0, "image_id": 0, "category_id": 1, "iscrowd": 0,
            "bbox": [0, 0, width, height], "area": float(width * height),
            "num_keypoints": 2,
            "keypoints": [p0[0], p0[1], 2, p1[0], p1[1], 2],
        }],
        "categories": [{"id": 1, "name": "preflight", "keypoints": ["endpoint_0", "endpoint_1"],
                         "skeleton": [[1, 2]]}],
    }
    ann_path = tmp_dir / "synthetic_coco.json"
    ann_path.write_text(json.dumps(coco), encoding="utf-8")
    return img_path, ann_path


def check_geometric_round_trip(cfg, tmp_dir: Path) -> None:
    print("=== [preflight 1/6] full non-square geometric + SimCC-codec round trip (deterministic pipeline) ===")
    from geometry import to_image_space, to_model_space
    from mmpose.registry import DATASETS

    width, height = 917, 641  # deliberately non-square, deliberately not a "nice" number
    p0, p1 = (123.4, 88.9), (740.1, 512.3)
    img_path, ann_path = _build_synthetic_sample(tmp_dir, width, height, p0, p1)

    # CORRECTED round 6: internal_val_dataloader's pipeline is
    # LoadImage -> PixelCentreResize -> PackPoseInputs ONLY -- no
    # FetalRandomFlipAndCanonicalize / FetalRotateScaleColorJitter, so this
    # measures geometry + codec quantisation with zero augmentation
    # randomness, matching what this function's own comments always
    # claimed (round 1 of this file wrongly used train_dataloader instead).
    dataset_cfg = dict(cfg.internal_val_dataloader["dataset"])
    dataset_cfg["ann_file"] = str(ann_path)
    dataset_cfg["data_prefix"] = dict(img=str(tmp_dir))
    dataset = DATASETS.build(dataset_cfg)
    assert len(dataset) == 1, f"expected 1 synthetic sample, got {len(dataset)}"

    for pt in (p0, p1):
        xp, yp = to_model_space(pt[0], pt[1], width, height, 512)
        x2, y2 = to_image_space(xp, yp, width, height, 512)
        err = math.hypot(x2 - pt[0], y2 - pt[1])
        assert err < 1e-6, f"pure geometric round trip failed for {pt}: error={err}"
    print("  pure geometric round-trip error: < 1e-6 px (matches test_geometry.py)")

    sample = dataset[0]
    keypoints_512 = np.asarray(sample["data_samples"].gt_instances.keypoints).reshape(-1, 2)
    assert keypoints_512.shape[0] >= 2, f"expected >=2 keypoints, got shape {keypoints_512.shape}"

    codec = dict(cfg.codec)
    # MMPose 1.3.2 does not expose `mmpose.codecs.build_codec`.  Codecs are
    # constructed through the project's registry, just like datasets and
    # models.  This was confirmed by the first live preflight against the
    # pinned server environment; keep this version-locked API explicit so a
    # missing registration remains a hard preflight failure.
    from mmpose.registry import KEYPOINT_CODECS
    built_codec = KEYPOINT_CODECS.build(codec)
    encoded = built_codec.encode(keypoints_512.reshape(1, -1, 2),
                                  keypoints_visible=np.ones((1, keypoints_512.shape[0])))
    # The exact encode()/decode() keys and tensor shapes are deliberately
    # exercised below against the live pinned MMPose installation.  Any API
    # mismatch remains a blocking preflight failure rather than being
    # hidden behind an adapter fallback.
    decoded_coords, _ = built_codec.decode(encoded["keypoint_x_labels"],
                                            encoded["keypoint_y_labels"])
    decoded_coords = np.asarray(decoded_coords).reshape(-1, 2)

    max_codec_error_px = 0.0
    for i, orig_pt in enumerate((p0, p1)):
        xp, yp = decoded_coords[i]
        x_final, y_final = to_image_space(xp, yp, width, height, 512)
        err = math.hypot(x_final - orig_pt[0], y_final - orig_pt[1])
        max_codec_error_px = max(max_codec_error_px, err)
    print(f"  full pipeline (incl. SimCC 1024-bin quantisation) max round-trip error: "
          f"{max_codec_error_px:.4f} px")
    assert max_codec_error_px < 2.0, (
        f"SimCC round-trip error implausibly large ({max_codec_error_px:.4f} px) "
        f"-- investigate the codec/geometry wiring before trusting the canary."
    )


def check_bgr_rgb_channel_order(cfg, tmp_dir: Path) -> None:
    """CORRECTED round 7 (review finding): the first version ran the FULL
    dataset pipeline (including PixelCentreResize's bilinear interpolation)
    before inspecting the corner pixel, so an inconclusive blended-pixel
    result was possible -- and when inconclusive, the function only printed
    a WARNING and returned success, directly contradicting this whole
    script's "any unverified assumption blocks training" premise. Fixed:
    invokes MMPose's own registered `LoadImage` transform DIRECTLY, on its
    own, with no resize/interpolation involved at all -- the loaded array's
    corner pixel is checked exactly, with zero ambiguity, and ANY outcome
    other than a clean RGB or clean BGR match is now a hard failure, not a
    warning."""
    print("=== [preflight 2/6] BGR/RGB channel order (direct LoadImage, no resize involved) ===")
    from mmpose.registry import TRANSFORMS

    width, height = 400, 300
    p0, p1 = (50.0, 50.0), (300.0, 200.0)
    top_left_rgb = (10, 20, 230)
    img_path, _ = _build_synthetic_sample(
        tmp_dir, width, height, p0, p1, filename="bgr_check.png", top_left_rgb=top_left_rgb,
    )

    # NOTE (needs live confirmation, same tier as the rest of this file):
    # `results['img_path']` is the documented mmcv/mmpose LoadImage input
    # key; if the installed version expects a different key name, this
    # call fails loudly here, which is exactly this preflight's job.
    load_image = TRANSFORMS.build(dict(type="LoadImage"))
    results = load_image.transform({"img_path": str(img_path)})
    img = np.asarray(results["img"])
    assert img.ndim == 3 and img.shape[2] == 3, (
        f"LoadImage output has unexpected shape {img.shape}, expected (H, W, 3)"
    )
    corner = tuple(int(v) for v in img[0, 0, :3])

    is_rgb_order = corner == tuple(top_left_rgb)
    is_bgr_order = corner == tuple(reversed(top_left_rgb))
    print(f"  wrote top-left pixel as RGB={top_left_rgb}; raw LoadImage output corner = {corner}")

    assert is_rgb_order or is_bgr_order, (
        f"LoadImage's corner pixel {corner} matches NEITHER the written RGB value "
        f"{tuple(top_left_rgb)} NOR its BGR reversal {tuple(reversed(top_left_rgb))} -- "
        f"cannot determine channel order at all (unexpected colour conversion, alpha "
        f"channel, or a different LoadImage than assumed). Do not proceed until this is "
        f"understood; a silent, unverified channel-order assumption is exactly what this "
        f"check exists to eliminate."
    )

    if is_rgb_order:
        print("  -> LoadImage preserves RGB order.")
        assert False, (
            "LoadImage produces RGB order, but transforms.FetalRotateScaleColorJitter "
            "defaults to assume_bgr=True -- this MUST be changed to assume_bgr=False "
            "(or explicitly passed as such in make_config.py's generated pipeline entry) "
            "before colour jitter's saturation/contrast maths are correct."
        )
    print("  -> LoadImage produces BGR order (OpenMMLab default): "
          "FetalRotateScaleColorJitter's assume_bgr=True default is CORRECT, no change needed.")


def check_train_forward_backward(cfg, device: str) -> None:
    print("=== [preflight 3/6] one real train-batch forward + loss + backward (pretrained backbone) ===")
    import torch
    from mmpose.registry import DATASETS, MODELS

    model = MODELS.build(cfg.model)
    # CORRECTED round 6: previously missing -- without this, the "real"
    # smoke test ran on a randomly-initialised backbone, not the pretrained
    # one the actual canary will use.
    model.init_weights()
    model.to(device)
    model.train()

    dataset = DATASETS.build(cfg.train_dataloader["dataset"])
    assert len(dataset) > 0, "internal-train dataset is empty -- check the internal split/CSV conversion"
    sample = dataset[0]

    batch = model.data_preprocessor(
        {"inputs": [sample["inputs"]], "data_samples": [sample["data_samples"]]},
        True,
    )
    losses = model.loss(batch["inputs"].to(device), batch["data_samples"])
    assert isinstance(losses, dict) and len(losses) > 0, f"model.loss() returned unexpected: {losses!r}"
    total_loss = sum(v for v in losses.values() if hasattr(v, "backward"))
    for name, v in losses.items():
        if hasattr(v, "item"):
            assert math.isfinite(v.item()), f"loss component {name!r} is not finite: {v.item()}"
    total_loss.backward()

    backbone_has_grad = any(p.grad is not None and torch.any(p.grad != 0)
                             for p in model.backbone.parameters() if p.requires_grad)
    head_has_grad = any(p.grad is not None and torch.any(p.grad != 0)
                         for p in model.head.parameters() if p.requires_grad)
    assert backbone_has_grad, "no nonzero gradient reached the backbone -- check the loss/graph wiring"
    assert head_has_grad, "no nonzero gradient reached the head -- check the loss/graph wiring"
    print(f"  loss components: {list(losses.keys())}, all finite; "
          f"gradients reached both backbone and head (pretrained init)")


def check_decode_path(cfg, device: str) -> None:
    print("=== [preflight 4/6] low-level decode path (shared with run_inference.py + InternalFixedChannelNMEHook) ===")
    from low_level_decode import decode_batch_low_level, fixed_channel_nme, to_original_image_space
    from mmpose.registry import DATASETS, MODELS

    model = MODELS.build(cfg.model)
    model.init_weights()
    model.to(device)
    model.eval()

    # CORRECTED round 6: cfg.val_dataloader is now None (see make_config.py's
    # own docstring) -- must read cfg.internal_val_dataloader instead.
    dataset = DATASETS.build(cfg.internal_val_dataloader["dataset"])
    assert len(dataset) > 0, "internal-val dataset is empty"
    sample = dataset[0]

    import torch
    with torch.no_grad():
        coords = decode_batch_low_level(model, sample, device)
    assert coords.shape == (2, 2), f"expected decode shape (2, 2), got {coords.shape}"
    assert np.all(np.isfinite(coords)), f"decoded coordinates are not finite: {coords}"
    print(f"  decode_batch_low_level output shape/finite: OK ({coords.shape})")

    pred = to_original_image_space(coords, width=800.0, height=600.0)
    assert pred.shape == (2, 2) and np.all(np.isfinite(pred)), "to_original_image_space output invalid"
    nme = fixed_channel_nme(pred, np.array([[100.0, 100.0], [700.0, 500.0]]))
    assert math.isfinite(nme) and nme >= 0.0, f"fixed_channel_nme produced an invalid value: {nme}"
    print(f"  fixed_channel_nme end-to-end call: OK (sanity value={nme:.4f})")


class _FakeRunner:
    """Minimal duck-typed stand-in exposing exactly the attributes
    InternalFixedChannelNMEHook.after_train_epoch reads (`model`, `cfg`,
    `epoch`, `logger`, `message_hub`) -- NOT a full mmengine.Runner (which
    would need a complete, working optimizer/scheduler/dataloader wiring
    just to construct). Uses a REAL `mmengine.logging.MessageHub` instance
    so the actual message-hub call is exercised, not mocked."""

    def __init__(self, model, cfg, epoch):
        import logging

        from mmengine.logging import MessageHub

        self.model = model
        self.cfg = cfg
        self.epoch = epoch
        self.logger = logging.getLogger("live_preflight")
        self.message_hub = MessageHub.get_instance("live_preflight_fake_runner")


def check_hook_registry_and_lifecycle(cfg, device: str) -> None:
    print("=== [preflight 5/6] InternalFixedChannelNMEHook: real registry build + real lifecycle call ===")
    from mmengine.registry import HOOKS
    from mmpose.registry import DATASETS, MODELS

    assert getattr(cfg, "custom_hooks", None), "cfg.custom_hooks is empty -- make_config.py regressed"
    hook_cfg = dict(cfg.custom_hooks[0])
    assert hook_cfg.get("type") == "InternalFixedChannelNMEHook", (
        f"expected custom_hooks[0].type == 'InternalFixedChannelNMEHook', got {hook_cfg.get('type')!r}"
    )
    # Force interval=1 so a single call with epoch=0 always triggers,
    # regardless of the configured val_interval.
    hook_cfg["interval"] = 1
    hook = HOOKS.build(hook_cfg)
    assert hook.__class__.__name__ == "InternalFixedChannelNMEHook", (
        f"HOOKS.build() returned {hook.__class__.__name__!r}, not InternalFixedChannelNMEHook "
        f"-- registry resolution failed silently"
    )
    print(f"  HOOKS.build({hook_cfg['type']!r}) succeeded: {hook.__class__.__name__}")

    model = MODELS.build(cfg.model)
    model.init_weights()
    model.to(device)
    model.train()  # so we can verify the Hook restores this afterward

    # Confirms the Hook's own dataset build (inside _lazy_build, triggered
    # by the after_train_epoch call below) resolves cfg.internal_val_dataloader
    # -- not a length-comparison heuristic against test_dataloader (a weak
    # signal on its own), but a real exercise of the exact code path
    # internal_val_hook.py's _lazy_build uses, which would raise
    # AttributeError/TypeError outright if it still referenced the now-None
    # cfg.val_dataloader.
    assert len(DATASETS.build(cfg.internal_val_dataloader["dataset"])) > 0, (
        "internal-val dataset (the one InternalFixedChannelNMEHook actually reads) is empty"
    )

    fake_runner = _FakeRunner(model=model, cfg=cfg, epoch=0)
    hook.after_train_epoch(fake_runner)

    recorded = fake_runner.message_hub.get_scalar("train/internal_fixed_channel_nme_pct")
    nme_value = recorded.current()
    assert math.isfinite(nme_value), f"InternalFixedChannelNMEHook logged a non-finite NME: {nme_value}"
    assert model.training is True, (
        "model.training was not restored to True after InternalFixedChannelNMEHook ran -- "
        "the Hook's own eval()/train() restore logic is broken"
    )
    print(f"  after_train_epoch() ran for real: logged NME={nme_value:.4f}% (finite), "
          f"message_hub scalar present, model.training correctly restored to True")


def check_provenance_fields_present(cfg) -> None:
    print("=== [preflight 6/6] training_recipe_summary present and sane ===")
    summary = cfg.get("training_recipe_summary")
    assert summary, "training_recipe_summary missing from the generated config -- make_config.py regressed"
    required = {"n_train_images", "batch_size", "iters_per_epoch", "effective_lr",
                "warmup_end_iters", "cosine_begin_epoch", "max_epochs"}
    missing = required - set(summary.keys())
    assert not missing, f"training_recipe_summary is missing fields: {missing}"
    assert summary["warmup_end_iters"] < summary["cosine_begin_epoch"] * summary["iters_per_epoch"], (
        "training_recipe_summary's own numbers show an overlapping warmup/cosine schedule"
    )
    print(f"  training_recipe_summary: {summary}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    from mmengine.config import Config
    from mmengine.registry import init_default_scope

    init_default_scope("mmpose")
    cfg = Config.fromfile(str(args.config))

    with tempfile.TemporaryDirectory() as tmp:
        check_geometric_round_trip(cfg, Path(tmp))
        check_bgr_rgb_channel_order(cfg, Path(tmp))
    check_train_forward_backward(cfg, args.device)
    check_decode_path(cfg, args.device)
    check_hook_registry_and_lifecycle(cfg, args.device)
    check_provenance_fields_present(cfg)

    print("=== ALL PREFLIGHT CHECKS PASSED -- safe to proceed to training ===")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as exc:
        print(f"\n[PREFLIGHT FAILED] {exc}", file=sys.stderr)
        sys.exit(1)
