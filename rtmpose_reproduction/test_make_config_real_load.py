"""Regression test for a real, blocking bug found by actually running
live_preflight.py's pipeline against a real MMEngine install (2026-08-07,
round 11): `Config.fromfile()` treats any top-level import of a non-builtin
module in a .py config as a trigger for MMEngine's "lazy import" config
mode (verified against MMEngine 0.10.7's own source,
`mmengine/config/config.py`'s `Config._is_lazy_import` +
`mmengine/config/utils.py`'s `_is_builtin_module`: ANY module that does not
resolve to `sys.builtin_module_names` or Python's own install root is
non-builtin). The generated config used to do exactly this --
`import transforms`, `import internal_val_hook`,
`from fetal_dataset_info import FETAL_DATASET_INFO`, and
`from dod_vectors import get_d_vect` were all top-level imports inside the
GENERATED config text, each independently sufficient to trigger lazy mode.
In lazy mode, imported names become `LazyObject`/`LazyAttr` PROXIES, not
the real objects, until a Runner actually builds the config -- so the
generated config's own `d_vect = get_d_vect(...)` line (calling a
not-yet-real proxy immediately, at parse time) could not work.

Fixed in make_config.py: `get_d_vect(dataset, task)` and
`FETAL_DATASET_INFO` are now resolved in make_config.py's OWN module scope
(a normal script, genuinely importing these modules for real) and embedded
into the generated config as plain LITERAL values (a tuple, a dict) --
no import, no function call, inside the generated config text at all.
`custom_imports = dict(...)` (a plain dict literal, not an AST
Import/ImportFrom node) is now the SOLE mechanism registering
`transforms`/`internal_val_hook`'s custom Transform/Hook classes.

THIS TEST REQUIRES A REAL, LIVE MMPOSE ENVIRONMENT TO RUN (same tier as
live_preflight.py, not test_geometry.py/test_fetal_augment.py's pure-Python
tier) -- `Config.fromfile()` unconditionally resolves `custom_imports`
internally (regardless of lazy/non-lazy mode), which imports
`transforms.py`, which itself requires `mmcv`/`mmpose` to be importable.
There is no meaningful way to run this test without them; it will raise
ImportError (not silently skip) if the environment is wrong -- run it
where `PY` in run_rtmpose_canary.sh/run_smoke_test.sh already points.

Run directly: python rtmpose_reproduction/test_make_config_real_load.py
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

from dod_vectors import get_d_vect
from fetal_dataset_info import FETAL_DATASET_INFO
from make_config import make_config

_REPO_ROOT = Path(__file__).resolve().parent


def _write_fake_coco_json(path: Path, n_images: int = 20) -> None:
    """A minimal, self-contained fake COCO json -- make_config() only reads
    len(data["images"]) from it (for the warmup/cosine schedule's real
    image-count requirement), so its contents need not be realistic beyond
    that, and no real dataset/images are needed to exercise this bug."""
    data = {
        "images": [{"id": i, "file_name": f"img{i}.jpg", "width": 300, "height": 600}
                   for i in range(n_images)],
        "annotations": [{"id": i, "image_id": i, "category_id": 1,
                          "keypoints": [10, 10, 2, 90, 10, 2]}
                        for i in range(n_images)],
        "categories": [{"id": 1, "name": "fetal_two_endpoint",
                         "keypoints": ["endpoint_0", "endpoint_1"]}],
    }
    path.write_text(json.dumps(data), encoding="utf-8")


def test_generated_config_avoids_lazy_import_and_resolves_d_vect_literally():
    tmp = Path(tempfile.mkdtemp())
    try:
        coco_dir = tmp / "coco"
        coco_dir.mkdir()
        for name in ("train.json", "val.json", "test.json"):
            _write_fake_coco_json(coco_dir / name)
        fake_ckpt = tmp / "fake_ckpt.pth"
        fake_ckpt.write_bytes(b"not a real checkpoint, only existence is checked here")

        config_path = make_config(
            dataset="UCL", task="BPD", seed=42,
            data_root=str(tmp), images_dir=str(tmp / "images"),
            internal_train_ann=str(coco_dir / "train.json"),
            internal_val_ann=str(coco_dir / "val.json"),
            test_ann=str(coco_dir / "test.json"),
            pretrained_checkpoint_path=str(fake_ckpt),
            work_dir=str(tmp / "work"),
            out_path=tmp / "config.py",
            repo_root=str(_REPO_ROOT),
        )

        from mmengine.config import Config

        is_lazy = Config._is_lazy_import(str(config_path))
        assert not is_lazy, (
            f"{config_path} was parsed in MMEngine's LAZY IMPORT mode -- a "
            f"top-level import of a non-builtin module has crept back into "
            f"the generated config (see this file's own module docstring, "
            f"and make_config.py's TEMPLATE 'LAZY IMPORT' comment)."
        )

        cfg = Config.fromfile(str(config_path))

        expected_d_vect = get_d_vect("UCL", "BPD")
        assert cfg.d_vect == expected_d_vect, (
            f"cfg.d_vect={cfg.d_vect} != get_d_vect('UCL','BPD')={expected_d_vect} "
            f"-- d_vect resolution in make_config.py itself is wrong, or the "
            f"generated config is no longer embedding it as a plain literal."
        )
        assert isinstance(cfg.d_vect, tuple), (
            f"cfg.d_vect should be a plain tuple literal, got {type(cfg.d_vect)} "
            f"-- if this is a LazyObject/LazyAttr, the lazy-import bug has returned."
        )

        assert cfg.FETAL_DATASET_INFO == FETAL_DATASET_INFO, (
            "cfg.FETAL_DATASET_INFO does not match the real "
            "fetal_dataset_info.FETAL_DATASET_INFO dict -- either the literal "
            "embedding broke, or fetal_dataset_info.py changed after this "
            "config was generated."
        )
        assert isinstance(cfg.FETAL_DATASET_INFO, dict), (
            f"cfg.FETAL_DATASET_INFO should be a plain dict literal, got "
            f"{type(cfg.FETAL_DATASET_INFO)}"
        )

        # custom_imports resolving without error (this line only runs if
        # Config.fromfile() above already succeeded) confirms transforms.py/
        # internal_val_hook.py's @*.register_module() side effects actually
        # ran for real -- the thing custom_imports exists to guarantee.
        assert cfg.custom_imports == dict(
            imports=["transforms", "internal_val_hook"], allow_failed_imports=False
        ), f"custom_imports unexpectedly changed: {cfg.custom_imports}"

        print(f"[PASS] test_generated_config_avoids_lazy_import_and_resolves_d_vect_literally "
              f"(is_lazy_import={is_lazy}, d_vect={cfg.d_vect})")
    finally:
        shutil.rmtree(tmp)


def main():
    test_generated_config_avoids_lazy_import_and_resolves_d_vect_literally()
    print("[ALL MAKE_CONFIG REAL-LOAD TESTS PASSED]")


if __name__ == "__main__":
    main()
