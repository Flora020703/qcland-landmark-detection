# ---------------------------------------------------------------
# MODIFIED: new entry-point for landmark detection training.
# Mirrors main.py but without segmentation-specific link_arguments
# (no num_classes / stuff_classes linking from data to model).
# ---------------------------------------------------------------

import logging
import warnings
from types import MethodType

import torch
from lightning.pytorch import cli
from lightning.pytorch.callbacks import LearningRateMonitor, ModelSummary
from lightning.pytorch.loops.fetchers import _DataFetcher, _DataLoaderIterDataFetcher
from lightning.pytorch.loops.training_epoch_loop import _TrainingEpochLoop

# Suppress PyTorch FX warnings for DINOv3 models
import os
os.environ["TORCH_LOGS"] = "-dynamo"


# -----------------------------------------------------------------------
# Identical val-check fix from main.py (step-based instead of batch-based)
# -----------------------------------------------------------------------

def _should_check_val_fx(self: _TrainingEpochLoop, data_fetcher: _DataFetcher) -> bool:
    if not self._should_check_val_epoch():
        return False

    is_infinite_dataset = self.trainer.val_check_batch == float("inf")
    is_last_batch = self.batch_progress.is_last_batch
    if is_last_batch and (
        is_infinite_dataset or isinstance(data_fetcher, _DataLoaderIterDataFetcher)
    ):
        return True

    if self.trainer.should_stop and self.trainer.fit_loop._can_stop_early:
        return True

    is_val_check_batch = is_last_batch
    if isinstance(self.trainer.limit_train_batches, int) and is_infinite_dataset:
        is_val_check_batch = (
            self.batch_idx + 1
        ) % self.trainer.limit_train_batches == 0
    elif self.trainer.val_check_batch != float("inf"):
        if self.trainer.check_val_every_n_epoch is not None:
            is_val_check_batch = (
                self.batch_idx + 1
            ) % self.trainer.val_check_batch == 0
        else:
            is_val_check_batch = (
                self.global_step
            ) % self.trainer.val_check_batch == 0 and not self._should_accumulate()

    return is_val_check_batch


# -----------------------------------------------------------------------
# Custom CLI — no segmentation-specific links
# -----------------------------------------------------------------------

class LandmarkCLI(cli.LightningCLI):
    def __init__(self, *args, **kwargs):
        logging.getLogger().setLevel(logging.INFO)
        torch.set_float32_matmul_precision("medium")
        torch.backends.cudnn.deterministic = True   # deterministic ConvTranspose2d / cuDNN kernels
        torch.backends.cudnn.benchmark = False      # disable auto-tuner (picks different kernels per run)
        torch._dynamo.config.capture_scalar_outputs = True
        torch._dynamo.config.suppress_errors = True
        warnings.filterwarnings(
            "ignore",
            message=r".*It is recommended to use .* when logging on epoch level.*",
        )
        super().__init__(*args, **kwargs)

    def add_arguments_to_parser(self, parser):
        parser.add_argument("--compile_disabled", action="store_true")

        # img_size: data → model and encoder (single source of truth)
        parser.link_arguments(
            "data.init_args.img_size", "model.init_args.img_size"
        )
        parser.link_arguments(
            "data.init_args.img_size",
            "model.init_args.network.init_args.encoder.init_args.img_size",
        )

        # heatmap_size: data → model only (EoMT no longer does the resize)
        parser.link_arguments(
            "data.init_args.heatmap_size", "model.init_args.heatmap_size"
        )

        # ckpt_path: model → encoder (load backbone weights)
        parser.link_arguments(
            "model.init_args.ckpt_path",
            "model.init_args.network.init_args.encoder.init_args.ckpt_path",
        )

    def fit(self, model, **kwargs):
        self.trainer.fit_loop.epoch_loop._should_check_val_fx = MethodType(
            _should_check_val_fx, self.trainer.fit_loop.epoch_loop
        )

        if not self.config[self.config["subcommand"]]["compile_disabled"]:
            model = torch.compile(model)

        self.trainer.fit(model, **kwargs)


if __name__ == "__main__":
    LandmarkCLI(
        save_config_callback=None,
    )
