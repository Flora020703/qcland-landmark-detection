# ---------------------------------------------------------------
# MODIFIED: new file — Exponential Moving Average (EMA) of model weights.
# ---------------------------------------------------------------

from typing import Any

import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback


class EMACallback(Callback):
    """
    Tracks an exponential moving average of the model's float parameters
    throughout training.

    Why not Lightning's built-in StochasticWeightAveraging: SWA takes over
    the LR schedule during its active phase (forces a constant/cyclic LR),
    which would conflict with this project's custom TwoStageWarmupPolySchedule
    (training/lightning_module.py). A plain shadow-weight EMA has zero
    interaction with the optimizer/scheduler — it only reads parameters
    after each step.

    Why the shadow is NOT swapped into the model during validation: doing so
    makes checkpoint-selection ordering depend on exactly when this
    callback's hook runs relative to ModelCheckpoint's — get that wrong and
    the "best" checkpoint silently saves the wrong weights. Instead, the EMA
    shadow rides along inside the checkpoint payload (on_save_checkpoint) and
    is evaluated afterwards via a separate materialize step (see apply_ema.py
    at the repo root). val_nme used for checkpoint selection during training
    always reflects the raw (non-EMA) weights, same as every other
    experiment, so ablation comparisons stay apples-to-apples.

    decay warmup: decay(t) = min(target, (1+t)/(10+t)), the same
    update-count-dependent schedule as TensorFlow's
    ExponentialMovingAverage(num_updates=...) -- NOT timm's ModelEmaV2,
    which uses a fixed decay with no warmup (verified against timm's
    model_ema.py source, 2026-07-23). This keeps the shadow from being
    dragged by full-strength decay while weights are still near their
    random init.
    """

    def __init__(self, decay: float = 0.999, use_warmup: bool = True):
        super().__init__()
        self.decay = decay
        self.use_warmup = use_warmup
        self.shadow: dict[str, torch.Tensor] = {}
        self._step = 0

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if not self.shadow:  # don't reset an already-loaded shadow on resume
            self.shadow = {
                name: p.detach().clone()
                for name, p in pl_module.named_parameters()
                if p.dtype.is_floating_point
            }

    def _current_decay(self) -> float:
        if not self.use_warmup:
            return self.decay
        return min(self.decay, (1 + self._step) / (10 + self._step))

    def on_train_batch_end(
        self, trainer: Trainer, pl_module: LightningModule, outputs, batch, batch_idx: int
    ) -> None:
        d = self._current_decay()
        with torch.no_grad():
            for name, p in pl_module.named_parameters():
                shadow_p = self.shadow.get(name)
                if shadow_p is not None:
                    shadow_p.mul_(d).add_(p.detach(), alpha=1 - d)
        self._step += 1

    def on_save_checkpoint(
        self, trainer: Trainer, pl_module: LightningModule, checkpoint: dict[str, Any]
    ) -> None:
        checkpoint["ema_state_dict"] = self.shadow
        checkpoint["ema_step"] = self._step

    def on_load_checkpoint(
        self, trainer: Trainer, pl_module: LightningModule, checkpoint: dict[str, Any]
    ) -> None:
        if "ema_state_dict" in checkpoint:
            self.shadow = checkpoint["ema_state_dict"]
            self._step = checkpoint.get("ema_step", 0)
