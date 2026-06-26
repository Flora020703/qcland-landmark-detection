# ---------------------------------------------------------------
# MODIFIED: new file — LightningModule for heatmap-based landmark detection.
# Replaces Hungarian matching + classification loss with direct MSE on heatmaps.
# ---------------------------------------------------------------

from typing import Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning
from lightning.fabric.utilities import rank_zero_info

from training.lightning_module import LightningModule

bold_green = "\033[1;32m"
reset = "\033[0m"


# -----------------------------------------------------------------------
# Coordinate utilities
# -----------------------------------------------------------------------

@torch.no_grad()
def heatmap_to_coords(heatmaps: torch.Tensor) -> torch.Tensor:
    """
    Extract (x, y) landmark coordinates from heatmaps via argmax +
    sub-pixel refinement (HRNet decode_preds style, sign-shift variant).

    Args:
        heatmaps: (B, N, H, W)  — raw or sigmoid'd predictions

    Returns:
        coords: (B, N, 2) float, in heatmap pixel space [x, y]
    """
    B, N, H, W = heatmaps.shape
    flat = heatmaps.view(B, N, -1)
    idx = flat.argmax(dim=-1)          # (B, N)

    xi = idx % W                       # column index, long
    yi = idx // W                      # row    index, long

    x = xi.float()
    y = yi.float()

    # index helpers for gathering neighbor values
    b = torch.arange(B, device=heatmaps.device).unsqueeze(1).expand(B, N)
    n = torch.arange(N, device=heatmaps.device).unsqueeze(0).expand(B, N)

    xi_l = (xi - 1).clamp(0, W - 1)
    xi_r = (xi + 1).clamp(0, W - 1)
    yi_u = (yi - 1).clamp(0, H - 1)
    yi_d = (yi + 1).clamp(0, H - 1)

    dx = heatmaps[b, n, yi, xi_r] - heatmaps[b, n, yi, xi_l]   # (B, N)
    dy = heatmaps[b, n, yi_d, xi] - heatmaps[b, n, yi_u, xi]   # (B, N)

    # shift ±0.25 pixel only when not on the heatmap border
    not_border_x = (xi > 0) & (xi < W - 1)
    not_border_y = (yi > 0) & (yi < H - 1)

    x = x + 0.25 * dx.sign() * not_border_x.float()
    y = y + 0.25 * dy.sign() * not_border_y.float()

    return torch.stack([x, y], dim=-1)   # (B, N, 2)


def weighted_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 5.0,
) -> torch.Tensor:
    """
    Weighted MSE: foreground pixels near the Gaussian peak get alpha+1× weight.
    weight = 1 + alpha * target  (target in [0,1], peak=1 → weight=1+alpha)
    Reduces background dominance when the heatmap is mostly zeros.
    """
    weight = 1.0 + alpha * target
    return ((pred - target) ** 2 * weight).mean()


def spatial_softmax(heatmap: torch.Tensor, temperature: float = 10.0) -> torch.Tensor:
    """
    Differentiable coordinate extraction via spatial softmax (soft-argmax).
    Args:
        heatmap: (B, Q, H, W)  — raw logits (no sigmoid needed)
        temperature: higher = sharper distribution, larger gradients
    Returns:
        coords: (B, Q, 2) as [x, y] in heatmap pixel space — matches lms_hm format
    """
    B, Q, H, W = heatmap.shape
    device = heatmap.device

    x_coords = torch.arange(W, device=device, dtype=heatmap.dtype)  # (W,)
    y_coords = torch.arange(H, device=device, dtype=heatmap.dtype)  # (H,)

    weights = torch.softmax(heatmap.reshape(B, Q, -1) * temperature, dim=-1)
    weights = weights.reshape(B, Q, H, W)  # (B, Q, H, W)

    # marginal sums: sum over H→ column weights, sum over W→ row weights
    pred_x = (weights.sum(dim=2) * x_coords).sum(dim=-1)   # (B, Q)
    pred_y = (weights.sum(dim=3) * y_coords).sum(dim=-1)   # (B, Q)

    return torch.stack([pred_x, pred_y], dim=-1)            # (B, Q, 2) [x, y]


def hybrid_loss(
    pred_hm: torch.Tensor,
    target_hm: torch.Tensor,
    gt_coords: torch.Tensor,
    alpha: float = 5.0,
    temperature: float = 10.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Weighted MSE on heatmaps + L1 on soft-argmax coordinates.
    Returns (L_hm, L_coord) separately so caller can weight and log them.
    gt_coords: (B, Q, 2) [x, y] in heatmap space — same format as spatial_softmax output.
    """
    weight = 1.0 + alpha * target_hm
    L_hm = ((pred_hm - target_hm) ** 2 * weight).mean()

    pred_coords = spatial_softmax(pred_hm, temperature=temperature)  # (B, Q, 2)
    L_coord = F.l1_loss(pred_coords, gt_coords)

    return L_hm, L_coord


class AdaptiveWingLoss(nn.Module):
    """
    Adaptive Wing Loss (Wang et al., ICCV 2019).
    Below theta: Wing-like log loss (larger gradient for small errors near peak).
    Above theta: linear loss (MSE-like behaviour for background).
    Recommended defaults: omega=14, theta=0.5, alpha=2.1, epsilon=1.
    """

    def __init__(
        self,
        omega: float = 14,
        theta: float = 0.5,
        alpha: float = 2.1,
        epsilon: float = 1,
    ):
        super().__init__()
        self.omega   = omega
        self.theta   = theta
        self.alpha   = alpha
        self.epsilon = epsilon

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        delta   = (target - pred).abs()
        alpha_t = self.alpha - target       # per-pixel adaptive exponent
        A = (
            self.omega
            * (1 / (1 + (self.theta / self.epsilon) ** (alpha_t - 1)))
            * alpha_t
            * ((self.theta / self.epsilon) ** (alpha_t - 2))
            / self.epsilon
        )
        C    = self.theta * A - self.omega * torch.log(
            1 + (self.theta / self.epsilon) ** alpha_t
        )
        loss = torch.where(
            delta < self.theta,
            self.omega * torch.log(1 + (delta / self.epsilon) ** alpha_t),
            A * delta - C,
        )
        return loss.mean()


@torch.no_grad()
def compute_nme(
    pred_coords: torch.Tensor,
    gt_coords: torch.Tensor,
    heatmap_size: tuple[int, int],
    img_size: tuple[int, int],
) -> torch.Tensor:
    """
    Normalised Mean Error (NME) per sample, normalised by the inter-landmark
    distance (i.e. the diameter length in image pixels).

    Args:
        pred_coords: (B, N, 2) in heatmap pixel space
        gt_coords:   (B, N, 2) in heatmap pixel space
        heatmap_size: (H, W) of the heatmap
        img_size:     (H, W) of the model input image

    Returns:
        nme: (B,) NME per sample
    """
    hm_h, hm_w = heatmap_size
    ih, iw = img_size

    # scale to image pixel space for physically meaningful distances
    scale = pred_coords.new_tensor([iw / hm_w, ih / hm_h])
    pred_img = pred_coords * scale    # (B, N, 2)
    gt_img   = gt_coords   * scale   # (B, N, 2)

    # normaliser: Euclidean distance between landmark 0 and landmark 1
    # (= diameter length in image pixels)
    norm = (gt_img[:, 0] - gt_img[:, 1]).norm(dim=-1).clamp(min=1.0)  # (B,)

    # Euclidean error per landmark, averaged across N landmarks
    errors = (pred_img - gt_img).norm(dim=-1)   # (B, N)
    nme_per_sample = errors.mean(dim=-1) / norm  # (B,)

    return nme_per_sample


# -----------------------------------------------------------------------
# Lightning module
# -----------------------------------------------------------------------

class LandmarkDetection(LightningModule):
    """
    EoMT adapted for heatmap-based landmark detection (BPD or OFD).

    Key differences from the segmentation LightningModule:
      - Loss    : MSE on heatmaps (no BCE/Dice/CE, no classification loss)
      - Matching: fixed 1-to-1, query i → landmark i (no Hungarian matching)
      - Metric  : NME normalised by inter-landmark distance
      - Annealing: attn_mask_annealing_enabled=False by default
      - forward : images expected as float32 [0, 1]; encoder handles ImageNet
                  normalisation internally (pixel_mean / pixel_std in ViT)
    """

    def __init__(
        self,
        network: nn.Module,
        img_size: tuple[int, int] = (512, 512),
        num_landmarks: int = 2,
        heatmap_size: tuple[int, int] = (64, 64),
        # Attention mask annealing — disabled initially; add as ablation later
        attn_mask_annealing_enabled: bool = False,
        attn_mask_annealing_start_steps: Optional[List[int]] = None,
        attn_mask_annealing_end_steps: Optional[List[int]] = None,
        # Loss
        loss_type: str = "mse",      # "mse" | "weighted_mse" | "adaptive_wing" | "hybrid" | "hybrid_awing"
        alpha: float = 5.0,          # foreground weight multiplier for weighted_mse / hybrid
        temperature: float = 10.0,   # soft-argmax temperature for hybrid coord loss
        lambda_coord: float = 0.1,   # coord loss weight for hybrid (total = L_hm + λ·L_coord)
        # AdaptiveWingLoss hyper-params (only used when loss_type="adaptive_wing")
        awing_omega:   float = 14.0,
        awing_theta:   float = 0.5,
        awing_alpha:   float = 2.1,
        awing_epsilon: float = 1.0,
        # Optimiser
        lr: float = 1e-4,
        llrd: float = 0.8,
        llrd_l2_enabled: bool = True,
        lr_mult: float = 1.0,
        weight_decay: float = 0.05,
        poly_power: float = 0.9,
        warmup_steps: List[int] = (500, 1000),
        # Checkpoint
        ckpt_path: Optional[str] = None,
        delta_weights: bool = False,
    ):
        super().__init__(
            network=network,
            img_size=img_size,
            num_classes=num_landmarks,          # reuses base field; no semantic meaning here
            attn_mask_annealing_enabled=attn_mask_annealing_enabled,
            attn_mask_annealing_start_steps=attn_mask_annealing_start_steps,
            attn_mask_annealing_end_steps=attn_mask_annealing_end_steps,
            lr=lr,
            llrd=llrd,
            llrd_l2_enabled=llrd_l2_enabled,
            lr_mult=lr_mult,
            weight_decay=weight_decay,
            poly_power=poly_power,
            warmup_steps=warmup_steps,
            ckpt_path=ckpt_path,
            delta_weights=delta_weights,
            load_ckpt_class_head=False,         # no class head in landmark task
        )

        self.heatmap_size = heatmap_size
        self.num_landmarks = num_landmarks
        self.loss_type   = loss_type
        self.alpha       = alpha
        self.temperature = temperature
        self.lambda_coord = lambda_coord
        self._awing = AdaptiveWingLoss(awing_omega, awing_theta, awing_alpha, awing_epsilon)

        # per-epoch NME accumulators (single-GPU; lists reset on epoch end)
        self._val_nme:  list[float] = []
        self._test_nme: list[float] = []

        self.save_hyperparameters(ignore=["_class_path", "network"])

    # ------------------------------------------------------------------
    # Forward — override base class to skip the /255 division.
    # Our DataLoader yields float32 [0, 1]; EoMT encoder normalises itself.
    # ------------------------------------------------------------------

    def forward(self, imgs: torch.Tensor):
        return self.network(imgs)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        imgs, gt_heatmaps, gt_coords = batch   # gt_coords: (B, N, 2) [x,y] in heatmap space

        mask_logits_per_layer, _ = self(imgs)

        total_loss    = torch.tensor(0.0, device=self.device)
        total_L_hm    = torch.tensor(0.0, device=self.device)
        total_L_coord = torch.tensor(0.0, device=self.device)
        n_layers = len(mask_logits_per_layer)

        for i, mask_logits in enumerate(mask_logits_per_layer):
            pred = F.interpolate(
                mask_logits, self.heatmap_size,
                mode="bilinear", align_corners=False,
            )                              # (B, N, hm_H, hm_W)

            if self.loss_type == "weighted_mse":
                layer_loss = weighted_mse_loss(pred, gt_heatmaps, self.alpha)
            elif self.loss_type == "adaptive_wing":
                layer_loss = self._awing(pred, gt_heatmaps)
            elif self.loss_type == "hybrid":
                L_hm, L_coord = hybrid_loss(
                    pred, gt_heatmaps, gt_coords,
                    alpha=self.alpha, temperature=self.temperature,
                )
                layer_loss     = L_hm + self.lambda_coord * L_coord
                total_L_hm    = total_L_hm    + L_hm.detach()
                total_L_coord = total_L_coord + L_coord.detach()
            elif self.loss_type == "hybrid_awing":
                L_hm   = self._awing(pred, gt_heatmaps)
                pred_coords = spatial_softmax(pred, temperature=self.temperature)
                L_coord = F.l1_loss(pred_coords, gt_coords)
                layer_loss     = L_hm + self.lambda_coord * L_coord
                total_L_hm    = total_L_hm    + L_hm.detach()
                total_L_coord = total_L_coord + L_coord.detach()
            else:
                layer_loss = F.mse_loss(pred, gt_heatmaps)

            postfix = f"_l{i}" if n_layers > 1 else ""
            self.log(f"train/layer_loss{postfix}", layer_loss, on_step=True, on_epoch=False)
            total_loss = total_loss + layer_loss

        total_loss = total_loss / n_layers
        self.log("train/loss", total_loss, on_step=True, on_epoch=False, prog_bar=True)

        if self.loss_type in ("hybrid", "hybrid_awing"):
            self.log("train/loss_hm",    total_L_hm    / n_layers, on_step=True, on_epoch=False)
            self.log("train/loss_coord", total_L_coord / n_layers, on_step=True, on_epoch=False)

        return total_loss

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validation_step(self, batch, batch_idx=0):
        return self.eval_step(batch, batch_idx, "val")

    def eval_step(self, batch, batch_idx, log_prefix):
        imgs, gt_heatmaps, gt_coords = batch   # gt_coords: (B, N, 2) float in heatmap space

        mask_logits_per_layer, _ = self(imgs)

        # use only the final layer prediction for evaluation metrics
        pred = F.interpolate(
            mask_logits_per_layer[-1], self.heatmap_size,
            mode="bilinear", align_corners=False,
        )
        val_loss = F.mse_loss(pred, gt_heatmaps)
        self.log(f"{log_prefix}/mse", val_loss, on_step=False, on_epoch=True)

        # NME: pred decoded from heatmap, GT taken directly from dataset (no quantisation error)
        pred_coords = heatmap_to_coords(pred.detach())          # (B, N, 2)

        nme_per_sample = compute_nme(
            pred_coords, gt_coords, self.heatmap_size, self.img_size
        )                                  # (B,)
        acc = self._test_nme if log_prefix == "test" else self._val_nme
        acc.extend(nme_per_sample.tolist())

    def on_validation_epoch_end(self):
        if not self._val_nme:
            return
        nme = sum(self._val_nme) / len(self._val_nme)
        self.log("metrics/val_nme", nme, prog_bar=True)
        self._val_nme.clear()
        if not self.trainer.sanity_checking:
            rank_zero_info(f"{bold_green}NME: {nme * 100:.2f}%{reset}")

    def on_validation_end(self):
        key = "metrics/val_nme"
        if not self.trainer.sanity_checking and key in self.trainer.callback_metrics:
            rank_zero_info(
                f"{bold_green}Final NME: "
                f"{self.trainer.callback_metrics[key] * 100:.2f}%{reset}"
            )

    def test_step(self, batch, batch_idx=0):
        return self.eval_step(batch, batch_idx, "test")

    def on_test_epoch_end(self):
        if not self._test_nme:
            return
        nme = sum(self._test_nme) / len(self._test_nme)
        self.log("metrics/test_nme", nme, prog_bar=True)
        self._test_nme.clear()
        rank_zero_info(f"{bold_green}Test NME: {nme * 100:.2f}%{reset}")
