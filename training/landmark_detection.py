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
    norm_pair: tuple[int, int] = (0, 1),
    endpoint_order_invariant: bool = False,
) -> torch.Tensor:
    """
    Normalised Mean Error (NME) per sample, normalised by the Euclidean
    distance between two fixed GT landmark indices in image pixel space.

    For BPD/OFD (N=2) the default norm_pair=(0,1) normalises by the
    inter-landmark distance (i.e. the diameter length itself). For 300W
    (N=68) pass norm_pair=(36, 45) — the standard inter-ocular distance
    between the two outer eye corners (Sagonas et al.; see
    HRNet-Facial-Landmark-Detection lib/core/evaluation.py, which computes
    exactly `norm(pts_gt[36] - pts_gt[45])`, NOT an eye-centre average —
    using eye-centre distance would make NME numbers incomparable to
    published 300W baselines.

    MODIFIED: endpoint_order_invariant (default False, preserves prior
    behaviour for 300W and any other caller). When True, matches
    Di Vece et al.'s published fetal-endpoint NME exactly:
        NME = min(d_std, d_swap) / (2 * ||gt_0 - gt_1||),
    where d_std/d_swap are the summed two-endpoint errors under the
    identity vs. swapped predicted-to-GT correspondence. Ground-truth
    endpoints are canonicalised left-to-right by x-coordinate in the
    dataset pipeline (datasets/landmark_dataset.py), but predicted
    endpoints are NOT re-sorted after heatmap_to_coords() decodes them —
    they keep whichever query channel produced them. So a query-channel
    "crossover" (query 0's peak landing right of query 1's) is exactly
    what this flag is for: it lets NME resolve that crossover the same
    way the published baseline does, instead of penalising it as a
    coordinate error. Only defined for N=2; raises if N != 2.

    Args:
        pred_coords: (B, N, 2) in heatmap pixel space
        gt_coords:   (B, N, 2) in heatmap pixel space
        heatmap_size: (H, W) of the heatmap
        img_size:     (H, W) of the model input image
        norm_pair:    indices of the two GT landmarks whose distance is
                      used as the normaliser. Ignored (must be (0, 1))
                      when endpoint_order_invariant=True.
        endpoint_order_invariant: use the published swap-min two-endpoint
                      formula instead of the fixed-channel formula.

    Returns:
        nme: (B,) NME per sample
    """
    hm_h, hm_w = heatmap_size
    ih, iw = img_size

    # scale to image pixel space for physically meaningful distances
    scale = pred_coords.new_tensor([iw / hm_w, ih / hm_h])
    pred_img = pred_coords * scale    # (B, N, 2)
    gt_img   = gt_coords   * scale   # (B, N, 2)

    if endpoint_order_invariant:
        n_landmarks = pred_img.shape[1]
        if n_landmarks != 2:
            raise ValueError(
                "endpoint_order_invariant=True is only defined for N=2 "
                f"two-endpoint fetal measurements, got N={n_landmarks}."
            )
        err_std = (
            (pred_img[:, 0] - gt_img[:, 0]).norm(dim=-1)
            + (pred_img[:, 1] - gt_img[:, 1]).norm(dim=-1)
        )
        err_swap = (
            (pred_img[:, 0] - gt_img[:, 1]).norm(dim=-1)
            + (pred_img[:, 1] - gt_img[:, 0]).norm(dim=-1)
        )
        diameter = (gt_img[:, 0] - gt_img[:, 1]).norm(dim=-1).clamp(min=1.0)
        nme_per_sample = torch.minimum(err_std, err_swap) / (2.0 * diameter)
        return nme_per_sample

    # normaliser: Euclidean distance between the two configured GT landmarks
    i0, i1 = norm_pair
    norm = (gt_img[:, i0] - gt_img[:, i1]).norm(dim=-1).clamp(min=1.0)  # (B,)

    # Euclidean error per landmark, averaged across N landmarks
    errors = (pred_img - gt_img).norm(dim=-1)   # (B, N)
    nme_per_sample = errors.mean(dim=-1) / norm  # (B,)

    return nme_per_sample


def compute_pixel_error(
    pred_coords: torch.Tensor,
    gt_coords: torch.Tensor,
    heatmap_size: tuple[int, int],
    img_size: tuple[int, int],
    endpoint_order_invariant: bool = False,
) -> torch.Tensor:
    """
    Mean per-landmark Euclidean pixel error per sample, in model-input
    pixel space (img_size, e.g. 512x512) — NOT the original un-cropped
    image space, so converting to mm still requires accounting for any
    crop/resize scale factor back to the original image. Separate
    function (not folded into compute_nme's return) so existing
    compute_nme call sites are untouched.

    MODIFIED: endpoint_order_invariant (default False) — when True, uses
    the SAME identity-vs-swap correspondence choice as compute_nme's
    swap-min formula (whichever pairing gives the smaller summed distance
    for that sample), so this stays consistent with a swap-min NME instead
    of silently reporting a fixed-channel distance while NME reports a
    swap-corrected one. Only defined for N=2, mirroring compute_nme.

    Returns:
        pixel_error: (B,) mean per-landmark pixel error per sample
    """
    hm_h, hm_w = heatmap_size
    ih, iw = img_size
    scale = pred_coords.new_tensor([iw / hm_w, ih / hm_h])
    pred_img = pred_coords * scale
    gt_img = gt_coords * scale

    if endpoint_order_invariant:
        n_landmarks = pred_img.shape[1]
        if n_landmarks != 2:
            raise ValueError(
                "endpoint_order_invariant=True is only defined for N=2 "
                f"two-endpoint fetal measurements, got N={n_landmarks}."
            )
        err_std = (
            (pred_img[:, 0] - gt_img[:, 0]).norm(dim=-1)
            + (pred_img[:, 1] - gt_img[:, 1]).norm(dim=-1)
        )
        err_swap = (
            (pred_img[:, 0] - gt_img[:, 1]).norm(dim=-1)
            + (pred_img[:, 1] - gt_img[:, 0]).norm(dim=-1)
        )
        return torch.minimum(err_std, err_swap) / n_landmarks

    return (pred_img - gt_img).norm(dim=-1).mean(dim=-1)


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
        nme_norm_pair: tuple[int, int] = (0, 1),  # MODIFIED: (36, 45) for 300W inter-ocular NME
        # MODIFIED: opt-in published-compatible swap-min NME for two-endpoint
        # fetal tasks (matches Di Vece et al.'s BiometryNet baseline metric
        # exactly, see compute_nme()'s docstring). Explicit per-config flag,
        # default False so existing behaviour (incl. 300W, which is N=68 and
        # must never set this) is unchanged unless a fetal config opts in.
        endpoint_order_invariant_nme: bool = False,
        # Attention mask annealing — disabled initially; add as ablation later
        attn_mask_annealing_enabled: bool = False,
        attn_mask_annealing_start_steps: Optional[List[int]] = None,
        attn_mask_annealing_end_steps: Optional[List[int]] = None,
        # Loss
        loss_type: str = "mse",      # "mse" | "weighted_mse" | "adaptive_wing" | "hybrid" | "hybrid_awing" | "triple"
        alpha: float = 5.0,          # foreground weight multiplier for weighted_mse / hybrid
        temperature: float = 10.0,   # soft-argmax temperature for hybrid coord loss
        lambda_coord: float = 0.1,   # coord loss weight for hybrid/triple
        lambda_awing: float = 0.01,  # AWing supplement weight for triple (WMSE + λ_a·AWing + λ_c·L1)
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
        # MODIFIED: opt-in per-sample NME dump for paired significance testing
        # against a second method's per-image errors on the same test set
        # (relies on test_dataloader's shuffle=False/drop_last=False for
        # deterministic, index-aligned ordering across separate eval runs).
        # Default None: byte-identical to prior behaviour (no file written).
        test_nme_dump_path: Optional[str] = None,
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
        self.nme_norm_pair = tuple(nme_norm_pair)
        self.endpoint_order_invariant_nme = endpoint_order_invariant_nme
        if endpoint_order_invariant_nme and num_landmarks != 2:
            raise ValueError(
                "endpoint_order_invariant_nme=True is only valid for "
                f"two-endpoint fetal tasks (num_landmarks=2), got "
                f"num_landmarks={num_landmarks}. Do not enable this for 300W."
            )
        self.loss_type   = loss_type
        self.alpha       = alpha
        self.temperature = temperature
        self.lambda_coord = lambda_coord
        self.lambda_awing = lambda_awing
        self._awing = AdaptiveWingLoss(awing_omega, awing_theta, awing_alpha, awing_epsilon)

        self.test_nme_dump_path = test_nme_dump_path

        # per-epoch NME accumulators (single-GPU; lists reset on epoch end)
        self._val_nme:  list[float] = []
        self._test_nme: list[float] = []
        self._test_pixel_error: list[float] = []
        # MODIFIED: raw (image-pixel-space) coordinates, dumped alongside NME
        # so per-image results can be re-scored offline under a different NME
        # definition later without re-running inference on the checkpoint.
        self._test_pred_coords: list = []
        self._test_gt_coords: list = []

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

        mask_logits_per_layer, _, coord_pred = self(imgs)
        # coord_pred: (B, N, 2) when heatmap_head=="coord_direct", else None

        total_loss    = torch.tensor(0.0, device=self.device)
        total_L_hm    = torch.tensor(0.0, device=self.device)
        total_L_coord = torch.tensor(0.0, device=self.device)
        n_layers = len(mask_logits_per_layer)

        for i, mask_logits in enumerate(mask_logits_per_layer):
            is_final_coord_direct = (
                coord_pred is not None and i == n_layers - 1
            )

            if is_final_coord_direct:
                # coord_direct final layer: pure L1 on direct regression output (no heatmap loss)
                L_coord    = F.l1_loss(coord_pred, gt_coords)
                layer_loss = L_coord
                total_L_coord = total_L_coord + L_coord.detach()
            else:
                pred = F.interpolate(
                    mask_logits, self.heatmap_size,
                    mode="bilinear", align_corners=False,
                )                              # (B, N, hm_H, hm_W)

                if self.loss_type == "weighted_mse":
                    layer_loss = weighted_mse_loss(pred, gt_heatmaps, self.alpha)
                elif self.loss_type == "adaptive_wing":
                    layer_loss = self._awing(pred, gt_heatmaps)
                elif self.loss_type in ("hybrid", "coord_direct"):
                    # "coord_direct" reuses hybrid for intermediate layers
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
                elif self.loss_type == "triple":
                    # WMSE (主导) + λ_awing·AWing (补充) + λ_coord·L1(coord)
                    L_hm, L_coord = hybrid_loss(
                        pred, gt_heatmaps, gt_coords,
                        alpha=self.alpha, temperature=self.temperature,
                    )
                    L_awing = self._awing(pred, gt_heatmaps)
                    layer_loss     = L_hm + self.lambda_awing * L_awing + self.lambda_coord * L_coord
                    total_L_hm    = total_L_hm    + L_hm.detach()
                    total_L_coord = total_L_coord + L_coord.detach()
                else:
                    layer_loss = F.mse_loss(pred, gt_heatmaps)

            postfix = f"_l{i}" if n_layers > 1 else ""
            self.log(f"train/layer_loss{postfix}", layer_loss, on_step=True, on_epoch=False)
            total_loss = total_loss + layer_loss

        total_loss = total_loss / n_layers
        self.log("train/loss", total_loss, on_step=True, on_epoch=False, prog_bar=True)

        if self.loss_type in ("hybrid", "hybrid_awing", "triple", "coord_direct"):
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

        mask_logits_per_layer, _, coord_pred = self(imgs)

        if coord_pred is not None:
            # coord_direct mode: use direct regression output for NME
            pred_coords = coord_pred.detach()          # (B, N, 2) already in heatmap space
            val_loss = F.l1_loss(pred_coords, gt_coords)
            self.log(f"{log_prefix}/mse", val_loss, on_step=False, on_epoch=True)
        else:
            # use only the final layer heatmap for evaluation metrics
            pred = F.interpolate(
                mask_logits_per_layer[-1], self.heatmap_size,
                mode="bilinear", align_corners=False,
            )
            val_loss = F.mse_loss(pred, gt_heatmaps)
            self.log(f"{log_prefix}/mse", val_loss, on_step=False, on_epoch=True)
            # NME: pred decoded from heatmap, GT taken directly from dataset (no quantisation)
            pred_coords = heatmap_to_coords(pred.detach())   # (B, N, 2)

        nme_per_sample = compute_nme(
            pred_coords, gt_coords, self.heatmap_size, self.img_size,
            norm_pair=self.nme_norm_pair,
            endpoint_order_invariant=self.endpoint_order_invariant_nme,
        )                                  # (B,)
        acc = self._test_nme if log_prefix == "test" else self._val_nme
        acc.extend(nme_per_sample.tolist())

        if log_prefix == "test" and self.test_nme_dump_path:
            pixel_error = compute_pixel_error(
                pred_coords, gt_coords, self.heatmap_size, self.img_size,
                endpoint_order_invariant=self.endpoint_order_invariant_nme,
            )
            self._test_pixel_error.extend(pixel_error.tolist())
            # MODIFIED: also stash raw coordinates (image-pixel space, same
            # frame as pixel_error) so per-image results can be re-scored
            # offline under a different NME definition later without
            # re-running inference on the checkpoint.
            hm_h, hm_w = self.heatmap_size
            ih, iw = self.img_size
            coord_scale = pred_coords.new_tensor([iw / hm_w, ih / hm_h])
            self._test_pred_coords.extend((pred_coords * coord_scale).tolist())
            self._test_gt_coords.extend((gt_coords * coord_scale).tolist())

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
        if self.test_nme_dump_path:
            # MODIFIED: also dump raw (image-pixel-space) pred/gt coordinates
            # per landmark, not just nme/pixel_error — enables offline
            # re-scoring under a different NME definition later without
            # re-running inference on the checkpoint (see eval_step).
            with open(self.test_nme_dump_path, "w") as f:
                header = ["index", "nme", "pixel_error"]
                for j in range(self.num_landmarks):
                    header += [f"pred_x{j}", f"pred_y{j}", f"gt_x{j}", f"gt_y{j}"]
                f.write(",".join(header) + "\n")
                for i, v in enumerate(self._test_nme):
                    pe = self._test_pixel_error[i] if i < len(self._test_pixel_error) else ""
                    row = [str(i), f"{v:.8f}", "" if pe == "" else f"{pe:.4f}"]
                    if i < len(self._test_pred_coords):
                        pred_c = self._test_pred_coords[i]
                        gt_c = self._test_gt_coords[i]
                        for j in range(self.num_landmarks):
                            row += [
                                f"{pred_c[j][0]:.8f}", f"{pred_c[j][1]:.8f}",
                                f"{gt_c[j][0]:.8f}", f"{gt_c[j][1]:.8f}",
                            ]
                    f.write(",".join(row) + "\n")
            rank_zero_info(
                f"{bold_green}Per-sample NME dumped to {self.test_nme_dump_path} "
                f"(n={len(self._test_nme)}){reset}"
            )
        self._test_nme.clear()
        self._test_pixel_error.clear()
        self._test_pred_coords.clear()
        self._test_gt_coords.clear()
        rank_zero_info(f"{bold_green}Test NME: {nme * 100:.2f}%{reset}")
