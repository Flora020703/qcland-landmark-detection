# CC Handover Document — EoMT Landmark Detection (Run 12 → Run 20e)

> Paste this entire document at the start of a new Claude Code conversation.
> Goal: bring the new session up to speed instantly on code state, run history, and next tasks.

---

## Project Overview

MSc thesis at UCL. Adapting **EoMT** (Encoder-Only Mask Transformer, DINOv2-ViT-S backbone) from semantic segmentation to **heatmap-based fetal biometry landmark detection** (BPD & OFD).

- Supervisors: Dr. Zhehua Mao, Dr. Sophia Bano (SRV group)
- Target: beat HRNet baseline → BPD NME=6.49%, OFD NME=4.41%
- **Current best: Run 18 → val NME 7.09%, test NME 13.66%**
- Training server: AutoDL RTX 4090, `/root/eomt/`
- W&B: https://wandb.ai/ucabnx1-ucl/eomt-landmark
- GitHub: https://github.com/Flora020703/eomt-landmark-detection (private)
- Run 2's claimed 17.68% was a **bug** (GT decoded from heatmap argmax with quantization error — that number is invalid)

---

## Complete Run History (seed-fixed baseline onward)

| Run | Loss | Refine | temp | λ | Val NME | Test NME | Notes |
|-----|------|--------|------|---|---------|----------|-------|
| 12 | MSE | ✗ | — | — | 50.89% | — | True baseline (seed=42) |
| 13 | MSE | ✓ | — | — | 47.02% | — | RefinementHead alone: marginal |
| 14 | WMSE α=5 | ✗ | — | — | 26.91% | — | Loss weighting is the lever |
| 15 | WMSE | ✗ | — | — | 26.67% | — | 400 ep, confirmed converged |
| 16 | WMSE | ✓ | — | — | 26.43% | — | Refine+WMSE: negligible gain |
| 17 | AWing | ✗ | — | — | 21.32% | — | AWing > WMSE standalone |
| **18** | **WMSE+L1** | **✗** | **10** | **0.1** | **7.09%** | **13.66%** | **⭐ BEST — coord loss is key** |
| 19 | AWing+L1 | ✗ | 10 | 0.1 | 27.04% | — | AWing >> WMSE in magnitude; coord loss ~1% of total |
| 20c | WMSE+L1 | ✓ | 10 | 0.1 | 17.46% | — | Refine disrupts soft-argmax |
| 20d | WMSE+L1 | ✗ | **50** | 0.1 | ~18% | — | High temp → instability |
| 20e | WMSE+L1 | ✗ | 10 | **0.5** | ~18-19% | — | Large λ → heatmap learning disrupted |

### Conclusions from ablations
- Coordinate loss (soft-argmax + L1) is the single most important factor (51%→7%)
- WMSE is a good heatmap loss; AWing works standalone but magnitude-mismatches with L1
- RefinementHead hurts when combined with hybrid loss
- Optimal config is **Run 18**: temp=10, λ=0.1, no refinement, WMSE+L1

---

## Architecture

```
DINOv2-ViT-S/14 backbone (vit_small_patch14_reg4_dinov2)
  ↓ last 3 blocks with injected learnable queries (num_q=2)
  ↓ multi-layer prediction: intermediate (×3) + final = 4 prediction heads
  ↓ mask_head (3-layer MLP) → query features (B, Q, C)
  ↓ upscale (2× ConvTranspose2d) → patch features (B, C, H/4, W/4)
  ↓ einsum("bqc,bchw→bqhw") → similarity maps (B, Q, H/4, W/4)
  ↓ F.interpolate → 64×64 heatmaps
Loss: Weighted MSE (heatmap) + λ × L1(soft-argmax coords)
Metric: NME normalised by inter-landmark distance in image pixel space
```

Key settings:
- `num_blocks=3, masked_attn_enabled=true` → 4 prediction layers (loss averaged)
- `freeze_backbone=false` (LLRD: backbone blocks get `lr × 0.8^(11-i)`)
- `seed_everything: 42` in yaml
- `torch.backends.cudnn.deterministic = True` in main_landmark.py

---

## All Code Changes Made (vs original EoMT repo)

### `main_landmark.py` (new file)
```python
# In LandmarkCLI.__init__:
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```
Also links img_size/heatmap_size from data → model → encoder via `link_arguments`.

### `models/eomt.py`
Added `RefinementHead` class (shared-weight Conv/BN/ReLU ×3 + residual):
```python
class RefinementHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
        )
    def forward(self, x):
        B, Q, H, W = x.shape
        flat = x.reshape(B * Q, 1, H, W)
        return (self.net(flat) + flat).reshape(B, Q, H, W)
```
Added `use_refinement_head: bool = False` to `EoMT.__init__`.
In `_predict()`: after einsum, `if self.refinement_head is not None: mask_logits = self.refinement_head(mask_logits)`.

### `training/landmark_detection.py` (new file)
Key components:

```python
# --- Loss functions ---

def weighted_mse_loss(pred, target, alpha=5.0):
    weight = 1.0 + alpha * target
    return ((pred - target) ** 2 * weight).mean()

def spatial_softmax(heatmap: torch.Tensor, temperature: float = 10.0) -> torch.Tensor:
    """(B, Q, H, W) → (B, Q, 2) [x, y] in heatmap space"""
    B, Q, H, W = heatmap.shape
    device = heatmap.device
    x_coords = torch.arange(W, device=device, dtype=heatmap.dtype)
    y_coords = torch.arange(H, device=device, dtype=heatmap.dtype)
    weights = torch.softmax(heatmap.reshape(B, Q, -1) * temperature, dim=-1)
    weights = weights.reshape(B, Q, H, W)
    pred_x = (weights.sum(dim=2) * x_coords).sum(dim=-1)
    pred_y = (weights.sum(dim=3) * y_coords).sum(dim=-1)
    return torch.stack([pred_x, pred_y], dim=-1)

def hybrid_loss(pred_hm, target_hm, gt_coords, alpha=5.0, temperature=10.0):
    weight = 1.0 + alpha * target_hm
    L_hm = ((pred_hm - target_hm) ** 2 * weight).mean()
    pred_coords = spatial_softmax(pred_hm, temperature=temperature)
    L_coord = F.l1_loss(pred_coords, gt_coords)
    return L_hm, L_coord

class AdaptiveWingLoss(nn.Module):
    def __init__(self, omega=14, theta=0.5, alpha=2.1, epsilon=1):
        ...
    def forward(self, pred, target):
        delta = (target - pred).abs()
        alpha_t = self.alpha - target
        A = (self.omega * (1/(1+(self.theta/self.epsilon)**(alpha_t-1)))
             * alpha_t * ((self.theta/self.epsilon)**(alpha_t-2)) / self.epsilon)
        C = self.theta * A - self.omega * torch.log(1 + (self.theta/self.epsilon)**alpha_t)
        return torch.where(delta < self.theta,
                           self.omega * torch.log(1+(delta/self.epsilon)**alpha_t),
                           A * delta - C).mean()
```

`LandmarkDetection.__init__` signature:
```python
def __init__(self, network, img_size=(512,512), num_landmarks=2, heatmap_size=(64,64),
             attn_mask_annealing_enabled=False,
             loss_type="mse",       # "mse"|"weighted_mse"|"adaptive_wing"|"hybrid"|"hybrid_awing"
             alpha=5.0, temperature=10.0, lambda_coord=0.1,
             awing_omega=14.0, awing_theta=0.5, awing_alpha=2.1, awing_epsilon=1.0,
             lr=1e-4, llrd=0.8, llrd_l2_enabled=True, lr_mult=1.0,
             weight_decay=0.05, poly_power=0.9, warmup_steps=(500,1000), ...):
```

`training_step` always unpacks `imgs, gt_heatmaps, gt_coords = batch` (gt_coords: (B,N,2) [x,y] in heatmap space).

### `configs/landmark/bpd_vit_small.yaml`
- Switched logger from CSVLogger → **WandbLogger** (project: eomt-landmark)
- Added `seed_everything: 42` at top level
- Loss hyperparams: `loss_type`, `alpha`, `temperature`, `lambda_coord`, `awing_*`
- `use_refinement_head: false` in network section
- `attn_mask_annealing_enabled: false`

### `configs/landmark/ofd_vit_small.yaml`
- Same WandbLogger switch (task=ofd, not yet trained)

---

## Current Config State (server, after Run 20e)

```yaml
seed_everything: 42

trainer:
  max_epochs: 400
  check_val_every_n_epoch: 5
  log_every_n_steps: 1
  callbacks: [ModelCheckpoint, EarlyStopping(patience=20), LearningRateMonitor]
  logger:
    class_path: lightning.pytorch.loggers.WandbLogger
    init_args:
      project: eomt-landmark
      name: bpd-run20e-hybrid-wmse-lambda05

model:
  class_path: training.landmark_detection.LandmarkDetection
  init_args:
    num_landmarks: 2
    loss_type: hybrid          # WMSE + L1 coord
    alpha: 5.0
    temperature: 10.0
    lambda_coord: 0.5          # was 0.1 in Run 18
    awing_omega: 14.0
    ...
    use_refinement_head: false
    backbone_name: vit_small_patch14_reg4_dinov2

data:
  images_dir: /root/autodl-tmp/images/UCL/Head
  ann_train_csv: /root/autodl-tmp/annotations/UCL/Head_Train.csv
  ann_test_csv: /root/autodl-tmp/annotations/UCL/Head_Test.csv
  task: bpd
  img_size: [512, 512]
  heatmap_size: [64, 64]
  sigma: 4.0
  batch_size: 16
```

**Note:** Local repo yaml is at Run 20a state. Server yaml reflects Run 20e. Next run should restore lambda_coord: 0.1.

---

## Run 18 Checkpoint (Best)

```
/root/eomt/logs/eomt-landmark/ff0ku8a6/checkpoints/epochepoch=29-nmemetrics/val_nme=0.0709.ckpt
```

To evaluate on test set:
```bash
python main_landmark.py test \
  --config configs/landmark/bpd_vit_small.yaml \
  --ckpt_path /root/eomt/logs/eomt-landmark/ff0ku8a6/checkpoints/epochepoch=29-nmemetrics/val_nme=0.0709.ckpt
```

---

## Critical Pitfalls

### 1. backbone_name must be exact
```
vit_small_patch14_reg4_dinov2   ← CORRECT (timm name)
dinov2_vits14                   ← WRONG (causes "Unknown model" error)
```
After any sed corruption: `sed -i 's|backbone_name: .*|backbone_name: vit_small_patch14_reg4_dinov2|'`

### 2. sed for wandb run name MUST use 6-space indent
```bash
# CORRECT — matches only the wandb name field (6-space indent):
sed -i 's|      name: .*|      name: bpd-run21-xxxx|' configs/landmark/bpd_vit_small.yaml

# WRONG — also matches backbone_name:
sed -i 's|name: .*|name: bpd-run21-xxxx|'
```

### 3. Always verify before running
```bash
grep -E "backbone_name|      name:|loss_type:|temperature:|lambda_coord:|use_refinement_head:" \
  configs/landmark/bpd_vit_small.yaml
```

### 4. No sigmoid on logits
Raw logits go into loss functions. Sigmoid kills gradients and collapses heatmaps to near-flat.

### 5. Server code may be stale
Always `git pull` or manually check file state on server before running. Previously lost runs because server had code from a different run.

---

## Next Steps (priority order)

### 1. Grad-CAM text interference diagnosis (supervisor request, HIGH)
Check whether the model is attending to yellow text annotations in the ultrasound images rather than the anatomy. This would explain the gap between val NME (7%) and test NME (13.66%).

```python
# Use pytorch-grad-cam or captum
# Visualise attention maps for a few val samples
# Check if highlighted regions correspond to text labels vs anatomy
```

### 2. Wing Loss as coordinate branch (supervisor recommended, MEDIUM)
Replace L1 coord loss with Wing Loss (CVPR 2018: "Wing Loss for Robust Facial Landmark Localisation").
Wing Loss has larger gradients for small-to-medium errors than L1.

```python
def wing_loss(pred, target, w=10.0, epsilon=2.0):
    C = w - w * math.log(1 + w / epsilon)
    delta = (pred - target).abs()
    return torch.where(delta < w,
                       w * torch.log(1 + delta / epsilon),
                       delta - C).mean()
```
Config: `loss_type: hybrid`, replace `F.l1_loss(pred_coords, gt_coords)` with `wing_loss(pred_coords, gt_coords)`.
Suggested run: **Run 21**.

### 3. OFD training (MEDIUM)
Use Run 18 exact hyperparams on `ofd_vit_small.yaml`:
```bash
sed -i 's|      name: .*|      name: ofd-run01-hybrid-wmse|' configs/landmark/ofd_vit_small.yaml
sed -i 's|loss_type: .*|loss_type: hybrid|' configs/landmark/ofd_vit_small.yaml
sed -i 's|temperature: .*|temperature: 10.0|' configs/landmark/ofd_vit_small.yaml
sed -i 's|lambda_coord: .*|lambda_coord: 0.1|' configs/landmark/ofd_vit_small.yaml
python main_landmark.py fit --config configs/landmark/ofd_vit_small.yaml
```

### 4. Replace einsum mechanism (LATER)
The `einsum("bqc,bchw→bqhw")` computes query-patch *semantic similarity* — inherently diffuse, not a sharp heatmap.
Reference: TokenPose (Li et al. 2021) uses deconv-based heatmap generation.
Consider: dedicate backbone features (no einsum), use transposed conv decoder to generate per-landmark heatmaps.

### 5. 300W face landmark experiments (LATER)
Validate generalization of the approach on a public benchmark.

### 6. Paper writing (target MICCAI/MedIA/TMI, Sept 2026 submission)

---

## Server Access

```bash
ssh -p 42739 root@connect.westb.seetacloud.com
# Note: port number may change after instance restart — check AutoDL console
```

```bash
# Standard run command:
cd /root/eomt
python main_landmark.py fit --config configs/landmark/bpd_vit_small.yaml

# Template: set up a new run
NEW_NAME="bpd-run21-wing-loss"
sed -i "s|      name: .*|      name: $NEW_NAME|" configs/landmark/bpd_vit_small.yaml
grep -E "backbone_name|      name:|loss_type:|temperature:|lambda_coord:|use_refinement_head:" configs/landmark/bpd_vit_small.yaml
python main_landmark.py fit --config configs/landmark/bpd_vit_small.yaml
```

---

## Key File Locations (Server)

| Path | Description |
|------|-------------|
| `/root/eomt/` | Project root |
| `/root/autodl-tmp/images/UCL/Head/` | Training images |
| `/root/autodl-tmp/annotations/UCL/Head_Train.csv` | Train annotations |
| `/root/autodl-tmp/annotations/UCL/Head_Test.csv` | Test annotations |
| `/root/eomt/logs/eomt-landmark/` | W&B run logs |
| `/root/eomt/configs/landmark/bpd_vit_small.yaml` | BPD config |
| `/root/eomt/configs/landmark/ofd_vit_small.yaml` | OFD config |
| `/root/eomt/training/landmark_detection.py` | Loss functions, training loop |
| `/root/eomt/models/eomt.py` | EoMT architecture + RefinementHead |

---

## Restore Run 18 Config for Next Run

Before starting any new experiment, reset the yaml to the Run 18 optimal baseline:

```bash
sed -i 's|loss_type: .*|loss_type: hybrid|' configs/landmark/bpd_vit_small.yaml
sed -i 's|temperature: .*|temperature: 10.0|' configs/landmark/bpd_vit_small.yaml
sed -i 's|lambda_coord: .*|lambda_coord: 0.1|' configs/landmark/bpd_vit_small.yaml
sed -i 's|use_refinement_head: .*|use_refinement_head: false|' configs/landmark/bpd_vit_small.yaml
sed -i 's|backbone_name: .*|backbone_name: vit_small_patch14_reg4_dinov2|' configs/landmark/bpd_vit_small.yaml
grep -E "backbone_name|      name:|loss_type:|temperature:|lambda_coord:|use_refinement_head:" configs/landmark/bpd_vit_small.yaml
```
