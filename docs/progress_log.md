# EoMT Landmark Detection — Progress Log
# EoMT 地标检测改造进度记录

> 对话时间 / Session date: 2026-06-12  
> 分支 / Branch: `master`  
> 基线参考 / Baseline reference: HRNet — BPD NME=0.0649, OFD NME=0.0441

---

## 项目背景 / Project Background

| | 中文 | English |
|---|---|---|
| **目标** | 将 EoMT（Encoder-only Mask Transformer）从语义分割改造为基于 Gaussian heatmap 的胎儿超声双径线（BPD/OFD）关键点检测 | Adapt EoMT from semantic segmentation to Gaussian heatmap-based landmark detection for fetal biometry (BPD/OFD) |
| **数据** | UCL Head 数据集：159 张图像，标注来自 `Head_Train.csv`（110 条）/ `Head_Test.csv`（49 条） | UCL Head dataset: 159 images, annotations from Head_Train.csv (110) / Head_Test.csv (49) |
| **关键差异** | 不需要 Hungarian matching，query 与 landmark 1-to-1 固定对应；loss 从 BCE+Dice+CE 改为 MSE on heatmaps；metric 从 mIoU 改为 NME | No Hungarian matching needed; fixed 1-to-1 query-to-landmark; loss changed from BCE+Dice+CE to MSE on heatmaps; metric changed from mIoU to NME |

---

## Step 1 — 新建 `datasets/landmark_dataset.py`

### 做了什么 / What was done

- 从头编写数据加载模块，**不复用**现有 `Dataset`（绑定分割 mask）和 `Transforms`（绑定 `tv_tensors.Mask`）
- 读取 CSV 标注，为每个 landmark 生成 64×64 Gaussian heatmap
- 按 subject（患者）分组划分 train/val，防止同一患者的多帧超声图像分别出现在 train 和 val（防数据泄漏）

- Written from scratch; does NOT reuse existing `Dataset` (tied to segmentation masks) or `Transforms` (tied to `tv_tensors.Mask`)
- Reads CSV annotations and generates 64×64 Gaussian heatmaps per landmark
- Splits train/val by subject group to prevent same-patient leakage across splits

### 新建文件 / New file

**`datasets/landmark_dataset.py`**

| 函数/类 | 作用 |
|---------|------|
| `generate_heatmap(x, y, heatmap_size, sigma)` | 生成单个 Gaussian heatmap，完全复现 HRNet `generate_target()` 的整数中心+截断 patch 逻辑；σ=1.0，peak=1.0 |
| `LandmarkDataset` | `torch.utils.data.Dataset`；每条 record 含 img_path + landmarks（原始像素坐标）；`__getitem__` 做：resize→scale coords→hflip(aug)→DOD sort→heatmap gen→to_tensor；返回 `(img:[3,512,512], heatmaps:[N,64,64])` |
| `HeadLandmarkDataModule` | `lightning.LightningDataModule`；解析 CSV，按 subject 分组 split，暴露 train/val/test 三个 DataLoader；自定义 `collate` 函数 stack 图像和 heatmap |

### 关键设计决策 / Key design decisions

| 决策 | 说明 |
|------|------|
| **img_size = (512, 512)** | 与 EOMT 分割 baseline 一致，ViT patch_size=14 时得到 36×36 patch grid |
| **Val split by subject** | 同一患者多帧相似度极高，random split 会导致 NME 虚高 |
| **DOD sort（x 升序）** | channel 0 = 左端点（x 较小），channel 1 = 右端点；解决水平翻转增强后 query-landmark 身份翻转的问题 |
| **task 参数化** | `task="bpd"` 或 `"ofd"` 选择 CSV 中对应的 4 列坐标 |
| **归一化移除** | Dataset 返回 `[0,1]` float tensor，**不做 ImageNet 归一化**；EoMT encoder（`models/vit.py`）内部通过 `pixel_mean/pixel_std` buffer 完成归一化，如果 dataset 再做一次会 double-normalize |

### 数据加载 smoke-test 结果 / Data loading smoke-test results

```
[generate_heatmap] OK
[task=bpd]  split: train=100  val=10  test=49
            img=(3,512,512)  heatmaps=(2,64,64)  hm_max=1.0  hm_min=0.0
            DOD sort OK (channel-0 x ≤ channel-1 x in all batch items)
[task=ofd]  (same)
Landmark data loading OK
```

---

## Step 2 — 新建 `training/landmark_detection.py`

### 做了什么 / What was done

- 继承现有 `LightningModule` 基类，复用 LLRD optimizer 和 TwoStageWarmupPoly scheduler
- 用 MSE loss 替代 Hungarian matching + BCE/Dice/CE
- 实现 argmax + sub-pixel refinement 坐标提取（HRNet `decode_preds` 风格）
- 实现 NME（归一化因子 = 两 landmark 之间的欧氏距离）

- Inherits existing `LightningModule` base; reuses LLRD optimizer + TwoStageWarmupPoly scheduler
- Replaces Hungarian matching + BCE/Dice/CE with direct MSE on heatmaps
- Implements argmax + sub-pixel refinement coord extraction (HRNet `decode_preds` style)
- Implements NME (normalised by inter-landmark Euclidean distance)

### 新建文件 / New file

**`training/landmark_detection.py`**

| 函数/类 | 作用 |
|---------|------|
| `heatmap_to_coords(heatmaps)` | `(B,N,H,W)` → `(B,N,2)` heatmap 像素坐标；argmax 定位峰值，±0.25px sub-pixel refinement（向高值邻居方向偏移） |
| `compute_nme(pred, gt, hm_size, img_size)` | heatmap 坐标 → image 像素坐标 → NME per sample；normalizer = GT landmark 0 与 landmark 1 的欧氏距离（即直径长度） |
| `LandmarkDetection` | `LightningModule` 子类；`forward` override（跳过基类 `/255`，直接传 `[0,1]` 给 network）；`training_step`（对每层 MSE，平均）；`eval_step`（final layer MSE + NME）；`on_validation_epoch_end`（log + print） |

### forward 归一化说明 / forward normalization

```
Base LightningModule.forward:  x = imgs / 255.0   ← 期望 uint8 输入
LandmarkDetection.forward:     return self.network(imgs)  ← 跳过 /255，直接传 [0,1]
EoMT.forward:                  x = (x - pixel_mean) / pixel_std  ← 正确归一化
```

### 训练 utils smoke-test 结果 / Training utils smoke-test results

```
[heatmap_to_coords] integer spike → coords OK
[compute_nme]       perfect prediction → NME=0 OK
[compute_nme]       4-px shift → NME=0.0447 (expected 0.0447) OK
Training utils OK
```

---

## Step 3 — 修改 `models/eomt.py`

### 做了什么 / What was done

在 `EoMT` 构造函数加 `heatmap_size` 可选参数；在 `_predict()` 末尾加 bilinear resize。
原有分割代码**完全不受影响**（`heatmap_size=None` 时逻辑不变）。

Added optional `heatmap_size` parameter to `EoMT.__init__`; appended bilinear resize at end of `_predict()`.
Existing segmentation code is **fully unaffected** (`heatmap_size=None` → no resize, identical behavior).

### 改动 / Changes to `models/eomt.py`

**`__init__` 新增一个参数：**
```python
# 改动前 / Before
def __init__(self, encoder, num_classes, num_q, num_blocks=4, masked_attn_enabled=True):

# 改动后 / After
def __init__(self, encoder, num_classes, num_q, num_blocks=4, masked_attn_enabled=True,
             heatmap_size: Optional[tuple[int,int]] = None):  # MODIFIED
    ...
    self.heatmap_size = heatmap_size  # MODIFIED
```

**`_predict()` 新增三行（在 `return` 前）：**
```python
# MODIFIED: resize to fixed heatmap_size when set (landmark detection).
# upscale naturally outputs patch_grid×2 (72×72 for ViT-S/14 at 512px input).
if self.heatmap_size is not None:
    mask_logits = F.interpolate(
        mask_logits, self.heatmap_size, mode="bilinear", align_corners=False
    )
```

### 为什么是 72×72 → 64×64 / Why 72×72 → 64×64

```
ViT-S, patch_size=14, input 512×512:
  patch grid = floor(512/14) = 36×36
  num_upscale = max(1, int(log2(14))-2) = 1
  ScaleBlock (ConvTranspose2d stride=2): 36×36 → 72×72
  bilinear resize to heatmap_size=(64,64): 72×72 → 64×64
```

---

## Step 4 — 新建 config 和入口脚本

### 做了什么 / What was done

- 新建独立训练入口 `main_landmark.py`（不复用 `main.py`，因为后者的 `link_arguments` 期望 `data.num_classes`，而 `HeadLandmarkDataModule` 没有这个字段）
- 新建 BPD 和 OFD 两个 YAML 配置

- Created standalone training entry `main_landmark.py` (does not reuse `main.py` whose `link_arguments` expects `data.num_classes`, absent in `HeadLandmarkDataModule`)
- Created YAML configs for BPD and OFD

### 新建文件 / New files

| 文件 | 说明 |
|------|------|
| `main_landmark.py` | `LandmarkCLI`：link `img_size`（data→model+encoder）、`heatmap_size`（data→model+EoMT）、`ckpt_path`（model→encoder）；保留 `torch.compile` 逻辑 |
| `configs/landmark/bpd_vit_small.yaml` | BPD，DINOv2-ViT-S，512×512，64×64 heatmap，`masked_attn_enabled=false` |
| `configs/landmark/ofd_vit_small.yaml` | OFD，其余与 BPD 完全相同 |

### 关键 config 参数 / Key config parameters

```yaml
model:
  num_landmarks: 2
  heatmap_size: [64, 64]        # linked from data
  attn_mask_annealing_enabled: false
  lr: 1.0e-4
  warmup_steps: [100, 200]      # ~1-2 epochs for 100-sample train set

  network (EoMT):
    num_classes: 1              # class_head 保留但 loss 中 ignore
    num_q: 2                    # 每个 landmark 一个 query
    num_blocks: 1               # LLRD 目标块数
    masked_attn_enabled: false  # 初始关闭，后续作为 ablation 开启

  encoder (ViT-S):
    backbone_name: vit_small_patch14_reg4_dinov2

data:
  task: bpd / ofd
  img_size: [512, 512]
  heatmap_size: [64, 64]
  sigma: 1.0
  val_fraction: 0.1
  batch_size: 16
  num_workers: 4
```

---

## CPU End-to-End Sanity Check

### 测试代码 / Test command
```python
encoder = ViT(img_size=(512,512), backbone_name="vit_small_patch14_reg4_dinov2")
network = EoMT(encoder=encoder, num_classes=1, num_q=2,
               num_blocks=1, masked_attn_enabled=False, heatmap_size=(64,64))
model = LandmarkDetection(network=network, img_size=(512,512),
                          num_landmarks=2, heatmap_size=(64,64))
# fake batch: imgs [0,1] float32, gt_heatmaps with spike at known coords
# forward → loss → backward
```

### 结果 / Results

```
Model instantiation : OK
Output layers       : 1     (masked_attn_enabled=False → single final prediction)
mask shape          : torch.Size([2, 2, 64, 64])   ✓
MSE loss            : 1.559835                      ✓ (reasonable for random init)
Backward            : OK                            ✓
=== CPU sanity check PASSED ===
```

---

## 文件变更汇总 / File Change Summary

### 新建文件 / New files

| 文件 | 说明 |
|------|------|
| `datasets/landmark_dataset.py` | 数据加载、heatmap 生成、LightningDataModule |
| `training/landmark_detection.py` | MSE loss、NME metric、LightningModule |
| `main_landmark.py` | 独立训练入口（LightningCLI） |
| `configs/landmark/bpd_vit_small.yaml` | BPD 训练配置 |
| `configs/landmark/ofd_vit_small.yaml` | OFD 训练配置 |
| `docs/landmark_dataset_design.md` | 数据模块设计文档 |
| `docs/progress_log.md` | 本文件 |

### 修改文件 / Modified files

| 文件 | 改动内容 |
|------|----------|
| `models/eomt.py` | `__init__` 加 `heatmap_size=None` 参数（1行）；`_predict()` 加 bilinear resize（3行）；原分割代码零影响 |
| `datasets/landmark_dataset.py` | 移除 `TF.normalize`（归一化改由 EoMT encoder 内部完成）；移除 `_IMAGENET_MEAN/STD` 常量 |
| `test_dataload.py` | 加入 `test_landmark()` 和 `test_training_utils()` 两个 smoke-test 函数；加 `landmark` / `training` 命令行参数 |

### 未改动文件 / Untouched files

```
models/vit.py           — pixel_mean/pixel_std 归一化逻辑不变
models/scale_block.py   — 无需改动
training/lightning_module.py     — 基类不变，configure_optimizers 直接复用
training/mask_classification_*.py — 分割训练逻辑不变
datasets/ade20k_semantic.py       — 不变
datasets/dataset.py               — 不变
main.py                           — 不变（分割入口）
```

---

## 当前项目状态 / Current Project Status

```
Phase 1 (已完成 / Done):
  ✅ EoMT 分割 pipeline 在 ADE20K_mini 验证通过（loss 下降，5 epoch）

Phase 2 (本次对话完成 / Completed this session):
  ✅ Step 1: datasets/landmark_dataset.py — 数据加载验证通过
  ✅ Step 2: training/landmark_detection.py — MSE + NME 数值验证通过
  ✅ Step 3: models/eomt.py 最小化修改 — heatmap_size 参数
  ✅ Step 4: configs/landmark/ + main_landmark.py
  ✅ CPU end-to-end sanity check PASSED

Phase 2 (待完成 / Next):
  ⬜ AutoDL GPU 训练 (BPD first)
  ⬜ 验证 val_nme 收敛，对比 HRNet baseline (BPD 0.0649, OFD 0.0441)
  ⬜ OFD 训练

Phase 3 (后续 ablations / Future ablations):
  ⬜ masked_attn_enabled=True + annealing (multi-layer supervision)
  ⬜ DOD GMM-based reassignment
  ⬜ Scale jitter / random crop augmentation
  ⬜ ViT-B / ViT-L backbone scaling
```

---

## AutoDL 启动命令 / AutoDL Launch Command

```bash
# BPD
python main_landmark.py fit \
    -c configs/landmark/bpd_vit_small.yaml \
    --data.init_args.images_dir /root/autodl-tmp/UCL/Head \
    --data.init_args.ann_train_csv /root/autodl-tmp/annotations/UCL/Head_Train.csv \
    --data.init_args.ann_test_csv /root/autodl-tmp/annotations/UCL/Head_Test.csv \
    --compile_disabled   # 先关 compile，稳定后再开

# OFD（训练完 BPD 后）
python main_landmark.py fit \
    -c configs/landmark/ofd_vit_small.yaml \
    --data.init_args.images_dir /root/autodl-tmp/UCL/Head \
    --data.init_args.ann_train_csv /root/autodl-tmp/annotations/UCL/Head_Train.csv \
    --data.init_args.ann_test_csv /root/autodl-tmp/annotations/UCL/Head_Test.csv \
    --compile_disabled
```

---

## 注意事项 / Notes

1. **归一化链路**：dataset → `[0,1]` → `LandmarkDetection.forward` → `EoMT.forward (x-mean)/std` → 正确。任何中间环节都不能再做 ImageNet 归一化。

   **Normalization chain**: dataset → `[0,1]` → `LandmarkDetection.forward` → `EoMT.forward (x-mean)/std` → correct. No extra ImageNet normalization at any intermediate step.

2. **`class_head` 保留**：EoMT 中 `class_head` 仍然存在（输出 `class_logits`），`LandmarkDetection.training_step` 用 `_` 忽略它，loss 里完全不涉及。

   **`class_head` kept**: `class_head` remains in EoMT (outputs `class_logits`), ignored via `_` in `LandmarkDetection.training_step`, never enters the loss.

3. **DOD sort**：每个 batch item 的 heatmap channel 0 始终对应 x 坐标较小的端点（左），channel 1 对应右端点。这是一个隐式的 landmark identity 约定，后续所有代码（NME 计算、可视化）都依赖此顺序。

   **DOD sort**: heatmap channel 0 always corresponds to the left endpoint (lower x), channel 1 to the right. This is an implicit landmark identity convention that all downstream code (NME, visualisation) depends on.

4. **`masked_attn_enabled=False`**：当前配置关闭了 masked attention，因此 `mask_logits_per_layer` 只有 1 个元素（最终层预测）。开启后可获得多层监督，预计有提升，作为后续 ablation。

   **`masked_attn_enabled=False`**: current config disables masked attention, so `mask_logits_per_layer` has only 1 element (final prediction). Enabling it enables multi-layer supervision, expected to improve NME — deferred to ablation study.
