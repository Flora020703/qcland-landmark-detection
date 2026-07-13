# CC Handover — EoMT Fetal Biometry Landmark Detection
*Last updated: 2026-06-19, after Run 10 completed*

---

## 1. 项目概述

**目标**：MSc 论文（UCL），把 EoMT（Encoder-only Mask Transformer）从语义分割改造为基于 heatmap 的胎儿双顶径/枕额径（BPD/OFD）关键点检测。

**导师**：Dr. Zhehua Mao, Dr. Sophia Bano（SRV group）

**要超过的 baseline（HRNet）**：
- Di Vece et al. 发表数字：BPD 8% (0.08±0.18)，OFD 5% (0.05±0.11)
- [2026-07-14 更正] 下面曾写的 "BPD 6.49% / OFD 4.41%" 是早期只用 10-20 张图跑通 pipeline 的产物，不是完整复现，不可信、不要引用对比。

**关键路径**：UCL Head 超声图像 → 2 个关键点（直径两端点） → Gaussian heatmap 监督 → 预测 heatmap → argmax 解码坐标 → NME 评估

**数据集**（服务器路径）：
- 图片：`/root/autodl-tmp/images/UCL/Head/`（JPEG）
- 标注：`/root/autodl-tmp/annotations/UCL/Head_Train.csv` / `Head_Test.csv`
- 数据量：~110 train（subject-split 后 ~100 train, 10 val），49 test

---

## 2. 环境

| 环境 | 配置 |
|------|------|
| 本地（开发调试） | Windows + WSL2，CPU only |
| 训练（AutoDL） | RTX 4090，PyTorch 2.8，CUDA 12.8，Python 3.12 |

**服务器运行命令**：
```bash
cd /root/eomt
export HF_ENDPOINT=https://hf-mirror.com
python main_landmark.py fit -c configs/landmark/bpd_vit_small.yaml \
    --data.init_args.images_dir /root/autodl-tmp/images/UCL/Head \
    --data.init_args.ann_train_csv /root/autodl-tmp/annotations/UCL/Head_Train.csv \
    --data.init_args.ann_test_csv /root/autodl-tmp/annotations/UCL/Head_Test.csv
```

---

## 3. 修改的文件清单

所有新增/改动文件：

| 文件 | 状态 | 说明 |
|------|------|------|
| `datasets/landmark_dataset.py` | **新建** | 数据加载、heatmap 生成 |
| `training/landmark_detection.py` | **新建** | LightningModule：MSE loss + NME metric |
| `models/eomt.py` | **改动** | 加 `freeze_backbone`、`upsample_bilinear` 参数 |
| `models/scale_block.py` | **改动** | 加 `use_bilinear` 参数 |
| `configs/landmark/bpd_vit_small.yaml` | **新建** | BPD 训练配置 |
| `configs/landmark/ofd_vit_small.yaml` | **新建** | OFD 训练配置 |
| `training/lightning_module.py` | **改动** | `configure_optimizers` 修复 lr_mult 生效 |

---

## 4. 当前各文件代码状态（关键部分）

### 4.1 `configs/landmark/bpd_vit_small.yaml`（当前正确配置）

```yaml
trainer:
  max_epochs: 200
  check_val_every_n_epoch: 5
  callbacks:
    - EarlyStopping: monitor=metrics/val_nme, mode=min, patience=20, min_delta=0.005
    - ModelCheckpoint: monitor=metrics/val_nme, mode=min, save_top_k=3

model:
  LandmarkDetection:
    num_landmarks: 2
    attn_mask_annealing_enabled: false
    lr: 2.0e-5          # ← Run 10 用的，可能太低（见第6节）
    llrd: 0.8
    llrd_l2_enabled: true
    lr_mult: 1.0
    weight_decay: 0.05
    poly_power: 0.9
    warmup_steps: [15, 30]
    network:
      EoMT:
        num_classes: 1
        num_q: 2
        num_blocks: 3           # ← 最后 3 个 block 插入 queries
        masked_attn_enabled: true  # ← 中间层预测作为 attention mask
        freeze_backbone: false     # ← 全参数训练
        upsample_bilinear: false   # ← ConvTranspose2d（原始）

data:
  HeadLandmarkDataModule:
    task: bpd
    img_size: [512, 512]
    heatmap_size: [64, 64]
    sigma: 4.0
    val_fraction: 0.1
    val_split_seed: 42
    batch_size: 16
```

`ofd_vit_small.yaml` 与 BPD 完全相同，只有 `task: ofd`。

---

### 4.2 `datasets/landmark_dataset.py` — 关键逻辑

**`generate_heatmap(x, y, heatmap_size, sigma=4.0)`**：
- 复刻 HRNet 的 Gaussian blob 生成
- `tmp_size = int(sigma * 3)`，sigma=4.0 → 24px window
- peak=1.0，出界直接截断（不全零）

**`LandmarkDataset.__getitem__()` 流程**：
1. 加载图片 PIL，获取原始 `orig_w, orig_h`
2. resize 到 `img_size`（512×512）
3. 等比缩放 landmarks 到 img_size 像素空间
4. 随机水平翻转（train only，prob=0.5）
5. **DOD sort**：按 x 坐标升序排列 → channel 0=左端点，channel 1=右端点（翻转后保持一致）
6. 换算到 heatmap 坐标空间（/8）
7. 生成 heatmaps（N, 64, 64）
8. `TF.to_tensor()` → [0,1] float32
9. Color jitter（train only，prob=0.5 each）：brightness±0.2，contrast±0.2，saturation±0.1

**返回 3-tuple**：`(img_t, heatmaps, lms_hm_float)`
- `lms_hm_float`：GT 关键点在 heatmap 空间的精确浮点坐标（用于 NME，无量化误差）

**已移除的增强**（曾经加过，后来删掉）：
- ~~文字区域涂黑~~（底部15%+右侧25%）— 把 BPD 右端点盖掉了
- ~~Scale jitter [0.6, 1.0]~~— 关键点越界
- ~~Rotation ±15°~~

---

### 4.3 `training/landmark_detection.py` — 关键逻辑

**`heatmap_to_coords(heatmaps)`**：
- argmax + ±0.25 sub-pixel refinement（HRNet 风格，边缘不 refine）
- 返回 (B, N, 2) 在 heatmap 空间

**`compute_nme(pred, gt, heatmap_size, img_size)`**：
- 先 ×8 换算到图像像素空间
- normaliser = `||landmark_0 - landmark_1||`（直径长度，图像像素）
- `nme = mean(||pred - gt||) / diameter`

**`training_step`**：
- `F.interpolate(mask_logits, heatmap_size, bilinear)` → 64×64
- **NO sigmoid**（raw logits MSE）
- 多层 loss 取均值：`total_loss / n_layers`
- num_blocks=3, masked_attn=true → 4 层（3 intermediate + 1 final）

**`eval_step`**：
- 只用最后一层（`mask_logits_per_layer[-1]`）做 NME
- **NO sigmoid**
- GT 直接用 3-tuple 第三个元素（精确 float，非 argmax 解码）

---

### 4.4 `models/eomt.py` — 关键改动

```python
def __init__(self, ..., freeze_backbone: bool = True, upsample_bilinear: bool = False):
    ...
    self.upscale = nn.Sequential(
        *[ScaleBlock(embed_dim, use_bilinear=upsample_bilinear) for _ in range(num_upscale)]
    )
    if freeze_backbone:
        # 冻结前 75% blocks，解冻后 25% + norm
        freeze_until = int(n_blocks * 0.75)  # e.g. 9/12 for ViT-S
```

**当前配置**：`freeze_backbone=false`（全参数训练），`upsample_bilinear=false`（ConvTranspose2d）

`_predict()` 不变（原始 EoMT 逻辑）：
```python
mask_logits = torch.einsum("bqc, bchw -> bqhw", self.mask_head(q), self.upscale(x))
```

`forward()` 流程：block 9 前插入 queries，block 9/10/11 各生成一次 intermediate prediction + attn_mask，block 12 后生成 final prediction → 共 4 层。

---

### 4.5 `models/scale_block.py` — 改动

```python
class ScaleBlock(nn.Module):
    def __init__(self, embed_dim, conv1_layer=nn.ConvTranspose2d, use_bilinear=False):
        if use_bilinear:
            self.conv1 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)
        else:
            self.conv1 = conv1_layer(embed_dim, embed_dim, kernel_size=2, stride=2)
```

`use_bilinear=True`：先 `F.interpolate(scale_factor=2, bilinear)` 再 Conv2d（消除棋盘格 artifact）  
`use_bilinear=False`（当前）：原始 ConvTranspose2d

---

### 4.6 `training/lightning_module.py` — 关键改动

`configure_optimizers` 修复（原来 lr_mult 是死代码）：

```python
# Head params (q, mask_head, upscale, class_head)
other_param_groups.append({"params": [param], "lr": self.lr * self.lr_mult, ...})
```

LLRD 规则：
- backbone block i：`lr * 0.8^(11 - i)`（block 11 满 lr，越早越小）
- l2_blocks（9,10,11）当 `llrd_l2_enabled=true` 且 `lr_mult=1.0` 时：**不**重置为满 lr，仍用 LLRD 衰减
- backbone.norm：重置为满 lr
- head params：`lr * lr_mult = lr * 1.0 = lr`（与 block 11 相同）

**注意**：`LandmarkDetection.forward()` 覆盖了基类的 `x = imgs / 255.0`，直接传 [0,1] tensor 给 EoMT（EoMT 内部自己做 ImageNet 归一化）。

---

## 5. 训练历史（所有 Run）

| Run | v_num | 关键配置差异 | 最佳 NME | 停止 epoch | 问题 |
|-----|-------|-------------|---------|-----------|------|
| Run 1 | ? | 早期版本 | ? | ? | 未知 |
| **Run 2** | 1 | num_blocks=3, masked_attn=true, lr=1e-4 | **17.68%** | ? | **目标！** |
| Run 3 | ? | 未知 | ~45% | ? | ? |
| Run 4 | ? | sigmoid 加回来了 | **125.16%（stuck）** | 全程 | sigmoid 梯度消失 |
| Run 5 | ? | 服务器还是旧代码（sigmoid 未删） | 125.16% | 全程 | 服务器 tar 打包时机错误 |
| Run 6 | ? | partial freeze(75%)+num_blocks=3+sigma=4 | 45.51% | ~50ep | 冻结早期 block 不适应 fetal US |
| Run 7 | ? | 加了文字区域涂黑（bottom15%+right25%） | 56.39% | ~50ep | 涂黑把 BPD 右端点遮了 |
| Run 8 | 9 | num_blocks=1, masked_attn=false（配置写错了！）, lr=1e-4, patience=10 | 39.91%@ep14 | ep69 | 配置用错了——用户最初把 Run 2 的 num_blocks 告诉 CC 时写错了 |
| Run 9 | 10 | 服务器仍是 Run 8 配置（tar 未更新） | ~42.91%@ep19 | ep69 | 证据：loss 与 Run 8 像素级相同；ep69 停止=patience=10 |
| Run 10 | — | 未记录（v_num 跳号） | — | — | — |
| **Run 11** | 11 | num_blocks=3, masked_attn=true, lr=**2e-5**, patience=20 | **46.41%@ep129** | ep184（用户中断） | train/loss 卡在 0.008-0.009 不降；比 Run 8 (lr=1e-4) 还差 |

---

## 6. 已修复的 Bug 列表

### Bug 1：Sigmoid 梯度消失（Run 4/5）
**症状**：NME 每轮都是 125.16%，完全不动。  
**原因**：`training_step` 和 `eval_step` 里加了 `torch.sigmoid(pred)`。模型初始化时 mask_logits 强负 → sigmoid≈0 → MSE gradient≈0 → 训练不动。  
**修复**：删掉两处 `torch.sigmoid()`，对 raw logits 直接算 MSE。  
**当前状态**：`landmark_detection.py` 已无 sigmoid。

### Bug 2：服务器运行旧代码（Run 5, 9, 10 前）
**症状**：结果与上一轮完全相同，或 loss 不符合新 config 预期。  
**原因**：tar.gz 在本地代码修改前就打包了，上传到服务器的是旧版。  
**诊断方法**：
```bash
grep -E "num_blocks|masked_attn|^    lr:|patience" /root/eomt/configs/landmark/bpd_vit_small.yaml
```
**修复方法**：直接 sed 改服务器文件（比重新上传 tar 更可靠）：
```bash
sed -i 's/num_blocks: 1/num_blocks: 3/' configs/landmark/bpd_vit_small.yaml
sed -i 's/masked_attn_enabled: false/masked_attn_enabled: true/' configs/landmark/bpd_vit_small.yaml
sed -i 's/lr: 1.0e-4/lr: 2.0e-5/' configs/landmark/bpd_vit_small.yaml
sed -i 's/patience: 10/patience: 20/' configs/landmark/bpd_vit_small.yaml
```

### Bug 3：文字区域涂黑把 BPD 右端点遮住（Run 7）
**症状**：NME 骤升至 56%（之前 45%）。  
**原因**：`img_arr[:, int(orig_w * 0.75):] = 0` 把图像右 25% 涂黑，而 BPD 右端点（颅骨右缘）经常落在 x=380-410px（512px 图像的 74-80% 位置）→ GT heatmap 有峰值，但图像对应区域全黑 → 训练信号冲突。  
**修复**：完全删除文字区域涂黑。  
**当前状态**：无任何涂黑。

### Bug 4：Scale jitter 导致关键点越界（曾加过）
**原因**：`crop_scale ∈ [0.6, 1.0]` 时，原图 x=400 → 等效到 x=667 → clip 到 511（图像边缘）→ GT heatmap 位置错误。  
**修复**：删除 scale jitter。

### Bug 5：Run 2 配置在 prompt 里写错了（关键！）
**原因**：用户在向 CC 描述 Run 2 时，错误写成 `num_blocks=1, masked_attn=false`，导致 Run 8 用了错误配置，一直没能复现 Run 2。  
**正确 Run 2 配置**：`num_blocks=3, masked_attn=true, lr=1e-4`。  
**影响**：Run 8 最好只能到 39.91%，Run 9 服务器代码未更新也是 ~42%。

### Bug 6：lr_mult 在 lightning_module.py 是死代码（已修复）
**原因**：`configure_optimizers` 里 head params 本来用的是 `lr` 而不是 `lr * lr_mult`。  
**修复**：`other_param_groups.append({"params": [param], "lr": self.lr * self.lr_mult, ...})`

---

## 7. 当前问题分析：为什么 Run 11 (46%) 远比 Run 2 (17.68%) 差？

### Run 11 详细 epoch 数据（v_num=11）

| Epoch | train/loss | val NME |
|-------|-----------|---------|
| 114 | 0.00855 | 46.57% |
| 119 | 0.00874 | 51.70% |
| 124 | 0.00869 | 51.26% |
| **129** | **0.00848** | **46.41%** ← 最佳 |
| 134 | 0.00853 | 51.28% |
| 139 | 0.00861 | 55.85% |
| 144 | 0.00837 | 55.82% |
| 149 | 0.00858 | 51.00% |
| 154 | 0.00884 | 51.04% |
| 159 | 0.00815 | 51.11% |
| 164 | 0.00841 | 51.02% |
| 169 | 0.00845 | 55.85% |
| 174 | 0.00831 | 46.81% |
| 179 | 0.00787 | 51.03% |
| 184 | 0.00826 | 51.03% [中断] |

**规律**：NME 在 3 个簇之间循环（46%/51%/56%），说明 10 个 val 样本里有 1-2 个在每次 checkpoint 间反复切换对错。train/loss 从 ep100+ 开始不再下降，说明模型已收敛到局部最优。

### 根本原因假设

**lr=2e-5 太低，导致模型收敛到更差的局部最优**

证据：
1. Run 8 (lr=1e-4) 最终 train/loss ~0.0071，比 Run 11 (lr=2e-5) 的 ~0.008 还低——更低的 lr 反而 train/loss 更高，说明模型没有充分探索损失景观
2. Run 11 train/loss 从 epoch 100+ 就不再下降（已收敛），而 Run 8 在全程持续下降
3. Run 2 (lr=1e-4) 能到 17.68% 的原因可能是：高 lr 帮助模型快速找到好的局部最优，在 val NME 过拟合上升前已经到位

**第二个可能**：Run 2 最佳 checkpoint 出现在很早的 epoch（如 ep10-20），当时训练才刚开始，val NME 还没因过拟合恶化。lr=2e-5 时模型到那个 epoch 还未充分收敛，错过了最优窗口。

### 对比

| | Run 8 | Run 11 | Run 2 |
|--|-------|--------|-------|
| lr | 1e-4 | **2e-5** | 1e-4 |
| num_blocks | 1 | 3 | 3 |
| masked_attn | false | true | true |
| 最佳 NME | 39.91% | 46.41% | **17.68%** |
| 最终 train/loss | 0.00712 | ~0.008 | ? |

**结论**：Run 11 用了 Run 2 的 num_blocks/masked_attn（对的），但 lr 从 1e-4 降到了 2e-5（错的）。

---

## 8. 下一步行动（优先级顺序）

### 立即要做（Run 12）

Run 11 已确认 lr=2e-5 导致收敛到更差局部最优。**只改一个变量：将 lr 恢复为 1e-4**，其他保持 Run 11 配置：

在服务器上直接改：
```bash
# 服务器直接改
sed -i 's/lr: 2.0e-5/lr: 1.0e-4/' /root/eomt/configs/landmark/bpd_vit_small.yaml

# 验证（改前必做）
grep -E "num_blocks|masked_attn|^    lr:|patience" /root/eomt/configs/landmark/bpd_vit_small.yaml
# 期望：num_blocks: 3, masked_attn_enabled: true, lr: 1.0e-4, patience: 20
```

**Run 12 完整配置（目标：复现 Run 2 的 17.68%）**：
```yaml
lr: 1.0e-4          # ← 恢复为 Run 2 的值（Run 11 是 2e-5，太低了）
num_blocks: 3
masked_attn_enabled: true
freeze_backbone: false
patience: 20        # 新增（Run 2 没有，但需要避免无限训练）
check_val_every_n_epoch: 5
warmup_steps: [15, 30]
```

预期行为：train/loss 前 20 epoch 应该快速下降（比 Run 11 快 5×），val NME 应该在 epoch 10-30 出现明显低谷（类似 Run 8 的 39.91%@ep14，但因为 masked_attn=true 应该更低）。

### 如果 Run 12 还没到 17.68%

排查方向：

1. **EarlyStopping 过早停止？**  
   Run 2 没有 EarlyStopping，而 Run 11 有（patience=20→100 epoch window）。如果 Run 2 的最佳出现在 epoch 200 附近，patience=20 会过早停止。临时把 patience 调大到 40 或去掉 EarlyStopping 来排除。

2. **warmup 不同？**  
   Run 2 的 warmup_steps 可能是默认值 `[500, 1000]` 而不是 `[15, 30]`。`[500, 1000]` 以 step 为单位，即 ~83 epoch 和 ~167 epoch 的 warmup——对于 200 epoch 总训练来说很不同。检查原始 segmentation config 的 warmup_steps 值。

3. **weight_decay=0.05 太强？**  
   100 个训练样本，0.05 的 weight decay 可能过度正则化。尝试 `weight_decay: 0.01`。

4. **NME 计算方式变化？**  
   Run 2 可能用的是 argmax 解码的 GT（量化到 heatmap 格子），现在用的是精确 float GT。差距约 2-3%，但不能解释 17.68% vs 46% 的巨大差距。

### 达到 <17.68% 后

1. 用最佳 BPD checkpoint 在 test set 上评估：  
   ```bash
   python main_landmark.py test -c configs/landmark/bpd_vit_small.yaml \
       --ckpt_path logs/lightning_logs/version_XX/checkpoints/best.ckpt
   ```

2. 训练 OFD（`ofd_vit_small.yaml` 已准备好）

3. Ablation 实验（论文需要）：
   - `masked_attn_enabled: false`（去掉 attention guidance）
   - `num_blocks: 1` vs `3`
   - sigma 对比（2.0 vs 4.0）
   - `upsample_bilinear: true` vs `false`

---

## 9. 重要提醒（避免踩过的坑）

1. **每次运行前，先在服务器 grep 验证 config**，不要相信 tar 上传成功就是代码正确：
   ```bash
   grep -E "num_blocks|masked_attn|^    lr:|patience" /root/eomt/configs/landmark/bpd_vit_small.yaml
   ```

2. **训练过程中观察两个信号**：
   - `train/loss` 是否还在下降（如果卡住说明 lr 太低或已收敛）
   - `metrics/val_nme` 是否有改善趋势（10 个 val 样本，NME 方差大，单次跳动属正常）

3. **v_num 与 Run 的对应**：
   - v_num=1 = Run 2（17.68%，目标）
   - v_num=9 = Run 8（39.91%，lr=1e-4 但 num_blocks=1 错误配置）
   - v_num=10 = Run 9（服务器跑的旧代码，~42%）
   - v_num=11 = Run 11（最新，lr=2e-5 太低，~46%）
   - v_num=12 = Run 12（下一个，lr=1e-4 恢复）
   - 每次新 run 自动递增

4. **不要对 raw logits 加 sigmoid**：loss 和 eval 里都是 raw MSE，这是刻意设计，加 sigmoid 会梯度消失。

5. **NME 的值域理解**：
   - 随机预测 NME ≈ 50-80%（依赖直径长度）
   - 17.68% 是 Run 2 达到的，当时仍比这里写的 HRNet 数字（6.49%）差很多
   - [2026-07-14 更正] "6.49%" 这个数字本身不可信（见第 1 节更正说明），真正要比的是 Di Vece 发表的 BPD 8%

---

## 10. 文件路径速查

```
eomt/
├── configs/landmark/
│   ├── bpd_vit_small.yaml     # BPD 训练配置
│   └── ofd_vit_small.yaml     # OFD 训练配置
├── datasets/
│   └── landmark_dataset.py    # 数据加载 + heatmap 生成
├── training/
│   ├── landmark_detection.py  # LightningModule: MSE loss + NME
│   └── lightning_module.py    # 基类: optimizer/scheduler/LLRD
├── models/
│   ├── eomt.py                # 主模型（改了 freeze_backbone, upsample_bilinear）
│   └── scale_block.py         # 上采样块（改了 use_bilinear）
└── main_landmark.py           # 训练入口
```

**HRNet 参考代码**（baseline）：
```
/mnt/d/download/Project coding/msc/Muti/Multicentre-Fetal-Biometry/
  lib/datasets/fetal.py          # heatmap 生成参考
  lib/core/evaluation.py         # NME 计算参考
  lib/utils/transforms.py        # generate_target() Gaussian blob
```
