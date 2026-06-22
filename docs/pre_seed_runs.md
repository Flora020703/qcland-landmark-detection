# Pre-Seed Exploratory Training Runs

> 记录时间：2026-06-23
> 说明：以下 11 轮实验未固定 random seed，用于探索架构和超参空间。后续实验将固定 seed 并使用 W&B 记录。

---

## 实验环境

- GPU: AutoDL RTX 4090
- Framework: PyTorch 2.8.0 + PyTorch Lightning
- Backbone: DINOv2-ViT-S/14 (pretrained)
- Dataset: UCL Head (BPD task), ~100 train / ~10 val (subject-level split)
- Heatmap size: 64×64
- Image size: 512×512

## Baseline

| Method | BPD NME | OFD NME |
|---|---|---|
| HRNet-W18 (Di Vece et al., Table 2) | 0.08 ± 0.18 | 0.05 ± 0.11 |
| HRNet-W18 (reproduced, 200 epochs) | 0.0649 | 0.0441 |

## 11 Runs Summary

| Run | σ | num_blocks | masked_attn | freeze | augmentation | lr | lr_mult | 其他改动 | Best Val NME | 问题/备注 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1.0 | 1 | false | 无 | hflip+color | 1e-4 | 1.0 | — | 98.44% | σ 太小，MSE 坍塌到全零 |
| 2 | 4.0 | 3 | true | 无 | hflip+color | 1e-4 | 1.0 | — | 17.68%* | epoch 110 后过拟合 |
| 3 | 4.0 | 1 | false | 全冻结 | hflip+color | 1e-4 | 1.0 | — | 45.83% | backbone 冻结太激进 |
| 4 | 4.0 | 3 | true | 部分冻结 | scale+rot+hflip | 1e-4 | 10.0 | +sigmoid | 125.16% | sigmoid 导致梯度消失 |
| 5 | 4.0 | 3 | true | 部分冻结 | scale+rot+hflip | 1e-4 | 10.0 | 去掉 sigmoid | 45.51% | 部分冻结仍限制太大 |
| 6 | 4.0 | 3 | true | 部分冻结 | scale+rot+hflip | 1e-4 | 10.0 | +bilinear+文字遮盖 | 45.51% | 同 Run 5，改动未生效 |
| 7 | 4.0 | 1 | false | 无 | scale+rot+hflip | 1e-4 | 2.0 | 文字遮盖+bilinear | 56.39% | 遮盖盖住了 BPD 右端点 |
| 8 | 4.0 | 1 | false | 无 | hflip+color | 1e-4 | 1.0 | config 错误 | 39.91% | num_blocks 应为 3 |
| 9 | 4.0 | 1 | false | 无 | hflip+color | 1e-4 | 1.0 | 服务器未更新代码 | 39.91% | 同 Run 8 |
| 10 | 4.0 | 3 | true | 无 | hflip+color | 1e-4 | 1.0 | 服务器未更新代码 | 42.91% | 代码版本问题 |
| 11 | 4.0 | 3 | true | 无 | hflip+color | 2e-5 | 1.0 | patience=20 | 46.41% | lr 太低 |

*Run 2 的 17.68% 使用了旧版 NME 代码（GT 从 heatmap argmax 解码，存在量化误差），修复量化 bug 后实际 NME 更高。待固定 seed 后复现验证。

## Key Findings

### 有效配置
- **σ=4.0** 是必需的（σ=1.0 导致 MSE 坍塌到全零，Run 1）
- **num_blocks=3 + masked_attn=true** 是目前最优的 decoder 配置
- **不冻结 backbone** 效果最好（所有 freeze 的 run 都卡在 ~45%）

### 失败的尝试
- **sigmoid 输出激活**：直接导致梯度消失，NME 固定在 125%（Run 4）
- **文字遮盖**：右侧 25% 涂黑把 BPD 右端点也盖了（Run 7）
- **部分冻结 backbone**：欠拟合，NME 卡在 ~45%（Run 3, 5, 6）
- **低学习率 2e-5**：收敛太慢（Run 11）

### 发现的 Bugs
1. **NME GT 坐标量化 bug**：eval_step 用 heatmap argmax 解码 GT 坐标，有 0.5 heatmap pixel 量化误差 → 修复为直接用原始 float 坐标
2. **warmup_steps 过长**：[100,200] 但每 epoch 只有 6 步 → 改为 [15,30]
3. **双重 interpolation**：eomt.py 和 landmark_detection.py 都有 F.interpolate → 删除 eomt.py 里的
4. **WandbLogger 在 AutoDL 崩溃** → 改为 CSVLogger（后续切回 W&B）

### 可视化分析（Run 2 checkpoint）
- 模型响应图像上的**黄色文字标注**（BPD/OFD 数值），而非解剖结构
- 预测 heatmap **弥散**：GT 是尖锐 Gaussian peak，pred 是大面积散开的激活
- **棋盘格伪影**：ConvTranspose2d(stride=2) 的经典问题
- Pred range: min=-0.24, max=0.91（模型在产生有意义输出，但定位不精确）

### 核心瓶颈
EoMT 的 `einsum("bqc, bchw -> bqhw", mask_head(q), upscale(x))` 产生的是 query 与 patch 的语义相似度图，天然弥散，不适合 landmark 需要的尖锐 Gaussian peak。

## Next Steps

1. 固定 random seed，复现 Run 2 的配置（σ=4.0, blocks=3, masked_attn=true, no freeze, lr=1e-4）
2. 集成 W&B logging
3. 方案 A：加 Refinement Head（einsum 输出后接 Conv 精化）
4. 方案 D：Hybrid Loss（MSE/AWing + L1/Wing Loss on coordinates）
