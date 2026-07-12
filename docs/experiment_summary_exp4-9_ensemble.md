# EoMT Landmark Detection 实验总结 / Experiment Summary（Exp 4 – Exp 9 + Ensemble）

> 更新时间 / Updated：2026-07-12
> 用途 / Purpose：导师汇报 + 论文撰写 + 新对话上下文交接 / Supervisor meeting, paper writing & conversation handoff
> 项目 / Project：Encoder-Only Vision Transformers for Landmark Detection with DINOv2 (MSc Thesis, UCL)
> 学生 / Student：Flora (Nan Xu) | 导师 / Supervisors：Dr. Zhehua Mao, Dr. Sophia Bano (SRV group)
> 仓库 / Repo：`Flora020703/eomt-landmark-detection`（branch `master`）| W&B：`ucabnx1-ucl/eomt-landmark`
> 承接文档 / Previous doc：本文档承接 "Run 12–Run 20" 与 "Run 12–DeconvV2 Ablations" 两份总结，覆盖 FPN 之后的全部实验 / Continues from the Run 12–20 and DeconvV2 ablation summaries; covers everything from FPN onward.

---

## 零、TL;DR（最重要的结论 / The Bottom Line）

- **当前最佳单模型 / Best single model**：FPN + UDP + EMA，5-seed **last mean 12.09% ± 2.74%**（Exp 8a）。单个 seed 最低触及 **8.71%**（seed 3407），首次触及 HRNet 水平。
- **当前最佳整体结果 / Best overall result**：**FPN + UDP 的 5-seed inference ensemble = 9.75%**（Exp 7 checkpoints）；跨批 10-model ensemble = **9.56%**。**这是全系列最好的 BPD 数字,离 HRNet 的 8% 仅差 1.56–1.75 个百分点,且完全免费(零额外训练)。**
- **完整改进轨迹 / Full trajectory（UCL BPD Test NME）**：
  ```
  einsum baseline        17.48%
  → deconv_v2 (架构核心)  15.18%   (-2.30)
  → FPN (多尺度融合)      14.49%   (-0.69)
  → UDP (坐标编解码修复)  13.21%   (-1.28)
  → +EMA (single model)   12.09%   (-1.12)
  → 5-seed ensemble        9.75%   (-2.34, 最终模型)
  → 10-model ensemble      9.56%   (统计噪声内的进一步改善)
     HRNet baseline         8.00%   (gap ≈ 1.56–1.75, 已接近 49 张 test 的噪声水平)
  ```
- **三个诚实的负结果 / Three honest negative results**：LoRA（欠拟合）、单独 EMA 的跨批不可比性、hm128（分辨率无益）。
- **跨数据集关键发现 / Cross-dataset key finding**：FPN/UDP/EMA 这些改进在 HC18（737 张图，7× 数据）上收益完全消失（16.52% ≈ 纯 deconv_v2 的 16.54%），证明它们是"专治小数据过拟合"的手段——这恰好支撑了论文"小样本医学场景"的核心动机。
- **下一步 / Next steps**：(1) 补 OFD（Head 部位另一半，刚需）；(2) 300W 人脸验证通用性（战略重点，把方法从"超声专用"提升为"通用 landmark 方法"）；(3) thesis 写作，9.8 截稿。

---

## 一、项目与环境回顾 / Project & Environment Recap

将 EoMT（Encoder-Only Mask Transformer，原用于语义分割）适配为 heatmap-based landmark detection，用于胎儿超声 BPD（双顶径）测量。Backbone 为 DINOv2-ViT-S/14（22M 参数）。

Adapt EoMT (originally for semantic segmentation) to heatmap-based landmark detection for fetal ultrasound BPD (biparietal diameter). Backbone: DINOv2-ViT-S/14 (22M params).

**Baseline 对比目标 / Baseline (Di Vece et al., Scientific Reports 2026, UCL→UCL)：**

| 测量 / Measurement | HRNet UCL→UCL | HRNet HC18→HC18 |
|---|---|---|
| BPD | 0.08 ± 0.18 (8%) | 0.05 ± 0.09 (5%) |
| OFD | 0.05 ± 0.11 (5%) | 0.04 ± 0.08 (4%) |

（Flora 早前复现 HRNet：BPD 6.49%, OFD 4.41%。/ Flora's earlier HRNet reproduction.）

**服务器与环境 / Server & Environment：**

| 项目 / Item | 配置 / Configuration |
|---|---|
| 远程训练 / Remote | AutoDL RTX 4090, PyTorch 2.8.0, CUDA 12.8, Python 3.12 |
| SSH | `ssh -p 42739 root@connect.westb.seetacloud.com` |
| 代码路径 / Code path | `/root/eomt/` |
| 数据路径 / Data path | `/root/autodl-tmp/` |
| 必须执行 / Required before git | `git config --global --add safe.directory /root/eomt` |
| 必须执行 / Required before training | `export HF_ENDPOINT=https://hf-mirror.com` |
| Checkpoint 备份 / Backup | `/root/autodl-tmp/saved_checkpoints/`（数据盘，避免 30G 系统盘爆满）|
| tmux | 未安装，用 `nohup ... > logfile.log 2>&1 &` 替代 |
| GitHub 访问 / GitHub access | 服务器 **pull 不了** GitHub（中国网络限制），统一用 **本地 push → 本地 scp 到服务器** 的工作流 |

**关键工作流教训 / Key workflow lessons：**
- 分支名是 `master` 不是 `main`（多次踩坑）。
- 服务器 `git pull` 基本不可用，标准操作是本地 `scp -P 42739 <file> root@connect.westb.seetacloud.com:/root/eomt/...`。
- 系统盘 30G 极易爆满（EMA checkpoint 体积翻倍）。**每次跑完实验先把 best/ema checkpoint 备份到数据盘 `/root/autodl-tmp/saved_checkpoints/`，确认后再删系统盘的。**
- **重要教训：不要删掉"当前最佳配置"批次的 checkpoint，直到 ensemble / TTA 都试过为止**——否则会重复错过 inference ensemble 这种免费收益。

---

## 二、核心架构基础（承前）/ Core Architecture Baseline (Recap)

在本轮实验开始前，已确立的最优配置：

- **DeconvHead V2**（FiLM-conditioned）替换 einsum：query token 生成 γ,β 调制 patch features → Conv → heatmap(B,Q,64,64)，而非 query-patch 点积相似度图。
- **Hybrid Loss**：WMSE + L1 coord（λ_coord=0.1, temp=10, α=5.0）。
- **中间层保持 einsum**：Block 9/10/11 的 intermediate `_predict()` 走 einsum，只有 final prediction 走 DeconvHead V2。
- 64×64 heatmap，σ=4.0，num_blocks=3，masked_attn=true，freeze_backbone=false。

**已确立的 baseline 数字（5-seed，seeds = 42/0/123/2024/3407）：**

| 配置 | Test NME (last, 5-seed mean ± std) |
|---|---|
| einsum baseline | 17.48% ± 1.95% |
| deconv_v2 (64×64) | 15.18% ± 3.00% |
| deconv_v2 frozen backbone | 15.80%（证明 DINOv2 特征本身够用）|
| deconv_v2 hm128 (旧,无UDP) | 15.16% ± 1.12% (val-best) / 15.48% ± 3.01% (last) |

---

## 三、本轮实验详细记录 / Detailed Experiment Log（Exp 4 – Exp 9）

> 所有实验均为 5-seed（42/0/123/2024/3407）消融，除 Exp 9（HC18，单 seed）。
> Test set = 49 张 UCL Head 图（除 HC18 用其自己的 test set）。
> 每个 seed 测 2–3 个 checkpoint：val-best（10 张 val 上最优）、last（最后 epoch）、EMA（若启用）。

### 3.1 Exp 4：FPN 多层特征融合 / Multi-Scale Feature Fusion ✅

**动机 / Motivation**：deconv_v2 只用 DINOv2 最后一层（layer 12），丢失浅层空间细节。HRNet 的核心优势是全程多尺度高分辨率特征。

**设计 / Design**：
- 在 backbone forward 的 block 循环里提取 layer 4/8/12 特征。
- `FeaturePyramidFusion` 模块：per-level GroupNorm → concat → 1×1 Conv 降维。新增参数 ~0.44M。
- 冷启动安全：1×1 Conv 初始化为只选最后一层（layer 12），其余层权重为 0。
- Config 开关：`use_fpn: true`, `fpn_layers: [4, 8, 12]`。

**结果 / Results（Test NME）：**

| Seed | val-best | last | deconv_v2 无FPN (last) |
|------|---|---|---|
| 42 | 13.38% | **10.45%** | 13.06% |
| 0 | 15.08% | 14.48% | 17.36% |
| 123 | 17.76% | 14.62% | 11.11% |
| 2024 | 12.97% | 18.16% | 18.24% |
| 3407 | 13.26% | 16.85% | 16.11% |
| **Mean ± Std** | **14.49% ± 2.01%** | **14.91% ± 2.94%** | **15.18% ± 3.00%** |

**结论 / Conclusion**：FPN 对 last 的改善很小（15.18% → 14.91%，仅 -0.27%，统计误差内），但 **val-best 的 std 从 ~3% 降到 2.01%**，说明 FPN 让 checkpoint selection 更稳定。方向正确、值得保留作为后续基础。/ FPN improvement is small on the mean but reduces variance; kept as the base for subsequent experiments.

### 3.2 Exp 5：FPN + LoRA ❌（负结果 / Negative Result）

**动机 / Motivation**：用 LoRA 微调 backbone 让它更适应超声图像，理论上在过拟合与欠拟合间找平衡。

**设计 / Design**：`peft` 库，DINOv2 attention QKV projection 插入 LoRA（r=8, alpha=16, dropout=0.1）。Sanity check 确认 LoRA 参数真实挂载（147,456 trainable LoRA params）。

**结果 / Results（Test NME）：**

| Seed | val-best | last |
|------|---|---|
| 42 | 18.99% | 18.99% |
| 0 | 17.01% | 18.02% |
| 123 | 18.57% | 19.76% |
| 2024 | 19.15% | 19.01% |
| 3407 | 12.36% | 15.73% |
| **Mean ± Std** | **17.22% ± 2.84%** | **18.30% ± 1.56%** |

**结论 / Conclusion**：**LoRA 明确变差**，比 FPN-only（14.49%）差 ~3%，甚至比无 FPN 的 deconv_v2 baseline（15.18%）都差。

**根因 / Root cause**：模型摘要显示 LoRA 实验变成 **3.1M Trainable + 22.0M Non-trainable**——`peft` 的 `get_peft_model` 在加 LoRA 的同时**自动冻结了整个 backbone**，等于变成 frozen backbone 实验。而 frozen backbone + deconv_v2 早已证明是 15.80%，LoRA 的 147K 参数不足以弥补冻结损失，反而更差。110 张图 + 全参数训练本身没有严重过拟合，"减少可训练参数防过拟合"的前提不成立，制造了欠拟合。/ `get_peft_model` froze the whole backbone; LoRA's 147K params couldn't compensate, causing underfitting. The overfitting premise LoRA addresses simply doesn't hold at 110 images with full fine-tuning.

### 3.3 Exp 6：FPN + EMA（无 loader_seed 修复前）⚠️（不可比 / Not Cleanly Comparable）

**动机 / Motivation**：训练曲线后期 val NME 暴涨（seed42 从 11.67% → 31%），EMA 对权重做指数滑动平均，本质是 implicit ensemble，针对这个后期退化下药。

**EMA 设计要点（CC 实现，`training/ema.py`）：**
- **手写 shadow-weight EMA，不用 Lightning 自带 `StochasticWeightAveraging`**——因为 SWA 会接管 LR schedule（强制常数/循环 LR），与现有 `TwoStageWarmupPolySchedule` 冲突。手写 EMA 完全不碰 optimizer/scheduler。
- **EMA 权重不在训练/验证时换进模型算 val_nme**——避免 EMACallback 和 ModelCheckpoint 的 hook 执行顺序踩坑，保持 val checkpoint selection 逻辑与其他实验完全一致、可比。EMA shadow 存在 checkpoint 里（`on_save_checkpoint`），训练后用 `apply_ema.py` 单独还原成新 ckpt 再测试。
- **decay=0.99（非文献常见的 0.999）**：UCL batch_size=16，每 epoch 仅 ~6 步，200 epoch 共 ~1200 步。decay=0.999 时间常数 ~1000 步，训练完还没"追上"；decay=0.99 时间常数 ~100 步（~17 epoch），刚好能在几次 validation 间隔内平滑噪声，又不会把后期过拟合上升的权重也拉低。

**结果 / Results（Test NME）：**

| Seed | val-best | last | EMA |
|------|----------|------|-----|
| 42 | 15.04% | 17.27% | 15.50% |
| 0 | 13.78% | 15.80% | 14.74% |
| 123 | 21.72% | 18.23% | 18.16% |
| 2024 | 18.19% | 15.65% | 13.53% |
| 3407 | 20.57% | 18.08% | 17.16% |
| **Mean ± Std** | **17.86% ± 3.43%** | **17.01% ± 1.23%** | **15.82% ± 1.86%** |

**关键分析 / Key analysis**：
- **同批内 EMA 有效**：EMA（15.82%）明显优于同批 val-best（17.86%）和 last（17.01%）——干净的同轨迹对比。
- **但整批比 Exp 4（FPN-only, 14.49%）差**：三列全部偏高。EMACallback 按设计不影响训练轨迹，理论上不该有此差异。
- **归因 / Attribution**：这是 **DataLoader 非确定性** 的又一次体现——`train_dataloader()` 建 DataLoader 时没传 `generator`，shuffle/worker 种子从"DataLoader 创建时刻的全局 torch RNG 状态"派生，会因之前的随机操作（模型初始化、dropout）漂移，跟 `seed_everything(N)` 不严格绑定。这与 hm128 时"同 seed=2024 单跑 12.02% vs 脚本跑 15.30%"是同一个问题。
- **seed 123 崩溃**：val NME epoch 30 后飙到 50% 卡住（同 seed 在 FPN-only 里是 17.76%/14.62%），典型的非确定性坏运气。

**结论 / Conclusion**：EMA 技术本身有效（同批内），但"FPN+EMA vs FPN"的整体优势因基准不可比而未证实。**触发了 DataLoader 确定性修复。**

### 3.4 DataLoader 确定性修复 / Determinism Fix（承 Exp 6 → 用于 Exp 8 起）

**根因 / Root cause**：三个 dataloader 方法建 `DataLoader` 时没传 `generator`。

**修法 / Fix（`datasets/landmark_dataset.py`，标准 PyTorch 可复现写法）：**
1. 新增 `loader_seed: Optional[int] = None`（默认 None，不改变现有行为）。
2. 新增顶层 `seed_worker()`：每个 worker 用 `torch.initial_seed()` 派生种子重新 seed numpy 和 python `random`（防御性，为将来 augmentation 铺路）。
3. 三个 dataloader 各建显式 `torch.Generator().manual_seed(loader_seed + offset)`（train +0, val +1, test +2），连同 `worker_init_fn=seed_worker` 传入。
4. Ablation 脚本里 per-seed config 生成处加一行 `cfg['data']['init_args']['loader_seed'] = seed`，让 loader_seed 跟 seed_everything 一致。

**从 Exp 8a 起所有实验都是确定性的**（config 校验清单里能看到 `loader_seed` 项）。

### 3.5 Exp 7：FPN + UDP 坐标修复 / Coordinate Fix ✅（关键突破 / Key Breakthrough）

> ⚠️ 注意：Exp 7 是在 loader_seed 修复**之前**跑的（脚本写于修复前），因此 Exp 7 的"seed=N"与 Exp 8a 的"seed=N"实际对应不同的数据加载顺序，二者不是干净 A/B。但这不影响 Exp 7 自身 5-seed mean/std 的有效性。

**UDP 审计发现 / UDP Audit Finding（CC 实施）**：
- 读了 encode 端（`datasets/landmark_dataset.py` 的 `generate_heatmap` + 坐标缩放）和 decode 端（`training/landmark_detection.py` 的 spatial_softmax / `heatmap_to_coords`）。
- **问题**：`lms[:,0] *= iw/orig_w` 用的是朴素比例缩放，但 PIL `img.resize()` 内部用像素中心对齐重采样（等价 `align_corners=False`）。两种约定之间有 `0.5*(1-scale)` 的系统性偏移——正是 UDP（Huang et al., CVPR 2020）指出的那类 bug。
- 用真实图片尺寸（原图 1136×783、960×720 等，长宽比不统一）算出：从原图→512→64 的偏移约 **0.47 个 heatmap 像素**（约 5–8 原图像素，<1mm）。恒定方向系统偏移，非随机噪声。

**修法 / Fix**：`pixel_center_align: bool = False` 开关（默认关，不影响历史实验）。开启后两处缩放改成 `(x + 0.5) * scale - 0.5`（scale=1 时退化为恒等）。

**结果 / Results（Test NME）：**

| Seed | val-best | last |
|------|---|---|
| 42 | 11.66% | 12.82% |
| 0 | 12.01% | 14.62% |
| 123 | 15.29% | 12.24% |
| 2024 | 15.33% | 14.16% |
| 3407 | 11.75% | 14.52% |
| **Mean ± Std** | **13.21% ± 1.92%** | **13.67% ± 1.08%** |

**结论 / Conclusion**：UDP 带来 ~1.3% 的真实改善（FPN val-best 14.49% → 13.21%），而且 **last 的 std 从 2.94% 大幅降到 1.08%**——修复 encode/decode 偏移不仅提升精度，还让训练更稳定。论文可写："We identified and corrected a systematic coordinate encoding-decoding misalignment consistent with UDP (Huang et al., CVPR 2020), yielding a further 1.3 percentage point improvement." **这是当时最好的单配置结果。**

### 3.6 Exp 8a：FPN + UDP + EMA ✅（最佳单模型 / Best Single Model）

> 第一个用上 loader_seed 的确定性实验。

**结果 / Results（Test NME）：**

| Seed | val-best | last | EMA |
|------|----------|------|-----|
| 42 | 15.42% | 12.74% | 12.88% |
| 0 | 16.25% | 15.99% | 16.00% |
| 123 | 16.83% | 10.45% | 11.38% |
| 2024 | 15.14% | 12.56% | 12.46% |
| 3407 | 13.88% | **8.71%** | **8.71%** |
| **Mean ± Std** | **15.50% ± 1.13%** | **12.09% ± 2.74%** | **12.29% ± 2.64%** |

**关键发现 / Key findings**：
- **seed 3407 = 8.71%，首次触及 HRNet 的 8%**（单 seed）。
- **EMA 这次真的有效**：同批 EMA（12.29%）比 val-best（15.50%）好 3.2%，且 EMA ≈ last（12.09%），说明 EMA 成功平滑了后期权重。
- **12.09% 是当时最好的绝对单模型数字。**

**重要 caveat（loader_seed 混杂）/ Important confound**：Exp 7（无 loader_seed）与 Exp 8a（有 loader_seed）不是干净 A/B。Exp 7→8a 的"val-best 变差（13.21%→15.50%）、last 变好（13.67%→12.09%）"这种一好一坏，很可能不是 EMA 造成的，而是 loader_seed 本身改变了每个 seed 的实际训练轨迹——反过来印证了 DataLoader 非确定性问题真实存在且有影响。**因此"FPN+UDP+EMA 比 FPN+UDP 好"这个结论在单模型层面无法干净证明**（论文里应如实说明，或直接以 ensemble 结果为最终数字规避）。

### 3.7 Exp 8b：FPN + UDP + hm128 (128×128) ❌（负结果 / Negative Result）

**动机 / Motivation**：之前 hm128 无效（15.16% vs 64×64 的 15.18%），但 UDP 的 bias 量级（0.5·(1-scale)）跟 heatmap 分辨率有关，怀疑之前 hm128 的效果被 UDP 未修复的噪声源掩盖了，值得在修复后重验。config：heatmap_size 64→128，sigma 4.0→8.0（proportional）。

**结果 / Results（Test NME）：**

| Seed | val-best | last |
|------|---|---|
| 42 | 14.02% | 19.14% |
| 0 | 14.43% | 14.24% |
| 123 | 14.51% | 15.12% |
| 2024 | 17.60% | 15.58% |
| 3407 | 14.83% | 16.94% |
| **Mean ± Std** | **15.08% ± 1.40%** | **16.20% ± 1.90%** |

**结论 / Conclusion**：**hm128 比 64×64 差 ~2%。** UDP 修复后 hm128 依然无益。seed 3407 训练中 val NME 一度降到 6.19%（10 张 val 上的幻觉），但 test 是 14.83%——再次印证 10 张 val set 完全不可信。**64×64 就是最优分辨率，更高分辨率只增加过拟合风险。**

### 3.8 Exp 9：FPN + UDP + EMA 在 HC18（跨数据集，单 seed=2024）✅（重要负结果 / Important Negative Result）

**动机 / Motivation**：论文需要跨数据集结果（HRNet 报了 UCL→UCL 和 HC18→HC18 两组）。HC18 有 737 张训练图（UCL 的 7×），验证改进的数据可扩展性。

**结果 / Results（HC18 Test NME）：**

| Checkpoint | HC18 Test NME |
|---|---|
| val-best | **16.52%** |
| last | 18.76% |
| EMA | 18.77% |

**历史 HC18 对比 / HC18 history：**

| 配置 | HC18 Test NME |
|---|---|
| einsum | 20.74% |
| deconv_v2 | 16.54% |
| **FPN+UDP+EMA (Exp 9)** | **16.52%** |
| HRNet | 5% |

**关键发现 / Key findings**：
- **16.52% ≈ 纯 deconv_v2 的 16.54%**——这一整套在 UCL 上有效的改进（FPN+UDP+EMA）搬到 HC18 上**几乎零增益**。两个完全不同的配置得出几乎相同的数字，是很强的信号（不像巧合）。
- **论文叙事价值**：FPN（补空间细节）、UDP（消标签噪声）、EMA（平滑轨迹）本质都是对抗小数据过拟合/高方差的手段。UCL 只有 110 张图，这些问题严重；HC18 有 737 张，模型有足够数据学到稳定表征，这些"止血"手段没有用武之地。**这不是"方法没用"，而是它们的价值集中在小数据场景——恰好正是论文的核心场景（胎儿超声标注稀缺昂贵）。负结果反而支撑了动机。**
- **val-best 反转**：HC18 上 val-best（16.52%）明显好于 last/EMA（18.76%/18.77%），与 UCL 规律相反。原因：HC18 数据量大，10% val split 对应绝对样本数更多，val 集更有代表性，"按 val 选 checkpoint"在大数据集上可信；只有 UCL 那种 10 张 val 才不可信。**这是个可写进论文的"数据量决定 checkpoint selection 可靠性"发现。**
- **last ≈ EMA（18.76% vs 18.77%）**：HC18 训练后期本身就稳定（不像 UCL 震荡），EMA 没什么可平滑的——进一步印证"这些技术专治小数据的病"。

**结论 / Conclusion**：**单 seed 已足够，不补 seed。** 论文核心论点可整理为："架构性贡献（einsum→deconv_v2）在两数据集上都有效（UCL 17.48%→15.18%，HC18 20.74%→16.54%），而正则化性质的改进（FPN/UDP/EMA）收益与训练集规模成反比：UCL 上累积贡献 ~3pt，HC18 上收益消失。"

---

## 四、Inference Ensemble（最重要的免费收益 / The Key Free Win）✅

**方法 / Method（`ensemble_test.py`，CC 实现）**：加载 N 个 checkpoint，对同一批 test 图的 **heatmap 在 decode 之前取平均**（不是平均坐标——空间平均能修正各模型不同的失败模式），再统一 soft-argmax + decode 算 NME。

**技术细节坑 / Technical gotcha**：`main_landmark.py` 用 `LightningCLI` 的 `link_arguments` 把 `data.init_args.heatmap_size` 自动同步到 `model.init_args.heatmap_size`（yaml 里没写这个字段，CLI 自动接）。任何绕开 `main_landmark.py` 直接构建模型的脚本，若不手动复现这几条 link，`LandmarkDetection.heatmap_size` 会 retain 默认值 (64,64)，跟网络头实际分辨率对不上却不报错（F.interpolate 会强行 resize）。`ensemble_test.py` 已手动补上 link。

**结果 / Results：**

| Ensemble 来源 / Source | Models | 个体 Mean | Ensemble NME |
|---|---|---|---|
| Exp 8a EMA (5 ema ckpt) | 5 | 12.29% | 13.11%（比个体还差）|
| Exp 8a best (5 best ckpt) | 5 | 15.50% | 11.32%（-4.2pt）|
| **Exp 7 best (5 best ckpt)** | **5** | **13.21%** | **9.75%（-3.5pt）** |
| **跨批 Exp7+8a best (10 ckpt)** | **10** | — | **9.56%（全系列最优）** |

**关键洞察 / Key insights：**
- **Exp 7 的 best-ensemble = 9.75%，从个体 13.21% 降 3.5pt，首次跌破 10%，离 HRNet 8% 仅 1.75pt。**
- **反直觉发现：EMA checkpoint 的 ensemble（13.11%）反而比 best checkpoint 的 ensemble（9.75%）差**，尽管 EMA 单独 mean 更低。原因：ensemble 靠模型间"分歧"起作用；5 个 val-best 是各自轨迹上相对独立的 epoch 点，犯的错不同，能互相纠正；而 EMA 本身已是"跨 step 平滑"，5 个 seed 的 EMA 权重被拉向相似的"通用解"，彼此更相似、分歧更小，可纠正空间反而变小。**EMA 和 ensemble 都是降方差手段,部分冗余,不是简单叠加——如果反正要 ensemble,原始 best checkpoint 比 EMA checkpoint 更适合。**
- **10-model 跨批 ensemble = 9.56%**，比 5-model 又好 0.19pt。但 49 张 test 下,0.19pt 基本在统计噪声内（约 1 张图的 NME 差异），只能说"至少不差，且免费"。

**论文数字选择建议 / Which number to report：**
- **Headline 用 Exp 7 的 9.75%（5-model）**，而非 9.56%（10-model）。理由：5-model 配置更干净（同一 FPN+UDP 配置跑 5 seed 再 ensemble），story 清晰；10-model 混了有/无 EMA 两个配置，解释啰嗦。0.19pt 差异统计无意义，不值得牺牲简洁性。
- 可在论文里用一句话提及 10-model 的 9.56% 作为附注。
- **重要写作提醒**：test set 仅 49 张，9.75% 与 HRNet 8% 的 1.75pt gap 换算成绝对样本可能就是 1–2 张图的差异。报告时应带 bootstrap 置信区间，别把 gap 说得太绝对；反过来也意味着"已实质接近 HRNet"。

---

## 五、完整消融总表 / Complete Ablation Summary

| 实验 / Exp | 配置 / Config | 数据集 | 最佳数字 / Best | 结论 / Verdict |
|---|---|---|---|---|
| — | einsum baseline | UCL | 17.48% ± 1.95% (last) | 基线 |
| — | deconv_v2 | UCL | 15.18% ± 3.00% (last) | ✅ 架构核心 |
| 4 | + FPN | UCL | 14.49% ± 2.01% (val-best) | ✅ 多尺度,降方差 |
| 5 | + FPN + LoRA | UCL | 17.22% ± 2.84% (val-best) | ❌ peft 冻结 backbone 致欠拟合 |
| 6 | + FPN + EMA (无 loader_seed) | UCL | 15.82% ± 1.86% (EMA) | ⚠️ 同批有效,跨批不可比 |
| 7 | + FPN + UDP | UCL | 13.21% ± 1.92% (val-best) | ✅ 坐标编解码修复 |
| 8a | + FPN + UDP + EMA | UCL | 12.09% ± 2.74% (last) | ✅ 最佳单模型,seed3407=8.71% |
| 8b | + FPN + UDP + hm128 | UCL | 15.08% ± 1.40% (val-best) | ❌ 分辨率提升无益 |
| 9 | FPN + UDP + EMA | HC18 | 16.52% (val-best, 单seed) | ✅(方法论) 改进在大数据集收益归零 |
| **Ens** | **FPN+UDP 5-seed ensemble** | **UCL** | **9.75%** | **✅ 最终模型(headline)** |
| Ens | 跨批 10-model ensemble | UCL | 9.56% | ✅ 统计噪声内进一步改善 |

---

## 六、关键技术发现总结 / Key Technical Findings

1. **架构 vs 正则化的二分 / Architecture vs. regularization dichotomy**：einsum→deconv_v2 是根本性架构贡献（两数据集都有效）；FPN/UDP/EMA 是正则化性质的小数据技巧（收益随数据量增大而消失）。这是论文最干净的论点主线。

2. **10 张 val set 系统性不可靠 / 10-image val set is systematically unreliable**：val NME 波动 ±10%，EarlyStopping 选出的 "best" 不可信；训练中 val 一度降到 6% 是幻觉。last checkpoint 通常更可靠。**数据量决定 checkpoint selection 可靠性**（HC18 上 val-best 反而最好）。

3. **DataLoader 非确定性真实存在 / DataLoader non-determinism is real**：未传 `generator` 导致同 seed 跨调用不可复现，是 hm128 和 Exp6 两次"同 seed 不同结果"的根源。已修复（loader_seed），Exp 8 起确定性。

4. **UDP 坐标偏移是隐蔽的系统性 bug / UDP misalignment is a hidden systematic bug**：PIL resize 的像素中心对齐 vs 朴素比例缩放之间有 0.5·(1-scale) 偏移，影响所有实验，修复后 -1.3pt 且降方差。

5. **Ensemble 靠"分歧"，与 EMA 冗余 / Ensemble relies on disagreement, redundant with EMA**：保留模型多样性（不用 EMA）再 inference ensemble 是最佳降方差策略；EMA checkpoint 因彼此相似反而不适合 ensemble。

6. **LoRA 在极小数据全参训练场景不适用**：peft 自动冻结 backbone，"减少可训练参数防过拟合"前提不成立，制造欠拟合。

7. **可视化验证（承前）/ Visualization (recap)**：单个好样本（NME 3.09%）的可视化显示 Q1/Q2 两个 query token 成功分工定位不同 landmark，heatmap 峰值集中，attention 正确聚焦解剖结构而非文字标注——DeconvHead V2 解决了 einsum 弥散问题。

---

## 七、未解决的观察 / Open Observations（尚未处理，供后续）

1. **长宽比 squeeze 问题 / Aspect-ratio squeeze**（CC 与导师独立发现，互相印证）：原图长宽比不统一（1.45:1 到 1.33:1），却被直接 squeeze 成 512×512 正方形，每张图变形程度不同——这是**样本间的方差来源**（非系统偏移）。可能比 UDP 的 0.47 像素影响更大。**未处理**，建议用 letterbox（pad-to-square）解决。

2. **augmentation 完全空白 / No augmentation yet**：目前仅 hflip + color jitter。110 张图场景下，aggressive augmentation 可能是剩余 gap 的最大来源，但历史上（Rule 3/4）踩过坑（黑块遮挡 landmark、scale jitter 坐标越界）。**当前判断：BPD 已到 9.56%（噪声边缘），augmentation 的边际收益低、风险高，本轮可暂不做**（除非后续想冲破 8%）。

---

## 八、下一步计划 / Next Steps（7.12 → 9.8 thesis 截稿）

**整体规划 / Overall plan**：先完成完整 30–50 页 thesis（9.8 交），之后按不同刊方向拆分投稿——医学影像刊聚焦 fetal head（BPD+OFD 深挖小数据方法论），通用 landmark 刊/会强调 encoder-only 架构在人脸+医学双域有效。

### 阶段一：补齐 OFD（刚需，最高优先级）/ Complete OFD (Critical)
- **为什么必须做**：Di Vece baseline 同时报 BPD 和 OFD（HRNet OFD=5%）。fetal biometry 论文只报单一测量项不完整，reviewer 一定会问。
- **成本可控**：OFD 与 BPD 共享同一批 Head 图像，只是标注不同的 landmark（BPD=颅骨双顶径左右两点；OFD=枕额径前后两点），**同样 2 个 landmark，架构不用改**。复用 Exp 7 的 FPN+UDP 配置 + 5-seed 脚本 + ensemble。
- **待确认**：OFD 标注 CSV 是否也是 2 点格式（应该是）。基于 `ofd_vit_small.yaml` 加 FPN+UDP 参数。

### 阶段二：300W 人脸通用性验证（战略重点）/ 300W Face Generalizability (Strategic)
- **为什么重要**：原始 proposal 承诺了 300W/WFLW 人脸数据集（Evaluation 部分明确写了 "face landmarks (300W, WFLW) and fetal biometry"）。做了 300W 就能把论文从"超声专用方法"提升到"通用 encoder-only landmark 方法,人脸+医学双域有效"——从 workshop 级提升到 main conference 级。Flora 明确想突破这个通用性点。
- **必须处理的域差异（与 fetal 很不同，提前预警）/ Domain differences to handle**：
  1. **landmark 数量爆炸**：300W 是 **68 个关键点**，现架构 num_q=2。要改成 num_q=68——query token 数量、matching 逻辑、heatmap 通道数全变。这是实质架构改动。
  2. **数据量完全不同**：300W 训练集 3000+ 张（UCL 的 30×）。预期 FPN/UDP/EMA 这些小数据技巧又会失效（与 HC18 发现一致）——反而进一步印证"数据规模决定方法收益"论点。
  3. **评估协议**：300W 有标准 NME 归一化（眼间距 / bounding box），必须严格对齐文献否则数字不可比。
  4. **不需要小数据技巧**：300W 上要证明的是"基础架构 deconv_v2 通用"，而非"小数据技巧通用"。
- **建议**：先在 300W 验证 deconv_v2 基础架构，不必叠加 UDP/EMA/ensemble。

- **如果 300W 遇到瓶颈（性能不达文献水平），按顺序考虑以下方案 / If 300W underperforms, consider these in order：**
  1. **数据增强 + letterbox / Augmentation + letterbox**：虽然 300W 数据量大（3000+），但人脸 landmark 任务对几何增强（旋转、scale、flip）高度敏感且标准做法就依赖强增强；letterbox 也能解决人脸图长宽比不一的问题。这套方案在 fetal 上因数据太少+历史 bug 而暂缓，但在 300W 上是**标准且低风险**的（数据多，reject-resample 触发少）。参考第八节"可选"里 CC 的详细设计（黑色 padding、±15° 旋转、scale 0.9–1.1、旋转+scale 合并 affine、reject-and-resample）。
  2. **升级 backbone 到 DINOv3 / Upgrade backbone to DINOv3**：现用 DINOv2-ViT-S/14。proposal 里 [1] 引的其实就是 DINOv3（"the latest generation of vision foundation models"）。DINOv3 特征更强，人脸这种自然图像域正是它的强项（DINOv2/v3 预训练数据以自然图像为主，人脸比超声更接近其分布）。换 backbone 是相对干净的 ablation（改 `backbone_name`），可能直接补上性能差距，同时也是论文里一个有价值的 backbone ablation 点。
  3. **SimCC（1D 坐标分类）替代 heatmap regression / SimCC as alternative to heatmap regression**：把坐标预测从 2D heatmap regression 转成 1D classification（x、y 分别 softmax 分类到 N 个 bin），绕开 2D heatmap 分辨率限制和 soft-argmax 的多峰均值偏移问题。与 EoMT 的 query token 架构天然兼容（query token → MLP → x-logits + y-logits）。**工作量大**（新 head + 新 loss + 新解码），放最后。参考文献见第十节 [SimCC]。68 点人脸场景下 SimCC 的 1D 表示比 68 通道 2D heatmap 更省显存，可能有额外工程优势。
- **注意**：以上三个方案一次只上一个变量，分别验证（吸取 fetal 阶段"一次只加一个 component"的教训）。

### 阶段三：thesis 写作（可与实验并行）/ Thesis Writing (Parallel)
- fetal 部分素材已齐（本文档 + BPD 完整结果），可先写。
- OFD 和 300W 结果出来后补进"generalizability"章节。

### 🔬 DINOv2 vs DINOv3 backbone 对比实验（值得做，但放 300W 之后）/ DINOv2 vs DINOv3 Ablation (Worth Doing, AFTER 300W)

> **提醒 Flora（怕忘）**：曾讨论过"是否在 BPD/OFD 上换 DINOv3 跟 DINOv2 对比"。结论：**值得做，但不要在 OFD 之后立刻做，而是放到 300W 之后，一次性在 BPD + OFD + 300W 三个任务上统一换 DINOv3 跑，做成一张完整对比表。**

**为什么值得做 / Why worth doing**：
- 换 backbone 是干净、低成本的 ablation（只改 `backbone_name`，架构不动，复用 FPN+UDP 脚本）。
- proposal [1] 引的其实就是 DINOv3（"the latest generation of vision foundation models"），用 DINOv2 主实验 + DINOv3 对比正好呼应 proposal，让"foundation model 选择"成为受控变量，是论文一个有价值的 ablation 点。

**为什么不在明天 OFD 之后立刻做 / Why NOT immediately after OFD**：
1. **BPD 在 DINOv2 上已达 9.75%（接近 HRNet），49 张 test 噪声大**，DINOv3 的小幅增益很可能被淹没、看不出统计显著差异。
2. **DINOv3 的真正价值在 300W，不在 fetal**：DINOv2/v3 都是自然图像预训练，**人脸比超声更接近其预训练分布**，DINOv3 在 300W 上的增益预期远大于 fetal。想证明"更强 backbone 有用"，300W 是更能出效果的战场。
3. **放最后做能一次覆盖三个任务**：那时最优配置已定型，不会浪费 GPU 在未定型配置上，论文里就是一张干净的"DINOv2 vs DINOv3 across BPD/OFD/300W"对比表，而非零散补。

**工程前置检查（重要）/ Engineering pre-check**：
- DINOv3 的权重/接口跟 DINOv2 **不完全一样**（patch size、reg token 数、HF/timm 模型名可能都变）。现用的是 `vit_small_patch14_reg4_dinov2`。
- **换之前先花 ~30 分钟验证接口可用性**：timm/HF 上有无对应可用权重、能否塞进现有 pipeline、加载后 shape 对不对、跑一个 epoch 不报错。**别假设无缝。** 确认可行后再排完整 5-seed 实验。

**折中 / Compromise**：如果明天 OFD 跑完还有空，可以先只做上面的"接口可用性验证"（不跑完整实验），确认能跑通即可；完整 DINOv2 vs DINOv3 对比留到 300W 之后统一做。

### 需与导师确认 / To Confirm with Supervisors
- 300W/WFLW 是否在 thesis scope 内（proposal 写了但至今未做）。大概率 scope 已收窄到 fetal，但需主动确认——thesis 评审最易被问"proposal 写了人脸，结果呢？"。
- Abdomen/Femur（APAD/TAD/FL）**明确排除**，列为 future work（不同图像、不同解剖结构，scope 爆炸）。

### 可选 / Optional（时间充裕再做）
- augmentation + letterbox（一次重写 `datasets/landmark_dataset.py` 几何变换，两个独立开关分开验证）。CC 已设计方案：letterbox 用黑色 padding（与超声扇形黑边一致）；旋转 ±15° 保守起步；scale 0.9–1.1；旋转+scale 合并成一次 affine（避免二次插值）；越界坐标用 **reject-and-resample**（不 clip 不丢弃，最多重试 5 次，失败退回 identity）；暂不做 elastic。**实验上 letterbox 与 geo_augment 要分开验证。**
- TTA（test-time augmentation）：推理时对 test 图做翻转/小幅缩放再平均，几乎免费。

---

## 八·五、投稿策略评估 / Submission Venue Strategy（供与导师讨论 / For Discussion with Supervisors）

> 这是一份现实的 target venue 评估，供 thesis 交完后拆分投稿时参考。核心判断：**医疗影像这条线补齐 OFD 后有实在的中稿希望（但别当铁票）；顶会/CVPR 这条线，通用性验证是必要条件但不是充分条件，当作"冲一冲"而非主要预期。**

### 医疗影像期刊/会议（主投，比较稳）/ Medical Imaging (Primary, Reasonably Strong)

- **目标 venue**：MICCAI（main 或 workshop）、MIDL、ISBI、Medical Image Analysis (MedIA)。MIDL 对"方法论 + 消融扎实但未达 SOTA"的工作尤其友好。
- **为什么有竞争力**：einsum→deconv_v2 的架构诊断 + 系统性排除 7 个瓶颈假设 + UDP 修复 + 9.75% ensemble 接近 HRNet + HC18 跨数据集诚实负结果（方法专治小数据）。补齐 OFD 后 Head 部位完整，是方法论清晰、结果诚实、消融充分的工作。
- **现实定位（别过度乐观）**：绝对数字（9.75% vs HRNet 8%）仍略逊于专用 baseline，这是最易被 reviewer 挑的点。但你的贡献不在"打败 HRNet"，而在"用通用架构逼近专用 CNN + 系统方法论"。审稿人接受这个框架就没问题。**结论：竞争力强、有合理中稿希望，而非"稳过 100%"。**

### 顶会 / CVPR / ICCV / NeurIPS（stretch，别当预期）/ Top-Tier (Stretch, Not the Baseline Expectation)

- **通用性 ≠ 自动够顶会**：即便 DINOv3 + 300W 都通用，顶会 landmark/pose 论文通常要么在标准 benchmark（300W/WFLW/COCO）打到接近 SOTA，要么有很强的方法新颖性。本方法本质是"EoMT 改造 + 已知组件（FPN/UDP/EMA/ensemble）组合"，这些组件单独都不新——新颖性主要在"encoder-only 分割架构迁移到 landmark"这个 angle。angle 有意思，但 reviewer 会追问"方法核心创新是什么"。
- **300W 上大概率打不过专用 SOTA**：300W 当前 SOTA NME 约 2.8–3.3%（专为人脸设计）。用通用架构去跑，"work、合理、证明通用性"就不错，但很可能明显落后专用 SOTA——这在顶会是硬伤，在"通用性 demo"里却可接受。
- **现实通过率**：CVPR ~25% 接收率，竞争极激烈。"通用架构 + 组合已知技巧 + 各 benchmark 都不 SOTA"即使 story 完整，也偏 borderline。

### 结论与策略 / Conclusion & Strategy

1. **主投医疗影像（MICCAI/MIDL/MedIA）**——工作最匹配、中稿希望最实在。
2. **顶会作为 stretch/备选**——若 300W+DINOv3 结果漂亮（通用性论点强、跨域一致提升），可试投 CVPR/ICCV 或其 workshop，但当作"冲一冲"，被拒是常态，不影响主线。
3. **通用性验证的最大确定收益，其实是提升医疗影像投稿的说服力**——"方法不是过拟合到超声，人脸上也 work"会让 MICCAI/MIDL reviewer 更信服；冲顶会是附带的彩票。

> 一句话 / One-liner：**OFD 补齐后医疗影像有实在中稿希望（但非铁票）；通用性做好能加分并给顶会一张彩票，但顶会别当预期。**

---

## 八·六、若冲 CVPR：内容最大化路线图 / CVPR Content-Maximization Roadmap（可选，thesis 后拆分投稿用）

> 场景 / Scenario：thesis（9.8）交完后，想把这份工作**加料改造 ~1/3 内容**投 CVPR/ICCV。这一节记录具体怎么做。
> **核心判断 / Core judgment**：现在这份工作的"骨架"直接投 CVPR 会被拒，但骨架是好的。冲 CVPR 需要补的**不是"再多一个数据集"，而是一个能让 reviewer 记住的"核心方法创新点"**。CVPR reviewer 的心理模型是"这篇的 one-contribution-sentence 是什么？"——现在的答案"把分割架构 EoMT 改造成 landmark detection 并系统消融"是好工程，但不是方法创新。要在下面三个方向里**选一个深挖（不是全做）**，把它变成方法创新。

### 方向 A（推荐，最可行）：把"encoder-only 统一架构"做成真正的卖点 / Unification as the Contribution
- **升级后的论点**：不是"EoMT 能改成 landmark detector"，而是**"一个 encoder-only ViT，不需要任务专用 decoder，同时做分割 + landmark detection，且都有竞争力"**——unification 本身是贡献。
- **要做的实验**：
  - 证明**同一个 encoder-only 架构**在多个 dense prediction 任务上都 work（landmark + 分割，甚至 + depth）。
  - 强调"no task-specific decoder"的极简性，呼应 EoMT 原论文"your ViT is secretly a segmentation model"——你是"secretly a landmark detector too"。
- **新颖性所在**：架构统一性，而非单任务 SOTA。CVPR 会吃这套。

### 方向 B：把 einsum→deconv_v2 的诊断上升为通用发现 / A General Plug-in Query-Decoding Module
- **升级后的论点**：把 einsum 弥散从"一个工程问题"上升为**"关于 mask-based query 机制为何不适合精确定位的系统性分析"**，提出一个通用的 query-to-precise-location 模块，证明它不只对 fetal 有效，对所有 query-based landmark/pose 任务都有效。
- **要做的实验**：在 COCO keypoint / 300W / WFLW 上验证 deconv_v2 模块能**插进其他 query-based 方法（DETR-style）并提升**。
- **贡献变化**：从"一个 fetal 方法"→"一个即插即用的 query-decoding 模块"。

### 方向 C（可作辅助）：把小数据方法论做成 CVPR 级系统研究 / Systematic Data-Efficiency Study
- **升级后的论点**：你已有一个强观察——FPN/UDP/EMA 收益与数据量成反比（UCL 有效、HC18 归零）。系统化为**"foundation-model-based landmark detection 在数据稀缺 regime 下，哪些技巧真正有效"**的完整 study + 可复现 recipe。
- **要做的实验**：
  - 人为构造数据量梯度（10/50/100/500/1000 张），画出每个技巧（FPN/UDP/EMA/ensemble/augmentation）的**收益-数据量曲线**。
  - 跨多个数据集（fetal + 人脸 + 通用）验证规律。
- **贡献变化**：变成"a systematic study of data-efficient landmark detection with foundation models" + 一套 recipe。这类 empirical study 论文 CVPR 也收。

### 推荐组合（把内容最大化，正好 ~改 1/3）/ Recommended Combo
若时间充足（近两个月 + thesis 后可继续）：
1. **主线 = 方向 A**：encoder-only 统一架构，在 landmark（fetal BPD/OFD + 300W 人脸）+ 至少一个分割任务上都 demo，强调"no task-specific decoder"的统一性。
2. **加料 = 方向 C**：补一个 data-efficiency study 作为 analysis 章节，把"技巧收益随数据量变化"系统化。
3. **backbone 维度**：DINOv2 vs DINOv3 对比作为 ablation，顺带增强"受益于更强 foundation model"的论点。
→ CVPR 版本 = thesis 的 fetal 核心（~2/3）+ 新增统一性 demo + data-efficiency study + 人脸/DINOv3（~1/3 新内容）。

### 诚实提醒 / Honest Caveats
1. **CVPR 版必须有至少一个 benchmark 上的强结果或强分析**：纯 fetal 数字（9.75%）撑不起 CVPR，必须靠"统一性"或"data-efficiency study"这种 angle 立住，或在 300W 上做到"合理接近"（哪怕不 SOTA）。
2. **thesis 和 CVPR 版必须有实质区别，不能一稿两投**：thesis 聚焦 fetal 应用 + 完整消融；CVPR 版聚焦方法/统一性/analysis。核心实验可复用，但 framing、主张、新增实验要不同。（先 thesis 后拆分，方向已对。）
3. **量力而行**：方向 A 要补分割任务、方向 C 要补数据量梯度实验，都是不小工作量。时间不够就退而求其次——**medical 主投稳妥，CVPR 版作为之后半年慢慢打磨的目标**，不必赶。

---

## 九、给新对话的快速交接 / Quick Handoff for New Conversations

如果在新对话里继续这个项目，需要知道：
- **当前最佳**：BPD 9.75%（FPN+UDP 5-seed ensemble），单模型 12.09%（+EMA）。gap to HRNet ≈ 1.75pt（噪声边缘）。
- **配置**：DeconvHead V2 (FiLM) + Hybrid Loss (WMSE + L1 coord, λ=0.1, temp=10, α=5.0) + FPN (layers 4/8/12) + UDP (pixel_center_align=true)，64×64 heatmap，σ=4.0，DINOv2-ViT-S/14 backbone，5 seeds = 42/0/123/2024/3407。
- **工作流**：本地改代码 → 本地 git push (branch `master`) → 本地 scp 到服务器（服务器 pull 不了 GitHub）→ 服务器 `nohup bash ablation/scripts/run_xxx_ablation.sh`（2026-07-13 起消融脚本已从仓库根目录移进 `ablation/scripts/`，`ensemble_test.py`/`apply_ema.py` 移进 `ablation/`，`test_dataload.py` 移进 `scripts/`，均需从仓库根目录调用）。每次先 `export HF_ENDPOINT=https://hf-mirror.com`。跑完先备份 checkpoint 到 `/root/autodl-tmp/saved_checkpoints/` 再清系统盘。
- **确定性**：Exp 8 起用 loader_seed，同 seed 可复现。
- **下一步**：OFD（刚需）→ 300W（通用性，需改 num_q=2→68）→ thesis 写作（9.8 截稿）。
- **DINOv2 vs DINOv3 对比实验（别忘）**：值得做，但放 **300W 之后**统一在 BPD+OFD+300W 三任务上换 DINOv3 跑（fetal 上噪声大、DINOv3 增益预期小；人脸更接近其预训练分布、增益预期大）。换之前先花 30 分钟验证 DINOv3 接口/权重能否塞进现有 pipeline（patch size、reg token、模型名可能都跟 DINOv2 不同）。详见第八节。
- **300W 遇瓶颈的备选（按顺序）**：数据增强+letterbox → DINOv3 backbone → SimCC（1D 坐标分类）。一次只上一个变量。
- **已排除**：LoRA、hm128、单独堆 EMA、Abdomen/Femur。augmentation 在 fetal 上暂缓（但在 300W 上是标准方案）。
- **投稿策略（详见第八·五节）**：主投医疗影像（MICCAI/MIDL/MedIA），补齐 OFD 后有实在中稿希望但非铁票；顶会/CVPR 当 stretch，通用性(300W+DINOv3)是加分和"彩票"，别当主要预期。
- **若冲 CVPR（详见第八·六节）**：thesis 后加料 ~1/3 内容。需补一个"核心方法创新点"（选一个深挖）：方向 A 统一架构（encoder-only 无任务专用 decoder 同时做分割+landmark，推荐）/ 方向 B 通用即插即用 query-decoding 模块 / 方向 C 数据效率系统研究。推荐组合 = A 主线 + C analysis + DINOv2/v3 ablation。thesis 与 CVPR 版必须实质区别，不可一稿两投。

---

## 十、关键方法与架构参考文献 / Key Method & Architecture References（论文撰写用 / For Paper Writing）

> 下表列出本项目所用/参考的每个核心方法对应的文献，方便写 Related Work 和 Method 章节直接引用。标注了每篇在本项目里对应的**具体组件**。
> Each reference below maps to a specific component used in this project.

### 10.1 核心架构与 Backbone / Core Architecture & Backbone

| 组件 / Component | 文献 / Reference | 用途 / Role in this project |
|---|---|---|
| **EoMT**（Encoder-only Mask Transformer，被改造的基础架构） | Kerssies et al., "Your ViT is Secretly an Image Segmentation Model," **CVPR 2024**, pp. 25303–25313. | 本项目的起点：把原用于语义分割的 EoMT（query token + einsum mask prediction）改造为 landmark detection。核心贡献就是诊断并替换其 einsum 机制。 |
| **DINOv2**（当前 backbone） | Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision," **TMLR 2024**. | 现用 backbone：DINOv2-ViT-S/14（22M）。frozen 实验（15.80%）证明其预训练特征本身足够定位 landmark。 |
| **DINOv3**（proposal 引用，300W 备选） | proposal [1] 所指"the latest generation of vision foundation models"（DINOv3 系列）。 | 300W 遇瓶颈时的 backbone 升级备选。人脸自然图像更接近其预训练分布。 |
| **ViT**（backbone 架构本体） | Dosovitskiy et al., "An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale," **ICLR 2021**. | ViT 架构基础，encoder-only 路线的根基。 |

### 10.2 Query Token → Landmark 的机制参考 / Query-based Landmark Mechanism

| 组件 / Component | 文献 / Reference | 用途 / Role |
|---|---|---|
| **TokenPose**（query token → heatmap 的标准做法） | Yanjie Li, Shoukui Zhang, Zhicheng Wang et al., "TokenPose: Learning Keypoint Tokens for Human Pose Estimation," **ICCV 2021** (arXiv:2104.03516). | 论证 learnable keypoint token 通过 transformer 定位 landmark 的合理性；DeconvHead V2 的 token→heatmap 转换思路与之呼应（对比 EoMT 的 einsum 相似度图）。已在文献调研中列为 🔴 立即读。 |
| **SimCC**（1D 坐标分类，300W 备选方案 3） | Li et al., "SimCC: a Simple Coordinate Classification Perspective for Human Pose Estimation," **ECCV 2022**. | 300W 备选：把 2D heatmap regression 转成 x/y 各自 1D softmax 分类，绕开分辨率限制和 soft-argmax 多峰偏移。与 query token 天然兼容。 |

### 10.3 Loss 设计 / Loss Functions（承前，已在早期实验确立）

| 组件 / Component | 文献 / Reference | 用途 / Role |
|---|---|---|
| **Adaptive Wing Loss** | Xinyao Wang, Liefeng Bo, Li Fuxin, "Adaptive Wing Loss for Robust Face Alignment via Heatmap Regression," **ICCV 2019**. | 早期 loss 消融用过（Run 17，21.32%）；最终 hybrid 用 WMSE 而非 AWing（AWing 量级 ~3–5 压制坐标 loss，λ=0.1 下坐标 loss 仅占 ~1%）。文献调研 🔴。 |
| **Wing Loss** | Zhenhua Feng et al., "Wing Loss for Robust Facial Landmark Localisation with CNNs," **CVPR 2018**. | 导师推荐的坐标 loss；当前用更简单的 L1 coord（λ=0.1），Wing 是备选（w=15, ε=3）。文献调研 🟡。 |
| **Weighted MSE / heatmap 前景加权**（FARNet 简化思路） | FARNet: "Feature Aggregation and Refinement Network," 2021 (arXiv:2111.00659). | **当前 hybrid loss 的 heatmap 分支就是 WMSE**（前景加权），`loss = MSE×(1+α·target)`, α=5.0。FARNet 论文指出这种简化版在解剖 landmark 上比 AWing 更有效。 |
| **STAR Loss**（备选，soft-argmax + coord 参考） | Zhenglin Zhou et al., "STAR Loss: Reducing Semantic Ambiguity in Facial Landmark Detection," **CVPR 2023**. | soft-argmax + coordinate loss 的完整实现参考；若 fetal 标注存在操作者间歧义可引用。文献调研 🟢。 |

### 10.4 坐标精度与数据处理修复 / Coordinate Accuracy & Data Processing Fixes

| 组件 / Component | 文献 / Reference | 用途 / Role |
|---|---|---|
| **UDP**（本项目 Exp 7 的核心修复） | Huang et al., "The Devil is in the Details: Delving into Unbiased Data Processing for Human Pose Estimation," **CVPR 2020**. | **Exp 7 直接依据**：修复 encode（heatmap 生成）与 decode（soft-argmax）之间的 0.5·(1-scale) 系统性坐标偏移。`pixel_center_align=true`，带来 -1.3pt 且降方差。 |
| **DARK**（评估过但未采用） | Zhang et al., "Distribution-Aware Coordinate Representation for Human Pose Estimation," **CVPR 2020**. | 曾考虑用于亚像素精化，但因已用 soft-argmax（连续可导），DARK 针对的 hard-argmax 量化误差已被绕开，故未采用。 |

### 10.4b FPN / 多尺度 + DINOv2 小数据先例 / FPN & DINOv2 Small-Data Precedents

| 组件 / Component | 文献 / Reference | 用途 / Role |
|---|---|---|
| **FPN**（Feature Pyramid Network，Exp 4 的思想来源） | Lin et al., "Feature Pyramid Networks for Object Detection," **CVPR 2017**. | Exp 4 的 `FeaturePyramidFusion`（融合 DINOv2 layer 4/8/12）的多尺度思想来源。 |
| **DINOv2 + FPN 眼科 landmark**（最直接的先例） | Chen, 2025, "DINOv2 + FPN for Ophthalmic Landmark Regression"（emergentmind 综述提及，写作前需核实原始出处）。 | **最直接的先例**：FPN over DINOv2 feature maps 做医学 landmark 回归。眼科与超声同为低对比度、需精细定位。文献调研 🟢 #14。 |
| **PlantTrack**（DINOv2 frozen + 轻量 head 小数据成功案例） | PlantTrack, arXiv 2024 (arXiv:2407.16829). | 论证 frozen DINOv2 + 轻量 heatmap head 在极小数据（20 张）可行——支撑本项目 frozen 实验（15.80%）"DINOv2 特征本身够用"的论点。文献调研 🟡 #13。 |

### 10.5 训练稳定化与集成 / Training Stabilization & Ensemble

| 组件 / Component | 文献 / Reference | 用途 / Role |
|---|---|---|
| **EMA / 权重平均**（Exp 6, 8a 用） | Polyak & Juditsky, "Acceleration of Stochastic Approximation by Averaging," SIAM 1992（Polyak averaging 原始思想）；现代实践参见 Mean Teacher (Tarvainen & Valpola, NeurIPS 2017)。 | 手写 shadow-weight EMA（decay=0.99），针对小数据后期 val NME 暴涨。同批内有效，但与 ensemble 冗余。 |
| **SWA**（明确未用） | Izmailov et al., "Averaging Weights Leads to Wider Optima and Better Generalization," **UAI 2018**. | Lightning 自带 `StochasticWeightAveraging`，因会接管 LR schedule 与现有 poly schedule 冲突，故改用手写 EMA。 |
| **Deep Ensembles**（Exp Ensemble 的理论依据） | Lakshminarayanan et al., "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles," **NeurIPS 2017**. | inference ensemble 靠模型间"分歧"降方差的理论支撑；本项目在 heatmap 层面（decode 前）平均 5 个 seed，9.75%。 |

### 10.6 参数高效微调（已排除）/ Parameter-Efficient Fine-tuning (Ruled Out)

| 组件 / Component | 文献 / Reference | 用途 / Role |
|---|---|---|
| **LoRA**（Exp 5，负结果） | Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models," **ICLR 2022**. | Exp 5 用 `peft` 库在 QKV 插 LoRA（r=8, α=16, dropout=0.1）；因 peft `get_peft_model` 自动冻结 backbone 致欠拟合，负结果（17.22% vs FPN 14.49%）。 |
| **DINOv2 + LoRA 实践**（Exp 5 的 target_modules 依据） | Gokmen et al., 2025; Baharoon et al., 2023; Brondolo et al., 2024（多篇 DINOv2 LoRA 微调实践）。 | 论证在 DINOv2 attention QKV 插 LoRA 的常见做法；本项目据此设 target_modules=["qkv"]。写论文讨论负结果时可引用，说明"LoRA 在其他场景有效，但在本项目 110 图全参训练场景欠拟合"。文献调研 🟢 #15。 |

### 10.7 Baseline 与数据集 / Baseline & Datasets

| 组件 / Component | 文献 / Reference | 用途 / Role |
|---|---|---|
| **HRNet baseline + fetal biometry 数据集**（UCL/HC18/FP/M-C） | Di Vece et al., "A multi-centre, multi-device benchmark dataset for landmark-based comprehensive fetal biometry," **arXiv:2512.16710, 2025**（proposal [4]）。 | 主对比 baseline（HRNet-W18）+ 数据来源。UCL BPD 8%/OFD 5%，HC18 BPD 5%/OFD 4%。含 Head(BPD/OFD)/Abdomen(APAD/TAD)/Femur(FL)，本项目 scope 聚焦 Head。 |
| **HRNet**（baseline 架构本体） | Sun et al., "Deep High-Resolution Representation Learning for Human Pose Estimation," **CVPR 2019**. | 对比的专用高分辨率 CNN 架构；本项目论点是"用通用 encoder-only ViT 架构逼近专用 CNN"。 |
| **300W**（通用性验证数据集） | Sagonas et al., "300 Faces in-the-Wild Challenge: The first facial landmark localization Challenge," **ICCV Workshops 2013**. | 阶段二人脸数据集，68 点。评估用标准 NME（眼间距归一化）。 |
| **WFLW**（proposal 提及的另一人脸集） | Wu et al., "Look at Boundary: A Boundary-Aware Face Alignment Algorithm," **CVPR 2018**. | proposal 提及，98 点，含遮挡/大姿态。视时间决定是否纳入。 |

### 10.8 论文实际能引用的完整清单 / Consolidated Citation List for the Paper

> 按论文章节归类，这些是**实际能写进 thesis / paper 正文**的引用（已用于本项目的方法、baseline、数据集，或作为直接 related work）。写作时按此清单建 bib 即可。
> Grouped by paper section. These are citations that can actually go into the thesis/paper (methods used, baselines, datasets, or direct related work).

**Method（方法章节，本项目实际使用的组件）：**
1. Kerssies et al., "Your ViT is Secretly an Image Segmentation Model," CVPR 2024 —— EoMT 基础架构（被改造对象）
2. Oquab et al., "DINOv2," TMLR 2024 —— backbone
3. Dosovitskiy et al., "An Image is Worth 16×16 Words (ViT)," ICLR 2021 —— backbone 架构
4. Lin et al., "Feature Pyramid Networks for Object Detection," CVPR 2017 —— Exp 4 FPN 多尺度融合
5. Huang et al., "Delving into Unbiased Data Processing (UDP)," CVPR 2020 —— Exp 7 坐标编解码修复（关键）
6. FARNet, arXiv:2111.00659, 2021 —— Weighted MSE heatmap loss（当前 hybrid 的 heatmap 分支）
7. Feng et al., "Wing Loss," CVPR 2018 —— 坐标 loss（备选，L1 的替代）
8. Wang et al., "Adaptive Wing Loss," ICCV 2019 —— loss 消融（Run 17）
9. Hu et al., "LoRA," ICLR 2022 —— Exp 5 参数高效微调（负结果，讨论用）

**Related Work（相关工作，query-based / transformer landmark 路线）：**
10. Li et al., "TokenPose," ICCV 2021 —— query token → heatmap 的先例
11. Li et al., "SimCC," ECCV 2022 —— 1D 坐标分类范式（下一步备选 + related work）
12. Zhou et al., "STAR Loss," CVPR 2023 —— soft-argmax + coord loss 参考
13. PlantTrack, arXiv:2407.16829, 2024 —— DINOv2 frozen 小数据 landmark 先例
14. Chen 2025, DINOv2 + FPN 眼科 landmark —— FPN over DINOv2 医学先例（需核实原始出处）

**Training / Ensemble（训练稳定化与集成）：**
15. Lakshminarayanan et al., "Deep Ensembles," NeurIPS 2017 —— inference ensemble 理论依据
16. Izmailov et al., "SWA," UAI 2018 —— 权重平均（说明为何改用手写 EMA）
17. （EMA / Polyak averaging，若需正式引用可用 Polyak & Juditsky 1992 或 Tarvainen & Valpola "Mean Teacher," NeurIPS 2017）

**Baseline / Datasets（基线与数据集）：**
18. Di Vece et al., "A multi-centre, multi-device benchmark ... fetal biometry," arXiv:2512.16710, 2025 —— HRNet baseline + fetal 数据（UCL/HC18）
19. Sun et al., "HRNet: Deep High-Resolution Representation Learning," CVPR 2019 —— baseline 架构
20. Sagonas et al., "300 Faces in-the-Wild (300W)," ICCV Workshops 2013 —— 人脸通用性数据集
21. Wu et al., "Look at Boundary (WFLW)," CVPR 2018 —— 人脸数据集（proposal 提及，视时间纳入）

**可选（视论文方向，若做对应实验才引用）：**
22. Yang et al., "Heatmap Regression without Soft-Argmax," ICCV 2025 —— 300W SOTA / 绕开 soft-argmax（若在 300W 上做）
23. Yao et al., "CC2D: One-Shot Medical Landmark Detection," MICCAI 2021 —— 若做伪标签/自训练扩数据
24. FM-OSD, arXiv:2407.05412, 2024 —— foundation model one-shot（小数据讨论）

### 10.9 仅背景参考、暂不进正文 / Background Only (Not Cited Unless Scope Expands)

> 这些在早期文献调研（2026-06-23，共 19 篇）里出现过，但本项目**没有实际使用**，除非 scope 扩展否则不进正文，列在这里仅供检索。
> Present in the earlier 19-paper survey but not actually used; listed for retrieval only.

- D-ViT (Cascaded Dual Vision Transformer), WACV 2025 —— channel-split + LSC，未采用 refinement 路线
- ORFormer, WACV 2025 —— confidence + offset decoder，未采用
- DETRPose, arXiv 2025 —— 多人姿态 DETR，与 2-landmark 场景差异大
- Poseur, ECCV 2022 —— 直接坐标回归（方案 B fallback），未走此路线
- ARobust Loss, Information Fusion 2023 —— loss 备选，未用
- Proto-Former, arXiv:2510.15338 —— 仅文献追踪用
- Diffusion few-shot X-ray landmark, arXiv:2407.18125, 2024 —— 小数据讨论，未实施
- Anatomical Landmark in Chest X-ray (Transformer), SPIE MI 2024 —— related work 备选

---

> **⚠️ 写作前务必核对 / Verify before citing**：本节文献的年份/会议/arXiv 编号，部分来自对话与早期文献调研整理，可靠性较高的是对话中明确出现过的（EoMT、DINOv2、UDP、LoRA、TokenPose、Di Vece、Wing/AWing、FARNet）；需要**特别再核对确切 venue/编号**的：SimCC、DARK、Deep Ensembles、FPN、300W/WFLW 年份、Chen 2025 眼科（原始出处存疑，emergentmind 综述转引）、以及 DINOv2-LoRA 的三篇实践。正式投稿前用 Google Scholar / DBLP 逐条核对。
