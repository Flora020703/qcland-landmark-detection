# 导师会议实验汇报整理（2026-08-08）

> 用途：集中记录 UCL/Multicentre 的 EoMT 与 HRNet 结果、RTMPose-s 实现验证、endpoint-ordering 审计，以及导师最终确认的评估协议和后续实验顺序。  

## 论文命名规则（2026-08-08 起）

- 论文题目不再使用 `EoMT` 作为核心名称，因为研究范围已经从“在 EoMT 上进行单一架构创新”扩展为胎儿生物测量 landmark localisation 的方法开发、跨 backbone 验证和多方法比较。
- 论文的贡献陈述、章节标题和正式结果表不再把本项目方法直接命名为 `EoMT`。在正式方法名确定前，表格暂用中性的 backbone-based 标签，例如 `DINOv2-based proposed model` 与 `DINOv3-based proposed model`；最终定稿时统一替换为正式方法名及其 backbone 变体。
- 这一命名调整不能抹去技术来源。Methods、implementation details 或 reproducibility statement 中仍须明确说明该模型是在 EoMT-style query-based segmentation architecture/codebase 基础上适配并发展为 landmark localisation 方法，并准确列出本项目新增和修改的组件。
- HRNet-W18 与 RTMPose-s 是外部比较方法，可以保留其正式方法名称；历史审计文件、代码路径和实验归档中的 `eomt` 字样保留，不做追溯重命名。
- 当前汇报材料中既有的 `EoMT–DINOv2/EoMT–DINOv3` 仅是历史可追溯标签，不应直接复制到论文最终表格。

## Chapter 5 方法论结果的统一重评估规则（2026-08-09 起）

- Chapter 5 中所有胎儿两端点任务的正式数值，包括 BPD staged-development、OFD supporting comparison、resolution/augmentation sensitivity 和仍保留的负结果，统一迁移到 permutation-invariant NME；不能在同一方法论论证中混用旧 fixed-channel 数字。
- 正式方法论表优先使用五个预设 seeds 的 final-checkpoint single-model mean ± seed-level sample SD；不以 inference ensemble 作为方法成立的主要证据。
- 已有逐图预测的实验直接重新聚合；只有 final checkpoint 的实验只补推理；只有历史 aggregate 且 checkpoint/逐图预测不可恢复的实验不得换算成新指标，需删除正式定量结论、降为有明确标签的历史探索记录，或在确属核心缺口时补跑。
- BPD 的核心四档是 original einsum head、DeconvHeadV2、+FPN、+FPN+UDP；OFD 的核心支持档是 DeconvHeadV2 baseline 与 +FPN+UDP。augmentation/rotation-scale 是否作为额外档位单独呈现，须在核心表冻结后决定，不能与无增强链混成严格单变量消融。
- Loss 与 sigma screening 属于早期 exploratory/single-seed 证据；若无法恢复为 permutation-invariant NME，不得继续把旧 fixed-channel 数字放进当前正式主表，只能保留其历史设计动机并明确证据限制。
- 300W 不属于无序两端点任务，继续使用标准 68-landmark、outer-eye-corner-normalised NME，不采用 permutation-invariant endpoint matching。

完整的 BPD 方法开发时间线必须覆盖以下阶段，而不能只呈现四档组件表：

1. early target-width screening：在早期 original-einsum/plain-MSE 条件下比较 Gaussian target sigma；
2. loss screening：plain/weighted heatmap loss、hybrid coordinate term，以及未保留的 Adaptive Wing 尝试；
3. core architecture sequence：original einsum head → DeconvHeadV2 → +FPN-style fusion → +FPN+UDP；
4. investigated but not retained：EMA、test-time augmentation、LoRA/frozen backbone、128×128 heatmap 等；
5. final augmentation development：rotation-only/rotation+scale 等实际存在且可核实的配置，最后形成用于正式跨方法比较的 rotation+scale recipe；
6. resolution sensitivity 与其他仍在 Chapter 5 中承担结论的实验，作为独立 sensitivity analysis 呈现。

这些阶段按真实时间顺序叙述，但至少拆成“早期单-seed screening”“五-seed核心架构开发”“未采用分支”“最终 augmentation/sensitivity”四个证据块。它们不能被排成一张暗示每一步都严格 matched、都由前一步单变量演进而来的总消融表。
> **重要更新（2026-08-07，当前权威口径）**：导师已经明确决定，论文最终主指标改为 **permutation-invariant NME**。下面原有的 fixed-channel、x-sort 和 frozen-DOD 内容保留为历史审计记录，用于解释为什么改变指标；凡与本节冲突之处，均以本节为准。

---

# 零、当前权威决定与实验执行计划（后续写作以本节为准）

## 0.1 导师最终确认的评估定义

胎儿生物测量的两个 endpoint 共同定义同一条测量直径，其临床含义不随两点交换而改变。因此，最终评估把预测与真值都视为**无序点对**。对每张图计算：

```text
E_direct  = d(p0, g0) + d(p1, g1)
E_crossed = d(p0, g1) + d(p1, g0)
E         = min(E_direct, E_crossed)
NME       = E / (2 * d(g0, g1))
```

随后沿用既有聚合方式：先在每个 seed 内对测试图像取平均，再报告五个预设 seeds 的 mean ± sample SD。该运算是**评估指标定义**，不是利用 GT 修改模型预测，也不是推理时的后处理。

该指标必须完全一致地用于：

- EoMT–DINOv2；
- EoMT–DINOv3；
- reproduced HRNet-W18；
- RTMPose-s。

主结果不再使用 prediction x-sort、frozen-DOD canonicalisation 或 raw fixed-channel NME。它们只保留在 Appendix/审计记录中，用于说明 endpoint-channel convention 为什么会显著影响旧指标。

正式跨方法比较必须同时满足：

- 同一 `dataset × task` 下使用跨方法共同图像文件名交集；
- 使用原图坐标空间中的相同 permutation-invariant evaluator；
- 显式报告每格的共同样本数 `n`；
- 报告五 seed 单模型 mean ± seed-level sample SD；
- 不用 ensemble 作为主结果。

## 0.1.1 2026-08-08 最新完成状态：UCL-BPD 已补齐，EoMT/HRNet 已达 30/30

本节取代下文所有“UCL-BPD EoMT checkpoint/逐图预测不可恢复”“仅完成 28/30 cells”或“UCL-BPD 仍为 `Unavailable`”的旧状态描述。旧段落仅作为时间顺序审计记录保留，不得再用于当前汇报或论文数字。

### 补跑范围与配置

仅重跑 UCL-BPD 的两个最终 EoMT 配置，没有重跑历史消融链：

- EoMT–DINOv2：`42, 0, 123, 2024, 3407` 五 seeds；
- EoMT–DINOv3：相同五 seeds；
- 每个 run 使用 `512×512` 整图直接 resize、DeconvHeadV2、FPN layers `[4,8,12]`、`64×64` heatmap、`sigma=4`、pixel-centre alignment、rotation `±30°`（概率 0.6）、scale `[0.75,1.25]`、horizontal flip、final/last checkpoint；
- 训练与历史最终配置一致，训练标签仍采用 x-coordinate channel ordering；训练期 validation/checkpoint 行为没有因最终评价指标改变而回溯修改；
- 正式比较只使用五个 final checkpoints 的单模型结果，不使用 inference ensemble。

### 逐图数据完整性

两个 backbone 均已保存：

- 5 个 `test_image_order.csv`；
- 5 个 `final_fixedchannel_per_image.csv`（历史/审计用途）；
- 5 个 `final_swapmin_per_image.csv`（文件名沿用历史命名，内容即当前正式 `permutation_invariant_nme`）；
- 每个 CSV 均为表头加 49 张 UCL-BPD Test 图像，即 50 行；
- 每个 seed 的 final checkpoint、实际运行配置、训练结果 TSV 和总日志均已归档。

DINOv2 本地归档：

```text
checkpoint_backups/ucl-bpd-eomt-dinov2-5seed-permutation-invariant-20260808.tar
SHA-256: 0baad54a8b4b93faf81127d526f561e8ba7c69f1d87db24e395eea9f61c9cef0
```

DINOv3 与完整 30/30 分析已完成本地归档并通过内容计数和可读性检查：

```text
checkpoint_backups/ucl-bpd-eomt-dinov3-5seed-and-complete-30of30-20260808.tar
SHA-256: 8bfc5c1bfce19798cc9bce6adf94674d7316fb0e4b2b9d156e63afe7302011bc
```

归档包含 5 个 final checkpoints、5 个逐 seed 配置、5 套 UCL-BPD DINOv3 image-order/fixed/permutation-invariant CSV、完整 30/30 分析输出，以及 30 个 common-subset per-image CSV。服务器上的 DINOv3 checkpoint 因而可在路径复核后清理；正式逐图结果和分析输出暂时保留到 RTMPose canary 汇总完成。

### 统一重评分验收

权威运行目录：

```text
/root/autodl-tmp/endpoint_ordering_analysis/
permutation_invariant_final_20260808_complete_30of30/
```

验收结果：

- `30/30 cells scored, 0/30 excluded`；
- `excluded_images.tsv` 只有表头；
- `cross_method_gt_consistency_warnings.tsv` 只有表头；
- 每个 `dataset × task` 均在跨方法共同文件名交集上重新聚合；
- UCL-BPD 三个方法均为 `n=49`；
- Multicentre BPD/OFD 的共同子集分别为 `n=1180/1189`；其余任务为 APAD `n=161`、TAD `n=161`、FL `n=362`；
- 汇总报告的是五个 seed 单模型均值与 seed-level sample SD。

### 当前 EoMT/HRNet 正式主表

**Permutation-invariant NME (%) ± seed-level sample SD；括号内为跨方法共同样本数。**

| Train/Test | Method | BPD | OFD | APAD | TAD | FL |
|---|---|---:|---:|---:|---:|---:|
| UCL | EoMT–DINOv2 | 5.92±0.66 (49) | 3.94±0.55 (49) | 6.50±1.15 (36) | 9.32±1.33 (36) | 1.61±0.11 (39) |
| UCL | EoMT–DINOv3 | 5.14±0.61 (49) | 3.75±0.09 (49) | 5.34±0.66 (36) | 6.47±0.86 (36) | 1.53±0.15 (39) |
| UCL | HRNet-W18 | 5.51±0.46 (49) | 4.59±0.49 (49) | 7.33±1.30 (36) | 6.74±0.98 (36) | 1.99±0.41 (39) |
| Multicentre | EoMT–DINOv2 | 6.90±0.17 (1180) | 6.13±0.45 (1189) | 6.99±0.23 (161) | 8.82±0.90 (161) | 2.78±0.11 (362) |
| Multicentre | EoMT–DINOv3 | 6.09±0.19 (1180) | 5.73±0.68 (1189) | 7.11±0.21 (161) | 8.71±0.54 (161) | 2.71±0.13 (362) |
| Multicentre | HRNet-W18 | 4.68±0.17 (1180) | 4.78±0.17 (1189) | 8.89±0.19 (161) | 8.75±0.33 (161) | 2.93±0.32 (362) |

这张表是当前 EoMT/HRNet 的正式比较底稿。不能再用旧 fixed-channel 表中的 BPD/TAD 大差距解释模型定位能力，也不能将 x-sort、DOD 或 raw-channel 重评分混入该主表。相关结果仅作为 Appendix 中解释评价协议演变的 implementation audit。

### 当前生成器的两个已知文档问题

本次 30/30 计算数值和 TSV 已通过门禁，但生成的 Markdown 尾注仍残留“UCL BPD EoMT cells are unavailable”，且代码中的 `EXPECTED_MISSING` 仍列出两个已经恢复的 cells。这两处属于过时说明，不影响数值；正式冻结前必须把缺失集合更新为空并将尾注改成按实际缺失集合动态生成，然后重跑同一分析。

### 周末并行推进的方法论评估

RTMPose seed-42 canary 完成、周一等待导师确认期间，不空等：

1. 盘点 BPD staged-development 与 OFD supporting experiment 的 checkpoint/逐图预测；
2. 有逐图预测的实验直接按 permutation-invariant NME 重新聚合；
3. 只有 checkpoint 的实验只重新推理，不重训；
4. 只有历史 aggregate、无预测和 checkpoint 的实验保留为 original ordered-channel historical result，不换算、不用于最终胜负结论；
5. 优先覆盖真正支撑 Chapter 5 方法结论的 BPD 核心 rungs 和 OFD baseline/final-recipe 对照；
6. 胎儿双端点实验统一采用 permutation-invariant NME；300W 仍使用其标准 68-point inter-ocular NME，因为任务定义不同。

## 0.1.2 RTMPose-s UCL-BPD seed-42 formal canary（2026-08-08）

本次 formal canary 的完整证据包已经复制到本地并通过归档完整性验收：

- 归档：`checkpoint_backups/rtmpose-ucl-bpd-seed42-formal-canary-20260808.tar`
- SHA-256：`64a4e52ff574da3e3a4e0474dc3b8a94e369ded61f6ca85b77d1e96a79f3d924`
- `sha256sum -c`：通过（`OK`）
- 内容计数：1 个 `epoch_200.pth`、1 个 predictions JSON、1 个 per-image CSV、1 个 summary JSON、1 个 provenance JSON、1 个正式 canary config、23 张 overlay PNG，以及 preflight/smoke/formal-canary 日志各 1 份。

因此 checkpoint、逐图预测、配置、provenance、可视化和三阶段运行日志已经形成可追溯且可恢复的本地证据链。服务器副本目前只占约 83 MB，无需为释放空间立即删除；至少保留到周一导师审阅 canary 并决定是否扩展正式 runs。

正式 canary 已完成 200 epochs，并严格使用 true final checkpoint `epoch_200.pth`：

```text
checkpoint SHA-256: 99dea67573eef660a055cc1f8ae0d7ee8113af00f6acb5310f9325a98fbe394e
```

实现与 provenance 验收：

- Train-only internal split：100 train / 10 internal-val；released Test 为 49 张；三者文件名交集均为 0；
- Test 只在 200-epoch 训练和 final-checkpoint 验证完成后读取；
- pretrained CSPNeXt-s backbone checkpoint SHA-256：`aa7d9335bf422ad02a803e36f357dfc6abb807eca42d79e8b3b6e7c5bd1f446b`；
- 242 个 backbone keys 全部精确加载，无 missing、unexpected 或 value mismatch；two-keypoint RTMCC head 随机初始化；
- 实际总参数量 `5,445,076`，其中 backbone `4,378,320`、head `1,066,756`；全部参数可训练；
- 输入为整图直接 resize 到 `512×512`，SimCC sigma `(8,8)`；preflight、非正方形坐标闭环、forward/backward 和独立 1-epoch Runner smoke 均通过；
- final Test 保存 49 条原图坐标空间 prediction、逐图 CSV、summary JSON 和 provenance JSON。

正式 canary 结果：

| Metric | Value |
|---|---:|
| Permutation-invariant NME（主指标） | 10.4338% |
| Fixed-channel NME（审计） | 10.4338% |
| Test images | 49 |
| Per-image median | 5.3530% |
| Per-image P75 | 9.6851% |
| Per-image P90 | 21.2800% |
| Per-image P95 | 33.6228% |
| Per-image maximum | 75.5875% |
| Images improved by crossed assignment | 0 |

这是单 seed canary，`14.8758%` 是 49 张图之间的 per-image sample SD，**不是 seed-level SD**，不得写成 `10.43±14.88%` 与五-seed方法结果比较。训练内部 validation 在 epoch 200 为约 `5.42%`，而 released Test 为 `10.43%`，显示明显的 validation–test generalisation gap。

人工审查了 best 10、median 附近 3 张和 worst 10 的原图 overlay（绿色为 GT，红色为 prediction）。结论：

- best 样本中 prediction 与 GT 几乎重合，排除统一的 resize/逆变换缩放或平移错误；
- 最严重的 `005_12HC.jpeg`（75.59%）和 `005_14HC.jpeg`（70.44%）中，模型预测了另一条解剖上合理但错误的颅骨直径，方向接近图中已有 OFD 测量，而不是 BPD GT；
- `009_HC.jpeg` 中一个预测端点落到超声扇形上方黑色区域，是明确的定位失败；
- 其他差样本主要表现为斜率、端点位置或直径选择错误，不呈现一致坐标偏移；
- subject `005` 同时包含 1.65%/2.80% 的优秀帧和 70% 以上的失败帧，因此目前更支持 frame/measurement-cue-specific failure，而不是简单 subject-level domain shift；
- fixed 与 permutation-invariant 数值完全一致，说明此次 canary 的高均值不是 endpoint assignment 引起。

因此该 canary 在工程与评价层面有效，但性能均值受少数极端失败显著拉高。按导师要求，周一先汇报本次 canary、逐图分布和 overlay，再决定是否启动其余 RTMPose runs；当前不得提前扩展五-seed/全任务训练。

## 0.2 RTMPose 在最终表中的结构

RTMPose-s 不使用 DINO backbone，也没有 DINOv2/DINOv3 两套变体。每个 dataset/task 只训练一套 measurement-specific RTMPose-s（CSPNeXt-s + two-keypoint RTMCCHead），因此最终表中只有**一行 RTMPose-s**：

| Method | BPD | OFD | APAD | TAD | FL |
|---|---|---|---|---|---|
| EoMT–DINOv2 | 5-seed result | 5-seed result | 5-seed result | 5-seed result | 5-seed result |
| EoMT–DINOv3 | 5-seed result | 5-seed result | 5-seed result | 5-seed result | 5-seed result |
| HRNet-W18 | 5-seed result | 5-seed result | 5-seed result | 5-seed result | 5-seed result |
| **RTMPose-s** | **one 5-seed result** | **one 5-seed result** | **one 5-seed result** | **one 5-seed result** | **one 5-seed result** |

RTMPose 不复制成 DINOv2/DINOv3 两行，也不为了与 EoMT 行数一致而人为增加 backbone 实验。

## 0.3 当前数据状态

- HRNet：UCL 25 runs 与 Multicentre 25 runs 已完成，并保留 predictions/per-image CSV。
- EoMT Multicentre：五任务 × DINOv2/DINOv3 × 五 seeds 的逐图数据可用于统一重评。
- EoMT UCL：OFD/APAD/TAD/FL 的两种 backbone 逐图数据可用。
- EoMT UCL BPD：DINOv2 与 DINOv3 两格的逐图预测和 checkpoint 当前不可恢复，需要补齐最终配置。
- RTMPose：真实环境 preflight 和独立 1-epoch engineering smoke 已通过；正式 seed-42 canary 尚待运行/完成。

## 0.4 后续实验顺序（固定执行顺序）

### 阶段 1：统一重评已有 EoMT/HRNet 结果并冻结评估器

1. 使用最新版 `endpoint_ordering_analysis/rescore_endpoint_conventions.py`。
2. 对现有 UCL 与 Multicentre 逐图预测统一计算 permutation-invariant NME。
3. 正式表使用每个 `dataset × task` 的跨方法共同图像子集；各方法自己的完整测试池结果仅作补充材料。
4. 通过以下门禁后冻结当前表：
   - 单元测试全部通过；
   - HRNet 独立 permutation-invariant sanity checks 全部通过；
   - `cross_method_gt_consistency_warnings.tsv` 只有表头；
   - 除 UCL-BPD 两个 EoMT cell 外没有其他缺失；
   - 同任务各可用方法的 common-subset `n` 完全一致。

### 阶段 2：补齐 UCL-BPD 的两个 EoMT 最终配置

只补：

- UCL BPD EoMT–DINOv2：五 seeds；
- UCL BPD EoMT–DINOv3：五 seeds。

不重跑整个历史消融链。优先寻找本地 checkpoint 或可恢复的原始预测；若均不存在，才重新训练这十个最终配置 runs。每个 run 必须保存 checkpoint、原始两通道预测坐标、GT、文件名顺序、配置、seed、provenance 和逐图 permutation-invariant NME。补齐后重新运行同一 evaluator，使两格从 `Unavailable` 变成正式结果，再冻结完整 EoMT/HRNet 表。

### 阶段 3：RTMPose-s UCL-BPD seed-42 canary —— 已完成，导师已书面确认（2026-08-10）

按以下顺序执行，已全部完成：

1. `PREFLIGHT_ONLY=1`；
2. 独立 1-epoch engineering smoke（若当前服务器版本已通过，只需保留日志，不把 smoke 数字当实验结果）；
3. UCL-BPD seed 42、完整训练周期的 formal canary；
4. 保存逐图原图空间预测，并使用相同 permutation-invariant evaluator；
5. 汇报 convergence、最终 NME、参数量、coordinate round trip、overlay 和 provenance；
6. 在导师查看 seed-42 canary 后再扩展正式 runs。

RTMPose 训练仍输出两个通道，但最终主评估把两 endpoint 当作无序点对，因此不需要为了主指标在推理时加入 prediction x-sort 或 frozen-DOD 后处理。

**导师回复（2026-08-10，原文节选，逐字记录）**：

> I think the seed-42 canary is technically valid based on the checks you have performed. In particular, the coordinate round-trip, SimCC encoding/decoding, gradient flow, pretrained-weight loading and full training-pipeline verification give me confidence that there is no obvious implementation issue. The fact that the fixed-channel and permutation-invariant NME are identical also confirms that the relatively high Test NME is not caused by endpoint assignment. ... as long as the implementation and experimental protocol are sound, I don't think we need to be concerned that its performance is lower than EoMT. We should report the result as obtained under the same evaluation protocol rather than trying to optimize the baseline based on the Test performance. Please proceed with the remaining four UCL BPD seeds using exactly the same configuration, so that we can obtain a reliable five-seed mean and seed-level standard deviation. Once we have the five-seed BPD result, we can proceed with the remaining measurements using the same RTMPose-s configuration.

要点：导师明确认可 RTMPose-s 作为"公平实现的 baseline"这一定位——它的分数比 proposed model 低不构成担忧，也**不应该为了让 Test 表现更好而反过来调整 baseline**，只需要在相同协议下如实报告。canary 阶段正式关闭，进入阶段 4。

### 阶段 4：RTMPose-s 完整实验 —— 已启动（2026-08-10）

导师已确认 canary，按批准的顺序推进：

1. **当前步骤**：完成 UCL BPD 其余四个 seeds，随后（同一脚本、显式确认后）继续完成全部 50 个 RTMPose-s run。

**2026-08-10 更新，脚本合并**：一个相关审阅发现原计划的 `run_rtmpose_bpd_remaining_seeds.sh`（只覆盖 BPD 4 个 seed）在防覆盖、共享 split 完整性、"完全相同配置"强制锁定这三方面都弱于本项目已有的 HRNet 50-run driver（`baseline_reproduction/run_hrnet_512_fixed_5seed.sh`）范式。与其维护两个各自实现同一套加固逻辑、容易互相漂移的脚本，不如直接按 HRNet driver 的成熟模式（resumable TSV、拒绝静默 resume 半成品目录、磁盘预检查）写一个覆盖全部 2 datasets × 5 tasks × 5 seeds = 50 run 的统一脚本，`run_rtmpose_bpd_remaining_seeds.sh` 已删除，替换为 `rtmpose_reproduction/run_rtmpose_full_sweep.sh`：

- **防覆盖**（相关审阅问题 1）：复用 HRNet driver 的 `RESULTS_TSV` + `is_recorded()` 跳过已完成项模式；若某 run 的工作目录存在但没有对应的 `summary.json`，直接报错要求人工检查/归档该目录，绝不自动 resume。
- **共享 split 完整性**（问题 2）：每个 (dataset, task) 的 internal-split/Train/Val/Test COCO json 只生成一次，生成时做内容自检（internal-train/internal-val/Test 三者互不重叠）并记录四个文件的 SHA-256 到独立 manifest；同一 cell 后续每个 seed 开跑前都会重新核对这份 manifest，一旦文件被改动/损坏立刻报错，不会静默用不同数据训练剩下的 seed。UCL/BPD 这一 cell 直接复用 seed-42 canary 已生成的四个文件（不重新生成任何"看起来一样"的副本）。
- **"完全相同配置"锁定**（问题 3）：`MAX_EPOCHS` 硬编码为 200（不再是可被环境变量覆盖的默认值）；预训练 checkpoint SHA-256 硬编码为 canary provenance 记录的 `aa7d9335bf422ad02a803e36f357dfc6abb807eca42d79e8b3b6e7c5bd1f446b`，每个 run 都强制核对；每个生成的 config 都会用 `mmengine.Config.fromfile` 读回后逐字段断言（`randomness.seed`、`train_cfg.max_epochs`、`backbone.init_cfg.checkpoint`、`codec.input_size` 等），比对存量 config 文件的文本 diff 更明确、也不依赖能找到某个历史 config 文件。

脚本按 UCL BPD 五个 seed（含预置的 seed-42 canary 结果）优先处理。`evaluate_rtmpose_fixed.py` 的逐图 CSV schema 与 HRNet 的完全一致（`swap_min_nme` 列即 permutation-invariant NME），最终聚合直接复用这一列，不需要额外解析逻辑。

**2026-08-10 第二轮加固（相关审阅发现第一版仍有 3 个真实问题，加上服务器磁盘现实约束）**：

- **完成状态可能与磁盘产物不一致**：原来的"TSV 有记录就跳过、work_dir 存在但没有 summary.json 就报错"漏掉了一种真实情况——训练评分全部完成、`summary.json` 已生成，但脚本在写入 TSV 之前中断，重新运行时会在已完整的目录上重新训练，覆盖好结果。已改为四态状态机（`run_artifacts_status`）：fresh（全新，正常开始）/ recoverable（5 个产物文件都在但 TSV 没记录，从 `summary.json` 严格恢复——同时核对逐图 CSV 行数与 `n` 一致、config 里记录的 seed 与预期一致，不是只看文件存在）/ complete（TSV 和 5 个产物都在，跳过）/ inconsistent（介于两者之间的任何情况，直接报错要求人工检查，绝不自动判断）。
- **配置断言范围不够**：之前只测了 5 个字段却在注释里宣称锁定了架构、batch size、augmentation、SimCC、checkpoint 策略等一整套配置。现在 `verify_config()` 把 `make_config.py` 模板里实际设置的每一个关键字段都读回来断言了一遍——CSPNeXt-s/RTMCCHead 的具体超参、SimCC codec（`sigma=(8.0,8.0)`、`simcc_split_ratio=2.0` 等）、batch_size=16、optimizer/scheduler（AdamW、按 batch size 线性缩放的 lr、`clip_grad`）、checkpoint 策略（`save_last=True`、`max_keep_ckpts=1`、确认没有 `save_best`）、train/val pipeline 的增强步骤顺序，以及 internal-train/val/Test 三个 annotation 文件路径是否真的接到了对应 dataloader 上——总计 40+ 项断言，每一条都是从 `make_config.py` 真实模板里核对出来的，不是猜的。
- **UCL BPD manifest 不是 canary 原始锁定值**：核实后发现 canary 归档本身就没有保存原始的 split/COCO json 文件（只有 checkpoint、predictions、per-image CSV、summary、provenance、config、overlay），所以"跟归档里的原始 json 比对哈希"这条路本来就走不通。改用一个更严谨、也真正可行的替代方案：把当前服务器上 `UCL_BPD_test.json` 的每张图 GT 坐标,跟已经归档并被导师批准的 `UCL_BPD_seed42_canary_per_image.csv`（`evaluate_rtmpose_fixed.py` 当时基于同一批 GT 算出来的）逐张比对，49 张图坐标必须完全一致——这样验证的其实是更本质的问题："现在的 Test 集内容是不是 canary 当时真正打分时用的那批"，而不是一个从未存在过的原始文件的哈希。
- **磁盘预算现实**：服务器系统盘和数据盘目前都只剩约 8GB，远不够连续跑完剩下 45 个 run（每个 5-seed cell 的 checkpoint+日志+预测大约 400–500MB）。脚本默认改为**每完成一个 cell（一个 dataset×task 的 5 个 seed）就暂停**（`STOP_AFTER_EACH_CELL=1`，默认开启），而不只是 BPD 之后暂停一次；脚本本身从不自动删除任何文件。新增配套脚本 `backup_and_clean_cell.sh`，严格按"训练→推理→打分→本地备份→本地校验→服务器删除 checkpoint"的顺序：`backup` 子命令打包一个 cell 的全部产物成 tar 并打印 SHA-256 和 scp 命令（不删除任何东西）；`clean` 子命令要求传入你在**本地**独立算出的 SHA-256，跟服务器上archive 自己的哈希核对一致后，才删除该 cell 的 `work_dir`（checkpoint+训练日志，真正占空间的部分）——config/provenance/predictions/per-image CSV/summary 这五个状态机需要的轻量文件永远保留，删除 checkpoint 后 `run_rtmpose_full_sweep.sh` 仍能正确识别该 cell 为已完成。

**2026-08-10 第三轮加固（相关审阅发现第二版仍有 3 个会立即卡住流程的阻塞性问题 + 3 个备份安全问题）**：

- **阻塞 1，seed-42 命名不一致**：状态机和 `backup_and_clean_cell.sh` 之前都统一用 `${dataset}_${task}_seed${seed}_run` 拼文件名，但 canary 的真实产物用的是 `_canary` 后缀（`UCL_BPD_seed42_canary_summary.json` 等）。TSV 预填了 seed 42 这一行，但脚本按错误的文件名去找 canary 产物，会在第一个 cell 的第一个 seed 就判定为 `inconsistent` 并硬退出。两个脚本都新增了 `run_name_for(dataset, task, seed)` 函数，只对 UCL/BPD/seed=42 返回 `UCL_BPD_seed42_canary`，其余一律返回原有的 `_run` 命名——两处实现必须保持同步，其中一处改了另一处忘改会重新引入这个 bug。
- **阻塞 2，重启后卡在第一个已完成的 cell**：`process_cell()` 原来处理完一个 cell（不管是不是全部 SKIP）就无条件 `exit 0`，导致 UCL/BPD 备份清理完之后重新运行脚本，会把 5 个 seed 全部判定为 SKIP，然后仍然退出，永远到不了 UCL/OFD。改为新增 `CELL_DID_WORK` 全局标志，每个 cell 开始前清零，只有 `run_one()` 真正训练了新 seed 或从 `recoverable` 状态恢复了结果才置 1；主循环只在 `CELL_DID_WORK=1` 且 `STOP_AFTER_EACH_CELL=1` 时才暂停，否则静默继续下一个 cell。
- **阻塞 3，backup→clean 不会真正释放磁盘空间**：`do_backup()` 会在 `cell_backups/` 下打一个 ~400-500MB 的 tar，但 `do_clean()` 原来只删除原始 work_dir，从不删除这个 tar 本身，净释放空间约等于 0。`do_clean()` 现在在确认本地 SHA-256 匹配后，同时删除服务器上的 archive tar 和它的哈希清单文件，才是真正释放空间。
- **备份安全 1，只验证 summary 会静默打包残缺档案**：原来只检查 TSV 有 5 行记录 + `summary.json` 存在，其余文件复制全部用 `[ -f "$f" ] && cp`（缺失就跳过、不报错），checkpoint 数量不对也只是 WARNING。`verify_cell_complete()` 现在对每个 seed 硬性要求：work directory、恰好一个 `epoch_*.pth`、config、provenance、predictions、per-image CSV、summary 全部存在，任一缺失或 checkpoint 数量不是 5，直接报错退出、拒绝生成 archive。
- **备份安全 2，archive 内部缺少可本地验证的 manifest**：原来只有服务器绝对路径的外层 tar 哈希清单，没有 archive 内部、相对路径的文件级 manifest。现在在打 tar 之前，会在 staging 目录内生成 `ARCHIVE_CONTENTS.sha256`（覆盖归档内每一个文件的相对路径哈希），本地解压后可以直接 `sha256sum -c ARCHIVE_CONTENTS.sha256` 逐文件校验，而不仅仅是校验外层 tar 没有传输损坏。
- **备份安全 3，删除目标缺少白名单和绝对路径约束**：`DATASET`/`TASK` 原来直接拼进 `rm -rf` 路径，没有任何校验。现在新增 `DATASET ∈ {UCL, MULTICENTRE}`、`TASK ∈ {BPD, OFD, APAD, TAD, FL}` 白名单校验；每个即将删除的目录（work_dir 和 archive 本身）都用 `realpath` 校验确实严格位于 `$ARTIFACT_ROOT` 内部，并核对 basename 与 `run_name_for()` 预期命名完全一致，才允许删除。
- **额外，"strict recovery" 名不副实**：`recoverable` 状态原来只做浅层核对（summary 有 3 个 key、逐图行数等于 n、config 里的 seed 匹配），不是真正重新核实数字正确。现在改为对 `recoverable` 状态重新纯打分：拿现有 predictions JSON + GT JSON 独立跑一遍 `evaluate_rtmpose_fixed.py` 到临时目录，与已有 summary/per-image 逐项（全部 summary key、全部逐图 CSV 列，按文件名对齐）核对一致后才把结果写回 TSV，不需要重新训练。
- **额外，配置断言文档超过代码**：注释曾声称锁定了 scheduler，但代码里没有任何 `param_scheduler` 断言；augmentation 检查也只覆盖 pipeline 步骤类型和顺序，没有覆盖实际参数值。现在 `verify_config()` 新增了 `param_scheduler`（LinearLR + CosineAnnealingLR 两项，逐字段核对，数值全部从 `make_config.py` 真实公式独立重新推导，不是照抄）、`auto_scale_lr` 确实不存在的断言、`train_pipeline` 里 `flip_prob` 的断言，以及 `training_recipe_summary` 全部 7 个字段的内部一致性核对。经过核实 `FetalRotateScaleColorJitter` 的旋转/缩放/颜色抖动具体范围是硬编码在 transform 类内部、根本不是 config 参数，因此没有为这部分再加断言——这些已经是 config 层面能核对的全部内容。

以上两个脚本的全部改动均已 `bash -n` 语法检查通过，提交到 `master`（commit `b15d799`）。

**2026-08-10 第四轮加固（相关审阅第三轮修复后又发现 3 个真实但更小的问题，正式训练前一并修完）**：

- **备份没有真正验证 epoch_200.pth**：原来的 checkpoint 检查只是"`epoch_*.pth` 数量恰好为 1"，如果目录里只有 `epoch_195.pth`（训练中途失败），一样能通过，静默备份一个没跑完 200 epoch 的 run。新增 `rtmpose_common.sh` 里的 `verify_final_checkpoint()`：要求 `epoch_200.pth` 存在且非空、目录内没有其它 `epoch_*.pth`、`last_checkpoint` 指针确实指向这个文件——两个脚本（训练完成时的确认、备份前的确认）统一调用同一个函数。
- **complete 状态仍然只检查文件存在**：四态状态机原来只统计 5 个文件是否都在，从不重新核对内容——如果一个"complete"的 cell 被后续拷贝/手动操作/磁盘故障破坏，会被永久静默信任并跳过。`recoverable` 恢复时也只核对了 config 的 seed 字段（没跑完整的 ~45 项配置断言），且逐图 CSV 是直接读进 `{filename: row}` 字典，重复文件名会被静默覆盖、不会报错。新增 `validate_run_content.py`，是配置断言 + 独立重新评分一致性 + 逐图 CSV 行数与 n 一致/无重复文件名 + provenance 交叉核对的唯一实现，`run_rtmpose_full_sweep.sh`（训练前的 config-only 模式；"recoverable"和现在也包括"complete"的 full 模式，跳过前会重新核实一遍）和 `backup_and_clean_cell.sh`（备份前的 full 模式）统一调用同一份代码，不再各自维护一份可能走样的逻辑。
- **cell 归档缺少两类轻量证据**：原来的 archive 里没有 results TSV、没有 COCO 转换时的 `*_excluded.json`、也没有软件版本记录。`do_backup()` 现在同时归档完整版和该 cell 单独提取的 results TSV、3 个 excluded-image 日志、以及两个 driver 脚本自身的副本；`record_run_provenance.py` 新增记录 python/torch/mmcv/mmengine/mmpose 版本号（向后兼容的新增字段，不需要重新跑已批准的 canary）。
- **清理不是 all-or-nothing**：`do_clean()` 原来是边验证边删除（每个 seed 验证通过就立刻删），如果第三个目录验证失败，前两个已经删了，形成部分清理状态。现在改为先验证全部 5 个 work_dir + archive 本身，全部通过后再统一进入第二个循环删除。
- **额外清理**：`run_name_for()`/共享 COCO json 路径/excluded-log 路径的计算逻辑现在都在共用的 `rtmpose_common.sh` 里，两个脚本 source 同一份，不再各自维护一份容易漂移的拷贝（第三轮就因为这个模式踩过一次真实 bug）；删除了本轮重构后完全废弃的 ~370 行历史代码（`legacy_run_one_unused`/`legacy_verify_config_unused`）。
- **新增 `test_sweep_lifecycle_synthetic.sh`**：按相关审阅的建议，做了一次不训练的端到端合成测试，覆盖 backup→本地式 manifest 校验→clean 全流程，使用真实的 `make_config.py`/`validate_run_content.py`/`record_run_provenance.py`/`evaluate_rtmpose_fixed.py`/`backup_and_clean_cell.sh` 代码路径，只有需要 GPU 的训练/推理两步是伪造的（predictions 直接等于 GT，确定性地产生 0% NME，只为验证机制而非产出有意义的数字）。`run_rtmpose_full_sweep.sh` 自身的 fresh/recoverable/complete 四态转换和跨调用的暂停/继续循环不在这次合成测试的覆盖范围内（该脚本顶层会立即执行本地测试套件/导入 mmpose/做真实磁盘检查，不能安全地当作库来 source）——建议直接让脚本处理第一个真实 seed 来验证这部分：它会立刻走到"recoverable"路径去恢复已批准的 seed-42 canary，这本身就是一次真实的生产行为验证。

以上全部改动已 `bash -n`/`python -m py_compile` 语法检查通过，提交到 `master`（commit `b142db3`）。RTMPose 完整 50-run 扫描尚未在服务器上真正启动；建议先在服务器上跑一次 `test_sweep_lifecycle_synthetic.sh` 确认 backup/clean 生命周期无误，再正式启动 UCL/BPD 剩余 4 个 seed。

2. 完成 UCL OFD/APAD/TAD/FL，各五 seeds；
3. 完成 Multicentre BPD/OFD/APAD/TAD/FL，各五 seeds。

总计为 `2 datasets × 5 tasks × 5 seeds = 50 RTMPose-s runs`（UCL BPD 的 5 个已经算在内：canary 1 个 + 本次 4 个），每个 cell 只有一个 RTMPose-s 方法结果，不存在 DINOv2/DINOv3 分支。

### 阶段 5：统一修改论文

等 EoMT UCL-BPD 缺口和 RTMPose 结果齐全后再一次性更新：

- Chapter 4：把 permutation-invariant NME 写成正式主指标；
- Chapter 5：更新 common-subset 主表、RTMPose 单一方法行和全部结果分析；
- Chapter 6：删除依赖旧 fixed-channel 胜负或 correspondence-gap 的结论并重新讨论；
- Chapter 1/3/Appendix：统一贡献表述、训练通道与无序点评估的区别，以及历史审计记录。

## 0.5 不再采用的旧方案

以下内容仅作为审计历史，不再作为最终主协议：

- fixed-channel NME 作为论文主指标；
- 用 prediction x-sort 或 frozen DOD 作为当前模型的主评估 canonicalisation；
- 要求 RTMPose 在 DOD 与 x-sort 之间做最终指标选择；
- 为追求主表数字而使用 five-model inference ensemble；
- 把 GT-informed minimum 描述成”推理修正”。它现在被正式定义为无序点对的 permutation-invariant evaluation metric。

## 0.6 本轮决策：命名规则适用边界、BPD 消融链取舍、OFD 是否补链、300W 处理方式（2026-08-09）

真实服务器运行已确认 UCL-BPD 缺口已补齐（30/30 cells scored, 0 excluded），`endpoint_ordering_analysis/` 脚本据此同步更新：`EXPECTED_MISSING` 已清空为空集合，`permutation_invariant_nme_final_table.md` 的缺失说明改为按本次实际缺失单元动态生成，不再硬编码”UCL BPD EoMT cells are Unavailable”这句现在已经不真实的话。28/28 本地测试通过，端到端脚本同步更新验证。

以下是本轮需要决策的四个问题。

### 决策 4：EoMT 命名规则的适用边界

命名规则本身（论文不用 EoMT，暂用 `Proposed model (DINOv2)` / `Proposed model (DINOv3)`，Methods 中披露技术来源）已经明确，无需进一步讨论。唯一需要补充的边界是**规则只在”从审计数据抄写进论文 LaTeX 表格”这一步生效**：

- 论文正文、章节标题、贡献陈述、最终结果表：一律使用中性名称，不出现 `EoMT`；
- 本项目自己的分析脚本、代码仓库、checkpoint 路径、审计归档——包括 `endpoint_ordering_analysis/` 生成的所有 TSV/MD，例如 `permutation_invariant_nme_final_table.md` 里 “EoMT-DINOv2”/”EoMT-DINOv3” 的列标签——**不需要重命名**，属于可追溯的内部审计产物，与最终论文表格是两个不同层级的制品，重命名反而会破坏可追溯性；
- 结论：往论文里誊抄数字时才做改名，脚本/归档保持原名不动。

### 决策 5：BPD 消融链——完整六阶段叙事 vs. 可验证的定量主表

当前资产盘点（本文档 0.1 节 + 周末并行推进计划）：`+FPN+UDP`（含 loader-seed 版本）本地有完整归档；augmentation 阶段的 `rotation+scale` 已经用新指标跑出正式结果（DINOv2 5.92±0.66%、DINOv3 5.14±0.61%，n=49）；resolution 256/512 已有逐图 permutation-invariant 结果；sigma/loss screening 多为单 seed exploratory；EMA/TTA/LoRA/128×128 多数有 checkpoint 可补推理；**early einsum head、DeconvHeadV2、+FPN 三档目前没有确认的 checkpoint**，是最大的缺口。

这里有一个新指标出现之后才凸显的风险，必须明确指出：**BPD 是 near-vertical 任务**（本项目 `dod_vectors.py` 的几何分析已确认，BPD 的 direction vector dx/dy 比值只有 0.005–0.16），也正是 fixed-channel 分数对 endpoint-correspondence 最敏感的两个任务之一（另一个是 TAD）。如果 early 三档只能用旧的 fixed-channel/native 数字，而 `+FPN+UDP` 及之后各档用新的 permutation-invariant 数字，放进**同一张**消融表，会制造一个”看起来是架构改进、实际上部分是指标定义改变”的假象——这正是本项目花了一整个审计流程（endpoint-ordering 分析、permutation-invariant metric 的采用）去避免的同一类问题；如果在内部消融表里放任它发生，会与主比较表已经达到的严谨程度自相矛盾。

**决策（2026-08-09 更新：用户明确选择补跑，不降档）**：

1. 不允许在同一张定量表里混用两种指标，哪怕只是权宜之计——BPD 恰好是最容易被指标混淆误导的任务，风险是真实的，不是吹毛求疵。这一条不变。
2. ~~降级为文字说明~~ **已否决**。用户明确要求：直接补跑 einsum head、DeconvHeadV2、+FPN 三档，跑完后与其余所有 BPD 实验一起，统一用当前的 permutation-invariant evaluator 重新计算，替换掉旧的 fixed-channel 数字——不再讨论”要不要降级”，默认动作改为”确认缺口 → 补跑 → 统一重算”。
3. 补跑范围与成本核算见下方 0.6.1 节的完整清单；确认的最小补跑规模是 3 个配置 × 5 seeds = **15 个新 training run**（einsum head、DeconvHeadV2、+FPN 各 5 seeds）。
4. 补跑必须复用这三档**历史上实际使用的设置**（sigma、loss、是否开启 augmentation 等），只把架构头替换回历史版本，其余全部与当时保持一致——否则会把”架构变化”和”顺手多改的其他设置”混在一起，破坏消融本身的单变量结构。正在核实这三档的确切历史配置（见 0.6.1 末尾），确认后会给出可以直接提交的 config。
5. 补跑完成后，这三档连同 `+FPN+UDP`、augmentation 阶段、resolution sensitivity 等所有已有逐图数据的 BPD 实验，全部按跟正式跨方法比较完全相同的方式重新计算：同一个 `permutation_invariant_nme` 定义、five-seed final-checkpoint single-model mean ± seed-level sample SD、不用 ensemble。旧的 fixed-channel 数字全部替换，不再并存。

### 0.6.1 BPD/OFD 完整补跑与统一重算清单

下表覆盖 BPD 全部六阶段时间线与 OFD 支持链的每一个分支，标注状态：

- **补跑**：确认没有可恢复的 checkpoint/逐图预测，必须重新训练；
- **只需推理**：有 final checkpoint，但还没有 permutation-invariant 逐图预测，只需重新跑一次推理（不需要重新训练）；
- **只需重新聚合**：已经有 permutation-invariant 逐图预测（例如已经跑过 `rescore_endpoint_conventions.py` 或等价流程），直接重新汇总五-seed mean±SD 即可；
- **待核实**：本文档目前的记录不足以判断，需要在服务器上确认后才能定案。

| 阶段 | 配置 | 状态 | Seeds | 备注 |
|---|---|---|---|---|
| 早期 target screening | sigma=1（original einsum/plain-MSE） | 只需推理 | 单 seed（历史如此） | 有 checkpoint；单 seed 结果不能算 seed-level SD，仍标注为 exploratory |
| 早期 target screening | sigma=4（同上） | 只需推理 | 单 seed | 同上 |
| Loss screening | plain MSE / weighted MSE / hybrid / Adaptive Wing | 待核实 | 单 seed | 需要在服务器确认哪些还有 checkpoint/逐图预测 |
| **核心架构** | **Original einsum head** | **补跑** | **5**（新） | 无 checkpoint；需先核实历史 sigma/loss/augmentation 设置 |
| **核心架构** | **DeconvHeadV2**（无 FPN） | **补跑** | **5**（新） | 同上 |
| **核心架构** | **+FPN-style fusion**（无 UDP） | **补跑** | **5**（新） | 同上 |
| 核心架构 | +FPN+UDP（无 augmentation） | 待核实 | 5 | 本地有”+FPN+UDP loader-seed”归档，但该归档是否等同于这一档本身的基线 checkpoint，需要确认 |
| Augmentation | +FPN+UDP + rotation only | 待核实 | 5 | 未在既有记录中明确确认是否单独存在（不同于 rotation+scale 最终版） |
| Augmentation | +FPN+UDP + rotation+scale（最终 recipe） | **已完成** | 5 | 已是新指标结果：DINOv2 5.92±0.66%、DINOv3 5.14±0.61%（n=49），与正式跨方法比较共用同一份数据 |
| Resolution sensitivity | 256×256 | 只需重新聚合 | 待确认 5 seeds 是否齐全 | 已有逐图 permutation-invariant 结果，需确认完整性 |
| Resolution sensitivity | 512×512 | 已完成 | 5 | 与最终 recipe 相同 |
| Not retained | EMA（raw vs materialised，同一 run 配对） | 待核实 | — | 历史上存在 checkpoint-来源阶段不确定的问题（见项目记忆 Rule 相关记录），需要先确认是否有真正配对干净的 raw/EMA checkpoint，否则只能补跑一次新的、干净配对的 run（不是 5-seed sweep，是单次 matched-pair 验证） |
| Not retained | TTA（flip，同一 checkpoint 有/无对比） | 只需推理 | 沿用已有 checkpoint 的 seeds | 纯推理时操作，任何已有 checkpoint 都可以直接做，不需要额外训练 |
| Not retained | LoRA/frozen backbone | 待核实 | — | 需确认是否有独立训练过的 checkpoint |
| Not retained | 128×128 heatmap | 待核实 | — | 改变模型输出分辨率，若无 checkpoint 需重新训练；需先核实 |
| OFD 支持链 | DeconvHeadV2 baseline | 只需推理 | 5（历史） | checkpoint 基本完整 |
| OFD 支持链 | +FPN+UDP | 只需推理 | 5（历史） | 同上 |
| OFD 支持链 | +FPN+UDP+rotation+scale（最终 recipe） | 只需推理/已部分完成 | 5 | 若尚未用新指标重新聚合，按同一流程补 |

**确认需要补跑（无争议）**：einsum head、DeconvHeadV2、+FPN 三档 × 5 seeds = **15 个新 run**。

**待核实清单**（需要在服务器上逐项确认 checkpoint/逐图预测是否存在，才能定出补跑总数的上限）：+FPN+UDP 无增强基线、rotation-only 阶段、loss screening 各配置、EMA 配对、LoRA/frozen backbone、128×128 heatmap、resolution 256 的 seeds 完整性。这些确认后，”补跑总数”可能从 15 个上升，具体取决于核实结果。

正在核实 einsum/DeconvHeadV2/+FPN 三档历史上使用的确切配置（sigma、loss、augmentation 开关等），确认后会在本节补充可直接提交的三份 config，以及一份服务器端 checkpoint/归档核实清单（针对上表”待核实”的所有条目）。

### 决策 6：OFD 是否需要补消融链——需要，”新指标下数字可能不好看”不构成跳过的理由

OFD baseline / `+FPN+UDP` / `+rotation+scale` 三档的 checkpoint 基本完整，只需补推理，成本很低（不需要重新训练）。用户的顾虑是”新指标下 OFD 的结果不一定好看”，但：

- 这个顾虑本身不能成为跳过分析的理由：本项目从 EMA、augmentation、correspondence diagnostic 到这次 permutation-invariant metric 的每一轮决策，一贯坚持”先算出来再如实报告”，从未因为担心结果不理想而回避某项已经可以低成本完成的验证；
- OFD 是本项目自己确认过的**near-horizontal 任务**（x-sort 与 DOD 几乎总是一致），从机制上讲，permutation-invariant 重评分预期不会像 BPD/TAD 那样发生数量级变化——如果实际算出来确实变化很大，那本身就是一个新的、值得写进论文的真实发现（可能说明 OFD 也存在此前未察觉的 correspondence sensitivity），而不是应该被藏起来的坏消息；
- 该分析成本低（推理而非重训），与”是否要为了赶时间牺牲某些实验”的资源约束无关，不适用”权衡截止日期”的例外条款。

**决策**：按计划正常补齐 OFD 消融链，用同一个 permutation-invariant 指标重新聚合，如实报告结果，无论方向如何。

### 决策 7：300W（DINOv2/DINOv3）实验——不需要按新指标重新评估，只需要改命名

300W 是标准 68-landmark 人脸关键点任务，每个 landmark 都有独立的语义身份，不存在”两个点互换意义不变”的情况，因此从一开始就不属于 permutation-invariant matching 想解决的问题范畴——这一点本文档 0.1 节已经写明，是既定结论，不需要重新讨论是否换指标。

需要做的只有两件事：

1. **命名**：300W 表格里如果出现 “EoMT-DINOv2/DINOv3” 字样，按决策 4 的同一规则改成 `Proposed model (DINOv2)` / `Proposed model (DINOv3)`，与胎儿两端点任务的表格保持全文一致；
2. **数据来源核实（可选，非阻塞）**：目前 300W 章节使用的数字来自 `EoMT_Landmark_Detection_Full_Chronological_Summary_2026-07-21.md` 的历史记录，不是直接从服务器原始 per-seed TSV 重新推导（此前原始 TSV 本地不可访问）。如果时间允许，值得做一次一次性核对：直接从 `face300w_5seed_results.tsv`（或等价文件）重新汇总，确认与 MD 摘要一致，再把引用来源冻结为原始 TSV。这不是阻塞项，时间不够可以维持现状。

### 小结

| 问题 | 决策 |
|---|---|
| EoMT 命名 | 论文表格/正文用 `Proposed model (DINOv2/DINOv3)`；脚本、代码、审计归档不改名，只在誊抄进论文时才改 |
| BPD 消融链 | 不允许同一张表混用新旧指标；早期三档（einsum/DeconvHeadV2/+FPN）若 checkpoint 找不到，降级为文字历史说明，不报具体数字；定量主表从 `+FPN+UDP` 开始，向后用统一 permutation-invariant 指标覆盖 augmentation 和 resolution sensitivity；重训 15 个 run 仅在导师明确要求时才启动 |
| OFD 消融链 | 正常补齐，成本低（只需推理）；”数字可能不好看”不是跳过的理由，如实报告 |
| 300W | 不换指标，继续用标准 68-point NME；只改方法名称标签；原始 TSV 复核为可选加分项，非阻塞 |

---

> **以下原文是 2026-08-07 导师最终决定之前形成的会议材料，保留用于追溯实验过程。其中 fixed-channel 主指标、common-DOD 推荐方案及待导师确认的问题均已被上面的第零节取代，不可直接复制进论文当前版本。**

## 一、建议先用一分钟说明目前的总体状态

1. EoMT 的 UCL 与 Multicentre 五任务、DINOv2/DINOv3、五 seeds 实验已经完成；UCL 和 Multicentre 的 HRNet 也都已经按最终协议重跑完成（合计 50/50 runs）。
2. 最终对比协议已统一到：五个 measurement-specific 模型、`512×512` 输入、相同 released Train/Test partition、相同五个 seeds，以及 fixed-channel NME。
3. RTMPose-s 的正式 UCL-BPD seed-42 canary 尚未开始，但代码级 preflight 和独立 1-epoch Runner smoke test 都已通过。
4. 在落实“RTMPose、HRNet、EoMT 使用相同 endpoint ordering”时，确认 EoMT 和 HRNet 历史上实际采用了两种不同规则：EoMT 是逐图 x-sort，HRNet 是训练集估计的 frozen DOD。现已对已有逐图预测做了统一外部重评分，需要导师确认 RTMPose 正式训练采用哪一种，以及最终比较如何表述。

---

# 二、UCL 与 Multicentre 的 HRNet 复现及 EoMT–HRNet 最终比较

## 2.1 实验对象与运行规模

Multicentre 不是一个联合输出五项测量的单模型。与原表格的真实训练方式一致，每项测量使用一个独立模型：

- Head：BPD、OFD；
- Abdomen：APAD、TAD；
- Femur：FL。

正式比较包含：

| 方法 | Backbone | 任务数 | Seeds | Multicentre 训练数 |
|---|---|---:|---|---:|
| EoMT | DINOv2 ViT-S/14（重采样为 patch-16 token grid） | 5 | 42, 0, 123, 2024, 3407 | 25 |
| EoMT | DINOv3 ViT-S/16 | 5 | 同上 | 25 |
| HRNet | HRNetV2-W18 | 5 | 同上 | 25 |

HRNet 自动脚本同时完成了 UCL 25 runs 和 Multicentre 25 runs，因此整批 HRNet 是 `2 datasets × 5 tasks × 5 seeds = 50 runs`。两组都属于正式复现结果，本节同时报告，不再只展开 Multicentre。

## 2.2 对齐的外部比较协议

| 项目 | EoMT–DINOv2 / DINOv3 | HRNet-W18 |
|---|---|---|
| 模型粒度 | 每项测量一个独立两点模型 | 每项测量一个独立两点模型 |
| 数据 | released Multicentre Train/Test partition | 相同 released partition |
| 输入 | 直接 resize 到 `512×512`，不保留长宽比 | `512×512` |
| Seeds | `42, 0, 123, 2024, 3407` | 相同五个 controlled seeds |
| 主报告 checkpoint | final/last | final state |
| 主评估 | fixed-channel NME | 同一 fixed-channel NME 实现 |
| 输出 | 每 seed aggregate + per-image 坐标/NME | 每 seed aggregate + predictions + per-image NME |

这里的“对齐”是 **external comparison protocol 对齐**，不是声称两个方法内部训练完全相同。两种架构仍保留各自的 target、loss、optimizer 和增强实现。

## 2.3 UCL 与 Multicentre EoMT 的完整训练配置

UCL 和 Multicentre 都使用在 BPD staged development 中形成的同一完整 recipe，每个任务和 backbone 独立训练。Multicentre 是把该 recipe 应用于 pooled dataset，不在 Multicentre 上重新逐组件选配置。两组的主要差异只是对应的数据模块、released CSV 路径与训练样本池；下列模型、输入、target、优化及 seeds 设置一致。

| 类别 | 配置 |
|---|---|
| Encoder | DINOv2 ViT-S/14-reg4 或 DINOv3 ViT-S/16；backbone 可训练 |
| Queries / blocks | 2 learned queries；3 decoder blocks；masked attention 开启 |
| Landmark head | DeconvHeadV2 + FPN-style fusion |
| FPN layers | `[4, 8, 12]` |
| 输入 / heatmap | `512×512` / `64×64` |
| Target | Gaussian heatmap，`sigma=4.0`，pixel-centre alignment 开启 |
| Loss | hybrid/weighted heatmap-coordinate recipe：`alpha=5`、`temperature=10`、`lambda_coord=0.1` |
| Optimisation | `lr=1e-4`、LLRD `0.8`、weight decay `0.05`、poly power `0.9`、warmup `[15,30]` |
| Augmentation | horizontal flip；rotation；scale；实际 matched config 中 `rotate_augment=true`、`scale_augment=true` |
| Epoch / batch | 最多 200 epochs；batch size 16 |
| Validation | Train 内部 10%，`val_split_seed=42`；每 5 epochs 验证；early-stopping patience 20、min delta 0.005 |
| Reproducibility | model seed 与 DataLoader seed 均设为当前 seed |

必须如实说明：这些 EoMT checkpoints 当时的 validation/checkpoint selection 使用了其历史配置中的 endpoint-order-invariant evaluator，但现在论文最终表统一对 **同一 final checkpoint** 使用 fixed-channel NME 重评分；没有利用新 fixed-channel test 结果重新选择 checkpoint。

## 2.4 UCL 与 Multicentre HRNet 的完整训练配置

| 类别 | 配置 |
|---|---|
| Architecture | released HRNetV2-W18 fetal landmark implementation；UCL/Multicentre 各五个 measurement-specific models |
| 输入 / heatmap | `512×512` / 原生 stride-4 `128×128` |
| Gaussian target | `sigma=1.0`（保留原方法设定；与 EoMT sigma=4、RTMPose SimCC sigma=8 含义不同） |
| Seeds | `42, 0, 123, 2024, 3407`，训练脚本补入 controlled seed 支持 |
| 其余训练 recipe | 保留 released HRNet 的 MSE、augmentation、optimizer 和 DOD 逻辑 |
| Checkpoint | final state 为主报告口径 |
| 评估 | 从保留 predictions 用统一 fixed-channel evaluator 重算，并保留逐图 CSV |

为正确支持 512 输入，做了两项必要的、已审计的 correctness adaptation：

1. 将 target heatmap 从原 256 输入对应的 `64×64` 改成模型在 512 输入下真实输出的 `128×128`，否则 loss shape 不一致；
2. 把 `function.py` 三处写死的 decode size `[64,64]` 改为 `config.MODEL.HEATMAP_SIZE`。否则训练可以继续，但 validation checkpoint selection 和最终坐标会被错误的 factor-of-two 变换静默污染。

此外还有 controlled-seed patch，以及为新版 Python/PyTorch 所需的兼容性补丁；这些不改变网络方法本身。修复前产生的 512 结果全部作废并删除，正式 50 runs 均在修复验证后重新运行。

## 2.5 UCL 与 Multicentre 总表：各方法五-seed final fixed-channel NME ± seed-level SD

下面把两个训练/测试数据集、三个已完成的方法配置和五个小任务融合在同一张表中，结构对应原论文的跨数据表，但本论文当前只比较 **within-dataset** 设置，即 UCL→UCL 和 Multicentre→Multicentre。`mean ± SD` 行是五个预设 seeds 的 final single-model **fixed-channel NME mean ± seed-level sample SD**；`ensemble` 行是相同五个 final checkpoints 组成的单个 heatmap-ensemble 点估计，因此没有 seed-level SD。两类统计量并列展示，但不能当成相同统计单位直接排名。

<table>
  <caption><strong>统一 512×512、final-checkpoint、fixed-channel NME (%)：single-model mean ± seed-level SD，并列 five-model ensemble 点估计</strong></caption>
  <thead>
    <tr>
      <th rowspan="3">Train</th>
      <th rowspan="3">Test</th>
      <th rowspan="3">Method</th>
      <th colspan="5">Fixed-channel NME (%) ± seed-level SD</th>
    </tr>
    <tr>
      <th colspan="2">Head</th>
      <th colspan="2">Abdomen</th>
      <th>Femur</th>
    </tr>
    <tr>
      <th>BPD</th>
      <th>OFD</th>
      <th>APAD</th>
      <th>TAD</th>
      <th>FL</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="5">UCL</td>
      <td rowspan="5">UCL</td>
      <td>EoMT–DINOv2</td>
      <td>10.99 ± 4.34</td>
      <td>4.31 ± 0.68</td>
      <td>7.53 ± 1.41</td>
      <td>12.61 ± 2.83</td>
      <td>1.83 ± 0.11</td>
    </tr>
    <tr>
      <td>EoMT–DINOv2 ensemble (5 models)</td>
      <td>9.94</td>
      <td>4.15</td>
      <td>7.08</td>
      <td>10.03</td>
      <td>1.69</td>
    </tr>
    <tr>
      <td>EoMT–DINOv3</td>
      <td>9.58 ± 3.15</td>
      <td>4.09 ± 0.12</td>
      <td>6.09 ± 0.84</td>
      <td>6.44 ± 0.89</td>
      <td>1.73 ± 0.16</td>
    </tr>
    <tr>
      <td>EoMT–DINOv3 ensemble (5 models)</td>
      <td>10.40</td>
      <td>4.13</td>
      <td>5.85</td>
      <td>5.82</td>
      <td>1.64</td>
    </tr>
    <tr>
      <td>HRNet-W18 (reproduced)</td>
      <td>5.57 ± 0.48</td>
      <td>4.72 ± 0.49</td>
      <td>7.36 ± 1.35</td>
      <td>6.90 ± 0.98</td>
      <td>1.99 ± 0.41</td>
    </tr>
    <tr>
      <td rowspan="5">Multicentre</td>
      <td rowspan="5">Multicentre</td>
      <td>EoMT–DINOv2</td>
      <td>20.33 ± 1.08</td>
      <td>7.34 ± 0.45</td>
      <td>9.17 ± 0.32</td>
      <td>29.26 ± 1.33</td>
      <td>3.29 ± 0.15</td>
    </tr>
    <tr>
      <td>EoMT–DINOv2 ensemble (5 models)</td>
      <td>22.29</td>
      <td>6.71</td>
      <td>8.68</td>
      <td>28.31</td>
      <td>3.06</td>
    </tr>
    <tr>
      <td>EoMT–DINOv3</td>
      <td>19.00 ± 0.90</td>
      <td>6.90 ± 0.52</td>
      <td>9.26 ± 0.24</td>
      <td>27.53 ± 0.57</td>
      <td>3.16 ± 0.18</td>
    </tr>
    <tr>
      <td>EoMT–DINOv3 ensemble (5 models)</td>
      <td>17.37</td>
      <td>6.59</td>
      <td>9.17</td>
      <td>26.53</td>
      <td>2.98</td>
    </tr>
    <tr>
      <td>HRNet-W18 (reproduced)</td>
      <td>4.81 ± 0.18</td>
      <td>4.90 ± 0.17</td>
      <td>8.89 ± 0.19</td>
      <td>8.78 ± 0.33</td>
      <td>2.94 ± 0.32</td>
    </tr>
    <tr>
      <td>UCL / Multicentre</td>
      <td>same dataset</td>
      <td>RTMPose-s</td>
      <td colspan="5"><em>Pending: first run UCL-BPD seed-42 formal canary after endpoint-ordering confirmation</em></td>
    </tr>
  </tbody>
</table>

测试图像数：UCL 的 BPD/OFD/APAD/TAD/FL 分别为 `49/49/36/36/39`；Multicentre EoMT 及其 ensemble 分别为 `1191/1191/161/161/362`。Multicentre HRNet loader 对 BPD/OFD 过滤无效行后分别为 `1180/1189`，其余为 `161/161/362`。因此 Multicentre BPD/OFD 的严格方法差异推断使用下一节的 common-subset 配对结果，而不是仅根据这张 full-pool 总表判断。

Ensemble 口径说明：UCL 数字是 matched rotation+scale 配置的 five-final-checkpoint heatmap ensemble；Multicentre 数字是原完整测试池上的 channel-aligned、final-decode 后重新 canonicalised five-model heatmap ensemble，均为 fixed-channel NME。Multicentre common-subset 分析没有重新生成 ensemble，因此不能把这里的 full-pool ensemble 数字冒充 common-subset ensemble。HRNet 没有运行与 EoMT 语义等价、预先审计的五模型 heatmap ensemble，所以表中不补一个 HRNet ensemble 行；跨方法主比较仍应使用每种方法的五-run single-model mean ± seed SD。

这张表暂不加粗“最佳方法”，原因有二：第一，RTMPose 尚未完成；第二，endpoint analysis 已证明 BPD/TAD 的 fixed-channel 数值对 channel convention 高度敏感。在导师决定最终 endpoint convention 和论文报告策略前，直接加粗排名可能造成过度结论。

### 2.5.1 UCL HRNet 25-run 复现结果与 UCL 方法比较

UCL HRNet 五个任务的逐 seed fixed-channel 结果如下，证明 UCL 的 25 runs 也已正式完成：

| Task | seed42 | seed0 | seed123 | seed2024 | seed3407 | Mean ± seed SD | Test n |
|---|---:|---:|---:|---:|---:|---:|---:|
| BPD | 4.8089 | 6.1399 | 5.6699 | 5.7200 | 5.5201 | **5.5718 ± 0.4845** | 49 |
| OFD | 5.5946 | 4.4527 | 4.5592 | 4.4657 | 4.5442 | **4.7233 ± 0.4893** | 49 |
| APAD | 8.1955 | 9.1031 | 7.3436 | 6.4893 | 5.6921 | **7.3647 ± 1.3488** | 36 |
| TAD | 8.0817 | 7.0870 | 7.4482 | 6.3119 | 5.5765 | **6.9011 ± 0.9784** | 36 |
| FL | 2.1197 | 2.1790 | 2.1790 | 1.2640 | 2.2139 | **1.9911 ± 0.4079** | 39 |

UCL 的 EoMT–HRNet 逐图 paired comparison 可恢复 8 个 cells；UCL-BPD 的 EoMT checkpoint/per-image 文件已缺失，因此 BPD 只能在上方总表使用权威 aggregate TSV，不能补作 image-level CI。

| Task | Backbone | n | EoMT | HRNet | EoMT−HRNet（pp） | Paired image-bootstrap 95% CI |
|---|---|---:|---:|---:|---:|---:|
| OFD | DINOv2 | 49 | 4.31 | 4.72 | -0.41 | [-1.56, +0.37] |
| OFD | DINOv3 | 49 | 4.09 | 4.72 | -0.63 | [-2.12, +0.32] |
| APAD | DINOv2 | 36 | 7.53 | 7.36 | +0.16 | [-2.21, +2.61] |
| APAD | DINOv3 | 36 | 6.09 | 7.36 | -1.27 | [-4.17, +1.50] |
| TAD | DINOv2 | 36 | 12.61 | 6.90 | +5.71 | [+1.64, +10.88] |
| TAD | DINOv3 | 36 | 6.44 | 6.90 | -0.46 | [-3.27, +2.17] |
| FL | DINOv2 | 39 | 1.83 | 1.99 | -0.16 | [-1.86, +0.77] |
| FL | DINOv3 | 39 | 1.73 | 1.99 | -0.26 | [-1.96, +0.68] |

除 TAD-DINOv2 外，七个 paired mean-difference CI 均跨 0。不能把点估计较低直接写成已证明优于 HRNet。TAD-DINOv2 的 EoMT NME 明显更高；DINOv3 在相同 TAD 数据上的点估计接近 HRNet，但 CI 仍较宽。

### 2.5.2 Multicentre 完整测试集上的方法内结果

| Task | EoMT–DINOv2 | EoMT–DINOv3 | HRNet-W18 |
|---|---:|---:|---:|
| BPD | 20.33 ± 1.08%（n=1191） | 19.00 ± 0.90%（n=1191） | 4.81 ± 0.18%（n=1180） |
| OFD | 7.34 ± 0.45%（n=1191） | 6.90 ± 0.52%（n=1191） | 4.90 ± 0.17%（n=1189） |
| APAD | 9.17 ± 0.32%（n=161） | 9.26 ± 0.24%（n=161） | 8.89 ± 0.19%（n=161） |
| TAD | 29.26 ± 1.33%（n=161） | 27.53 ± 0.57%（n=161） | 8.78 ± 0.33%（n=161） |
| FL | 3.29 ± 0.15%（n=362） | 3.16 ± 0.18%（n=362） | 2.94 ± 0.32%（n=362） |

注意：BPD/OFD 的 HRNet released loader 会过滤少量无效 landmark 行，因此其 n 分别是 1180/1189，而 EoMT 完整测试池为 1191。不能直接把上述 full-pool 均值当成严格逐图配对统计。

### 2.5.3 Multicentre 相同 common subset 上的 EoMT–HRNet 逐图配对比较

下表中的差值为 `EoMT fixed-channel NME − HRNet fixed-channel NME`；正值表示 EoMT 数值更高。CI 是逐图 paired bootstrap 95% CI。

| Task | Backbone | n | EoMT | HRNet | 差值（pp） | 95% CI（pp） |
|---|---|---:|---:|---:|---:|---:|
| BPD | DINOv2 | 1180 | 20.25 | 4.81 | +15.43 | [13.78, 17.12] |
| BPD | DINOv3 | 1180 | 18.86 | 4.81 | +14.05 | [12.43, 15.76] |
| OFD | DINOv2 | 1189 | 7.31 | 4.90 | +2.40 | [2.07, 2.76] |
| OFD | DINOv3 | 1189 | 6.87 | 4.90 | +1.97 | [1.66, 2.29] |
| APAD | DINOv2 | 161 | 9.17 | 8.89 | +0.28 | [-0.63, 1.15] |
| APAD | DINOv3 | 161 | 9.26 | 8.89 | +0.37 | [-0.45, 1.13] |
| TAD | DINOv2 | 161 | 29.26 | 8.78 | +20.48 | [15.25, 25.90] |
| TAD | DINOv3 | 161 | 27.53 | 8.78 | +18.75 | [13.73, 24.14] |
| FL | DINOv2 | 362 | 3.29 | 2.94 | +0.35 | [-0.31, 0.96] |
| FL | DINOv3 | 362 | 3.16 | 2.94 | +0.22 | [-0.41, 0.79] |

### 2.5.4 Multicentre DINOv2–DINOv3 方法内比较

五个任务的 paired seed difference（DINOv3 − DINOv2）如下：

| Task | 平均差值（pp） | 5-seed t-CI | 方向性说明 |
|---|---:|---:|---|
| BPD | -1.33 | [-3.74, +1.07] | 5/5 seeds 数值偏向 DINOv3，但 exact sign p=0.0625 |
| OFD | -0.43 | [-0.96, +0.09] | 5/5 偏向 DINOv3，CI 跨 0 |
| APAD | +0.08 | [-0.59, +0.76] | 方向混合 |
| TAD | -1.73 | [-3.51, +0.04] | 5/5 偏向 DINOv3，CI 跨 0 |
| FL | -0.13 | [-0.50, +0.24] | 4/5 偏向 DINOv3，CI 跨 0 |

因此，fixed-channel 下没有任何一个任务的五-seed CI 排除 0。不能再写“Multicentre 上 backbone effect 显著反转”或“DINOv3 已被证明更好”；最多只能报告点估计方向和不确定性。

## 2.6 当前对 Multicentre 结果最稳妥的解释

- 在统一的历史 fixed-channel 定义下，HRNet 在 BPD、OFD、TAD 上明显低于当前 EoMT；APAD 和 FL 的差值较小，paired mean CI 跨 0。
- 这不等于“EoMT 的全部方法开发没有贡献”。消融回答的是 EoMT 内部哪些设计变化与性能变化相关；最终比较回答的是该 EoMT recipe 相对其他架构的表现。二者是不同研究问题。
- BPD/TAD 的异常大差距与 endpoint ordering 非常敏感，后面的统一重评分表明：报告数值的一大部分取决于 x-sort 与 DOD 的 fixed-channel 定义。但重评分不是重训，因此不能直接把 DOD 重评分后的低值替换进最终主表，然后声称完成了完全公平训练比较。
- released Multicentre partition 继续与原工作保持一致；其中已确认存在 UCL patient overlap。按会议决定不做 corrected-split 重训，但论文必须披露，且不能称为 unseen-centre 或严格 patient-disjoint generalisation。

## 2.7 数据与归档状态

- HRNet 512/fixed 50-run archive：50 个 predictions、50 个 fixed-channel CSV、50 个 final states 在归档时均验证存在；本地归档约 1.9 GB，SHA-256 已记录。服务器 final states 后来删除以回收约 1.83 GiB，但 predictions、CSV 和 paired analyses 保留。
- EoMT Multicentre common-subset archive：已验证可读。
- Multicentre paired-analysis archive：10 个逐图 paired CSV + 1 个 summary TSV，已验证可读。
- Endpoint-ordering archive：28 个逐图结果文件 + summary/seed summary/DOD vectors/exclusions/consistency warnings，已验证可读，约 730 KB。

---

# 三、RTMPose-s：为什么选、如何适配、preflight/smoke 分别证明了什么

## 3.1 为什么由 YOLO-Pose 改为 RTMPose-s

导师指出：YOLO-Pose 会联合预测 bounding box 与 keypoints，而胎儿数据只有两个 endpoint，没有有意义的 object box；使用全图框会引入人为的 box target 和无关 box loss。RTMPose 是 top-down landmark model，bbox/region 只定义输入区域，不由模型预测，因此可把整张图作为固定输入区域，并直接学习两个 endpoint。

本项目采用 RTMPose-s/CSPNeXt-s。实际构建后的参数量不是照抄官方 17 点、256×192 配置的数字，而是现场记录：

| 部分 | 参数量 |
|---|---:|
| CSPNeXt-s backbone | 4,378,320 |
| 两关键点 RTMCC head | 1,066,756 |
| 总计 | **5,445,076** |

## 3.2 正式 canary 的锁定配置

| 类别 | 设置 |
|---|---|
| Canary | UCL BPD，seed 42；导师查看初始结果前不启动其余 runs |
| 模型 | RTMPose-s architecture：CSPNeXt-s + 新初始化两关键点 RTMCCHead |
| 预训练 | 只加载官方 CSPNeXt-s backbone；head 从头训练 |
| 输入 | 整张原图直接非等比例 resize 到 `512×512` |
| 几何公式 | `(x+0.5)×512/W−0.5`，推理用精确逆变换；不使用默认 padded TopdownAffine |
| Feature map | `16×16`（512/32） |
| SimCC | split ratio 2.0；x/y 各 1024 bins；provisional `sigma=(8,8)` |
| Train/val | released UCL Train 共 110；按 EoMT 的 subject grouping 从 Train 内部分成 100 train / 10 internal-val |
| Test | 正式训练与 checkpoint 完成后才转换/读取；不参与 validation 或 checkpoint selection |
| Batch / epoch | batch 16；formal canary 200 epochs，需根据 canary convergence 再判断是否合适 |
| Optimizer | AdamW；LR=`4e-3×16/1024=6.25e-5`；norm/bias zero decay；gradient clipping max norm 35 |
| Scheduler | 最多 5 epochs short warmup（BPD 为 35 iterations）；cosine 从 epoch 100 开始；两者不重叠 |
| Augmentation | flip p=0.5；rotation ±30° p=0.6；scale 0.75–1.25；无 translation；移除 RandomHalfBody；保留与 EoMT 对齐的 color jitter |
| Checkpoint | final/last 为主，不用 Test 选 checkpoint |
| Evaluation | fixed-channel NME；保存逐图原图空间坐标、GT、NME、配置与 provenance |
| Endpoint rule | 当前代码按 HRNet frozen DOD 实现，但正式 canary 前等待导师最终确认 |

必须披露：这是“RTMPose-s architecture under the project's common fetal protocol”，不是官方 COCO RTMPose-s recipe 的逐项 reproduction。官方 EMA hook、原生 human augmentation、stage-2 pipeline switch 与 420 epochs 没有直接照搬。

## 3.3 环境与预训练来源

- MMPose `v1.3.2`, commit `5408bc76f5b848cf925a0d1857899011d8c5b497`；权威 recipe 路径锁定为 `projects/rtmpose/...`。
- PyTorch `2.1.0+cu121`、torchvision `0.16.0+cu121`、MMCV `2.1.0`、MMEngine `0.10.7`、MMDetection `3.2.0`。
- NumPy `1.26.4`、OpenCV-headless `4.10.0.84`；`setuptools=80.9.0`。
- 预训练权重：`cspnext-s_udp-aic-coco_210e-256x192-92f5a029_20230130.pth`。
- SHA-256：`aa7d9335bf422ad02a803e36f357dfc6abb807eca42d79e8b3b6e7c5bd1f446b`。
- Provenance audit 验证了 242 个 backbone state entries：无 missing、unexpected 或 value mismatch；实际加载文件与被哈希文件是同一个本地 checkpoint。

## 3.4 Live preflight 做了什么、能证明什么

最终 live preflight 在真实锁定环境通过六类 gate：

1. **配置真实加载测试**：证明 MMEngine 不是把本地 imports 解析成未执行的 LazyObject；custom transforms/hook 能从 registry 构建。
2. **非正方形图像几何 + SimCC codec round trip**：合成非正方形原图经过显式 512 resize、SimCC encode/decode 和精确逆变换后坐标闭环，验证没有把 MMPose 默认 aspect-ratio padding/inverse 混进来。
3. **BGR/RGB 已知像素测试**：直接调用真实 `LoadImage`，明确验证通道顺序，而不是“看起来合理”。
4. **真实 forward/backward**：用真实 pretrained backbone 和新 head 跑一个 batch，确认 loss 有限，且梯度确实流到 backbone 和 head。
5. **共享 low-level decode**：验证 RTMCCHead 在 MMPose 1.3.2 返回的 `InstanceData` 能正确解析为两个有限坐标；同一函数同时供内部验证和最终 inference 使用。
6. **内部验证 Hook 生命周期**：从 registry 真实构建 hook，调用 `after_train_epoch()`，确认 fixed-channel NME 被写入 MessageHub、模型 training 状态恢复、输出有限。

Preflight 中随机初始化 head 的 Hook NME 为 `122.8730%`。它只说明数值路径没有崩溃或 NaN，**不是模型性能，不得写进结果表**。

Preflight 还提前发现并修复了：MMEngine lazy-config、SimCC codec registry API、RTMCCHead decode 返回类型、Test-as-validation 风险、inference 绕过 preprocessor、checkpoint 字典序选择、权重来源未闭环等问题。这说明 gate 起到了“在正式训练前暴露集成错误”的作用。

## 3.5 独立 1-epoch Runner smoke test 做了什么、能证明什么

Smoke test 使用独立 throwaway work directory，强制 `MAX_EPOCHS=1`，只读 Train-derived 100/10 split，绝不读取 Test。它不是简化版实验，而是对真实 Runner 全链路的工程验证。

通过结果：

- 真实 `Runner.from_cfg()` 构建成功；
- 完成全部 `7/7` training iterations；
- optimizer、scheduler、backward、gradient clipping 均实际执行；
- 最终 aggregate loss 有限：`0.1241`；
- internal fixed-channel hook 对 10 张内部 validation 图完整运行，有限诊断 NME=`69.4972%`；
- 成功写出可读的 `epoch_1.pth`。

这证明的是“环境—数据—模型—优化器—scheduler—hook—checkpoint”真实集成链可运行。`0.1241` 和 `69.4972%` 来自故意只训练一 epoch 的工程 smoke，**不得当作 canary 或论文实验结果**。

Smoke 过程中另行修复了三个真实环境/确定性问题：

- `setuptools 83` 缺少 PyTorch 2.1 所需的 `pkg_resources`，因此固定到 80.9；
- 一 epoch 情形不应生成 `LinearLR(begin=0,end=0)`，smoke-only 配置会完全省略 warmup；
- deterministic algorithms 需要提前设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`，现已在 smoke 和 formal driver 中固定。

## 3.6 当前 RTMPose 状态

代码和环境已经通过 preflight 与 smoke；下一项不是继续无止境做代码审计，而是确定 endpoint convention。导师确认后即可启动 **formal UCL-BPD seed-42、200-epoch canary**，完成后先汇报 convergence、最终 fixed-channel NME、逐图预测、overlay 和 provenance，再决定是否运行剩余 seeds/tasks。

---

# 四、Endpoint ordering：问题、历史、真实数据分析与导师需要确认的事项

## 4.1 两种规则实际上是什么

### EoMT：逐图 x-coordinate sorting

- 每张图独立按 endpoint 的 x 坐标升序排列；相同 x 时以 y 作为 tie-break；
- 约定 channel 0 为“更左”的点，channel 1 为“更右”的点；
- 在数据增强后重新计算；
- 优点是简单、对水平直径直观；缺点是直径接近竖直时，微小定位扰动可能改变左右次序。

### HRNet：training-derived frozen DOD

- 只从 training partition 估计一个 task-specific direction vector；
- 每张图把两点投影到该 frozen direction 上决定 channel 顺序；
- validation/Test 不重新估计方向，因此不会用 Test GT 选择规则；
- released 实现的 flip 行为使用 static、未变换的 `d_vect`，这是代码审计确认的真实实现。

RTMPose 当前 adapter 复刻 HRNet 的 frozen DOD，因为导师要求与已有方法统一时，最初优先对齐了 baseline 的 channel convention。但 EoMT 的历史模型已经是在 x-sort targets 下训练的，所以三者不能在不重训的情况下同时拥有完全相同的训练标签定义。

## 4.2 为什么我记得以前已经讨论过 DOD/x-sort——确实讨论过

以前项目中确实多次出现过这些词，但当时存在一个术语混用：

1. 项目早期设计文档把“按 x 升序”称为 **DOD sort**，写的是 `channel 0 = left endpoint`。当时的判断是：x-sort 足以处理 horizontal flip 后左右端点交换，完整 training-set DOD 可作为以后消融。因此早期所谓“DOD”其实大多是 x-sort，不是 HRNet 的 learned direction vector。
2. TTA 阶段已经发现 near-vertical diameter 在 flip 后有 endpoint-channel ambiguity，因此没有保留 flip TTA。这是第一次明确看到 x-sort 对近竖直直径不稳定。
3. Multicentre ensemble 阶段又使用了“DOD x-sort”或“DOD-final”字样：每个模型 heatmap 先做 channel alignment，ensemble decode 后再次按 x 排序，并检查 GT 已 x-sort。这里仍然是 x-coordinate canonicalisation，不是 frozen-vector DOD。
4. 当时之所以继续保留 x-sort，是因为：
   - EoMT 所有历史消融和已完成模型都按 x-sort 训练/评估；
   - 同协议内的相对消融仍可解释；
   - 临近截止期不适合仅因诊断发现就重训所有 EoMT；
   - 当时主要把大 fixed-channel gap 视为 near-vertical correspondence sensitivity，并未把 EoMT 与 HRNet 的真实训练规则放到同一原图坐标系做系统重评分。
5. 这次为 RTMPose 落实“三方法同一 ordering”时，才逐行核实 HRNet 代码并确认：HRNet 的 DOD 和 EoMT 的 x-sort 是数学上不同的规则。随后用已有逐图坐标做了共同外部重评分，问题才被定量闭环。

因此，这不是“以前知道却漏掉”，而是：以前知道 x-sort 有 near-vertical 稳定性问题，但把若干实现都笼统叫 DOD；这次才严格区分 **x-sort canonicalisation** 与 **training-derived frozen-vector DOD**。

## 4.3 本次重评分如何保证有效

- 共尝试 30 cells（2 datasets × 5 tasks × HRNet/EoMT-DINOv2/EoMT-DINOv3）；成功 28/30。
- 仅缺 UCL-BPD 的两个 EoMT cells，因为对应旧 checkpoint/逐图坐标文件已不存在；HRNet UCL-BPD 可评分。
- 每个文件首先在自身 native 坐标和 native convention 下复算 NME，全部在容差内复现 stored NME。
- HRNet 坐标本来在原图空间；EoMT dump 先恢复 pixel-centre heatmap chain 的 `3.5 px` 常量 offset，再按每张真实图像的 W/H 精确逆变换回原图空间。
- 跨方法 GT 比较允许两个物理点的 channel 全局互换，只比较最接近的配对；所有 common GT 最大误差约 `0.0001 px`，warning 文件为空。
- DOD vector 只来自 training data，未用 Test GT 重新拟合。
- 对同一逐图预测分别计算 native、统一 x-sort、统一 DOD；x-sort−DOD 的 CI 用逐图 bootstrap。

### 4.3.1 “Unified x-sort / Unified DOD”到底如何计算

这里的 `unified` **不是一种新的排序数学公式**。x-sort 仍然是普通 x-sort，DOD 仍然是 HRNet 的 frozen-direction projection。区别在于：所有方法的坐标先进入同一个原图坐标系，然后对 **prediction 和 GT 两边同时、分别重新排序**，最后才计算相同的 fixed-channel NME。

对每个 dataset、task、method、seed、test image，设恢复到原图空间的两个预测点为

```text
P = {p0, p1}
```

两个真实端点为

```text
G = {g0, g1}
```

#### Unified x-sort

分别对预测和 GT 按 `(x, y)` 字典序排序：

```text
(p_x0, p_x1) = sort(P, key=(x, y))
(g_x0, g_x1) = sort(G, key=(x, y))
```

即 x 较小的端点为 channel 0、x 较大的为 channel 1；若 x 完全相同，再以 y 打破 tie。然后计算：

```text
NME_xsort =
  ( ||p_x0-g_x0||₂ + ||p_x1-g_x1||₂ )
  / ( 2 × ||g_x0-g_x1||₂ )
```

#### Unified DOD

每个 `(dataset, task)` 都有一个只从其 Train partition 得到并冻结的方向原型：

```text
d_vect = (d0, d1)
v = d1 - d0
```

注意：`unified` 并不表示 UCL、Multicentre 和五个任务共用一根方向向量；而是说在同一个 dataset/task cell 内，EoMT、HRNet、RTMPose 和所有 seeds 共用该 cell 的同一个 training-derived vector。

对一个点 `q=(x,y)` 的投影为：

```text
projection(q) = dot(q, v) / (||v||₂ + ε)
```

预测的两个点按 projection 从小到大排序，GT 的两个点也独立做相同排序：

```text
(p_d0, p_d1) = DOD_sort(P, frozen d_vect)
(g_d0, g_d1) = DOD_sort(G, frozen d_vect)
```

若两个 projection 完全相等，则沿用输入顺序，与 HRNet 源码的 `proj0 <= proj1` tie rule 一致。最后计算：

```text
NME_DOD =
  ( ||p_d0-g_d0||₂ + ||p_d1-g_d1||₂ )
  / ( 2 × ||g_d0-g_d1||₂ )
```

因为重新排序只改变点的 channel 编号，不改变两个 GT 物理点，所以分母的 inter-endpoint distance 在两种规则下相同；改变的是预测点与 GT 点的 fixed-channel 配对。

### 4.3.2 它和训练/原生评估里的“常规 x-sort / DOD”有什么区别

| 项目 | 常规 EoMT x-sort | 常规 HRNet DOD | Unified external x-sort/DOD |
|---|---|---|---|
| 发生阶段 | 数据加载与增强后，为训练 GT heatmap 指定 channel | 数据加载时用 Train-derived frozen vector 为 GT 指定 channel | 训练完成后的 retrospective evaluation |
| 坐标空间 | EoMT 的 resized/augmented `512×512` 空间 | HRNet 原图/affine pipeline 所使用的坐标逻辑 | 所有方法统一恢复到真实原图像素空间 |
| GT 是否排序 | 是 | 是 | 是，按所选共同规则重新排序 |
| Prediction 是否显式重新排序 | 原生单模型评估通常按网络输出 channel 原样比较，假定网络学会 GT channel identity | 原生评估同样主要依赖模型输出 channel 已遵循 DOD supervision | **是**；prediction 两个坐标也按同一规则独立重新排序 |
| 是否改变训练标签 | 是，定义 EoMT supervision | 是，定义 HRNet supervision | 否；只改变 test-time external pairing |
| 跨方法用途 | EoMT 内部协议 | HRNet 内部协议 | 让所有已有预测接受同一个外部 correspondence rule |

因此，下面三种数值回答的问题不同：

- **Native**：这个已训练模型按自己的历史 channel 输出与历史 evaluator，原本报告了多少 NME？
- **Unified x-sort**：如果不重训，只把所有方法已有预测和 GT 在原图空间都重新按 x 排序，NME 是多少？
- **Unified DOD**：如果不重训，只把所有方法已有预测和 GT 都按共同 frozen DOD 排序，NME 是多少？

这解释了为什么：

- EoMT 的 `native` 不一定等于 `unified x-sort`。EoMT 的 GT supervision 虽然由 x-sort 产生，但 native single-model evaluator通常把预测 channel 原样使用；unified x-sort 额外对预测坐标重新排序，可以修正模型输出 channel crossover。
- HRNet 的 `native` 通常接近但不一定逐位等于 `unified DOD`。HRNet 训练 channel 由 DOD 定义，但 unified DOD 会再次对实际预测坐标做 projection re-sort，而 native 输出主要假定网络已经保持该通道身份。
- `Unified DOD < Unified x-sort` 并不表示移动了预测点或提高了几何定位；两个预测坐标完全没变，只是与两个 GT endpoint 的 channel pairing 改变了。

### 4.3.3 坐标统一与五-seed汇总顺序

在重新排序以前，坐标经过以下处理：

1. HRNet per-image 坐标已经处于原图空间，直接读取；
2. EoMT dump 先逆转其 `512/64` pixel-centre heatmap encode 与 dump 中 naive scale-back 组合产生的 `3.5 px` 常量偏移；
3. 再使用每张真实图像的宽、高和精确 pixel-centre inverse，把 EoMT 的 `512×512` 坐标恢复到原图空间；
4. 完成 native-NME reproduction sanity check与跨方法 GT 物理坐标一致性检查；
5. 对同一原图空间坐标分别应用 unified x-sort 和 unified DOD；
6. 每个 seed 先在全部 test images 上得到 mean NME，最后对五个 seed means 计算表中的 `mean ± seed-level sample SD`；
7. `x-sort − DOD` 的 bootstrap CI 则先对同一图像的五-seed NME 求平均，再以图像为单位 bootstrap，避免把同一张图的五次预测当成五个独立样本。

### 4.3.4 一个直观例子

假设某张近竖直 BPD 图像的真实上端点和下端点 x 坐标非常接近。预测只有很小的横向误差，却使预测的“上端点”比“下端点”稍微更靠右：

```text
GT：   upper.x < lower.x     → x-sort: [upper, lower]
Pred： upper.x > lower.x     → x-sort: [lower, upper]
```

此时 unified x-sort 会把 prediction 和 GT 排成相反的物理对应，fixed-channel error 变大。若 frozen DOD 的方向主要沿着该直径的上下轴，projection 排序仍可能得到：

```text
GT：   [upper, lower]
Pred： [upper, lower]
```

因此 unified DOD NME 较低。这里没有改变 heatmap、模型或预测位置，只是 DOD 在部分竖直任务上提供了比左右 x 坐标更稳定的 endpoint correspondence。

相反，OFD/APAD 出现 `100%` GT disagreement 时，往往只是两种规则对所有图做固定的全局反转：

```text
x-sort: [A, B]
DOD:    [B, A]
```

只要 prediction 和 GT 两边同时反转，配对仍是 `A↔A, B↔B`，所以 unified x-sort 与 unified DOD 的 NME 可以完全相同。故“GT disagreement rate 很高”本身不等于“指标一定变化很大”；关键是全局一致反转，还是 BPD/TAD 那种随图像发生的部分分歧。

## 4.4 GT 层面：两种规则在多少图像上不同

| Dataset | BPD | OFD | APAD | TAD | FL |
|---|---:|---:|---:|---:|---:|
| UCL | 46.94% | 100% | 0% | 38.89% | 0% |
| Multicentre | 46.53% | 100% | 100% | 42.24% | 0% |

解释要点：

- `100% disagreement` 不必然造成 NME 差异。它可能只是两条规则对所有图都做一致的全局 channel reversal；只要 prediction 与 GT 同时按同一规则重排，NME 可以完全相同。
- 真正危险的是 BPD/TAD 这种约 39–47% 的 **部分分歧**：规则并非全局固定反转，而是随图像方向变化，因而会实质改变 fixed-channel 配对和报告值。

## 4.5 UCL：统一外部重评分结果

数值为五-seed mean ± sample SD；最后一列是 `x-sort − DOD` 的逐图 bootstrap 95% CI。

| Task | Method | n | Native | Unified x-sort | Unified DOD | x-sort−DOD（pp, 95% CI） |
|---|---|---:|---:|---:|---:|---:|
| BPD | HRNet | 49 | 5.57±0.48 | 12.97±1.49 | 5.57±0.48 | +7.40 [2.19, 13.96] |
| OFD | HRNet | 49 | 4.72±0.49 | 4.72±0.49 | 4.72±0.49 | 0.00 [0.00, 0.00] |
| OFD | EoMT-DINOv2 | 49 | 4.31±0.68 | 3.96±0.60 | 3.96±0.60 | 0.00 |
| OFD | EoMT-DINOv3 | 49 | 4.09±0.12 | 3.75±0.09 | 3.75±0.09 | 0.00 |
| APAD | HRNet | 36 | 7.36±1.35 | 7.33±1.30 | 7.33±1.30 | 0.00 |
| APAD | EoMT-DINOv2 | 36 | 7.53±1.41 | 6.50±1.15 | 6.50±1.15 | 0.00 |
| APAD | EoMT-DINOv3 | 36 | 6.09±0.84 | 5.34±0.66 | 5.34±0.66 | 0.00 |
| TAD | HRNet | 36 | 6.90±0.98 | 13.65±3.06 | 6.90±0.98 | +6.75 [1.34, 14.03] |
| TAD | EoMT-DINOv2 | 36 | 12.61±2.83 | 12.31±2.16 | 9.66±1.39 | +2.65 [0.00, 6.70] |
| TAD | EoMT-DINOv3 | 36 | 6.44±0.89 | 9.20±0.87 | 6.63±0.87 | +2.57 [-0.47, 8.18] |
| FL | HRNet | 39 | 1.99±0.41 | 1.99±0.41 | 1.99±0.41 | 0.00 |
| FL | EoMT-DINOv2 | 39 | 1.83±0.11 | 1.61±0.11 | 1.61±0.11 | 0.00 |
| FL | EoMT-DINOv3 | 39 | 1.73±0.16 | 1.53±0.15 | 1.53±0.15 | 0.00 |

UCL-BPD 的 EoMT 两行无法恢复，因此不能据此给出 UCL-BPD 三方法统一 DOD 的完整新主表。

## 4.6 Multicentre：统一外部重评分结果

| Task | Method | n | Native | Unified x-sort | Unified DOD | x-sort−DOD（pp, 95% CI） |
|---|---|---:|---:|---:|---:|---:|
| BPD | HRNet | 1180 | 4.81±0.18 | 15.82±0.86 | 4.80±0.19 | +11.02 [9.53, 12.54] |
| BPD | EoMT-DINOv2 | 1191 | 20.33±1.08 | 23.78±0.67 | 7.16±0.15 | +16.62 [15.00, 18.26] |
| BPD | EoMT-DINOv3 | 1191 | 19.00±0.90 | 20.50±1.00 | 6.31±0.20 | +14.18 [12.63, 15.77] |
| OFD | HRNet | 1189 | 4.90±0.17 | 4.87±0.16 | 4.87±0.16 | -0.00 [-0.00, 0.00] |
| OFD | EoMT-DINOv2 | 1191 | 7.34±0.45 | 6.32±0.44 | 6.33±0.45 | -0.01 [-0.04, 0.00] |
| OFD | EoMT-DINOv3 | 1191 | 6.90±0.52 | 5.98±0.62 | 5.99±0.63 | -0.02 [-0.05, 0.00] |
| APAD | HRNet | 161 | 8.89±0.19 | 8.93±0.21 | 8.93±0.21 | +0.00 |
| APAD | EoMT-DINOv2 | 161 | 9.17±0.32 | 6.99±0.23 | 6.99±0.23 | 0.00 |
| APAD | EoMT-DINOv3 | 161 | 9.26±0.24 | 7.11±0.21 | 7.11±0.21 | 0.00 |
| TAD | HRNet | 161 | 8.78±0.33 | 32.50±2.98 | 8.79±0.34 | +23.72 [18.59, 29.11] |
| TAD | EoMT-DINOv2 | 161 | 29.26±1.33 | 29.16±1.92 | 8.89±0.92 | +20.26 [15.34, 25.38] |
| TAD | EoMT-DINOv3 | 161 | 27.53±0.57 | 27.94±1.45 | 8.81±0.59 | +19.13 [14.39, 24.19] |
| FL | HRNet | 362 | 2.94±0.32 | 2.96±0.31 | 2.96±0.32 | +0.00 [-0.00, 0.00] |
| FL | EoMT-DINOv2 | 362 | 3.29±0.15 | 2.79±0.11 | 2.79±0.11 | 0.00 |
| FL | EoMT-DINOv3 | 362 | 3.16±0.18 | 2.72±0.13 | 2.72±0.13 | 0.00 |

## 4.7 这张表能证明什么、不能证明什么

能证明：

- fixed-channel 结果对 channel canonicalisation 的定义高度敏感，尤其是部分竖直/斜向的 BPD 和 TAD；
- 在已有预测上，training-derived frozen DOD 是 BPD/TAD 更稳定的外部对应规则；
- EoMT 原 fixed-channel 表中 BPD/TAD 的高值不能简单解释为纯几何定位失败；其中包含很大的 channel-assignment sensitivity；
- 旧的“EoMT 在 Multicentre BPD/TAD 比 HRNet 差约 14–20 pp”结论会随着统一外部 DOD 重评分大幅缩小，方法排名/差距必须重新谨慎表述。

不能证明：

- 不能把 unified-DOD EoMT 值称为“DOD 训练后的 EoMT 性能”，因为 EoMT 训练 targets 仍是 x-sort；
- 不能证明若重新用 DOD labels 训练，模型必然得到同样结果；
- 不能用重评分代替所有方法在统一训练标签 convention 下的完整重训；
- 不能根据这些 Test 重评分结果反过来挑 checkpoint 或调超参数。

## 4.8 建议请导师确认的三个具体问题

### 决策 1：RTMPose 正式训练使用哪一种 endpoint convention？

建议提问：

> 我已经对现有 HRNet 和 EoMT 逐图预测做了统一外部重评分。BPD/TAD 上 x-sort 与 training-derived DOD 的部分分歧会显著改变 fixed-channel NME，而 OFD/APAD 的 100% 分歧主要是全局 channel reversal、实际 NME 几乎不变。当前 RTMPose adapter 按 HRNet 的 frozen DOD 实现。您是否同意正式 UCL-BPD seed-42 canary 继续使用 DOD？

我的建议：**RTMPose 使用 frozen DOD**。理由是 DOD 只由 Train 估计、Test 不参与，而且在部分竖直任务上显著更稳定。

### 决策 2：论文最终 EoMT–HRNet 主表如何处理？

可给导师两个清晰方案：

- **方案 A（当前推荐）**：不重训现有 EoMT；使用每个 dataset/task 对应 Train partition 估计并冻结的同一 DOD，对所有可获得的 EoMT、HRNet 和后续 RTMPose predictions 与 GT 做共同外部 canonicalisation，并以统一 DOD 后的 fixed-channel NME 作为最终跨方法主表。历史原生结果另放 sensitivity/审计表。必须明确：这只统一 evaluation correspondence，没有追溯统一历史 training supervision。
- **方案 B（最严格但代价极高）**：EoMT、HRNet、RTMPose 全部按同一 endpoint target convention 重训。至少涉及 EoMT 50 runs + HRNet 50 runs + RTMPose 50 runs，且会推翻已冻结的大量结果，不适合当前论文时间线，除非导师明确认为这是 submission 必需条件。

需要特别询问：导师此前说“same endpoint-ordering conventions across RTMPose, HRNet and EoMT”，在看到这次真实数据后，是否接受 **训练规则历史不完全一致但使用共同外部 sensitivity analysis 透明披露**，还是要求全部重训。

#### 推荐方案 A 的准确协议

```text
EoMT：    x-sort supervision → common frozen-DOD external evaluation
HRNet：   DOD supervision    → common frozen-DOD external evaluation
RTMPose： DOD supervision    → common frozen-DOD external evaluation
```

每个 dataset/task 的 DOD 必须只由其对应 Train partition 估计一次，并在所有方法、所有 seeds、internal validation/Test 上冻结复用；不得利用 Test GT 重新估计方向。最终主表应明确命名为：

> Common training-set-derived DOD evaluation: final single-model fixed-channel NME, mean ± seed-level SD.

论文中可以写：

> All available predictions were evaluated using a common endpoint canonicalisation based on a training-set-derived frozen direction vector. The existing EoMT checkpoints were not retrained: their historical supervision used per-image x-coordinate sorting, whereas HRNet and RTMPose used DOD-based endpoint ordering.

不能写：

> All methods were trained and evaluated using identical endpoint ordering.

#### 采用方案 A 前必须关闭的四个边界

1. **导师书面或会议明确同意**：确认“共同外部 DOD evaluation，但不重训历史 EoMT”满足最终论文比较要求。
2. **UCL-BPD 缺口**：UCL-BPD 的 EoMT checkpoints 和逐图坐标已缺失，因此目前无法统一 DOD 重评分。不能把其历史 x-sort aggregate 混入 unified-DOD 主表。优先询问导师是否接受该 cell 标为 unavailable；若他要求完整，才有限重跑 UCL-BPD，而不是重跑全部 EoMT。
3. **Ensemble 不能直接复用**：当前总表里的 EoMT ensemble 是历史原生 canonicalisation 下的点估计，不能自动当作 unified-DOD ensemble，也不能从五个单模型 unified-DOD 均值反推。除非已有 ensemble 逐图坐标，或恢复 checkpoints 后重新推理，否则 unified-DOD 主表只报告五个单模型 `mean±SD`；历史 ensemble 留在 EoMT 内部补充表并标明原生 ordering。
4. **重新生成正式总表**：主表必须全部使用 common DOD：EoMT-DINOv2、EoMT-DINOv3、HRNet 和 RTMPose。原生 fixed-channel 与 common-DOD 数字不得在同一排名栏混排。

#### 为什么当前优先推荐方案 A

- 现有 28/30 cells 已完成真实逐图重评分，native sanity、原图坐标恢复与跨方法 GT 一致性检查均通过；
- BPD/TAD 的大差距已被证明对 external correspondence rule 高度敏感，继续用不同原生规则直接排名会混入 evaluator confounding；
- common DOD 至少统一了最终评价阶段的物理端点对应定义；
- 全量重训会新增约 100 次 EoMT/HRNet 训练，并推翻大量已完成分析，而当前证据没有表明必须这样做；
- 该方案的剩余限制可以通过透明披露准确表达，不需要把 retrospective re-scoring 包装成共同训练协议。

### 决策 3：能否在导师确认 DOD 后立即开始 formal canary？

> Preflight 和独立 1-epoch smoke 已通过；Test 从未用于内部 validation；正式 UCL-BPD seed-42 canary 尚未启动。若您确认 RTMPose 使用 frozen DOD，我计划会后立即启动 200-epoch canary，并在开始其余 runs 前先把 convergence、fixed-channel NME、逐图预测与 overlays 发给您检查。

---

# 五、会议时建议的口头汇报顺序

1. **先纠正概念**：“五项测量一直是五个独立模型，我之前这部分理解已经确认正确。”
2. **展示 Multicentre 固定口径主表**：先说 HRNet/EoMT 都已跑完，再指出 BPD/TAD 差距异常大。
3. **展示 endpoint 统一重评分表**：重点只展示 disagreement rate，以及 BPD/TAD 的 x-sort−DOD 差值；强调这是诊断，不是重训。
4. **解释历史原因**：以前所谓 DOD 多数其实是 x-sort；近竖直不稳定早已在 TTA/ensemble 中出现，但这次才把 HRNet frozen-vector DOD 与 EoMT x-sort 严格分开并量化。
5. **说明 RTMPose 已准备到什么程度**：preflight 证明坐标/codec/梯度/Hook，1-epoch smoke 证明真实 Runner/optimizer/scheduler/checkpoint；两个数值都不是实验结果。
6. **最后只要求导师做决策**：RTMPose 是否用 DOD；历史 EoMT/HRNet 是否接受透明披露 + sensitivity analysis 而不全部重训；确认后立即启动 seed-42 canary。

---

# 六、可以直接对导师说的简短总结

> Multicentre 的 EoMT-DINOv2、EoMT-DINOv3 和 HRNet fixed-channel 比较已经完成。按照当前原生 channel 定义，HRNet 在 BPD、OFD 和 TAD 上明显更低，APAD 和 FL 的 paired CI 跨零。但在实现 RTMPose 时，我确认 EoMT 实际按逐图 x 坐标排序 endpoints，而 HRNet 使用 training-derived frozen direction vector；以前项目文档曾把 x-sort 也简称为 DOD，因此这个差异没有被清楚区分。  
> 我现在已经在原图坐标空间对 28/30 个已有结果 cell 做了两种共同规则的外部重评分。所有 native NME 均成功复现，跨方法 GT 最大差异约 0.0001 像素。OFD/APAD 的规则差异主要是全局通道反转，NME 基本不变；但 BPD/TAD 是约 39–47% 的部分分歧，统一 DOD 会显著降低两种方法的 NME，说明原表的大差距包含 endpoint-channel convention sensitivity。这个分析不是重训练，所以我不会把它当作 DOD-trained EoMT 的性能。  
> RTMPose 的 512 非等比例几何、SimCC round trip、预训练权重、真实 forward/backward、内部 fixed-channel hook 和完整一 epoch Runner smoke 都已经通过，正式 Test 尚未用于训练或选 checkpoint。现在只需要您确认：RTMPose 是否按更稳定的 training-derived DOD 训练，以及论文是否可以保留历史训练、增加统一外部 sensitivity analysis 并透明披露，而不重新训练全部 EoMT/HRNet。确认后我会立即开始 UCL-BPD seed-42 formal canary，并先把结果发给您再继续其余 runs。
