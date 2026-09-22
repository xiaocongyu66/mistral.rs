# 模型优化待办（从用户资源清单筛出，按对我们的价值排序）

当前基线：Qwen3-0.6B + decision head，软标签 KL 蒸馏（教师 jev-1.13.0），
37,802 题决策数据，T4/Kaggle 训练。

## P0 — 直接可用于当前管线

**1. Seal 的弃答能力（ACL 2025）**
SFT 阶段加 `[REJ]` token，让模型学会"证据不足就拒答"。我们的 decide 层正需要此能力
（verdict 数据已含 `__insufficient_evidence__` 候选，天然对齐）。
落地：`colab_train.py` 的候选集里显式保留 abstain 候选 + 训练时给它真实概率。

**2. 选择性重加权（PNAS 2026）**
仅重复 5% 训练样本即降 40% 幻觉率，不牺牲准确率。做法是刻意注入轻度校准偏差，
抑制稀有事实的过度泛化。
落地：对 `convert_nanojev.py` 加样本权重字段（按教师分布的熵/置信度分层）。

**3. DASD 的三件套（Distribution-Aligned Sequence Distillation）**
① 温度调度学习 ② 散度感知采样 ③ 混合策略蒸馏——少量数据达 SOTA。
落地：我们已有固定 T=3 网格搜索，升级为"训练中退火 T"；散度感知采样即按
教师-学生 KL 大小动态选样本（我们的 45% 软分布样本可直接用）。

## P1 — 架构级改进（APK 端收益大）

**4. KDA 线性注意力（QINGYI-KDA-0.6B，`Sisyphbaous-DT-Project/open-qingyi`）**
Qwen3-0.6B-Base 的 28 层里 21 层换成 KDA 线性注意力（657.5M 参数）。
线性注意力把推理复杂度从 O(n²) 降到 O(n)，**长上下文的端侧推理速度大幅提升**。
参考其 HF 权重 `shiershuiheseixiliya/qingyi-kda-0.6b` 与 GenDistill 蒸馏配方。
落地：作为我们学生的 v3 架构候选（需 candle 侧实现 KDA kernel，工程量中等）。

**5. DistilQwen 配方（alibaba-pai）**
同规格（0.6B/1.5B）蒸馏成果，Proof-weighted 蒸馏——按样本证明难度加权。
落地：与我们的置信度加权互补，可合并成复合权重。

## P2 — 推理侧（APK 部署后优化）

6. **RE-IAG / GuarantRAG**：推理与证据解耦，仅在内部推理停滞时触发检索
7. **对比解码 / ActLCD**：惩罚"通用但可能错误"的 token，抑制幻觉
8. **DeLask（IEEE 2026）**：幻觉倾向从深层解码器层产生，动态跳过"问题层"
9. **NOVA（ACL 2025）**：内部一致性探测筛除模型"不熟悉"的训练数据

## 已纳入基线（无需再做）

- 教师校准优先（2508.20224）、LoCa 错误修正（2409.04778）、UNDO 迭代蒸馏
- RLCR 校准奖励（2507.16806）→ 阶段 5b
- ModernBERT 编码器路线 → v2 学生候选


## 评估后判定「不需要」（2026-09-21，防重复踩坑）

**语义增强（nlpaug / 回译 / 规则改写）**
它解决的是「数据不足」。我们的瓶颈是**教师配额 2 万/天**，而种子池有 9 万条待标注——
种子过剩、标注稀缺，给过剩库存做增强没有收益。
实测规则化增强（同义替换+实体替换+语气拼接）产出 16,307 条，但样本中英混杂、
拼接生硬（例：`在鸿蒙平板上，The mobile app has been down...，谢谢（订单 4401）`），
直接污染训练集，已全部删除。**结论：不做语义增强，配额优先投给高质量真实种子。**

**搜索增强（Search-R1 / SAIL / s3 / SimpleDeepSearcher）**
训练模型「推理-搜索交替」的能力。我们的 decide 层是**单次前向的分类决策**
（state + 候选 → 概率分布），没有检索循环，架构上不适用。
其中 s3 的「2.4k 数据通过 RL 超过 70 倍数据基线」是数据效率的旁证，
与 LiteCoT 的「10 万短推理 > 80 万长推理」同向，支持我们「质量优先」的路线。

**结论**：资源清单要按「我们的瓶颈是否被它解决」筛选，不按「它是否优秀」筛选。

## 2026-09-22 用户资料批（三批，全部已核验存在性）

### P1：stage 2 数据源（下一轮直接接）
- **Data Turnstile**：API 规范→函数调用数据生成框架；1000+ API、10 万+多轮交互。
  Qwen3-0.6B 微调后 BFCL 75.9%、tau2-bench pass^1 3.5%->24.6%（7x）。
  用途：函数调用=decide 层引擎（任务 #14）的调用层原语数据。
- **OmniThought-0528 + EasyDistill**：36.5 万变长思维链（DeepSeek-R1-0528 蒸馏），
  HF/ModelScope 开源。用途：变长 CoT 与我们的温度退火互补，推理增强源。
- **混合数据配方警告（Opus-3000x 教训）**：纯推理 SFT 使 ARC-Challenge -24.31%
  +模式崩溃。Unsloth 配方（unsloth/OpenMathReasoning-mini + mlabonne/FineTome-100k
  混合）是下一轮 stage 2 数据配比的参考线。我们 stage 1 的 38 源热身正是防这个。

### P2：多模态启动时用（任务 #16）
- **Light-MER**（GAIR-Lab）：隐藏态 Wasserstein 蒸馏，Qwen3-8B 教师->0.6B 学生，
  峰值显存 2.54GB。蒸馏 pipeline 可迁移到我们的多模态任务。
- **Reyes-0.6B**（yujunhuics/Reyes）：SigLIP2-Base + 2 层 MLP + Qwen3-0.6B，
  MMMU 38.7，ModelScope 权重。若做多模态：直接领域 LoRA，不从零训。

### 新教师通道：knox.chat（2026-09-22 实测）
- POST /v1/systemone，state+questions{criteria} 格式，返回 answers.q.probabilities
  软分布；jev-1.13.0 正版在位。实测 128 并发 335/min（429 上限），
  CONC=16 稳定 ~55/min，不限日配额。蒸馏器：distill/distill_knox.py。
  nm2b 25k 过夜烧制中。意图种子 4,900（clinc/BlendX）排队。gold 直通已支持
  （convert_nanojev.py gold-only rows + --objective gold）。
