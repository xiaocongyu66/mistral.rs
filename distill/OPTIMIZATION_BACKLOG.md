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
