# 训练资源索引（按我们的相关性过滤）

外部资源清单经评估后入库，只留与我们管线直接相关的，并标注用途。

## 立即可用

| 资源 | 与我们的关系 | 用法 |
| --- | --- | --- |
| **openJev-verdict-2.0**（Heman10x-NGU，MIT） | **151M 编码器学生**（GLiClass+ModernBERT，自报 ECE 0.0144/Brier 0.0636/acc 77.1%）；其 `core/calibration.py` 是严格实现（CE+Brier 复合损失、L-BFGS 温度、10-bin ECE） | ① 移植 calibration.py 到我们的产物分析（predictions.jsonl 离线算 ECE/Brier）② GLiClass 作为 v2 学生架构候选（量化后 ~100MB，原生接口=标签集→logits） |
| **open-jev-typed-decision-engine**（intikhab49） | 150M 三题型引擎，自报 0.697 vs Jev 0.727、2.5 倍校准、T4 30 分钟 | 与 verdict 互为参照的编码器学生实现 |
| **modelscope/easydistill**（阿里，2026-09-18） | 系统一/系统二蒸馏工具包 | 蒸馏管线工具化备选；当前自研脚本已跑通，暂不引入 |
| **腾讯云 TI-ONE GRPO 教程（Qwen3-0.6B 例）** | 底模与我们的完全一致；GRPO≈RLCD 同族，是 Jev 的原训练法，也是 NanoJev roadmap 未做的 RLCD 后训练 | 阶段 5b：SFT 蒸馏完成并验证后，用 GRPO/RLCD 对决策头做校准后训练 |
| **HuggingFace 训练手册**（200+ 页，SmolLM3 3B @384 H100 实录） | 端到端训练经验（哪些有效哪些失败）+ 调试技巧 | 训练决策遇阻时的对照手册；run2 放宽 max_length、多任务混训前先翻它 |
| **阿里云 ROLL SFT 流水线**（Ray+Megatron/FSDP、Sequence Packing） | 我们的决策样本全是短序列，**sequence packing 直接提升训练吞吐** | 训练数据量大后（>10 万题）引入 packing；当前 17.7k 题暂不需要 |
| **DeepSeek 实战资源（训练篇）** | SFT 复现 / GRPO 复现 / 数据蒸馏配套 | 与 TI-ONE 互为参照的 GRPO 实现参考 |
| **阿里云百炼**（API 调优：SFT/CPT/DPO） | 免运维的训练替代通道（付费） | Colab 不够用时的 B 计划；DPO 可用于置信度偏好校准 |

## 学生架构候选对比（当前决策点）

| 架构 | 体积（量化后） | 接口契合 | 状态 |
| --- | --- | --- | --- |
| **NanoJev 0.6B 解码器+头**（当前，run1 训练中） | ~500MB | 首读 logits，需 patch 或自建引擎 | dev acc 83.6%（step 174 best） |
| **GLiClass 151M 编码器**（verdict 路线） | **~100MB** | 原生「文本+标签集→每标签logit」，无需改造 | v2 实验：等 run1 指标出来后对比 |

## 暂不需要

- Azure ML / Vertex AI：托管流程重，我们已有 Colab 遥控体系
- Windows ML ONNX 训练指南：目标平台是 Android，不走 ONNX 训练路径（推理侧 candle）
- Happy-LLM / LLM design patterns：教材向，认知补充
- 团体标准《高质量数据集实施指南》：数据质量控制条目可对照，非操作文档
- YOLOv5 训练指南：域不相关

## 论文笔记（已学习，可直接落配方）

**[RLCR: Beyond Binary Rewards](https://arxiv.org/abs/2507.16806)**（MIT，Damani/Shenfeld/Choshen/Kim/Andreas）
- 问题：RL 用二元奖励只罚错误不罚瞎猜 → 校准退化、幻觉率上升
- 配方：奖励 = 二元正确性 + **Brier 分数**（严格本征评分规则），模型同时输出预测和数值置信度
- **落点**：阶段 5b 的奖励设计直接采用 RLCR（不是裸 GRPO）——我们决策头的 RLCD 后训练 = 正确性 + Brier 联合优化；verdict 仓库 calibration.py 已有 Brier 实现，配合无墙
- 关键句："binary reward functions do not penalize guessing"——正是我们在钓鱼任务看到的教师高置信错误问题的 RL 端解法

**[ModernBERT](https://arxiv.org/abs/2412.13663)**（Warner et al.）
- 2T token 训练、**原生 8192 序列长度**、双向编码器 Pareto 改进、推理速度/显存最优
- **落点**：确认编码器学生路线（GLiClass 底座即 ModernBERT）的两个优势：①原生 8192 上下文（对比我们 0.6B 解码器学生 768 训练窗 + 前缀 K 倍重复）②速度/显存为移动端友好设计；v2 学生实验有了理论背书

## 论文批量学习（11 篇，按命中度排序）

**🎯 直接改我们的数据预处理（三篇）**

1. **[Teacher Calibration in KD](https://arxiv.org/abs/2508.20224)**：教师校准误差与学生性能强相关 → **蒸馏前先量教师 ECE、先校准教师分布再喂学生**。行动：convert 阶段对 fixtures 上的教师分布做温度拟合（我们已有 temperize，缺"先测后用"的顺序）
2. **[LoCa: Logit Calibration](https://arxiv.org/abs/2409.04778)**：发现 **mis-instruction**——教师 logits 与标签冲突时会误导学生；提出校准教师 logits 的方法。行动：对 verdict 摄入数据（有**外部金标** target_id，非教师代理）套用：教师软标签与金标冲突处修正后再入训练——直接治理我们钓鱼 62.6% 的高置信错误传播
3. **[UNDO: Distillation as Optimization](https://arxiv.org/abs/2504.02521)**：one-shot 蒸馏的教师产物与学生需求错配 → 迭代蒸馏、按学生错误定向合成数据。行动：run2 起用 holdout 上学生的错误清单驱动 cron 的种子合成方向

**有道参考（三篇）**

4. **[EasyDistill 论文](https://arxiv.org/abs/2505.20888)**：工具包的正式论文（System 1/2、数据合成+SFT+排序+RL 全家桶）
5. **[BiLD: Bi-directional Logits Difference](https://aclanthology.org/2025.coling-main.78/)**：LLM 蒸馏损失创新（双向 logits 差）
6. **[Encoder-only Cross-Encoders 控制研究](https://arxiv.org/abs/2603.03010)**（2026-03）：**LLM ranker 蒸馏 vs 强 cross-encoder 教师 vs 纯监督**的受控对比——直接回答"该选哪种教师通道"的方法论问题

**索引/工具（五项）**

7. **[KD 综述](https://arxiv.org/abs/2503.12067)**：全景参考
8. **[distillKitPlus](https://github.com/agokrani/distillKitPlus)**：**Pre-Computed Logits**（预生成 logits 省显存——正对我们 2-3GB 约束）、跨 tokenizer 损失（ULD/multi-OT）、LoRA+4bit
9. **[Soft Decision Tree](https://arxiv.org/abs/1711.09784)**（Hinton 经典）：蒸馏到可解释软决策树——"可审计决策"产品叙事的先例，二期可给决策头加树状解释层
10. **[ModernBERT-small-v2](https://huggingface.co/johnnyboycurtis/ModernBERT-small-v2)**：MSE 蒸馏的句子向量模型（编码器路线上限参考）
11. KD 综述与 distillKit 的关系：distillKitPlus ≈ DistillKit + 低算力 PEFT 强化

## 已在用的同类资源（本仓库内）

- `distill/TRAINING_DESIGN.md`：训练契约（含 NanoJev 同步要点）
- `distill/RUNTIME_CONTRACT.md`：decide 层接口契约 + 教师审计口径
- `distill/COLAB.md`：云上训练运行手册
- NanoJev `research/algorithm_training_contract_zh.md`：损失与批处理契约（已吸收）
