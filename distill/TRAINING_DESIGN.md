# Student distillation design (mapped to our pipeline)

教师：classifier.dev 的 jev-1.13.0（黑盒 API，拿不到 logits，但 `scores` 存了完整
概率分布）。学生：Qwen3.5-4B（部署定死 4B，用户拍板）+ 读选项 logits 的 decision
readout（SemIf/kev 式，走 mistral.rs `return_raw_logits` 路径，零生成 token）。

## 训练阶梯（2026-09-21 修订：默认走向动底模）

既然部署锁定 4B，LoRA 只是第一级数据验证手段，不是终态：

1. 第一级 LoRA 蒸馏：验证 17k 行数据质量（金标 + holdout 指标）。成本 ~¥2/时
2. 第二级 动底模（默认路径）：LoRA merge 进底模 → 全参 FT（BF16 + 8-bit AdamW
   + 梯度检查点，~24-32GB 显存，17k 样本 1-2h）或直接全参
3. 部署产物：微调后的权重用 mistralrs ISQ 自己量化出 Q4 单文件（自主量化 +
   UQFF），APK 单文件加载，FFI 无需 adapter 参数
4. 灾难性遗忘不影响本部署形态（只做单前向决策读出，不做生成），故全参无顾虑

## 与标准蒸馏框架的映射

| 通用概念 | 我们的实现 |
| --- | --- |
| 蒸馏类型 | 响应式（response-based），且教师是黑盒 → 离线蒸馏：先收数据（generate.py），后训练 |
| 教师"logits" | 存储的概率分布重加温：`q_i(T) = p_i^(1/T) / Σ_j p_j^(1/T)`。T=1 时回到原始校准概率；T>1 时暗知识显影。因为存了完整分布，T 可以训练时任意调，不需要重调 API |
| 学生 logits | 单次前向后，在最后位置读候选标签首 token 的 logits，对候选集合 softmax。设备端 gather（侵入式优化二期再做） |
| 损失 | `α · T² · KL(q_T ‖ softmax(z_s/T)) + (1-α) · CE(z_s, y)`，T² 梯度补偿，α 默认 0.7 |
| 混合硬/软标签 | confidence ≥ 0.9 的样本（质检里 department 21/24）走硬标签为主；低置信样本加大 α（软目标权重），置信度作为样本权重 |
| 教师错误传播 | 已实锤（钓鱼任务独立评测 62.6%）。对策：人工抽查集（金标）单独成 eval 集；训练集按置信度加权；金标集上学生超过教师的样本直接进金标 |
| 容量差距 | 学生首选 4B（与教师任务智能同级）。若 4B 拟合不动 → 助教路线：先用 14B 教师数据再降。学生最小 0.8B 需实验验证 |
| 评估 | 不止准确率：ECE 校准误差（教师卖点就是校准）、真机 prefill 延迟、内存。金标集 + held-out 保留集分离 |

## 候选 token 化的关键细节

choice 的标签（如 billing/technical）token 化后取首 token id，要求候选间首 token
互不冲突（训练脚本里 assert，冲突时改用全序列 logprob 和做分数，长度归一）。
中文标签同理。`questions.json` 里的标签就是候选集合，与 mistralrs-decide 的
runtime 接口同构——训练时的候选集合 = 部署时的候选集合。

## 训练形态

云 GPU（Colab/AutoDL）上跑 `train_student.py`（本机只写码不训练）：
- QLoRA/LoRA + 新增 readout（或直接用原始 logits 读出，kev 路线两者都支持）
- 数据：distill/data/teacher-*.jsonl（含 scores 软目标）
- 产出：LoRA adapter + readout 权重 → 转回 mistral.rs 可加载格式（GGUF/UQFF
  或 safetensors），本地 APK 侧仍走 `return_raw_logits` 读出，推理引擎零改动

## 为什么这条路线成立

教师（70ms 云端 API）→ 学生（同接口，端侧 ~0 网络延迟、无配额、隐私数据不出
设备）。学生接口与教师完全同构（state + typed questions → 分布 + confidence），
数据管线产出的每一行 JSONL 直接就是一条训练样本。
