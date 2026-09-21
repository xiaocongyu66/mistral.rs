# Decide 层运行时契约（对齐 jev-arena 已实测接口）

来源：NanmiCoder/jev-arena `docs/CONTRACT.md`（2026-09-19 实跑验证过真实 Jev，
非文档臆测）。mistralrs-decide 的 HTTP/FFI schema 与此对齐，保证与 Jev 生态
（arena 工具、awesome-jev prompt 库）drop-in 兼容。

## 请求

```
POST /v1/decisions
{
  "model":  "<可选，默认本地学生>",
  "state":  <string | object | array>,
  "questions": {
    "<id>": { "type": "noul",
              "instructions": "...",
              "criteria": {"true": "...", "false": "..."} },
    "<id>": { "type": "choice",
              "instructions": "...",
              "criteria": {"选项": "说明或 null"} },
    "<id>": { "type": "score",
              "instructions": "...",
              "criteria": ["第0档", "第1档", "..."] }
  }
}
```

## 响应

```
{ "model": "<checkpoint id>",
  "answers": {
    "<id>": { "type": "noul",   "noul": 0.95 },
    "<id>": { "type": "choice", "choice": "mixed",
              "probabilities": {"...": 0.xx}, "confidence": 0.99 },
    "<id>": { "type": "score",  "score": 1.25,
              "legend": {"...": 0.xx}, "probabilities": {"...": 0.xx},
              "confidence": 0.79 }
  },
  "usage": { "input_tokens": 509, "output_tokens": 0 },
  "id": "<gen id>" }
```

## 语义铁律（实测确认）

1. `score` 是**档位下标的期望**（可落在两档之间，如 1.25），不是原始分值
2. 题型命名用 `noul|choice|score`（注意：Vercel AI SDK 层用 `boolean`，勿混淆）
3. `choice` 返回完整 `probabilities` 分布 + `confidence`；`noul` 单值概率
4. 报错形态 `{"error":{"message":"...","code":400}}`

## 第二教师通道（Overflow）

classifier.dev 配额干涸时的备用通道（jev-arena 同款）：

```
POST https://openrouter.ai/api/alpha/decisions     ← 注意没有 /v1
Authorization: Bearer $OPENROUTER_API_KEY
model: "typesafe/jev-1.13"
```

- `usage.cost` 由 OpenRouter 直接返回真实美元费用（勿自算）
- 实测成本约 $0.84 / 1 万题 → 2 万题/天 ≈ $1.7/天（比免费贵，做溢出通道）
- 需要用户提供 OPENROUTER_API_KEY

## 教师质量审计（借他们的口径，本地已复算验证）

jev-arena 万条评论，两组数字**口径不同，勿混**（曾混用，已修正）：

1. **一致率**（两模型互判，分母=双方均成功的 10,000 条）：
   is_relevant 89.39% / sentiment 73.59% / intent 67.85%
   ——一致率不是准确率；无人工标准答案时不能宣布谁更对（本地复算精确对齐 ✓）
2. **准确率**（GPT-6 Astra 全量复核 + 400 条随机复标，AI 参考口径）：
   Jev 94.7/82.9/77.9%，三项全对 62.69%；DeepSeek 96.3/84.5/80.3%，67.26%
   ——分母是**各自判定相关**的评论（Jev 8,204 vs DeepSeek 8,549），比较前先对齐

其他实测：Jev 真实费用 $0.84/万题（供应商返回）；端到端 latency mean ≈ 2.1s
（含宿主摘录 evidence_quote 的开销）；confidence mean 0.732；
情感分歧集中于 B 站（1091 条）。

## 操作硬限（jev.mjs 踩坑注释，MIT 代码照抄）

- state + 问题定义 ≤ **32k token**；单条评论 ≈ **2.0k token**（25 个问题定义是大头）
- 发请求前**本地估算**，超线抛 STATE_TOO_LARGE 让上层二分——别打过去吃 422（白花钱）
- 实用批量 maxBatchSize = 10；429/5xx 重试 3 次；timeout 180s
- `score` 5 档：强烈负面/偏负面/中性/偏正面/强烈正面 → -1 + score*0.5 映射 -1..1
- Jev 不产文本，evidence_quote 由宿主机械摘取并标 evidenceSource:"host"——
  不要把摘录说成模型推理

## 新题型模板（同步他们的词表，进 generate.py）

- sentiment：choice over [positive, negative, neutral, mixed]（四档说明照抄）
- intent：choice over 10 类（praise/complaint/question/suggestion/correction/
  agreement/disagreement/joke/information/other，中文说明照抄）
- sentiment_score：5 档 score（上述 SCORE_TIERS）
- aspects / emotion：多选（multi: true，频率和可超 100%）

其 1 万条评论数据版权归原作者与平台（README 明示不属于 MIT 代码许可），
标签结构可学，内容不入训练集。
