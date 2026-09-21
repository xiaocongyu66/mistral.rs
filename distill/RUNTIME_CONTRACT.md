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

## 教师质量审计（借他们的口径）

jev-arena 用 GPT-6 Astra 对 1 万条评论全量复核 + 400 条随机复标：

| 指标 | Jev 1.13 | DeepSeek Flash |
| --- | ---: | ---: |
| 相关性 / 情感 / 意图 准确率 | 94.7 / 82.9 / 77.9% | 96.3 / 84.5 / 80.3% |
| 三项同时正确 | 62.69% | 67.26% |

结论：教师在评论域三项全对仅 62.7%（与我们钓鱼 62.6% 的独立评测一致）——
双教师 + 金标抽查的既定方针不变；评论/情感/意图类样本应进入金标池优先复核。
其 1 万条评论数据版权归原作者与平台（README 明示不属于 MIT 代码许可），
只作本地参考，不入我们的训练集。
