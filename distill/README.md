# Distillation data pipeline (POC)

Teacher: classifier.dev (free, no key). Default lane is TypeSafe Jev
(jev-1.13.0); `model: "laya"` opts into the open-source Laya trial.

- `seeds_zh.jsonl`: seed states (Chinese-first support tickets, 20 zh + 4 en)
- `questions.json`: typed question templates (choice / score / probability)
- `generate.py`: bulk-classifies all seeds per question, writes JSONL rows
  with state, question, labels, answer, per-label `scores` (soft targets),
  confidence, latency, teacher id
- `data/`: generated datasets (committed as evidence + eval fixtures)

Quota (trial tier): fast lane 60 q/min, 2000/day; bulk 1000 inputs/call,
1000 q/min, 20000/day. A question counts per input per call.

## Findings (2026-09-21)

1. `instructions` is mandatory for binary lanes. With a bare question
   ("是否要求退款？"), P(yes) barely discriminates (praise -> 0.74). With
   operationalized definitions of both labels, discrimination is perfect
   (0.00-1.00) and mean confidence jumps 0.53 -> 0.998.
2. Choice lanes are strong out of the box (department mean conf 0.934).
3. Score lanes benefit from level definitions in instructions
   (urgency 0.616 -> 0.743).
4. Soft `scores` are returned per label and are the supervision signal for
   KL distillation; argmax-only labels would lose calibration info.
5. Teacher quality is domain-dependent (independent evals found ~62.6%
   accuracy on phishing) - keep a human spot-check set before trusting
   any lane at scale.
