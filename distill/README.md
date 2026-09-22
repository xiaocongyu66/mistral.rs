# Apeireth-Decis Training Pipeline

**Apeireth-Decis-2.6B-128K** — a parallel decision model built on Qwen3-0.6B, upcycled to 8-expert MoE with YaRN 128K context extension. States and typed questions in, complete probability distributions out. Zero output-token decoding.

> Inspired by [Jev (TypeSafe)](https://typesafe.ai/blog/introducing-system-one-models-and-jev) and [NanoJev](https://github.com/TianyuCodings/NanoJev). Built on [mistral.rs](https://github.com/EricLBuehler/mistral.rs) (MIT).

## Architecture

```
Qwen3-0.6B (dense, 751.6M)
    │ upcycling: importance-based neuron split
    ▼
8-expert MoE (2,446M total, ~1.02B active per token)
    │ + YaRN rope_scaling (factor 3.2, 40K → 131K)
    │ + Switch-Transformer load-balancing aux loss
    │ + set-attention decision head (~500K params)
    ▼
Apeireth-Decis-2.6B-128K
    output: P(candidate_1), ..., P(candidate_K)  ← single forward, no decoding
```

## Full Training Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│  1. DATA PRODUCTION                                         │
│     teacher: jev-1.13.0 via classifier.dev / knox.chat      │
│     → soft distribution over candidate labels per state     │
│     → 5-split: train / dev / calibration / test / ood      │
├─────────────────────────────────────────────────────────────┤
│  2. MoE UPCYCLING                                           │
│     upcycle_to_moe.py                                       │
│     strategy: importance (E[a_j²]·‖down[:,j]‖² neuron rank) │
│     → shared expert (top-k) + routed (serpentine fill)     │
│     → α least-squares scale calibration                    │
│     → YaRN rope_scaling injected into config.json           │
├─────────────────────────────────────────────────────────────┤
│  3. STREAMING CORPUS WARMUP (stage 1)                       │
│     42+ streaming HF sources, weighted round-robin         │
│     → domain coverage for router + backbone                 │
├─────────────────────────────────────────────────────────────┤
│  4. DECISION FINETUNE (stage 2)                             │
│     colab_train.py on Kaggle T4×2                           │
│     loss: soft-target CE ≡ forward KL (distillation)       │
│     + entropy-weighted sampling (teacher-hesitant ↑)       │
│     + temperature annealing T: 5.0 → 1.5                   │
│     + MoE router load-balancing (anti-collapse)            │
│     + gradient checkpointing + AdamW8bit                   │
│     → dev-set best.safetensors                             │
├─────────────────────────────────────────────────────────────┤
│  5. MULTIMODAL EXTENSION (phase 2)                          │
│     multimodal/mm_decision_model.py                        │
│     CLIP-ViT-B-16 → visual prefix → backbone               │
│     SWD-H: sliced Wasserstein hidden-state alignment       │
│     signal: jev labels text description of image           │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Produce teacher data
```bash
# via knox.chat (unlimited daily quota)
export KNOX_KEY="your_key"
python distill_knox.py --seeds seeds_*.jsonl --out data/labeled.jsonl

# via classifier.dev (20K/day/IP free tier)
python generate_fast.py --seeds seeds_*.jsonl --budget 19800
```

### 2. Upcycle to MoE
```bash
python upcycle_to_moe.py \
    --model Qwen/Qwen3-0.6B \
    --out ./moe-8e \
    --num-experts 8 \
    --top-k 2 \
    --split-strategy importance
```

### 3. Train on Kaggle (T4×2)
```bash
# push kernel
cd kaggle-train && kaggle kernels push -p .

# or run locally
python colab_train.py \
    --nanojev-scripts ./ \
    --input unified.jsonl \
    --model ./moe-8e \
    --steps 600 --head-steps 24 \
    --batch-questions 4 \
    --temperature 5.0 --temperature-final 1.5 \
    --entropy-weighting --dtype fp32 --adam8bit
```

### 4. Multimodal extension (phase 2)
```bash
cd multimodal
python gen_mm_data.py --source coco --count 500 --out mm_coco.jsonl
python train_mm.py --checkpoint ./moe-8e --data mm_coco.jsonl --steps 200
```

## Repository Structure

```
distill/
├── README.md                  ← this file
├── LICENSE                    ← MIT (training code)
├── CITATIONS.md               ← all referenced repos/datasets/papers
├── colab_train.py             ← core decision finetune script
├── upcycle_to_moe.py          ← dense → MoE upcycling
├── stream_sources.py          ← 42+ streaming corpus sources
├── generate_fast.py           ← concurrent classifier.dev labeling
├── generate_perrow.py         ← per-row candidate labeling
├── distill_knox.py            ← knox.chat jev gateway distiller
├── distill_soft.py            ← vLLM logprobs-based soft distiller
├── convert_nanojev.py         ← teacher labels → unified 5-split format
├── seeds_*.jsonl              ← seed pools (tickets, books, code, math...)
├── nanojev_format/            ← unified.jsonl (5-split training data)
├── multimodal/
│   ├── mm_decision_model.py   ← visual decision model
│   ├── train_mm.py            ← multimodal training script
│   ├── gen_mm_data.py         ← COCO → decision data pipeline
│   └── train_mm_helper.py     ← shared knox labeling helper
├── RESOURCES_LEDGER.md        ← full resource inventory (12+ batches)
├── OPTIMIZATION_BACKLOG.md    ← P0/P1 optimization items
└── RUNTIME_CONTRACT.md        ← jev-compatible API schema
```

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Importance split > copy+noise | UPCYCLE.md: activation-based split prevents expert homogenization (CV≈0 vs ~1) |
| Soft labels > hard labels | Dark knowledge (class similarity, hesitation) transfers via KL |
| Entropy-weighted sampling | Teacher-hesitant questions are harder; upweight them |
| Temperature annealing 5→1.5 | High T early = exploration; low T late = calibration |
| MoE 8-expert top-2 | 3.25× capacity, same FLOPs (token-level routing) |
| YaRN factor 3.2 | 40K native → 131K cap, official NVIDIA recipe values |

## Hardware Requirements

| Phase | Min GPU | Peak Memory | Training Time |
|-------|---------|-------------|---------------|
| Data production | CPU only | - | ~2-4h per 25K rows |
| Upcycling | 1× T4 (16GB) | ~10GB | ~5 min |
| Warmup (stage 1) | 2× T4 (32GB) | ~12GB/GPU | 90 min |
| Decision finetune | 2× T4 (32GB) | ~12.3GB/GPU | ~35 min (600 steps) |
| Multimodal (SWD-H) | 1× T4 (16GB) | ~2.54GB | ~30 min (200 steps) |

## License

Training pipeline code: [MIT](LICENSE)
Base model (Qwen3-0.6B): [Apache-2.0](https://huggingface.co/Qwen/Qwen3-0.6B)
Inference engine (mistral.rs): [MIT](https://github.com/EricLBuehler/mistral.rs)

See [CITATIONS.md](CITATIONS.md) for full attribution of all referenced repos, datasets, and papers.
