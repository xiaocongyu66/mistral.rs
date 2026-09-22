# Citations & References

All repos, datasets, papers, and frameworks referenced in the Apeireth-Decis training pipeline.

## Core Dependencies

| Project | License | Role |
|---------|---------|------|
| [mistral.rs](https://github.com/EricLBuehler/mistral.rs) | MIT | Inference engine (fork base) |
| [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | Apache-2.0 | Base model (upcycled to MoE) |
| [NanoJev](https://github.com/TianyuCodings/NanoJev) | MIT | Decision model architecture + training recipe |
| [NanoJev-Data](https://huggingface.co/datasets/C-Tianyu/NanoJev) | CC0-1.0 | Game decision training data (18,760 questions) |
| [Jev (TypeSafe)](https://typesafe.ai/blog/introducing-system-one-models-and-jev) | Closed | Concept + teacher model (jev-1.13.0) |

## Teacher Gateways

| Gateway | Endpoint | Quota |
|---------|----------|-------|
| [classifier.dev](https://classifier.dev) | POST /v1/classify | 20K/day/IP free |
| [knox.chat](https://api.knox.chat) | POST /v1/systemone | Unlimited (API key) |
| vLLM (133.186.146.224) | OpenAI-compatible logprobs | Unlimited |

## Training Frameworks Referenced

| Framework | Source | What We Took |
|-----------|--------|--------------|
| [Light-MER](https://github.com/GAIR-Lab/Light-MER) | Apache-2.0 | SWDProjector + sliced_wasserstein_loss (ot_loss.py) |
| [Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) | NVIDIA | Qwen3-0.6B YaRN 128K SFT recipe |
| [Unsloth](https://github.com/unslothai/unsloth) | Apache-2.0 | MoE finetuning optimization reference |
| [Drop-Upcycling](https://arxiv.org/abs/2502.19261) | arXiv | Expert splitting theory |
| [UPCYCLE.md](https://huggingface.co/AlexWortega/moe-600m-qwen3-upcycle) | - | Importance-based neuron split, engineering traps |

## Datasets Used in Training

### Decision & Intent (direct hits)
| Dataset | Source | Rows | License |
|---------|--------|------|---------|
| choices13k | [github](https://github.com/pankajcs/choices13k) | 13,006 | CC-BY |
| Newcomb-like Questions | [github](https://github.com/casparoe/newcomblike_questions_dataset) | 341 | CC-BY |
| [jev-bench](https://huggingface.co/datasets/Praveenrajus/jev-bench) | HF | 133,953 | - |
| [Open-Jev](https://huggingface.co/datasets/ZefanCai/Open-Jev) | HF | - | - |
| CLINC 150 | [HF](https://huggingface.co/datasets/clinc/clinc_oos) | ~23K | CC-BY-4.0 |
| BlendX | [HF](https://huggingface.co/datasets/HYU-NLP/BlendX) | - | - |
| MASSIVE | [HF](https://huggingface.co/datasets/AmazonScience/massive) | 1M+ | CC-BY-4.0 |
| IntentGrasp | [HF](https://huggingface.co/datasets/yuweiyin/IntentGrasp) | 262K | - |
| MultiWOZ | [github](https://github.com/budzianowski/multiwoz) | 120K | MIT |
| RiSAWOZ | [HF](https://huggingface.co/datasets/GEM/RiSAWOZ) | 11.2K | - |

### Math & Reasoning
| Dataset | Source | Rows | License |
|---------|--------|------|---------|
| [NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) | HF | 860K | Apache-2.0 |
| [OpenMathInstruct-2](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2) | HF | 14M | NVIDIA |
| [OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k) | HF | 220K | Apache-2.0 |
| [MathInstruct](https://huggingface.co/datasets/TIGER-Lab/MathInstruct) | HF | 260K | MIT |

### Code
| Dataset | Source | Rows | License |
|---------|--------|------|---------|
| [codeparrot-clean](https://huggingface.co/datasets/codeparrot/codeparrot-clean) | HF | - | - |
| [OpenCodeReasoning](https://huggingface.co/datasets/nvidia/OpenCodeReasoning) | HF | 736K | NVIDIA |
| MIT/pr-agent/reviewdog | local repos | 1,500 files | MIT/Apache-2.0 |

### Comprehension (backbone warmup)
| Dataset | Source | Size | License |
|---------|--------|------|---------|
| [FineWiki](https://huggingface.co/datasets/HuggingFaceFW/finewiki) | HF | 325-lang | ODC-BY |
| [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | HF | 1.3T tok | ODC-BY |
| [Cosmopedia](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia) | HF | 30M | Apache-2.0 |
| [Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) | HF | 300+ lang | CC-BY-SA |
| [OpenHermes-2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5) | HF | 1M | Apache-2.0 |
| [UltraFeedback](https://huggingface.co/datasets/openbmb/UltraFeedback) | HF | 64K | MIT |
| [COIG-CQIA](https://huggingface.co/datasets/m-a-p/COIG-CQIA) | HF | - | - |
| Bactrian-X | [HF](https://huggingface.co/datasets/MBZUAI/Bactrian-X) | 3.4M | Apache-2.0 |

### Multimodal (phase 2)
| Dataset | Source | Size | License |
|---------|--------|------|---------|
| COCO Captions | [HF](https://huggingface.co/datasets/jxie/coco_captions) | 330K | CC-BY-4.0 |
| Flickr30k | [HF](https://huggingface.co/datasets/nlphuji/flickr30k) | 31K | CC-BY |
| LLaVA-Instruct-150K | [HF](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K) | 150K | Apache-2.0 |
| VQAv2 | [HF](https://huggingface.co/datasets/HuggingFaceM4/VQAv2) | 1.3M | CC-BY-4.0 |

## Papers Referenced

| Paper | arXiv | Key Concept |
|-------|-------|-------------|
| Drop-Upcycling | [2502.19261](https://arxiv.org/abs/2502.19261) | Training sparse MoE from dense |
| Dense2MoE | [2605.26496](https://arxiv.org/pdf/2605.26496) | On-device MoE Pareto frontier |
| Dirichlet-Prior Shaping | [2510.01185](https://arxiv.org/abs/2510.01185) | ICML 2026, router init |
| MoE Survey | [2407.06204](https://arxiv.org/abs/2407.06204) | Comprehensive MoE overview |
| YaRN | [2309.00071](https://arxiv.org/abs/2309.00071) | RoPE extension method |
| Switch-Transformer | [2101.03961](https://arxiv.org/abs/2101.03961) | Load-balancing aux loss |

## Community Projects Referenced

| Project | Source | What We Learned |
|---------|--------|-----------------|
| [jev-forge](https://github.com/zwliJay/jev-forge) | - | End-to-end Jev toolchain |
| [jimothy](https://github.com/AndrewPrifer/jimothy) | - | 15-45MB local distillation |
| [open-jev-typed-decision-engine](https://github.com/intikhab49/open-jev-typed-decision-engine) | - | 150M encoder, +0.03 acc, 4× speed |
| [open-code-review](https://github.com/alibaba/open-code-review) | Apache-2.0 | Code review dataset |
| [aacr-bench](https://github.com/aacr-bench) | - | PR review evaluation |
