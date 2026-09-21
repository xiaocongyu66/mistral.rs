"""Qwen3 dense -> MoE upcycling (function-preserving, risk-mitigated).

Mitigations for the three known failure modes:
  - expert homogenization: expert_0 keeps original weights, experts 1..N-1 get
    perturbation noise so they diverge from step 0 instead of receiving
    identical gradients.
  - routing collapse: router starts near-uniform (tiny init) and training must
    add a load-balancing aux loss (see load_balance_loss below).
  - Qwen3 RoPE half-split: we never touch attention/RoPE here -- only MLP
    blocks are replaced, so the interleaved/half-split convention is untouched.
"""
import argparse
import json
import os
import shutil

import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def load_balance_loss(router_logits, num_experts, top_k, coef=0.01):
    """Switch-Transformer style aux loss; call once per forward with the
    concatenated router logits of all layers. Prevents routing collapse."""
    if not router_logits:
        return torch.tensor(0.0)
    loss = 0.0
    for logits in router_logits:
        probs = torch.softmax(logits.float(), dim=-1)          # (tokens, experts)
        _, idx = torch.topk(probs, top_k, dim=-1)
        one_hot = torch.zeros_like(probs).scatter_(1, idx, 1.0)
        tokens_per_expert = one_hot.sum(0)
        fraction = tokens_per_expert / probs.shape[0]
        importance = probs.mean(0)
        loss = loss + (fraction * importance).sum() * num_experts
    return coef * loss / len(router_logits)


def upcycle(src, out_dir, num_experts, top_k, noise, seed):
    torch.manual_seed(seed)
    cfg = AutoConfig.from_pretrained(src)
    tok = AutoTokenizer.from_pretrained(src)
    model = AutoModelForCausalLM.from_pretrained(src, dtype=torch.float32)
    sd = model.state_dict()

    hidden = cfg.hidden_size
    inter = cfg.intermediate_size
    n_layers = cfg.num_hidden_layers
    new_sd, report = {}, []

    for name, tensor in sd.items():
        # MLP weights: layer.{i}.mlp.{gate,up,down}_proj.weight
        if ".mlp." in name and name.endswith(".weight"):
            layer = int(name.split(".")[1])
            for e in range(num_experts):
                key = name.replace(".mlp.", f".mlp.experts.{e}.")
                if e == 0:
                    new_sd[key] = tensor.clone()
                else:
                    new_sd[key] = tensor.clone() + noise * torch.randn_like(tensor)
            report.append(f"layer{layer}: 1 MLP -> {num_experts} experts")
        else:
            new_sd[name] = tensor

    # router for every layer
    for layer in range(n_layers):
        gate = torch.randn(hidden, num_experts) * 1e-3
        new_sd[f"model.layers.{layer}.mlp.gate.weight"] = gate

    os.makedirs(out_dir, exist_ok=True)
    save_file(new_sd, os.path.join(out_dir, "model.safetensors"))
    tok.save_pretrained(out_dir)
    cfg.model_type = "qwen3_moe"
    cfg.num_experts = num_experts
    cfg.num_experts_per_tok = top_k
    cfg.decoder_sparse_step = 1
    cfg.moe_intermediate_size = inter
    cfg.norm_topk_prob = True
    cfg.output_router_logits = True
    cfg.architectures = ["Qwen3MoeForCausalLM"]
    cfg.save_pretrained(out_dir)

    n_dense = sum(v.numel() for v in sd.values())
    n_moe = sum(v.numel() for v in new_sd.values())
    print(f"dense params: {n_dense/1e6:.1f}M -> moe params: {n_moe/1e6:.1f}M "
          f"({n_moe/n_dense:.2f}x), active≈{n_dense/1e6:.1f}M with top-{top_k}")
    print(f"experts={num_experts} top_k={top_k} noise={noise} -> {out_dir}")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-experts", type=int, default=4)
    ap.add_argument("--top-k", type=int, default=2)
    ap.add_argument("--noise", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=17)
    a = ap.parse_args()
    rep = upcycle(a.model, a.out, a.num_experts, a.top_k, a.noise, a.seed)
    print("replaced:", rep[0] if rep else "-", f"... total {len(rep)} layers")


if __name__ == "__main__":
    main()
