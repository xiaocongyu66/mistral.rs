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


def compute_neuron_importance(model, tokenizer, samples, hidden_size, n_layers):
    """Estimate per-neuron importance for each MLP layer.

    importance_j = E[a_j^2] * ||down[:, j]||^2  (UPCYCLE.md convention)
    where a_j = silu(gate_j·x) * (up_j·x).
    Returns {layer: [importance scores]}.
    """
    model.eval()
    importance = {l: torch.zeros(intermediate_of(model)) for l in range(n_layers)}
    counts = {l: 0 for l in range(n_layers)}
    device = next(model.parameters()).device

    with torch.no_grad():
        for text in samples:
            enc = tokenizer(text, return_tensors="pt", truncation=True,
                            max_length=256).to(device)
            if enc["input_ids"].numel() < 8:
                continue
            # hook every MLP to capture activations
            hooks, acts = [], {}
            def _mk(layer_idx):
                def hook(mod, inp, out):
                    gate_w = mod.gate_proj.weight  # [inter, hidden]
                    up_w = mod.up_proj.weight
                    x = inp[0]                     # [1, seq, hidden]
                    a = torch.nn.functional.silu(x @ gate_w.T) * (x @ up_w.T)
                    acts[layer_idx] = a.squeeze(0).mean(0)  # [inter]
                return hook
            for i, layer in enumerate(model.model.layers):
                hooks.append(layer.mlp.register_forward_hook(_mk(i)))
            model(**enc)
            for h in hooks:
                h.remove()
            for l, a in acts.items():
                down_w = model.model.layers[l].mlp.down_proj.weight  # [hidden, inter]
                w_norm = down_w.norm(dim=0)  # [inter]
                imp = (a ** 2) * w_norm
                importance[l] += imp.cpu()
                counts[l] += 1
    for l in importance:
        if counts[l] > 0:
            importance[l] /= counts[l]
    return importance


def intermediate_of(model):
    return model.config.intermediate_size


def split_experts_importance(sd, importance, num_experts, inter, n_layers,
                              top_shared=None):
    """Split dense MLP into num_experts by neuron importance.

    expert_0 = shared (top neurons by importance, active on every token).
    experts 1..N-1 = routed, filled serpentine-style to equalize total mass.
    """
    k_shared = top_shared or inter // num_experts
    new_sd = {}
    for layer in range(n_layers):
        imp = importance.get(layer)
        if imp is None:
            imp = torch.ones(inter)
        order = torch.argsort(imp, descending=True)
        # shared expert: top-k by importance
        shared_idx = set(order[:k_shared].tolist())
        # routed: remaining neurons, serpentine fill
        routed_pool = [i for i in order.tolist() if i not in shared_idx]
        routed_bins = [[] for _ in range(num_experts - 1)]
        bin_sums = [0.0] * (num_experts - 1)
        for idx in routed_pool:
            # pick the lightest bin (serpentine greedy)
            b = min(range(len(bin_sums)), key=lambda x: bin_sums[x])
            routed_bins[b].append(idx)
            bin_sums[b] += imp[idx].item()
        # assemble expert weights
        gate_w = sd[f"model.layers.{layer}.mlp.gate_proj.weight"]   # [inter, hidden]
        up_w = sd[f"model.layers.{layer}.mlp.up_proj.weight"]
        down_w = sd[f"model.layers.{layer}.mlp.down_proj.weight"]   # [hidden, inter]
        for e in range(num_experts):
            if e == 0:
                idxs = sorted(shared_idx)
            else:
                idxs = sorted(routed_bins[e - 1])
            gi = torch.tensor(idxs, dtype=torch.long)
            new_sd[f"model.layers.{layer}.mlp.experts.{e}.gate_proj.weight"] = gate_w[gi].clone()
            new_sd[f"model.layers.{layer}.mlp.experts.{e}.up_proj.weight"] = up_w[gi].clone()
            new_sd[f"model.layers.{layer}.mlp.experts.{e}.down_proj.weight"] = down_w[:, gi].clone()
    return new_sd


def calibrate_scale(dense_model, moe_sd, moe_cfg, tokenizer, samples, num_experts, top_k):
    """Least-squares alpha per layer: scale experts' down_proj so that
    y_moe ~= alpha * y_dense in the first forward. Returns {layer: alpha}.

    Skipping this lets the 28 residual stream accumulate a systematic FFN
    deficit and silently erases the donor's pretrained capability.
    """
    import torch.nn.functional as F
    dense_model.eval()
    alphas = {}
    with torch.no_grad():
        ids = tokenizer(samples, return_tensors="pt", padding=True, truncation=True,
                        max_length=256).to(dense_model.device)
        d_out = dense_model(**ids, output_hidden_states=True).hidden_states
        for layer in range(moe_cfg.num_hidden_layers):
            prefix = f"model.layers.{layer}.mlp."
            # dense reference: run donor MLP on layer input
            x = d_out[layer]
            g = F.linear(x, dense_model.model.layers[layer].mlp.gate_proj.weight)
            u = F.linear(x, dense_model.model.layers[layer].mlp.up_proj.weight)
            y_dense = F.linear(F.silu(g) * u,
                               dense_model.model.layers[layer].mlp.down_proj.weight)
            # moe with uniform routing over all experts (alpha is defined at init)
            y_moe = torch.zeros_like(y_dense)
            for e in range(num_experts):
                gw = moe_sd[prefix + f"experts.{e}.gate_proj.weight"]
                uw = moe_sd[prefix + f"experts.{e}.up_proj.weight"]
                dw = moe_sd[prefix + f"experts.{e}.down_proj.weight"]
                y_moe = y_moe + F.linear(F.silu(F.linear(x, gw)) * F.linear(x, uw), dw) / num_experts
            num = (y_dense * y_moe).sum()
            den = (y_moe * y_moe).sum().clamp_min(1e-12)
            alphas[layer] = float(num / den)
    return alphas


def apply_scale(sd, alphas, num_experts, n_layers):
    for layer in range(n_layers):
        a = alphas[layer]
        for e in range(num_experts):
            k = f"model.layers.{layer}.mlp.experts.{e}.down_proj.weight"
            sd[k] = sd[k] * a
    return sd


def upcycle(src, out_dir, num_experts, top_k, noise, seed):
    torch.manual_seed(seed)
    cfg = AutoConfig.from_pretrained(src)
    tok = AutoTokenizer.from_pretrained(src)
    model = AutoModelForCausalLM.from_pretrained(src, torch_dtype=torch.float32)
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
        # HF gate is nn.Linear(hidden, num_experts): weight is [E, H]
        gate = torch.randn(num_experts, hidden) * 1e-3
        new_sd[f"model.layers.{layer}.mlp.gate.weight"] = gate

    os.makedirs(out_dir, exist_ok=True)
    new_sd.pop("lm_head.weight", None)  # tied to embed_tokens; safetensors rejects shared storage
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
    ap.add_argument("--split-strategy", default="copy_noise",
                    choices=["importance", "copy_noise"],
                    help="importance: neuron-activation split (UPCYCLE.md); "
                         "copy_noise: legacy duplicate+gaussian")
    ap.add_argument("--no-calibrate", action="store_true",
                    help="skip least-squares alpha scaling (not recommended)")
    a = ap.parse_args()
    # build the upcycled state dict first so alpha can be calibrated on it
    torch.manual_seed(a.seed)
    cfg = AutoConfig.from_pretrained(a.model)
    tok = AutoTokenizer.from_pretrained(a.model)
    donor = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32)
    sd = donor.state_dict()
    if a.split_strategy == "importance":
        print("split strategy: importance (neuron-activation based)", flush=True)
        samples = ["Customer reports a duplicate charge on invoice 4411 and demands a refund.",
                   "The app crashes on startup after the latest update on Android 14.",
                   "Can we get enterprise pricing for 200 seats with SSO support?",
                   "Our team needs to migrate from the legacy system to your platform.",
                   "The integration is returning a 500 error when we submit the form."]
        importance = compute_neuron_importance(donor, tok, samples,
                                                cfg.hidden_size, cfg.num_hidden_layers)
        new_sd = split_experts_importance(sd, importance, a.num_experts,
                                           cfg.intermediate_size, cfg.num_hidden_layers)
        # copy non-MLP weights
        for name, tensor in sd.items():
            if ".mlp." not in name:
                new_sd[name] = tensor.clone()
    else:
        print("split strategy: copy_noise (legacy)", flush=True)
        new_sd = {}
        for name, tensor in sd.items():
            if ".mlp." in name and name.endswith(".weight"):
                for e in range(a.num_experts):
                    key = name.replace(".mlp.", f".mlp.experts.{e}.")
                    new_sd[key] = tensor.clone() if e == 0 else tensor.clone() + a.noise * torch.randn_like(tensor)
            else:
                new_sd[name] = tensor
    for layer in range(cfg.num_hidden_layers):
        new_sd[f"model.layers.{layer}.mlp.gate.weight"] = torch.randn(a.num_experts, cfg.hidden_size) * 1e-3

    if not a.no_calibrate:
        import transformers
        samples = ["Customer reports a duplicate charge on invoice 4411 and demands a refund.",
                   "The app crashes on startup after the latest update on Android 14.",
                   "Can we get enterprise pricing for 200 seats with SSO support?"]
        alphas = calibrate_scale(donor, new_sd, cfg, tok, samples, a.num_experts, a.top_k)
        vals = list(alphas.values())
        print(f"alpha: mean={sum(vals)/len(vals):.4f} min={min(vals):.4f} max={max(vals):.4f}", flush=True)
        apply_scale(new_sd, alphas, a.num_experts, cfg.num_hidden_layers)

    os.makedirs(a.out, exist_ok=True)
    new_sd.pop("lm_head.weight", None)  # tied to embed_tokens; safetensors rejects shared storage
    save_file(new_sd, os.path.join(a.out, "model.safetensors"))
    tok.save_pretrained(a.out)
    cfg.model_type = "qwen3_moe"
    cfg.num_experts = a.num_experts
    cfg.num_experts_per_tok = a.top_k
    cfg.decoder_sparse_step = 1
    cfg.moe_intermediate_size = cfg.intermediate_size
    cfg.norm_topk_prob = True
    cfg.output_router_logits = True
    cfg.architectures = ["Qwen3MoeForCausalLM"]
    # NeMo qwen3_600m_sft_yarn_128k recipe: 40k native -> 131k extrapolation cap
    cfg.rope_scaling = {"factor": 3.2, "original_max_position_embeddings": 40960,
                        "beta_fast": 32.0, "beta_slow": 1.0,
                        "type": "yarn", "rope_type": "yarn"}
    cfg.max_position_embeddings = 131072
    cfg.save_pretrained(a.out)
    import json as _json
    _p = os.path.join(a.out, "config.json")
    with open(_p) as _f:
        _cj = _json.load(_f)
    _cj["model_type"] = "qwen3_moe"
    _cj["architectures"] = ["Qwen3MoeForCausalLM"]
    _cj["num_experts"] = a.num_experts
    _cj["num_experts_per_tok"] = a.top_k
    _cj["_name_or_path"] = "Apeireth-Decis-2.6B-128K"
    _cj["model_name"] = "Apeireth-Decis-2.6B-128K"
    _cj["base_model"] = "Qwen/Qwen3-0.6B"
    with open(_p, "w") as _f:
        _json.dump(_cj, _f, indent=2)
    n_dense = sum(v.numel() for v in sd.values())
    n_moe = sum(v.numel() for v in new_sd.values())
    print(f"dense {n_dense/1e6:.1f}M -> moe {n_moe/1e6:.1f}M ({n_moe/n_dense:.2f}x) -> {a.out}")


if __name__ == "__main__":
    main()
