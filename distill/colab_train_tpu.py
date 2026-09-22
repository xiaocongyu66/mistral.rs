"""Decision finetune on Kaggle TPU (PyTorch XLA port of colab_train.py).

Key changes from CUDA version:
  - device: xm.xla_device() instead of .cuda()
  - optimizer: regular AdamW (no bitsandbytes — CUDA-only)
  - autocast: torch_xla.amp.autocast(xm.xla_device())
  - no device_map / max_memory (single XLA device, 96GB HBM)
  - gradient checkpointing: same API, works on XLA
  - all tensor ops: MarkStep after each optimizer step

Usage (Kaggle TPU v3-8 kernel):
  python colab_train_tpu.py --nanojev-scripts ./ --input unified.jsonl \
      --output-dir ./run1 --model ./moe-8e --steps 600 --head-steps 24 \
      --batch-questions 8 --temperature 5.0 --temperature-final 1.5 \
      --entropy-weighting --dtype fp32
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch_xla.core.xla_model as xm
import torch_xla.amp as xa
import torch_xla.runtime as xr
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file, save_file


def _round(x, n):
    return round(x, n) if isinstance(x, (int, float)) else x


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def load_unified(path):
    rows, skipped = [], 0
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            skipped += 1
    print(f"unified: loaded {len(rows)} rows, skipped {skipped} malformed", flush=True)
    return rows


def fix_gate_orientation(model_dir):
    from safetensors.torch import load_file as _lf, save_file as _sf
    _p = os.path.join(model_dir, "model.safetensors")
    if not os.path.exists(_p):
        return
    sd = _lf(_p)
    g0 = sd.get("model.layers.0.mlp.gate.weight")
    if g0 is None or g0.shape[0] <= g0.shape[1]:
        return
    for k in list(sd):
        if k.endswith("mlp.gate.weight"):
            sd[k] = sd[k].t().contiguous()
    _sf(sd, _p)
    print("gate orientation self-healed (transposed)", flush=True)


def fit_temperature(model, calib_examples, pad_token, dev):
    model.eval()
    n_cand = max(len(ex["candidate_ids"]) for ex in calib_examples)
    max_len = max(max(len(ids) for ids in ex["candidate_ids"]) for ex in calib_examples)
    bs = max(1, (8 * 1024 * 1024 * 1024) // (n_cand * max_len * 1024 * 4))
    all_logits = []
    with torch.no_grad():
        for s in range(0, len(calib_examples), bs):
            chunk = calib_examples[s:s + bs]
            lg, _ = model(chunk, pad_token)
            all_logits.append(lg.float().cpu())
    logits = torch.cat(all_logits, 0)
    target = torch.zeros_like(logits)
    for i, ex in enumerate(calib_examples):
        target[i, :len(ex["candidate_ids"])] = torch.tensor(ex["teacher_probs"])
    best_t, best_nll = 1.0, float("inf")
    for t in [x * 0.25 for x in range(2, 33)]:
        p = (logits / t).log_softmax(-1)
        nll = -(target * p).sum(-1)
        score = float(nll.mean())
        if score < best_nll:
            best_t, best_nll = t, score
    return best_t, best_nll


def _load_balance(logits_list, coef):
    total = None
    for logits in logits_list:
        probs = torch.softmax(logits.float(), dim=-1)
        k = min(2, probs.shape[-1])
        _, idx = torch.topk(probs, k, dim=-1)
        one_hot = torch.zeros_like(probs).scatter_(1, idx, 1.0)
        frac = one_hot.sum(0) / probs.shape[0]
        term = (frac * probs.mean(0)).sum() * probs.shape[-1]
        loss = coef * (term - 1.0 / probs.shape[-1])
        total = loss if total is None else total + loss
    return total if total is not None else torch.tensor(0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nanojev-scripts", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--objective", default="teacher", choices=["teacher", "gold"])
    ap.add_argument("--set-head", default="attention", choices=["none", "attention"])
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--head-steps", type=int, default=24)
    ap.add_argument("--batch-questions", type=int, default=8,
                    help="TPU 96GB HBM allows larger batch than T4's 4")
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--no-grad-checkpoint", action="store_true")
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--backbone-lr", type=float, default=1e-5)
    ap.add_argument("--head-lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"],
                    help="TPU supports bf16 natively")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--temperature-final", type=float, default=None)
    ap.add_argument("--entropy-weighting", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, args.nanojev_scripts)
    from train_toy_decisions import (DecisionModel, benchmark, dump as nj_dump,
                                      evaluate, load_examples, loss_for)

    dev = xm.xla_device()
    print(f"XLA device: {dev}", flush=True)
    print(f"TPU runtime: {xr.get_runtime_version()}", flush=True)
    print(f"torch_xla: {__import__('torch_xla').__version__}", flush=True)

    # dtype: TPU native bf16
    param_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    fix_gate_orientation(args.model)
    _cfg = AutoConfig.from_pretrained(args.model)
    _mt = getattr(_cfg, "model_type", "")
    if _mt == "qwen3_moe":
        try:
            from transformers import Qwen3MoeForCausalLM as _Moe
            lm = _Moe.from_pretrained(args.model, torch_dtype=param_dtype,
                                       attn_implementation="sdpa")
            lm = lm.to(dev)
            print(f"loaded {_Moe.__name__} (explicit MoE class)", flush=True)
        except ImportError:
            print("FATAL: Qwen3MoeForCausalLM missing", flush=True)
            raise
    else:
        lm = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=param_dtype, attn_implementation="sdpa").to(dev)
        print(f"loaded {type(lm).__name__} (model_type={_mt})", flush=True)

    lm.config.use_cache = False
    if not args.no_grad_checkpoint:
        lm.gradient_checkpointing_enable()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # write clean copy
    rows = load_unified(args.input)
    clean_input = out / "unified_clean.jsonl"
    clean_input.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    examples, audit = load_examples(str(clean_input), tokenizer, args.max_length)
    dump(out / "target_audit.json", audit)
    bysplit = {s: [e for e in examples if e["split"] == s]
               for s in ["train", "dev", "calibration", "test", "ood"]}
    missing = [s for s, g in bysplit.items() if not g]
    if missing:
        raise ValueError(f"empty splits: {missing}")

    model = DecisionModel(lm.model, args.set_head).to(dev)
    del lm

    # router hook (same as CUDA version)
    router_buf = []
    def _router_hook(module, inputs, output):
        if model.training and inputs and hasattr(module, "gate"):
            router_buf.append((module.gate, inputs[0]))
    n_moe = 0
    for mod in model.backbone.modules():
        if type(mod).__name__.endswith("SparseMoeBlock"):
            mod.register_forward_hook(_router_hook)
            n_moe += 1
    if n_moe:
        print(f"moe blocks hooked: {n_moe} (router load-balancing active)", flush=True)

    # optimizer: regular AdamW (no bitsandbytes on TPU)
    head = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    body = list(model.backbone.parameters())
    groups = [{"params": body, "lr": args.backbone_lr},
              {"params": head, "lr": args.head_lr}]
    optimizer = torch.optim.AdamW(groups, weight_decay=0.0)
    print("optimizer=AdamW (full precision, TPU 96GB HBM)", flush=True)

    # entropy weighting
    sample_weights = None
    if args.entropy_weighting:
        sample_weights = []
        for e in bysplit["train"]:
            q = (e.get("source") or {}).get("questions", {}).get(e.get("qid"), {})
            ent = float(q.get("teacher_entropy", 0.5))
            sample_weights.append(max(ent, 0.05))
        wsum = sum(sample_weights)
        print(f"entropy weighting on: mean_w={wsum/len(sample_weights):.4f}", flush=True)

    # lr warmup
    warmup_iters = 30
    def _lr_lambda(step):
        if step < warmup_iters:
            return (step + 1) / warmup_iters
        return 1.0
    schedulers = [torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)]
    print(f"lr warmup: {warmup_iters} iters linear 0->1", flush=True)

    # training loop
    micro_bs = max(1, args.batch_questions // args.accum)
    total_steps = args.head_steps + args.steps
    logs, best, best_step = [], float("inf"), None
    start = time.perf_counter()

    for step in range(total_steps):
        warm = step < args.head_steps
        for param in body:
            param.requires_grad = not warm
        if warm:
            head_lr_override = 1e-3
            for g in optimizer.param_groups:
                if g is not groups[0]:
                    g["lr"] = head_lr_override

        if sample_weights:
            batch = random.choices(bysplit["train"], weights=sample_weights,
                                    k=args.batch_questions)
        else:
            batch = random.sample(bysplit["train"], args.batch_questions)

        step_loss = 0.0
        n_micro = max(1, len(batch) // micro_bs)
        for mi in range(n_micro):
            mb = batch[mi * micro_bs:(mi + 1) * micro_bs]
            if not mb:
                continue
            with xa.autocast(dev, enabled=(args.dtype == "bf16")):
                ce, aux = loss_for(model, mb, tokenizer.pad_token_id)
                mloss = ce / n_micro
            if router_buf:
                hiddens = [h.detach() for _, h in router_buf]
                gates = [fn(h) for fn, h in router_buf]
                mloss = mloss + _load_balance(gates, 0.01).to(mloss.device)
            router_buf.clear()
            if not torch.isfinite(mloss):
                continue
            mloss.backward()
            router_buf.clear()
            step_loss += mloss.item()

        if router_buf:
            router_buf.clear()
        xm.optimizer_step(optimizer)  # XLA optimizer step (handles all-reduce)
        for s in schedulers:
            s.step()

        if (step + 1) % 50 == 0:
            _e = batch[0]
            _tp = _e.get("teacher_probs") or []
            print(f"[sample] {_e.get('qid', '?')} "
                  f"state={str(_e.get('state', ''))[:80]!r} "
                  f"teacher={[_round(x, 2) for x in _tp[:5]] if _tp else 'gold-only'}",
                  flush=True)

        item = {"step": step + 1, "phase": "head" if warm else "full",
                "loss": round(step_loss, 4),
                "elapsed_seconds": time.perf_counter() - start}

        if not warm and ((step + 1 - args.head_steps) % args.eval_every == 0
                          or step + 1 == total_steps):
            router_buf.clear()
            metrics = evaluate(model, bysplit["dev"], tokenizer.pad_token_id,
                                args.batch_questions)
            router_buf.clear()
            item["dev"] = metrics
            score = metrics["teacher_ce" if args.objective == "teacher" else "gold_nll"]
            if score is not None and score < best:
                best, best_step = score, step + 1
                save_file({k: v.detach().cpu().contiguous().clone()
                            for k, v in model.state_dict().items()},
                          out / "best.safetensors")
                print(f"  ** best {score:.4f} at step {step+1}", flush=True)

        logs.append(item)
        if step % 12 == 0 or "dev" in item:
            print(json.dumps(item, ensure_ascii=False), flush=True)

    # final temperature fit
    if best_step:
        temperature, calib_nll = fit_temperature(model, bysplit["calibration"],
                                                   tokenizer.pad_token_id, dev)
        summary = {"best_step": best_step, "best_score": best,
                    "temperature": temperature, "calib_nll": calib_nll,
                    "objective": args.objective}
        dump(out / "summary.json", summary)
        print(f"final: {json.dumps(summary)}", flush=True)
    else:
        raise RuntimeError("No checkpoint selected on dev")

    # save train log
    dump(out / "train_log.json", logs)
    print(f"TPU_TRAIN_DONE best={best:.4f} step={best_step}", flush=True)


if __name__ == "__main__":
    main()
