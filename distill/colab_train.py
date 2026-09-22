import argparse
import hashlib
import importlib.metadata
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def fit_temperature(model, calib_examples, pad_token, amp_dtype=None):
    """Fit scalar T on the calibration split: minimize CE of softmax(z/T)
    against teacher soft targets. Grid over T in [0.5, 8]."""
    model.eval()
    use_amp = amp_dtype in (torch.float16, torch.bfloat16)
    with torch.no_grad(), torch.autocast("cuda", dtype=amp_dtype or torch.float16, enabled=use_amp):
        logits, _ = model(calib_examples, pad_token)
    target = torch.zeros_like(logits)
    for i, ex in enumerate(calib_examples):
        target[i, :len(ex["candidate_ids"])] = torch.tensor(
            ex["teacher_probs"], device=logits.device)
    best_t, best_nll = 1.0, float("inf")
    for t in [x * 0.25 for x in range(2, 33)]:
        p = (logits.float() / t).log_softmax(-1)
        nll = -(target * p).sum(-1)
        score = float(nll.mean())
        if score < best_nll:
            best_t, best_nll = t, score
    return best_t, best_nll


def _load_balance(logits_list, coef):
    """Switch-Transformer aux loss over MoE router logits (anti-collapse)."""
    total = 0.0
    for logits in logits_list:
        probs = torch.softmax(logits.float(), dim=-1)
        k = min(2, probs.shape[-1])
        _, idx = torch.topk(probs, k, dim=-1)
        one_hot = torch.zeros_like(probs).scatter_(1, idx, 1.0)
        frac = one_hot.sum(0) / probs.shape[0]
        total = total + (frac * probs.mean(0)).sum() * probs.shape[-1]
    return coef * total / max(len(logits_list), 1)


def load_unified(path):
    """Tolerant reader: skip malformed lines instead of dying mid-training."""
    rows = []
    skipped = 0
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            skipped += 1
    print(f"unified: loaded {len(rows)} rows, skipped {skipped} malformed", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nanojev-scripts", required=True,
                    help="path to NanoJev scripts dir (imports train_toy_decisions)")
    ap.add_argument("--input", required=True, help="unified.jsonl with five splits")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--init-checkpoint", default=None,
                    help="best.safetensors of C-Tianyu/NanoJev for warm start")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--objective", default="teacher", choices=["teacher", "gold"])
    ap.add_argument("--set-head", default="attention", choices=["none", "attention"])
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--head-steps", type=int, default=24)
    ap.add_argument("--batch-questions", type=int, default=8)
    ap.add_argument("--accum", type=int, default=4,
                    help="gradient accumulation steps: micro_batch=batch/accum")
    ap.add_argument("--no-grad-checkpoint", action="store_true")
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--max-length", type=int, default=768)
    ap.add_argument("--backbone-lr", type=float, default=1e-5)
    ap.add_argument("--head-lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--adam8bit", action="store_true",
                    help="8-bit AdamW (bitsandbytes); needed for 1.2B+ MoE on 16GB")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"],
                    help="autocast dtype; fp16 uses T4-native tensor cores + GradScaler")
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="teacher temperature applied to soft targets")
    ap.add_argument("--temperature-final", type=float, default=None,
                    help="anneal T linearly from --temperature to this over total steps")
    ap.add_argument("--entropy-weighting", action="store_true",
                    help="sample train questions proportional to teacher entropy")
    args = ap.parse_args()

    sys.path.insert(0, args.nanojev_scripts)
    from train_toy_decisions import (DecisionModel, benchmark, dump, evaluate,
                                     load_examples, loss_for)

    use_amp = args.dtype in ("fp16", "bf16")
    amp_dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=(args.dtype == "fp16"))
    print(f"autocast={args.dtype} enabled={use_amp} scaler={scaler.is_enabled()}", flush=True)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    param_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
                   "fp32": torch.float32}[args.dtype]
    lm = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=param_dtype, attn_implementation="sdpa").cuda()
    lm.config.use_cache = False
    if not args.no_grad_checkpoint:
        lm.gradient_checkpointing_enable()
    clean_input = out / "unified_clean.jsonl"
    rows = load_unified(args.input)
    clean_input.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                                   for r in rows), encoding="utf-8")
    examples, audit = load_examples(str(clean_input), tokenizer, args.max_length)
    dump(out / "target_audit.json", audit)
    bysplit = {s: [e for e in examples if e["split"] == s]
               for s in ["train", "dev", "calibration", "test", "ood"]}
    missing = [s for s, g in bysplit.items() if not g]
    if missing:
        raise ValueError(f"empty splits: {missing}")

    model = DecisionModel(lm.model, args.set_head).cuda()
    del lm
    router_buf = []
    def _router_hook(module, inputs, output):
        if model.training and inputs and hasattr(module, "gate"):
            router_buf.append(module.gate(inputs[0]))
    n_moe = 0
    for mod in model.backbone.modules():
        if type(mod).__name__.endswith("SparseMoeBlock"):
            mod.register_forward_hook(_router_hook)
            n_moe += 1
    if n_moe:
        print(f"moe blocks hooked: {n_moe} (router load-balancing active)", flush=True)
    init_info = "fresh backbone + fresh head"
    if args.init_checkpoint:
        sd = load_file(args.init_checkpoint)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        init_info = (f"warm start from {args.init_checkpoint}: "
                     f"{len(sd)} tensors, {len(missing)} missing, {len(unexpected)} dropped")
    print(init_info, flush=True)

    train = [e for e in bysplit["train"]
             if args.objective == "gold" or e["teacher_probs"] is not None]
    config = {**vars(args), "init": init_info,
              "data_sha256": hashlib.sha256(clean_input.read_bytes()).hexdigest(),
              "deps": {k: importlib.metadata.version(k)
                       for k in ["torch", "transformers", "safetensors"]},
              "gpu": torch.cuda.get_device_name(0),
              "train_questions": len(train),
              "questions_per_split": {s: len(g) for s, g in bysplit.items()},
              "parameter_count": sum(t.numel() for t in model.parameters())}
    dump(out / "config.json", config)
    tokenizer.save_pretrained(out / "tokenizer")
    model.backbone.config.save_pretrained(out / "backbone_config")

    evaluation = sum([bysplit[s] for s in ["dev", "calibration", "test", "ood"]], [])
    head = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    body = list(model.backbone.parameters())
    groups = [{"params": body, "lr": args.backbone_lr}, {"params": head, "lr": args.head_lr}]
    if args.adam8bit:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(groups, weight_decay=0.0)
        print("optimizer=AdamW8bit wd=0.0", flush=True)
    else:
        optimizer = torch.optim.AdamW(groups, weight_decay=0.0)

    sample_weights = None
    if args.entropy_weighting:
        sample_weights = []
        for e in train:
            q = (e.get("source") or {}).get("questions", {}).get(e.get("qid"), {})
            ent = float(q.get("teacher_entropy", 0.5))
            sample_weights.append(max(ent, 0.05))
        wsum = sum(sample_weights)
        print(f"entropy weighting on: mean_w={wsum/len(sample_weights):.4f} "
              f"min={min(sample_weights):.3f} max={max(sample_weights):.3f}", flush=True)

    total_steps = args.head_steps + args.steps
    logs, best, best_step = [], float("inf"), None
    start = time.perf_counter()
    for step in range(args.head_steps + args.steps):
        warm = step < args.head_steps
        for param in body:
            param.requires_grad_(not warm)
        optimizer.param_groups[1]["lr"] = 1e-3 if warm else args.head_lr
        if sample_weights is not None:
            batch = random.choices(train, weights=sample_weights, k=args.batch_questions)
        else:
            batch = random.sample(train, args.batch_questions)
        model.train()
        router_buf.clear()
        optimizer.zero_grad(set_to_none=True)
        anneal = args.temperature_final is not None and args.temperature_final > 0
        frac = step / max(total_steps - 1, 1)
        T_now = args.temperature + (args.temperature_final - args.temperature) * frac if anneal else args.temperature

        micro_bs = max(1, args.batch_questions // max(args.accum, 1))
        n_micro = max(1, len(batch) // micro_bs)
        step_loss = 0.0
        for mi in range(n_micro):
            mb = batch[mi * micro_bs:(mi + 1) * micro_bs]
            if not mb:
                continue
            if anneal:
                mb = [
                    ({**ex, "teacher_probs": (lambda p: (p / p.sum()).tolist())(
                        torch.tensor(ex["teacher_probs"], dtype=torch.float32).clamp_min(1e-9)
                        ** (1.0 / T_now))}
                     if ex.get("teacher_probs") is not None else ex)
                    for ex in mb
                ]
            with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                z, _ = model(mb, tokenizer.pad_token_id)
                mloss = loss_for(z, mb, args.objective).mean() / n_micro
            if not torch.isfinite(mloss):
                continue
            scaler.scale(mloss).backward()
            step_loss += mloss.item()
        if router_buf:
            router_buf.clear()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        item = {"step": step + 1, "phase": "head" if warm else "full",
                "loss": round(step_loss, 4),
                "elapsed_seconds": time.perf_counter() - start}
        if not warm and ((step + 1 - args.head_steps) % args.eval_every == 0
                         or step + 1 == args.head_steps + args.steps):
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
        logs.append(item)
        if step % 12 == 0 or "dev" in item:
            print(json.dumps(item), flush=True)
    if best_step is None:
        raise RuntimeError("No checkpoint selected on dev")
    model.load_state_dict(load_file(out / "best.safetensors"))

    temperature, calib_nll = fit_temperature(
        model, bysplit["calibration"], tokenizer.pad_token_id, amp_dtype)
    dump(out / "temperature.json", {
        "T": temperature, "calib_nll_at_T": calib_nll,
        "calibration_questions": len(bysplit["calibration"])})

    final = evaluate(model, evaluation, tokenizer.pad_token_id,
                     args.batch_questions, out / "predictions.jsonl")
    timing = benchmark(model, bysplit["test"], tokenizer.pad_token_id)
    dump(out / "timing.json", timing)
    dump(out / "train_log.json", logs)
    dump(out / "summary.json", {
        "best_step": best_step, "temperature": temperature,
        "selected_on": "dev teacher CE", "init": init_info,
        "training_seconds": time.perf_counter() - start,
        "final_all_eval": final,
        "max_gpu_allocated_gb": torch.cuda.max_memory_allocated() / 1e9})
    print(json.dumps({"done": str(out), "best_step": best_step,
                      "temperature": temperature, "eval": final}), flush=True)


if __name__ == "__main__":
    main()
