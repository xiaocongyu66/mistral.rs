import argparse
import hashlib
import os
import importlib.metadata
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
from transformers import AutoConfig
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer


def _round(x, n):
    return round(x, n) if isinstance(x, (int, float)) else x


def rlcd_loss(logits, target, reference_logits, utility_weight=1.0,
              calibration_weight=0.5, kl_weight=0.02):
    """JevForge-style RLCD objective over finite candidates.

    loss = -utility + calibration_weight*Brier + kl_weight*KL(policy||ref)
    where utility = sum(p * target/peak): put mass on teacher-favored candidates.
    """
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    probs = log_probs.exp()
    brier = ((probs - target) ** 2).sum(-1)
    peak = target.max(-1, keepdim=True).values.clamp_min(1e-8)
    utility = (probs * (target / peak)).sum(-1)
    ref_log_probs = torch.log_softmax(reference_logits.float(), dim=-1)
    kl = (probs * (log_probs - ref_log_probs)).sum(-1)
    loss = (utility_weight * -utility
            + calibration_weight * brier
            + kl_weight * kl).mean()
    return loss, {"utility": float(utility.mean()), "brier": float(brier.mean()),
                   "kl": float(kl.mean())}


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def fit_temperature(model, calib_examples, pad_token, amp_dtype=None):
    """Fit scalar T on the calibration split: minimize CE of softmax(z/T)
    against teacher soft targets. Grid over T in [0.5, 8]."""
    model.eval()
    use_amp = amp_dtype in (torch.float16, torch.bfloat16)
    # batch to avoid a 20GB+ one-shot allocation on the calibration set
    n_cand = max(len(ex["candidate_ids"]) for ex in calib_examples)
    L = max(len(ex["candidate_ids"]) for ex in calib_examples)
    max_len = max(max(len(ids) for ids in ex["candidate_ids"]) for ex in calib_examples)
    bs = max(1, (8 * 1024 * 1024 * 1024) // (n_cand * max_len * 1024 * 4))
    all_logits, order = [], []
    with torch.no_grad(), torch.autocast("cuda", dtype=amp_dtype or torch.float16, enabled=use_amp):
        for s in range(0, len(calib_examples), bs):
            chunk = calib_examples[s:s + bs]
            lg, _ = model(chunk, pad_token)
            all_logits.append(lg.float().cpu())
            order.extend(range(s, s + len(chunk)))
    logits = torch.cat(all_logits, 0)
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
    total = None
    for logits in logits_list:
        probs = torch.softmax(logits.float(), dim=-1)
        k = min(2, probs.shape[-1])
        _, idx = torch.topk(probs, k, dim=-1)
        one_hot = torch.zeros_like(probs).scatter_(1, idx, 1.0)
        frac = one_hot.sum(0) / probs.shape[0]
        term = (frac * probs.mean(0)).sum() * probs.shape[-1]
        total = term if total is None else total + term.to(total.device)
    if total is None:
        return 0.0
    return coef * total / max(len(logits_list), 1)


def fix_gate_orientation(model_dir):
    """self-heal: older upcycle builds saved gate as [H,E]; HF wants [E,H]"""
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
    ap.add_argument("--eval-batch", type=int, default=None)
    ap.add_argument("--eval-subset", type=int, default=0,
                    help="periodic dev size stratified by family; 0 = full dev")
    ap.add_argument("--max-length", type=int, default=2048)
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
    ap.add_argument("--rlcd-steps", type=int, default=0,
                    help="RLCD refinement after SFT (JevForge port: utility + "
                         "Brier calibration + KL anchor); 0 = disabled")
    ap.add_argument("--rlcd-utility-weight", type=float, default=1.0)
    ap.add_argument("--rlcd-calibration-weight", type=float, default=0.5)
    ap.add_argument("--rlcd-kl-weight", type=float, default=0.02)
    ap.add_argument("--entropy-weighting", action="store_true",
                    help="sample train questions proportional to teacher entropy")
    args = ap.parse_args()

    sys.path.insert(0, args.nanojev_scripts)
    from train_toy_decisions import (DecisionModel, benchmark, dump,
                                     evaluate as evaluate_raw, load_examples, loss_for)

    def evaluate(model, examples, pad_token, batch, path=None):
        # group similar-length sequences so padding inside a batch stays small;
        # metrics are order-independent sums, so the sort changes nothing else
        ordered = sorted(examples, key=lambda e: max(map(len, e["leaf_tokens"])))
        return evaluate_raw(model, ordered, pad_token, batch, path)

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
    # master weights stay fp32 for fp16 mode: GradScaler cannot unscale
    # fp16 gradients; autocast handles the forward conversion (standard AMP)
    param_dtype = {"fp16": torch.float32, "bf16": torch.bfloat16,
                   "fp32": torch.float32}[args.dtype]
    fix_gate_orientation(args.model)
    _cfg = AutoConfig.from_pretrained(args.model)
    _mt = getattr(_cfg, "model_type", "")
    lm = None
    if _mt == "qwen3_moe":
        try:
            from transformers import Qwen3MoeForCausalLM as _Moe
            kw = dict(torch_dtype=param_dtype, attn_implementation="sdpa")
            if torch.cuda.device_count() > 1:
                lm = _Moe.from_pretrained(args.model, device_map="auto", **kw)
            else:
                lm = _Moe.from_pretrained(args.model, **kw).cuda()
            print(f"loaded {_Moe.__name__} (explicit MoE class)", flush=True)
        except ImportError:
            print("FATAL: qwen3_moe config but Qwen3MoeForCausalLM missing "
                  "in this transformers build — refusing dense fallback", flush=True)
            raise
    if lm is None:
        if torch.cuda.device_count() > 1:
            lm = AutoModelForCausalLM.from_pretrained(
                args.model, torch_dtype=param_dtype, attn_implementation="sdpa",
                device_map="auto")
        else:
            lm = AutoModelForCausalLM.from_pretrained(
                args.model, torch_dtype=param_dtype, attn_implementation="sdpa").cuda()
        print(f"loaded {type(lm).__name__} via AutoModel (model_type={_mt})", flush=True)
    if torch.cuda.device_count() > 1:
        print(f"device_map=auto across {torch.cuda.device_count()} GPUs", flush=True)
    lm.config.use_cache = False
    if not args.no_grad_checkpoint:
        lm.gradient_checkpointing_enable()
    clean_input = out / "unified_clean.jsonl"
    rows = load_unified(args.input)
    # ascii-escaped: NanoJev load_examples reads via read_text().splitlines(),
    # raw U+2028/U+2029 inside strings would split rows mid-JSON
    clean_input.write_text("".join(json.dumps(r) + "\n"
                                   for r in rows), encoding="utf-8")
    examples, audit = load_examples(str(clean_input), tokenizer, args.max_length)
    dump(out / "target_audit.json", audit)
    bysplit = {s: [e for e in examples if e["split"] == s]
               for s in ["train", "dev", "calibration", "test", "ood"]}
    missing = [s for s, g in bysplit.items() if not g]
    if missing:
        raise ValueError(f"empty splits: {missing}")
    eval_batch = args.eval_batch or args.batch_questions
    dev_full = bysplit["dev"]
    dev_periodic = dev_full
    if args.eval_subset and args.eval_subset < len(dev_full):
        by_family = {}
        for ex in dev_full:
            by_family.setdefault(ex.get("family_id", ""), []).append(ex)
        rng = random.Random(args.seed + 3)
        for group in by_family.values():
            rng.shuffle(group)
        families = sorted(by_family)
        dev_periodic = []
        while len(dev_periodic) < args.eval_subset and any(by_family.values()):
            for family in families:
                if by_family[family] and len(dev_periodic) < args.eval_subset:
                    dev_periodic.append(by_family[family].pop())
    print(f"eval: periodic={len(dev_periodic)} full={len(dev_full)} batch={eval_batch}",
          flush=True)

    model = DecisionModel(lm.model, args.set_head).to(param_dtype)
    if torch.cuda.device_count() > 1:
        # move only the head: a whole-model .to() collapses the device_map=auto
        # sharding and desyncs accelerate per-layer execution devices
        tail_dev = next(model.backbone.layers[-1].parameters()).device
        for name, mod in model.named_modules():
            if name and not name.startswith("backbone"):
                mod.to(tail_dev)
        print(f"decision head on {tail_dev}", flush=True)
    else:
        model.to(next(lm.parameters()).device)
    del lm
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

    warmup_iters = 30
    def _lr_lambda(step):
        if step < warmup_iters:
            return (step + 1) / warmup_iters
        return 1.0
    schedulers = [torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)]
    print(f"lr warmup: {warmup_iters} iters linear 0->1", flush=True)

    total_steps = args.head_steps + args.steps
    logs, best, best_step = [], float("inf"), None
    start = time.perf_counter()
    for step in range(args.head_steps + args.steps):
        warm = step < args.head_steps
        for param in body:
            param.requires_grad_(not warm)
        if warm:
            # head-phase boost only; else LambdaLR's 30-iter warmup would be
            # clobbered for this group every step
            optimizer.param_groups[1]["lr"] = 1e-3
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
            if router_buf:
                # buffered hidden states are graph-less under reentrant checkpointing
                # (initial pass runs no_grad); recompute gate logits on detached
                # inputs so the aux grad reaches gate weights in any mode
                gates = [fn(h.detach()) for fn, h in router_buf]
                mloss = mloss + _load_balance(gates, 0.01).to(mloss.device)
            router_buf.clear()
            if not torch.isfinite(mloss):
                continue
            scaler.scale(mloss).backward()
            router_buf.clear()
            step_loss += mloss.item()
        if router_buf:
            router_buf.clear()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        for s in schedulers:
            s.step()
        if (step + 1) % 50 == 0:
            _e = batch[0]
            _tp = _e.get("teacher_probs") or []
            print(f"[sample] {_e.get('qid', '?')} "
                  f"state={str(_e.get('state', _e.get('text', '')))[:80]!r} "
                  f"teacher={[_round(x, 2) for x in _tp[:5]] if _tp else 'gold-only'}",
                  flush=True)
            for _gi in range(torch.cuda.device_count()):
                print(f"[mem] GPU {_gi}: {torch.cuda.memory_allocated(_gi)/2**30:.2f}G",
                      flush=True)
        item = {"step": step + 1, "phase": "head" if warm else "full",
                "loss": round(step_loss, 4),
                "elapsed_seconds": time.perf_counter() - start}
        if not warm and ((step + 1 - args.head_steps) % args.eval_every == 0
                         or step + 1 == args.head_steps + args.steps):
            # cooldown before eval: release activation memory
            torch.cuda.empty_cache()
            time.sleep(5)
            router_buf.clear()
            try:
                eval_set = (dev_full if step + 1 == args.head_steps + args.steps
                            else dev_periodic)
                metrics = evaluate(model, eval_set, tokenizer.pad_token_id, eval_batch)
                router_buf.clear()
                item["dev"] = metrics
                score = metrics["teacher_ce" if args.objective == "teacher" else "gold_nll"]
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                time.sleep(10)
                router_buf.clear()
                print("[eval] OOM, retrying with halved batch", flush=True)
                try:
                    metrics = evaluate(model, eval_set, tokenizer.pad_token_id,
                                       max(1, eval_batch // 2))
                    router_buf.clear()
                    item["dev"] = metrics
                    score = metrics["teacher_ce" if args.objective == "teacher" else "gold_nll"]
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    time.sleep(15)
                    router_buf.clear()
                    print("[eval] OOM again, skipping this round", flush=True)
                    metrics = None
                    score = None
            if score is not None and score < best:
                best, best_step = score, step + 1
                save_file({k: v.detach().cpu().contiguous().clone()
                           for k, v in model.state_dict().items()},
                          out / "best.safetensors")
            # cooldown after eval before training resumes
            torch.cuda.empty_cache()
            time.sleep(5)
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
                     eval_batch, out / "predictions.jsonl")
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
    # ---- stage 3: RLCD refinement (JevForge port) ----
    if args.rlcd_steps > 0:
        print(f"[rlcd] starting {args.rlcd_steps} steps "
              f"(utility={args.rlcd_utility_weight} "
              f"calib={args.rlcd_calibration_weight} kl={args.rlcd_kl_weight})",
              flush=True)
        # reference = the SFT best checkpoint; cache its logits once per
        # example so no second model copy is needed on GPU
        ref_sd = load_file(out / "best.safetensors")
        model.load_state_dict(ref_sd)
        model.eval()
        ref_cache = []
        with torch.no_grad():
            for s in range(0, len(bysplit["train"]), args.batch_questions):
                group = bysplit["train"][s:s + args.batch_questions]
                lg, _ = model(group, tokenizer.pad_token_id)
                for i, ex in enumerate(group):
                    k = len(ex["candidate_ids"])
                    ref_cache.append(lg[i, :k].float().cpu())
        model.train()
        print(f"[rlcd] reference logits cached for {len(ref_cache)} questions",
              flush=True)

        rlcd_opt = torch.optim.AdamW(
            [{"params": list(model.backbone.parameters()),
              "lr": args.backbone_lr * 0.5},
             {"params": [p for n, p in model.named_parameters()
                         if not n.startswith("backbone.")],
              "lr": args.head_lr * 0.5}], weight_decay=0.01)
        rlcd_logs, rlcd_best, rlcd_best_step = [], float("inf"), None
        rng = random.Random(args.seed + 7)
        t_rlcd = time.perf_counter()
        for rstep in range(args.rlcd_steps):
            idxs = rng.sample(range(len(bysplit["train"])),
                              min(args.batch_questions, len(bysplit["train"])))
            group = [bysplit["train"][i] for i in idxs]
            lg, _ = model(group, tokenizer.pad_token_id)
            losses, diags = [], []
            for i, ex in enumerate(group):
                k = len(ex["candidate_ids"])
                if ex["teacher_probs"] is None:
                    continue
                target = torch.tensor(ex["teacher_probs"][:k], device=lg.device)
                loss, diag = rlcd_loss(lg[i, :k], target,
                                        ref_cache[idxs[i]].to(lg.device),
                                        args.rlcd_utility_weight,
                                        args.rlcd_calibration_weight,
                                        args.rlcd_kl_weight)
                losses.append(loss)
                diags.append(diag)
            if not losses:
                continue
            rloss = torch.stack(losses).mean()
            rlcd_opt.zero_grad(set_to_none=True)
            rloss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            rlcd_opt.step()
            entry = {"rlcd_step": rstep + 1, "rlcd_loss": round(float(rloss), 4),
                     **{k: round(sum(d[k] for d in diags) / len(diags), 4)
                        for k in diags[0]}}
            if (rstep + 1) % args.eval_every == 0 or rstep + 1 == args.rlcd_steps:
                torch.cuda.empty_cache()
                try:
                    m = evaluate(model, bysplit["dev"][:300],
                                 tokenizer.pad_token_id, 2)
                    entry["dev"] = m
                    sc = m["teacher_ce" if args.objective == "teacher"
                            else "gold_nll"]
                    if sc is not None and sc < rlcd_best:
                        rlcd_best, rlcd_best_step = sc, rstep + 1
                        save_file({kk: v.detach().cpu().contiguous().clone()
                                   for kk, v in model.state_dict().items()},
                                  out / "rlcd_best.safetensors")
                except torch.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    print("[rlcd] eval OOM, skipped", flush=True)
            rlcd_logs.append(entry)
            if (rstep + 1) % 25 == 0:
                print(json.dumps(entry), flush=True)
        dump(out / "rlcd_log.json", rlcd_logs)
        if rlcd_best_step:
            model.load_state_dict(load_file(out / "rlcd_best.safetensors"))
            temperature, calib_nll = fit_temperature(
                model, bysplit["calibration"], tokenizer.pad_token_id, amp_dtype)
            dump(out / "temperature.json", {
                "T": temperature, "calib_nll_at_T": calib_nll,
                "stage": "rlcd", "rlcd_best_step": rlcd_best_step,
                "calibration_questions": len(bysplit["calibration"])})
            final = evaluate(model, evaluation, tokenizer.pad_token_id,
                             args.batch_questions, out / "predictions.jsonl")
            dump(out / "summary.json", {
                "best_step": best_step, "rlcd_best_step": rlcd_best_step,
                "temperature": temperature,
                "selected_on": "dev CE after RLCD", "init": init_info,
                "training_seconds": time.perf_counter() - start,
                "rlcd_seconds": time.perf_counter() - t_rlcd,
                "final_all_eval": final,
                "max_gpu_allocated_gb": torch.cuda.max_memory_allocated() / 1e9})
            print(f"[rlcd] DONE best={rlcd_best:.4f} step={rlcd_best_step}",
                  flush=True)
        else:
            print("[rlcd] no dev improvement; keeping SFT checkpoint", flush=True)


if __name__ == "__main__":
    main()
