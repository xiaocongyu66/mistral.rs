import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", nargs="+", required=True, help="teacher-*.jsonl paths")
    p.add_argument("--model", default="Qwen/Qwen3.5-4B")
    p.add_argument("--out", default="checkpoints/student")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--temperature", type=float, default=3.0, help="KD temperature T")
    p.add_argument("--alpha", type=float, default=0.7, help="weight of KL vs CE")
    p.add_argument("--conf-threshold", type=float, default=0.9,
                   help="teacher confidence above this -> hard-label dominant sample")
    p.add_argument("--max-len", type=int, default=1024)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


class DecisionDataset(Dataset):
    """Each row: state + instructions -> prompt; candidate labels -> token ids;
    teacher scores -> softened target distribution."""

    def __init__(self, rows, tokenizer, max_len: int):
        self.samples = []
        for r in rows:
            labels = r["labels"]
            prompt = f"{r['instructions']}\n\nState: {r['state']}\nAnswer label:"
            enc = tokenizer(prompt, truncation=True, max_length=max_len, return_tensors="pt")
            ids = []
            for lab in labels:
                tok = tokenizer(lab, add_special_tokens=False)["input_ids"]
                if not tok:
                    raise ValueError(f"label {lab!r} tokenizes to nothing")
                ids.append(tok[0])
            if len(set(ids)) != len(ids):
                # first-token collision: fall back to full-sequence scoring later
                raise ValueError(f"first-token collision among {labels}: {ids}")
            scores = r["scores"]
            target = torch.tensor([float(scores[lab]) for lab in labels])
            target = target / target.sum().clamp_min(1e-9)
            self.samples.append({
                "input_ids": enc["input_ids"][0],
                "attention_mask": enc["attention_mask"][0],
                "candidate_ids": torch.tensor(ids, dtype=torch.long),
                "target": target,
                "confidence": float(r["confidence"] or 0.0),
                "label_idx": max(range(len(labels)), key=lambda i: float(scores[labels[i]])),
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


def collate(batch, pad_id: int):
    n = max(len(b["input_ids"]) for b in batch)
    input_ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    attn = torch.zeros((len(batch), n), dtype=torch.long)
    for i, b in enumerate(batch):
        L = len(b["input_ids"])
        input_ids[i, :L] = b["input_ids"]
        attn[i, :L] = b["attention_mask"]
    return {
        "input_ids": input_ids,
        "attention_mask": attn,
        "candidate_ids": torch.stack([b["candidate_ids"] for b in batch]),
        "target": torch.stack([b["target"] for b in batch]),
        "confidence": torch.tensor([b["confidence"] for b in batch]),
        "label_idx": torch.tensor([b["label_idx"] for b in batch]),
    }


def student_candidate_logits(model, input_ids, attention_mask, candidate_ids):
    """One forward pass; gather logits at the last non-pad position for each
    candidate first-token id. Reuses the same readout the runtime uses."""
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    last = attention_mask.sum(dim=1) - 1
    pos_logits = out.logits[torch.arange(len(input_ids)), last]
    gather = candidate_ids.to(pos_logits.device)
    return pos_logits.gather(1, gather)


def temperize(target: torch.Tensor, T: float) -> torch.Tensor:
    """Black-box teacher stored at T=1: re-soften with p^(1/T), renormalized."""
    return F.softmax(target.clamp_min(1e-9).log() / T, dim=-1)


def kd_loss(student_logits, target, confidence, label_idx, T, alpha, conf_threshold):
    q = temperize(target.to(student_logits.device), T)
    log_p = F.log_softmax(student_logits / T, dim=-1)
    per_sample_kl = F.kl_div(log_p, q, reduction="none").sum(-1) * (T * T)
    per_sample_ce = F.cross_entropy(student_logits, label_idx.to(student_logits.device),
                                    reduction="none")
    # hard labels dominate high-confidence samples; soft targets carry the rest
    a = torch.where(confidence.to(student_logits.device) >= conf_threshold,
                    torch.full_like(per_sample_kl, 1.0 - alpha),
                    torch.full_like(per_sample_kl, alpha))
    return (a * per_sample_kl + (1 - a) * per_sample_ce).mean()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    rows = []
    for path in args.data:
        rows += [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    random.shuffle(rows)
    ds = DecisionDataset(rows, tokenizer, args.max_len)
    print(f"dataset: {len(ds)} samples")

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto")
    model.config.use_cache = False
    try:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM"))
        model.print_trainable_parameters()
    except ImportError:
        print("peft not installed: full finetune")

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    collate_fn=lambda b: collate(b, pad_id))
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    step = 0
    for epoch in range(args.epochs):
        for batch in dl:
            logits = student_candidate_logits(
                model, batch["input_ids"].to(model.device),
                batch["attention_mask"].to(model.device),
                batch["candidate_ids"])
            loss = kd_loss(logits, batch["target"], batch["confidence"],
                           batch["label_idx"], args.temperature, args.alpha,
                           args.conf_threshold)
            loss.backward()
            opt.step()
            opt.zero_grad()
            step += 1
            if step % 20 == 0:
                print(f"epoch {epoch} step {step} loss {loss.item():.4f}")
        model.save_pretrained(out / f"epoch{epoch}")

    print(f"done -> {out}")


if __name__ == "__main__":
    main()
