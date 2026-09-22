"""Multimodal decision training: image state + jev teacher labels.

Two-stage bridge:
  1. For each image, generate a text description (BLIP or caption)
  2. jev labels the description → soft distribution over candidates
  3. Student sees the IMAGE (not the description) and learns to match jev's
     distribution — this forces visual understanding.

Usage:
  python train_mm.py --checkpoint <apeireth_dir> --data mm_data.jsonl --steps 200
"""
import argparse
import json
import os
import sys
import time

import torch
from torch.utils.data import Dataset, DataLoader
from safetensors.torch import load_file, save_file

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class MultimodalDecisionDataset(Dataset):
    """Each row: {image_path, state_text, question, candidates, soft_label}"""

    def __init__(self, data_path, tokenizer, max_length=512):
        self.rows = []
        for line in open(data_path, encoding="utf-8"):
            if line.strip():
                self.rows.append(json.loads(line))
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        # tokenize text state + question
        text = f"State: {r['state_text']}\nQuestion: {r['question']}\n"
        for i, c in enumerate(r["candidates"]):
            text += f"  {chr(65+i)}: {c}\n"
        enc = self.tok(text, truncation=True, max_length=self.max_length,
                       return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "image_path": r["image_path"],
            "soft_label": r.get("soft_label", {}),
            "candidates": r["candidates"],
        }


def train_mm(args):
    from transformers import AutoTokenizer
    from PIL import Image
    import io

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.checkpoint)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # load the text decision model
    from train_toy_decisions import DecisionModel
    backbone = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, torch_dtype=torch.float32).to(device)
    text_model = DecisionModel(backbone.model, "attention").to(device)

    # wrap with multimodal
    from mm_decision_model import MultimodalDecisionModel
    model = MultimodalDecisionModel(
        text_model, backbone.config.hidden_size).to(device)
    model.train()

    dataset = MultimodalDecisionDataset(args.data, tok)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr)
    scaler = torch.amp.GradScaler("cuda")

    step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        for batch in loader:
            # load images
            pixel_values = []
            for img_path in batch["image_path"]:
                img = Image.open(img_path).convert("RGB")
                from transformers import CLIPImageProcessor
                proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16")
                pv = proc(img, return_tensors="pt")["pixel_values"]
                pixel_values.append(pv)
            pixel_values = torch.cat(pixel_values).to(device)

            input_ids = torch.nn.utils.rnn.pad_sequence(
                batch["input_ids"], batch_first=True, padding_value=tok.pad_token_id
            ).to(device)
            attention_mask = torch.nn.utils.rnn.pad_sequence(
                batch["attention_mask"], batch_first=True, padding_value=0
            ).to(device)

            with torch.autocast("cuda", dtype=torch.float16):
                logits, losses = model(input_ids=input_ids,
                                       attention_mask=attention_mask,
                                       pixel_values=pixel_values)

                # CE against soft labels
                # logits: [B, num_candidates] or [B*K] depending on DecisionModel
                # for simplicity, assume DecisionModel returns [B, max_cands]
                target = torch.zeros_like(logits)
                for i, sl in enumerate(batch["soft_label"]):
                    for j, c in enumerate(batch["candidates"][i]):
                        if c in sl:
                            target[i, j] = sl[c]
                ce = -(target.clamp_min(1e-9) * logits.log_softmax(-1)).sum(-1).mean()
                total_loss = ce + 0.1 * losses.get("swd", torch.tensor(0.0, device=device))

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step % 20 == 0:
                print(f"[mm] step={step} ce={ce.item():.4f} "
                      f"swd={losses.get('swd', 0):.4f} "
                      f"gate={losses.get('visual_gate', 0):.3f} "
                      f"min={((time.time()-t0)/60):.1f}", flush=True)
            if step >= args.steps:
                break
        if step >= args.steps:
            break

    # save
    os.makedirs(args.out, exist_ok=True)
    save_file({k: v.cpu().contiguous() for k, v in model.state_dict().items()},
              os.path.join(args.out, "mm_model.safetensors"))
    print(f"saved {step} steps -> {args.out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="mm_output")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=3)
    args = ap.parse_args()
    train_mm(args)
