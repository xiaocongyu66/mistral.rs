"""TPU kernel wrapper: upcycle + warmup + decision finetune on Kaggle TPU v3-8."""
import os, subprocess, sys, time, json
os.environ["HF_HOME"] = "/kaggle/temp/hf"
os.environ["XLA_USE_BF16"] = "0"  # we control dtype ourselves

W = "/kaggle/working"
D = "/kaggle/input/duan-distill-corpus"
RAW = "https://raw.githubusercontent.com/xiaocongyu66/mistral.rs/decide/distill"
PRETRAIN_MINUTES = 90
NUM_EXPERTS = 8

# install: torch_xla pre-installed on Kaggle TPU images
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "transformers>=4.55,<5", "datasets==3.6.0", "pyarrow==16.1.0",
                "accelerate", "safetensors", "huggingface_hub", "pdfplumber"],
               check=False)

import urllib.request
def _has_net():
    try:
        urllib.request.urlopen("https://huggingface.co", timeout=15)
        return True
    except Exception:
        return False

NET = _has_net()
print("network:", NET, flush=True)
print("TPU available:", flush=True)
# verify XLA
import torch
import torch_xla.core.xla_model as xm
dev = xm.xla_device()
print(f"XLA device: {dev}", flush=True)

def dl(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print("cached", dest, flush=True); return
    print("fetch", url, flush=True)
    urllib.request.urlretrieve(url, dest)

def fetch(name, url):
    src = os.path.join(D, name)
    if os.path.exists(src):
        import shutil; shutil.copy(src, f"{W}/{name}"); print("mounted", name, flush=True)
        return
    try:
        dl(url, f"{W}/{name}")
        return
    except Exception as e:
        print(f"[fetch] raw failed for {name}: {e}", flush=True)
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("conyu778/duan-distill", name, repo_type="dataset")
    import shutil; shutil.copy(p, f"{W}/{name}")
    print("hf-fetched", name, flush=True)

for name, url in [("upcycle_to_moe.py", f"{RAW}/upcycle_to_moe.py"),
                  ("colab_train_tpu.py", f"{RAW}/colab_train_tpu.py"),
                  ("train_toy_decisions.py", "https://raw.githubusercontent.com/TianyuCodings/NanoJev/main/scripts/train_toy_decisions.py"),
                  ("stream_sources.py", f"{RAW}/stream_sources.py"),
                  ("unified.jsonl", f"{RAW}/nanojev_format/unified.jsonl")]:
    fetch(name, url)
print("unified rows:", sum(1 for _ in open(f"{W}/unified.jsonl")), flush=True)
print("MODEL_NAME: Apeireth-Decis-2.6B-128K (TPU v3-8)", flush=True)

# ---- stage 0: upcycle ----
moe_dir = f"{W}/moe-{NUM_EXPERTS}e"
if not os.path.exists(f"{moe_dir}/model.safetensors"):
    subprocess.run([sys.executable, f"{W}/upcycle_to_moe.py",
                    "--model", "Qwen/Qwen3-0.6B",
                    "--out", moe_dir,
                    "--num-experts", str(NUM_EXPERTS),
                    "--top-k", "2",
                    "--split-strategy", "importance"], check=True)
print("moe dir size GB:", round(sum(os.path.getsize(os.path.join(r, f))
      for r, _, fs in os.walk(moe_dir) for f in fs) / 1e9, 2), flush=True)

# ---- stage 1: streaming warmup in a SUBPROCESS (XLA) ----
WARMUP_SRC = r'''
import os, sys, time, json, gc, subprocess
moe_dir, minutes = sys.argv[1], float(sys.argv[2])
W = "/kaggle/working"
sys.path.insert(0, W)
import torch
import torch_xla.core.xla_model as xm
import torch_xla.amp as xa
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(moe_dir)
dev = xm.xla_device()
print(f"warmup XLA device: {dev}", flush=True)
from transformers import Qwen3MoeForCausalLM as _Moe
model = _Moe.from_pretrained(moe_dir, torch_dtype=torch.float32,
                             attn_implementation="sdpa").to(dev)
model.config.use_cache = False
model.config.output_router_logits = True
model.gradient_checkpointing_enable()
# full AdamW (TPU 96GB HBM, no need for 8-bit)
opt = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.0)
from stream_sources import build_stream
def mixed():
    import random as _r
    rng = _r.Random(17)
    ext = build_stream(tok)
    while True:
        try:
            yield next(ext)
        except StopIteration:
            return
stream = mixed()
t0, step, tok_seen = time.time(), 0, 0
model.train()
try:
    for raw in stream:
        if (time.time() - t0) / 60 > minutes:
            break
        text = raw.strip()
        if len(text) < 200:
            continue
        enc = tok(text, return_tensors="pt", truncation=True, max_length=512).to(dev)
        out = model(**enc, labels=enc["input_ids"], output_router_logits=True)
        loss = out.loss + 0.01 * (out.aux_loss if getattr(out, "aux_loss", None) is not None else 0.0)
        loss.backward()
        xm.optimizer_step(opt)
        opt.zero_grad(set_to_none=True)
        step += 1; tok_seen += enc["input_ids"].numel()
        if step % 20 == 0:
            print(f"[warmup] step={step} loss={loss.item():.4f} tokens={tok_seen} "
                  f"min={((time.time()-t0)/60):.1f}", flush=True)
except Exception as e:
    print(f"[warmup] aborted by: {type(e).__name__}: {str(e)[:300]}", flush=True)
print(f"[warmup] done steps={step} tokens={tok_seen}", flush=True)
del stream; gc.collect()
model.save_pretrained(f"{W}/moe-warmup")
tok.save_pretrained(f"{W}/moe-warmup")
print("[warmup] saved", flush=True)
'''
with open(f"{W}/warmup_tpu.py", "w") as f:
    f.write(WARMUP_SRC)

r = subprocess.run([sys.executable, f"{W}/warmup_tpu.py", moe_dir, str(PRETRAIN_MINUTES)])
warm_ok = os.path.exists(f"{W}/moe-warmup/model.safetensors") and \
          os.path.getsize(f"{W}/moe-warmup/model.safetensors") > 1e6
print(f"warmup subprocess rc={r.returncode} warm_ok={warm_ok}", flush=True)
stage2_model = f"{W}/moe-warmup" if warm_ok else moe_dir
if not warm_ok:
    print(f"WARN: falling back to un-warmed {moe_dir}", flush=True)

# ---- stage 2: decision finetune (TPU) ----
subprocess.run([sys.executable, f"{W}/colab_train_tpu.py",
                "--nanojev-scripts", W,
                "--input", f"{W}/unified.jsonl",
                "--output-dir", f"{W}/apeireth-decis-2.6b-128k",
                "--model", stage2_model,
                "--steps", "600", "--head-steps", "24", "--eval-every", "50",
                "--batch-questions", "8",
                "--temperature", "5.0", "--temperature-final", "1.5",
                "--entropy-weighting", "--dtype", "fp32"], check=True)
print("ALL_DONE_TPU", flush=True)
