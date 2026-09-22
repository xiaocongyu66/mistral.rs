import os, subprocess, sys, time, json
os.environ["HF_HOME"] = "/kaggle/temp/hf"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

W = "/kaggle/working"
D = "/kaggle/input/duan-distill-corpus"
RAW = "https://raw.githubusercontent.com/xiaocongyu66/mistral.rs/decide/distill"
PRETRAIN_MINUTES = int(os.environ.get("PRETRAIN_MINUTES", "90"))
NUM_EXPERTS = int(os.environ.get("NUM_EXPERTS", "8"))

# datasets/pyarrow mismatch crashes the parquet streaming reader at GC time
# and core-dumps the process; pin a known-good pair
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "transformers>=4.55,<5", "datasets==3.6.0", "pyarrow==16.1.0",
                "bitsandbytes", "accelerate", "safetensors", "huggingface_hub", "pdfplumber"],
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

def dl(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print("cached", dest, flush=True); return
    print("fetch", url, flush=True)
    urllib.request.urlretrieve(url, dest)

def fetch(name, url):
    """three-way fetch: mounted Kaggle dataset -> raw GitHub -> HF dataset repo"""
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
    p = hf_hub_download("congyu778/duan-distill", name, repo_type="dataset")
    import shutil; shutil.copy(p, f"{W}/{name}")
    print("hf-fetched", name, flush=True)

for name, url in [("upcycle_to_moe.py", f"{RAW}/upcycle_to_moe.py"),
                  ("colab_train.py", f"{RAW}/colab_train.py"),
                  ("train_toy_decisions.py", "https://raw.githubusercontent.com/TianyuCodings/NanoJev/main/scripts/train_toy_decisions.py"),
                  ("stream_sources.py", f"{RAW}/stream_sources.py"),
                  ("unified.jsonl", f"{RAW}/nanojev_format/unified.jsonl")]:
    fetch(name, url)
print("unified rows:", sum(1 for _ in open(f"{W}/unified.jsonl")), flush=True)
print("MODEL_NAME: Apeireth-Decis-2.6B-128K (base: Qwen3-0.6B -> 8-expert MoE, YaRN 128k)", flush=True)

# ---- stage 0: upcycle dense -> MoE ----
moe_dir = f"{W}/moe-{NUM_EXPERTS}e"
if not os.path.exists(f"{moe_dir}/model.safetensors"):
    subprocess.run([sys.executable, f"{W}/upcycle_to_moe.py", "--model", "Qwen/Qwen3-0.6B",
                    "--out", moe_dir, "--num-experts", str(NUM_EXPERTS), "--top-k", "2",
                    "--noise", "1e-3", "--split-strategy", "copy_noise"], check=True)
print("moe dir size GB:", round(sum(os.path.getsize(os.path.join(r, f))
      for r, _, fs in os.walk(moe_dir) for f in fs) / 1e9, 2), flush=True)

# ---- stage 1: streaming warmup in a SUBPROCESS ----
# the HF streaming stack can abort() its own process (parquet reader threads);
# keep that blast radius away from stage 2
WARMUP_SRC = r'''
import os, sys, time, json, gc, subprocess
moe_dir, minutes = sys.argv[1], float(sys.argv[2])
W = "/kaggle/working"
D = "/kaggle/input/duan-distill-corpus"
sys.path.insert(0, W)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import safetensors.torch as _st
_gp = os.path.join(moe_dir, "model.safetensors")
if os.path.exists(_gp):
    _sd = _st.load_file(_gp)
    _g = _sd.get("model.layers.0.mlp.gate.weight")
    if _g is not None and _g.shape[0] > _g.shape[1]:
        for _k in list(_sd):
            if _k.endswith("mlp.gate.weight"):
                _sd[_k] = _sd[_k].t().contiguous()
        _st.save_file(_sd, _gp)
        print("gate orientation self-healed in warmup", flush=True)
tok = AutoTokenizer.from_pretrained(moe_dir)
from transformers import AutoConfig as _AC
_mt = getattr(_AC.from_pretrained(moe_dir), "model_type", "")
from transformers import Qwen3MoeForCausalLM as _Moe
_mm = {i: "13GiB" for i in range(torch.cuda.device_count())}
model = _Moe.from_pretrained(moe_dir, torch_dtype=torch.float32,
                             attn_implementation="sdpa", device_map="auto", max_memory=_mm)
print(f"warmup loaded {_Moe.__name__} (model_type={_mt})", flush=True)
model.config.use_cache = False
model.config.output_router_logits = True
model.gradient_checkpointing_enable()
import bitsandbytes  # noqa
opt = bitsandbytes.optim.AdamW8bit(model.parameters(), lr=1e-5, weight_decay=0.0)
scaler = torch.amp.GradScaler("cuda", enabled=True)

from stream_sources import build_stream

# runtime-fetch MACHIAVELLI game data (50 万决策场景) into /kaggle/tmp
os.makedirs("/kaggle/tmp", exist_ok=True)
if not os.path.exists("/kaggle/tmp/mach_data"):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gdown"], check=False)
    r = subprocess.run(["gdown", "19PXa2bgjkfFfTTI3EZIT3-IJ_vxrV0Rz", "-O", "/kaggle/tmp/mach.zip"],
                       capture_output=True, text=True, timeout=1800)
    if r.returncode == 0 and os.path.exists("/kaggle/tmp/mach.zip"):
        import zipfile
        try:
            with zipfile.ZipFile("/kaggle/tmp/mach.zip") as z:
                z.extractall("/kaggle/tmp/mach_data", pwd=b"machiavelli")
            os.remove("/kaggle/tmp/mach.zip")
            print("MACHIAVELLI data ready at /kaggle/tmp/mach_data", flush=True)
        except Exception as e:
            print(f"[mach] unzip failed: {e}", flush=True)
    else:
        print(f"[mach] gdown rc={r.returncode} (non-fatal, skip)", flush=True)
def local_corpus():
    paths = []
    for name in ("book_corpus.jsonl", "code_corpus.jsonl"):
        for cand in (os.path.join(D, name), f"{W}/{name}"):
            if os.path.exists(cand):
                paths.append(cand); break
    if not paths:
        print("WARN: no local corpus found", flush=True); return
    print("local corpus:", paths, flush=True)
    while True:
        for p in paths:
            with open(p, encoding="utf-8") as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    t = (r.get("text") or "").strip()
                    if len(t) >= 100:
                        yield t

def mixed():
    import random as _r
    rng = _r.Random(17)
    loc, ext = local_corpus(), build_stream(tok)
    while True:
        if rng.random() < 0.7:
            try:
                yield next(loc); continue
            except StopIteration:
                pass
        try:
            yield next(ext)
        except StopIteration:
            try:
                yield next(loc)
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
        enc = tok(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
        opt.zero_grad(set_to_none=True)
        out = model(**enc, labels=enc["input_ids"], output_router_logits=True)
        loss = out.loss + 0.01 * (out.aux_loss if getattr(out, "aux_loss", None) is not None else 0.0)
        scaler.scale(loss).backward()
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        step += 1; tok_seen += enc["input_ids"].numel()
        if step % 20 == 0:
            print(f"[warmup] step={step} loss={loss.item():.4f} tokens={tok_seen} "
                  f"min={((time.time()-t0)/60):.1f}", flush=True)
except Exception as e:
    print(f"[warmup] aborted by: {type(e).__name__}: {str(e)[:300]}", flush=True)
print(f"[warmup] done steps={step} tokens={tok_seen}", flush=True)
del stream
gc.collect()
model.save_pretrained(f"{W}/moe-warmup")
tok.save_pretrained(f"{W}/moe-warmup")
print("[warmup] saved", flush=True)
'''
with open(f"{W}/warmup_main.py", "w") as f:
    f.write(WARMUP_SRC)

r = subprocess.run([sys.executable, f"{W}/warmup_main.py", moe_dir, str(PRETRAIN_MINUTES)])
warm_ok = os.path.exists(f"{W}/moe-warmup/model.safetensors") and \
          os.path.getsize(f"{W}/moe-warmup/model.safetensors") > 1e6
print(f"warmup subprocess rc={r.returncode} warm_ok={warm_ok}", flush=True)
stage2_model = f"{W}/moe-warmup" if warm_ok else moe_dir
if not warm_ok:
    print(f"WARN: falling back to un-warmed {moe_dir}", flush=True)
import torch
torch.cuda.empty_cache()

# ---- stage 2: decision finetune ----
subprocess.run([sys.executable, f"{W}/colab_train.py",
                "--nanojev-scripts", W,
                "--input", f"{W}/unified.jsonl",
                "--output-dir", f"{W}/apeireth-decis-2.6b-128k",
                "--model", stage2_model,
                "--steps", "600", "--head-steps", "24", "--eval-every", "50",
                "--batch-questions", "4",
                "--temperature", "5.0", "--temperature-final", "1.5",
                "--entropy-weighting", "--dtype", "fp32", "--adam8bit"], check=True)
print("ALL_DONE", flush=True)
