"""Multi-source streaming corpus for MoE warmup.

Each source declares its own text field; `extract_text` falls back through
common aliases so heterogeneous schemas (text / content / raw_content) all
normalize to one stream. Weights drive interleaving so the router sees a
mixed-domain distribution instead of one corpus.
"""

SOURCES = [
    # (repo, config, split, preferred field, weight, note)
    ("HuggingFaceTB/cosmopedia", "web_v1", "train", "text", 0.20, "synthetic textbooks"),
    ("HuggingFaceFW/fineweb-edu", "sample-10BT", "train", "text", 0.20, "curated web"),
    ("wikimedia/wikipedia", "20231101.zh", "train", "text", 0.20, "chinese wiki"),
    ("wikimedia/wikipedia", "20231101.en", "train", "text", 0.10, "english wiki"),
    ("HuggingFaceTB/finemath", "finemath-4plus", "train", "text", 0.10, "math web"),
    ("codeparrot/codeparrot-clean", None, "train", "content", 0.10, "python code"),
    ("HuggingFaceFW/fineweb", "sample-10BT", "train", "text", 0.10, "open web"),
]

ALIASES = ("text", "content", "raw_content", "raw", "body", "document", "markdown")


def extract_text(row, preferred=None):
    if preferred and isinstance(row.get(preferred), str):
        return row[preferred]
    for k in ALIASES:
        v = row.get(k)
        if isinstance(v, str) and len(v) > 50:
            return v
    return ""


def build_stream(tokenizer, seed=17, max_sources=None):
    """Yield text from weighted-round-robin over streaming sources."""
    from datasets import load_dataset
    import itertools, random
    streams = []
    for repo, cfg, split, field, w, note in SOURCES[: max_sources or len(SOURCES)]:
        try:
            ds = load_dataset(repo, cfg, split=split, streaming=True)
            streams.append({"it": iter(ds), "field": field, "w": w, "note": note, "repo": repo})
            print(f"[stream] {repo}/{cfg} w={w} ({note})", flush=True)
        except Exception as e:
            print(f"[stream] SKIP {repo}/{cfg}: {str(e)[:120]}", flush=True)
    if not streams:
        return
    total_w = sum(s["w"] for s in streams)
    rng = random.Random(seed)
    while True:
        r = rng.random() * total_w
        acc = 0.0
        for s in streams:
            acc += s["w"]
            if r <= acc:
                try:
                    row = next(s["it"])
                except StopIteration:
                    continue
                txt = extract_text(row, s["field"])
                if len(txt) >= 200:
                    yield txt
                break


if __name__ == "__main__":
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    n = 0
    for t in build_stream(tok):
        n += 1
        if n <= 3:
            print(f"--- sample {n} ({len(t)} chars) ---")
            print(t[:180].replace("\n", " "))
        if n >= 5:
            break
    print(f"streamed {n} samples OK")
