"""Multi-source streaming corpus for MoE warmup.

Each source declares its own text field; `extract_text` falls back through
common aliases so heterogeneous schemas (text / content / raw_content) all
normalize to one stream. Weights drive interleaving so the router sees a
mixed-domain distribution instead of one corpus.
"""

SOURCES = [
    # (repo, config, split, preferred field, weight, note)
    # -- comprehension capacity: the backbone must UNDERSTAND before it can decide --
    ("HuggingFaceFW/finewiki", None, "train", "text", 0.16, "325-lang wiki (clean)"),
    ("wikimedia/wikipedia", "20231101.zh", "train", "text", 0.14, "chinese wiki"),
    ("HuggingFaceTB/cosmopedia", "web_v1", "train", "text", 0.12, "synthetic textbooks"),
    ("HuggingFaceFW/fineweb-edu", "sample-10BT", "train", "text", 0.12, "curated web"),
    ("Helsinki-NLP/opus_books", "en-zh", "train", "translation", 0.06, "parallel books zh/en"),
    ("wikimedia/wikipedia", "20231101.en", "train", "text", 0.06, "english wiki"),
    ("sedthh/gutenberg_multilang", None, "train", "text", 0.04, "multilingual books"),
    # -- reasoning / math --
    ("HuggingFaceTB/finemath", "finemath-4plus", "train", "text", 0.08, "math web"),
    ("AI-MO/NuminaMath-CoT", None, "train", "problem", 0.03, "competition math"),
    # -- code --
    ("codeparrot/codeparrot-clean", None, "train", "content", 0.08, "python code"),
    ("bigcode/the-stack-smol", None, "train", "content", 0.03, "multilingual code"),
    # -- open web --
    ("HuggingFaceFW/fineweb", "sample-10BT", "train", "text", 0.08, "open web"),
    # -- comprehension: books / wiki / long-form (user batch, verified in registry) --
    ("HuggingFaceTB/cosmopedia", "stories", "train", "text", 0.04, "synthetic stories"),
    ("HuggingFaceTB/cosmopedia", "stanford", "train", "text", 0.03, "synthetic textbooks"),
    ("CohereForAI/aya_collection", "aya_eng", "train", "inputs", 0.03, "multilingual instructions"),
    ("open-thoughts/OpenThoughts2-1M", None, "train", "conversations", 0.03, "reasoning traces"),
    ("nvidia/OpenCodeReasoning", None, "train", "input", 0.03, "code reasoning"),
    ("TIGER-Lab/MathInstruct", None, "train", "instruction", 0.02, "math cot"),
    ("nguha/legalbench", None, "train", "text", 0.02, "legal reasoning"),
    ("PolyAI/banking77", None, "train", "text", 0.02, "banking intents"),
    ("clinc/clinc_oos", "plus", "train", "text", 0.02, "oos utterances"),
    # -- P0 verified from search batch (COIG-CQIA / Nemotron / OpenHermes / UltraFeedback) --
    ("m-a-p/COIG-CQIA", None, "train", "instruction", 0.03, "chinese human-verified"),
    ("nvidia/Nemotron-SFT-Agentic-v2", None, "train", "messages", 0.03, "agentic w/ thinking toggle"),
    ("nvidia/Nemotron-SFT-Multilingual-v1", None, "train", "messages", 0.03, "multilingual reasoning"),
    ("teknium/OpenHermes-2.5", None, "train", "conversations", 0.03, "1M high-quality general"),
    ("openbmb/UltraFeedback", None, "train", "prompt", 0.02, "preference calibration"),
    # -- user-verified registry batch: chinese long-form / decision / math-scale --
    ("MegaScience/Chinese-Reasoning-Dataset-v1", None, "train", None, 0.03, "chinese reasoning"),
    ("Mxode/Meow-Reasoning-100K", None, "train", None, 0.02, "chinese cot 100k"),
    ("nvidia/OpenMathInstruct-2", None, "train", "problem", 0.02, "14M math scale"),
    ("open-r1/OpenR1-Math-220k", None, "train", "problem", 0.02, "r1 math traces"),
    ("bespokelabs/Bespoke-Stratos-17k", None, "train", None, 0.02, "stratos reasoning"),
    ("MBZUAI/Bactrian-X", "en", "train", "input", 0.02, "52-lang instructions"),
    ("tasksource/100k-choice-dilemmas", None, "train", None, 0.02, "choice dilemmas"),
    # -- decision-domain direct hits (registry batch) --
    ("KRAFTON/Orak", None, "train", None, 0.02, "game decision corpus"),
    # -- chinese long-form fiction (registry batch) --
    ("hsilvosa/hsc-wuxia-200k", None, "train", None, 0.02, "wuxia 200k"),
    ("wdndev/webnovel-chinese", None, "train", None, 0.02, "chinese webnovels"),
    ("bh2821/LightNovel5000", None, "train", None, 0.02, "light novels"),
    # -- long-context 128K (YaRN RoPE scaling + continual pretrain) --
    ("caskcsg/entropylong_128k", None, "train", "text", 0.03, "EntropyLong 128K (ICLR 2026, FineWeb-Edu+Cosmopedia, dep-verified)"),
    ("allenai/Mix-Context-Post-Training-128K", None, "train", "text", 0.02, "Mix-Context 128K packed (FineWeb-Edu+RedPajama, 64-200K)"),
    # -- general web for coverage --
    ("openwebtext", None, "train", "text", 0.04, "OpenWebText (general web, ~38GB)"),
    # -- long-form books & fiction (natural long sequences) --
    ("sedthh/gutenberg_multilang", None, "train", "text", 0.04, "gutenberg multilang books"),
    ("wikimedia/wikipedia", "20231101.en", "train", "text", 0.06, "english wiki (long articles)"),
    # -- reasoning traces (naturally long chains) --
    ("open-thoughts/OpenThoughts2-1M", None, "train", "conversations", 0.03, "reasoning traces (long CoT)"),
    ("teknium/OpenHermes-2.5", None, "train", "conversations", 0.03, "1M general (some long)"),
    # -- code (long files = long sequences) --
    ("codeparrot/codeparrot-clean", None, "train", "content", 0.08, "python code (long files)"),
    # -- additional 128K sources (verified HF API) --
    ("HuggingFaceFW/fineweb-2", "eng_Latn", "train", "text", 0.04, "FineWeb-2 English (massive web)"),
    ("HuggingFaceFW/fineweb-2", "zho_Hans", "train", "text", 0.04, "FineWeb-2 Chinese (massive web)"),
]

# opus_books yields dict translations, not a plain string field
PARALLEL = {"Helsinki-NLP/opus_books"}

TRUST_REMOTE_CODE = {"PolyAI/banking77", "clinc/clinc_oos", "nguha/legalbench"}

# config/split corrections from the v9 kernel run's SKIP log
SOURCE_FIXES = {
    "nvidia/Nemotron-SFT-Agentic-v2": {"cfg": None, "split": "search"},
    "nvidia/Nemotron-SFT-Multilingual-v1": {"cfg": None, "split": "math_zh"},
    "nvidia/OpenCodeReasoning": {"cfg": "split_0", "split": "train"},
    "m-a-p/COIG-CQIA": {"cfg": "coig_pc", "split": "train"},
    "KRAFTON/Orak": {"cfg": "ace_attorney", "split": "train"},
}

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
        fx = SOURCE_FIXES.get(repo)
        if fx:
            cfg = fx.get("cfg", cfg)
            split = fx.get("split", split)
        try:
            # trust all hub sources: streaming iter() lazily executes
            # loader code long after load_dataset returns
            ds = load_dataset(repo, cfg, split=split, streaming=True,
                              trust_remote_code=True)
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
