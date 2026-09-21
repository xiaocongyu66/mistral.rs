"""Registry of every dataset named in the resource lists.

Batch-verifies existence against the HF API and classifies each entry by
domain + training stage. The stream loader and seed queue draw from here, so
adding a dataset is one line instead of a manual pipeline edit.
"""
import json
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

# (hf_id, domain, stage, note)
REGISTRY = [
    # ---- chinese general / instruction ----
    ("Mxode/Chinese-Instruct", "chinese", "sft", "dpsk-r1-distil + reasoning configs"),
    ("Mxode/Chinese-Reasoning-Distil-Data", "chinese", "sft", "reasoning distill"),
    ("Mxode/Meow-Reasoning-100K", "chinese", "sft", "100k reasoning"),
    ("Chinese-Data-Distill-From-R1", "chinese", "sft", "110k w/ math-verify"),
    ("MegaScience/Chinese-Reasoning-Dataset-v1", "chinese", "sft", "10k verified math"),
    ("openbmb/UltraData-Math", "math", "sft", "curated math"),
    ("hsc-wuxia-200k", "chinese", "pretrain", "wuxia style"),
    ("webnovel-chinese", "chinese", "pretrain", "9B tokens novels"),
    ("LightNovel5000", "chinese", "pretrain", "light novels"),
    ("CAPC-CG", "chinese", "sft", "policy instructions"),
    # ---- decision / reasoning ----
    ("tasksource/100k-choice-dilemmas", "decision", "sft", "real life dilemmas"),
    ("SamSJackson/Newcomb-like-Questions", "decision", "sft", "newcomb problems"),
    ("KRAFTON/Orak", "decision", "sft", "expert video-game traces"),
    ("Max-Suslov/game-bench", "decision", "sft", "slay-the-spire"),
    ("rushes", "decision", "sft", "44k human choice events"),
    ("open-thoughts/OpenThoughts2-1M", "reasoning", "sft", "ICLR 2026"),
    ("bespokelabs/Bespoke-Stratos-17k", "reasoning", "sft", "curated reasoning"),
    ("nvidia/OpenMathReasoning", "math", "sft", "3.2M AoPS"),
    ("nvidia/OpenMathInstruct-2", "math", "sft", "14M pairs"),
    ("nvidia/Nemotron-Math", "math", "sft", "7.5M traces"),
    ("AI-MO/NuminaMath-CoT", "math", "sft", "860k cot"),
    ("TIGER-Lab/MathInstruct", "math", "sft", "chain-of-thought math"),
    ("open-r1/OpenR1-Math-220k", "math", "sft", "220k verified"),
    # ---- code ----
    ("nvidia/OpenCodeReasoning", "code", "sft", "736k python"),
    ("nvidia/SWE-Zero2Hero", "code", "sft", "two-stage swe"),
    ("SWE-bench/SWE-bench_Multilingual", "code", "eval", "multilingual swe"),
    ("openai/openai_humaneval", "code", "eval", "humaneval"),
    ("codeparrot/codeparrot-clean", "code", "pretrain", "python"),
    ("bigcode/the-stack-smol", "code", "pretrain", "multilingual code"),
    # ---- multilingual ----
    ("CohereForAI/aya_dataset", "multilingual", "sft", "204k 65 langs"),
    ("MBZUAI/Bactrian-X", "multilingual", "sft", "52 langs 3.4M"),
    ("HuggingFaceFW/finewiki", "multilingual", "pretrain", "325 langs wiki"),
    ("Helsinki-NLP/opus_books", "multilingual", "pretrain", "parallel books"),
    ("sedthh/gutenberg_multilang", "multilingual", "pretrain", "7.9k books"),
    # ---- english/general pretrain ----
    ("HuggingFaceTB/cosmopedia", "pretrain", "pretrain", "25B synth textbooks"),
    ("HuggingFaceFW/fineweb-edu", "pretrain", "pretrain", "curated web"),
    ("HuggingFaceTB/finemath", "math", "pretrain", "math web"),
    ("wikimedia/wikipedia", "pretrain", "pretrain", "wiki multi"),
    # ---- domain ----
    ("bitext/Bitext-retail-banking-llm-chatbot-training-dataset", "domain", "sft", "banking intents"),
    ("PolyAI/banking77", "domain", "sft", "77 intents"),
    ("clinc/clinc_oos", "domain", "sft", "oos detection"),
    ("osunlp/Adaptive-Clinical-Decision", "domain", "sft", "clinical"),
    ("FinGPT/fingpt-sentiment-train", "domain", "sft", "finance"),
    ("nguha/legalbench", "domain", "sft", "legal"),
]


def probe(item):
    hf_id, domain, stage, note = item
    url = "https://huggingface.co/api/datasets/" + urllib.parse.quote(hf_id, safe="/")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "duan-registry"})
        with urllib.request.urlopen(req, timeout=25) as r:
            d = json.load(r)
        return {"id": hf_id, "domain": domain, "stage": stage, "note": note, "ok": True,
                "downloads": d.get("downloads", 0), "gated": d.get("gated", False)}
    except Exception as e:
        return {"id": hf_id, "domain": domain, "stage": stage, "note": note, "ok": False,
                "err": str(e)[:80]}


def main():
    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(probe, REGISTRY))
    ok = [r for r in results if r["ok"]]
    bad = [r for r in results if not r["ok"]]
    print(f"verified {len(ok)}/{len(results)} datasets exist")
    for r in bad:
        print(f"  MISSING {r['id']} ({r['err']})")
    by_domain = {}
    for r in ok:
        by_domain.setdefault(r["domain"], []).append(r["id"])
    for dom, ids in sorted(by_domain.items()):
        print(f"  {dom:12s} {len(ids)}: {', '.join(ids[:4])}{'...' if len(ids) > 4 else ''}")
    json.dump(results, open("resource_registry.json", "w"), ensure_ascii=False, indent=1)
    print("-> resource_registry.json")


if __name__ == "__main__":
    main()
