"""Sample seed states from the materials bank via HTTP Range requests.

We only need the first ~20MB of each train file to extract a few hundred
seed states -- no full 9GB download. Each dataset gets a parser mapping its
row schema to {"id","text","tag"}.
"""
import json
import random
import urllib.request
from pathlib import Path

random.seed(7)
OUT = Path(__file__).parent / "seeds_materials.jsonl"
PER_SOURCE = 400
MAX_CHARS = 850
RANGE_BYTES = 25_000_000

SOURCES = [
    ("belle35m", "https://huggingface.co/datasets/BelleGroup/train_3.5M_CN/resolve/main/train_3.5M_CN.jsonl", "belle"),
    ("firefly", "https://huggingface.co/datasets/YeungNLP/firefly-train-1.1M/resolve/main/firefly-train-1.1M.jsonl", "firefly"),
    ("multiturn", "https://huggingface.co/datasets/BelleGroup/multiturn_chat_0.8M/resolve/main/data/multiturn_chat_0.8M.jsonl", "multiturn"),
]


def fetch_head(url, nbytes=RANGE_BYTES):
    req = urllib.request.Request(url, headers={"Range": f"bytes=0-{nbytes}", "User-Agent": "duan-distill"})
    with urllib.request.urlopen(req, timeout=90) as r:
        return r.read().decode("utf-8", errors="ignore")


def parse_belle(line):
    d = json.loads(line)
    text = d.get("instruction", "")
    if d.get("input"):
        text += "\n" + str(d["input"])[:300]
    return text, "belle"


def parse_firefly(line):
    d = json.loads(line)
    text = d.get("instruction", "")
    if d.get("input"):
        text += "\n" + str(d["input"])[:300]
    return text, f"firefly_{d.get('task_type', 'misc')}"


def parse_multiturn(line):
    d = json.loads(line)
    conv = d.get("conversations", [])
    first = next((c["value"] for c in conv if c.get("from") == "human"), "")
    return first, "multiturn"


PARSERS = {"belle": parse_belle, "firefly": parse_firefly, "multiturn": parse_multiturn}


def main():
    random.seed(7)
    rows = []
    for name, url, tag in SOURCES:
        try:
            head = fetch_head(url)
            lines = [l for l in head.splitlines() if l.strip()]
            if lines and not lines[-1].strip().endswith("}"):
                lines = lines[:-1]
            random.shuffle(lines)
            got = 0
            for l in lines:
                if got >= PER_SOURCE:
                    break
                try:
                    text, sub = PARSERS[tag](l)
                except Exception:
                    continue
                text = (text or "").strip()
                if not (30 <= len(text) <= MAX_CHARS):
                    continue
                got += 1
                rows.append({"id": f"mat-{name}-{got:04d}", "text": text, "tag": sub,
                             "holdout": random.random() < 0.05})
            print(f"{name}: {got} seeds")
        except Exception as e:
            print(f"{name}: ERROR {str(e)[:200]}")

    random.shuffle(rows)
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"wrote {len(rows)} seeds -> {OUT}")


if __name__ == "__main__":
    main()
