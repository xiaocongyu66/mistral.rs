"""Ingest openJev-verdict data (Apache-2.0) into our teacher-labeling pipeline.

openjev-jsonl-v1 (id/question/text|context/candidates/target_id) -> our seeds
jsonl (id/text/tag/holdout). Teacher soft labels are then produced by
generate.py via classifier.dev, and the rows flow into convert_nanojev.py.
"""
import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).parent

QUESTION_TAG = {
    "What is the primary customer inquiry or banking request?": "verdict_banking",
    "Classify whether this email is a fraudulent phishing attempt, legitimate correspondence, or requires abstention.": "verdict_phishing",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", required=True, help="openjev jsonl files")
    ap.add_argument("--limit-per-file", type=int, default=2000,
                    help="cap per file to spread across quota days")
    ap.add_argument("--out", default=str(ROOT / "seeds_verdict.jsonl"))
    args = ap.parse_args()

    random.seed(99)
    rows = []
    for src in args.src:
        n_in = 0
        for line in Path(src).read_text().splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            text = d.get("text") or d.get("context") or ""
            question = d.get("question", "")
            cands = d.get("candidates") or []
            if not text or not cands:
                continue
            tag = QUESTION_TAG.get(question, "verdict_misc")
            rows.append({
                "id": f"ing-{d['id']}",
                "text": text,
                "tag": tag,
                "holdout": random.random() < 0.05,
                "_question": question,
                "_candidates": {c["id"]: c.get("description", c["id"]) for c in cands},
                "_target": d.get("target_id"),
            })
            n_in += 1
            if n_in >= args.limit_per_file:
                break
        print(f"{src}: {n_in} rows")

    random.shuffle(rows)
    out = Path(args.out)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    tags = {}
    for r in rows:
        tags[r["tag"]] = tags.get(r["tag"], 0) + 1
    print(f"wrote {len(rows)} seeds -> {out}")
    print("tags:", tags)


if __name__ == "__main__":
    main()
