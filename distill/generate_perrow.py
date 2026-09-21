"""Teacher-label per-row-candidate seeds via classifier.dev.

Each row carries its own candidate set (from ingest_verdict.py), so requests
are made per candidate-group (fast lane). Output: teacher-*.jsonl rows in the
same schema as generate.py, ready for convert_nanojev.py.
"""
import argparse
import json
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parent
API = "https://classifier.dev/v1/classify"


def call(payload, retries=5):
    body = json.dumps(payload).encode()
    for attempt in range(retries):
        req = urllib.request.Request(API, data=body,
                                     headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            if e.code in (429, 503) and attempt < retries - 1:
                time.sleep(2 ** attempt * 5)
                continue
            raise
        except (urllib.error.URLError, json.JSONDecodeError) as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt * 5)
                continue
            raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", required=True)
    ap.add_argument("--max-rows", type=int, default=300, help="per-night quota budget")
    ap.add_argument("--max-chars-per-req", type=int, default=40000,
                    help="cap on summed input length per HTTP request")
    ap.add_argument("--prefix", default="train-verdict")
    args = ap.parse_args()

    rows = []
    for p in args.seeds:
        # open() iteration, NOT read_text().splitlines(): splitlines() breaks
        # on Unicode line separators (\u2028/\u2029) inside JSON strings
        with Path(p).open() as f:
            for l in f:
                if l.strip():
                    rows.append(json.loads(l))

    # group by identical candidate sets
    groups = defaultdict(list)
    for r in rows:
        key = json.dumps(sorted(r["_candidates"]), ensure_ascii=False)
        groups[key].append(r)

    out_path = ROOT / "data" / f"{args.prefix}-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    out_path.parent.mkdir(exist_ok=True)
    n = 0
    with out_path.open("w") as out:
        for key, group in groups.items():
            if n >= args.max_rows:
                break
            cands = json.loads(key)
            instructions = next((r["_question"] for r in group), None)
            # split group into char-budgeted chunks
            batch, chars = [], 0
            chunks = []
            for r in group:
                if n + len(batch) >= args.max_rows:
                    break
                L = len(r["text"])
                if batch and chars + L > args.max_chars_per_req:
                    chunks.append(batch)
                    batch, chars = [], 0
                batch.append(r)
                chars += L
            if batch:
                chunks.append(batch)
            for batch in chunks:
                payload = {"model": "jev", "inputs": [r["text"] for r in batch], "labels": cands}
                if instructions:
                    payload["instructions"] = instructions
                resp = call(payload)
                for r, res in zip(batch, resp["results"]):
                    row = {
                        "id": r["id"], "state": r["text"], "question": r["tag"],
                        "qtype": "choice", "instructions": r["_question"],
                        "labels": cands, "answer": res.get("label") or res.get("labels"),
                        "scores": res.get("scores"), "confidence": res.get("confidence"),
                        "ms": res.get("ms"), "teacher": res.get("model"),
                    }
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    n += 1
                print(f"chunk ({len(cands)} cands): {len(batch)} rows done, total {n}", flush=True)
                time.sleep(2)
    print(f"wrote {n} rows -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
