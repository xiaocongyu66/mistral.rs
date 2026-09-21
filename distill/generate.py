import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

API = "https://classifier.dev/v1/classify"
MODEL = "jev"
TIMEOUT = 120
RETRIES = 5

ROOT = Path(__file__).parent


def call_classify(payload: dict) -> dict:
    body = json.dumps(payload).encode()
    for attempt in range(RETRIES):
        req = urllib.request.Request(
            API,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            if e.code in (429, 503) and attempt < RETRIES - 1:
                wait = 2 ** attempt * 5
                print(f"HTTP {e.code}, retry in {wait}s (attempt {attempt + 1}/{RETRIES})")
                time.sleep(wait)
                continue
            raise
        except urllib.error.URLError as e:
            if attempt < RETRIES - 1:
                wait = 2 ** attempt * 5
                print(f"network error ({e.reason}), retry in {wait}s")
                time.sleep(wait)
                continue
            raise


CHUNK = 250
CHUNK_SLEEP = 1.5


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=str(ROOT / "seeds_zh.jsonl"))
    ap.add_argument("--prefix", default="teacher")
    args = ap.parse_args()

    seeds = [json.loads(line) for line in Path(args.seeds).read_text().splitlines() if line.strip()]
    spec = json.loads((ROOT / "questions.json").read_text())
    questions = spec["questions"]

    out_path = ROOT / "data" / f"{args.prefix}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.jsonl"
    out_path.parent.mkdir(exist_ok=True)

    n_rows = 0
    t0 = time.time()
    with out_path.open("w") as out:
        for q in questions:
            inputs = [s["text"] for s in seeds]
            results = []
            usage_total = {"classifications": 0, "escalated": 0, "ms": 0}
            for i in range(0, len(inputs), CHUNK):
                payload = {
                    "model": MODEL,
                    "inputs": inputs[i:i + CHUNK],
                    "labels": q["labels"],
                    "instructions": q["instructions"],
                }
                resp = call_classify(payload)
                results.extend(resp["results"])
                for k in usage_total:
                    usage_total[k] += resp.get("usage", {}).get(k, 0)
                if i + CHUNK < len(inputs):
                    time.sleep(CHUNK_SLEEP)
            assert len(results) == len(seeds), f"{q['name']}: {len(results)} != {len(seeds)}"
            print(f"{q['name']}: {len(results)} results, usage={json.dumps(usage_total)}", flush=True)
            for seed, r in zip(seeds, results):
                row = {
                    "id": seed["id"],
                    "state": seed["text"],
                    "question": q["name"],
                    "qtype": q["type"],
                    "instructions": q["instructions"],
                    "labels": q["labels"],
                    "answer": r.get("label") or r.get("labels"),
                    "scores": r.get("scores"),
                    "confidence": r.get("confidence"),
                    "ms": r.get("ms"),
                    "teacher": r.get("model") or resp.get("model"),
                }
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_rows += 1

    print(f"wrote {n_rows} rows to {out_path} in {time.time() - t0:.1f}s")
    return 0 if n_rows else 1


if __name__ == "__main__":
    sys.exit(main())
