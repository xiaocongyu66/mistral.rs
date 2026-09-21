import json
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

API = "https://classifier.dev/v1/classify"
MODEL = "jev"
TIMEOUT = 120

ROOT = Path(__file__).parent


def call_classify(payload: dict) -> dict:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        API,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return json.load(resp)


def main() -> None:
    seeds = [json.loads(line) for line in (ROOT / "seeds_zh.jsonl").read_text().splitlines() if line.strip()]
    spec = json.loads((ROOT / "questions.json").read_text())
    questions = spec["questions"]

    out_path = ROOT / "data" / f"teacher-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.jsonl"
    out_path.parent.mkdir(exist_ok=True)

    n_rows = 0
    t0 = time.time()
    with out_path.open("w") as out:
        for q in questions:
            payload = {
                "model": MODEL,
                "inputs": [s["text"] for s in seeds],
                "labels": q["labels"],
                "instructions": q["instructions"],
            }
            resp = call_classify(payload)
            results = resp["results"]
            assert len(results) == len(seeds), f"{q['name']}: {len(results)} != {len(seeds)}"
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
            print(f"{q['name']}: {len(results)} results, usage={json.dumps(resp.get('usage', {}))}")

    print(f"wrote {n_rows} rows to {out_path} in {time.time() - t0:.1f}s")
    return 0 if n_rows else 1


if __name__ == "__main__":
    sys.exit(main())
