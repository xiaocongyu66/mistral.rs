import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent

TYPE_MAP = {
    "choice": "choice",
    "score": "score",
    "probability": "boolean",
}

BOOLEAN_KEY = {"yes": "true", "no": "false"}


def convert(rows):
    """Group teacher-*.jsonl rows (one per question) into NanoJev unified rows
    (one per state, with a questions dict and teacher native_probs)."""
    by_state = collections.defaultdict(lambda: collections.defaultdict(dict))
    meta = {}
    for r in rows:
        sid = r["id"]
        if sid not in meta:
            meta[sid] = {"state": r["state"], "holdout": None}
        by_state[sid][r["question"]] = r
    out = []
    for sid in sorted(by_state):
        questions = {}
        gold = {}
        native = {}
        for qname, r in by_state[sid].items():
            labels = r["labels"]
            scores = r["scores"]
            typ = TYPE_MAP[r["qtype"]]
            if typ == "boolean":
                ids = ["false", "true"]
                texts = ["The proposition is true."]
                probs = {BOOLEAN_KEY[k]: float(scores[k]) for k in labels if k in BOOLEAN_KEY}
                if set(probs) != {"false", "true"}:
                    continue
                native[qname] = probs
                questions[qname] = {"type": "boolean", "instructions": r["instructions"]}
            elif typ == "choice":
                native[qname] = {lab: float(scores[lab]) for lab in labels}
                questions[qname] = {"type": "choice", "instructions": r["instructions"],
                                    "criteria": {lab: lab for lab in labels}}
            else:
                ids = [str(i) for i in range(len(labels))]
                native[qname] = {str(i): float(scores[lab]) for i, lab in enumerate(labels)}
                questions[qname] = {"type": "score", "instructions": r["instructions"],
                                    "criteria": list(labels)}
            best = max(native[qname], key=native[qname].get)
            gold[qname] = {"true": True, "false": False}.get(best, best)
        out.append({
            "id": sid,
            "state_id": sid,
            "family_id": "support_triage_v1",
            "split": "calibration" if meta[sid]["holdout"] else "train",
            "state": meta[sid]["state"],
            "questions": questions,
            "gold": gold,
            "teacher": {"native_probs": native,
                        "source": "classifier.dev jev-1.13.0 via generate.py"},
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--holdout-seeds", default=str(ROOT / "seeds_v3.jsonl"),
                    help="seeds file carrying holdout marks")
    ap.add_argument("--out", default=str(ROOT / "nanojev_format"))
    args = ap.parse_args()

    holdout = set()
    for line in Path(args.holdout_seeds).read_text().splitlines():
        if line.strip():
            s = json.loads(line)
            if s.get("holdout"):
                holdout.add(s["id"])

    rows = []
    for p in args.data:
        rows += [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
    grouped = convert(rows)
    for row in grouped:
        row["split"] = "calibration" if row["id"] in holdout else "train"

    outdir = Path(args.out)
    outdir.mkdir(exist_ok=True)
    counts = collections.Counter()
    n_q = collections.Counter()
    with (outdir / "unified.jsonl").open("w") as f:
        for row in grouped:
            split = row["split"]
            counts[split] += 1
            n_q[split] += len(row["questions"])
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (outdir / "train.jsonl").write_text("".join(
        json.dumps(r, ensure_ascii=False) + "\n" for r in grouped if r["split"] == "train"))
    (outdir / "calibration.jsonl").write_text("".join(
        json.dumps(r, ensure_ascii=False) + "\n" for r in grouped if r["split"] == "calibration"))
    print(f"states: {dict(counts)}, questions: {dict(n_q)}, total {sum(n_q.values())}")
    return 0 if sum(n_q.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
