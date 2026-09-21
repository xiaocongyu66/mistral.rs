import argparse
import collections
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).parent

TYPE_MAP = {
    "choice": "choice",
    "score": "score",
    "probability": "boolean",
}

BOOLEAN_KEY = {"yes": "true", "no": "false"}

# Split assignment contract (NanoJev trainer requires all five nonempty):
#   train        - non-holdout, non-English states (Chinese-dominant learning signal)
#   dev          - holdout slice 1: best-checkpoint selection
#   calibration  - holdout slice 2: post-hoc temperature fitting only
#   test         - holdout slice 3: final held-out report, never used for decisions
#   ood          - English-tagged states (distribution shift for a Chinese-dominant
#                  model); excluded from train on purpose
DEV_FRAC, CAL_FRAC = 0.34, 0.33


def convert(rows, id2tag, id2holdout):
    by_state = collections.defaultdict(lambda: collections.defaultdict(dict))
    states = {}
    for r in rows:
        sid = r["id"]
        if sid not in states:
            states[sid] = {"state": r["state"]}
        by_state[sid][r["question"]] = r
    out = []
    for sid in sorted(by_state):
        questions, gold, native = {}, {}, {}
        for qname, r in by_state[sid].items():
            labels, scores = r["labels"], r["scores"]
            typ = TYPE_MAP[r["qtype"]]
            if typ == "boolean":
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
                native[qname] = {str(i): float(scores[lab]) for i, lab in enumerate(labels)}
                questions[qname] = {"type": "score", "instructions": r["instructions"],
                                    "criteria": list(labels)}
            best = max(native[qname], key=native[qname].get)
            gold[qname] = {"true": True, "false": False}.get(best, best)
            probs = [max(v, 1e-9) for v in native[qname].values()]
            s = sum(probs)
            ent = -sum((v / s) * math.log(v / s) for v in probs)
            questions[qname]["teacher_entropy"] = round(ent, 6)
            questions[qname]["teacher_confidence"] = round(max(probs) / s, 6)
        tag = id2tag.get(sid, "")
        if tag.startswith("en"):
            split = "ood"
        elif id2holdout.get(sid):
            split = "dev_or_cal"
        else:
            split = "train"
        out.append({
            "id": sid, "state_id": sid, "family_id": "support_triage_v1",
            "split": split, "state": states[sid]["state"],
            "questions": questions, "gold": gold,
            "teacher": {"native_probs": native,
                        "source": "classifier.dev jev-1.13.0 via generate.py"},
        })
    return out


def assign_holdout_slices(grouped):
    """Deterministically split dev_or_cal states into dev/calibration/test."""
    dev_or_cal = [r for r in grouped if r["split"] == "dev_or_cal"]
    dev_n = max(1, int(len(dev_or_cal) * DEV_FRAC))
    cal_n = max(1, int(len(dev_or_cal) * CAL_FRAC))
    for i, r in enumerate(sorted(dev_or_cal, key=lambda r: r["id"])):
        r["split"] = "dev" if i < dev_n else ("calibration" if i < dev_n + cal_n else "test")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", default=[],
                    help="seed files carrying tag/holdout marks")
    ap.add_argument("--out", default=str(ROOT / "nanojev_format"))
    args = ap.parse_args()

    id2tag, id2holdout = {}, {}
    for p in args.seeds:
        with Path(p).open() as f:
            for line in f:
                if line.strip():
                    s = json.loads(line)
                    id2tag[s["id"]] = s.get("tag", "")
                    id2holdout[s["id"]] = bool(s.get("holdout"))

    rows = []
    for p in args.data:
        # open() iteration, NOT read_text().splitlines(): splitlines() breaks
        # on Unicode line separators (\u2028/\u2029) inside JSON strings
        with Path(p).open() as f:
            for l in f:
                if l.strip():
                    rows.append(json.loads(l))
    grouped = convert(rows, id2tag, id2holdout)
    assign_holdout_slices(grouped)

    outdir = Path(args.out)
    outdir.mkdir(exist_ok=True)
    counts, n_q = collections.Counter(), collections.Counter()
    for name in ("train", "dev", "calibration", "test", "ood"):
        part = [r for r in grouped if r["split"] == name]
        (outdir / f"{name}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in part))
        counts[name] = len(part)
        n_q[name] = sum(len(r["questions"]) for r in part)
    (outdir / "unified.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in grouped))
    print(f"states: {dict(counts)}")
    print(f"questions: {dict(n_q)}, total {sum(n_q.values())}")
    return 0 if sum(n_q.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
