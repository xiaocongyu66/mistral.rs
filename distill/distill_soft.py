"""TRUE distillation: probe the teacher's next-token distribution over the
candidate label tokens via logprobs, store the full soft distribution.

Why not argmax: the entire value of a System-1 teacher is its CALIBRATED
DISTRIBUTION. Hard labels discard the dark knowledge (class similarity,
hesitation) that KL distillation transfers to the student. This script keeps
it: rows carry `scores` (softmax over candidates) + `entropy` (sample weight).
"""
import json
import math
import os
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = os.environ.get("DUAN_BASE", "http://133.186.146.224:8081/v1")
KEY = os.environ.get("DUAN_KEY", "sk-7878")
CONC = int(os.environ.get("CONC", "512"))


def probe(model, state, labels, max_probe=20):
    """One request; read top_logprobs of the answer token; softmax candidates.

    Only sources that return a >=2-candidate distribution count: a single
    1.0 logit is an argmax, not a distribution.
    """
    prompt = ("The ticket: " + state + "\n\nPick the single most fitting word, "
              "output only that word, no explanation:\n" + ", ".join(labels))
    payload = {"model": model,
               "messages": [{"role": "user", "content": prompt}],
               "max_tokens": 8, "temperature": 0,
               "logprobs": True, "top_logprobs": max_probe,
               "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(
        BASE + "/chat/completions", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + KEY}, method="POST")
    last = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                d = json.load(r)
            break
        except Exception as e:
            last = e
            time.sleep(1.5 ** attempt)
    else:
        raise last
    lp = d["choices"][0].get("logprobs")
    if not lp or not lp.get("content"):
        raise RuntimeError("no logprobs in response")
    lset = set(x.lower() for x in labels)
    for position in lp["content"][:3]:
        cand = {}
        for t in position.get("top_logprobs", []):
            tok = "".join(ch for ch in t["token"].strip().lower()
                          if ch.isalnum() or ch == "_")
            if tok in lset:
                for lab in labels:
                    if tok == lab.lower():
                        cand[lab] = t["logprob"]
        if len(cand) >= 2:
            mx = max(cand.values())
            Z = sum(math.exp(v - mx) for v in cand.values())
            scores = {k: math.exp(v - mx) / Z for k, v in cand.items()}
            n = len(labels)
            miss = n - len(scores)
            eps = (1.0 - sum(scores.values())) / miss if miss else 0.0
            full = {lab: scores.get(lab, eps) for lab in labels}
            ent = -sum(p * math.log(max(p, 1e-9)) for p in full.values())
            return full, ent
    raise RuntimeError("candidates not in top tokens")


def load_todo(seed_files, out_path):
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path, encoding="utf-8"):
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                pass
    todo = []
    for p in seed_files:
        for line in open(p, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            if r["id"] in done:
                continue
            labels = r.get("_candidates") or r.get("labels")
            if not labels:
                continue
            todo.append((r, list(labels) if isinstance(labels, dict) else labels))
    return todo


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="GLM-5.3-Flash")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    todo = load_todo(a.seeds, a.out)
    if a.limit:
        todo = todo[: a.limit]
    print(f"todo: {len(todo)} model={a.model} conc={CONC}", flush=True)

    t0, n_ok, n_fail = time.time(), 0, 0
    out = open(a.out, "a", encoding="utf-8")
    wl = threading.Lock()

    def work(item):
        r, labels = item
        try:
            scores, ent = probe(a.model, r["text"], labels)
            row = {"id": r["id"], "state": r["text"], "question": r.get("tag", "mass"),
                   "qtype": "choice", "instructions": r.get("_question", ""),
                   "labels": labels, "scores": {k: round(v, 6) for k, v in scores.items()},
                   "entropy": round(ent, 6),
                   "answer": max(scores, key=scores.get),
                   "confidence": round(max(scores.values()), 6),
                   "teacher": a.model}
            return ("ok", row)
        except Exception:
            return ("fail", r)

    with ThreadPoolExecutor(CONC) as ex:
        for status, obj in ex.map(work, todo):
            if status == "ok":
                with wl:
                    out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    if (n_ok + 1) % 500 == 0:
                        out.flush()
                n_ok += 1
            else:
                n_fail += 1
            if (n_ok + n_fail) % 2000 == 0:
                rate = (n_ok + n_fail) / (time.time() - t0) * 60
                print(f"  {n_ok+n_fail}/{len(todo)} ok={n_ok} fail={n_fail} {rate:.0f}/min", flush=True)

    out.close()
    m = (time.time() - t0) / 60
    print(f"DONE ok={n_ok} fail={n_fail} in {m:.1f} min ({n_ok/m:.0f} rows/min)", flush=True)


if __name__ == "__main__":
    main()
