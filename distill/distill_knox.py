"""Distill via the knox.chat jev gateway (POST /v1/systemone).

Returns full soft distributions over candidate labels; schema maps 1:1 to
our generate.py rows so convert_nanojev.py consumes it unchanged.
"""
import json
import math
import os
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

KEY = os.environ["KNOX_KEY"]  # required; never commit keys
URL = "https://api.knox.chat/v1/systemone"
CONC = int(os.environ.get("CONC", "48"))


def probe(state, question, labels, max_len=4000):
    crit = {lab: lab for lab in labels}
    payload = {"model": "jev-latest",
               "state": state[:max_len],
               "questions": {"q": {"text": question, "type": "choice",
                                    "instructions": question,
                                    "criteria": crit}}}
    body = json.dumps(payload).encode()
    last = None
    for attempt in range(8):
        req = urllib.request.Request(URL, data=body,
                                     headers={"Content-Type": "application/json",
                                              "Authorization": "Bearer " + KEY},
                                     method="POST")
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                d = json.load(r)
            probs = d["answers"]["q"]["probabilities"]
            return {lab: float(probs.get(lab, 0.0)) for lab in labels}
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(min(4 * (attempt + 1), 30))
                continue
            raise
        except Exception as e:
            last = e
            time.sleep(2 ** attempt)
    raise last or RuntimeError("knox probe failed")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    done = set()
    if os.path.exists(a.out):
        for line in open(a.out, encoding="utf-8"):
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                pass
    todo = []
    for p in a.seeds:
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
    if a.limit:
        todo = todo[:a.limit]
    print(f"todo: {len(todo)} conc={CONC}", flush=True)

    t0, n_ok, n_fail = time.time(), 0, 0
    out = open(a.out, "a", encoding="utf-8")
    wl = threading.Lock()

    err_samples = []

    def work(item):
        r, labels = item
        try:
            scores = probe(r["text"], r["_question"], labels)
            s = sum(scores.values()) or 1.0
            scores = {k: v / s for k, v in scores.items()}
            ent = -sum(p * math.log(max(p, 1e-9)) for p in scores.values())
            row = {"id": r["id"], "state": r["text"], "question": r["tag"],
                   "qtype": "choice", "instructions": r["_question"],
                   "labels": labels, "scores": {k: round(v, 6) for k, v in scores.items()},
                   "entropy": round(ent, 6),
                   "answer": max(scores, key=scores.get),
                   "confidence": round(max(scores.values()), 6),
                   "teacher": "jev-1.13.0@knox"}
            return ("ok", row)
        except Exception as e:
            if len(err_samples) < 3:
                err_samples.append(f"{type(e).__name__}: {str(e)[:120]}")
                print(f"[FAIL] {r['id']}: {err_samples[-1]}", flush=True)
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
    print(f"DONE ok={n_ok} fail={n_fail} in {m:.1f} min ({n_ok/max(m,0.01):.0f} rows/min)", flush=True)


if __name__ == "__main__":
    main()
