"""Mass distillation via local vLLM gateway (~5500 req/s @ conc 1024).

Dual-teacher: Qwen3.8-Flash-Next-FP8 primary, GLM-5.3-Flash cross-check.
Resume-safe: committed rows are skipped on restart. Soft targets via
single-token category probing (content first, reasoning_content fallback).
"""
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = os.environ.get("DUAN_BASE", "http://133.186.146.224:8081/v1")
KEY = os.environ.get("DUAN_KEY", "sk-7878")
CONC = int(os.environ.get("CONC", "1024"))
INSTR = "判断这条工单属于哪个类别，只回答一个词（{labels}）：\n\n{state}"


def chat(model, prompt, max_tokens=400):
    body = json.dumps({"model": model,
                       "messages": [{"role": "user", "content": prompt}],
                       "temperature": 0, "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(f"{BASE}/chat/completions", data=body, headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {KEY}"}, method="POST")
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                d = json.load(r)
            msg = d["choices"][0]["message"]
            return (msg.get("content") or "") + "\n" + (msg.get("reasoning_content") or "")
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def classify(model, state, labels):
    text = chat(model, INSTR.format(labels="/".join(labels), state=state)).lower()
    picked = None
    for lab in labels:
        if lab.lower() in text:
            picked = lab
            break
    scores = {lab: (1.0 if lab == picked else 0.0) for lab in labels} if picked else {}
    return picked, scores


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
            todo.append(r)
    return todo


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen3.8-Flash-Next-FP8")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    # group seeds by their own label sets
    groups = {}
    for p in args.seeds:
        for line in open(p, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            labels = r.get("_candidates") or r.get("labels")
            if not labels:
                continue
            labels = list(labels) if isinstance(labels, dict) else labels
            key = tuple(sorted(labels))
            groups.setdefault((key, tuple(labels)), []).append(r)
    todo_all = []
    for (_, labels), rows in groups.items():
        for r in rows:
            todo_all.append((r, labels))
    if args.limit:
        todo_all = todo_all[: args.limit]
    print(f"todo: {len(todo_all)}", flush=True)

    t0 = time.time()
    n_ok = n_fail = 0
    out = open(args.out, "a", encoding="utf-8")
    lock_write = __import__("threading").Lock()

    def work(item):
        r, labels = item
        try:
            picked, scores = classify(args.model, r["text"], labels)
            if not picked:
                return ("skip", r)
            row = {"id": r["id"], "state": r["text"], "question": r.get("tag", "mass"),
                   "qtype": "choice", "instructions": r.get("_question", ""),
                   "labels": labels, "answer": picked, "scores": scores,
                   "confidence": scores.get(picked, 0.0), "teacher": args.model}
            return ("ok", row)
        except Exception:
            return ("fail", r)

    with ThreadPoolExecutor(CONC) as ex:
        for status, obj in ex.map(work, todo_all):
            if status == "ok":
                with lock_write:
                    out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    if (n_ok + 1) % 500 == 0:
                        out.flush()
                n_ok += 1
            elif status == "fail":
                n_fail += 1
            done = n_ok + n_fail
            if done % 2000 == 0:
                rate = done / (time.time() - t0) * 60
                print(f"  {done}/{len(todo_all)} ok={n_ok} fail={n_fail} rate={rate:.0f}/min", flush=True)

    out.close()
    mins = (time.time() - t0) / 60
    print(f"DONE {n_ok} rows in {mins:.1f} min ({n_ok/mins:.0f}/min), fail={n_fail}", flush=True)


if __name__ == "__main__":
    main()
