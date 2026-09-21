"""Concurrent teacher-labeling via classifier.dev for burning the daily quota.

Same row schema as generate_perrow.py (convert_nanojev.py compatible).
Groups rows by candidate set, splits into char-budgeted chunks, fans out
with a small thread pool; 429/5xx get exponential backoff.
"""
import argparse
import json
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).parent
API = "https://classifier.dev/v1/classify"

_print_lock = threading.Lock()
_pace_last = [0.0]
_pace_lock = threading.Lock()


def _pace_gate(pace):
    with _pace_lock:
        now = time.time()
        wait = _pace_last[0] + pace - now
        if wait > 0:
            time.sleep(wait)
        _pace_last[0] = time.time()
        return 0.0


def call(payload, retries=12):
    body = json.dumps(payload).encode()
    last = None
    for attempt in range(retries):
        req = urllib.request.Request(API, data=body,
                                     headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (429, 503):
                try:
                    body = e.read().decode(errors="replace")
                except Exception:
                    body = ""
                # a daily-cap 429 counts attempts too: retrying burns the
                # remaining quota, so stop the whole run instead
                if "rate_limit_day" in body:
                    raise SystemExit(f"DAILY_LIMIT: {body[:200]}") from e
                if attempt < retries - 1:
                    print(f"[{e.code}] attempt={attempt} body={body[:200]!r}", flush=True)
                    time.sleep(90 + attempt * 45)
                    continue
            raise
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            last = e
            if attempt < retries - 1:
                time.sleep(2 ** attempt * 3)
                continue
            raise
    raise last


def load_rows(paths):
    rows = []
    for p in paths:
        with Path(p).open(encoding="utf-8") as f:
            for l in f:
                if l.strip():
                    rows.append(json.loads(l))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", required=True)
    ap.add_argument("--budget", type=int, default=20000)
    ap.add_argument("--max-chars-per-req", type=int, default=40000)
    ap.add_argument("--max-rows-per-req", type=int, default=150,
                    help="cap inputs per request; big batches get silently "
                         "truncated server-side under load")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--prefix", default="train-fast")
    ap.add_argument("--pace", type=float, default=0.0,
                    help="minimum seconds between request starts (rate-limit friendly)")
    ap.add_argument("--skip-done", default="", help="jsonl files whose ids count as labeled")
    args = ap.parse_args()

    done = set()
    if args.skip_done:
        for p in args.skip_done.split(","):
            pp = Path(p)
            if pp.exists():
                for l in pp.open(encoding="utf-8"):
                    try:
                        done.add(json.loads(l)["id"])
                    except Exception:
                        pass

    rows = [r for r in load_rows(args.seeds) if r["id"] not in done]
    n_total = len(rows)
    print(f"todo={n_total} budget={args.budget} conc={args.concurrency}", flush=True)

    groups = defaultdict(list)
    for r in rows:
        groups[json.dumps(sorted(r["_candidates"]), ensure_ascii=False)].append(r)

    out_path = ROOT / "data" / f"{args.prefix}-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    out_path.parent.mkdir(exist_ok=True)
    out = out_path.open("w", encoding="utf-8")
    wl = threading.Lock()

    def make_chunks(group):
        batch, chars, chunks = [], 0, []
        for r in group:
            L = len(r["text"])
            if batch and (chars + L > args.max_chars_per_req
                          or len(batch) >= args.max_rows_per_req):
                chunks.append(batch)
                batch, chars = [], 0
            batch.append(r)
            chars += L
        if batch:
            chunks.append(batch)
        return chunks

    tasks = []
    cands_by_group = {}
    for key, group in groups.items():
        cands_by_group[key] = json.loads(key)
        instructions = group[0]["_question"]
        for batch in make_chunks(group):
            if len(tasks) * 50 >= args.budget and tasks:
                break
            tasks.append((key, instructions, batch))
        if len(tasks) * 50 >= args.budget:
            break
    # hard budget by row count
    tasks, budget_used, n_left = [], 0, args.budget
    for key, group in groups.items():
        if n_left <= 0:
            break
        for batch in make_chunks(group):
            if len(batch) > n_left:
                batch = batch[:n_left]
            tasks.append((key, group[0]["_question"], batch))
            n_left -= len(batch)
            if n_left <= 0:
                break

    print(f"http requests: {len(tasks)}", flush=True)
    t0 = time.time()
    n_ok, n_fail, n_rows = 0, 0, 0

    def work(task):
        key, instructions, batch = task
        payload = {"model": "jev", "inputs": [r["text"] for r in batch],
                   "labels": cands_by_group[key]}
        if instructions:
            payload["instructions"] = instructions
        if args.pace > 0:
            time.sleep(_pace_gate(args.pace))
        try:
            resp = call(payload)
            return task, batch, resp["results"], None
        except Exception as e:
            return task, batch, None, str(e)

    with ThreadPoolExecutor(args.concurrency) as ex:
        from concurrent.futures import as_completed
        futs = {ex.submit(work, t): t for t in tasks}
        for fut in as_completed(futs):
            try:
                task, batch, results, err = fut.result()
            except SystemExit as e:
                # stop firing new requests but keep draining already-completed
                # ones: their quota was already spent
                print(str(e), flush=True)
                for f in futs:
                    f.cancel()
                continue
            if err is None:
                if len(results) != len(batch):
                    print(f"[TRUNC] server returned {len(results)}/{len(batch)} "
                          f"results; {len(batch)-len(results)} rows lost", flush=True)
                with wl:
                    for r, res in zip(batch, results):
                        out.write(json.dumps({
                            "id": r["id"], "state": r["text"], "question": r["tag"],
                            "qtype": "choice", "instructions": r["_question"],
                            "labels": cands_by_group[task[0]],
                            "answer": res.get("label") or res.get("labels"),
                            "scores": res.get("scores"), "confidence": res.get("confidence"),
                            "ms": res.get("ms"), "teacher": res.get("model"),
                        }, ensure_ascii=False) + "\n")
                n_ok += 1
                n_rows += len(batch)
            else:
                n_fail += 1
                with _print_lock:
                    print(f"[FAIL] {len(batch)} rows: {err[:200]}", flush=True)
            if (n_ok + n_fail) % 20 == 0:
                rate = n_rows / max(time.time() - t0, 1) * 60
                with _print_lock:
                    print(f"  req={n_ok+n_fail}/{len(tasks)} rows={n_rows} "
                          f"fail={n_fail} {rate:.0f} rows/min", flush=True)

    out.close()
    m = (time.time() - t0) / 60
    print(f"DONE rows={n_rows} ok_req={n_ok} fail_req={n_fail} in {m:.1f} min "
          f"({n_rows/max(m,0.01):.0f} rows/min) -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
