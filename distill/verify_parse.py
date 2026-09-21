"""Actually load every registered source and confirm we can parse text from it.

Registration without this step is worthless: a wrong config name or split name
fails at load time, and a wrong field name yields empty text. Only sources that
pass here belong in the warmup stream.
"""
import json, sys, traceback

RESULT = "verify_parse_result.json"


def probe(entry):
    from datasets import load_dataset
    repo, cfg, split, field = entry["id"], entry["config"], entry["split"], entry["field"]
    out = dict(entry)
    try:
        ds = load_dataset(repo, cfg, split=split, streaming=True)
        it = iter(ds)
        for _ in range(3):
            row = next(it)
        keys = list(row.keys())
        # try declared field, then aliases, then any string field long enough
        text = ""
        if field in row and isinstance(row[field], str):
            text = row[field]
        else:
            for k in ("text", "content", "raw_content", "body", "document", "problem",
                      "question", "instruction", "input", "prompt", "query", "review_body"):
                if isinstance(row.get(k), str) and len(row[k]) > 30:
                    text = row[k]
                    out["field_fallback"] = k
                    break
        out["keys"] = keys[:12]
        out["text_len"] = len(text)
        out["preview"] = text[:120].replace("\n", " ")
        out["ok"] = len(text) >= 30
        out["err"] = "" if out["ok"] else "no usable text field"
    except Exception as e:
        out["ok"] = False
        out["err"] = f"{type(e).__name__}: {str(e)[:120]}"
    return out


def main():
    data = json.load(open("registry_ext_result.json"))
    usable = data["usable"]
    done = {}
    try:
        done = {r["id"] + str(r["config"]): r for r in json.load(open(RESULT))}
    except Exception:
        pass
    results = []
    for e in usable:
        key = e["id"] + str(e["config"])
        if key in done:
            results.append(done[key])
            continue
        r = probe(e)
        results.append(r)
        status = "OK  " if r.get("ok") else "FAIL"
        print(f"{status} {r['id']:55s} {r.get('text_len',0):6d} {r.get('err','')[:60]}", flush=True)
        json.dump(results, open(RESULT, "w"), ensure_ascii=False, indent=1)
    good = [r for r in results if r.get("ok")]
    print(f"\nparseable: {len(good)}/{len(results)}")
    json.dump(results, open(RESULT, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
