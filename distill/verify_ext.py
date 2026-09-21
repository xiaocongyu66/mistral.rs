import json, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor
from registry_ext import EXTENDED

def probe(e):
    hf_id = e[0]
    url = "https://huggingface.co/api/datasets/" + urllib.parse.quote(hf_id, safe="/")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "duan"})
        with urllib.request.urlopen(req, timeout=25) as r:
            d = json.load(r)
        return {**{"id": hf_id, "config": e[1], "split": e[2], "field": e[3], "domain": e[4]},
                "ok": True, "downloads": d.get("downloads", 0), "gated": bool(d.get("gated"))}
    except Exception as ex:
        return {"id": hf_id, "domain": e[4], "ok": False, "err": str(ex)[:60]}

with ThreadPoolExecutor(max_workers=10) as ex:
    res = list(ex.map(probe, EXTENDED))

uniq, seen = [], set()
for r in res:
    if r["id"] in seen:
        continue
    seen.add(r["id"])
    uniq.append(r)

ok = [r for r in uniq if r["ok"] and not r["gated"]]
gated = [r for r in uniq if r["ok"] and r["gated"]]
bad = [r for r in uniq if not r["ok"]]
print(f"unique ids: {len(uniq)} | usable: {len(ok)} | gated: {len(gated)} | missing: {len(bad)}")
for r in bad[:12]:
    print(f"  MISSING {r['id']}")
for r in gated[:6]:
    print(f"  GATED   {r['id']}")
json.dump({"usable": ok, "gated": gated, "missing": bad},
          open("registry_ext_result.json", "w"), ensure_ascii=False, indent=1)
print("-> registry_ext_result.json")
