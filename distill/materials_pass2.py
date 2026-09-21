import json
import os

manifest = json.load(open("/content/materials/MANIFEST.json"))
FIREFLY_SIZE = 1_700_000_000  # 预期，下载后校验

try:
    from huggingface_hub import snapshot_download
    p = snapshot_download("YeungNLP/firefly-train-1.1M", repo_type="dataset",
                          local_dir="/content/materials/YeungNLP__firefly-train-1.1M",
                          allow_patterns=["*.jsonl", "*.json", "*.md"])
    size = sum(os.path.getsize(os.path.join(r, f))
               for r, _, fs in os.walk(p) for f in fs if ".cache" not in r)
    print("firefly ok:", round(size / 1e9, 3), "GB", flush=True)
    for m in manifest:
        if m.get("repo") == "YeungNLP/firefly-train-1.1M":
            m["bytes"], m["gb"], m["files"] = size, round(size / 1e9, 3), 1
except Exception as e:
    print("firefly ERROR:", str(e)[:300], flush=True)

try:
    p2 = snapshot_download("BelleGroup/multiturn_chat_0.8M", repo_type="dataset",
                           local_dir="/content/materials/BelleGroup__multiturn_chat_0.8M",
                           allow_patterns=["*.jsonl", "*.json", "*.md", "*.gz"])
    size2 = sum(os.path.getsize(os.path.join(r, f))
                for r, _, fs in os.walk(p2) for f in fs if ".cache" not in r)
    print("multiturn ok:", round(size2 / 1e9, 3), "GB", flush=True)
    manifest.insert(-1, {"repo": "BelleGroup/multiturn_chat_0.8M", "note": "multi-turn dialogues",
                         "files": 1, "bytes": size2, "gb": round(size2 / 1e9, 3)})
except Exception as e:
    print("multiturn ERROR:", str(e)[:300], flush=True)

total = sum(m["bytes"] for m in manifest if "bytes" in m)
manifest[-1] = {"TOTAL_GB": round(total / 1e9, 3)}
with open("/content/materials/MANIFEST.json", "w") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)
print(json.dumps(manifest, ensure_ascii=False)[:600], flush=True)
