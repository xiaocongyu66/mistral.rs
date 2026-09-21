import json
import os
import subprocess

os.makedirs("/content/materials", exist_ok=True)
TARGETS = [
    # (repo, repo_type, allow_patterns or None, note)
    ("BelleGroup/train_3.5M_CN", "dataset", ["*.jsonl", "*.json"], "belle 3.5M chinese instructions"),
    ("YeungNLP/firefly-train-1.1M", "dataset", ["*.json"], "firefly 1.1M"),
    ("shibing624/medical", "dataset", ["*.json*"], "chinese medical"),
    ("C-Tianyu/NanoJev-Data", "dataset", ["unified/soft/*"], "nanojev game questions (aux mix)"),
]

manifest = []
for repo, rtype, patterns, note in TARGETS:
    dest = "/content/materials/" + repo.replace("/", "__")
    print(f"downloading {repo} ({note}) ...", flush=True)
    try:
        from huggingface_hub import snapshot_download
        p = snapshot_download(repo, repo_type=rtype, local_dir=dest,
                              allow_patterns=patterns or None)
        size = 0
        nfiles = 0
        for root, _, files in os.walk(p):
            for f in files:
                fp = os.path.join(root, f)
                if ".cache" in fp:
                    continue
                size += os.path.getsize(fp)
                nfiles += 1
        manifest.append({"repo": repo, "note": note, "files": nfiles,
                         "bytes": size, "gb": round(size / 1e9, 3)})
        print(f"  ok {round(size/1e9, 2)} GB, {nfiles} files", flush=True)
    except Exception as e:
        manifest.append({"repo": repo, "note": note, "error": str(e)[:300]})
        print(f"  ERROR {e}", flush=True)

total = sum(m["bytes"] for m in manifest if "bytes" in m)
manifest.append({"TOTAL_GB": round(total / 1e9, 3)})
with open("/content/materials/MANIFEST.json", "w") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)
print(json.dumps(manifest, ensure_ascii=False, indent=1), flush=True)
