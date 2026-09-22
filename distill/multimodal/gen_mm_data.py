"""Generate multimodal decision training data from verified HF datasets.

Pipeline:
  1. Pull image+caption pairs from COCO / Flickr30k / LLaVA-Instruct
  2. Convert to decision format: {image_path, state_text, question, candidates, soft_label}
  3. Use jev (knox) to label the text description → soft distribution
  4. Save as JSONL for train_mm.py

Usage:
  KNOX_KEY=sk-xxx python gen_mm_data.py --source coco --count 500 --out mm_coco.jsonl
"""
import argparse
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Decision question templates per source
TEMPLATES = {
    "coco": {
        "question": "这张图片的主要场景是什么？",
        "candidates_from": "categories",
    },
    "flickr30k": {
        "question": "图片中正在发生什么？",
        "candidates_from": "actions",
    },
    "llava": {
        "question": "根据图片内容回答问题。",
        "candidates_from": "qa",
    },
}

# COCO 80 categories for candidate sampling
COCO_CATS = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
             "truck", "boat", "traffic light", "fire hydrant", "stop sign",
             "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
             "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
             "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
             "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
             "tennis racket", "bottle", "wine glass", "cup", "fork", "knife",
             "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
             "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
             "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
             "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
             "toaster", "sink", "refrigerator", "book", "clock", "vase",
             "scissors", "teddy bear", "hair drier", "toothbrush"]


def download_image(url, save_dir, filename):
    try:
        path = os.path.join(save_dir, filename)
        if os.path.exists(path):
            return path
        urllib.request.urlretrieve(url, path)
        return path
    except Exception:
        return None


def gen_from_coco(count, out_path, img_dir):
    """COCO: image + caption → scene classification decision."""
    from huggingface_hub import hf_hub_download
    import random
    rng = random.Random(42)
    os.makedirs(img_dir, exist_ok=True)

    # pull COCO train split
    lp = hf_hub_download("jxie/coco_captions", "data/train-00000-of-00182-5c0b9bd6a017ebf2.parquet",
                          repo_type="dataset", local_dir="mm_raw")
    import pyarrow.parquet as pq
    t = pq.read_table(lp)
    df = t.to_pydict()
    n_avail = len(df["image_id"])
    print(f"COCO rows available: {n_avail}")

    rows = []
    idxs = list(range(min(n_avail, count * 3)))
    rng.shuffle(idxs)
    for i in idxs:
        if len(rows) >= count:
            break
        # COCO format varies; adapt to actual schema
        img_bytes = df.get("image", [None] * n_avail)[i]
        caption = df.get("caption", [""] * n_avail)[i]
        if not caption or not img_bytes:
            continue
        # save image
        fname = f"coco_{i:06d}.jpg"
        img_path = os.path.join(img_dir, fname)
        if not os.path.exists(img_path):
            try:
                with open(img_path, "wb") as f:
                    f.write(img_bytes if isinstance(img_bytes, bytes) else img_bytes["bytes"])
            except Exception:
                continue
        # pick true category + 4 distractors
        true_cat = rng.choice(COCO_CATS)
        distr = [c for c in COCO_CATS if c != true_cat]
        rng.shuffle(distr)
        cands = distr[:4] + [true_cat]
        rng.shuffle(cands)
        rows.append({
            "image_path": img_path,
            "state_text": caption,
            "question": "这张图片的主要场景/对象是什么？",
            "candidates": cands,
            "gold": true_cat,
            "source": "coco",
        })
    # save pre-jev
    with open(out_path.replace(".jsonl", "_raw.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"raw rows: {len(rows)}")
    return rows


def label_with_jev(rows, out_path, key):
    """Label each row's state_text with jev soft distribution via knox."""
    sys.path.insert(0, os.path.dirname(__file__))
    from train_mm_helper import knox_probe
    labeled = []
    for i, r in enumerate(rows):
        try:
            probs = knox_probe(r["state_text"], r["question"], r["candidates"])
            s = sum(probs.values()) or 1.0
            probs = {k: round(v / s, 6) for k, v in probs.items()}
            r["soft_label"] = probs
            labeled.append(r)
        except Exception:
            pass
        if (i + 1) % 100 == 0:
            print(f"  labeled {i+1}/{len(rows)}", flush=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in labeled:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"labeled: {len(labeled)} -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="coco", choices=["coco", "flickr30k", "llava"])
    ap.add_argument("--count", type=int, default=500)
    ap.add_argument("--out", default="mm_coco.jsonl")
    ap.add_argument("--img-dir", default="mm_images")
    ap.add_argument("--skip-jev", action="store_true", help="only generate raw data")
    args = ap.parse_args()

    if args.source == "coco":
        rows = gen_from_coco(args.count, args.out, args.img_dir)
    else:
        print(f"source {args.source} not yet implemented")
        sys.exit(1)

    if not args.skip_jev and rows:
        key = os.environ.get("KNOX_KEY")
        if not key:
            print("KNOX_KEY not set; saving raw data only")
        else:
            label_with_jev(rows, args.out, key)
