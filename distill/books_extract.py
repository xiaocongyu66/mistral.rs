"""Extract seed blocks from downloaded book PDFs.

Blocks are 300-800 char knowledge statements (chapter-anchored). Output:
seeds_books.jsonl in our seed schema. Text stays on the VM; only counts go
back to the phone.
"""
import json
import os

MIN_LEN, MAX_LEN = 250, 800
CHAPTER_KEYS = ("chapter", "part", "Chapter", "CHAPTER")

BOOKS = [
    ("alg_dm", "/content/books/alg_dm.pdf", "book_decision_algorithms"),
    ("rl_sutton", "/content/books/rl_sutton.pdf", "book_reinforcement_learning"),
    ("rl_bertsekas", "/content/books/rl_bertsekas.pdf", "book_rl_optimal_control"),
    ("llm_fudan", "/content/books/llm_fudan.pdf", "book_llm_theory"),
]


def main():
    try:
        import fitz
    except ImportError:
        subprocess = __import__("subprocess")
        subprocess.run(["pip", "install", "-q", "PyMuPDF"], check=True)
        import fitz

    rows = []
    n = 0
    for name, path, tag in BOOKS:
        if not os.path.exists(path) or os.path.getsize(path) < 100_000:
            print(f"SKIP {name} (missing/too small)", flush=True)
            continue
        doc = fitz.open(path)
        chapter = "front"
        buf = []
        pages_used = 0
        for page in doc:
            pages_used += 1
            text = page.get_text("text").strip()
            for line in text.splitlines():
                s = line.strip()
                if any(s.startswith(k) for k in CHAPTER_KEYS) and 0 < len(s) < 80:
                    chapter = s[:60]
                if not s:
                    continue
                buf.append(s)
                joined = " ".join(buf)
                if len(joined) >= MIN_LEN:
                    if len(joined) <= MAX_LEN:
                        n += 1
                        rows.append({"id": f"bk-{name}-{n:05d}", "text": joined,
                                     "tag": f"{tag}::{chapter[:40]}",
                                     "holdout": False, "_chapter": chapter})
                    buf = buf[-2:] if len(joined) > MAX_LEN else []
        print(f"EXTRACT {name}: {pages_used} pages, {sum(1 for r in rows if r['id'].startswith('bk-'+name))} blocks", flush=True)
        doc.close()

    with open("/content/seeds_books.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"TOTAL {len(rows)} blocks -> /content/seeds_books.jsonl", flush=True)


if __name__ == "__main__":
    main()
