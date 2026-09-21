"""Download open-access decision-theory books, extract text, emit seed blocks.

Books with confirmed open-access URLs only. Extracted text stays local (VM);
only scripts and row counts are committed to git (license hygiene).
"""
import json
import os
import subprocess
import urllib.request

BOOKS = [
    ("alg_dm", "https://algorithmsbook.com/files/dm.pdf", "Algorithms for Decision Making (MIT Press, CC-BY-NC-ND)"),
    ("rl_sutton", "http://incompleteideas.net/book/RLbook2020.pdf", "Sutton & Barto RL 2nd ed (free PDF, copyright retained)"),
    ("rl_bertsekas", "https://web.mit.edu/dimitrib/www/RLlO.pdf", "Bertsekas RL & Optimal Control 2nd (free from author)"),
    ("llm_fudan", "https://raw.githubusercontent.com/datawhalechina/so-large-lm/main/docs/chapter1.pdf", "Fudan LLM textbook chapter probe"),
]

os.makedirs("/content/books", exist_ok=True)


def download(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 100_000:
        return "cached"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
            f.write(r.read())
        return "ok"
    except Exception as e:
        return f"ERR {str(e)[:120]}"


def main():
    for name, url, note in BOOKS:
        dest = f"/content/books/{name}.pdf"
        status = download(url, dest)
        size = os.path.getsize(dest) if os.path.exists(dest) else 0
        print(f"DL {name}: {status} {round(size/2**20,1)}MB ({note})", flush=True)


if __name__ == "__main__":
    main()
