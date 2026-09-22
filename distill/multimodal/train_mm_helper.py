"""Shared helper for jev labeling via knox systemone."""
import json
import os
import time
import urllib.request

KEY = os.environ.get("KNOX_KEY", "")
URL = "https://api.knox.chat/v1/systemone"


def knox_probe(state, question, labels, max_len=4000):
    if not KEY:
        raise RuntimeError("KNOX_KEY not set")
    crit = {lab: lab for lab in labels}
    payload = {"model": "jev-latest",
               "state": state[:max_len],
               "questions": {"q": {"text": question, "type": "choice",
                                    "instructions": question,
                                    "criteria": crit}}}
    body = json.dumps(payload).encode()
    for attempt in range(6):
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
        except Exception:
            time.sleep(2 ** attempt)
    raise RuntimeError("knox probe failed after retries")
