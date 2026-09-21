#!/bin/bash
# Loop a classifier.dev labeling run until the seed file is exhausted.
# On rate_limit_day the run aborts; the proxy pool rotates IPs, so wait
# and retry with skip-done (idempotent) until todo hits zero.
cd /root/mistral.rs/distill
SEEDS="$1"; PREFIX="$2"; BUDGET="${3:-19800}"
SKIP="$(ls data/*.jsonl | tr '\n' ',' | sed 's/,$//')"
ROUND=0
while true; do
    ROUND=$((ROUND+1))
    python3 generate_fast.py --seeds $SEEDS --budget "$BUDGET" --concurrency 2 --pace 4.0 \
        --prefix "${PREFIX}-r${ROUND}" --skip-done "$SKIP" > /tmp/burn_${PREFIX}_r${ROUND}.log 2>&1
    RC=$?
    if grep -q "DAILY_LIMIT" /tmp/burn_${PREFIX}_r${ROUND}.log; then
        echo "[burn] round $ROUND hit daily limit, waiting 180s for IP rotation"
        sleep 180
        continue
    fi
    if [ $RC -ne 0 ]; then
        echo "[burn] round $ROUND failed rc=$RC, waiting 120s"
        sleep 120
        continue
    fi
    REMAIN=$(python3 - "$SEEDS" <<'EOF'
import json, sys, glob
done = set()
for p in glob.glob("data/*.jsonl"):
    for line in open(p, encoding="utf-8"):
        try: done.add(json.loads(line)["id"])
        except: pass
rem = 0
for line in open(sys.argv[1], encoding="utf-8"):
    if line.strip():
        try:
            if json.loads(line)["id"] not in done: rem += 1
        except: rem += 1
print(rem)
EOF
)
    echo "[burn] round $ROUND done, remaining=$REMAIN"
    if [ "$REMAIN" -eq 0 ]; then
        echo "[burn] ALL DONE"
        break
    fi
done
