#!/bin/bash
# classifier.dev burner with per-node IP rotation via sing-box clash API.
# Usage: node_burner.sh <seed_file> <prefix>
# On daily limit, switch to the next alive node (fresh IP = fresh quota).
cd /root/mistral.rs/distill
SEEDS="$1"; PREFIX="$2"
NODES_FILE=/root/alive_nodes.json

next_node() {
    local idx=$((NODE_IDX % $(python3 -c "import json;print(len(json.load(open('$NODES_FILE'))))")))
    local tag=$(python3 -c "import json;print(json.load(open('$NODES_FILE'))[$idx]['tag'])")
    curl -sS -m 5 -X PUT http://127.0.0.1:9090/proxies/POOL \
         -H "Content-Type: application/json" -d "{\"name\": \"$tag\"}"
    echo "[burner] switched to node $idx: $tag"
    NODE_IDX=$((NODE_IDX+1))
}

NODE_IDX=0
ROUND=0
FULL_SWEEPS=0
while true; do
    ROUND=$((ROUND+1))
    SKIP="$(ls data/*.jsonl | tr '\n' ',' | sed 's/,$//')"
    python3 generate_fast.py --seeds "$SEEDS" --budget 19800 --concurrency 2 --pace 4.0 \
        --prefix "${PREFIX}-n${NODE_IDX}r${ROUND}" --skip-done "$SKIP" \
        > /tmp/burn_${PREFIX}_r${ROUND}.log 2>&1
    RC=$?
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
    echo "[burner] round $ROUND rc=$RC remaining=$REMAIN"
    if [ "$REMAIN" -eq 0 ]; then
        echo "[burner] ALL DONE for $SEEDS"
        break
    fi
    N_TOTAL=$(python3 -c "import json;print(len(json.load(open('$NODES_FILE'))))")
    if [ $((NODE_IDX % N_TOTAL)) -eq 0 ] && [ $ROUND -gt 1 ]; then
        FULL_SWEEPS=$((FULL_SWEEPS+1))
        echo "[burner] full sweep $FULL_SWEEPS done; quotas reset on rolling windows, waiting 900s"
        sleep 900
    fi
    next_node
done
