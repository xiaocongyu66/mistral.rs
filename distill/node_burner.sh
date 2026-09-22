#!/bin/bash
# classifier.dev burner with per-node IP rotation via sing-box clash API.
# Usage: node_burner.sh <seed_file> <prefix>
# - On daily limit, blacklist the node and switch to the next live one.
# - CF-egress nodes (WARP pool) rotate their egress IP, so their blacklist
#   entries expire after 1h; fixed datacenter nodes stay blacklisted.
cd /root/mistral.rs/distill
SEEDS="$1"; PREFIX="$2"
NODES_FILE=/root/alive_nodes.json
BLACKLIST=/root/burned_nodes.json
[ -f "$BLACKLIST" ] || echo '{}' > "$BLACKLIST"

node_tag() {
    python3 - "$1" <<'EOF'
import json, sys
print(json.load(open("/root/alive_nodes.json"))[int(sys.argv[1])]["tag"])
EOF
}

is_blacklisted() {
    python3 - "$1" <<'EOF'
import json, sys, time
tag = sys.argv[1]
bl = json.load(open("/root/burned_nodes.json"))
e = bl.get(tag)
if not e:
    print("no"); raise SystemExit
cf = e.get("cf", False)
if cf and time.time() - e["t"] > 3600:
    print("no")   # egress rotated: treat as usable again
else:
    print("yes")
EOF
}

blacklist_node() {
    python3 - "$1" <<'EOF'
import json, sys, time
tag = sys.argv[1]
nodes = json.load(open("/root/alive_nodes.json"))
cf = any(n["tag"] == tag and n["ip"].startswith("104.28.") for n in nodes)
bl = json.load(open("/root/burned_nodes.json"))
bl[tag] = {"t": time.time(), "cf": cf}
json.dump(bl, open("/root/burned_nodes.json", "w"))
print(f"[burner] blacklisted {tag} (cf={cf})")
EOF
}

next_node() {
    local N=$(python3 -c "import json;print(len(json.load(open('$NODES_FILE'))))")
    local tries=0
    while [ $tries -lt $N ]; do
        local tag=$(node_tag $NODE_IDX)
        NODE_IDX=$(( (NODE_IDX+1) % N ))
        if [ "$(is_blacklisted "$tag")" = "no" ]; then
            curl -sS -m 5 -X PUT http://127.0.0.1:9090/proxies/POOL \
                 -H "Content-Type: application/json" -d "{\"name\": \"$tag\"}" > /dev/null
            echo "[burner] -> node: $tag"
            return 0
        fi
        tries=$((tries+1))
    done
    echo "[burner] WARN: all nodes blacklisted; waiting 600s for quota/egress rotation"
    sleep 600
    echo '{}' > "$BLACKLIST"
}

ROUND=0
NODE_IDX=0
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
    if grep -q "DAILY_LIMIT" /tmp/burn_${PREFIX}_r${ROUND}.log; then
        blacklist_node "$(node_tag $(( (NODE_IDX-1) % $(python3 -c "import json;print(len(json.load(open('$NODES_FILE'))))") )))"
    fi
    next_node
done
