#!/usr/bin/env python3
"""Probe proxy nodes one at a time in an isolated sing-box instance.

Never touches the main pool (55 nodes serving the burner on 7897/9090).
Each candidate gets its own single-outbound config on 7898, gets one
chance to fetch an IP through itself, and dies immediately after.
"""
import json, subprocess, sys, time, urllib.request

SRC = sys.argv[1] if len(sys.argv) > 1 else "/tmp/sub2_real.json"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/root/sub2_alive.json"
CFG = "/root/verify_node.json"

def build_cfg(ob):
    return {
        "log": {"level": "fatal"},
        "inbounds": [{"type": "mixed", "tag": "in", "listen": "127.0.0.1", "listen_port": 7898}],
        "outbounds": [ob],
        "route": {"final": ob["tag"], "default_interface": "wlan0"},
    }


def clash_to_singbox(n):
    """Minimal Clash->sing-box conversion (same rules as the pool builder)."""
    t = n["type"]
    ob = {"tag": n["name"], "server": str(n["server"]), "server_port": int(n["port"])}
    sni = n.get("sni") or n.get("servername") or n["server"]
    insecure = bool(n.get("skip-cert-verify"))
    tls = {"enabled": True, "insecure": insecure, "server_name": str(sni)}
    if t == "vless":
        ob["type"] = "vless"
        ob["uuid"] = n["uuid"]
        if n.get("reality-opts"):
            tls["reality"] = {"enabled": True,
                              "public_key": n["reality-opts"]["public-key"],
                              "short_id": n["reality-opts"].get("short-id", "")}
            tls["utls"] = {"enabled": True, "fingerprint": "chrome"}
        ob["tls"] = tls
        if n.get("flow"):
            ob["flow"] = str(n["flow"])
    elif t == "anytls":
        ob["type"] = "anytls"; ob["password"] = str(n.get("password") or ""); ob["tls"] = tls
    elif t == "hysteria2":
        ob["type"] = "hysteria2"; ob["password"] = str(n.get("password") or ""); ob["tls"] = tls
    elif t == "trojan":
        ob["type"] = "trojan"; ob["password"] = str(n["password"]); ob["tls"] = tls
    elif t == "ss":
        ob["type"] = "shadowsocks"; ob["method"] = n["cipher"]; ob["password"] = str(n["password"])
    elif t == "vmess":
        ob["type"] = "vmess"; ob["uuid"] = n["uuid"]
        ob["security"] = n.get("cipher") or "auto"; ob["alter_id"] = int(n.get("alterId") or 0)
        if n.get("tls"):
            ob["tls"] = tls
    else:
        return None
    net = n.get("network")
    if net == "ws":
        hdr = n.get("ws-headers") or {}
        ob["transport"] = {"type": "ws", "path": str(n.get("ws-path") or n.get("path") or "/"),
                           "headers": ({"Host": str(hdr["Host"])} if hdr.get("Host") else {})}
    elif net == "grpc":
        ob["transport"] = {"type": "grpc", "service_name": str(n.get("grpc-service-name") or "")}
    return ob

def probe(ob, timeout=10):
    open(CFG, "w").write(json.dumps(build_cfg(ob)))
    p = subprocess.Popen(["sing-box", "run", "-c", CFG],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        # let the inbound come up
        ok = False
        for _ in range(20):
            try:
                urllib.request.urlopen("http://127.0.0.1:7898", timeout=1)
            except Exception:
                pass
            time.sleep(0.25)
        PH = urllib.request.ProxyHandler({"http": "http://127.0.0.1:7898",
                                          "https": "http://127.0.0.1:7898"})
        op = urllib.request.build_opener(PH)
        with op.open("http://ip-api.com/json/?fields=query,country", timeout=timeout) as r:
            return json.load(r)
    finally:
        p.kill()
        p.wait()

nodes = json.load(open(SRC))
alive = []
for i, n in enumerate(nodes):
    tag = n["name"] if "name" in n else n["tag"]
    try:
        ob = n if "tag" in n else clash_to_singbox(n)
        if ob is None:
            print(f"--  [{i+1}/{len(nodes)}] {tag}: unsupported type {n['type']}", flush=True)
            continue
        res = probe(ob)
        alive.append({"tag": tag, "ip": res["query"], "country": res["country"],
                      "outbound": ob})
        print(f"OK  [{i+1}/{len(nodes)}] {tag}: {res['query']} ({res['country']})", flush=True)
    except Exception as e:
        print(f"--  [{i+1}/{len(nodes)}] {tag}: {str(e)[:50]}", flush=True)

json.dump(alive, open(OUT, "w"), ensure_ascii=False, indent=1)
print(f"ALIVE: {len(alive)}/{len(nodes)} -> {OUT}", flush=True)
