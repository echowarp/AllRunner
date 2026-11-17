"""LORA-SUFFIX: "<model>:<adapter>" (or body "lora" / header X-LLMT-LoRA) goes only to backends with lora true, with the
adapter name as the engine's model name; bad names -> 400, no LoRA backend -> 503, unknown adapter -> the engine's 404
passes through; plain requests are untouched and may use every backend. Fake backends, no GPU."""
import asyncio
import importlib.util
import json
import os
import sys
import tempfile

sp = importlib.util.spec_from_file_location("r", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "llmt-router.py"))
r = importlib.util.module_from_spec(sp)
sp.loader.exec_module(r)
r.DRAIN_DIR = tempfile.mkdtemp()
fails = []


def check(cond, what):
    print(("ok   " if cond else "FAIL ") + what)
    if not cond:
        fails.append(what)


def unit():
    req = {"model": "qwen3.8-27b:terse", "messages": []}
    check(r.lora_select(req, [], ":") == ("terse", None) and req["model"] == "terse", "suffix -> adapter, model rewritten")
    req = {"model": "qwen3.8-27b", "lora": "terse", "messages": []}
    check(r.lora_select(req, [], ":") == ("terse", None) and req["model"] == "terse" and "lora" not in req,
          "body field 'lora' -> adapter, field removed")
    req = {"model": "qwen3.8-27b", "messages": []}
    check(r.lora_select(req, [("X-LLMT-LoRA", "terse")], ":") == ("terse", None), "header X-LLMT-LoRA -> adapter")
    req = {"model": "qwen3.8-27b", "messages": []}
    check(r.lora_select(req, [], ":") == (None, None) and req == {"model": "qwen3.8-27b", "messages": []},
          "no adapter -> request untouched")
    for bad in ("qwen3.8-27b:../x", "qwen3.8-27b:", "qwen3.8-27b:a b", ":terse", "qwen3.8-27b:.hidden", "m:" + "a" * 65):
        a, e = r.lora_select({"model": bad}, [], ":")
        check(a is None and e, f"bad name {bad[:30]!r} refused ({e})")
    a, e = r.lora_select({"model": "m:x", "lora": "y"}, [], ":")
    check(a is None and "conflicting" in e, "suffix and field disagree -> refused")
    a, e = r.lora_select({"model": "m:x", "lora": 3}, [], ":")
    check(a is None and e, "non-string lora field refused")
    check(r.lora_select({"model": "qwen3.8-27b:terse"}, [], "") == (None, None), "lora_sep '' turns the suffix off")
    b = r.Backend({"name": "a", "url": "http://x:1", "lora": True})
    check(b.lora, "backend lora: true (JSON bool)")
    check(r.Backend("name=a,url=http://x:1,lora=1").lora and not r.Backend("name=a,url=http://x:1").lora,
          "backend lora=1 (CLI spec) / default off")
    b.update({"lora": False})
    check(not b.lora, "hot reload turns lora off")


async def e2e():
    seen = {"arc": [], "rtx5090": []}

    def mk(name):
        async def h(rd, wr):
            head = await r.read_head(rd)
            line, hdrs = head
            body = await r.read_body(rd, hdrs)
            path = line.split(" ")[1]
            if path == "/health":
                wr.write(r.simple("200 OK", b"{}"))
            else:
                req = json.loads(body)
                seen[name].append(req)
                if req["model"] == "nope":
                    wr.write(r.simple("404 Not Found", json.dumps({"error": {"message": "LoRA adapter nope does not exist",
                                                                        "type": "NotFoundError", "code": 404}}).encode()))
                else:
                    wr.write(r.simple("200 OK", json.dumps({"backend": name, "model": req["model"],
                                                            "usage": {"prompt_tokens": 5, "completion_tokens": 3}}).encode()))
            await wr.drain()
            wr.close()
        return h
    sa = await asyncio.start_server(mk("arc"), "127.0.0.1", 0)
    sb = await asyncio.start_server(mk("rtx5090"), "127.0.0.1", 0)
    pa, pb = sa.sockets[0].getsockname()[1], sb.sockets[0].getsockname()[1]
    A = r.Backend({"name": "arc", "url": f"http://127.0.0.1:{pa}", "lora": True, "prefill": 2000})
    B = r.Backend({"name": "rtx5090", "url": f"http://127.0.0.1:{pb}", "prefill": 6000, "prefer_until": 8})
    R = r.Router([A, B])
    S = r.Server(R, health_interval=0.2)
    s = await asyncio.start_server(S.handle, "127.0.0.1", 0)
    port = s.sockets[0].getsockname()[1]
    hl = asyncio.create_task(S.health_loop())
    await asyncio.sleep(0.5)

    async def post(obj, hdrs=None):
        rd, wr = await asyncio.open_connection("127.0.0.1", port)
        raw = json.dumps(obj).encode()
        hh = "".join(f"{k}: {v}\r\n" for k, v in (hdrs or {}).items())
        wr.write(f"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n{hh}"
                 f"Content-Length: {len(raw)}\r\n\r\n".encode() + raw)
        await wr.drain()
        data = await rd.read()
        wr.close()
        head, _, rest = data.partition(b"\r\n\r\n")
        return int(head.split(b" ")[1]), rest
    msg = [{"role": "user", "content": "hi"}]
    st, body = await post({"model": "qwen3.8-27b", "messages": msg})
    check(st == 200 and json.loads(body)["backend"] == "rtx5090", "plain request: preferred 5090 (lora not needed)")
    for i in range(3):
        st, body = await post({"model": "qwen3.8-27b:terse", "messages": msg + [{"role": "user", "content": str(i)}]})
        j = json.loads(body)
        check(st == 200 and j["backend"] == "arc" and j["model"] == "terse",
              f"suffix request {i} -> arc only, engine sees model 'terse' ({j})")
    st, body = await post({"model": "qwen3.8-27b", "lora": "terse", "messages": msg})
    check(st == 200 and json.loads(body)["backend"] == "arc" and "lora" not in seen["arc"][-1], "body field -> arc, field stripped")
    st, body = await post({"model": "qwen3.8-27b", "messages": msg}, {"X-LLMT-LoRA": "terse"})
    check(st == 200 and seen["arc"][-1]["model"] == "terse", "header -> arc with model 'terse'")
    st, body = await post({"model": "qwen3.8-27b:nope", "messages": msg})
    check(st == 404 and b"does not exist" in body, f"unknown adapter -> engine 404 passed through ({st})")
    st, body = await post({"model": "qwen3.8-27b:../etc", "messages": msg})
    check(st == 400 and b"invalid LoRA adapter name" in body, f"bad adapter name -> 400 ({st})")
    open(os.path.join(r.DRAIN_DIR, "drain-arc"), "w").close()
    await asyncio.sleep(0.1)
    st, body = await post({"model": "qwen3.8-27b:terse", "messages": msg})
    check(st == 200 and json.loads(body)["backend"] == "arc", f"drained LoRA backend is still the last resort ({st})")
    open(os.path.join(r.DRAIN_DIR, "hold-arc"), "w").close()
    st, body = await post({"model": "qwen3.8-27b:terse", "messages": msg})
    check(st == 503 and b"no available backend serves LoRA" in body, f"LoRA backend held+drained -> 503, never the 5090 ({st})")
    check(not any(q["model"] in ("terse", "nope") for q in seen["rtx5090"]), "the non-LoRA backend never saw an adapter request")
    st, body = await post({"model": "qwen3.8-27b", "messages": msg})
    check(st == 200 and json.loads(body)["backend"] == "rtx5090", "plain traffic unaffected")
    m = S.metrics().decode()
    check('llmt_router_lora_requests_total{outcome="routed"}' in m and 'outcome="no-backend"} 1' in m
          and 'outcome="bad-name"} 1' in m, "lora metrics exported")
    # prefix separation: the adapter is part of the chain root (KV for a LoRA request differs from the base's)
    c1 = r.prefix_chain("/v1/chat/completions", {"model": "qwen3.8-27b", "messages": msg}, 3.3)
    c2 = r.prefix_chain("/v1/chat/completions", {"model": "terse", "messages": msg}, 3.3)
    check(c1[0][0] != c2[0][0], "base and adapter requests have different prefix hashes")
    hl.cancel()
    for x in (s, sa, sb):
        x.close()


unit()
asyncio.run(e2e())
print("ALL PASS" if not fails else f"{len(fails)} FAILED")
sys.exit(1 if fails else 0)
