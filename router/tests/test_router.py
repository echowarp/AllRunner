#!/usr/bin/env python3
"""Offline test of llmt-router.py with two fake vLLM backends (no GPU, synthetic prompts only).
Checks: affinity for follow-up turns, new conversations spread by load, steal when the affinity backend is saturated,
failover when a backend is down, drain files, streaming passthrough, /health, /router/metrics.
  python3 tests/test_router.py"""
import asyncio
import importlib.util
import json
import os
import pathlib
import sys
import tempfile

HERE = pathlib.Path(__file__).parent
spec = importlib.util.spec_from_file_location("router", HERE.parent / "llmt-router.py")
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)


class Fake:
    """Minimal vLLM stand-in: /health, /v1/models, /v1/chat/completions (SSE when stream=true), records who it served."""

    def __init__(self, name, delay=0.0):
        self.name, self.delay, self.served, self.up = name, delay, [], True
        self.ktp = []   # kv_transfer_params seen per request
        self.reqs = []  # (last message, max_tokens, stream) per request (split-mode checks)

    async def handle(self, rd, wr):
        head = await R.read_head(rd)
        if head is None:
            wr.close()
            return
        line, hdrs = head
        body = await R.read_body(rd, hdrs)
        path = line.split(" ")[1]
        if not self.up:
            wr.close()
            return
        if path == "/health":
            wr.write(R.simple("200 OK", b""))
        elif path == "/v1/models":
            wr.write(R.simple("200 OK", json.dumps({"backend": self.name}).encode()))
        else:
            req = json.loads(body)
            self.served.append(req["messages"][-1]["content"])
            self.ktp.append(req.get("kv_transfer_params"))
            self.reqs.append((req["messages"][-1]["content"], req.get("max_tokens"), req.get("stream")))
            await asyncio.sleep(self.delay)
            if req.get("stream"):
                wr.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n")
                for i in range(3):
                    ev = f"data: {json.dumps({'backend': self.name, 'i': i})}\n\n".encode()
                    wr.write(f"{len(ev):x}\r\n".encode() + ev + b"\r\n")
                    await wr.drain()
                    await asyncio.sleep(0.01)
                wr.write(b"0\r\n\r\n")
            else:
                wr.write(R.simple("200 OK", json.dumps({"backend": self.name}).encode()))
        await wr.drain()
        wr.close()


async def http(port, method, path, obj=None):
    rd, wr = await asyncio.open_connection("127.0.0.1", port)
    body = json.dumps(obj).encode() if obj is not None else b""
    wr.write(f"{method} {path} HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: {len(body)}\r\n"
             f"\r\n".encode() + body)
    await wr.drain()
    data = await rd.read()
    wr.close()
    head, _, rest = data.partition(b"\r\n\r\n")
    return int(head.split(b" ")[1]), head.decode(), rest


def conv(sys_prompt, turns):
    msgs = [{"role": "system", "content": sys_prompt}]
    for t in turns:
        msgs.append({"role": "user", "content": t})
        msgs.append({"role": "assistant", "content": "ok " + t})
    return msgs[:-1]


async def main():
    tmp = tempfile.mkdtemp()
    R.DRAIN_DIR = tmp
    fa, fb = Fake("arc", delay=0.3), Fake("rtx5090", delay=0.3)
    sa = await asyncio.start_server(fa.handle, "127.0.0.1", 0)
    sb = await asyncio.start_server(fb.handle, "127.0.0.1", 0)
    pa, pb = sa.sockets[0].getsockname()[1], sb.sockets[0].getsockname()[1]
    router = R.Router([R.Backend(f"name=arc,url=http://127.0.0.1:{pa},cap=2,prefill=2000,stream_cost=0.6"),
                       R.Backend(f"name=rtx5090,url=http://127.0.0.1:{pb},cap=4,prefill=6000,stream_cost=0.3")])
    srv = R.Server(router, health_interval=0.2)
    router.trace_dir = os.path.join(tmp, "trace")   # per-request JSONL trace
    s = await asyncio.start_server(srv.handle, "127.0.0.1", 0)
    port = s.sockets[0].getsockname()[1]
    hl = asyncio.create_task(srv.health_loop())
    await asyncio.sleep(0.5)
    fails = []

    def check(cond, what):
        print(("ok   " if cond else "FAIL ") + what)
        if not cond:
            fails.append(what)

    st, _, body = await http(port, "GET", "/health")
    check(st == 200, "/health 200 with both backends up")
    long_sys = "x" * 60000                                    # ~18k-token shared system prompt + tools

    # 1. a new long conversation goes to the faster-prefill idle backend (5090); its follow-ups stick there
    st, _, body = await http(port, "POST", "/v1/chat/completions", {"model": "m", "messages": conv(long_sys, ["a1"])})
    first = json.loads(body)["backend"]
    check(st == 200 and first == "rtx5090", f"new long conversation -> rtx5090 (got {first})")
    for t in (["a1", "a2"], ["a1", "a2", "a3"]):
        st, _, body = await http(port, "POST", "/v1/chat/completions", {"model": "m", "messages": conv(long_sys, t)})
        check(json.loads(body)["backend"] == first, f"follow-up turn {len(t)} stays on {first}")

    # 2. a conversation started on the Arc (5090 drained) stays on the Arc after the drain is lifted
    open(os.path.join(tmp, "drain-rtx5090"), "w").close()
    st, _, body = await http(port, "POST", "/v1/chat/completions", {"model": "m", "messages": conv("y" * 30000, ["b1"])})
    check(json.loads(body)["backend"] == "arc", "drained rtx5090 -> new conversation on arc")
    os.remove(os.path.join(tmp, "drain-rtx5090"))
    st, _, body = await http(port, "POST", "/v1/chat/completions", {"model": "m", "messages": conv("y" * 30000, ["b1", "b2"])})
    check(json.loads(body)["backend"] == "arc", "follow-up stays on arc after the drain is lifted (affinity)")

    # 2b. prefer_until: with rtx5090.prefer_until = 8 the Arc's own conversation is routed to the 5090 while
    #     it has fewer than 8 in flight (reason "prefer"); at 8 in flight the cost model decides again
    b5 = router.by_name["rtx5090"]; b5.prefer_until = 8
    st, _, body = await http(port, "POST", "/v1/chat/completions", {"model": "m", "messages": conv("y" * 30000, ["b1", "b2", "b3"])})
    check(json.loads(body)["backend"] == "rtx5090", "prefer_until 8: the Arc's conversation goes to the idle 5090")
    check(b5.n.get("prefer", 0) >= 1, f"reason 'prefer' counted on rtx5090 ({dict(b5.n)})")
    b5.inflight += 8
    o, reason, _ = router.choose([(b"h", 100), (b"h2", 200)])
    check(reason != "prefer", f"at 8 in flight the cost model decides (reason {reason})")
    b5.inflight -= 8; b5.prefer_until = 0
    # 3. saturate the 5090: a follow-up of a conversation that lives there is stolen by the idle Arc only when it pays
    fb.delay = 1.5
    hold = [asyncio.create_task(http(port, "POST", "/v1/chat/completions",
                                     {"model": "m", "messages": conv(f"z{i}" * 2000, [f"c{i}"])})) for i in range(6)]
    await asyncio.sleep(0.3)
    st, _, body = await http(port, "POST", "/v1/chat/completions", {"model": "m", "messages": conv(long_sys, ["a1", "a2", "a3", "a4"])})
    got = json.loads(body)["backend"]
    print(f"     (5090 in flight {router.by_name['rtx5090'].inflight}, arc {router.by_name['arc'].inflight}) -> {got}")
    res = await asyncio.gather(*hold)
    spread = [json.loads(b)["backend"] for _, _, b in res]
    check("arc" in spread and "rtx5090" in spread, f"6 concurrent new conversations spread over both GPUs ({spread})")
    fb.delay = 0.3

    # 4. streaming passthrough (chunked SSE) and backend down -> failover
    st, head, body = await http(port, "POST", "/v1/chat/completions",
                                {"model": "m", "stream": True, "messages": conv(long_sys, ["a1", "a2", "a3", "a4", "a5"])})
    check(st == 200 and body.count(b"data: ") == 3 and "chunked" in head.lower(), "SSE stream passes through intact")
    owner = json.loads(body.split(b"data: ")[1].split(b"\n")[0])["backend"]
    (fa if owner == "arc" else fb).up = False
    await asyncio.sleep(0.8)
    st, _, body = await http(port, "POST", "/v1/chat/completions", {"model": "m", "messages": conv(long_sys, ["a1", "a2", "a3", "a4", "a5", "a6"])})
    other = "rtx5090" if owner == "arc" else "arc"
    check(st == 200 and json.loads(body)["backend"] == other, f"affinity backend {owner} down -> served by {other}")
    (fa if owner == "arc" else fb).up = True

    # 4b. shared-tier namespace injection: off by default, merged into client params, client's llmt_ns wins
    seen = lambda: (fa.ktp + fb.ktp)[-1] if (fa.ktp or fb.ktp) else None
    last = lambda f: f.ktp[-1] if f.ktp else None
    st, _, _ = await http(port, "POST", "/v1/chat/completions", {"model": "m", "messages": conv("inj", ["q0"])})
    check(st == 200 and all(k is None for k in fa.ktp + fb.ktp), "no kv_transfer_params injected while inject_kv_ns is off")
    router.apply_config({"inject_kv_ns": "shared"})
    n0 = len(fa.ktp) + len(fb.ktp)
    st, _, _ = await http(port, "POST", "/v1/chat/completions", {"model": "m", "messages": conv("inj", ["q1"]),
                                                                  "kv_transfer_params": {"llmt_nostore": True}})
    got = [k for k in (fa.ktp + fb.ktp)[n0:]]
    check(st == 200 and got and got[-1] == {"llmt_nostore": True, "llmt_ns": "shared"}, "llmt_ns injected, client params kept")
    st, _, _ = await http(port, "POST", "/v1/chat/completions", {"model": "m", "messages": conv("inj", ["q2"]),
                                                                  "kv_transfer_params": {"llmt_ns": "xgpu-test"}})
    got = [k for k in (fa.ktp + fb.ktp)[n0:]]
    check(st == 200 and got[-1] == {"llmt_ns": "xgpu-test"}, "client's own llmt_ns wins")
    router.apply_config({"inject_kv_ns": ""})

    # 4c. split mode: decode GPU busy + other GPU idle -> prefill-only copy on the idle GPU, then the
    # full request on the decode GPU; no split when the decode GPU is idle, when it doesn't pay, or for images
    router.apply_config({"inject_kv_ns": "shared", "split": {"on": True, "min_tokens": 1000, "overhead_s": 0.0,
                                                               "store_wait_s": 0.05, "weight": 0.1}})
    na, nb = len(fa.reqs), len(fb.reqs)
    st, _, body = await http(port, "POST", "/v1/chat/completions", {"model": "m", "messages": conv("s" * 60000, ["s1"])})
    check(st == 200 and len(fa.reqs) == na and json.loads(body)["backend"] == "rtx5090",
          "split on, decode GPU idle -> no split (served by rtx5090 directly)")
    fb.delay = 2.0
    open(os.path.join(tmp, "drain-arc"), "w").close()          # pin the long-running request to the 5090
    held = asyncio.create_task(http(port, "POST", "/v1/chat/completions", {"model": "m", "messages": conv("w" * 2000, ["h"])}))
    await asyncio.sleep(0.3)
    os.remove(os.path.join(tmp, "drain-arc"))
    check(router.by_name["rtx5090"].inflight == 1 and router.by_name["arc"].inflight == 0, "setup: 5090 busy, Arc idle")
    for bk in router.backends:                                 # earlier steps left the estimated prefill queues busy
        bk.prefill_busy_until = 0.0
    na, nb = len(fa.reqs), len(fb.reqs)
    st, _, body = await http(port, "POST", "/v1/chat/completions",
                             {"model": "m", "stream": True, "max_tokens": 500, "messages": conv("t" * 60000, ["t1"])})
    pre = fa.reqs[na:]
    check(st == 200 and len(pre) == 1 and pre[0][0] == "t1" and pre[0][1] == 1 and pre[0][2] is None,
          f"5090 busy, Arc idle -> prefill-only copy on arc (max_tokens 1, not streamed) {pre}")
    check(fa.ktp[-1] == {"llmt_ns": "shared"}, "prefill copy carries the shared namespace")
    full = [r for r in fb.reqs[nb:] if r[0] == "t1"]
    check(body.count(b"data: ") == 3 and full and full[0][1] == 500 and full[0][2] is True,
          "then the full streamed request on rtx5090")
    na = len(fa.reqs)
    router.apply_config({"split": {"on": True, "min_tokens": 1000, "weight": 50.0}})
    st, _, body = await http(port, "POST", "/v1/chat/completions", {"model": "m", "messages": conv("u" * 60000, ["u1"])})
    check(st == 200 and not any(r[1] == 1 for r in fa.reqs[na:]), "no split when it doesn't pay (weight 50)")
    router.apply_config({"split": {"on": True, "min_tokens": 1000, "weight": 0.1, "store_wait_s": 0.05}})
    na = len(fa.reqs)
    img = conv("v" * 60000, ["v1"])
    img[-1]["content"] = [{"type": "text", "text": "v1"}, {"type": "image_url", "image_url": {"url": "data:x"}}]
    st, _, body = await http(port, "POST", "/v1/chat/completions", {"model": "m", "messages": img})
    check(st == 200 and not any(r[1] == 1 for r in fa.reqs[na:]), "no split for image requests")
    await held
    fb.delay = 0.3
    st, _, body = await http(port, "GET", "/router/metrics")
    check(b'llmt_router_split_total{result="ok"} 1' in body, "split counted in /router/metrics")
    router.apply_config({"inject_kv_ns": "", "split": {"on": False}})

    # 4d. the SSE tracker (pure parsing, split at odd byte boundaries; chunked framing interleaved)
    ev = lambda o: f"data: {json.dumps(o)}\n\n".encode()
    sse = (ev({"choices": [{"delta": {"reasoning": "a"}}]}) + ev({"choices": [{"delta": {"reasoning": "b"}}]})
           + ev({"choices": [{"delta": {"content": "c"}, "token_ids": [1, 2]}]})
           + ev({"choices": [{"delta": {"tool_calls": [{"function": {"name": "f"}}]}}]})
           + ev({"choices": [{"delta": {"tool_calls": [{"function": {"arguments": "{}"}}]}}]})
           + ev({"choices": [{"delta": {}, "finish_reason": "tool_calls"}], "usage": {"completion_tokens": 9}})
           + b"data: [DONE]\n\n")
    framed = b"".join(f"{len(sse[i:i + 37]):x}\r\n".encode() + sse[i:i + 37] + b"\r\n" for i in range(0, len(sse), 37))
    trk = R.StreamTracker(); safe_mid = None
    for i in range(0, len(sse), 7):
        trk.feed(sse[i:i + 7])
        if trk.tool_calls == 1 and trk.finish is None:
            safe_mid = trk.safe()
    check(trk.chunks == 6 and trk.reasoning == 2 and trk.content == 1 and trk.ids == 2 and trk.tool_calls == 1
          and trk.finish == "tool_calls" and trk.usage == {"completion_tokens": 9} and safe_mid is False and trk.safe() is False,
          f"StreamTracker counts (chunks {trk.chunks}, reasoning {trk.reasoning}, content {trk.content}, ids {trk.ids}, "
          f"tool_calls {trk.tool_calls}, finish {trk.finish}, safe inside tool call {safe_mid})")
    trk2 = R.StreamTracker(chunked=True)
    for i in range(0, len(framed), 11):
        trk2.feed(framed[i:i + 11])
    trk2.feed(b"0\r\n\r\n")
    check(trk2.chunks == 6 and trk2.finish == "tool_calls" and trk2.usage == {"completion_tokens": 9},
          f"tracker undoes HTTP chunked framing (chunks {trk2.chunks}, finish {trk2.finish})")
    trk3 = R.StreamTracker(); trk3.feed(ev({"choices": [{"delta": {"content": "x"}}]}))
    check(trk3.safe() is True and trk3.tokens == 1, "tracker: plain content stream is a safe point")

    # 5. metrics and state hold no content
    st, _, body = await http(port, "GET", "/router/metrics")
    check(st == 200 and b"llmt_router_requests_total" in body and b"affinity" in body, "/router/metrics")
    sm = [l for l in body.decode().splitlines() if l.startswith("llmt_router_stream_")]
    check(any(l.startswith('llmt_router_stream_streams_total{backend="rtx5090"} ') and int(l.split()[-1]) >= 1 for l in sm),
          f"the streamed requests were followed (stream metrics {sm})")
    st, _, body = await http(port, "GET", "/router/state")
    check(st == 200 and b"xxxx" not in body and b"a1" not in body, "/router/state has no prompt content")
    print(body.decode())
    hl.cancel()
    # hold flag: a held backend gets no traffic, not even as the fallback
    open(os.path.join(tmp, "hold-5090"), "w").close(); open(os.path.join(tmp, "drain-rtx5090"), "w").close()
    st, _, body = await http(port, "POST", "/v1/chat/completions", {"model": "m", "messages": conv("h" * 50000, ["held1"])})
    check(st == 200 and json.loads(body)["backend"] == "arc", "new long conversation avoids the held+drained rtx5090 -> arc")
    open(os.path.join(tmp, "drain-arc"), "w").close()
    st, _, body = await http(port, "POST", "/v1/chat/completions", {"model": "m", "messages": conv("h" * 50000, ["held2"])})
    check(st == 200 and json.loads(body)["backend"] == "arc", "drained arc still serves when the other GPU is held+drained (fallback excludes it)")
    fa.up = False
    await asyncio.sleep(0.6)
    st, _, body = await http(port, "POST", "/v1/chat/completions", {"model": "m", "messages": conv("h" * 50000, ["held3"])})
    check(st == 502, f"arc down + rtx5090 held+drained -> 502, no silent reroute (got {st})")
    fa.up = True
    os.remove(os.path.join(tmp, "drain-rtx5090"))        # hold stays, drain lifted = hand-started test stack: routable again
    await asyncio.sleep(0.6)
    st, _, body = await http(port, "POST", "/v1/chat/completions", {"model": "m", "messages": conv("h" * 50000, ["held4"])})
    check(st == 200 and json.loads(body)["backend"] == "rtx5090", f"held but undrained rtx5090 serves again (got {st} {body[:40]!r})")
    os.remove(os.path.join(tmp, "hold-5090")); os.remove(os.path.join(tmp, "drain-arc"))
    await asyncio.sleep(0.6)
    # trace: one JSON line per routed request, no content
    import glob as _glob
    tf = _glob.glob(os.path.join(tmp, "trace", "trace-*.jsonl"))
    recs = [json.loads(l) for f in tf for l in open(f) if l.strip()]
    check(len(recs) >= 10, f"trace has one line per routed request ({len(recs)})")
    ok_fields = recs and all(set(r) >= {"t", "backend", "code", "first_s", "done_s", "prompt_est", "conv", "reason"} for r in recs)
    check(bool(ok_fields), "trace lines carry t/backend/code/first_s/done_s/prompt_est/conv/reason")
    check(all(r["backend"] in ("arc", "rtx5090") and r["code"] == "200" for r in recs if r["code"] == "200"), "trace backends named")
    check(not any("messages" in r or "prompt" in r or "content" in json.dumps(r) for r in recs), "trace has no request content")
    strm = [r for r in recs if r.get("stream")]
    check(bool(strm) and all(r.get("finish") is not None or r.get("tool_calls") is not None for r in strm), f"streamed requests traced with stream fields ({len(strm)})")

    print("ALL PASS" if not fails else f"{len(fails)} FAILED")
    sys.exit(1 if fails else 0)


asyncio.run(main())
