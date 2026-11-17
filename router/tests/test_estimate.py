"""Prompt estimate (whole messages + tools), calibration, exact count near max_ctx (no network but a local fake)."""
import importlib.util, tempfile, asyncio, json
import os as _os
sp = importlib.util.spec_from_file_location("r", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "llmt-router.py"))
r = importlib.util.module_from_spec(sp); sp.loader.exec_module(r)
r.DRAIN_DIR = tempfile.mkdtemp()
tools = [{"type": "function", "function": {"name": "search_docs", "description": "x" * 3000, "parameters": {}}}]
msgs = [{"role": "user", "content": "q" * 3300}, {"role": "assistant", "content": None,
        "tool_calls": [{"id": "1", "type": "function", "function": {"name": "search_docs", "arguments": "a" * 6600}}]}]
c0 = r.prefix_chain("/v1/chat/completions", {"messages": msgs}, 3.3)[-1][1]
c1 = r.prefix_chain("/v1/chat/completions", {"messages": msgs, "tools": tools}, 3.3)[-1][1]
assert c0 > 3000 and c1 > c0 + 900, (c0, c1); print(f"ok   tool_calls + tools counted ({c0} -> {c1})")
h0 = r.prefix_chain("/v1/chat/completions", {"messages": msgs, "tools": tools}, 3.3)[0][0]
h1 = r.prefix_chain("/v1/chat/completions", {"messages": msgs, "tools": tools}, 2.0)[0][0]
assert h0 == h1; print("ok   hashes independent of the estimate")
a = r.Backend({"name": "arc", "url": "http://127.0.0.1:18312", "max_ctx": 196608})
g = r.Backend({"name": "rtx5090", "url": "http://127.0.0.1:18311", "max_ctx": 124800, "prefer_until": 2})
for b in (a, g): b.healthy = True
R = r.Router([a, g])
for _ in range(60): R.calibrate(100000, 130000)
assert 1.28 < R.est_scale < 1.31, R.est_scale; print(f"ok   calibration -> {R.est_scale:.3f}")
o, why, _ = R.choose([(b"x", 10), (b"", 100000)], 2000); assert o[0] is a and why == "ctx", (o[0].name, why); print("ok   scaled estimate 130k -> arc")
o, why, _ = R.choose([(b"x", 10), (b"", 100000)], 2000, exact=110000); assert why != "ctx" and o[0] is g, (o[0].name, why); print("ok   exact 110k + 2k fits the 5090 (no margin)")
async def main():
    async def h(rd, wr):
        hd = await rd.readuntil(b"\r\n\r\n"); n = int([l for l in hd.split(b"\r\n") if l.lower().startswith(b"content-length")][0].split(b":")[1])
        await rd.readexactly(n); body = json.dumps({"count": 123456}).encode()
        wr.write(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n" % len(body) + body); await wr.drain(); wr.close()
    srv = await asyncio.start_server(h, "127.0.0.1", 18312)
    S = r.Server.__new__(r.Server); S.r = R
    n = await S.exact_tokens("/v1/chat/completions", {"model": "m", "messages": msgs}, [(b"x", 10), (b"", 90000)], 2000)
    assert n == 123456, n; print("ok   near the limit: exact count from the largest backend's /tokenize")
    n = await S.exact_tokens("/v1/chat/completions", {"model": "m", "messages": msgs}, [(b"x", 10), (b"", 20000)], 2000)
    assert n is None; print("ok   far below the limit: no /tokenize call")
    srv.close()
asyncio.run(main())
print("ALL PASS")
