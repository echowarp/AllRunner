"""Per-backend max_ctx and image limits, ctx-retry, balance_new, affinity_min/ttl, summary_target (local fakes only)."""
import importlib.util
import os as _os
sp = importlib.util.spec_from_file_location("r", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "llmt-router.py"))
r = importlib.util.module_from_spec(sp); sp.loader.exec_module(r)
import tempfile; r.DRAIN_DIR = tempfile.mkdtemp()
def mk():
    a = r.Backend({"name": "arc", "url": "http://127.0.0.1:1", "cap": 16, "prefill": 2800, "dec": [83, 74, 63, 57], "max_ctx": 196608})
    g = r.Backend({"name": "rtx5090", "url": "http://127.0.0.1:2", "cap": 6, "prefill": 6500, "dec": [245, 203, 139, 137], "prefer_until": 2, "max_ctx": 124800})
    for b in (a, g): b.healthy = True
    return r.Router([a, g]), a, g
R, a, g = mk()
o, why, _ = R.choose([(b"x", 100), (b"y", 30000)], 4000); assert o[0] is g, o[0].name; print("ok   30k -> 5090 (prefer)")
o, why, _ = R.choose([(b"x", 100), (b"y", 150000)], 4000); assert o[0] is a and why == "ctx" and g not in o, (o[0].name, why, [b.name for b in o]); print("ok   150k -> arc only (ctx)")
o, why, _ = R.choose([(b"x", 100), (b"y", 112000)], 2000); assert o[0] is a and g not in o; print("ok   112k x1.1 margin + 2k -> arc only")
import os; open(os.path.join(r.DRAIN_DIR, "drain-arc"), "w").close()
o, why, _ = R.choose([(b"x", 100), (b"y", 150000)], 4000); assert o[0] is g, o[0].name; print("ok   arc drained: 150k falls back to 5090 (engine decides)")
os.remove(os.path.join(r.DRAIN_DIR, "drain-arc")); g.max_ctx = 196608; g.update({"max_ctx": 196608})
o, why, _ = R.choose([(b"x", 100), (b"y", 150000)], 4000); assert why != "ctx"; print("ok   both 192k: no ctx rule")
print("ALL PASS")
# no backend fits the estimate: the largest max_ctx wins
R, a, g = mk()
o, why, _ = R.choose([(b"x", 100), (b"y", 185000)], 40); assert o[0] is a, o[0].name; print("ok   nothing fits (185k x1.1): arc (largest max_ctx)")
# ctx-retry: fake 5090 answers a context-length 400, the request lands on the arc
import asyncio, json
async def fake(port, code, body):
    async def h(rd, wr):
        await rd.readuntil(b"\r\n\r\n"); hd = {}
        wr.write(f"HTTP/1.1 {code}\r\nContent-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n".encode() + body)
        await wr.drain(); wr.close()
    return await asyncio.start_server(h, "127.0.0.1", port)
async def main():
    s1 = await fake(18301, "400 Bad Request", b'{"error":{"message":"This model\'s maximum context length is 124800 tokens. However, you requested 130000 tokens"}}')
    s2 = await fake(18302, "200 OK", b'{"ok":true,"usage":{"completion_tokens":3}}')
    a = r.Backend({"name": "arc", "url": "http://127.0.0.1:18302", "max_ctx": 196608})
    g = r.Backend({"name": "rtx5090", "url": "http://127.0.0.1:18301", "max_ctx": 124800})
    for b in (a, g): b.healthy = True
    S = r.Server.__new__(r.Server); S.r = r.Router([a, g]); S.r.trace_dir = ""
    class W:
        def __init__(self): self.buf = b""
        def write(self, d): self.buf += d
        async def drain(self): pass
    w = W(); body = b'{"model":"m","messages":[]}'
    await S.forward([g, a], "affinity", [(b"x", 10), (b"", 90000)], "POST", "/v1/chat/completions", [("Host", "x")], body, w, 16000, req={})
    assert w.buf.startswith(b"HTTP/1.1 200") and g.n["ctx-retry"] == 1, w.buf[:80]; print("ok   context-length 400 on the 5090 -> retried on the arc (200)")
    w = W(); g.max_ctx = 196608
    await S.forward([g, a], "affinity", [(b"x", 10), (b"", 90000)], "POST", "/v1/chat/completions", [("Host", "x")], body, w, 16000, req={})
    assert w.buf.startswith(b"HTTP/1.1 400") and b"maximum context" in w.buf, w.buf[:80]; print("ok   no larger backend: the 400 goes to the client unchanged")
    s1.close(); s2.close()
asyncio.run(main())
print("ALL PASS (2)")
# images: 6 images -> only the backend allowing them
R, a, g = mk(); a.max_images, g.max_images = 32, 4
o, why, _ = R.choose([(b"x", 100), (b"y", 20000)], 2000, None, 6); assert o[0] is a and why == "ctx" and g not in o, (o[0].name, why); print("ok   6 images -> arc (5090 limit 4)")
o, why, _ = R.choose([(b"x", 100), (b"y", 20000)], 2000, None, 3); assert o[0] is g; print("ok   3 images -> 5090 as usual")
import importlib; print("ALL PASS (3)")
# balance_new: 5090 busy with 2 active conversations -> a NEW conversation goes to the arc; existing ones keep affinity
R, a, g = mk(); R.balance_new = {"rtx5090": 2, "arc": 1}
g.inflight = 2
o, why, _ = R.choose([(b"new1", 100), (b"", 30000)], 2000); assert o[0] is a and why == "balance", (o[0].name, why); print("ok   new conversation -> arc (5090 has 2 active)")
R.record([(b"old1", 100), (b"", 30000)], g)
o, why, _ = R.choose([(b"old1", 100), (b"x", 30500), (b"", 31000)], 2000); assert o[0] is g, (o[0].name, why); print("ok   existing conversation keeps its GPU")
g.inflight = 0
o, why, _ = R.choose([(b"new2", 100), (b"", 30000)], 2000); assert o[0] is g, (o[0].name, why); print("ok   idle 5090: new conversation -> 5090")
print("ALL PASS (4)")
# affinity_min: a 26k shared beginning is not a session; a 60k match of a 62k prompt is
R, a, g = mk(); R.balance_new = {"rtx5090": 2, "arc": 1}; R.affinity_min_tokens = 32768; R.affinity_min_frac = 0.5
R.record([(b"sys", 26000), (b"convA", 60000), (b"", 60000)], g); g.inflight = 2
o, why, _ = R.choose([(b"sys", 26000), (b"convB", 30000), (b"", 30000)], 2000); assert o[0] is a and why == "balance", (o[0].name, why); print("ok   26k shared system prompt only -> placed by load (arc)")
o, why, _ = R.choose([(b"sys", 26000), (b"convA", 60000), (b"t", 62000), (b"", 62000)], 2000); assert o[0] is g, (o[0].name, why); print("ok   60k of 62k matched -> stays on its GPU")
print("ALL PASS (5)")
# affinity_ttl_s: an old placement no longer forces routing
import time as _t
R, a, g = mk(); R.balance_new = {"rtx5090": 2, "arc": 1}; R.affinity_min_tokens = 32768; R.affinity_min_frac = 0.5; R.affinity_ttl_s = 600
R.record([(b"sys", 26000), (b"convA", 60000), (b"", 60000)], g); g.inflight = 2
o, why, _ = R.choose([(b"sys", 26000), (b"convA", 60000), (b"", 62000)], 2000); assert o[0] is g; print("ok   fresh placement keeps affinity")
for h in list(R.map.d): n, tk, ts = R.map.d[h]; R.map.d[h] = (n, tk, ts - 700)
o, why, _ = R.choose([(b"sys", 26000), (b"convA", 60000), (b"", 62000)], 2000); assert o[0] is a and why == "balance", (o[0].name, why); print("ok   placement older than the TTL -> placed by load")
print("ALL PASS (6)")
# compaction-like (no tools, 33k) skips balance -> cost model (5090 faster); small no-tool calls still balance
R, a, g = mk(); R.balance_new = {"rtx5090": 2, "arc": 1}; g.inflight = 2
o, why, _ = R.choose([(b"cmpA", 100), (b"", 33000)], 6000, None, 0, False); assert o[0] is g and why != "balance", (o[0].name, why); print("ok   33k no-tool summary -> 5090 by finish time")
o, why, _ = R.choose([(b"ttlA", 100), (b"", 4000)], 600, None, 0, False); assert why == "balance" and o[0] is a, (o[0].name, why); print("ok   4k title/summary still balanced to the arc")
print("ALL PASS (7)")
# summary_target: with pre-compaction, large no-tool jobs go to the arc first
R, a, g = mk(); R.balance_new = {"rtx5090": 2, "arc": 1}; R.summary_target = "arc"
o, why, _ = R.choose([(b"cmpB", 100), (b"", 33000)], 6000, None, 0, False); assert o[0] is a and why == "summary", (o[0].name, why); print("ok   summary_target arc: 33k summary -> arc")
R.summary_target = ""
o, why, _ = R.choose([(b"cmpC", 100), (b"", 33000)], 6000, None, 0, False); assert o[0] is g, (o[0].name, why); print("ok   summary_target off: -> 5090 by finish time")
print("ALL PASS (8)")
