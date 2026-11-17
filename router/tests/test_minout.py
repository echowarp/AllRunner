"""min_out_tokens raises a small client output cap (OpenCode 16384), clamped to max_ctx (local fake backend)."""
import importlib.util, tempfile, asyncio, json
import os as _os
sp = importlib.util.spec_from_file_location("r", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "llmt-router.py"))
r = importlib.util.module_from_spec(sp); sp.loader.exec_module(r)
r.DRAIN_DIR = tempfile.mkdtemp()
got = []
async def main():
    async def h(rd, wr):
        hd = await rd.readuntil(b"\r\n\r\n"); n = int([l.split(b":")[1] for l in hd.split(b"\r\n") if l.lower().startswith(b"content-length")][0])
        got.append(json.loads(await rd.readexactly(n)))
        b = b'{"ok":true,"usage":{"prompt_tokens":5,"completion_tokens":3}}'
        wr.write(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n" % len(b) + b); await wr.drain(); wr.close()
    srv = await asyncio.start_server(h, "127.0.0.1", 18322)
    port = 18323
    a = r.Backend({"name": "arc", "url": "http://127.0.0.1:18322", "max_ctx": 196608}); a.healthy = True
    R = r.Router([a]); R.min_out_tokens = 32768.0; R.trace_dir = tempfile.mkdtemp()
    S = r.Server.__new__(r.Server); S.r = R
    srv2 = await asyncio.start_server(S.handle, "127.0.0.1", port)
    async def send(body):
        raw = json.dumps(body).encode(); rd, wr = await asyncio.open_connection("127.0.0.1", port)
        wr.write(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: %d\r\nConnection: close\r\n\r\n" % len(raw) + raw)
        await wr.drain(); await rd.read(); wr.close()
    msg = [{"role": "user", "content": "hi " * 100}]
    await send({"model": "m", "messages": msg, "max_completion_tokens": 16384}); assert got[-1]["max_completion_tokens"] == 32768, got[-1]; print("ok   16384 -> 32768")
    await send({"model": "m", "messages": msg, "max_tokens": 40000}); assert got[-1]["max_tokens"] == 40000; print("ok   larger cap kept")
    await send({"model": "m", "messages": msg}); assert "max_tokens" not in got[-1] and "max_completion_tokens" not in got[-1]; print("ok   no cap stays uncapped")
    big = [{"role": "user", "content": "x" * int(3.3 * 150000)}]
    await send({"model": "m", "messages": big, "max_completion_tokens": 16384})
    assert "max_completion_tokens" not in got[-1], got[-1].get("max_completion_tokens"); print("ok   near max_ctx: cap dropped (engine computes the exact room)")
    R.min_out_tokens = 196608.0
    await send({"model": "m", "messages": msg, "max_tokens": 16384}); assert "max_tokens" not in got[-1], got[-1]; print("ok   196608: cap dropped (context-limited)")
    await send({"model": "m", "messages": big, "max_completion_tokens": 16384}); assert "max_completion_tokens" not in got[-1]; print("ok   196608 big prompt: cap dropped")
    R.min_out_tokens = 0.0
    await send({"model": "m", "messages": msg, "max_completion_tokens": 16384}); assert got[-1]["max_completion_tokens"] == 16384; print("ok   off = unchanged")
    srv.close(); srv2.close()
asyncio.run(main()); print("ALL PASS")
