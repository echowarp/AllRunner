"""Session tags -> X-Request-Id (session-hold tag format), via a local fake backend that echoes headers."""
import importlib.util, tempfile, asyncio, json, re
import os as _os
sp = importlib.util.spec_from_file_location("r", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "llmt-router.py"))
r = importlib.util.module_from_spec(sp); sp.loader.exec_module(r)
r.DRAIN_DIR = tempfile.mkdtemp()
TAG = re.compile(r"^arcs([0-9a-f]{12})([0-9a-f]{12})([a-z]?)-[0-9a-f]{16}$")
seen = []
async def main():
    async def h(rd, wr):
        hd = (await rd.readuntil(b"\r\n\r\n")).decode()
        n = int(re.search(r"(?i)content-length: (\d+)", hd).group(1)); await rd.readexactly(n)
        seen.append({l.split(":", 1)[0].lower(): l.split(":", 1)[1].strip() for l in hd.split("\r\n")[1:] if ":" in l})
        b = b'{"ok":true,"usage":{"prompt_tokens":5,"completion_tokens":3}}'
        wr.write(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n" % len(b) + b); await wr.drain(); wr.close()
    srv = await asyncio.start_server(h, "127.0.0.1", 18331)
    a = r.Backend({"name": "arc", "url": "http://127.0.0.1:18331"}); a.healthy = True
    R = r.Router([a]); R.trace_dir = tempfile.mkdtemp()
    S = r.Server.__new__(r.Server); S.r = R
    srv2 = await asyncio.start_server(S.handle, "127.0.0.1", 18332)
    async def send(extra):
        raw = json.dumps({"model": "m", "messages": [{"role": "user", "content": "hi"}]}).encode()
        hh = "".join(f"{k}: {v}\r\n" for k, v in extra.items())
        rd, wr = await asyncio.open_connection("127.0.0.1", 18332)
        wr.write(f"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n{hh}Content-Length: {len(raw)}\r\nConnection: close\r\n\r\n".encode() + raw)
        await wr.drain(); await rd.read(); wr.close()
        return seen[-1]
    R.session_tag_ids = False
    g = await send({"x-opencode-session-id": "ses_A"}); assert "x-request-id" not in g; print("ok   off: no tag")
    R.session_tag_ids = True
    g1 = await send({"x-opencode-session-id": "ses_A"}); m1 = TAG.match(g1["x-request-id"]); assert m1 and m1.group(2) == "0" * 12 and m1.group(3) == ""
    g2 = await send({"x-opencode-session-id": "ses_A", "x-opencode-agent": "compaction"}); m2 = TAG.match(g2["x-request-id"])
    assert m2.group(1) == m1.group(1) and m2.group(3) == "c" and g1["x-request-id"] != g2["x-request-id"]; print("ok   same session -> same 12-hex, compaction -> c, ids unique")
    g3 = await send({"x-session-id": "chat-7", "x-parent-session-id": "chat-1", "x-agent": "worker"}); m3 = TAG.match(g3["x-request-id"])
    assert m3 and m3.group(1) != m1.group(1) and m3.group(2) != "0" * 12 and m3.group(3) == "m"; print("ok   chat agents: session + parent + agent")
    g4 = await send({"x-request-id": "client-id", "x-opencode-session-id": "ses_A"}); assert g4["x-request-id"].startswith("arcs"); print("ok   client X-Request-Id replaced when tagged")
    g5 = await send({"x-request-id": "client-id"}); assert g5["x-request-id"] == "client-id"; print("ok   untagged requests keep theirs")
    assert "ses_A" not in json.dumps(seen[-1]) and all("ses_A" not in v for v in g1.values() if v != "ses_A") ; print("ok   raw id not in the tag")
    srv.close(); srv2.close()
asyncio.run(main()); print("ALL PASS")
