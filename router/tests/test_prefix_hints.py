#!/usr/bin/env python3
"""Prefix hints (AllBroker HINT / WHERE, docs/prefix-hints.md there): a fake broker on a Unix socket answers WHERE;
fake vLLM backends record what they get. No GPU, no real broker, synthetic prompts only.
Checks: off by default (no broker contact, no anchor injected); broker missing / hung / without the feature = exactly the
old routing; a RAM holder wins a forgotten conversation (reason "hint", no balance_new); the shared_tier guess is
replaced by what the broker reports; drains and caps still win; the anchor is the router's full-prompt hash and a
client's own llmt_anchor is kept; the WHERE frame matches the broker's encoding.
  python3 tests/test_prefix_hints.py"""
import asyncio
import importlib.util
import json
import os
import pathlib
import struct
import tempfile
import time

HERE = pathlib.Path(__file__).parent
spec = importlib.util.spec_from_file_location("router", HERE.parent / "llmt-router.py")
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)

fails = []


def check(cond, what):
    print(("ok   " if cond else "FAIL ") + what)
    if not cond:
        fails.append(what)


class FakeBroker:
    """HELLO / FEATURES / WHERE as the broker encodes them. places: anchor -> [(engine_id, name, n_resident, flags)]."""

    def __init__(self, path, features=1 << 5, hang=False):
        self.path, self.features, self.hang = path, features, hang
        self.places, self.wheres, self.hello = {}, [], None
        self.chunk = 1664

    async def handle(self, rd, wr):
        try:
            while True:
                (n,) = struct.unpack("<I", await rd.readexactly(4))
                f = await rd.readexactly(n)
                op, _, rid = struct.unpack_from("<HHQ", f)
                p = f[12:]
                if op == 1:
                    eng, ver, ln = struct.unpack_from("<HHH", p)
                    self.hello = (eng, p[6:6 + ln].decode())
                    name = b"fake"
                    out = struct.pack("<HH", 0, 1) + struct.pack("<H", len(name)) + name + struct.pack("<Q", 7) \
                        + struct.pack("<H", 2) + b"/p" + struct.pack("<QQI", 1, 1, 1)
                elif op == 18:
                    out = struct.pack("<HQI", 0, self.features, 0)
                elif op == 31:
                    if self.hang:
                        await asyncio.sleep(3600)
                    (k,) = struct.unpack_from("<I", p)
                    anchors = [p[4 + 16 * i:20 + 16 * i] for i in range(k)]
                    self.wheres.append(anchors)
                    rows = []
                    seen = set()
                    for i, a in enumerate(anchors):
                        for eng, name, nres, fl in self.places.get(a, ()):
                            if eng in seen:
                                continue
                            seen.add(eng)
                            nb = name.encode()
                            rows.append(struct.pack("<IHH", i, eng, len(nb)) + nb
                                        + struct.pack("<IQIIIB", 0x80000001, 1234, self.chunk, 40, nres, fl))
                    out = struct.pack("<HI", 0, len(rows)) + b"".join(rows)
                else:
                    out = struct.pack("<H", 6)
                wr.write(struct.pack("<IHHQ", 12 + len(out), op, 1, rid) + out)
                await wr.drain()
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        finally:
            wr.close()


class Fake:
    """vLLM stand-in: records backend + kv_transfer_params."""

    def __init__(self, name):
        self.name, self.ktp = name, []

    async def handle(self, rd, wr):
        head = await R.read_head(rd)
        if head is None:
            wr.close()
            return
        line, hdrs = head
        body = await R.read_body(rd, hdrs)
        path = line.split(" ")[1]
        if path == "/health":
            wr.write(R.simple("200 OK", b""))
        else:
            req = json.loads(body)
            self.ktp.append(req.get("kv_transfer_params"))
            wr.write(R.simple("200 OK", json.dumps({"backend": self.name}).encode()))
        await wr.drain()
        wr.close()


async def http(port, obj):
    rd, wr = await asyncio.open_connection("127.0.0.1", port)
    body = json.dumps(obj).encode()
    wr.write(f"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
             f"Content-Length: {len(body)}\r\n\r\n".encode() + body)
    await wr.drain()
    data = await rd.read()
    wr.close()
    head, _, rest = data.partition(b"\r\n\r\n")
    return int(head.split(b" ")[1]), json.loads(rest)


def conv(tag, n):
    msgs = [{"role": "system", "content": tag * 40000}]           # ~12k estimated tokens per message pair
    for i in range(n):
        msgs.append({"role": "user", "content": f"{tag}{i} " + "q" * 30000})
        msgs.append({"role": "assistant", "content": f"ok {i}"})
    return msgs[:-1]


def backends(pa, pb):
    return [R.Backend({"name": "arc", "url": f"http://127.0.0.1:{pa}", "cap": 16, "prefill": 2800, "dec": [83, 74, 63, 57],
                       "load_gbps": 2.5, "prefill_slowdown": 0.65}),
            R.Backend({"name": "rtx5090", "url": f"http://127.0.0.1:{pb}", "cap": 32, "prefill": 6500,
                       "dec": [383, 317, 264, 212], "load_gbps": 2.0, "prefill_slowdown": 0.65})]


CFG = {"shared_tier": 0.9, "balance_new": {"rtx5090": 2, "arc": 1}, "affinity_min_tokens": 32768,
       "affinity_min_frac": 0.5, "affinity_ttl_s": 600, "split": {"kv_bytes_per_token": 41000}}


def unit():
    """choose() with a WHERE answer given directly."""
    R.DRAIN_DIR = tempfile.mkdtemp()                  # never the live /run/llmt-router drain / hold flags
    bs = backends(1, 2)
    for b in bs:
        b.healthy = True
    r = R.Router(bs)
    r.apply_config(CFG)
    chain = R.prefix_chain("/v1/chat/completions", {"model": "m", "messages": conv("u", 3)}, r.cpt)
    total = chain[-1][1]
    check(r.hints is None and not r.prefix_hints["on"], "prefix hints are off by default")
    o0, reason0, _ = r.choose(chain)
    check(reason0 in ("new", "balance"), f"no hints: a forgotten conversation is placed as new ({reason0} -> {o0[0].name})")
    # the Arc holds 90% of the prompt in RAM; a fresh router would balance it to the 5090
    ct = 1664
    nres = int(total * 0.9) // ct
    places = [{"anchor_idx": 0, "engine_id": 1, "name": "arc", "ns": 1, "age_ms": 1000, "chunk_tokens": ct,
               "n_chunks": nres + 1, "n_resident": nres, "flags": 0}]
    o1, reason1, _ = r.choose(chain, places=places)
    check(o1[0].name == "arc" and reason1 == "hint", f"Arc holds {nres * ct}/{total} tokens in RAM -> arc, reason hint "
                                                     f"(got {o1[0].name}, {reason1})")
    check(r._last_uncached == total - nres * ct, f"arc's prefill clock gets only the uncached part ({r._last_uncached})")
    # broker answered, nobody holds it: no shared_tier credit for anyone, placed as new
    r.balance_new = {}
    o2, reason2, _ = r.choose(chain, places=[])
    check(reason2 == "new" and r._last_hint == {"arc": (0, False), "rtx5090": (0, False)},
          f"empty WHERE answer: no holder, placed as new ({reason2})")
    # a small resident prefix (system prompt only) does not count as a holder
    small = [dict(places[0], n_resident=2)]
    _, reason3, _ = r.choose(chain, places=small)
    check(reason3 != "hint", f"a 3k-token resident prefix is a shared beginning, not a holder ({reason3})")
    # drained holder: never chosen
    tmp = tempfile.mkdtemp()
    R.DRAIN_DIR = tmp
    open(os.path.join(tmp, "drain-arc"), "w").close()
    o4, reason4, _ = r.choose(chain, places=places)
    check(o4[0].name == "rtx5090", f"drained holder never chosen ({o4[0].name}, {reason4})")
    os.remove(os.path.join(tmp, "drain-arc"))
    # holder far over its cap: the cap penalty still applies (goes to the 5090)
    bs[0].inflight = 40
    o5, reason5, _ = r.choose(chain, places=places)
    check(o5[0].name == "rtx5090", f"holder 24 over cap -> rtx5090 ({reason5})")
    bs[0].inflight = 0
    # the 5090 holds it (cross-GPU or its own): with a balanced load the 5090 wins even though shared_tier would have
    # credited both
    p5 = [dict(places[0], engine_id=2, name="rtx5090")]
    o6, reason6, _ = r.choose(chain, places=p5)
    check(o6[0].name == "rtx5090" and reason6 in ("hint", "prefer"), f"5090 holder -> rtx5090 ({reason6})")
    # mapping by engine id
    r.prefix_hints = {**r.prefix_hints, "engines": {"arc": 1, "rtx5090": 2}}
    o7, reason7, _ = r.choose(chain, places=[dict(places[0], name="arc-b70-i4")])
    check(o7[0].name == "arc" and reason7 == "hint", f"engines map by engine id ({o7[0].name}, {reason7})")
    # VRAM flag only within vram_ttl_s
    r.prefix_hints = {**r.prefix_hints, "engines": {}, "vram_ttl_s": 120}
    r.choose(chain, places=[dict(places[0], flags=1, age_ms=5000)])
    check(r._last_hint["arc"][1] is True, "fresh VRAM hint reported")
    r.choose(chain, places=[dict(places[0], flags=1, age_ms=500_000)])
    check(r._last_hint["arc"][1] is False, "old VRAM hint not reported as VRAM")
    # parse_where: the broker's Place encoding (crates/proto Reply::Where)
    nb = b"arc"
    body = struct.pack("<I", 1) + struct.pack("<IHH", 1, 2, 3) + nb + struct.pack("<IQIIIB", 3, 4, 1664, 6, 5, 1)
    check(R.BrokerHints.parse_where(body) == [{"anchor_idx": 1, "engine_id": 2, "name": "arc", "ns": 3, "age_ms": 4,
                                              "chunk_tokens": 1664, "n_chunks": 6, "n_resident": 5, "flags": 1}],
          "WHERE reply decodes (anchor_idx u32, engine u16, name, ns u32, age u64, chunk u32, n u32, resident u32, flags u8)")


async def e2e():
    tmp = tempfile.mkdtemp()
    R.DRAIN_DIR = tmp
    sock = os.path.join(tmp, "broker.sock")
    fa, fb = Fake("arc"), Fake("rtx5090")
    sa = await asyncio.start_server(fa.handle, "127.0.0.1", 0)
    sb = await asyncio.start_server(fb.handle, "127.0.0.1", 0)
    pa, pb = sa.sockets[0].getsockname()[1], sb.sockets[0].getsockname()[1]
    r = R.Router(backends(pa, pb))
    r.apply_config(CFG)
    srv = R.Server(r, health_interval=0.2)
    s = await asyncio.start_server(srv.handle, "127.0.0.1", 0)
    port = s.sockets[0].getsockname()[1]
    hl = asyncio.create_task(srv.health_loop())
    await asyncio.sleep(0.5)

    # off: no anchor, no broker
    st, got = await http(port, {"model": "m", "messages": conv("a", 1)})
    check(st == 200 and not any("llmt_anchor" in (k or {}) for k in fa.ktp + fb.ktp), "off: no llmt_anchor forwarded")

    # on, broker missing: same routing as off, request served
    r.apply_config({"prefix_hints": {"on": True, "socket": sock, "timeout_ms": 5}})
    r.map.d.clear()
    msgs = conv("b", 2)
    t0 = time.monotonic()
    st, got = await http(port, {"model": "m", "messages": msgs})
    dt = time.monotonic() - t0
    check(st == 200 and r.hints.n["error"] >= 1, f"broker missing: served ({got['backend']}), WHERE error counted "
                                                 f"({dict(r.hints.n)}, {dt * 1000:.0f} ms)")
    last = (fa.ktp + fb.ktp)[-1] or {}
    chain = R.prefix_chain("/v1/chat/completions", {"model": "m", "messages": msgs}, r.cpt)
    check(last.get("llmt_anchor") == chain[-2][0].hex(), "anchor injected = the router's hash of the full prompt")

    # broker up: the Arc reports a conversation the router forgot
    fbk = FakeBroker(sock)
    bs = await asyncio.start_unix_server(fbk.handle, sock)
    r.hints.next_try = 0.0
    msgs3 = conv("c", 4)
    chain3 = R.prefix_chain("/v1/chat/completions", {"model": "m", "messages": msgs3}, r.cpt)
    prev = chain3[-3][0]            # the previous turn's full-prompt anchor (= its llmt_anchor)
    fbk.places[prev] = [(1, "arc", int(chain3[-3][1] * 0.95) // 1664, 0)]
    # (idle 5090 vs idle Arc, the 5090 still wins on finish time: its decode is 4.6x faster; with 4 agents decoding
    # there the Arc's RAM copy pays)
    r.by_name["rtx5090"].inflight += 4
    r.prefix_hints["engines"] = {}
    _, why_off, _ = r.choose(chain3[:-1] + [chain3[-1]])        # same load, no hint information
    st, got = await http(port, {"model": "m", "messages": msgs3})
    r.by_name["rtx5090"].inflight -= 4
    check(why_off != "hint", f"same load without hints: not a hint placement ({why_off})")
    check(got["backend"] == "arc" and r.by_name["arc"].n.get("hint", 0) == 1,
          f"broker says the Arc holds the previous turn in RAM -> arc, reason hint (got {got['backend']}, {dict(r.by_name['arc'].n)}, hint {r._last_hint}, {dict(r.hints.n)})")
    check(fbk.hello == (0xBB01, "llmt-router"), f"router HELLOs as an observer ({fbk.hello})")
    check(fbk.wheres and fbk.wheres[-1][0] == chain3[-2][0] and prev in fbk.wheres[-1],
          "WHERE asks the newest boundary first and includes the previous turn's anchor")

    # a client's own llmt_anchor wins
    st, got = await http(port, {"model": "m", "messages": conv("d", 1), "kv_transfer_params": {"llmt_anchor": "ab" * 16}})
    check(((fa.ktp + fb.ktp)[-1] or {}).get("llmt_anchor") == "ab" * 16, "client's own llmt_anchor kept")

    # broker hangs: timeout, request still served promptly
    r.hints.close()
    bs.close()
    if os.path.exists(sock):
        os.remove(sock)
    hang = FakeBroker(sock, hang=True)
    bs2 = await asyncio.start_unix_server(hang.handle, sock)
    r.hints.next_try = 0.0
    t0 = time.monotonic()
    for i in range(3):
        st, got = await http(port, {"model": "m", "messages": conv(f"e{i}", 1)})
    dt = (time.monotonic() - t0) / 3
    check(st == 200 and r.hints.n["timeout"] >= 2, f"hung broker: requests served, WHERE timed out ({dict(r.hints.n)}, "
                                                   f"{dt * 1000:.0f} ms per request)")
    bs2.close()

    # broker without the feature: hints off for a minute, no error loop
    r.hints.close()
    if os.path.exists(sock):
        os.remove(sock)
    old = FakeBroker(sock, features=1)
    bs3 = await asyncio.start_unix_server(old.handle, sock)
    r.hints.next_try = 0.0
    st, got = await http(port, {"model": "m", "messages": conv("f", 1)})
    st, got = await http(port, {"model": "m", "messages": conv("g", 1)})
    check(st == 200 and r.hints.n["off"] == 1 and r.hints.n["down"] >= 1, f"old broker: off once, then not asked ({dict(r.hints.n)})")
    bs3.close()

    m = srv.metrics().decode()
    check('llmt_router_hint_where_total{result="ok"}' in m, "metrics: llmt_router_hint_where_total")
    # off again via config: client dropped
    r.apply_config({"prefix_hints": {"on": False}})
    check(r.hints is None, "turning prefix_hints off drops the broker client")
    hl.cancel()


unit()
asyncio.run(e2e())
print("ALL PASS" if not fails else f"FAILED: {fails}")
raise SystemExit(1 if fails else 0)
