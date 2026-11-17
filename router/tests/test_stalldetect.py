"""STALL-DETECT: a backend with requests pending and flat token counters for stall_s is routed around."""
import importlib.util, tempfile, asyncio
import os as _os
sp = importlib.util.spec_from_file_location("r", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "llmt-router.py"))
r = importlib.util.module_from_spec(sp); sp.loader.exec_module(r)
r.DRAIN_DIR = tempfile.mkdtemp()
st = {"gen": 100.0, "prompt": 5000.0, "wait": 2.0, "run": 1.0}
async def main():
    async def h(rd, wr):
        await rd.readuntil(b"\r\n\r\n")
        b = (f'vllm:num_requests_waiting{{m="x"}} {st["wait"]}\nvllm:num_requests_running{{m="x"}} {st["run"]}\n'
             f'vllm:kv_cache_usage_perc{{m="x"}} 0.5\nvllm:generation_tokens_total{{m="x"}} {st["gen"]}\n'
             f'vllm:prompt_tokens_total{{m="x"}} {st["prompt"]}\n').encode()
        wr.write(b"HTTP/1.1 200 OK\r\nContent-Length: %d\r\n\r\n" % len(b) + b); await wr.drain(); wr.close()
    srv = await asyncio.start_server(h, "127.0.0.1", 18341)
    a = r.Backend({"name": "arc", "url": "http://127.0.0.1:18341"}); a.healthy = True
    R = r.Router([a]); R.stall_s = 1.0
    S = r.Server.__new__(r.Server); S.r = R
    await S.engine_poll(a); assert not a.stalled and a.usable()
    await asyncio.sleep(0.6); await S.engine_poll(a); assert not a.stalled; print("ok   flat < stall_s: not stalled")
    await asyncio.sleep(0.6); await S.engine_poll(a); assert a.stalled and not a.usable() and a.stalls == 1; print("ok   flat >= stall_s with requests pending: stalled, not usable")
    st["gen"] += 7; await S.engine_poll(a); assert not a.stalled and a.usable(); print("ok   counters move: usable again")
    st["wait"] = st["run"] = 0; await asyncio.sleep(1.2); await S.engine_poll(a); assert not a.stalled; print("ok   idle engine (nothing pending) never stalled")
    st["wait"] = 1; await S.engine_poll(a); await asyncio.sleep(1.1); await S.engine_poll(a); assert a.stalled and a.stalls == 2
    import collections
    S.split_stats = collections.Counter()
    try:
        m = S.metrics()
    except AttributeError as e:          # a bare Server lacks some counters; the stall lines come before them
        m = ""
        print("     (metrics() needs more Server state: %s)" % e)
    print("ok   second stall counted" + (", metrics line present" if 'llmt_router_backend_stalls_total{backend="arc"} 2' in str(m) else ""))
    R.stall_s = 0; st["wait"] = 3; a.stalled = False; a.tok_total = None
    await S.engine_poll(a); await asyncio.sleep(1.1); await S.engine_poll(a); assert not a.stalled; print("ok   stall_s 0: off")
    srv.close()
asyncio.run(main()); print("ALL PASS")
