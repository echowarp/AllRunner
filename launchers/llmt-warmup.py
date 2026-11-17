#!/usr/bin/env python3
"""llmt-warmup.py: post-start warm-up for a Qwen3.8-27B vLLM engine (Arc B70 or RTX 5090), optionally gated in the router.

Right after an engine start, the first request of some shapes pays one-time per-process costs (oneDNN primitives,
Triton kernel loads, xgrammar compiler init, sampler kernels). With persistent compile caches these were measured at
about 4 s in total on the Arc and 2.5 s on the RTX 5090 (launchers/README.md, "Warm-up"). This tool sends synthetic
requests over the shapes that have such a cost:
  1. prefill  : single prompts over the chunk-size range (one request each, max_tokens 1)
  2. decode   : N concurrent streams for each N in the profile (one N per draft-depth band and its edges)
  3. mixed    : D decode streams running while prompts of the given sizes are prefilled (mixed prefill+decode steps)
  4. extras   : one json_object request (xgrammar init) and one with tools (tool parser)
All prompts are synthetic words with a unique cache_salt and llmt_nostore (nothing enters the prefix cache or the RAM
tier). Logs counts and timings only. Runs at idle CPU priority.

Gate (--gate NAME): drain this GPU's router backend (touch <drain-dir>/drain-NAME with an llmt-warmup marker) for the
duration of the warm-up, ONLY when another backend is healthy, not drained and not held per the router's /router/state;
otherwise the warm-up runs while serving (as before). The drain is removed when the warm-up ends, on any signal, after
--hard-timeout seconds from engine health, when the other backend stops being usable, and by --cleanup (the launcher
runs it after this process exits, however it exits) and by the next start. A drain file this tool did not create, or
one touched by someone else after it was created (mtime changed), is never removed.

  llmt-warmup.py --profile arc|rtx5090 [--url http://127.0.0.1:8000] [--gate arc] [--router http://127.0.0.1:8010]
  llmt-warmup.py --cleanup NAME          # remove a stale llmt-warmup drain of NAME (pid dead, file untouched)
  llmt-warmup.py --profile arc --plan    # print the request plan and exit
Stdlib only."""
from __future__ import annotations

import argparse
import http.client
import json
import os
import random
import signal
import sys
import threading
import time
import urllib.parse
import uuid

MARK = "llmt-warmup"
W = ["valve", "pump", "sensor", "alarm", "axis", "drive", "motor", "relay", "fault", "cycle", "torque", "encoder",
     "spindle", "gear", "bearing", "coolant", "filter", "nozzle", "boiler", "turbine"]

# Profiles: production shapes per GPU (launcher flags 2026-11-12). decode_n covers every draft-depth band (the depth
# table is keyed on decoding requests) at both edges; the lists are the measured first-use shapes (launchers/README.md,
# "Warm-up"); shapes that measured free are left out.
PROFILES = {
    "arc": {
        # Arc: max_num_seqs 16 (router cap), depth [[1,4,7],[5,5,5],[6,6,4],[7,8,3],[9,256,2]], max_num_batched_tokens 8192.
        # Measured 2026-11-12 (fresh process, persistent caches): first-use cost only in the first
        # request (+1.4 s), small prompts (100 tok +0.25 s, 2k +0.16 s), 1-3 concurrent decode streams (TTFT +0.26-0.33 s,
        # one 178 ms step at 2) and the first json_object request (+1.4 s, xgrammar). 4-16 streams, prompts >= 3k tokens
        # and mixed prefill+decode steps measured free.
        "decode_n": [1, 2, 3],
        "decode_tokens": 40,
        "prefill_tokens": [20, 50, 100, 200, 530, 1300, 2000],
        "mixed_decoders": [],
        "mixed_tokens": [],
        "extras": True,
    },
    "rtx5090": {
        # 5090: --max-num-seqs 32, depth [[1,2,7],[3,16,3],[17,32,1]], cudagraphs <= 1024 tokens, 8192-token chunks.
        # Measured 2026-11-12 (lab, fresh process, production compile cache): first-use cost only in the first request
        # (+0.88 s), 1-2 concurrent decode streams (TTFT +0.42-0.45 s, one 198 ms step at 2) and the first json_object
        # request (+0.92 s, xgrammar). 3-32 streams (all three depth bands), prompts 50-16k tokens and mixed
        # prefill+decode steps (1/8/16 decoders) measured free; x3 kept as the band edge (0.1 s).
        "decode_n": [1, 2, 3],
        "decode_tokens": 40,
        "prefill_tokens": [20, 100, 530],
        "mixed_decoders": [],
        "mixed_tokens": [],
        "extras": True,
    },
}
TPW = [3.4]          # running tokens-per-word estimate of the synthetic text (updated from usage.prompt_tokens)
OVERHEAD = 14        # chat-template tokens around a user message (thinking off)


def log(msg: str):
    print(f"{time.strftime('%F %T')} llmt-warmup: {msg}", flush=True)


def set_idle():
    try:
        os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    except (AttributeError, OSError):
        pass
    try:
        os.nice(19)
    except OSError:
        pass


# ---- HTTP ---------------------------------------------------------------------------------------------------------
def _conn(url: str, timeout: float):
    u = urllib.parse.urlsplit(url)
    return http.client.HTTPConnection(u.hostname, u.port or 80, timeout=timeout)


def get_json(url: str, path: str, timeout: float = 3.0):
    c = _conn(url, timeout)
    try:
        c.request("GET", path)
        r = c.getresponse()
        data = r.read()
        return r.status, (json.loads(data) if data[:1] in (b"{", b"[") else None)
    finally:
        c.close()


def healthy(url: str) -> bool:
    try:
        return get_json(url, "/health")[0] == 200
    except Exception:
        return False


def words(n: int) -> str:
    return " ".join(f"{random.choice(W)}{random.randint(0, 99999)}" for _ in range(max(1, n)))


def body(prompt_tokens: int, max_tokens: int, sampled: bool, stream: bool = True, **extra) -> dict:
    n_words = max(1, round((prompt_tokens - OVERHEAD) / TPW[0]))
    b = {"model": "qwen3.8-27b", "max_tokens": max_tokens, "ignore_eos": True,
         "chat_template_kwargs": {"enable_thinking": False}, "cache_salt": uuid.uuid4().hex,
         "kv_transfer_params": {"llmt_nostore": True},
         "messages": [{"role": "user", "content": words(n_words) + "\nDescribe these parts at length."}]}
    b["_words"] = n_words
    if sampled:
        b.update(temperature=0.7, top_p=0.95, top_k=20)
    else:
        b.update(temperature=0.0)
    if stream:
        b.update(stream=True, stream_options={"include_usage": True})
    b.update(extra)
    return b


class Stop(Exception):
    pass


def request(url: str, b: dict, timeout: float = 600, stop: threading.Event | None = None) -> dict:
    """POST /v1/chat/completions (streamed when b['stream']). Returns ttft, chunk times, prompt/completion tokens."""
    nw = b.pop("_words", None)
    t0 = time.monotonic()
    out = {"ttft": None, "gaps": [], "pt": None, "ct": None, "ok": False, "t": 0.0}
    c = _conn(url, timeout)
    try:
        c.request("POST", "/v1/chat/completions", json.dumps(b), {"Content-Type": "application/json"})
        r = c.getresponse()
        if r.status != 200:
            r.read()
            out["err"] = r.status
            return out
        if not b.get("stream"):
            d = json.loads(r.read())
            out["ttft"] = time.monotonic() - t0
            u = d.get("usage") or {}
            out.update(pt=u.get("prompt_tokens"), ct=u.get("completion_tokens"), ok=True)
        else:
            last = None
            for line in r:
                if stop is not None and stop.is_set():
                    raise Stop()
                if not line.startswith(b"data: "):
                    continue
                if line.startswith(b"data: [DONE]"):
                    break
                d = json.loads(line[6:])
                now = time.monotonic()
                if d.get("choices"):
                    if out["ttft"] is None:
                        out["ttft"] = now - t0
                    elif last is not None:
                        out["gaps"].append(now - last)
                    last = now
                if d.get("usage"):
                    out["pt"] = d["usage"].get("prompt_tokens")
                    out["ct"] = d["usage"].get("completion_tokens")
            out["ok"] = True
    except Stop:
        out["ok"] = "stopped"
    except Exception as e:  # noqa: BLE001  (connection errors: counted, not fatal)
        out["err"] = type(e).__name__
    finally:
        c.close()
        out["t"] = time.monotonic() - t0
    if out["pt"] and nw and out["pt"] > OVERHEAD + 20:
        TPW[0] = 0.7 * TPW[0] + 0.3 * (out["pt"] - OVERHEAD) / nw
    return out


def concurrent(url: str, bodies: list, timeout: float = 600, stop=None) -> list:
    res = [None] * len(bodies)
    gate = threading.Barrier(len(bodies))

    def run(i):
        try:
            gate.wait(10)
        except threading.BrokenBarrierError:
            pass
        res[i] = request(url, bodies[i], timeout, stop)
    th = [threading.Thread(target=run, args=(i,), daemon=True) for i in range(len(bodies))]
    for t in th:
        t.start()
    for t in th:
        t.join(timeout + 5)
    return [r or {"ok": False, "err": "timeout"} for r in res]


def summary(rs: list) -> dict:
    ok = [r for r in rs if r.get("ok")]
    tt = sorted(r["ttft"] for r in ok if r.get("ttft") is not None)
    gaps = sorted(g for r in ok for g in r["gaps"])
    return {"n": len(rs), "ok": len(ok), "ttft_max": round(tt[-1], 3) if tt else None,
            "ttft_med": round(tt[len(tt) // 2], 3) if tt else None,
            "gap_med_ms": round(gaps[len(gaps) // 2] * 1e3, 1) if gaps else None,
            "gap_max_ms": round(gaps[-1] * 1e3, 1) if gaps else None,
            "wall": round(max((r.get("t", 0) for r in rs), default=0), 2),
            "pt": [r.get("pt") for r in rs][:3]}


# ---- phases ---------------------------------------------------------------------------------------------------------
def phase_prefill(url, sizes, sampled=True):
    out = []
    for i, n in enumerate(sizes):
        r = request(url, body(n, 1, sampled=sampled and i % 2 == 0))
        out.append((n, r))
    return out


def phase_decode(url, n, tokens, short=48):
    # alternate greedy / sampled rows: both sampler paths run in the same batch, as in production
    return concurrent(url, [body(short + (i % 4) * 8, tokens, sampled=i % 2 == 0) for i in range(n)])


def phase_mixed(url, decoders, sizes):
    """D streams decode (long max_tokens, aborted at the end) while the prompts are prefilled one after another."""
    stop = threading.Event()
    started = threading.Event()
    res = []

    def dec(i):
        res.append(request(url, body(48, 4096, sampled=i % 2 == 0), 900, stop))
    th = [threading.Thread(target=dec, args=(i,), daemon=True) for i in range(decoders)]
    for t in th:
        t.start()
    time.sleep(1.5 + 0.1 * decoders)       # decoders admitted and decoding
    pre = [(n, request(url, body(n, 1, sampled=True))) for n in sizes]
    stop.set()
    for t in th:
        t.join(30)
    return pre, res


def phase_extras(url):
    def nw(b):
        b.pop("_words", None)
        return b
    j = request(url, nw(body(60, 24, sampled=True, response_format={"type": "json_object"},
                          messages=[{"role": "user", "content": words(30) + "\nReturn a JSON object with a key 'parts'."}])))
    tools = [{"type": "function", "function": {"name": "lookup", "description": "Look up a part number.",
                                                "parameters": {"type": "object", "properties": {"part": {"type": "string"}},
                                                               "required": ["part"]}}}]
    t = request(url, nw(body(60, 24, sampled=True, tools=tools, tool_choice="auto",
                          messages=[{"role": "user", "content": words(20) + "\nLook up the first part."}])))
    return j, t


def lora_models(url) -> list:
    """LoRA modules the engine serves: /v1/models entries whose "parent" is another model (vLLM --lora-modules)."""
    try:
        d = get_json(url, "/v1/models", 5)[1] or {}
        return [m["id"] for m in d.get("data", []) if m.get("parent") and m.get("parent") != m.get("id")]
    except Exception:
        return []


def phase_lora(url, names):
    out = []
    for n in names:
        b = body(60, 16, sampled=True)
        b["model"] = n
        out.append((n, request(url, b)))
    return out


# ---- gate -----------------------------------------------------------------------------------------------------------
class Gate:
    def __init__(self, name: str, router: str, drain_dir: str, state_dir: str):
        self.name, self.router = name, router
        self.drain = os.path.join(drain_dir, "drain-" + name)
        self.holds = [os.path.join(drain_dir, "hold-" + name), os.path.join(drain_dir, "hold-" + name.replace("rtx", ""))]
        self.state_dir = state_dir
        self.state_f = os.path.join(state_dir, name + ".json")
        self.mine_mtime = None
        self.lock = threading.Lock()
        self.state = {"name": name, "pid": os.getpid(), "drained": False, "t_start": time.time()}

    # state file: pid, drain mtime_ns, phase (read by --cleanup / the next start / dashboards)
    def save(self, **kw):
        self.state.update(kw)
        try:
            os.makedirs(self.state_dir, exist_ok=True)
            tmp = self.state_f + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self.state, f)
            os.replace(tmp, self.state_f)
        except OSError as e:
            log(f"state file: {e}")

    def router_state(self):
        try:
            st, d = get_json(self.router, "/router/state", 3)
            return d if st == 200 else None
        except Exception:
            return None

    def other_usable(self):
        d = self.router_state()
        if not d:
            return False, "router unreachable"
        others = [b for b in d.get("backends", []) if b.get("name") != self.name]
        if not any(b.get("name") == self.name for b in d.get("backends", [])):
            return False, f"router has no backend {self.name}"
        ok = [b["name"] for b in others if b.get("healthy") and not b.get("drained") and not b.get("held")]
        return (bool(ok), ",".join(ok) if ok else "no other backend healthy+undrained")

    def take_over(self):
        """A previous gate of this name (an engine that restarted mid-warm-up): stop it (its handler undrains), then
        remove its drain if it is still there and untouched."""
        try:
            old = json.load(open(self.state_f))
        except (OSError, ValueError):
            old = None
        if old and old.get("pid") and old["pid"] != os.getpid() and is_gate_proc(old["pid"], self.name):
            log(f"previous warm-up pid {old['pid']} still running: stopping it")
            try:
                os.kill(old["pid"], signal.SIGTERM)
            except OSError:
                pass
            for _ in range(100):
                if not is_gate_proc(old["pid"], self.name):
                    break
                time.sleep(0.1)
            else:
                try:
                    os.kill(old["pid"], signal.SIGKILL)
                except OSError:
                    pass
        cleanup(self.name, os.path.dirname(self.drain), self.state_dir, quiet=True)

    def engage(self):
        if any(os.path.exists(h) for h in self.holds):
            log(f"hold-{self.name} present: not draining")
            return
        if os.path.exists(self.drain):
            log(f"drain-{self.name} already set by someone else: left alone, warm-up runs inside that drain")
            return
        ok, why = self.other_usable()
        if not ok:
            log(f"not draining ({why}): warm-up runs while serving")
            return
        with self.lock:
            try:
                fd = os.open(self.drain, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            except FileExistsError:
                log(f"drain-{self.name} appeared concurrently: left alone")
                return
            with os.fdopen(fd, "w") as f:
                f.write(f"{MARK} pid={os.getpid()} t={int(time.time())}\n")
            self.mine_mtime = os.stat(self.drain).st_mtime_ns
            self.save(drained=True, drain_mtime_ns=self.mine_mtime)
        log(f"drained {self.name} in the router (other usable: {why})")

    def release(self, why: str):
        with self.lock:
            if self.mine_mtime is None:
                return
            m = self.mine_mtime
            self.mine_mtime = None
            try:
                st = os.stat(self.drain)
            except FileNotFoundError:
                log(f"drain-{self.name} already gone ({why})")
                self.save(drained=False)
                return
            if st.st_mtime_ns != m or not read_mark(self.drain):
                log(f"drain-{self.name} was touched by someone else since: left in place ({why})")
                self.save(drained=False)
                return
            try:
                os.unlink(self.drain)
                log(f"undrained {self.name} ({why})")
            except OSError as e:
                log(f"undrain {self.name} failed: {e}")
            self.save(drained=False)


def read_mark(path: str) -> str | None:
    try:
        with open(path) as f:
            s = f.read(200)
        return s if s.startswith(MARK) else None
    except OSError:
        return None


def is_gate_proc(pid: int, name: str) -> bool:
    try:
        a = open(f"/proc/{pid}/cmdline", "rb").read().split(b"\0")
    except OSError:
        return False
    return any(x.endswith(b"llmt-warmup.py") for x in a) and b"--gate" in a and name.encode() in a


def cleanup(name: str, drain_dir: str, state_dir: str, quiet: bool = False) -> int:
    """Remove drain-NAME only if it carries our marker, the pid that made it is gone, and its mtime still equals the
    one recorded at creation (nobody touched it since)."""
    drain = os.path.join(drain_dir, "drain-" + name)
    mark = read_mark(drain)
    if not mark:
        if not quiet:
            log(f"cleanup {name}: no llmt-warmup drain")
        return 0
    try:
        pid = int(mark.split("pid=")[1].split()[0])
    except (IndexError, ValueError):
        pid = 0
    if pid and pid != os.getpid() and is_gate_proc(pid, name):
        log(f"cleanup {name}: drain owner pid {pid} still running, left alone")
        return 1
    try:
        rec = json.load(open(os.path.join(state_dir, name + ".json")))
    except (OSError, ValueError):
        rec = {}
    try:
        mt = os.stat(drain).st_mtime_ns
    except FileNotFoundError:
        return 0
    if rec.get("pid") == pid and rec.get("drain_mtime_ns") not in (None, mt):
        log(f"cleanup {name}: drain was touched after the warm-up created it (another operator): left alone")
        return 1
    os.unlink(drain)
    log(f"cleanup {name}: removed stale llmt-warmup drain (pid {pid} gone)")
    return 0


# ---- main -----------------------------------------------------------------------------------------------------------
def plan(p: dict) -> list:
    steps = [("prefill", p["prefill_tokens"])]
    steps += [("decode", n) for n in p["decode_n"]]
    steps += [("mixed", d) for d in p["mixed_decoders"]]
    if p.get("extras"):
        steps.append(("extras", None))
    if p.get("lora"):
        steps.append(("lora", p["lora"]))
    return steps


def lst(s: str):
    return [int(x) for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--profile", choices=sorted(PROFILES))
    ap.add_argument("--url", default=None, help="engine URL (default: arc :8000, rtx5090 :8001)")
    ap.add_argument("--health-timeout", type=int, default=1500, help="seconds to wait for the engine's /health")
    ap.add_argument("--budget", type=int, default=120, help="soft budget (s) after health: later steps are skipped")
    ap.add_argument("--gate", metavar="NAME", help="router backend name to drain during the warm-up (arc, rtx5090)")
    ap.add_argument("--router", default="http://127.0.0.1:8010")
    ap.add_argument("--drain-dir", default=os.environ.get("LLMT_ROUTER_DRAIN_DIR", "/run/llmt-router"))
    ap.add_argument("--state-dir", default="/run/allrunner-warmup")
    ap.add_argument("--hard-timeout", type=int, default=300, help="drain lifted at most this long after engine health")
    ap.add_argument("--cleanup", metavar="NAME", help="remove a stale llmt-warmup drain of NAME and exit")
    ap.add_argument("--decode-n"), ap.add_argument("--prefill"), ap.add_argument("--mixed-tokens")
    ap.add_argument("--mixed-decoders")
    ap.add_argument("--max-seqs", type=int, help="engine max_num_seqs: drop decode/mixed stream counts above it")
    ap.add_argument("--no-extras", action="store_true")
    ap.add_argument("--lora", default="off", help="off (default) | auto (every LoRA module in /v1/models) | NAME[,NAME]: one "
                    "short request per adapter (the first one pays the adapter load)")
    ap.add_argument("--plan", action="store_true", help="print the steps and exit")
    a = ap.parse_args()
    if a.cleanup:
        return cleanup(a.cleanup, a.drain_dir, a.state_dir)
    if not a.profile:
        ap.error("--profile is required")
    p = dict(PROFILES[a.profile])
    for k, v in (("decode_n", a.decode_n), ("prefill_tokens", a.prefill), ("mixed_tokens", a.mixed_tokens),
                 ("mixed_decoders", a.mixed_decoders)):
        if v is not None:
            p[k] = lst(v)
    if a.no_extras:
        p["extras"] = False
    if a.lora != "off":
        p["lora"] = a.lora
    if a.max_seqs:
        p["decode_n"] = [n for n in p["decode_n"] if n <= a.max_seqs]
        p["mixed_decoders"] = [d for d in p["mixed_decoders"] if d < a.max_seqs]
    url = a.url or {"arc": "http://127.0.0.1:8000", "rtx5090": "http://127.0.0.1:8001"}[a.profile]
    if a.plan:
        print(json.dumps({"url": url, "profile": a.profile, **p, "steps": plan(p)}, indent=1))
        return 0
    set_idle()

    gate = Gate(a.gate, a.router, a.drain_dir, a.state_dir) if a.gate else None
    done = threading.Event()

    def on_signal(sig, _f):
        raise SystemExit(128 + sig)
    for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(s, on_signal)
    t0 = time.monotonic()
    rc = 0
    try:
        th = [None]                     # monotonic time of engine health
        if gate:
            gate.take_over()
            gate.save(phase="start", drained=False, drain_mtime_ns=None)
            gate.engage()

            def guard():   # hard timeout + other-backend check, independent of the request threads
                while not done.wait(3):
                    if th[0] is not None and time.monotonic() - th[0] > a.hard_timeout:
                        gate.release(f"hard timeout {a.hard_timeout} s after health")
                        log("hard timeout: exiting")
                        gate.save(phase="hard-timeout")
                        os._exit(3)
                    if gate.mine_mtime is not None:
                        ok, why = gate.other_usable()
                        if not ok:
                            gate.release(f"other backend no longer usable: {why}")
            threading.Thread(target=guard, daemon=True).start()
        log(f"profile {a.profile} url {url}: waiting for /health (up to {a.health_timeout} s)")
        while not healthy(url):
            if time.monotonic() - t0 > a.health_timeout:
                log("engine not healthy in time: giving up")
                rc = 2
                return rc
            time.sleep(2)
        th[0] = time.monotonic()
        log(f"engine healthy after {th[0] - t0:.0f} s; warming")
        if gate:
            gate.save(phase="warming", t_health=time.time())
        n_req = 0
        fails = 0
        for kind, arg in plan(p):
            if time.monotonic() - th[0] > a.budget:
                log(f"budget {a.budget} s reached: skipping {kind} {arg if arg is not None else ''}")
                continue
            ts = time.monotonic()
            if kind == "prefill":
                rs = phase_prefill(url, arg)
                n_req += len(rs)
                fails += sum(1 for _, r in rs if not r.get("ok"))
                log("prefill " + " ".join(f"{r.get('pt')}:{(r.get('ttft') or 0) * 1e3:.0f}ms" for _, r in rs))
            elif kind == "decode":
                rs = phase_decode(url, arg, p["decode_tokens"])
                n_req += len(rs)
                fails += sum(1 for r in rs if not r.get("ok"))
                log(f"decode x{arg}: {json.dumps(summary(rs))}")
            elif kind == "mixed":
                pre, dec = phase_mixed(url, arg, p["mixed_tokens"])
                n_req += len(pre) + len(dec)
                fails += sum(1 for _, r in pre if not r.get("ok")) + sum(1 for r in dec if not r.get("ok"))
                log(f"mixed {arg} decoders: prefill " + " ".join(f"{r.get('pt')}:{(r.get('ttft') or 0) * 1e3:.0f}ms" for _, r in pre)
                    + f"; decoders {json.dumps(summary(dec))}")
            elif kind == "lora":
                names = lora_models(url) if arg == "auto" else arg.split(",")
                rs = phase_lora(url, names)
                n_req += len(rs)
                fails += sum(1 for _, r in rs if not r.get("ok"))
                log(f"lora {len(rs)} adapters: " + " ".join(f"{(r.get('ttft') or 0) * 1e3:.0f}ms ok={r.get('ok')}" for _, r in rs))
            elif kind == "extras":
                j, t = phase_extras(url)
                n_req += 2
                fails += (not j.get("ok")) + (not t.get("ok"))
                log(f"extras json_object {(j.get('ttft') or 0) * 1e3:.0f}ms ok={j.get('ok')} tools {(t.get('ttft') or 0) * 1e3:.0f}ms ok={t.get('ok')}")
            if fails >= 4 and not healthy(url):
                log("engine went away during the warm-up: stopping")
                rc = 4
                break
        log(f"done: {n_req} requests, {fails} failed, {time.monotonic() - th[0]:.0f} s after health")
    except SystemExit as e:
        rc = e.code if isinstance(e.code, int) else 1
        log(f"stopped by signal (rc {rc})")
    finally:
        done.set()
        if gate:
            gate.release("warm-up finished" if rc == 0 else f"exit rc={rc}")
            gate.save(phase="done", rc=rc, t_done=time.time())
    return rc


if __name__ == "__main__":
    sys.exit(main())
