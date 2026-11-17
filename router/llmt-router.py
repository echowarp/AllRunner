#!/usr/bin/env python3
"""llmt-router: prefix-affinity router for several vLLM backends serving the same model (e.g. an Arc Pro B70 on :8000
and an RTX 5090 on :8001).

Sits behind an API gateway (e.g. llama-swap, which holds the API key) and in front of the vLLM engines. An agent's
follow-up turn repeats its earlier messages, so the engine that served the earlier turn still holds that prefix in its KV cache (VRAM, then its
RAM tier). Routing the turn back there skips re-prefilling 10-25k tokens. Routing is a cost estimate, per backend:

    cost = uncached prompt tokens / prefill tok/s  +  in-flight requests x stream_cost  (+ over-cap penalty)

where "cached" = the longest message prefix this router last sent to that backend (times shared_tier for the other
backend, once both use one shared RAM tier). So a request stays on its GPU unless the other one is much less busy.

Finish-time model: with `dec=` given (never leave a GPU idle while another queues work; offload greedily when it
finishes sooner), the cost is the request's estimated FINISH TIME on each backend,
    wait for prefills already queued there (a per-backend "prefill busy until" clock: prefills serialize)
  + uncached prompt tokens / prefill tok/s
  + expected output tokens / per-stream decode tok/s of that backend at (running requests + 1)
plus the delay it causes the requests already running there (weight `externality`): each running request decodes
its remaining ~gen/2 tokens at the slower per-stream rate of n+1 streams, and is slowed by `prefill_slowdown` while
this request's prefill runs. The request goes to the backend with the smallest total.
Parameters live in a JSON file (--config, see router.example.json); it is re-read when it changes (no restart; backends are
matched by name, adding/removing one needs a restart). dec = per-stream decode tok/s at 1, 2, ... streams
(measured; beyond the list the aggregate stays flat). Expected output tokens = running mean of observed completions
(from the responses' usage), capped by the request's max_tokens.

Privacy: only 16-byte hashes of message prefixes and token estimates are kept beyond a request; no prompt or
completion text is logged or written to disk.

Endpoints: POST /v1/chat/completions, /v1/completions (routed); GET /health (200 if any backend is up),
GET /router/metrics (Prometheus), GET /router/state (JSON, no contents); anything else goes to the first healthy backend.
Drain a backend (no new requests; running ones finish): touch /run/llmt-router/drain-<name>; remove the file to undo.

  llmt-router.py --listen 127.0.0.1:8010 \\
     --backend name=arc,url=http://127.0.0.1:8000,cap=8,prefill=2000,stream_cost=0.6 \\
     --backend name=rtx5090,url=http://127.0.0.1:8001,cap=16,prefill=6000,stream_cost=0.3
Stdlib only (asyncio). Tests: tests/test_*.py (fake backends, no GPU).
"""
from __future__ import annotations

import uuid
import argparse
import asyncio
import collections
import fnmatch
import hashlib
import json
import logging
import logging.handlers
import os
import re
import struct
import time
import urllib.parse

log = logging.getLogger("llmt-router")
dlog = logging.getLogger("llmt-router.decisions")   # one line per routed request, no content (also to --decision-log)
HOP = {"connection", "keep-alive", "transfer-encoding", "content-length", "proxy-connection", "te", "trailer", "upgrade"}
ROUTED = ("/v1/chat/completions", "/v1/completions")


def _has_image(req) -> bool:
    """Split mode skips multimodal requests (image encoder outputs are not in the RAM tier)."""
    try:
        for m in req.get("messages") or ():
            c = m.get("content")
            if isinstance(c, list) and any(isinstance(p, dict) and p.get("type") not in (None, "text") for p in c):
                return True
    except AttributeError:
        return False
    return False
def _n_images(req) -> int:
    """Images in a chat request (engines reject prompts above limit_mm_per_prompt)."""
    n = 0
    try:
        for m in req.get("messages") or ():
            c = m.get("content")
            if isinstance(c, list):
                n += sum(1 for p in c if isinstance(p, dict) and p.get("type") in ("image_url", "image", "input_image"))
    except AttributeError:
        return 0
    return n


# SESSION TAGS: requests that carry a session id (OpenCode and other agent clients) get X-Request-Id = a session-hold
# tag "arcs" + 12 hex (session) + 12 hex (parent session or zeros) + one agent letter + "-" + random, so an engine-side
# session hold (and anything else keyed by session) sees it. Opaque: a keyed blake2b with a per-start random key, never
# logged. Headers: x-opencode-session-id / x-session-affinity / x-session-id (session), x-opencode-parent-session-id /
# x-parent-session-id (parent), x-opencode-agent / x-agent (compaction -> c, title -> t, summary -> s, other -> m).
_SESS_KEY = os.urandom(16)
_SESS_H = ("x-opencode-session-id", "x-session-affinity", "x-session-id")
_PARENT_H = ("x-opencode-parent-session-id", "x-parent-session-id")
_AGENT_H = ("x-opencode-agent", "x-agent")


def _sess_hash(v: str) -> str:
    return hashlib.blake2b(v.encode("utf-8", "replace"), key=_SESS_KEY, digest_size=6).hexdigest()


def session_tag(hdrs) -> str | None:
    """X-Request-Id value for a request with a session header, else None."""
    h = {k.lower(): v.strip() for k, v in hdrs}
    sess = next((h[k] for k in _SESS_H if h.get(k)), None)
    if not sess:
        return None
    par = next((h[k] for k in _PARENT_H if h.get(k)), None)
    ag = next((h[k].lower() for k in _AGENT_H if h.get(k)), "")
    letter = ("c" if "compact" in ag else "t" if "title" in ag else "s" if "summar" in ag else "m") if ag else ""
    return "arcs" + _sess_hash(sess) + (_sess_hash(par) if par else "0" * 12) + letter + "-" + uuid.uuid4().hex[:16]


# SESSION + PRIORITY POLICY (router.json "session_policy", off by default). If a request carries session tags, they
# are used; untagged requests are routed exactly as without the policy. The mechanisms are generic; which agent gets
# which treatment lives in a policy file (router.json "policy_file", hot-reloaded; see router-policy.example.json).
# Request headers (client contract): X-Session-Id (one per conversation), X-Parent-Session-Id (children),
# X-Agent (role: main / summary / compaction / ...), X-Agent-Name (slug [a-z0-9][a-z0-9_.:-]*, <= 64), X-Priority
# (a tier name).
#   tiers: name -> {rank (0 = most urgent), vllm_priority (engines with priority_sched), cache (normal / low / lowest:
#          the RAM tier's value class), background (true: never takes a GPU that has more agent work than another,
#          waits up to low_wait_s while every GPU is busy)}. X-Priority picks the tier; else agent_priority[name];
#          else the class learned for that agent (AgentStats); else default_tier.
#   rules: first match on the agent name (globs) and role: retention (session = the session's chain stays normal until
#          a terminal_roles request, which drops what only that session used to lowest; group = normal; oneshot =
#          its new blocks low; lowest), place (session affinity always first; then group = the replica of the group
#          key, parent = the parent session's replica; background tiers otherwise go to the least agent load), group
#          (name / name+session: requests that share a prefix or form one batch go together).
#   value class sent to the RAM tier = the lower of the tier's cache class and the retention's.
# Names and raw session ids never leave the router: the engines get 64-bit keyed hashes and classes; agent names appear
# only in local metrics / logs (sanitized, capped).
_NAME_H = ("x-agent-name", "x-llmt-agent-name")
_PRIO_H = ("x-priority", "x-llmt-priority")
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_.:-]{0,63}$")
AGENT_LABEL_MAX = 64           # distinct agent keys in metrics / stats (more -> "other")
CLASS_RANK = {"normal": 0, "low": 1, "lowest": 2}
RANK_CLASS = {0: "normal", 1: "low", 2: "lowest"}
RETENTION_CLASS = {"session": "normal", "group": "normal", "oneshot": "low", "lowest": "lowest"}
# Generic defaults (no policy file): four tiers, every named request a session-retained agent
DEFAULT_TIERS = {
    "high": {"rank": 0, "vllm_priority": 0, "cache": "normal"},
    "normal": {"rank": 1, "vllm_priority": 0, "cache": "normal"},
    "low": {"rank": 2, "vllm_priority": 10, "cache": "low", "background": True},
    "background": {"rank": 3, "vllm_priority": 20, "cache": "lowest", "background": True},
}
DEFAULT_RULE = {"retention": "session", "place": "parent"}


def _sess_hash64(v: str) -> str:
    """64-bit keyed session id for the engines' RAM tier (kv_transfer_params llmt_sess; the broker's session tag)."""
    return hashlib.blake2b(v.encode("utf-8", "replace"), key=_SESS_KEY, digest_size=8).hexdigest()


def req_tags(hdrs) -> dict:
    """Tags of a request: sess / parent (64-bit keyed hex or None), name (agent name slug or ""), role (X-Agent, lower
    case, <= 32 chars), letter (role letter: c compaction, t title, s summary, m main), prio (X-Priority tier name,
    lower case, or None), tagged (any)."""
    h = {k.lower(): v.strip() for k, v in hdrs}
    sess = next((h[k] for k in _SESS_H if h.get(k)), None)
    par = next((h[k] for k in _PARENT_H if h.get(k)), None)
    role = next((h[k].lower() for k in _AGENT_H if h.get(k)), "")[:32]
    name = next((h[k].lower() for k in _NAME_H if h.get(k)), "")
    name = name if _NAME_RE.match(name) else ""
    p = next((h[k].lower() for k in _PRIO_H if h.get(k)), "")[:32] or None
    letter = ("c" if "compact" in role else "t" if "title" in role else "s" if "summar" in role else "m") if role else ""
    return {"sess": _sess_hash64(sess) if sess else None, "parent": _sess_hash64(par) if par else None, "name": name,
            "role": role, "letter": letter, "prio": p, "tagged": bool(sess or par or role or name or p)}


def agent_key(t: dict) -> str:
    """Stats / learning key of a request: its agent name, else "role:<X-Agent>", "" = none."""
    return t.get("name") or (f"role:{t['role']}" if t.get("role") else "")


def agent_label(a: str) -> str:
    """Metrics label for an agent key: [a-z0-9_.:-], <= 64 chars ("(none)" = no name / role)."""
    if not a:
        return "(none)"
    return re.sub(r"[^a-z0-9_.:-]", "_", a)[:64]


def match_rule(rules: list, name: str, role: str) -> dict | None:
    for ru in rules or ():
        pats = ru.get("match")
        if pats is not None and not any(fnmatch.fnmatchcase(name, p) for p in ([pats] if isinstance(pats, str) else pats)):
            continue
        if ru.get("roles") and role not in ru["roles"]:
            continue
        return ru
    return None


class AgentStats:
    """Per agent key: what its traffic looks like (no content). Window of the last `window` requests: follow-up (the
    request continues a conversation this router routed before: prefix match >= affinity_min_tokens and
    >= affinity_min_frac), prefix reuse (matched tokens / prompt tokens, shared beginnings included), concurrency (other
    requests of the same key in flight at arrival)."""

    def __init__(self, window: int = 200):
        self.win = collections.deque(maxlen=window)
        self.requests = 0
        self.inflight = 0
        self.learned = None          # "low" / "normal" once learn_min requests are in
        self.logged = None
        self.tiers = collections.Counter()

    def add(self, follow: bool, reuse: float, conc: int):
        self.requests += 1
        self.win.append((follow, reuse, conc))

    def summary(self) -> dict:
        n = len(self.win) or 1
        return {"n": len(self.win), "follow": sum(1 for f, _, _ in self.win if f) / n,
                "reuse": sum(r for _, r, _ in self.win) / n, "conc": sum(c for _, _, c in self.win) / n}

    def learn(self, p: dict) -> str | None:
        """Learned class from the window (None until learn_min requests): low = single-stream, no follow-ups, low reuse."""
        sm = self.summary()
        if sm["n"] < int(p.get("learn_min", 20)):
            return None
        low = (sm["follow"] <= float(p.get("learn_follow_max", 0.1)) and sm["reuse"] <= float(p.get("learn_reuse_max", 0.6))
               and sm["conc"] <= float(p.get("learn_conc_max", 0.5)))
        self.learned = "low" if low else "normal"
        return self.learned


DRAIN_DIR = os.environ.get("LLMT_ROUTER_DRAIN_DIR", "/run/llmt-router")


class Backend:
    def __init__(self, spec):
        kv = dict(p.split("=", 1) for p in spec.split(",")) if isinstance(spec, str) else dict(spec)
        self.name = kv["name"]
        u = urllib.parse.urlsplit(kv["url"])
        self.host, self.port = u.hostname, u.port or 80
        self.cap = int(kv.get("cap", 8))                       # concurrent requests before the over-cap penalty
        # the engine's max_model_len; 0 = unknown. A request whose estimated prompt
        # (x ctx_margin, the estimate is chars / 3.3) + max_tokens exceeds it never goes there while a backend that fits
        # is usable (the engine would answer 400)
        self.max_ctx = int(kv.get("max_ctx", 0))
        self.max_images = int(kv.get("max_images", 0))         # engine limit_mm_per_prompt image (0 = unknown)
        # LORA-SUFFIX: the engine serves LoRA adapters by name (vLLM --enable-lora + a resolver, e.g. llmt_lora);
        # requests for "<model>:<adapter>" go only to backends with lora true
        self.lora = _truthy(kv.get("lora", False))
        self.prefill = float(kv.get("prefill", 2000))          # prefill tokens/s (uncached part of the prompt)
        self.stream_cost = float(kv.get("stream_cost", 0.5))   # seconds of expected delay per in-flight request
        # v2: per-stream decode tok/s at 1, 2, ... running requests ("dec=134/149/134"), "/"-separated
        d = kv.get("dec")
        self.dec = [float(x) for x in (d.split("/") if isinstance(d, str) else d)] if d else []
        self.prefill_busy_until = 0.0                           # monotonic time the queued prefills should be done
        self.prefill_slowdown = float(kv["prefill_slowdown"]) if "prefill_slowdown" in kv else None   # per-GPU override
        self.load_gbps = float(kv.get("load_gbps", 4.0))       # split mode: RAM-tier load bandwidth into this GPU (GB/s)
        # prefer this GPU for every request while fewer than prefer_until requests are in
        # flight on it, regardless of the per-request cost model and of affinity (the prefix is reachable through the
        # shared RAM tier). Evidence (a day of production traffic on an Arc B70 + RTX 5090): the Arc served 29% of
        # requests but 52% of agent wait; per
        # stream it decodes 90/67/53/50/41/37/32/24 tok/s with 0..7 others vs the 5090's 219/170/120/98/89/~70, and the
        # 5090's 8-stream aggregate (851 tok/s) is 4.4x the Arc's. The cost model is per request and cannot see that a
        # request sent to the Arc is slowed by the next arrivals there. 0 = off.
        self.prefer_until = int(kv.get("prefer_until", 0))
        # the engine runs --scheduling-policy priority: low-priority requests carry vLLM's "priority" field
        # (session_policy tiers' vllm_priority; lower = earlier). False = never sent
        self.priority_sched = bool(kv.get("priority_sched", False))
        self.low_inflight = 0                                   # low-priority requests in flight (session policy)
        self.healthy = False
        self.fails = 0
        self.inflight = 0
        # work conservation (no GPU idle while requests wait elsewhere): the engine's own view, polled
        # from its /metrics by Server.engine_loop: requests queued in the engine, running, KV cache fill (0..1)
        self.eng_waiting = 0
        self.eng_running = 0
        self.eng_kv = 0.0
        self.eng_t = 0.0                                        # monotonic time of the last good poll
        # STALL-DETECT: requests pending in the engine but its
        # generation + prompt token counters flat for stall_s (scheduler deadlock, GPU hang that still answers /health):
        # no new traffic (not even affinity) until the counters move again
        self.stalled = False
        self.stalls = 0
        self.tok_total = None
        self.tok_t = 0.0
        self.n = collections.Counter()                          # requests by reason

    @property
    def drained(self) -> bool:
        return os.path.exists(os.path.join(DRAIN_DIR, "drain-" + self.name))

    @property
    def held(self) -> bool:
        """A GPU deliberately out of production (hold flag) may run a lab engine on the same port; never
        send it traffic, not even as the last-resort fallback (a fast 503 beats a silent reroute into an experiment)."""
        d = DRAIN_DIR
        return os.path.exists(os.path.join(d, "hold-" + self.name)) or os.path.exists(os.path.join(d, "hold-" + self.name.replace("rtx", "")))

    @property
    def excluded(self) -> bool:
        """held AND drained = a GPU out of production that may run a lab engine on its port: no traffic at all. A held GPU
        whose drain was lifted is a hand-started test stack that wants traffic (5090 test windows)."""
        return self.held and self.drained

    def usable(self) -> bool:
        return self.healthy and not self.drained and not self.stalled

    def update(self, cfg: dict):
        """Hot reload of the tunables (not the address)."""
        for k in ("cap", "prefill", "stream_cost", "load_gbps", "max_ctx", "max_images"):
            if k in cfg:
                setattr(self, k, type(getattr(self, k))(cfg[k]))
        if "lora" in cfg:
            self.lora = _truthy(cfg["lora"])
        self.prefill_slowdown = float(cfg["prefill_slowdown"]) if "prefill_slowdown" in cfg else None
        if "dec" in cfg:
            self.dec = [float(x) for x in cfg["dec"]]
        self.prefer_until = int(cfg.get("prefer_until", 0))
        self.priority_sched = bool(cfg.get("priority_sched", False))

    def per_stream_f(self, n: float) -> float:
        """per_stream for a fractional load (linear between the measured points)."""
        lo = max(int(n), 1)
        f = n - int(n) if n >= 1 else 0.0
        return self.per_stream(lo) * (1 - f) + self.per_stream(lo + 1) * f

    def per_stream(self, n: int) -> float:
        """Per-stream decode tok/s with n requests running (aggregate flat beyond the measured list)."""
        if n <= len(self.dec):
            return self.dec[n - 1]
        return self.dec[-1] * len(self.dec) / n


class PrefixMap:
    """LRU: prefix hash -> (backend name, prompt tokens up to that prefix, time)."""

    def save(self, path: str, max_age: float) -> int:
        """Persist entries younger than max_age (hashes + backend + counts only, no content)."""
        now = time.time(); n = 0
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            for h, (b, tok, ts) in self.d.items():
                if now - ts <= max_age:
                    f.write(f"{h.hex()} {b} {tok} {ts:.1f}\n"); n += 1
        os.replace(tmp, path)
        return n

    def load(self, path: str, max_age: float) -> int:
        now = time.time(); n = 0
        try:
            for ln in open(path):
                hx, b, tok, ts = ln.split()
                if now - float(ts) <= max_age:
                    self.d[bytes.fromhex(hx)] = (b, int(tok), float(ts)); n += 1
        except (OSError, ValueError):
            return n
        return n

    def __init__(self, max_entries: int):
        self.d: collections.OrderedDict[bytes, tuple[str, int, float]] = collections.OrderedDict()
        self.max = max_entries

    def get(self, h: bytes):
        v = self.d.get(h)
        if v is not None:
            self.d.move_to_end(h)
        return v

    def put(self, h: bytes, backend: str, tokens: int):
        self.d[h] = (backend, tokens, time.time())
        self.d.move_to_end(h)
        while len(self.d) > self.max:
            self.d.popitem(last=False)


def _h(prev: bytes, obj) -> bytes:
    return hashlib.blake2b(prev + json.dumps(obj, sort_keys=True, separators=(",", ":")).encode(), digest_size=16).digest()


def prefix_chain(path: str, body: dict, chars_per_token: float) -> list[tuple[bytes, int]]:
    """Cumulative (hash, estimated prompt tokens) at each message boundary. The root covers what the chat template
    puts before the first message (tools, template options) and the model, so different tool sets never match."""
    root = _h(b"", {"m": body.get("model"), "tools": body.get("tools"), "ct": body.get("chat_template_kwargs")})
    out, h, chars = [], root, 0
    if path.startswith("/v1/chat/completions"):
        # the whole message (tool_calls arguments, reasoning) and the tools array count; content alone
        # missed ~20-40% of agent prompts (context-length 400s at a 124.8k max_model_len)
        if body.get("tools"):
            chars += len(json.dumps(body.get("tools"), ensure_ascii=False))
        for m in body.get("messages") or []:
            h = _h(h, m)
            chars += len(json.dumps(m, ensure_ascii=False)) if isinstance(m, dict) else 16
            out.append((h, int(chars / chars_per_token)))
    else:                                               # /v1/completions: 2048-char pieces of the prompt
        p = body.get("prompt")
        if isinstance(p, list) and p and all(isinstance(t, int) for t in p):
            # token-id prompt: exact token counts, 512-token pieces (json text would count ~2x tokens)
            for i in range(512, len(p) + 1, 512):
                h = _h(root, p[:i])
                out.append((h, i))
            out.append((b"", len(p)))
            return out
        p = p if isinstance(p, str) else json.dumps(p)
        for i in range(2048, len(p) + 1, 2048):
            h = _h(root, p[:i])
            out.append((h, int(i / chars_per_token)))
        chars = len(p)
    out.append((b"", int(chars / chars_per_token)))     # last entry: total estimate (hash unused)
    return out


class BrokerHints:
    """Prefix-affinity hints from an AllBroker KV broker (docs/prefix-hints.md in AllBroker): WHERE asks, for a request's
    message-prefix hashes (longest first), which engines hinted one of them and how many leading chunks are in its
    host-RAM tier right now. Engines send the hints (HINT) when a request carrying `kv_transfer_params.llmt_anchor`
    finishes; the router sets that anchor = its hash of the full prompt.
    Read-only on the broker; one request in flight (asyncio lock), `timeout_ms` per call. Any failure (no socket, no
    FEAT_PREFIX_HINTS, timeout, bad frame) = None: the router then routes exactly as without hints. A dead connection is
    retried at most every `retry_s`. Stdlib only: the broker's frame format is a few struct calls."""

    OP_HELLO, OP_FEATURES, OP_WHERE = 1, 18, 31
    FEAT_PREFIX_HINTS = 1 << 5
    OBSERVER_ENGINE_ID = 0xBB01
    HINT_VRAM = 1

    def __init__(self, socket_path: str, timeout_ms: float = 5.0, retry_s: float = 1.0):
        self.path, self.timeout, self.retry_s = socket_path, timeout_ms / 1000.0, retry_s
        self.rd = self.wr = None
        self.lock = asyncio.Lock()
        self.req_id = 0
        self.next_try = 0.0
        self.timeouts = 0
        self.n = collections.Counter()       # ok, error, timeout, off (no feature), connect
        self.seconds = 0.0

    def close(self):
        if self.wr is not None:
            try:
                self.wr.close()
            except (OSError, RuntimeError):
                pass
        self.rd = self.wr = None

    async def _call(self, op: int, payload: bytes) -> tuple[int, bytes]:
        self.req_id += 1
        rid = self.req_id
        self.wr.write(struct.pack("<IHHQ", 12 + len(payload), op, 0, rid) + payload)
        await self.wr.drain()
        while True:                                  # replies to earlier calls that timed out are skipped by req_id
            (n,) = struct.unpack("<I", await self.rd.readexactly(4))
            if not 12 <= n <= 4 << 20:
                raise ValueError("bad frame")
            f = await self.rd.readexactly(n)
            rop, flags, got = struct.unpack_from("<HHQ", f)
            if got == rid and rop == op:
                return struct.unpack_from("<H", f, 12)[0], f[14:]

    async def _connect(self):
        self.n["connect"] += 1
        self.rd, self.wr = await asyncio.open_unix_connection(self.path)
        name = b"llmt-router"
        st, _ = await self._call(self.OP_HELLO, struct.pack("<HHH", self.OBSERVER_ENGINE_ID, 1, len(name)) + name)
        if st != 0:
            raise ValueError(f"hello status {st}")
        st, body = await self._call(self.OP_FEATURES, struct.pack("<QI", 0, 0))
        feats = struct.unpack_from("<Q", body)[0] if st == 0 and len(body) >= 8 else 0
        if not feats & self.FEAT_PREFIX_HINTS:
            raise LookupError("broker without FEAT_PREFIX_HINTS")

    @staticmethod
    def parse_where(body: bytes) -> list[dict]:
        (n,) = struct.unpack_from("<I", body)
        o, out = 4, []
        for _ in range(n):
            idx, eng, ln = struct.unpack_from("<IHH", body, o)
            o += 8
            name = body[o:o + ln].decode("utf-8", "replace")
            o += ln
            ns, age, ct, nch, nres, fl = struct.unpack_from("<IQIIIB", body, o)
            o += 25
            out.append({"anchor_idx": idx, "engine_id": eng, "name": name, "ns": ns, "age_ms": age, "chunk_tokens": ct,
                        "n_chunks": nch, "n_resident": nres, "flags": fl})
        if o != len(body):
            raise ValueError("trailing bytes")
        return out

    async def where(self, anchors: list[bytes]) -> list[dict] | None:
        anchors = [a for a in anchors if len(a) == 16][:256]
        if not anchors:
            return []
        if self.wr is None and time.monotonic() < self.next_try:
            self.n["down"] += 1
            return None
        t0 = time.monotonic()
        try:
            async with self.lock:
                async def go():
                    if self.wr is None:
                        await self._connect()
                    st, body = await self._call(self.OP_WHERE, struct.pack("<I", len(anchors)) + b"".join(anchors))
                    if st != 0:
                        raise ValueError(f"where status {st}")
                    return self.parse_where(body)
                r = await asyncio.wait_for(go(), self.timeout if self.wr is not None else max(self.timeout, 0.05))
            self.timeouts = 0
            self.n["ok"] += 1
            return r
        except asyncio.TimeoutError:
            self.n["timeout"] += 1
            self.timeouts += 1
            if self.timeouts >= 8:                       # hung: drop it (a late reply would be skipped anyway)
                self.close()
                self.next_try = time.monotonic() + self.retry_s
            return None
        except LookupError:
            self.n["off"] += 1
            self.close()
            self.next_try = time.monotonic() + 60.0      # an old broker: ask again in a minute (it may be upgraded)
            return None
        except (OSError, ValueError, struct.error, asyncio.IncompleteReadError, ConnectionError):
            self.n["error"] += 1
            self.close()
            self.next_try = time.monotonic() + self.retry_s
            return None
        finally:
            self.seconds += time.monotonic() - t0


def hint_tokens(places: list[dict] | None, engines: dict, backends, total: int, vram_ttl_s: float):
    """Per backend name: (tokens of the prompt's prefix in its RAM tier, VRAM-recent hint) from a WHERE answer; None if
    the broker gave none. `engines`: backend name -> broker HELLO name or engine id (default: the backend name)."""
    if places is None:
        return None
    out = {}
    for b in backends:
        key = engines.get(b.name, b.name)
        best = (0, False)
        for p in places:
            if p["name"] == key or (isinstance(key, int) and p["engine_id"] == key):
                tok = min(p["n_resident"] * p["chunk_tokens"], total)
                vram = bool(p["flags"] & BrokerHints.HINT_VRAM) and p["age_ms"] < vram_ttl_s * 1000
                best = max(best, (tok, vram))
        out[b.name] = best
    return out


class Router:
    def __init__(self, backends: list[Backend], max_entries=200_000, shared_tier=0.0, chars_per_token=3.3):
        self.backends = backends
        self.by_name = {b.name: b for b in backends}
        self.map = PrefixMap(max_entries)
        self.shared_tier = shared_tier
        self.cpt = chars_per_token
        self.started = time.time()
        self.gen_mean = 400.0                                    # running mean of completion tokens (v2)
        self.externality = 1.0                                   # weight of the delay caused to running requests
        self.prefill_slowdown = 0.4                              # running streams' slowdown while a prefill runs
        # agents come back within seconds after a tool call: a conversation that finished a request on a backend in
        # the last return_window_s counts as returning_weight of a running request there
        self.return_window_s = 20.0
        self.returning_weight = 0.5
        self._recent: dict[str, collections.deque] = {b.name: collections.deque() for b in backends}
        # work-conserving spill: a GPU whose engine has requests waiting or a KV cache above spill_kv is
        # "saturated"; while another usable GPU is not, new requests go there (reason "spill"), whatever the cost model,
        # prefer_until or affinity say. Engine numbers older than spill_stale_s are ignored. spill_kv 0 = off.
        self.spill_kv = 0.0
        self.stall_s = 90.0     # STALL-DETECT; config "stall_s", 0 = off
        self.spill_stale_s = 3.0
        self.ctx_margin = 1.1                                    # safety factor on the prompt estimate vs max_ctx
        # NEW conversations (no prefix on any GPU; incl. the
        # first turn after a client compaction) go to the backend with the fewest active conversations per unit of
        # balance weight (active = in flight + returning within return_window_s). {name: weight}; {} = off.
        self.balance_new: dict = {}
        self.affinity_min_tokens = 0.0                          # 0 = any prefix match is affinity (old behaviour)
        self.affinity_ttl_s = 0.0                               # 0 = affinity never expires (old behaviour)
        self.balance_skip_tokens = 8192.0                       # no-tool requests this large skip balance_new
        self.session_tag_ids = False                            # session headers -> X-Request-Id tag
        self.session_tagged = 0
        # session + priority policy (router.json "session_policy"; {} / on false = off: requests are routed exactly as
        # before and nothing new is sent downstream)
        self.session_policy: dict = {}
        self.policy_doc: dict = {}                              # the policy file's contents (policy_file)
        self.policy_path = ""
        self.policy_override = False                            # --policy-file / LLMT_ROUTER_POLICY wins over router.json
        self._policy_mtime = 0.0
        self.sess_map: dict[str, tuple[str, float]] = {}       # session tag -> (backend, last dispatch, wall clock)
        self.group_map: dict[str, tuple[str, float]] = {}      # group key -> (backend, last dispatch)
        self.agents: dict[str, AgentStats] = {}
        self.cls_n = collections.Counter()                      # (tier, class, source) -> requests
        self.low_waits = 0                                      # background requests held while every GPU was busy
        self.lora_sep = ":"                                     # "<model><sep><adapter>" selects a LoRA adapter
        self.lora_requests = collections.Counter()              # by outcome: routed / bad-name / no-backend
        self.min_out_tokens = 0.0                               # raise smaller client max_tokens to this (0 = off)
        # loop stop: end a stream whose generated text repeats: over the last `words` words, the share
        # of distinct `ngram`-word phrases stays below `min_distinct` for `confirm` checks `every` words apart. Calibrated
        # on 2,987 sampled outputs >= 1,600 words: normal minimum 0.349; a 190k-token runaway fell below 0.15 by ~2,800
        # words. {} = off.
        self.loop_stop: dict = {}
        self.summary_target = ""                                # "" = large no-tool jobs by finish time; name = there first
        self.affinity_min_frac = 0.0
        # real prompt tokens (response usage) / raw estimate, running mean (clamped 0.5..2), applied to every
        # estimate; exact_band: estimates within [lo, hi] x the smallest usable max_ctx are counted exactly (/tokenize)
        self.est_scale = 1.0
        self.est_n = 0
        self.exact_band = (0.7, 1.4)
        # prefix hints (AllBroker HINT / WHERE; docs/prefix-hints.md there): off by default. When on and the broker
        # answers, a backend's RAM-tier share of the prompt comes from the broker (engines report what they hold)
        # instead of the shared_tier guess; a RAM holder is not a "new" conversation (no balance_new). Broker down or
        # off = exactly the old behaviour.
        self.prefix_hints = dict(Router.prefix_hints)
        self.hints: BrokerHints | None = None

    # ------------------------------------------------------------ session + priority policy

    @property
    def policy_on(self) -> bool:
        return bool(self.session_policy.get("on"))

    @property
    def pol(self) -> dict:
        """router.json session_policy with the policy file's keys over it."""
        return {**self.session_policy, **self.policy_doc}

    def tiers(self) -> dict:
        return self.policy_doc.get("tiers") or self.session_policy.get("tiers") or DEFAULT_TIERS

    def stats_for(self, k: str) -> AgentStats:
        if k not in self.agents and len(self.agents) >= AGENT_LABEL_MAX:
            k = "other"
        return self.agents.setdefault(k, AgentStats(int(self.pol.get("learn_window", 200))))

    def classify(self, t: dict) -> dict:
        """The request's policy: tier (name, spec, source), rule, value class for the RAM tier, terminal flag,
        group key. Untagged requests: default tier, no rule, normal class."""
        pol, tiers = self.pol, self.tiers()
        dflt = pol.get("default_tier") or ("normal" if "normal" in tiers else min(tiers, key=lambda n: tiers[n].get("rank", 0)))
        k = agent_key(t)
        ap = {str(a).lower(): str(v).lower() for a, v in (pol.get("agent_priority") or {}).items()}
        if t.get("prio") and t["prio"] in tiers:
            tier, src = t["prio"], "header"
        elif t.get("prio"):
            tier, src = dflt, "header-unknown"
        elif k and k in ap and ap[k] in tiers:
            tier, src = ap[k], "map"
        elif k and pol.get("learn", True) and k in self.agents and self.agents[k].learned == "low" \
                and (pol.get("learn_tier") or "low") in tiers:
            tier, src = pol.get("learn_tier") or "low", "learned"
        else:
            tier, src = dflt, "default"
        spec = tiers[tier]
        rule = None
        if t.get("tagged") and (t.get("name") or t.get("role") or t.get("sess")):
            rule = match_rule(pol.get("rules") or [], t.get("name") or "", t.get("role") or "") \
                or pol.get("default_rule") or DEFAULT_RULE
        ret = (rule or {}).get("retention", "session")
        rank = max(CLASS_RANK.get(str(spec.get("cache", "normal")), 0), CLASS_RANK[RETENTION_CLASS.get(ret, "normal")] if rule else 0)
        cls = RANK_CLASS[rank]
        top = min(x.get("rank", 0) for x in tiers.values())
        term = bool(rule and t.get("role") and t["role"] in (rule.get("terminal_roles") or ()))
        gk = None
        if rule and rule.get("group"):
            gk = (t.get("name") or t.get("role") or "") + ("|" + (t.get("sess") or "") if rule["group"] == "name+session" else "")
        return {"tier": tier, "spec": spec, "src": src, "rule": rule, "cls": cls, "term": term, "group": gk,
                "agent": cls == "normal" and spec.get("rank", 0) == top and bool(t.get("tagged")),
                "background": bool(spec.get("background"))}

    def observe(self, t: dict, pc: dict, matched: int, total: int, follow: bool) -> AgentStats:
        """Count one routed request in its agent key's stats; re-learn and log a class change once."""
        k = agent_key(t)
        st = self.stats_for(k)
        st.add(follow, min(matched / total, 1.0) if total > 0 else 0.0, st.inflight)
        st.tiers[pc["tier"]] += 1
        if k:
            old = st.learned
            new = st.learn(self.pol)
            if new and new != old and new != st.logged:
                st.logged = new
                sm = st.summary()
                log.info("agent %r: traffic looks %s after %d requests (follow-ups %.0f%%, prefix reuse %.0f%%, "
                         "concurrency %.2f)", agent_label(k), "single-stream one-off (learned low)" if new == "low"
                         else "interactive (normal)", st.requests, 100 * sm["follow"], 100 * sm["reuse"], sm["conc"])
        return st

    def _fresh(self, m: dict, key: str | None, ttl: float) -> Backend | None:
        if not key:
            return None
        v = m.get(key)
        if v is None or time.time() - v[1] > ttl:
            return None
        return self.by_name.get(v[0])

    def sess_backend(self, sess: str | None) -> Backend | None:
        return self._fresh(self.sess_map, sess, float(self.pol.get("session_ttl_s", 1800)))

    def group_backend(self, key: str | None) -> Backend | None:
        return self._fresh(self.group_map, key, float(self.pol.get("group_ttl_s", 600)))

    @staticmethod
    def _note(m: dict, key: str | None, b: Backend) -> None:
        if not key:
            return
        m[key] = (b.name, time.time())
        if len(m) > 20000:                                # drop the oldest quarter
            for k, _ in sorted(m.items(), key=lambda x: x[1][1])[:5000]:
                del m[k]

    def note_session(self, sess: str | None, b: Backend, group: str | None = None) -> None:
        self._note(self.sess_map, sess, b)
        self._note(self.group_map, group, b)

    def agent_load(self, b: Backend) -> float:
        """Agent work on a backend: its requests in flight that are not background + returning conversations."""
        return (b.inflight - b.low_inflight) + self.returning_weight * len(self._recent[b.name])

    def busy(self, b: Backend, now: float) -> bool:
        """For background placement: the engine has requests waiting or its KV is above low_kv (default spill_kv or
        0.9), from its /metrics (fresh within spill_stale_s); spill's saturation counts too."""
        if self.saturated(b, now):
            return True
        if not b.eng_t or now - b.eng_t > self.spill_stale_s:
            return False
        return b.eng_waiting > 0 or b.eng_kv >= float(self.pol.get("low_kv", self.spill_kv or 0.9))

    def apply_policy(self, order: list[Backend], reason: str, tags: dict | None, pc: dict | None, fit: list[Backend],
                     now: float) -> tuple[list[Backend], str]:
        """Session / group / parent placement and background placement on top of the cost-model order."""
        if not self.policy_on or tags is None or pc is None or not tags.get("tagged") or reason == "ctx":
            return order, reason
        ok = [b for b in order if b in fit and not self.saturated(b, now) and not b.stalled]
        bg = pc["background"]
        if bg:
            ok = [b for b in ok if not self.busy(b, now)] or ok
        place = (pc["rule"] or {}).get("place", "parent")
        cands = [(self.sess_backend(tags.get("sess")), "session")]
        if place == "group":
            cands.append((self.group_backend(pc["group"]), "group"))
        if place in ("parent", "group") and self.pol.get("parent_affinity", True):
            pb = self.sess_backend(tags.get("parent"))
            cands.append((pb if pb is not None and pb.inflight < pb.cap else None, "parent"))
        for sb, why in cands:
            # a background request keeps its session's / group's replica (its prefix) unless another GPU carries less
            # agent work while that one is busy with agents
            if sb is not None and sb in ok and (not bg or self.agent_load(sb) <= min(self.agent_load(b) for b in ok)
                                                or why != "parent"):
                return [sb] + [b for b in order if b is not sb], (reason if order[0] is sb and reason == "affinity" else why)
        if bg and ok:
            # never take an agent's GPU slot when another GPU has less agent work: least agent load, low_target, cost
            tgt = self.pol.get("low_target") or ""
            best = min(ok, key=lambda b: (round(self.agent_load(b), 3), b.name != tgt, order.index(b)))
            return [best] + [b for b in order if b is not best], "background"
        return order, reason

    def saturated(self, b: Backend, now: float) -> bool:
        if not self.spill_kv or now - b.eng_t > self.spill_stale_s:
            return False
        return b.eng_waiting > 0 or b.eng_kv >= self.spill_kv

    def choose(self, chain: list[tuple[bytes, int]], max_tokens: int | None = None,
               exact: int | None = None, images: int = 0, tools: bool = True,
               places: list[dict] | None = None, tags: dict | None = None,
               pc: dict | None = None) -> tuple[list[Backend], str, int]:
        """Backends in preference order, the reason for the first, and the matched tokens on the affinity backend.
        exact = the prompt's real token count (from /tokenize) when known. places = the broker's WHERE answer for this
        prompt (prefix hints), None = no hint information (hints off or broker unreachable): old behaviour."""
        total = exact if exact else int(chain[-1][1] * self.est_scale)
        ph = self.prefix_hints
        hint = hint_tokens(places, ph.get("engines") or {}, self.backends, total, float(ph.get("vram_ttl_s", 0)))
        aff, matched = None, 0
        stale = False
        for h, tok in reversed(chain[:-1]):
            v = self.map.get(h)
            if v is not None and v[0] in self.by_name:
                aff, matched = self.by_name[v[0]], tok
                # affinity needs recency: a prefix last routed longer ago than
                # affinity_ttl_s is likely gone from that GPU's VRAM; it no longer forces placement (load + RAM tier decide)
                stale = bool(self.affinity_ttl_s) and time.time() - v[2] > self.affinity_ttl_s
                break
        # force-route only on matches much longer than a system prompt: a prefix shorter than
        # affinity_min_tokens or than affinity_min_frac of the prompt is a shared beginning (OpenCode: ~26k tokens of system
        # prompt + tools), not the same session: no affinity, placed by load (balance_new), the shared part still counts
        # through shared_tier on every backend
        had_aff, aff0 = aff is not None, aff
        weak = aff is not None and (matched * self.est_scale < self.affinity_min_tokens
                                    or matched * self.est_scale < self.affinity_min_frac * total)
        if weak or stale:
            aff = None
        usable = [b for b in self.backends if b.usable()] or [b for b in self.backends if b.healthy and not b.excluded]
        if not usable:
            return [b for b in self.backends if not b.excluded], "none-healthy", 0

        gen = min(self.gen_mean, max_tokens) if max_tokens else self.gen_mean
        now = time.monotonic()
        need = total * (1.0 if exact else self.ctx_margin) + (max_tokens or 0)
        fit = [b for b in usable if (not b.max_ctx or need <= b.max_ctx) and (not b.max_images or images <= b.max_images)]
        too_long = len(fit) < len(usable)
        if not fit and images and any(b.max_images for b in usable):   # too many images everywhere: the largest limit
            topi = max(b.max_images or 1 << 30 for b in usable)
            fit = [b for b in usable if (b.max_images or 1 << 30) == topi]
        if not fit and any(b.max_ctx for b in usable):   # nothing fits the estimate: the largest limit is the best bet
            top = max(b.max_ctx or 1 << 30 for b in usable)
            fit = [b for b in usable if (b.max_ctx or 1 << 30) == top]
        if fit:
            usable = fit

        for b in self.backends:                              # expire finished conversations past the return window
            q = self._recent[b.name]
            while q and now - q[0] > self.return_window_s:
                q.popleft()

        def load(b: Backend) -> float:
            # returning conversations count on their backend, except the one being routed right now
            ret = len(self._recent[b.name]) - (1 if b is aff and self._recent[b.name] else 0)
            return b.inflight + self.returning_weight * max(ret, 0)

        kvb = float(self.split.get("kv_bytes_per_token", 41000))

        def cached_of(b: Backend) -> tuple[float, float]:
            """(prompt tokens this backend need not prefill, seconds to load the RAM-tier part of them)"""
            vram = matched if b is aff else 0
            if hint is None or b.name not in hint:
                return (matched if b is aff else matched * self.shared_tier), 0.0
            ram = hint[b.name][0]
            if ram <= vram:
                return vram, 0.0
            return ram, (ram - vram) * kvb / (b.load_gbps * 1e9)

        def cost(b: Backend) -> float:
            cached, ram_s = cached_of(b)
            uncached = max(total - cached, 0)
            if b.dec:                                            # v2: estimated finish time + delay caused to others
                n = load(b)
                wait = max(b.prefill_busy_until - now, 0.0)
                pf = uncached / b.prefill
                c = wait + pf + ram_s + gen / b.per_stream_f(n + 1)
                run = b.inflight                             # requests actually decoding there now
                if run:
                    # only running requests are slowed: returning conversations (between tool calls) don't stall
                    ext = run * (gen / 2) * (1 / b.per_stream_f(n + 1) - 1 / b.per_stream_f(n)) \
                        + run * pf * (b.prefill_slowdown if b.prefill_slowdown is not None else self.prefill_slowdown)
                    c += self.externality * max(ext, 0.0)
            else:
                c = uncached / b.prefill + ram_s + b.inflight * b.stream_cost
            if b.inflight >= b.cap:
                c += (b.inflight - b.cap + 1) * 30.0
            return c

        costs = {b.name: cost(b) for b in usable}
        order = sorted(usable, key=lambda b: costs[b.name])
        pref = [b for b in order if b.prefer_until and b.inflight < b.prefer_until]
        preferred = bool(pref) and order[0] is not pref[0]
        if pref:                                            # prefer_until GPUs first (cheapest of them), then the rest by cost
            order = pref + [b for b in order if b not in pref]
        balanced = False
        # a large request without tools (client compaction / long summary:
        # big prefill + long output while the session waits) is placed by estimated finish time, never balanced away
        big_job = (not tools) and total >= self.balance_skip_tokens
        # summary_target = backend name -> large no-tool
        # jobs go there first (background work off the interactive GPU) while it is usable and not saturated; "" = finish time
        summary_first = None
        if big_job and self.summary_target and self.summary_target in self.by_name:
            tb = self.by_name[self.summary_target]
            if tb in usable and not self.saturated(tb, time.monotonic()):
                summary_first = tb
        if summary_first is not None and order[0] is not summary_first:
            order = [summary_first] + [b for b in order if b is not summary_first]
        # a conversation the router forgot (affinity TTL, restart, weak match) whose prefix one GPU still holds in its
        # RAM tier (broker WHERE) is not new: the cost model places it (RAM load vs prefill), no balance_new
        hint_aff = None
        if aff is None and hint:
            hb = max((b for b in usable if b.name in hint), key=lambda b: hint[b.name][0], default=None)
            if hb is not None and hint[hb.name][0] >= max(float(ph.get("min_tokens", 0)), self.affinity_min_tokens,
                                                          self.affinity_min_frac * total):
                hint_aff = hb
        if aff is None and hint_aff is None and self.balance_new and len(order) > 1 and not big_job:
            cand = [b for b in order if self.balance_new.get(b.name, 0) > 0]
            if len(cand) > 1:
                act = lambda b: b.inflight + len(self._recent[b.name])
                best_b = min(cand, key=lambda b: (act(b) / self.balance_new[b.name], -self.balance_new[b.name]))
                if best_b is not order[0]:
                    order = [best_b] + [b for b in order if b is not best_b]
                    balanced = True
        spill = False
        if self.spill_kv and self.saturated(order[0], now):
            free = [b for b in order if not self.saturated(b, now) and b.inflight < b.cap + 8]
            if free:                                        # never queue behind a full GPU while another has room
                order = free + [b for b in order if b not in free]
                spill = True
        best = order[0]
        if too_long and fit:
            reason = "ctx"
        elif spill:
            reason = "spill"
        elif balanced:
            reason = "balance"
        elif summary_first is not None and best is summary_first:
            reason = "summary"
        elif preferred:
            reason = "prefer"
        elif aff is None and hint_aff is not None and best is hint_aff:
            reason = "hint"
        elif aff is None:
            reason = "new"
        elif best is aff:
            reason = "affinity"
        elif aff in usable:
            reason = "steal"
        else:
            reason = "affinity-down"
        order, reason = self.apply_policy(order, reason, tags, pc, usable, now)
        best = order[0]
        if aff is None and aff0 is not None and best is aff0 and reason in ("session", "parent", "group", "background"):
            aff = aff0                                   # its (stale or short) prefix is likely still there
        self._last_follow = had_aff and not weak       # continues a conversation routed before (stale or not)
        self._last_matched = matched
        self._last_total = total
        rest = [b for b in self.backends if b not in order and not b.excluded
                and (not b.max_ctx or need <= b.max_ctx or not fit) and (not b.max_images or images <= b.max_images or not fit)]   # failover never into a held+drained GPU
        self._last_uncached = max(total - cached_of(best)[0], 0)
        self._last_hint = hint
        self._last_choice = (aff, matched, total, usable)
        # decision log (no request content): chosen backend, reason, estimated finish time per backend, token counts
        dlog.info("route %s reason=%s tier=%s est=%s prompt_tok=%d cached_tok=%d gen_est=%d hint=%s inflight=%s returning=%s state=%s",
                  best.name, reason, pc["tier"] if pc else "-", ",".join(f"{k}:{v:.1f}s" for k, v in costs.items()), total,
                  matched if best is aff else 0, gen,
                  "-" if hint is None else ",".join(f"{k}:{v[0]}{'v' if v[1] else ''}" for k, v in hint.items()),
                  ",".join(f"{b.name}:{b.inflight}" for b in self.backends),
                  ",".join(f"{b.name}:{len(self._recent[b.name])}" for b in self.backends),
                  ",".join(f"{b.name}:{'up' if b.healthy else 'DOWN'}{'+drained' if b.drained else ''}"
                           for b in self.backends))
        return order + rest, reason, matched if best is aff else 0

    def dispatched(self, b: Backend, uncached: int):
        """v2: the chosen backend's prefill clock advances by this request's uncached prefill."""
        now = time.monotonic()
        b.prefill_busy_until = max(b.prefill_busy_until, now) + uncached / b.prefill

    def split_plan(self, order: list[Backend]):
        """Split mode: (prefill backend P, info) if prefilling elsewhere pays, else None. Uses choose()'s last inputs."""
        sp = self.split
        if not sp.get("on") or not self.inject_kv_ns or not order or not getattr(self, "_last_choice", None):
            return None
        aff, matched, total, usable = self._last_choice
        D = order[0]
        run = D.inflight
        if D not in usable or run < int(sp["min_running"]):
            return None
        now = time.monotonic()
        unc = lambda b: max(total - (matched if b is aff else matched * self.shared_tier), 0)
        slow = D.prefill_slowdown if D.prefill_slowdown is not None else self.prefill_slowdown
        unc_d = unc(D)
        tail = min(float(sp["tail_tokens"]), unc_d)
        t_local = max(D.prefill_busy_until - now, 0.0) + unc_d / D.prefill
        benefit = self.externality * run * slow * (unc_d - tail) / D.prefill
        best = None
        for P in usable:
            if P is D or unc(P) < float(sp["min_tokens"]) or P.inflight > int(sp.get("max_prefill_running", 99)):
                continue
            transfer = max(unc_d - tail, 0) * float(sp["kv_bytes_per_token"]) / (D.load_gbps * 1e9)
            t_split = (max(P.prefill_busy_until - now, 0.0) + unc(P) / P.prefill + float(sp["overhead_s"])
                       + float(sp["store_wait_s"]) + transfer + tail / D.prefill)
            extra = t_split - t_local
            # a busy P is allowed too (with 8 agents neither GPU is ever idle); charge the stall the
            # prefill causes P's own running streams against the stall it saves D's
            slow_p = P.prefill_slowdown if P.prefill_slowdown is not None else self.prefill_slowdown
            cost_p = self.externality * P.inflight * slow_p * unc(P) / P.prefill
            net = benefit - cost_p
            if net > float(sp["weight"]) * max(extra, 0.0) and (best is None or extra < best[1]["extra_s"]):
                best = (P, {"unc_p": int(unc(P)), "unc_d": int(unc_d), "tail": int(tail), "benefit_s": round(benefit, 2),
                            "cost_p_s": round(cost_p, 2), "p_running": P.inflight,
                            "extra_s": round(extra, 2), "t_local_s": round(t_local, 2), "t_split_s": round(t_split, 2)})
        return best

    def finished(self, b: Backend):
        self._recent[b.name].append(time.monotonic())

    def completed(self, completion_tokens: int):
        self.gen_mean = 0.9 * self.gen_mean + 0.1 * completion_tokens

    def calibrate(self, est_raw: int, prompt_tokens: int):
        """learn est_scale from the engine's real prompt token count (only prompts > 2000 estimated tokens)"""
        if est_raw < 2000 or prompt_tokens <= 0:
            return
        r = min(max(prompt_tokens / est_raw, 0.5), 2.0)
        self.est_n += 1
        w = max(0.02, 1.0 / self.est_n)                     # plain mean for the first 50, then a slow running mean
        self.est_scale = (1 - w) * self.est_scale + w * r

    config_path = None
    _cfg_mtime = 0.0

    inject_kv_ns = ""   # shared RAM tier: llmt_ns merged into forwarded kv_transfer_params ("" = off)
    trace_dir = ""      # per-request JSONL trace (order, pacing, backend; no content) in trace-YYYYMMDD.jsonl ("" = off)
    # split mode: disaggregated prefill through the shared RAM tier. When the chosen decode
    # GPU D has requests decoding and another GPU P is idle, the router first sends a prefill-only copy (max_tokens 1,
    # same cache_salt / llmt_ns) to P, whose RAM tier stores the prompt's blocks + GDN state, then the real request to
    # D, which loads them from RAM and recomputes only the tail. Taken when the stall it saves D's running streams
    # (running x prefill time x prefill_slowdown x externality) exceeds the extra delay it costs this request (P's
    # queue + prefill + RAM load into D + tail recompute + overhead). Needs inject_kv_ns (shared namespace). Off by default.
    split = {"on": False, "min_tokens": 2048, "min_running": 1, "kv_bytes_per_token": 35000, "tail_tokens": 832,
             "overhead_s": 0.3, "store_wait_s": 0.3, "weight": 1.0, "max_prefill_running": 99}
    # prefix hints (AllBroker WHERE; off by default): socket = the broker's Unix socket; engines = backend name ->
    # broker HELLO name (string) or engine id (int), default the backend name; timeout_ms per WHERE (on a miss the
    # request is routed without hints); max_anchors = message-boundary hashes asked (newest first); inject_anchor =
    # put llmt_anchor (this prompt's hash) into kv_transfer_params so the engine HINTs it when the request finishes;
    # min_tokens = RAM-resident tokens that make a forgotten conversation "not new" (also >= affinity_min_tokens and
    # affinity_min_frac x prompt); vram_ttl_s = a hint younger than this from an engine that had just served it is
    # reported as VRAM-recent (logged only)
    prefix_hints = {"on": False, "socket": "/run/llmt-kv/broker.sock", "engines": {}, "timeout_ms": 5.0,
                    "max_anchors": 64, "inject_anchor": True, "min_tokens": 4096, "vram_ttl_s": 120.0}

    def apply_config(self, cfg: dict):
        if isinstance(cfg.get("split"), dict):
            self.split = {**Router.split, **cfg["split"]}
        if "prefix_hints" in cfg:
            ph = {**Router.prefix_hints, **(cfg["prefix_hints"] or {})}
            if self.hints is not None and (not ph.get("on") or ph["socket"] != self.prefix_hints.get("socket")
                                           or float(ph["timeout_ms"]) != float(self.prefix_hints.get("timeout_ms"))):
                self.hints.close()
                self.hints = None
            self.prefix_hints = ph
            if ph.get("on") and self.hints is None:
                self.hints = BrokerHints(str(ph["socket"]), float(ph["timeout_ms"]))
        if "inject_kv_ns" in cfg:
            self.inject_kv_ns = str(cfg["inject_kv_ns"] or "")
        if "trace_dir" in cfg:
            self.trace_dir = str(cfg["trace_dir"] or "")
        for k in ("externality", "prefill_slowdown", "shared_tier", "chars_per_token", "return_window_s", "returning_weight",
                  "spill_kv", "spill_stale_s", "stall_s", "ctx_margin", "affinity_min_tokens", "affinity_min_frac", "affinity_ttl_s", "balance_skip_tokens", "min_out_tokens"):
            if k in cfg:
                setattr(self, "cpt" if k == "chars_per_token" else k, float(cfg[k]))
        if "session_tag_ids" in cfg:
            self.session_tag_ids = bool(cfg["session_tag_ids"])
        # session + priority policy and the policy file (hot-reloaded; a key left out of the file falls back to the
        # defaults / learned class)
        if "session_policy" in cfg:
            self.session_policy = dict(cfg["session_policy"] or {})
        if "policy_file" in cfg and not self.policy_override:
            self.policy_path = str(cfg["policy_file"] or "")
            self._policy_mtime = 0.0
            if not self.policy_path:
                self.policy_doc = {}
            self.load_policy()
        if "lora_sep" in cfg:
            self.lora_sep = str(cfg["lora_sep"] or "")
        if "loop_stop" in cfg:
            self.loop_stop = dict(cfg["loop_stop"] or {})
        if "summary_target" in cfg:
            self.summary_target = str(cfg["summary_target"] or "")
        if "balance_new" in cfg:
            self.balance_new = {str(k): float(v) for k, v in (cfg["balance_new"] or {}).items()}
        if "est_scale" in cfg and self.est_n == 0:
            self.est_scale = float(cfg["est_scale"])
        if "exact_band" in cfg:
            self.exact_band = tuple(float(x) for x in cfg["exact_band"])
        if "gen_default" in cfg and self.gen_mean == 400.0:
            self.gen_mean = float(cfg["gen_default"])
        for bc in cfg.get("backends", []):
            if bc.get("name") in self.by_name:
                self.by_name[bc["name"]].update(bc)

    def load_policy(self):
        """The policy file (tiers, rules, agent_priority, ...): re-read when it changes; a bad file keeps the last one."""
        if not self.policy_path:
            return
        try:
            m = os.stat(self.policy_path).st_mtime
            if m != self._policy_mtime:
                with open(self.policy_path) as f:
                    doc = json.load(f)
                if not isinstance(doc, dict):
                    raise ValueError("not an object")
                self.policy_doc = {k: v for k, v in doc.items() if not k.startswith("_")}
                if self._policy_mtime:
                    log.info("policy %s reloaded", self.policy_path)
                self._policy_mtime = m
        except (OSError, ValueError) as e:
            log.warning("policy %s not loaded: %s", self.policy_path, e)

    def maybe_reload(self):
        self.load_policy()
        if not self.config_path:
            return
        try:
            m = os.stat(self.config_path).st_mtime
            if m != self._cfg_mtime:
                cfg = json.load(open(self.config_path))
                self.apply_config(cfg)
                if self._cfg_mtime:
                    log.info("config %s reloaded", self.config_path)
                self._cfg_mtime = m
        except (OSError, ValueError) as e:
            log.warning("config %s not reloaded: %s", self.config_path, e)

    def record(self, chain, backend: Backend):
        for h, tok in chain[:-1]:
            self.map.put(h, backend.name, tok)


class StreamTracker:
    """Follow the state of a streamed response while it
    is proxied byte for byte: how far the generation is, reasoning vs content, whether a tool call is open (no handoff
    inside one), finish reason, usage. Only counts and flags are kept, never text. Chunk counts approximate tokens
    (a spec-decode step can put several tokens into one delta); `usage` at the end is exact."""

    def __init__(self, chunked: bool = False, loop_stop: dict | None = None):
        self.buf = b""
        # loop stop: a rolling window of the generated words (reasoning, content, tool arguments), in
        # memory for this stream only; never logged
        ls = loop_stop or {}
        self.ls_words = int(ls.get("words", 0) or 0)
        self.ls_ngram = int(ls.get("ngram", 6))
        self.ls_min = float(ls.get("min_distinct", 0.15))
        self.ls_every = int(ls.get("every", 200))
        self.ls_confirm = int(ls.get("confirm", 2))
        self.words: collections.deque = collections.deque(maxlen=max(1, self.ls_words))
        self.partial = ""
        self.new_words = 0
        self.low = 0
        self.loop = False
        self.loop_ratio = None
        self.nwords = 0
        self.last_meta = {}
        # loop resume: the reply so far (thinking, answer) to continue from; only when resume is on
        self.keep_text = bool(ls.get("resume")) and self.ls_words > 0
        self.rtext: list = []
        self.ctext: list = []
        self.tool_sent = False
        self.force_after = int(ls.get("test_force_after_words", 0) or 0)   # test only: treat as a loop after N words
        self.chunks = self.reasoning = self.content = self.tool_calls = self.ids = 0
        self.tool_open = False
        self.finish = None
        self.usage = None
        self.chunked = chunked          # HTTP chunked transfer coding on the proxied bytes: undo it before parsing
        self.raw = b""
        self.remaining = 0              # payload bytes left in the current HTTP chunk; -1 = waiting for a size line
        self.skip = 0                   # CRLF bytes to drop after a chunk's payload
        self.broken = False

    def _dechunk(self, data: bytes) -> bytes:
        self.raw += data
        out = []
        while self.raw and not self.broken:
            if self.skip:
                n = min(self.skip, len(self.raw))
                self.raw, self.skip = self.raw[n:], self.skip - n
                continue
            if self.remaining == 0:
                j = self.raw.find(b"\r\n")
                if j < 0:
                    if len(self.raw) > 64:
                        self.broken = True
                    break
                try:
                    size = int(self.raw[:j].split(b";")[0].strip() or b"0", 16)
                except ValueError:
                    self.broken = True
                    break
                self.raw = self.raw[j + 2:]
                if size == 0:            # last chunk (trailers ignored)
                    self.broken = True
                    break
                self.remaining = size
                continue
            n = min(self.remaining, len(self.raw))
            out.append(self.raw[:n])
            self.raw, self.remaining = self.raw[n:], self.remaining - n
            if self.remaining == 0:
                self.skip = 2
        return b"".join(out)

    def _text(self, t: str) -> None:
        if not self.ls_words or self.loop or not t:
            return
        t = self.partial + t
        parts = t.split()
        if t and not t[-1].isspace():
            self.partial = parts.pop() if parts else t
        else:
            self.partial = ""
        for w in parts:
            self.words.append(w)
        self.nwords += len(parts)
        self.new_words += len(parts)
        if self.force_after and self.nwords >= self.force_after:
            self.loop, self.loop_ratio = True, -1.0
            return
        if self.new_words >= self.ls_every and len(self.words) >= self.ls_words:
            self.new_words = 0
            w, n = list(self.words), self.ls_ngram
            g = len(w) - n + 1
            r = len({tuple(w[i:i + n]) for i in range(g)}) / g
            self.low = self.low + 1 if r < self.ls_min else 0
            if self.low >= self.ls_confirm:
                self.loop, self.loop_ratio = True, round(r, 3)

    @property
    def tokens(self) -> int:
        return self.ids or (self.reasoning + self.content)

    def safe(self) -> bool:
        """a handoff would be allowed now: not inside a tool call, not finished"""
        return not self.tool_open and self.finish is None

    def feed(self, data: bytes) -> None:
        if self.chunked:
            data = self._dechunk(data)
        self.buf += data
        while True:
            i = self.buf.find(b"\n")
            if i < 0:
                if len(self.buf) > 1 << 20:          # not SSE after all: stop buffering
                    self.buf = b""
                return
            line, self.buf = self.buf[:i].rstrip(b"\r"), self.buf[i + 1:]
            if not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                continue
            try:
                obj = json.loads(payload)
            except ValueError:
                continue
            self.chunks += 1
            self.last_meta = {k: obj[k] for k in ("id", "object", "created", "model") if k in obj}
            for ch in obj.get("choices") or ():
                d = ch.get("delta") or {}
                if self.ls_words:
                    rt = d.get("reasoning") or d.get("reasoning_content") or ""
                    ct = (d.get("content") if "delta" in ch else ch.get("text")) or ""
                    self._text(rt)
                    self._text(ct)
                    for t in d.get("tool_calls") or ():
                        self._text((t.get("function") or {}).get("arguments") or "")
                    if self.keep_text:
                        if rt:
                            self.rtext.append(rt)
                        if ct:
                            self.ctext.append(ct)
                    if d.get("tool_calls"):
                        self.tool_sent = True
                if ch.get("token_ids"):
                    self.ids += len(ch["token_ids"])
                if d.get("reasoning") or d.get("reasoning_content"):
                    self.reasoning += 1
                if (d.get("content") if "delta" in ch else ch.get("text")):
                    self.content += 1
                tc = d.get("tool_calls")
                if tc:
                    self.tool_open = True
                    self.tool_calls += sum(1 for t in tc if (t.get("function") or {}).get("name"))
                if ch.get("finish_reason"):
                    self.finish = ch["finish_reason"]
                    self.tool_open = False
            if obj.get("usage"):
                self.usage = obj["usage"]


async def read_head(reader: asyncio.StreamReader) -> tuple[str, list[tuple[str, str]]] | None:
    try:
        raw = await reader.readuntil(b"\r\n\r\n")
    except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, ConnectionError):
        return None
    lines = raw.decode("latin-1").split("\r\n")
    hdrs = []
    for ln in lines[1:]:
        if ":" in ln:
            k, v = ln.split(":", 1)
            hdrs.append((k.strip(), v.strip()))
    return lines[0], hdrs


async def read_body(reader, hdrs) -> bytes:
    h = {k.lower(): v for k, v in hdrs}
    if "chunked" in h.get("transfer-encoding", "").lower():
        out = bytearray()
        while True:
            size = int((await reader.readline()).split(b";")[0].strip() or b"0", 16)
            if size == 0:
                while (await reader.readline()) not in (b"\r\n", b""):
                    pass
                return bytes(out)
            out += await reader.readexactly(size)
            await reader.readexactly(2)
    n = int(h.get("content-length", "0") or 0)
    return await reader.readexactly(n) if n else b""


def _truthy(v) -> bool:
    return v is True or str(v).strip().lower() in ("1", "true", "yes", "on")


def _err_json(msg: str, typ: str, code: int) -> bytes:
    return json.dumps({"error": {"message": msg, "type": typ, "code": code}}).encode()


LORA_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def lora_select(req, hdrs, sep: str = ":") -> tuple[str | None, str | None]:
    """LORA-SUFFIX: the LoRA adapter a request asks for, as (adapter, error). Three spellings, all optional:
    model "<base><sep><adapter>" (e.g. "qwen3.8-27b:terse"), a body field "lora": "<adapter>", a header
    X-LLMT-LoRA: <adapter> (the last two survive gateways that rewrite or route on the model name). If they disagree the
    request is refused. On success req["model"] is set to the adapter name (vLLM serves an adapter as a model of that
    name) and the "lora" field is removed; without an adapter req is left untouched."""
    if not isinstance(req, dict):
        return None, None
    names = []
    f = req.get("lora")
    if f is not None:
        if not isinstance(f, str):
            return None, "'lora' must be a string (the adapter name)"
        if f:
            names.append(f)
    for k, v in hdrs:
        if k.lower() == "x-llmt-lora" and v.strip():
            names.append(v.strip())
    m = req.get("model")
    if sep and isinstance(m, str) and sep in m:
        base, _, suf = m.partition(sep)
        if not base:
            return None, f"model '{m}': no base model before '{sep}'"
        names.append(suf)
    if not names:
        return None, None
    if len(set(names)) > 1:
        return None, f"conflicting LoRA adapter names {sorted(set(names))}"
    a = names[0]
    if not LORA_NAME.match(a):
        return None, f"invalid LoRA adapter name '{a}' (letters, digits, '.', '_', '-'; at most 64)"
    req.pop("lora", None)
    req["model"] = a
    return a, None


def simple(status: str, body: bytes, ctype="application/json") -> bytes:
    return (f"HTTP/1.1 {status}\r\nContent-Type: {ctype}\r\nContent-Length: {len(body)}\r\nConnection: close\r\n\r\n"
            ).encode() + body


class Server:
    def __init__(self, router: Router, health_interval=2.0):
        self.split_stats = collections.Counter(ok=0, failed=0, seconds=0.0)
        # per backend, streamed responses followed and the handoff opportunity seen in them
        self.stream_stats = collections.defaultdict(lambda: {"streams": 0, "eligible": 0, "eligible_s": 0.0, "tool_calls": 0})
        self.r = router
        self.health_interval = health_interval

    def agent_view(self):
        """Per agent key (local only: agent names can be business data; they stay in local metrics / logs): requests,
        follow-ups, reuse, concurrency, the tiers its requests came with, the learned class."""
        for a, st in sorted(self.r.agents.items(), key=lambda x: -x[1].requests):
            yield a, st, st.summary(), ",".join(f"{t}:{n}" for t, n in st.tiers.most_common(4)), st.learned or "-"

    def agents_line(self) -> str:
        """One line per agent key (counts only)."""
        return "; ".join(f"{agent_label(a)}: req={st.requests} follow={100 * sm['follow']:.0f}% reuse={100 * sm['reuse']:.0f}% "
                         f"conc={sm['conc']:.2f} tiers={tiers} learned={lr}" for a, st, sm, tiers, lr in self.agent_view())

    async def health_loop(self):
        last_agents = time.monotonic()
        while True:
            self.r.maybe_reload()
            every = float(self.r.pol.get("agents_log_s", 600)) if self.r.policy_on else 0
            if every and self.r.agents and time.monotonic() - last_agents >= every:
                last_agents = time.monotonic()
                log.info("agents: %s", self.agents_line())
            await asyncio.gather(*(self.check(b) for b in self.r.backends))
            await asyncio.sleep(self.health_interval)

    async def engine_loop(self, interval=0.5):
        """Poll each engine's /metrics (num_requests_waiting/running, kv_cache_usage_perc) for the spill rule."""
        while True:
            if self.r.spill_kv or self.r.stall_s:
                await asyncio.gather(*(self.engine_poll(b) for b in self.r.backends if b.healthy))
            await asyncio.sleep(interval)

    async def engine_poll(self, b: Backend):
        try:
            rd, wr = await asyncio.wait_for(asyncio.open_connection(b.host, b.port), 1)
            wr.write(f"GET /metrics HTTP/1.1\r\nHost: {b.host}\r\nConnection: close\r\n\r\n".encode())
            await wr.drain()
            data = await asyncio.wait_for(rd.read(), 2)
            wr.close()
        except (OSError, asyncio.TimeoutError):
            return
        got = {}
        cap_wait = None     # waiting for capacity only: "deferred" = loading its prefix from the RAM tier (not a queue)
        for line in data.split(b"\n"):
            if line.startswith(b"vllm:num_requests_waiting_by_reason{") and b'reason="capacity"' in line:
                try:
                    cap_wait = (cap_wait or 0.0) + float(line.rsplit(b" ", 1)[1])
                except (ValueError, IndexError):
                    pass
                continue
            for key in (b"vllm:num_requests_waiting{", b"vllm:num_requests_running{", b"vllm:kv_cache_usage_perc{",
                        b"vllm:generation_tokens_total{", b"vllm:prompt_tokens_total{"):
                if line.startswith(key):
                    try:
                        got[key] = got.get(key, 0.0) + float(line.rsplit(b" ", 1)[1])
                    except (ValueError, IndexError):
                        pass
        tok_keys = (b"vllm:generation_tokens_total{", b"vllm:prompt_tokens_total{")
        if self.r.stall_s and all(k in got for k in tok_keys) and b"vllm:num_requests_running{" in got:
            now = time.monotonic()
            tok = got[tok_keys[0]] + got[tok_keys[1]]
            pending = got.get(b"vllm:num_requests_waiting{", 0) + got[b"vllm:num_requests_running{"]
            if tok != b.tok_total or pending == 0:
                if b.stalled:
                    log.warning("STALL-DETECT: backend %s progressing again after %.0f s", b.name, now - b.tok_t)
                b.tok_total, b.tok_t, b.stalled = tok, now, False
            elif not b.stalled and now - b.tok_t >= self.r.stall_s:
                b.stalled = True
                b.stalls += 1
                log.warning("STALL-DETECT: backend %s: %d requests pending, no token for %.0f s: routing around it",
                            b.name, pending, now - b.tok_t)
        if all(k in got for k in (b"vllm:num_requests_waiting{", b"vllm:num_requests_running{", b"vllm:kv_cache_usage_perc{")):
            b.eng_waiting = int(cap_wait if cap_wait is not None else got[b"vllm:num_requests_waiting{"])
            b.eng_running = int(got[b"vllm:num_requests_running{"])
            b.eng_kv = got[b"vllm:kv_cache_usage_perc{"]
            b.eng_t = time.monotonic()

    async def check(self, b: Backend):
        ok = False
        try:
            rd, wr = await asyncio.wait_for(asyncio.open_connection(b.host, b.port), 2)
            wr.write(f"GET /health HTTP/1.1\r\nHost: {b.host}\r\nConnection: close\r\n\r\n".encode())
            await wr.drain()
            line = await asyncio.wait_for(rd.readline(), 3)
            ok = line.split(b" ")[1:2] == [b"200"]
            wr.close()
        except (OSError, asyncio.TimeoutError, IndexError):
            ok = False
        if ok:
            if not b.healthy:
                log.info("backend %s up", b.name)
            b.healthy, b.fails = True, 0
        else:
            b.fails += 1
            if b.healthy and b.fails >= 2:
                log.warning("backend %s down", b.name)
                b.healthy = False

    def metrics(self) -> bytes:
        out = ["# TYPE llmt_router_requests_total counter"]
        for b in self.r.backends:
            for reason, n in sorted(b.n.items()):
                out.append(f'llmt_router_requests_total{{backend="{b.name}",reason="{reason}"}} {n}')
        for name, f in (("inflight", lambda b: b.inflight), ("healthy", lambda b: int(b.healthy)),
                        ("drained", lambda b: int(b.drained)), ("stalled", lambda b: int(b.stalled))):
            out.append(f"# TYPE llmt_router_{name} gauge")
            out += [f'llmt_router_{name}{{backend="{b.name}"}} {f(b)}' for b in self.r.backends]
        out.append(f"llmt_router_prefix_entries {len(self.r.map.d)}")
        if self.r.hints is not None:
            out.append("# TYPE llmt_router_hint_where_total counter")
            out += [f'llmt_router_hint_where_total{{result="{k}"}} {v}' for k, v in sorted(self.r.hints.n.items())]
            out.append("# TYPE llmt_router_hint_where_seconds_total counter")
            out.append(f"llmt_router_hint_where_seconds_total {self.r.hints.seconds:.4f}")
        out.append("# TYPE llmt_router_backend_stalls_total counter")
        out += [f'llmt_router_backend_stalls_total{{backend="{b.name}"}} {b.stalls}' for b in self.r.backends]
        out.append("# TYPE llmt_router_session_tagged_total counter")
        out.append(f"llmt_router_session_tagged_total {self.r.session_tagged}")
        # request classes and per agent string traffic (labels sanitized, at most AGENT_LABEL_MAX strings, the rest
        # "other")
        out.append("# TYPE llmt_router_class_requests_total counter")
        out += [f'llmt_router_class_requests_total{{tier="{agent_label(t)}",cls="{c}",source="{src}"}} {n}'
                for (t, c, src), n in sorted(self.r.cls_n.items())]
        out.append("# TYPE llmt_router_background_waits_total counter")
        out.append(f"llmt_router_background_waits_total {self.r.low_waits}")
        out.append("# TYPE llmt_router_background_inflight gauge")
        out += [f'llmt_router_background_inflight{{backend="{b.name}"}} {b.low_inflight}' for b in self.r.backends]
        for name, typ in (("requests_total", "counter"), ("followup_ratio", "gauge"), ("reuse_ratio", "gauge"),
                          ("concurrency", "gauge"), ("learned_low", "gauge")):
            out.append(f"# TYPE llmt_router_agent_{name} {typ}")
        out.append(f"llmt_router_sessions {len(self.r.sess_map)}")
        out.append(f"llmt_router_groups {len(self.r.group_map)}")
        for a, st, sm, tiers, lr in self.agent_view():
            lab = f'agent="{agent_label(a)}"'
            out.append(f"llmt_router_agent_requests_total{{{lab}}} {st.requests}")
            out.append(f"llmt_router_agent_followup_ratio{{{lab}}} {sm['follow']:.3f}")
            out.append(f"llmt_router_agent_reuse_ratio{{{lab}}} {sm['reuse']:.3f}")
            out.append(f"llmt_router_agent_concurrency{{{lab}}} {sm['conc']:.3f}")
            out.append(f"llmt_router_agent_learned_low{{{lab}}} {int(lr == 'low')}")
        out.append("# TYPE llmt_router_lora_requests_total counter")
        for k, v in sorted(self.r.lora_requests.items()):
            out.append(f'llmt_router_lora_requests_total{{outcome="{k}"}} {v}')
        out += ["# TYPE llmt_router_split_total counter"] + [f'llmt_router_split_total{{result="{k}"}} {self.split_stats[k]}'
                                                             for k in ("ok", "failed")]
        out += ["# TYPE llmt_router_split_prefill_seconds_total counter",
                f"llmt_router_split_prefill_seconds_total {self.split_stats['seconds']:.3f}"]
        # streamed responses followed, and how many / how long a mid-generation handoff would have been
        # possible and wanted (safe point, other GPU idle, this one >= 2 streams, >= 200 tokens left)
        for key, typ in (("streams", "counter"), ("eligible", "counter"), ("tool_calls", "counter")):
            out.append(f"# TYPE llmt_router_stream_{key}_total {typ}")
            out += [f'llmt_router_stream_{key}_total{{backend="{n}"}} {s[key]}' for n, s in sorted(self.stream_stats.items())]
        out.append("# TYPE llmt_router_stream_eligible_seconds_total counter")
        out += [f'llmt_router_stream_eligible_seconds_total{{backend="{n}"}} {s["eligible_s"]:.1f}'
                for n, s in sorted(self.stream_stats.items())]
        return ("\n".join(out) + "\n").encode()

    def state(self) -> bytes:
        return json.dumps({"backends": [{"name": b.name, "healthy": b.healthy, "drained": b.drained, "held": b.held,
                                         "inflight": b.inflight, "cap": b.cap, "requests": dict(b.n),
                                         "engine": {"waiting": b.eng_waiting, "running": b.eng_running,
                                                    "kv": round(b.eng_kv, 3),
                                                    "age_s": round(time.monotonic() - b.eng_t, 1) if b.eng_t else None}}
                                        for b in self.r.backends],
                           "prefix_entries": len(self.r.map.d), "gen_mean": round(self.r.gen_mean),
                           "agents": {agent_label(a): {"requests": st.requests, **{k: round(v, 3) for k, v in sm.items()},
                                                       "tiers": tiers, "learned": lr}
                                      for a, st, sm, tiers, lr in self.agent_view()} if self.r.policy_on else {},
                           "sessions": len(self.r.sess_map), "groups": len(self.r.group_map),
                           "uptime_s": int(time.time() - self.r.started)}).encode()

    async def handle(self, reader, writer):
        t_arr = time.time()
        try:
            head = await read_head(reader)
            if head is None:
                return
            reqline, hdrs = head
            method, target = reqline.split(" ")[:2]
            body = await read_body(reader, hdrs)
            # session + priority policy: tags and class (the raw header values never leave this function)
            tags = req_tags(hdrs) if self.r.policy_on else None
            pc = self.r.classify(tags) if tags is not None else None
            if self.r.session_tag_ids:
                tag = session_tag(hdrs)
                if tag:
                    hdrs = [(k, v) for k, v in hdrs if k.lower() != "x-request-id"] + [("X-Request-Id", tag)]
                    self.r.session_tagged += 1
            path = target.split("?")[0]
            if method == "GET" and path == "/health":
                ok = any(b.healthy for b in self.r.backends)
                writer.write(simple("200 OK" if ok else "503 Service Unavailable", b"{}"))
                return
            if path == "/router/metrics":
                writer.write(simple("200 OK", self.metrics(), "text/plain; version=0.0.4"))
                return
            if path == "/router/state":
                writer.write(simple("200 OK", self.state()))
                return
            chain, reason, matched, mt, req, hint_now = None, "other", 0, None, None, None
            adapter = None
            if method == "POST" and path in ROUTED:
                try:
                    req = json.loads(body)
                    adapter, err = lora_select(req, hdrs, self.r.lora_sep)
                    if err:
                        self.r.lora_requests["bad-name"] += 1
                        writer.write(simple("400 Bad Request", _err_json(err, "BadRequestError", 400)))
                        return
                    if adapter:
                        body = json.dumps(req, ensure_ascii=False, separators=(",", ":")).encode()
                    chain = prefix_chain(path, req, self.r.cpt)
                    mt = req.get("max_tokens") or req.get("max_completion_tokens")
                    mt = int(mt) if mt else None
                except (ValueError, AttributeError, TypeError):
                    chain = None
            if chain is not None and mt and self.r.min_out_tokens and mt < self.r.min_out_tokens:
                # OpenCode "Compaction summary reached the output token limit": OpenCode asks for 16384
                # output tokens and Qwen3.8's thinking alone can use that up (compaction summaries at 89k context hit it).
                # Raise the cap to min_out_tokens, as far as the prompt estimate leaves room in the largest max_ctx.
                lims = [b.max_ctx for b in self.r.backends if b.max_ctx]
                room = (max(lims) - chain[-1][1] * self.r.est_scale * self.r.ctx_margin) if lims else self.r.min_out_tokens
                new_mt = int(min(self.r.min_out_tokens, room))
                if new_mt > mt and self.r.min_out_tokens >= room:
                    # when the raise reaches the context limit, drop the client cap instead of computing one from the
                    # prompt estimate (a few tokens low on short prompts: 196594 + 15 > 196608 -> 400). vLLM then
                    # allows exactly max_model_len - prompt tokens.
                    for k in ("max_tokens", "max_completion_tokens"):
                        req.pop(k, None)
                    mt = int(room)
                    body = json.dumps(req, ensure_ascii=False, separators=(",", ":")).encode()
                elif new_mt > mt:
                    for k in ("max_tokens", "max_completion_tokens"):
                        if req.get(k):
                            req[k] = new_mt
                    mt = new_mt
                    body = json.dumps(req, ensure_ascii=False, separators=(",", ":")).encode()
            if chain is not None and tags is not None and isinstance(req, dict):
                # the engines' RAM tier: class and session tags (AllBroker llmt_kv vllm_tier: ENTRY_LOW stores,
                # TOUCH_TAGGED)
                ktp = req.get("kv_transfer_params")
                ktp = dict(ktp) if isinstance(ktp, dict) else {}
                ktp["llmt_cls"] = "agent" if pc["agent"] else pc["cls"]
                for k_, v_ in (("llmt_sess", tags.get("sess")), ("llmt_parent", tags.get("parent")), ("llmt_agent", tags.get("letter")),
                               ("llmt_term", 1 if pc["term"] else None)):
                    if v_:
                        ktp[k_] = v_
                req["kv_transfer_params"] = ktp
                body = json.dumps(req, ensure_ascii=False, separators=(",", ":")).encode()
            if chain is not None and self.r.inject_kv_ns:
                # shared RAM tier: put the request into the shared broker namespace; a client's own
                # kv_transfer_params are kept, and an llmt_ns it set itself wins. Instantly reversible via router.json.
                ktp = req.get("kv_transfer_params")
                ktp = dict(ktp) if isinstance(ktp, dict) else {}
                if "llmt_ns" not in ktp:
                    ktp["llmt_ns"] = self.r.inject_kv_ns
                    req["kv_transfer_params"] = ktp
                    body = json.dumps(req, ensure_ascii=False, separators=(",", ":")).encode()
            places = None
            if chain is not None and self.r.hints is not None and self.r.prefix_hints.get("on"):
                # prefix hints: ask the broker where this prompt's prefixes are (newest boundary first), and tag the
                # request with its own anchor so the engine hints it when it finishes. The anchor is the router's
                # prefix hash (no content); a client's own llmt_anchor wins.
                ph = self.r.prefix_hints
                anchors = [h for h, _ in reversed(chain[:-1]) if len(h) == 16][:int(ph.get("max_anchors", 64))]
                places = await self.r.hints.where(anchors)
                if ph.get("inject_anchor", True) and anchors and isinstance(req, dict):
                    ktp = req.get("kv_transfer_params")
                    ktp = dict(ktp) if isinstance(ktp, dict) else {}
                    if "llmt_anchor" not in ktp:
                        ktp["llmt_anchor"] = anchors[0].hex()
                        req["kv_transfer_params"] = ktp
                        body = json.dumps(req, ensure_ascii=False, separators=(",", ":")).encode()
            if chain is not None:
                exact = await self.exact_tokens(path, req, chain, mt)
                if pc is not None and pc["background"]:
                    await self.low_wait(chain, mt, exact)
                order, reason, matched = self.r.choose(chain, mt, exact, _n_images(req),
                                                       bool(req.get("tools")) if isinstance(req, dict) else True,
                                                       places=places, tags=tags, pc=pc)
                hint_now = self.r._last_hint
                if tags is not None:
                    st = self.r.observe(tags, pc, getattr(self.r, "_last_matched", 0), getattr(self.r, "_last_total", 0),
                                        getattr(self.r, "_last_follow", False))
                    self.r.cls_n[(pc["tier"], pc["cls"], pc["src"])] += 1
                plan = self.r.split_plan(order) if not _has_image(req) and not adapter else None
                if plan is not None:
                    P, info = plan
                    if await self.prefill_split(P, order[0], info, method, target, hdrs, req, chain):
                        reason = "split"
                        self.r._last_uncached = info["tail"]     # D's prefill clock: only the tail is left
            else:
                order = [b for b in self.r.backends if b.usable()] + [b for b in self.r.backends if not b.usable() and not b.excluded]
            if adapter:
                order = [b for b in order if b.lora]
                if not order:
                    self.r.lora_requests["no-backend"] += 1
                    writer.write(simple("503 Service Unavailable", _err_json(
                        f"LoRA adapter '{adapter}' requested, but no available backend serves LoRA adapters",
                        "ServiceUnavailableError", 503)))
                    return
                self.r.lora_requests["routed"] += 1
                reason = "lora-" + reason
            if tags is not None and chain is not None:
                st.inflight += 1
                try:
                    await self.forward(order, reason, chain, method, target, hdrs, body, writer, mt, req=req, t_arr=t_arr,
                                       hint=hint_now, tags=tags, pc=pc)
                finally:
                    st.inflight -= 1
            else:
                await self.forward(order, reason, chain, method, target, hdrs, body, writer, mt, req=req, t_arr=t_arr,
                                   hint=hint_now)
        except Exception as e:  # noqa: one request must never take the router down
            log.warning("request failed: %s: %s", type(e).__name__, e)
        finally:
            try:
                await writer.drain()
                writer.close()
            except (ConnectionError, RuntimeError):
                pass

    async def low_wait(self, chain, mt, exact):
        """A low-priority request waits (up to session_policy.low_wait_s, 0 = never) while every backend that fits it
        is saturated (engine queue or KV full: agent work would wait behind it), instead of taking a slot."""
        lim = float(self.r.pol.get("low_wait_s", 0))
        if lim <= 0:
            return
        t0 = time.monotonic()
        waited = False
        while time.monotonic() - t0 < lim:
            now = time.monotonic()
            ok = [b for b in self.r.backends if b.usable() and not self.r.busy(b, now)]
            if ok or not any(b.usable() for b in self.r.backends):
                break
            waited = True
            await asyncio.sleep(0.25)
        if waited:
            self.r.low_waits += 1
            dlog.info("low-wait %.1f s (every backend saturated)", time.monotonic() - t0)

    async def exact_tokens(self, path, req, chain, mt):
        """The real prompt token count from an engine's /tokenize when the estimate is near the smallest
        usable max_ctx (exact_band); None otherwise or on any failure (2 s cap; the estimate is used then)."""
        lims = [b.max_ctx for b in self.r.backends if b.max_ctx and b.usable()]
        if not lims:
            return None
        est = chain[-1][1] * self.r.est_scale + (mt or 0)
        lo, hi = self.r.exact_band
        if not (lo * min(lims) <= est <= hi * min(lims)):
            return None
        tb = next((b for b in sorted(self.r.backends, key=lambda x: -x.max_ctx) if b.usable()), None)
        if tb is None:
            return None
        if path.startswith("/v1/chat/completions"):
            q = {k: req[k] for k in ("model", "messages", "tools", "chat_template_kwargs") if k in req}
            q["add_generation_prompt"] = True
        else:
            q = {k: req[k] for k in ("model", "prompt") if k in req}
        raw = json.dumps(q, ensure_ascii=False, separators=(",", ":")).encode()
        t0 = time.monotonic()
        try:
            async def go():
                rd, wr = await asyncio.wait_for(asyncio.open_connection(tb.host, tb.port), 1)
                try:
                    wr.write(f"POST /tokenize HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
                             f"Content-Length: {len(raw)}\r\nConnection: close\r\n\r\n".encode() + raw)
                    await wr.drain()
                    head = await read_head(rd)
                    if head is None or head[0].split(" ")[1:2] != ["200"]:
                        return None
                    return int(json.loads(await read_body(rd, head[1]))["count"])
                finally:
                    wr.close()
            n = await asyncio.wait_for(go(), 2.0)
        except (OSError, asyncio.TimeoutError, ConnectionError, asyncio.IncompleteReadError, ValueError, KeyError):
            n = None
        dlog.info("exact-count %s est=%d -> %s (%.0f ms, %s)", path, chain[-1][1], n, (time.monotonic() - t0) * 1000, tb.name)
        if n:
            self.r.calibrate(chain[-1][1], n)
        return n

    async def prefill_split(self, P, D, info, method, target, hdrs, req, chain) -> bool:
        """Split mode phase 1: prefill-only copy of the request on P (max_tokens 1, not streamed); its RAM tier stores
        the prompt. True if P answered 200 (then wait store_wait_s so the stores land before D looks them up)."""
        st = self.split_stats
        body_p = dict(req)
        body_p["max_tokens"] = 1
        if "max_completion_tokens" in body_p:
            body_p["max_completion_tokens"] = 1
        for k in ("stream", "stream_options", "n", "logprobs", "top_logprobs"):
            body_p.pop(k, None)
        raw = json.dumps(body_p, ensure_ascii=False, separators=(",", ":")).encode()
        lines = [f"{method} {target} HTTP/1.1"] + [f"{k}: {v}" for k, v in hdrs if k.lower() not in HOP]
        lines += [f"Content-Length: {len(raw)}", "Connection: close", "", ""]
        P.inflight += 1
        self.r.dispatched(P, info["unc_p"])
        t0 = time.monotonic()
        ok = False
        try:
            async def go():
                rd, wr = await asyncio.wait_for(asyncio.open_connection(P.host, P.port), 3)
                try:
                    wr.write("\r\n".join(lines).encode() + raw)
                    await wr.drain()
                    head = await read_head(rd)
                    if head is None:
                        return False
                    await read_body(rd, head[1])
                    return head[0].split(" ")[1:2] == ["200"]
                finally:
                    wr.close()
            ok = await asyncio.wait_for(go(), 30 + 3 * info["unc_p"] / P.prefill)
        except (OSError, asyncio.TimeoutError, ConnectionError, asyncio.IncompleteReadError, ValueError):
            ok = False
        finally:
            P.inflight -= 1
        dt = time.monotonic() - t0
        st["ok" if ok else "failed"] += 1
        st["seconds"] += dt
        P.n["split-prefill"] += 1
        dlog.info("split prefill=%s decode=%s ok=%s %.2fs %s", P.name, D.name, ok, dt,
                  " ".join(f"{k}={v}" for k, v in info.items()))
        if ok:
            self.r.record(chain, P)          # P now holds the prefix in VRAM too (D's record after the decode wins)
            await asyncio.sleep(float(self.r.split["store_wait_s"]))
        return ok

    def trace(self, rec: dict) -> None:
        """One JSON line per routed request (pacing and order, e.g. to replay a load shape): wall-clock
        arrival, first-byte and completion offsets, backend, route reason, HTTP code, path, estimated prompt tokens and
        message count, conversation prefix hashes (first three chain links, 12 hex each: group requests of one agent),
        completion tokens, tool calls and finish reason from the stream. No prompt or completion content."""
        d = self.r.trace_dir
        if not d:
            return
        try:
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, time.strftime("trace-%Y%m%d.jsonl", time.localtime(rec["t"]))), "a") as f:
                f.write(json.dumps(rec, separators=(",", ":")) + "\n")
        except OSError as e:
            if not getattr(self, "_trace_warned", False):
                self._trace_warned = True
                log.warning("trace: cannot write to %s: %s", d, e)

    def _stream_step(self, trk: StreamTracker, b: Backend, mt, st: dict, now: float) -> None:
        """After each proxied chunk, note whether a handoff of this stream would be possible AND wanted now:
        a safe point (no open tool call), another backend healthy and idle while this one runs >= 2 streams, and
        enough output left (max_tokens - generated >= 200, or no cap). Sums the time spent in that state."""
        want = (trk.safe() and b.inflight >= 2 and (mt is None or mt - trk.tokens >= 200)
                and any(o is not b and o.usable() and o.inflight == 0 for o in self.r.backends))
        if want:
            if st["eligible_since"] is None:
                st["eligible_since"] = now
                st["eligible_chunks"] = 0
            st["eligible_chunks"] += 1
            st["eligible_s"] += now - st["last_t"]
        else:
            st["eligible_since"] = None
        st["last_t"] = now

    def _stream_done(self, trk: StreamTracker, b: Backend, mt, st: dict, code: str) -> None:
        s = self.stream_stats[b.name]
        s["streams"] += 1
        s["tool_calls"] += trk.tool_calls
        if st["eligible_s"] > 0:
            s["eligible"] += 1
            s["eligible_s"] += st["eligible_s"]
        u = trk.usage or {}
        dlog.info("stream backend=%s code=%s chunks=%d tokens~%d reasoning~%d content~%d tool_calls=%d finish=%s "
                  "completion_tokens=%s max_tokens=%s eligible_s=%.1f elapsed_s=%.1f", b.name, code, trk.chunks, trk.tokens,
                  trk.reasoning, trk.content, trk.tool_calls, trk.finish, u.get("completion_tokens"), mt,
                  st["eligible_s"], time.monotonic() - st["t0"])

    async def _resume(self, b, target, hdrs, body, trk, writer, chunked) -> bool:
        """Loop resume (in a replay, 56/56 continuations of sampled loops at T 1.3 ended normally): continue the same
        turn on the same backend from exactly what the client already has, at loop_stop.resume_temperature, and relay it into the client's stream
        (original id; role / stray whitespace-thinking deltas dropped). A second loop ends the stream (finish length).
        False = not resumed (nothing written; the caller ends the stream as before)."""
        ls = self.r.loop_stop
        try:
            j = json.loads(body)
        except ValueError:
            return False
        msgs = j.get("messages")
        if not isinstance(msgs, list) or not msgs or j.get("continue_final_message"):
            return False
        rt, ct = "".join(trk.rtext), "".join(trk.ctext)
        if ct.strip():                                   # loop in the answer: the turn as generated, thinking closed
            msg = {"role": "assistant", "content": ct}
            if rt:
                msg["reasoning"] = rt
            answer_mode = True
        else:                                            # loop in the thinking: keep it open
            msg = {"role": "assistant", "content": "<think>\n" + rt}
            answer_mode = False
        j["messages"] = msgs + [msg]
        j["continue_final_message"], j["add_generation_prompt"], j["stream"] = True, False, True
        j["temperature"] = float(ls.get("resume_temperature", 1.3))
        for k in ("max_tokens", "max_completion_tokens"):
            if j.get(k):
                j[k] = max(256, int(j[k]) - trk.tokens)
        nb = json.dumps(j, ensure_ascii=False, separators=(",", ":")).encode()
        rq = [f"POST {target} HTTP/1.1"] + [f"{k}: {v}" for k, v in hdrs if k.lower() not in HOP]
        rq += [f"Content-Length: {len(nb)}", "Connection: close", "", ""]
        try:
            rd, wr = await asyncio.wait_for(asyncio.open_connection(b.host, b.port), 3)
        except (OSError, asyncio.TimeoutError):
            return False
        b.inflight += 1
        t0 = time.monotonic()
        try:
            wr.write("\r\n".join(rq).encode() + nb)
            await wr.drain()
            head = await read_head(rd)
            if head is None or head[0].split(" ")[1:2] != ["200"]:
                dlog.info("loop-resume %s: continuation refused (%s); ending the stream", b.name, head[0] if head else "closed")
                return False
            ch2 = any(k.lower() == "transfer-encoding" and "chunked" in v.lower() for k, v in head[1])
            dech = StreamTracker(ch2)                    # framing only
            trk2 = StreamTracker(False, {k: v for k, v in ls.items() if k not in ("resume", "test_force_after_words")})
            meta = {k: v for k, v in trk.last_meta.items() if k in ("id", "created", "model")}
            buf, fin, stopped = b"", None, False

            def out(payload: bytes):
                writer.write(b"%x\r\n%s\r\n" % (len(payload), payload) if chunked else payload)

            while True:
                chunk = await rd.read(65536)
                if not chunk:
                    break
                buf += dech._dechunk(chunk) if ch2 else chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.rstrip(b"\r")
                    if not line.startswith(b"data:"):
                        continue
                    pl = line[5:].strip()
                    if pl == b"[DONE]":
                        out(b"data: [DONE]\n\n")
                        continue
                    try:
                        obj = json.loads(pl)
                    except ValueError:
                        continue
                    obj.update(meta)
                    keep = []
                    for c in obj.get("choices") or ():
                        d = c.get("delta") or {}
                        d.pop("role", None)
                        if answer_mode:
                            for rk in ("reasoning", "reasoning_content"):
                                if rk in d and not (d[rk] or "").strip():
                                    d.pop(rk)
                        if c.get("finish_reason"):
                            fin = c["finish_reason"]
                        if any(v not in (None, "") for v in d.values()) or c.get("finish_reason") or c.get("logprobs"):
                            keep.append(c)
                    if not keep and not obj.get("usage"):
                        continue
                    obj["choices"] = keep
                    ev = b"data: " + json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode() + b"\n\n"
                    trk2.feed(ev)
                    out(ev)
                    if trk2.loop and fin is None:        # looped again: end it
                        ev = {**trk.last_meta, "choices": [{"index": 0, "delta": {}, "finish_reason": "length"}]}
                        out(b"data: " + json.dumps(ev, separators=(",", ":")).encode() + b"\n\ndata: [DONE]\n\n")
                        stopped = True
                        break
                await writer.drain()
                if stopped:
                    break
            if chunked:
                writer.write(b"0\r\n\r\n")
            await writer.drain()
            trk.finish = "loop-again" if stopped else f"loop-resumed:{fin}"
            if trk2.usage:
                trk.usage = trk2.usage
            b.n["loop-resume"] += 1
            dlog.info("loop-resume %s: %s loop after %d words (distinct %s); continuation at T %.2f: %d chunks, finish %s, %.1f s",
                      b.name, "answer" if answer_mode else "thinking", trk.nwords, trk.loop_ratio, j["temperature"],
                      trk2.chunks, "length (looped again)" if stopped else fin, time.monotonic() - t0)
            return True
        except (ConnectionError, asyncio.IncompleteReadError, OSError):
            return True                                  # client or backend went away mid-continuation
        finally:
            b.inflight -= 1
            wr.close()

    async def forward(self, order, reason, chain, method, target, hdrs, body, writer, mt=None, req=None, t_arr=None,
                      hint=None, tags=None, pc=None):
        body_req, t_arr = req, t_arr or time.time()
        rec = None
        if chain is not None and self.r.trace_dir:
            msgs = body_req.get("messages") if isinstance(body_req, dict) else None
            rec = {"t": round(t_arr, 3), "path": target.split("?")[0], "reason": reason, "prompt_est": chain[-1][1],
                   **({"hint": {k: v[0] for k, v in hint.items()}} if hint else {}),
                   "msgs": len(msgs) if isinstance(msgs, list) else None, "chain": len(chain),
                   "conv": [c[0].hex()[:12] for c in chain[:3] if c[0]], "max_tokens": mt,
                   "stream": bool(isinstance(body_req, dict) and body_req.get("stream"))}
        if rec is not None and tags is not None:
            rec["tier"], rec["cls"] = pc["tier"], pc["cls"]   # tier + class only (no session ids, no agent name)

        def frame(bd: bytes) -> bytes:
            rq = [f"{method} {target} HTTP/1.1"] + [f"{k}: {v}" for k, v in hdrs if k.lower() not in HOP]
            rq += [f"Content-Length: {len(bd)}", "Connection: close", "", ""]
            return "\r\n".join(rq).encode() + bd
        req_bytes = frame(body)
        prio_bytes = None
        low = pc is not None and pc["background"]
        vp = int(pc["spec"].get("vllm_priority", 0)) if pc is not None else 0
        if vp and isinstance(body_req, dict) and any(b.priority_sched for b in order):
            # vLLM priority scheduling (lower = earlier): the tier's priority, only to engines that run
            # --scheduling-policy priority
            pb = dict(body_req)
            pb["priority"] = vp
            prio_bytes = frame(json.dumps(pb, ensure_ascii=False, separators=(",", ":")).encode())
        for i, b in enumerate(order):
            req_bytes_b = prio_bytes if (prio_bytes is not None and b.priority_sched) else req_bytes
            b.inflight += 1                          # count it before the first await: concurrent choices see it
            if low:
                b.low_inflight += 1
            try:
                rd, wr = await asyncio.wait_for(asyncio.open_connection(b.host, b.port), 3)
            except (OSError, asyncio.TimeoutError):
                b.inflight -= 1
                if low:
                    b.low_inflight -= 1
                b.fails += 1
                if b.inflight <= 0:
                    b.prefill_busy_until = min(b.prefill_busy_until, time.monotonic())
                continue
            b.n[reason if i == 0 else "failover"] += 1
            if chain is not None:
                self.r.dispatched(b, getattr(self.r, "_last_uncached", 0) if i == 0 else chain[-1][1])
            sent_any = False
            tail = b""
            try:
                wr.write(req_bytes_b)
                await wr.drain()
                head = await read_head(rd)
                if head is None:                     # backend closed before answering: try the next one
                    continue
                status, rhdrs = head
                code = status.split(" ")[1] if " " in status else "502"
                # (tool schemas were once missing from the prompt estimate: context-length 400s at a 124.8k
                # max_model_len): a context-length 400 is retried on a later backend with a larger max_ctx
                if code == "400" and any((b.max_ctx and x.max_ctx > b.max_ctx) or (b.max_images and x.max_images > b.max_images)
                                         for x in order[i + 1:]):
                    try:
                        rb = await asyncio.wait_for(read_body(rd, rhdrs), 10)
                    except (asyncio.TimeoutError, asyncio.IncompleteReadError, ConnectionError, ValueError):
                        rb = b""
                    if re.search(rb"maximum context length|max_model_len|context length|too long|image\(s\) may be provided", rb, re.I):
                        b.n["ctx-retry"] += 1
                        dlog.info("ctx-retry %s (max_ctx %d): 400 context length -> next backend with a larger max_ctx",
                                  b.name, b.max_ctx)
                        continue
                    out = [status] + [f"{k}: {v}" for k, v in rhdrs
                                      if k.lower() not in ("connection", "keep-alive", "transfer-encoding", "content-length")]
                    out += [f"Content-Length: {len(rb)}", "Connection: close", "", ""]
                    writer.write("\r\n".join(out).encode() + rb)
                    if rec is not None:
                        now = time.time()
                        rec.update(backend=b.name, code=code, first_s=round(now - t_arr, 3), done_s=round(now - t_arr, 3),
                                   completion_tokens=None, failover=i, tool_calls=None, finish=None)
                        self.trace(rec)
                    return
                out = [status] + [f"{k}: {v}" for k, v in rhdrs if k.lower() not in ("connection", "keep-alive")]
                out += ["Connection: close", "", ""]
                writer.write("\r\n".join(out).encode())
                sent_any = True
                t_first = None
                # follow streamed responses (bytes still pass through unchanged, chunked framing included;
                # the tracker undoes the framing for its own parsing)
                ctype = next((v for k, v in rhdrs if k.lower() == "content-type"), "")
                chunked = any(k.lower() == "transfer-encoding" and "chunked" in v.lower() for k, v in rhdrs)
                trk = (StreamTracker(chunked, self.r.loop_stop) if chain is not None and "text/event-stream" in ctype
                       else None)
                loop_stopped = False
                tst = {"t0": time.monotonic(), "last_t": time.monotonic(), "eligible_since": None, "eligible_s": 0.0,
                       "eligible_chunks": 0}
                while True:
                    chunk = await rd.read(65536)
                    if not chunk:
                        break
                    if t_first is None:
                        t_first = time.time()
                    writer.write(chunk)
                    tail = (tail + chunk)[-8192:]
                    if trk is not None:
                        trk.feed(chunk)
                        self._stream_step(trk, b, mt, tst, time.monotonic())
                        if trk.loop and trk.finish is None and not trk.buf and (not chunked or (trk.remaining == 0 and not trk.raw)):
                            if trk.keep_text and not trk.tool_sent:
                                wr.close()                       # abort the looping request before the continuation
                                done = await self._resume(b, target, hdrs, body, trk, writer, chunked)
                                if done:
                                    loop_stopped = True
                                    break
                            # end the client's stream at an event boundary; closing the backend connection (finally)
                            # makes vLLM abort the request
                            ev = {**trk.last_meta, "choices": [{"index": 0, "delta": {}, "finish_reason": "length"}]}
                            payload = (b"data: " + json.dumps(ev, separators=(",", ":")).encode() + b"\n\ndata: [DONE]\n\n")
                            if chunked:
                                payload = b"%x\r\n%s\r\n0\r\n\r\n" % (len(payload), payload)
                            writer.write(payload)
                            trk.finish = "loop"
                            loop_stopped = True
                            b.n["loop-stop"] += 1
                            dlog.info("loop-stop %s: generated text repeats (distinct %d-word phrases %.3f over the last %d "
                                      "words, after %d words, ~%d chunks): stream ended, request aborted", b.name,
                                      trk.ls_ngram, trk.loop_ratio, trk.ls_words, trk.nwords, trk.chunks)
                            await writer.drain()
                            break
                    await writer.drain()             # raises if the client went away -> backend aborts on close
                if trk is not None:
                    self._stream_done(trk, b, mt, tst, code)
                m = re.findall(rb'"completion_tokens":\s*(\d+)', tail)
                mp = re.findall(rb'"prompt_tokens":\s*(\d+)', tail)
                if code == "200" and chain is not None:
                    self.r.record(chain, b)
                    if tags is not None and pc is not None:
                        self.r.note_session(tags.get("sess"), b, pc["group"])
                    self.r.finished(b)
                    if m:
                        self.r.completed(int(m[-1]))
                    if mp:
                        self.r.calibrate(chain[-1][1], int(mp[-1]))
                if rec is not None:
                    now = time.time()
                    rec.update(backend=b.name, code=code, first_s=round((t_first or now) - t_arr, 3), done_s=round(now - t_arr, 3),
                               completion_tokens=int(m[-1]) if m else None, failover=i,
                               prompt_tokens=int(mp[-1]) if mp else None, est_scale=round(self.r.est_scale, 3),
                               tool_calls=trk.tool_calls if trk is not None else None, finish=trk.finish if trk is not None else None)
                    self.trace(rec)
                return
            except (ConnectionError, asyncio.IncompleteReadError):
                if sent_any:
                    return
            finally:
                b.inflight -= 1
                if low:
                    b.low_inflight -= 1
                # an idle GPU is not busy prefilling. Failed / aborted dispatches (e.g. retries against a
                # crashing engine) advanced the clock by their whole prompt and froze the Arc out for minutes (est 407 s)
                if b.inflight <= 0 and b.prefill_busy_until > time.monotonic():
                    b.prefill_busy_until = time.monotonic()
                wr.close()
        if rec is not None:
            rec.update(backend=None, code="502", first_s=None, done_s=round(time.time() - t_arr, 3), completion_tokens=None, failover=None, tool_calls=None, finish=None)
            self.trace(rec)
        writer.write(simple("502 Bad Gateway", b'{"error":"no backend available"}'))


async def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--listen", default="127.0.0.1:8010")
    ap.add_argument("--backend", action="append", help="name=..,url=..[,cap=..,prefill=..,dec=a/b/c] (or use --config)")
    ap.add_argument("--config", help="JSON file with listen/backends/tunables (router.json); re-read when it changes")
    ap.add_argument("--max-entries", type=int, default=200_000)
    ap.add_argument("--shared-tier", type=float, default=0.0,
                    help="fraction of an affinity prefix the OTHER backend can load from a shared RAM tier (0 = none)")
    ap.add_argument("--chars-per-token", type=float, default=3.3)
    ap.add_argument("--decision-log", default="/var/log/llmt-router/decisions.log",
                    help="rotating file for the per-request decision lines (no request content); '' = stderr only")
    ap.add_argument("--policy-file", default=os.environ.get("LLMT_ROUTER_POLICY", ""),
                    help="session + priority policy file (see router-policy.example.json; env LLMT_ROUTER_POLICY); "
                         "overrides router.json policy_file; the policy itself is switched on by session_policy.on")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if a.decision_log:
        try:
            os.makedirs(os.path.dirname(a.decision_log), exist_ok=True)
            h = logging.handlers.RotatingFileHandler(a.decision_log, maxBytes=20 << 20, backupCount=5)
            h.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
            dlog.addHandler(h)
        except OSError as e:
            log.warning("decision log %s not writable: %s", a.decision_log, e)
    if not a.config:                                     # default: router.json next to this script, if present
        d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "router.json")
        a.config = d if os.path.exists(d) else None
    cfg = json.load(open(a.config)) if a.config else {}
    specs = cfg.get("backends") or a.backend
    if not specs:
        ap.error("no backends: give --config or --backend")
    if cfg.get("listen"):
        a.listen = cfg["listen"]
    r = Router([Backend(s) for s in specs], a.max_entries, a.shared_tier, a.chars_per_token)
    if a.policy_file:
        r.policy_path, r.policy_override = a.policy_file, True
        r.load_policy()
    if a.config:
        r.config_path = a.config
        r.maybe_reload()
    srv = Server(r)
    host, port = a.listen.rsplit(":", 1)
    server = await asyncio.start_server(srv.handle, host, int(port), limit=64 << 20)
    asyncio.create_task(srv.health_loop())
    asyncio.create_task(srv.engine_loop())
    # session placements survive router restarts (only entries younger than the affinity TTL matter)
    map_path = os.path.join(r.trace_dir or "/var/lib/llmt-router", "prefix-map.txt")
    keep = r.affinity_ttl_s or 3600.0
    n = r.map.load(map_path, keep)
    if n:
        log.info("prefix map: %d placements loaded from %s", n, map_path)

    async def map_saver():
        while True:
            await asyncio.sleep(30)
            try:
                r.map.save(map_path, r.affinity_ttl_s or 3600.0)
            except OSError as e:
                log.warning("prefix map: save failed: %s", e)
    asyncio.create_task(map_saver())
    log.info("listening on %s, backends %s", a.listen, ", ".join(f"{b.name}={b.host}:{b.port}" for b in r.backends))
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
