"""Session + priority policy (router.json session_policy + policy_file): tagged, untagged, mixed,
background tiers, parent/child, group placement, retention classes, agent-name map and learned classes,
kv_transfer_params / vLLM priority downstream. Fictional agent names only. No network beyond 127.0.0.1, no GPU. Run: python3 router/tests/test_sessprio.py
SESSPRIO_REF=<git ref> picks the pre-policy router for check 1 (default: the commit before the policy was added);
without git or that ref, check 1 compares untagged traffic with the policy on against the policy off instead."""
import importlib.util, tempfile, asyncio, json, os, re, subprocess, time, logging
HERE = os.path.dirname(os.path.abspath(__file__))
ROUTER = os.path.join(HERE, "..", "llmt-router.py")
sp = importlib.util.spec_from_file_location("r", ROUTER); r = importlib.util.module_from_spec(sp); sp.loader.exec_module(r)
r.DRAIN_DIR = tempfile.mkdtemp()
n_ok = 0


def ok(msg):
    global n_ok
    n_ok += 1
    print("ok  ", msg)


def mk(policy=None, agent_priority=None):
    a = r.Backend({"name": "arc", "url": "http://127.0.0.1:1", "cap": 16, "prefill": 2800, "dec": [83, 74, 63, 57]})
    g = r.Backend({"name": "rtx5090", "url": "http://127.0.0.1:2", "cap": 32, "prefill": 6500, "dec": [383, 317, 264, 212],
                   "prefer_until": 2})
    for b in (a, g):
        b.healthy = True
    R = r.Router([a, g], shared_tier=0.9)
    cfg = {"balance_new": {"rtx5090": 2, "arc": 1}, "affinity_min_tokens": 32768, "affinity_min_frac": 0.5,
           "affinity_ttl_s": 600, "spill_kv": 0.9}
    if policy is not None:
        cfg["session_policy"] = policy
    if agent_priority is not None:
        cfg["agent_priority"] = agent_priority
    R.apply_config(cfg)
    return R, a, g


def chain(conv: str, n_tok: int, parts: int = 4):
    """A prompt of n_tok estimated tokens in `parts` messages; the same conv + more parts = a follow-up turn."""
    body = {"model": "m", "messages": [{"role": "user", "content": f"{conv}-{i}-" + "x" * int(n_tok * 3.3 / parts)}
                                       for i in range(parts)]}
    return r.prefix_chain("/v1/chat/completions", body, 3.3)


def turn(conv: str, i: int, per_msg: int = 8000):
    """Turn i of conversation conv: the same earlier messages + one more (a real follow-up: prefix match)."""
    body = {"model": "m", "messages": [{"role": "user", "content": f"{conv}-{j}-" + "x" * int(per_msg * 3.3)} for j in range(i + 2)]}
    return r.prefix_chain("/v1/chat/completions", body, 3.3)


def tags(sess=None, parent=None, agent=None, prio=None, name=None):
    h = []
    if sess: h.append(("X-Session-Id", sess))
    if parent: h.append(("X-Parent-Session-Id", parent))
    if agent: h.append(("X-Agent", agent))
    if name: h.append(("X-Agent-Name", name))
    if prio: h.append(("X-Priority", prio))
    return r.req_tags(h)


def route(R, ch, t=None, b_done=True):
    """choose + record (what forward does on a 200), returns (backend, reason)."""
    pc = R.classify(t) if t is not None and R.policy_on else None
    order, reason, _ = R.choose(ch, tags=t if R.policy_on else None, pc=pc)
    if R.policy_on and t is not None:
        R.observe(t, pc, R._last_matched, R._last_total, R._last_follow)
    if b_done:
        R.record(ch, order[0])
        if t is not None and pc is not None:
            R.note_session(t.get("sess"), order[0], pc["group"])
    return order[0], reason


# ---- 1. policy off: identical to the pre-policy router (same order and reason for the same inputs)
def _reference():
    ref = os.environ.get("SESSPRIO_REF", "012d5cc")
    try:
        txt = subprocess.run(["git", "-C", HERE, "show", f"{ref}:router/llmt-router.py"], capture_output=True, check=True,
                             timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    f = tempfile.NamedTemporaryFile("wb", suffix=".py", delete=False); f.write(txt); f.close()
    s0 = importlib.util.spec_from_file_location("p", f.name); m0 = importlib.util.module_from_spec(s0); s0.loader.exec_module(m0)
    return m0


p0 = _reference()
REF_NAME = "the pre-policy router"
if p0 is None:          # no git checkout: untagged traffic with the policy on must route like the policy off
    p0, REF_NAME = r, "the policy-off router (untagged requests, policy on)"
p0.DRAIN_DIR = r.DRAIN_DIR
R, a, g = mk() if p0 is not r else mk({"on": True})
P = p0.Router([p0.Backend({"name": "arc", "url": "http://127.0.0.1:1", "cap": 16, "prefill": 2800, "dec": [83, 74, 63, 57]}),
               p0.Backend({"name": "rtx5090", "url": "http://127.0.0.1:2", "cap": 32, "prefill": 6500, "dec": [383, 317, 264, 212],
                           "prefer_until": 2})], shared_tier=0.9)
for b in P.backends:
    b.healthy = True
P.apply_config({"balance_new": {"rtx5090": 2, "arc": 1}, "affinity_min_tokens": 32768, "affinity_min_frac": 0.5,
                "affinity_ttl_s": 600, "spill_kv": 0.9})
seq = []
for i in range(12):
    ch = chain(f"c{i % 5}", 20000 + 9000 * (i // 5), 2 + i // 5)
    o1, r1, _ = R.choose(ch); o2, r2, _ = P.choose(ch)
    seq.append((o1[0].name, r1) == (o2[0].name, r2))
    R.record(ch, o1[0]); P.record(ch, o2[0])
    o1[0].inflight += 1; P.by_name[o2[0].name].inflight += 1
assert all(seq), seq
ok(f"policy off: same placement and reason as {REF_NAME} on 12 mixed requests")

# ---- 2. tagged session: sticks to its GPU even when the prefix match is too short for affinity
R, a, g = mk({"on": True})
t1 = tags("ses-A", agent="build")
b1, why = route(R, chain("A", 6000, 2), t1)
g.inflight += 1                                # busy 5090: a new conversation would be balanced elsewhere
b2, why2 = route(R, chain("A", 6500, 3), t1)
assert b2 is b1 and why2 == "session", (b1.name, b2.name, why2)
ok(f"session: second request of ses-A back on {b1.name} (reason session, prefix only 6.5k tokens)")
t2 = tags("ses-B", agent="build")
bB, _ = route(R, chain("B", 6000, 2), t2)
ok(f"another session placed by load -> {bB.name}")

# ---- 3. session GPU saturated: work-conserving spill wins
R, a, g = mk({"on": True})
tS = tags("ses-S")
bS, _ = route(R, chain("S", 30000, 2), tS)
bS.eng_waiting, bS.eng_t = 2, time.monotonic()
b, why = route(R, chain("S", 31000, 3), tS)
assert b is not bS and why == "spill", (b.name, why)
ok(f"session GPU {bS.name} saturated -> spill to {b.name}")

# ---- 4. parent / child: a subagent's first request goes to its parent's GPU, unless that GPU is full
R, a, g = mk({"on": True})
tp = tags("ses-P", agent="build")
R.note_session(tp["sess"], a)                     # the parent runs on the Arc
tc = tags("ses-C1", parent="ses-P", agent="explore")
b, why = route(R, chain("C1", 26000, 2), tc)
assert b is a and why == "parent", (b.name, why)
ok("subagent first request -> parent's GPU (arc, reason parent)")
b, why = route(R, chain("C1", 27000, 3), tc)
assert b is a and why in ("session", "affinity"), (b.name, why)
ok(f"subagent's next request: its own session (reason {why})")
a.inflight = a.cap
b, why = route(R, chain("C2", 26000, 2), tags("ses-C2", parent="ses-P"))
assert b is not a and why != "parent", (b.name, why)
ok("parent's GPU at cap -> subagent placed by load")

# ---- 5. low priority: never takes the GPU with agent work while another has less
R, a, g = mk({"on": True})
g.inflight = 1                                   # an agent decoding on the 5090 (prefer_until would send it there)
b, why = route(R, chain("L", 24000, 2), tags(prio="low"))
assert b is a and why == "background", (b.name, why)
ok("X-Priority: low (a background tier) with an agent on the 5090 -> arc (prefer_until bypassed)")
a.inflight, a.low_inflight = 1, 1                # the Arc runs a low request, the 5090 an agent
b, why = route(R, chain("L2", 24000, 2), tags(prio="low"))
assert b is a, b.name
ok("second low request: still the GPU without agent work (low work does not count as agent load)")
R, a, g = mk({"on": True, "low_target": "rtx5090"})
b, why = route(R, chain("L3", 24000, 2), tags(prio="low"))
assert b is g and why == "background", (b.name, why)
ok("idle box: low_target (rtx5090) wins the tie")
R._recent["rtx5090"].append(time.monotonic())    # an agent between tool calls on the 5090
b, why = route(R, chain("L4", 24000, 2), tags(prio="low"))
assert b is a, b.name
ok("returning agent on the 5090 counts as agent load -> low goes to arc")
for x in (a, g):
    x.eng_waiting, x.eng_t = 1, time.monotonic()
b, why = route(R, chain("L5", 24000, 2), tags(prio="low"))
ok(f"every GPU saturated: low placed by cost ({b.name}, reason {why}); the router-side low_wait_s holds it first")

# ---- 6. tiers: X-Priority picks the tier; unknown tier -> default; agent_priority by name; learned
EX = json.load(open(os.path.join(HERE, "..", "router-policy.example.json")))
R, a, g = mk({"on": True})
assert R.classify(tags(prio="low"))["tier"] == "low" and R.classify(tags(prio="low"))["background"]
assert R.classify(tags(prio="weird"))["tier"] == "normal" and R.classify(tags(prio="weird"))["src"] == "header-unknown"
assert R.classify(tags(name="assistant"))["tier"] == "normal"
R.policy_doc = {**EX, "agent_priority": {"Sorter": "background"}}
c = R.classify(tags("s1", name="sorter"))
assert (c["tier"], c["src"], c["cls"]) == ("background", "map", "low"), c
c = R.classify(tags("s1", name="sorter", prio="interactive"))
assert (c["tier"], c["src"]) == ("interactive", "header"), c
ok("tiers: X-Priority wins, unknown tier -> default, agent_priority by name (case-insensitive)")

# ---- 7. rules: retention classes, terminal roles, group / parent placement (example policy, fictional runners)
R, a, g = mk({"on": True})
R.policy_doc = dict(EX)
c = R.classify(tags("s-a", agent="main", name="assistant", prio="interactive"))
assert (c["cls"], c["agent"], c["term"], c["group"]) == ("normal", True, False, None), c
c = R.classify(tags("s-a", agent="main", name="assistant", prio="background"))
assert (c["cls"], c["background"]) == ("low", True), c
c = R.classify(tags("s-h", parent="s-a", agent="summary", name="helper-maps", prio="interactive"))
assert (c["cls"], c["term"], c["group"]) == ("normal", True, "helper-maps"), c
c = R.classify(tags("s-n", agent="main", name="nightly:tidy", prio="background"))
assert (c["cls"], c["group"]) == ("lowest", "nightly:tidy"), c
c = R.classify(tags("s-x", agent="main", name="one-off-thing", prio="interactive"))
assert c["cls"] == "low" and c["rule"] == EX["default_rule"], c
c = R.classify(tags("s-c", parent="s-a", agent="compaction", prio="interactive"))
assert c["cls"] == "low" and c["rule"]["roles"] == ["compaction"], c
assert R.classify(tags())["rule"] is None and R.classify(tags())["cls"] == "normal"
ok("retention: session normal, background tier low, terminal role, group key, lowest batch, default one-shot, role rule")
# group placement: a second session of the same helper goes to the replica holding the helper's prefix
R.note_session(tags("s-a")["sess"], g)
b1, w1 = route(R, chain("H1", 26000, 2), tags("s-h1", parent="s-a", agent="main", name="helper-maps", prio="interactive"))
a.inflight = 0; g.inflight = 3                      # the 5090 is busier now
b2, w2 = route(R, chain("H2", 26000, 2), tags("s-h2", parent="s-zz", agent="main", name="helper-maps", prio="interactive"))
assert b2 is b1 and w2 == "group", (b1.name, b2.name, w2)
ok(f"group: the second helper-maps session joins the first on {b1.name} (shared specialist prefix)")
# compaction (role rule, place parent): to the parent conversation's replica, low class (never pushes the parent out)
R.note_session(tags("s-p")["sess"], a)
b, why = route(R, chain("CP", 40000, 2), tags("s-p2", parent="s-p", agent="compaction", prio="interactive"))
assert b is a and why == "parent", (b.name, why)
ok("compaction: on the parent's replica (reason parent), stores low")
# re-run at a background tier: same session -> its replica (prefix), background priority
R, a, g = mk({"on": True})
R.policy_doc = dict(EX)
R.note_session(tags("s-r")["sess"], g)
g.inflight = 1
b, why = route(R, chain("R", 30000, 2), tags("s-r", agent="main", name="assistant", prio="background"))
assert b is g and why == "session", (b.name, why)
ok("re-run at a background tier: the session's replica (its prefix), background tier for the engine")
# a batch of one job goes together
R, a, g = mk({"on": True})
R.policy_doc = dict(EX)
seen_b = set()
for i in range(6):
    b, why = route(R, chain(f"J{i}", 9000, 2), tags(f"s-j{i}", agent="main", name="nightly:tidy", prio="background"))
    seen_b.add(b.name)
    b.inflight += 1
assert len(seen_b) == 1, seen_b
ok(f"batch items of one job placed together ({seen_b.pop()})")

# ---- 7c. a typical agent traffic shape (fictional names): a root conversation (2 turns, one session), one delegated
# helper child (5 rounds + its summary, one session), 9 readouts each in its own fresh session with parent = the root;
# every request at the top tier
R, a, g = mk({"on": True})
R.policy_doc = dict(EX)
top = "interactive"
root = tags("s-root", agent="main", name="assistant", prio=top)
b_root, _ = route(R, turn("root", 0), root)
b_root2, w = route(R, turn("root", 1), root)
assert b_root2 is b_root and w in ("session", "affinity"), (b_root2.name, w)
hb = None
for i in range(5):
    t = tags("s-help", parent="s-root", agent="main", name="helper-maps", prio=top)
    b, w = route(R, turn("help", i), t)
    assert R.classify(t)["cls"] == "normal" and not R.classify(t)["term"]
    hb = hb or b
    assert b is hb, (i, b.name, hb.name)
ts = tags("s-help", parent="s-root", agent="summary", name="helper-maps", prio=top)
b, w = route(R, turn("help", 5), ts)
assert b is hb and R.classify(ts)["term"], (b.name, w)
reasons = []
for i in range(9):
    t = tags(f"s-read{i}", parent="s-root", agent="main", name="readout-x", prio=top)
    c = R.classify(t)
    assert c["cls"] == "low" and not c["background"] and c["spec"]["vllm_priority"] == 0, c
    b, w = route(R, chain(f"read{i}", 30000, 2), t)
    assert b is b_root and w in ("parent", "affinity"), (i, b.name, w)
    reasons.append(w)
assert all(not R.classify(tags(f"s{i}", prio=top))["background"] for i in range(3))
assert R.low_waits == 0 and all(x.low_inflight == 0 for x in (a, g))
ok(f"observed shape: root sticks, helper stays on {hb.name} through its terminal summary, 9 fresh-session readouts "
   f"go to the root's GPU ({set(reasons)}), one-shot class, nothing background when every request is top tier")

# ---- 7b. learned classes: single-stream one-offs -> learned low after learn_min (only without X-Priority)
logs = []
h = logging.Handler(); h.emit = lambda rec: logs.append(rec.getMessage()); r.log.addHandler(h); r.log.setLevel(logging.INFO)
R, a, g = mk({"on": True, "learn_min": 20})
for i in range(25):
    b, why = route(R, chain(f"book{i}", 24000, 2), tags(f"ses-k{i}", name="ledger-writer"))
assert R.classify(tags(name="ledger-writer"))["src"] == "learned", R.agents["ledger-writer"].summary()
assert R.classify(tags(name="ledger-writer", prio="high"))["tier"] == "high"
assert sum("ledger-writer" in m and "learned low" in m for m in logs) == 1, logs
ok("unknown agent 'ledger-writer': 25 single-stream one-offs -> learned low (only when X-Priority is absent), logged once")
for i in range(25):                               # a coding agent: one conversation, growing, follow-up turns
    route(R, turn("code", i), tags("ses-code", name="builder"))
assert R.classify(tags(name="builder"))["src"] == "default", R.agents["builder"].summary()
ok(f"agent 'builder' with follow-ups stays default (follow-ups {R.agents['builder'].summary()['follow']:.0%})")
R.policy_doc = {"agent_priority": {"ledger-writer": "normal"}}
assert R.classify(tags(name="ledger-writer"))["src"] == "map"
ok("agent_priority overrides a learned low")
for i in range(30):                               # untagged one-offs: never inferred low
    route(R, chain(f"u{i}", 24000, 2), tags())
assert R.classify(tags())["src"] == "default"
ok("untagged traffic (no headers) is never inferred low")
R2, _, _ = mk({"on": True, "learn_min": 20})
st = R2.stats_for("par")
for i in range(25):
    st.add(False, 0.1, 2)                          # one-offs but 2 concurrent: not single-stream
assert st.learn(R2.pol) == "normal"
ok("concurrent one-off traffic is not learned low (single-stream rule)")

# ---- 8. mixed tagged + untagged: untagged keep today's reasons
R, a, g = mk({"on": True})
route(R, chain("T", 30000, 2), tags("ses-T"))
b, why = route(R, turn("U", 2, 12000))
assert why in ("new", "balance", "prefer"), why
b2, why2 = route(R, turn("U", 4, 12000))
assert b2 is b and why2 == "affinity", (b2.name, why2)
ok(f"untagged conversation next to tagged traffic: {why} then affinity, as today")


# ---- 9. end to end: headers -> kv_transfer_params, vLLM priority only to priority_sched engines, metrics
async def e2e():
    seen = []

    async def h(rd, wr):
        hd = (await rd.readuntil(b"\r\n\r\n")).decode()
        n = int(re.search(r"(?i)content-length: (\d+)", hd).group(1))
        body = json.loads(await rd.readexactly(n))
        seen.append((wr.get_extra_info("sockname")[1], body,
                     {l.split(":", 1)[0].lower(): l.split(":", 1)[1].strip() for l in hd.split("\r\n")[1:] if ":" in l}))
        b = b'{"ok":true,"usage":{"prompt_tokens":5,"completion_tokens":3}}'
        wr.write(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n" % len(b) + b)
        await wr.drain(); wr.close()
    s1 = await asyncio.start_server(h, "127.0.0.1", 18431)
    s2 = await asyncio.start_server(h, "127.0.0.1", 18432)
    a = r.Backend({"name": "arc", "url": "http://127.0.0.1:18431", "priority_sched": True, "dec": [83, 74], "prefill": 2800})
    g = r.Backend({"name": "rtx5090", "url": "http://127.0.0.1:18432", "dec": [383, 317], "prefill": 6500})
    for b in (a, g):
        b.healthy = True
    R = r.Router([a, g]); R.trace_dir = ""
    pf = tempfile.mktemp(suffix=".json")
    json.dump({**EX, "tiers": {**EX["tiers"], "background": {**EX["tiers"]["background"], "vllm_priority": 7}},
               "low_target": "arc"}, open(pf, "w"))
    R.apply_config({"inject_kv_ns": "shared", "session_policy": {"on": True}, "policy_file": pf})
    S = r.Server.__new__(r.Server); S.r = R
    S.split_stats = r.collections.Counter(ok=0, failed=0, seconds=0.0)
    S.stream_stats = r.collections.defaultdict(lambda: {"streams": 0, "eligible": 0, "eligible_s": 0.0, "tool_calls": 0})
    s3 = await asyncio.start_server(S.handle, "127.0.0.1", 18433)

    async def send(extra, path="/v1/chat/completions"):
        raw = json.dumps({"model": "m", "messages": [{"role": "user", "content": "hi"}]}).encode()
        hh = "".join(f"{k}: {v}\r\n" for k, v in extra.items())
        rd, wr = await asyncio.open_connection("127.0.0.1", 18433)
        wr.write(f"POST {path} HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n{hh}Content-Length: {len(raw)}\r\n"
                 f"Connection: close\r\n\r\n".encode() + raw)
        await wr.drain(); out = await rd.read(); wr.close()
        return out

    await send({"X-Session-Id": "ses_A", "X-Parent-Session-Id": "ses_root", "X-Agent": "main", "X-Agent-Name": "assistant",
                "X-Priority": "interactive"})
    port, body, hd = seen[-1]
    k = body["kv_transfer_params"]
    assert k["llmt_ns"] == "shared" and k["llmt_cls"] == "agent" and re.fullmatch(r"[0-9a-f]{16}", k["llmt_sess"])
    assert re.fullmatch(r"[0-9a-f]{16}", k["llmt_parent"]) and k["llmt_agent"] == "m" and "priority" not in body
    assert "ses_A" not in json.dumps(body) and "assistant" not in json.dumps(body) and "llmt_term" not in k
    ok("tagged request: llmt_cls agent + 64-bit session / parent hashes + letter; raw ids and names not sent")
    await send({"X-Session-Id": "ses_H", "X-Agent": "summary", "X-Agent-Name": "helper-maps", "X-Priority": "interactive"})
    assert seen[-1][1]["kv_transfer_params"]["llmt_term"] == 1
    ok("terminal role -> llmt_term 1")
    await send({"X-Session-Id": "ses_B", "X-Agent": "main", "X-Agent-Name": "assistant", "X-Priority": "background"})
    port, body, hd = seen[-1]
    assert port == 18431 and body["kv_transfer_params"]["llmt_cls"] == "low" and body["priority"] == 7
    ok("background tier: arc (low_target, priority_sched) with the tier's vLLM priority 7, class low")
    await send({"X-Session-Id": "ses_N", "X-Agent": "main", "X-Agent-Name": "nightly:tidy", "X-Priority": "background"})
    assert seen[-1][1]["kv_transfer_params"]["llmt_cls"] == "lowest"
    ok("lowest-retention batch item -> llmt_cls lowest")
    a.healthy = False                              # only the FCFS 5090 left: no priority field
    await send({"X-Priority": "background"})
    port, body, hd = seen[-1]
    assert port == 18432 and "priority" not in body and body["kv_transfer_params"]["llmt_cls"] == "low"
    ok("background request on an engine without priority scheduling: no priority field")
    a.healthy = True
    await send({})
    port, body, hd = seen[-1]
    assert body["kv_transfer_params"] == {"llmt_ns": "shared", "llmt_cls": "normal"} and "priority" not in body
    ok("untagged: llmt_cls normal only")
    m = S.metrics().decode()
    assert 'llmt_router_agent_requests_total{agent="assistant"} 2' in m, m
    assert 'llmt_router_class_requests_total{tier="background",cls="low",source="header"} 2' in m, m
    assert "llmt_router_sessions 4" in m and "llmt_router_groups 2" in m, m
    line = S.agents_line()
    assert "assistant: req=2" in line and "tiers=" in line, line
    ok("metrics + agents line (local only): per agent name requests / follow-ups / reuse / concurrency / tiers")
    # low_wait: every backend saturated -> the low request waits until one frees up
    R.session_policy["low_wait_s"] = 3
    R.policy_doc.pop("low_wait_s", None)
    for b in (a, g):
        b.eng_waiting, b.eng_t = 1, time.monotonic()

    async def free_later():
        await asyncio.sleep(0.6)
        a.eng_waiting = 0; a.eng_t = time.monotonic()
    t0 = time.monotonic()
    await asyncio.gather(send({"X-Priority": "background"}), free_later())
    dt = time.monotonic() - t0
    assert 0.5 < dt < 2.5 and seen[-1][0] == 18431, (dt, seen[-1][0])
    assert R.low_waits == 1
    ok(f"low_wait: held {dt:.1f} s while both GPUs were saturated, then sent to the one that freed up")
    t0 = time.monotonic()
    for b in (a, g):
        b.eng_waiting, b.eng_t = 1, time.monotonic()
    await send({"X-Agent": "main", "X-Agent-Name": "assistant", "X-Priority": "interactive"})
    assert time.monotonic() - t0 < 0.5
    ok("top-tier requests never wait")
    for s_ in (s1, s2, s3):
        s_.close()
asyncio.run(e2e())
print(f"ALL PASS ({n_ok})")
