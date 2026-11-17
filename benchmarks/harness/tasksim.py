#!/usr/bin/env python3
"""time per multi-agent task, modeled. Numbers only (no content).
Task = router root conversation (conv[0]); agent = full conv chain; each agent's requests run in order, agents in
parallel. The outside world is kept from the trace: the gap between a response's end and the agent's next request
(tool execution, user) is replayed unchanged. Each request is re-timed under an engine model:
  prefill: modeled wait (perreq.jsonl: stock vLLM cache rules vs ours) x engine prefill factor; runs exclusively on the
           GPU (FIFO) and pauses every decoding request meanwhile;
  decode:  completion_tokens at the per-stream rate agg(n)/n of the engine's measured / published throughput curve,
           n = requests decoding at that moment.
Event-driven per GPU. Prints median / p90 task time for multi-agent tasks, all tasks, and a calibration against the
measured trace.  tasksim.py <perreq.jsonl> "<trace globs>" """
import sys, json, glob, math, heapq, collections, statistics as st

perreq = {r["cid"]: r for r in (json.loads(l) for l in open(sys.argv[1]))}
tr = {}
for g in sys.argv[2].split():
    for f in sorted(glob.glob(g)):
        for l in open(f, errors="ignore"):
            try: r = json.loads(l)
            except Exception: continue
            if r.get("cid") in perreq: tr[r["cid"]] = r

ENGINES = {   # name: (throughput curve {n: aggregate tok/s}, prefill factor vs our 5090 cold curve, cache arm)
    "RTX 5090 · ours":         ({1: 287.5, 2: 449.0, 4: 608.6, 8: 775.2, 16: 1121.0, 32: 1178.2}, 1.0, "ours"),
    "RTX 5090 · vLLM 0.28":    ({1: 86.8, 2: 134.5, 4: 266.6, 8: 545.8, 16: 546.6}, 1.0, "stock"),
    "RTX 5090 · best published": ({1: 206.1, 2: 162.1, 4: 266.6, 8: 545.8, 16: 546.6}, 1.0, "stock"),
    "Arc B70 · ours":          ({1: 102.7, 2: 162.3, 4: 274.4, 8: 442.4, 16: 528.4}, 11.494 / 4.55, "ours"),
    "Arc B70 · exl3xpu upstream": ({1: 78.7, 2: 138.4, 4: 237.8, 8: 339.5, 16: 357.5}, 11.494 / 4.55 * 2851 / 2124, "stock"),
}


def rate(curve, n):
    ks = sorted(curve)
    if n <= ks[0]: return curve[ks[0]] / ks[0]
    if n >= ks[-1]: return curve[ks[-1]] / n
    for a, b in zip(ks, ks[1:]):
        if a <= n <= b:
            fa, fb = math.log(a), math.log(b); x = (math.log(n) - fa) / (fb - fa)
            agg = curve[a] + (curve[b] - curve[a]) * x
            return agg / n


# agents: ordered request lists with the observed outside-world gap before each request
agents = collections.defaultdict(list)
for cid, r in tr.items():
    conv = r.get("conv") or [cid]
    agents[tuple(conv)].append(r)
chains = []
for key, rs in agents.items():
    rs.sort(key=lambda r: r["t"]); seq = []
    for i, r in enumerate(rs):
        gap = 0.0 if i == 0 else max(0.0, r["t"] - (rs[i - 1]["t"] + (rs[i - 1].get("done_s") or 0)))
        seq.append((r["cid"], gap))
    chains.append((key[0], rs[0]["t"], seq))


def simulate(eng, gpu_filter):
    curve, pf, arm = ENGINES[eng]
    ch = [c for c in chains if gpu_filter(c)]
    t = 0.0; arrivals = []          # heap (time, chain index, step)
    for i, (_, t0, _) in enumerate(ch):
        heapq.heappush(arrivals, (t0, i, 0))
    pq = collections.deque(); pre = None; dec = {}   # dec: cid -> remaining tokens
    start = {}; end = {}; owner = {}
    t = arrivals[0][0] if arrivals else 0.0
    while arrivals or pq or pre or dec:
        # next events
        t_arr = arrivals[0][0] if arrivals else math.inf
        t_pre = pre[1] if pre else math.inf
        if dec and not pre:
            rr = rate(curve, len(dec)); t_dec = t + max(0.0, min(dec.values())) / rr
        else:
            rr = 0.0; t_dec = math.inf
        tn = min(t_arr, t_pre, t_dec)
        if tn == math.inf: break
        if rr:
            for c in dec: dec[c] -= rr * (tn - t)
        t = tn
        if t_pre == tn and pre:
            c = pre[0]; pre = None
            n = (tr[c].get("completion_tokens") or 1)
            dec[c] = max(1, n)
        elif t_dec == tn:
            done = [c for c, v in dec.items() if v <= 1e-3]
            for c in done:
                del dec[c]; end[c] = t
                i, k = owner[c]
                if k + 1 < len(ch[i][2]):
                    heapq.heappush(arrivals, (t + ch[i][2][k + 1][1], i, k + 1))
        else:
            _, i, k = heapq.heappop(arrivals)
            c = ch[i][2][k][0]; owner[c] = (i, k); start[c] = t
            pq.append(c)
        if not pre and pq:
            c = pq.popleft(); w = perreq[c]["w_" + arm] * pf
            pre = (c, t + w)
    tasks = collections.defaultdict(list)
    for i, (root, _, seq) in enumerate(ch):
        tasks[root].extend(c for c, _ in seq)
    out = {}
    for root, cs in tasks.items():
        cs = [c for c in cs if c in end]
        if cs:
            iv = sorted((start[c], end[c]) for c in cs); busy = 0.0; a0, b0 = iv[0]
            for a, b in iv[1:]:
                if a > b0: busy += b0 - a0; a0, b0 = a, b
                else: b0 = max(b0, b)
            busy += b0 - a0
            out[root] = (min(start[c] for c in cs), max(end[c] for c in cs), len({tuple(tr[c].get("conv") or [c]) for c in cs}), busy)
    return out


def q(xs, f):
    xs = sorted(xs); return xs[min(len(xs) - 1, int(f * len(xs)))]


# calibration: measured task times on the RTX 5090 (trace)
meas = collections.defaultdict(list)
for cid, r in tr.items():
    if r["backend"] == "rtx5090" and r.get("done_s") is not None:
        meas[(r.get("conv") or [cid])[0]].append((r["t"], r["t"] + r["done_s"], tuple(r.get("conv") or [cid])))
on5090 = lambda c: tr[c[2][0][0]]["backend"] == "rtx5090"
res = {}
for eng in ENGINES:
    res[eng] = simulate(eng, on5090)      # the same production tasks on each GPU model
multi = [k for k, v in res["RTX 5090 · ours"].items() if v[2] >= 2]
print(f"{len(res['RTX 5090 · ours'])} tasks, {len(multi)} multi-agent (2+ agents)")
mt = [max(b for _, b, _ in meas[k]) - min(a for a, _, _ in meas[k]) for k in multi if k in meas]
print(f"measured (trace, RTX 5090 ours): multi-agent task median {q(mt, .5) / 60:.1f} min  p90 {q(mt, .9) / 60:.1f} min")
def mbusy(k):
    iv = sorted((a, b) for a, b, _ in meas[k]); busy = 0.0; a0, b0 = iv[0]
    for a, b in iv[1:]:
        if a > b0: busy += b0 - a0; a0, b0 = a, b
        else: b0 = max(b0, b)
    return busy + b0 - a0
mb = [mbusy(k) for k in multi if k in meas]
print(f"measured model-wait per multi-agent task (union of request intervals): median {q(mb, .5) / 60:.2f} min  p90 {q(mb, .9) / 60:.2f} min  mean {st.mean(mb) / 60:.2f} min")
for eng, o in res.items():
    b = [o[k][3] for k in multi if k in o]
    print(f"{eng:28s}: model-wait per multi-agent task median {q(b, .5) / 60:6.2f} min  p90 {q(b, .9) / 60:6.2f} min  mean {st.mean(b) / 60:6.2f} min  total {sum(b) / 3600:6.1f} h")
for eng, o in res.items():
    d = [o[k][1] - o[k][0] for k in multi if k in o]
    print(f"{eng:28s}: multi-agent task median {q(d, .5) / 60:6.1f} min  p90 {q(d, .9) / 60:6.1f} min  mean {st.mean(d) / 60:6.1f} min")
