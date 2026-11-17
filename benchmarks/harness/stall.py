#!/usr/bin/env python3
"""Decode time taken from OTHER agents by each request's prefill: while a request prefills (modeled wait w), every
other request in flight on the same GPU stalls (a prefill step costs far more than a decode step). stall = w x others."""
import json, bisect, collections
R = [json.loads(l) for l in open("perreq.jsonl")]
by = collections.defaultdict(list)
for r in R:
    if r["done_s"] is not None: by[r["be"]].append(r)
res = collections.Counter(); n_oth = []
for be, rs in by.items():
    iv = sorted((r["t"], r["t"] + r["done_s"]) for r in rs)
    starts = [a for a, _ in iv]
    for r in rs:
        t = r["t"]
        # requests in flight at t (started before, not finished), excluding r itself
        i = bisect.bisect_right(starts, t)
        others = sum(1 for a, b in iv[max(0, i - 400):i] if b > t) - 1
        others = max(0, others); n_oth.append(others)
        for arm in ("stock", "ours"):
            res[arm + "_self"] += r["w_" + arm]; res[arm + "_others"] += r["w_" + arm] * others
n = len(n_oth); conc = [x for x in n_oth if x > 0]
print(f"{n} requests; {len(conc)} ({len(conc) / n:.0%}) prefilled while other agents were in flight (mean {sum(conc) / max(1, len(conc)):.1f} others)")
for arm in ("stock", "ours"):
    print(f"{arm:5s}: own prefill {res[arm + '_self'] / 3600:.2f} h; decode time taken from other agents {res[arm + '_others'] / 3600:.2f} agent-h")
