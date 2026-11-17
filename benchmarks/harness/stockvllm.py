#!/usr/bin/env python3
"""what STOCK vLLM's prefix cache would have reused on the same production window (numbers only).
Stock rules for this hybrid (GDN) model with prefix caching on (measured 10-14):
  - KV is reused in whole blocks of B tokens of an identical prefix;
  - the GDN state is saved only at the last full block of each earlier request's PROMPT, so a request can reuse an
    earlier request's prefix only if that request's saved state (floor(L1/B)*B) lies inside the shared prefix;
  - the MTP drafter drops the last block of a hit: reuse = (floor(L1/B) - 1) * B;
  - no RAM tier, no token-exact tail reuse.
UPPER BOUND for stock: nothing is ever evicted (stock 16-bit KV holds ~3.4x fewer tokens than our int4, so it would
evict more). Content is read locally and never printed.
  stockvllm.py <t0> <t1> <backend> <cold curve k:s,...> "<trace globs>" "<capture globs>" <B>[:<capacity tokens>] ...
With a capacity, cached prefixes are evicted least-recently-used once the distinct cached tokens exceed it (a reuse
touches the entry it came from; each request adds only the tokens it did not reuse). Approximation of vLLM's block LRU."""
import collections
import sys, json, glob, bisect

t0, t1, be, cur, tg, cg = float(sys.argv[1]), float(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6]
curve = sorted((float(k) * 1000, float(v)) for k, v in (x.split(":") for x in cur.split(",")))


def cold(n):
    xs = [c[0] for c in curve]
    if n <= 0: return 0.0
    if n <= xs[0]: return curve[0][1] * n / xs[0]
    i = min(bisect.bisect_left(xs, n), len(xs) - 1); (x0, y0), (x1, y1) = curve[i - 1], curve[i]
    return y0 + (y1 - y0) * (n - x0) / (x1 - x0)


def lines(globs):
    for g in globs.split():
        for f in sorted(glob.glob(g)):
            for l in open(f, errors="ignore"):
                try: yield json.loads(l)
                except Exception: pass


T = {r["cid"]: r for r in lines(tg) if str(r.get("code")) == "200" and r.get("backend") == be and r.get("prompt_tokens")
     and t0 <= r["t"] < t1 and r.get("cid")}
C = {}
for r in lines(cg):
    if r.get("cid") in T and isinstance(r.get("body"), dict) and r["body"].get("messages") is not None:
        b = r["body"]
        C[r["cid"]] = json.dumps(b.get("tools", []), sort_keys=True) + "".join(
            json.dumps({k: m.get(k) for k in ("role", "content", "tool_calls", "tool_call_id")}, sort_keys=True) for m in b["messages"])


def lcp(a, b):
    n = min(len(a), len(b)); i = 0; step = 4096
    while i < n and a[i:i + step] == b[i:i + step]: i += step
    i = max(0, i - step) if i else 0
    while i < n and a[i] == b[i]: i += 1
    return i


reqs = sorted((T[c]["t"], c) for c in C)
P_all = sum(T[c]["prompt_tokens"] for _, c in reqs)
nocache = sum(cold(T[c]["prompt_tokens"]) for _, c in reqs)
for spec in sys.argv[7:]:
    B, cap = (int(x) for x in spec.split(":")) if ":" in spec else (int(spec), 0)
    reused = 0.0; with_cache = 0.0; prev = []
    lru = collections.OrderedDict(); used = 0          # cid -> distinct tokens it added
    for t, cid in reqs:
        s = C[cid]; P = T[cid]["prompt_tokens"]; tpc = P / max(1, len(s))
        best = 0; src = None
        for e in prev[-300:]:
            if cap and e not in lru: continue
            L1 = T[e]["prompt_tokens"]; nb = L1 // B
            if nb < 2: continue
            if nb * B <= lcp(s, C[e]) * tpc and (nb - 1) * B > best:
                best = (nb - 1) * B; src = e
        best = min(best, P - 1)
        reused += best; with_cache += cold(P) - cold(best)
        prev.append(cid)
        if cap:
            if src is not None: lru.move_to_end(src)
            add = P - best; lru[cid] = add; used += add
            while used > cap and len(lru) > 1:
                k, v = lru.popitem(last=False); used -= v
    print(f"stock rules, B={B}, cache {'unlimited' if not cap else f'{cap:,} tokens (LRU)'}: reused {reused / P_all:.1%} of {P_all:,.0f} prompt tokens; "
          f"prefill {nocache / 3600:.2f} h -> {with_cache / 3600:.2f} h ({nocache / max(with_cache, 1e-9):.2f}x less)")
