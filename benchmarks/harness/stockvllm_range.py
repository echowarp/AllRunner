#!/usr/bin/env python3
"""(per-turn version) what STOCK vLLM's prefix cache would have reused on the same production window (numbers only).
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
import hashlib
BES = be.split(",")
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


T = {r["cid"]: r for r in lines(tg) if str(r.get("code")) == "200" and r.get("backend") in BES and r.get("prompt_tokens")
     and t0 <= r["t"] < t1 and r.get("cid")}
C = {}; vision = set(); last_role = {}


def strip(c):
    if isinstance(c, list):
        out = []
        for p in c:
            if isinstance(p, dict) and p.get("type") in ("image_url", "image", "input_image"):
                out.append({"type": "img", "h": hashlib.sha1(json.dumps(p, sort_keys=True).encode()).hexdigest()[:16]})
            else:
                out.append(p)
        return out
    return c


for r in lines(cg):
    if r.get("cid") in T and isinstance(r.get("body"), dict) and r["body"].get("messages") is not None:
        b = r["body"]; m = b["messages"]
        if any(isinstance(x.get("content"), list) and any(isinstance(p, dict) and p.get("type") in ("image_url", "image", "input_image")
                                                       for p in x["content"]) for x in m):
            vision.add(r["cid"])
        last_role[r["cid"]] = (m[-1] if m else {}).get("role")
        C[r["cid"]] = json.dumps(b.get("tools", []), sort_keys=True) + "".join(
            json.dumps({k: (strip(x.get(k)) if k == "content" else x.get(k)) for k in ("role", "content", "tool_calls", "tool_call_id")}, sort_keys=True) for x in m)


def lcp(a, b):
    n = min(len(a), len(b)); i = 0; step = 4096
    while i < n and a[i:i + step] == b[i:i + step]: i += step
    i = max(0, i - step) if i else 0
    while i < n and a[i] == b[i]: i += 1
    return i


reqs = sorted((T[c]["t"], c) for c in C)
conv = {c: (T[c].get("conv") or [c])[0] for _, c in reqs}
def run(B, cap, tail):
    prev = []; lru = collections.OrderedDict(); used = 0; out = {}
    for t, cid in reqs:
        s = C[cid]; P = T[cid]["prompt_tokens"]; tpc = P / max(1, len(s)); best = 0; src = None
        for e in prev[-300:]:
            if cap and e not in lru: continue
            l = lcp(s, C[e]) * tpc
            L1 = T[e]["prompt_tokens"]; nb = L1 // B
            if nb >= 2 and nb * B <= l and (nb - 1) * B > best:
                best = (nb - 1) * B; src = e
            if tail and conv[e] == conv[cid] and l > best:     # token-exact hit on the same chat's held tail
                best = int(l); src = e
        best = min(best, P - 1); prev.append(cid)
        out[cid] = (P, best)
        if cap:
            if src is not None: lru.move_to_end(src)
            add = P - best; lru[cid] = add; used += add
            while used > cap and len(lru) > 1:
                k, v = lru.popitem(last=False); used -= v
    return out


def wait(P, r):          # prefill seconds for the part not reused, at its depth (marginal cost on the cold curve)
    return cold(P) - cold(r)


def q(xs, f):
    xs = sorted(xs); return xs[min(len(xs) - 1, int(f * len(xs)))]


allreqs = reqs
arms = {}
for bname in BES:
    reqs = [x for x in allreqs if T[x[1]]["backend"] == bname]
    for k, v in run(832, 140000, False).items(): arms.setdefault("stock vLLM (832-token blocks, 140k cache per GPU)", {})[k] = v
    for k, v in run(3200, 480000, True).items(): arms.setdefault("ours (3,200-token blocks + tail hold, 480k cache per GPU)", {})[k] = v
reqs = allreqs
print({b: sum(1 for _, c in reqs if T[c]["backend"] == b) for b in BES}, "requests per backend")
_st, _ou = arms["stock vLLM (832-token blocks, 140k cache per GPU)"], arms["ours (3,200-token blocks + tail hold, 480k cache per GPU)"]
with open("perreq.jsonl", "w") as f:       # numbers only: no content
    for _, c in reqs:
        r = T[c]
        f.write(json.dumps({"cid": c, "be": r["backend"], "t": r["t"], "first_s": r.get("first_s"), "done_s": r.get("done_s"),
                            "P": r["prompt_tokens"], "vision": c in vision, "tool": last_role.get(c) == "tool",
                            "w_stock": round(wait(*_st[c]), 4), "w_ours": round(wait(*_ou[c]), 4)}) + "\n")
tool = [c for _, c in reqs if last_role.get(c) == "tool" and c not in vision]
agent = [c for _, c in reqs if c not in vision]
vis = [c for _, c in reqs if c in vision]
Ptot = lambda sel: sum(T[c]["prompt_tokens"] for c in sel)
print(f"{len(reqs)} requests: {len(vis)} vision ({Ptot(vis) / max(1, Ptot([c for _, c in reqs])):.0%} of prompt tokens), "
      f"{len(agent)} text/tool agent, of which {len(tool)} tool-result turns")
for name, o in arms.items():
    for lab, sel in (("agent turns", agent), ("agent tool-result turns", tool), ("vision requests", vis)):
        if not sel: continue
        w = [wait(*o[c]) for c in sel]; comp = [o[c][0] - o[c][1] for c in sel]
        share = sum(o[c][1] for c in sel) / max(1, Ptot(sel))
        print(f"{name:52s} {lab:18s}: prefill wait median {q(w, .5):5.2f} s  p90 {q(w, .9):5.2f} s  mean {sum(w) / len(w):5.2f} s | "
              f"recomputed tokens median {q(comp, .5):6.0f}  p90 {q(comp, .9):6.0f} | reused {share:.0%}")
