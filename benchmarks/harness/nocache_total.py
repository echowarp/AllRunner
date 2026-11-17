#!/usr/bin/env python3
"""Sum of cold-prefill seconds over routed requests in a window (trace only, no content): the no-cache GPU prefill time."""
import sys, json, glob, bisect
t0, t1, bes, cur, tg = float(sys.argv[1]), float(sys.argv[2]), sys.argv[3].split(","), sys.argv[4], sys.argv[5]
curve = sorted((float(k) * 1000, float(v)) for k, v in (x.split(":") for x in cur.split(",")))
def cold(n):
    xs = [c[0] for c in curve]
    if n <= 0: return 0.0
    if n <= xs[0]: return curve[0][1] * n / xs[0]
    i = min(bisect.bisect_left(xs, n), len(xs) - 1); (x0, y0), (x1, y1) = curve[i - 1], curve[i]
    return y0 + (y1 - y0) * (n - x0) / (x1 - x0)
n = 0; s = 0.0
for g in tg.split():
    for f in sorted(glob.glob(g)):
        for l in open(f, errors="ignore"):
            try: r = json.loads(l)
            except Exception: continue
            if str(r.get("code")) == "200" and r.get("backend") in bes and r.get("prompt_tokens") and t0 <= r.get("t", 0) < t1:
                n += 1; s += cold(r["prompt_tokens"])
print(f"{n} requests, no-cache prefill {s / 3600:.2f} h")
