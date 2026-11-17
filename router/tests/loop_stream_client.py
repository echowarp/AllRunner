"""Manual loop-stop harness, part 2: send one streamed chat request (JSON body in argv[1]) to a router on
127.0.0.1:18010 and print counts: events, ids, finish reasons, reasoning/content word counts, tool names, whether a
think tag leaked into the content. See ../README.md (Tests) for the full recipe."""
import json, sys, urllib.request, time
body = json.loads(sys.argv[1]); t0 = time.time()
r = urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:18010/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600)
reasoning, content, tools, ids, fins, n, roles, order = [], [], [], set(), [], 0, 0, []
for line in r:
    line = line.strip()
    if not line.startswith(b"data:") or line == b"data: [DONE]": continue
    d = json.loads(line[5:]); ids.add(d.get("id")); n += 1
    for c in d.get("choices") or []:
        dl = c.get("delta") or {}
        if "role" in dl: roles += 1
        if dl.get("reasoning"): reasoning.append(dl["reasoning"]); order.append("R")
        if dl.get("content"): content.append(dl["content"]); order.append("C")
        for t in dl.get("tool_calls") or []: tools.append(t); order.append("T")
        if c.get("finish_reason"): fins.append(c["finish_reason"])
# R after C means thinking text arrived after the answer started (a misclassified continuation)
switch = "".join(o for i, o in enumerate(order) if i == 0 or o != order[i - 1])
rtxt, ctxt = "".join(reasoning), "".join(content)
names = sorted({(t.get("function") or {}).get("name") for t in tools if (t.get("function") or {}).get("name")})
print(json.dumps({"events": n, "ids": len(ids), "role_deltas": roles, "finish": fins, "sequence": switch[:20],
                  "reasoning_words": len(rtxt.split()), "content_words": len(ctxt.split()), "tool_names": names,
                  "think_tag_leak": ("<think>" in ctxt or "</think>" in ctxt), "s": round(time.time() - t0, 1)}))
print("content tail:", repr(ctxt[-160:]))
