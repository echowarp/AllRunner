#!/usr/bin/env python3
"""Published-benchmark replicas on a live production engine that yield to real traffic (no restart, no config change:
run the prompts and cancel them when a real request comes in). A poller reads the engine's /metrics every 0.25 s; when running + waiting exceeds our own in-flight count, a real
request is there: every benchmark request is closed (vLLM aborts it), the cell is discarded, and the cell reruns after
the engine has been idle for IDLE_S. Prompts come only from public datasets (fetch_data.sh: HumanEval, GSM8K, MT-Bench, json-mode-eval, Gutenberg).

  yieldbench.py --base http://127.0.0.1:8001 --suite spark|sgl8k|agg|all --label L --out results.jsonl

Suites (replicas of published RTX 5090 harnesses):
  spark  gittensor-model-hub/Qwen3.8-27B-NVFP4 cross-engine harness: 24 prompts (6 chat / 6 code / 6 math / 6 JSON),
         max 256 tokens, temperature 0, thinking off, sequential. Their RTX 5090 rows: SGLang 85.8, SparkInfer 92.9
         (no spec); SGLang + DSpark 161.7; SparkInfer + DSpark bench harness 264.8 overall (420.2 code).
  sgl8k  SGLang cookbook envelope: ISL 8192 / OSL 1024 (ignore_eos), concurrency 1, temperature 0. DFlash2 + NVFP4 on
         one RTX 5090: TPOT 4.92 ms median (~203 tok/s), the "206.1 tok/s" figure.
  agg    gittensor concurrency test: distinct prompts per request, 128 output tokens (ignore_eos), 32 requests per level at
         concurrency 1/2/4/8/16/32. Their RTX 5090: SparkInfer 88.9/162.1/249.6/344.3/345.1/318.2, vLLM 0.28
         -/134.5/266.6/545.8/546.6 aggregate tok/s.
Per cell: tok/s (per request decode and aggregate), TTFT, tokens/step from the engine's spec-decode counters (valid
because the cell ran with no foreign request).
"""
import argparse, json, random, re, threading, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="http://127.0.0.1:8001")
ap.add_argument("--model", default="qwen3.8-27b")
ap.add_argument("--suite", default="all")
ap.add_argument("--label", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--idle-s", type=float, default=10.0)
ap.add_argument("--max-tries", type=int, default=20)
ap.add_argument("--reps", type=int, default=2)
ap.add_argument("--data", required=True, help="dir from fetch_data.sh")
a = ap.parse_args()

D = a.data
HE = [json.loads(l) for l in open(f"{D}/HumanEval.jsonl")]                 # openai/human-eval
GSM = [json.loads(l) for l in open(f"{D}/gsm8k_test.jsonl")]               # openai/grade-school-math test split
MTB = [json.loads(l) for l in open(f"{D}/mt_bench_question.jsonl")]        # lm-sys/FastChat MT-Bench questions
JME = [json.loads(l) for l in open(f"{D}/json_mode_eval.jsonl")]           # NousResearch/json-mode-eval
GUT = open(f"{D}/pg1342.txt", encoding="utf-8", errors="ignore").read()   # Project Gutenberg #1342
import os
GUTL = GUT + "".join(open(f"{D}/{b}", encoding="utf-8", errors="ignore").read() for b in ("pg2701.txt", "pg2600.txt") if os.path.exists(f"{D}/{b}"))   # + #2701, #2600 for long prompts
# chat = the first-turn question of the first MT-Bench item of each of 6 categories (writing, roleplay, reasoning, math,
# coding, extraction order as in the file); JSON = json-mode-eval rows 0-5 (system schema + user turn, as published)
_seen = {}
for q in MTB:
    _seen.setdefault(q["category"], q["turns"][0])
CHAT = list(_seen.values())[:6]
JSONM = [r["prompt"] for r in JME[:6]]


def get_metrics():
    t = urllib.request.urlopen(a.base + "/metrics", timeout=5).read().decode()
    def g(name):
        m = re.findall(r"^%s\{[^}]*\} ([0-9.e+]+)$" % re.escape(name), t, re.M)
        return sum(float(x) for x in m)
    return {"running": g("vllm:num_requests_running"), "waiting": g("vllm:num_requests_waiting"),
            "drafts": g("vllm:spec_decode_num_drafts_total"), "acc": g("vllm:spec_decode_num_accepted_tokens_total")}


class Cell:
    def __init__(self):
        self.own = 0; self.lock = threading.Lock(); self.abort = threading.Event(); self.foreign = 0

    def watch(self, stop):
        while not stop.is_set():
            try:
                m = get_metrics()
                with self.lock:
                    own = self.own
                if m["running"] + m["waiting"] > own + 0.5:
                    self.foreign = int(m["running"] + m["waiting"] - own); self.abort.set(); return
            except Exception:
                pass
            time.sleep(0.25)


def one(cell, body, res):
    req = urllib.request.Request(a.base + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with cell.lock:
        cell.own += 1
    t0 = time.time(); tf = None; usage = None
    try:
        r = urllib.request.urlopen(req, timeout=900)
        for line in r:
            if cell.abort.is_set():
                r.close(); res.append(None); return
            line = line.strip()
            if not line.startswith(b"data:") or line == b"data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                usage = d["usage"]
            ch = d.get("choices") or []
            if ch and tf is None and ((ch[0].get("delta") or {}).get("content") or (ch[0].get("delta") or {}).get("reasoning_content")):
                tf = time.time()
        te = time.time()
        n = (usage or {}).get("completion_tokens", 0)
        res.append({"ttft": (tf or te) - t0, "dec": (n - 1) / max(1e-6, te - (tf or te)) if n > 1 else 0, "n": n,
                    "prompt": (usage or {}).get("prompt_tokens"), "t0": t0, "te": te})
    except Exception as e:
        res.append(None if cell.abort.is_set() else {"error": repr(e)[:200]})
    finally:
        with cell.lock:
            cell.own -= 1


def wait_idle():
    t_idle = None
    while True:
        try:
            m = get_metrics()
            busy = m["running"] + m["waiting"] > 0
        except Exception:
            busy = True
        if busy:
            t_idle = None
        elif t_idle is None:
            t_idle = time.time()
        elif time.time() - t_idle >= a.idle_s:
            return
        time.sleep(0.5)


def run_cell(name, bodies, conc):
    """Run bodies with `conc` workers; retry until no foreign request appeared during the cell."""
    for tr in range(1, a.max_tries + 1):
        wait_idle()
        cell = Cell(); stop = threading.Event(); res = []
        m0 = get_metrics(); t0 = time.time()
        w = threading.Thread(target=cell.watch, args=(stop,), daemon=True); w.start()
        q = list(bodies); ql = threading.Lock()

        def worker():
            while not cell.abort.is_set():
                with ql:
                    if not q:
                        return
                    b = q.pop(0)
                one(cell, b, res)
        ths = [threading.Thread(target=worker) for _ in range(conc)]
        [t.start() for t in ths]; [t.join() for t in ths]
        stop.set(); m1 = get_metrics(); wall = time.time() - t0
        if cell.abort.is_set():
            print(f"  {name}: yielded to {cell.foreign} real request(s) (try {tr}), will rerun", flush=True)
            continue
        ok = [r for r in res if r and "error" not in r]
        errs = [r for r in res if r and "error" in r]
        ntok = sum(r["n"] for r in ok)
        dr = m1["drafts"] - m0["drafts"]; acc = m1["acc"] - m0["acc"]
        out = {"label": a.label, "cell": name, "concurrency": conc, "requests": len(ok), "errors": len(errs),
               "tok": ntok, "wall_s": round(wall, 2), "agg_tok_s": round(ntok / wall, 1),
               "dec_tok_s_mean": round(sum(r["dec"] for r in ok) / max(1, len(ok)), 1),
               "dec_tok_s_tokweighted": round(ntok / max(1e-6, sum(r["n"] / r["dec"] for r in ok if r["dec"])), 1),
               "ttft_s_mean": round(sum(r["ttft"] for r in ok) / max(1, len(ok)), 3),
               "tok_per_step": round(1 + acc / dr, 3) if dr else None, "tries": tr, "time": time.strftime("%F %T")}
        if errs:
            out["first_error"] = errs[0]["error"]
        print("  " + json.dumps(out), flush=True)
        with open(a.out, "a") as f:
            f.write(json.dumps(out) + "\n")
        return out
    print(f"  {name}: gave up after {a.max_tries} tries (busy production)", flush=True)


def body(msg, max_tokens, temp=0.0, ignore_eos=False):
    msgs = msg if isinstance(msg, list) else [{"role": "user", "content": msg}]
    b = {"model": a.model, "messages": msgs, "max_tokens": max_tokens, "stream": True,
         "stream_options": {"include_usage": True}, "temperature": temp,
         "chat_template_kwargs": {"enable_thinking": False}}
    if ignore_eos:
        b["ignore_eos"] = True; b["min_tokens"] = max_tokens
    return b


def tok_prefix(text, n):
    """First n tokens of text (exact, via the engine's /tokenize + /detokenize)."""
    r = json.loads(urllib.request.urlopen(urllib.request.Request(a.base + "/tokenize", json.dumps(
        {"model": a.model, "prompt": text}).encode(), {"Content-Type": "application/json"}), timeout=60).read())
    ids = r["tokens"][:n]
    d = json.loads(urllib.request.urlopen(urllib.request.Request(a.base + "/detokenize", json.dumps(
        {"model": a.model, "tokens": ids}).encode(), {"Content-Type": "application/json"}), timeout=60).read())
    return d["prompt"]


suites = a.suite.split(",") if a.suite != "all" else ["spark", "sgl8k", "agg"]
if "cold" in suites:
    # cold prefill cost vs prompt length (the "no cache" baseline for production cache savings): unique first line per
    # request so nothing is cached, max_tokens 1, TTFT = time to the first streamed token
    longtxt = tok_prefix(GUTL, 200000)
    for rep in range(a.reps):
        for n in (4000, 8000, 16000, 32000, 64000, 96000, 128000, 160000, 190000):
            txt = tok_prefix(longtxt, n - 40)
            run_cell(f"cold-{n // 1000}k-r{rep}", [body(f"[cold {rep}-{n}-{random.randrange(1 << 30)}]\n" + txt + "\n\nSummarize.", 1)], 1)
print(f"yieldbench {a.label}: {a.base} suites {suites} start {time.strftime('%T')}", flush=True)
if "spark" in suites:
    cats = {"chat": CHAT, "code": [h["prompt"] for h in HE[:6]], "math": [g["question"] for g in GSM[:6]], "json": JSONM}
    for rep in range(a.reps):
        for c, ps in cats.items():
            run_cell(f"spark-{c}-r{rep}", [body(p if c != "code" else "Complete this Python function:\n\n" + p, 256)
                                           for p in ps], 1)
if "sgl8k" in suites:
    base = tok_prefix(GUT[20000:200000], 8150)
    for rep in range(a.reps):
        bs = []
        for i in range(3):
            nonce = f"[doc {rep}-{i}-{random.randrange(1 << 30)}]\n"     # cold prefix per request, as bench_serving's random set
            bs.append(body(nonce + base + "\n\nContinue the story from here.", 1024, ignore_eos=True))
        run_cell(f"sgl8k-c1-r{rep}", bs, 1)
if "agg" in suites:
    pool = [g["question"] for g in GSM[100:400]] + [h.get("prompt", "") for h in HE[10:164]]
    for c in (1, 2, 4, 8, 16, 32):
        ps = random.Random(c).sample(pool, 32)
        run_cell(f"agg-c{c}", [body(f"[req {c}-{i}] " + p, 128, ignore_eos=True) for i, p in enumerate(ps)], c)
print(f"yieldbench {a.label}: done {time.strftime('%T')}", flush=True)
