# llmt-router: prefix-affinity routing over several vLLM backends

A small OpenAI-compatible proxy (one file, Python standard library only, asyncio) that spreads chat and completion
requests over several vLLM engines serving the **same model**, here one Intel Arc Pro B70 and one RTX 5090 serving
Qwen3.8-27B. It is built for agent traffic: many concurrent conversations, long prompts (20k+ tokens), tool calls,
and follow-up turns that repeat the whole conversation so far.

It does not authenticate anyone. Run it on localhost **behind an API gateway that holds the API key** (we use
llama-swap: the public model name proxies to the router, the router talks to the engines on 127.0.0.1).

```
client --(API key)--> gateway :1234 --> llmt-router :8010 --> vLLM arc :8000
                                                         \--> vLLM rtx5090 :8001
```

## What it does

- **Prefix affinity.** An agent's next turn repeats its earlier messages, so the engine that served the previous turn
  still holds that prefix in its KV cache (VRAM, then its host-RAM tier). The router hashes the request at every
  message boundary (cumulative 16-byte blake2b; the root covers model, tools and chat-template options, so different
  tool sets never match) and remembers which backend last received each prefix. Routing a turn back there skips
  re-prefilling 10-25k tokens. A match only forces placement when it is long (`affinity_min_tokens`,
  `affinity_min_frac`; a shared system prompt is not a session) and recent (`affinity_ttl_s`). With a KV tier shared
  by all GPUs, `shared_tier` credits the other backends with a fraction of the cached prefix.
- **Finish-time cost model.** For each usable backend it estimates when this request would finish there: the wait for
  prefills already queued on that GPU (prefills serialize), plus the uncached prompt tokens / `prefill` tok/s, plus the
  expected output tokens / the per-stream decode rate at (running + 1) streams (`dec`, measured). Expected output =
  running mean of observed completions, capped by the request's `max_tokens`. Conversations that finished a request in
  the last `return_window_s` (an agent running a tool) count as `returning_weight` of a running request. Without `dec`,
  a simpler `uncached / prefill + inflight x stream_cost` is used. Over `cap` in flight, each extra request adds 30 s.
- **Externality.** The new request also slows the requests already running on that GPU: each decodes its remaining
  ~gen/2 tokens at the slower (n+1)-stream rate, and stalls by `prefill_slowdown` while this request's prefill runs
  (measured on both GPUs: running streams lose about 2/3 of their rate during a mixed prefill+decode step). That delay,
  times `externality`, is added to the cost; the request goes to the backend with the smallest total.
- **Placement rules on top of the cost** (each with its own reason label):
  `ctx` (a request that would not fit a backend's `max_ctx` or `max_images` never goes there while one that fits is
  usable; a context-length or image-limit 400 is retried on a backend with a larger limit), `spill` (work-conserving:
  a GPU whose engine has requests waiting or KV fill >= `spill_kv` gets no new requests while another one has room),
  `prefer` (`prefer_until`: send everything to the fast GPU while it has fewer than N in flight), `balance`
  (`balance_new`: new conversations go to the GPU with the fewest active conversations per weight), `summary`
  (`summary_target`: large requests without tools, e.g. client compactions, go to one GPU first), `split`
  (disaggregated prefill through a shared KV tier, off by default).
- **Prompt estimate.** Whole messages (tool-call arguments, reasoning) and the tools array, in characters /
  `chars_per_token`, scaled by a running mean of real prompt sizes from the responses' `usage`. Near the smallest
  `max_ctx` (`exact_band`) the router asks the largest backend's `/tokenize` for the exact count (2 s cap).
- **Output cap.** `min_out_tokens` raises a client's small `max_tokens` / `max_completion_tokens`; when the raise
  reaches the context room, the cap is dropped so vLLM allows exactly `max_model_len - prompt`.
- **Loop stop** (`loop_stop`). Streamed responses are followed while they pass through byte for byte. If, over the
  last `words` generated words, the share of distinct `ngram`-word phrases stays below `min_distinct` for `confirm`
  checks, the stream is a runaway repetition: the router ends it at an event boundary (`finish_reason: length`) and
  aborts the request. With `resume`, it first continues the same turn on the same backend from exactly what the client
  already has, at `resume_temperature`, and relays the continuation into the client's stream; a second loop ends it.
  Calibration: on 2,987 sampled outputs of 1,600+ words the lowest normal value was 0.349; a 190k-token runaway fell
  below 0.15 by ~2,800 words. In a replay, 56/56 continuations of sampled loops at T 1.3 ended normally.
- **Health and stalls.** `GET /health` on every backend every 2 s (down after 2 failures). With `spill_kv` or
  `stall_s` set, each engine's `/metrics` is polled every 0.5 s; a backend with requests pending whose generation and
  prompt token counters stay flat for `stall_s` gets no new traffic (not even affinity) until they move.
- **Failover.** Connection errors, or a backend closing before answering, move the request to the next backend in
  cost order. Once response bytes have reached the client there is no retry.
- **Session tags** (`session_tag_ids`, off by default). Requests with a session header (`x-opencode-session-id`,
  `x-session-affinity`, `x-session-id`; parent `x-opencode-parent-session-id`, `x-parent-session-id`; agent
  `x-opencode-agent`, `x-agent`) get `X-Request-Id: arcs<12 hex session><12 hex parent><agent letter>-<random>` for an
  engine-side session hold. The hex parts are a keyed blake2b with a random key chosen at each start; raw ids are never
  logged or forwarded in the tag.
- **Shared KV namespace** (`inject_kv_ns`). Adds `llmt_ns` to the forwarded `kv_transfer_params` (a client's own
  params are kept, its own `llmt_ns` wins) so all GPUs read and write one namespace of a shared host-RAM KV tier
  (AllBroker). Set it to `""` for engines without that connector.
- **Prefix hints** (`prefix_hints`, off by default). With an AllBroker KV broker (protocol ops `HINT` / `WHERE`,
  `docs/prefix-hints.md` there), the router asks the broker, for each request's prefix hashes (newest message boundary
  first), which engine holds how many leading tokens of it in its host-RAM tier right now, and tags the request with
  `kv_transfer_params.llmt_anchor` (its full-prompt hash) so the engine reports what it stored when it finishes. A
  backend's RAM share then comes from the broker instead of the `shared_tier` guess (plus the RAM load time at
  `load_gbps`), and a conversation the router forgot (affinity TTL, restart, weak match) whose prefix one GPU still
  holds is not "new" (no `balance_new`; reason `hint` when the holder wins). Drains, holds, caps, `spill` and `ctx`
  still apply first. Broker missing, slow (`timeout_ms`, 5 ms), or without the feature: routing is exactly as without
  hints. Metrics `llmt_router_hint_where_total{result}`, `llmt_router_hint_where_seconds_total`; the decision log
  gains `hint=<backend>:<tokens>[v]`, the trace a `hint` field (tokens per backend).

- **LoRA adapters** (per-backend `lora`). A request for `<model>:<adapter>` (e.g. `qwen3.8-27b:terse`; separator
  `lora_sep`), or with a body field `"lora": "<adapter>"` or a header `X-LLMT-LoRA: <adapter>` (both survive gateways
  that route on the model name), goes only to backends with `"lora": true`, with `model` set to the adapter name (vLLM
  serves an adapter as a model of that name; with a resolver such as llmt_lora's it is loaded on first use). The
  adapter is part of the prefix-hash root, so base and adapter conversations never share affinity. Names outside
  `[A-Za-z0-9][A-Za-z0-9._-]{0,63}` or conflicting spellings get a 400; no usable LoRA backend gets a 503 (never a
  silent fallback to the base model); an unknown adapter gets the engine's 404. Counted in
  `llmt_router_lora_requests_total{outcome}`.

Hot reload: the config file is re-read when its mtime changes (checked every 2 s). Backends are matched by name;
adding or removing one needs a restart.

## Running it

```
cp router.example.json router.json        # edit backends and measured rates
python3 llmt-router.py --config router.json --decision-log /var/log/llmt-router/decisions.log
```

or without a config file:

```
python3 llmt-router.py --listen 127.0.0.1:8010 \
   --backend name=arc,url=http://127.0.0.1:8000,cap=8,prefill=2000,stream_cost=0.6 \
   --backend name=rtx5090,url=http://127.0.0.1:8001,cap=16,prefill=6000,stream_cost=0.3
```

(`dec=a/b/c` in a `--backend` spec gives per-stream decode rates and turns on the finish-time model.) Without
`--config`, a `router.json` next to the script is used if present.

| Option | Default | Meaning |
|---|---|---|
| `--listen` | `127.0.0.1:8010` | address to listen on (the config's `listen` wins) |
| `--backend` | | `name=..,url=..[,cap=..,prefill=..,dec=a/b/c,...]`, repeatable; ignored when the config has `backends` |
| `--config` | `router.json` next to the script | JSON config, re-read on change |
| `--max-entries` | `200000` | prefix-map size (LRU) |
| `--shared-tier` | `0.0` | start value of `shared_tier` |
| `--chars-per-token` | `3.3` | start value of `chars_per_token` |
| `--decision-log` | `/var/log/llmt-router/decisions.log` | rotating file (20 MB x 5) for one line per routing decision; `''` = stderr only |

Environment: `LLMT_ROUTER_DRAIN_DIR` (default `/run/llmt-router`) holds the drain and hold files.

### Endpoints

- `POST /v1/chat/completions`, `POST /v1/completions`: routed. SSE streams pass through unchanged (chunked framing
  included).
- `GET /health`: 200 if any backend is up, else 503.
- `GET /router/metrics`: Prometheus text. `llmt_router_requests_total{backend,reason}` with reasons `new`, `affinity`,
  `steal`, `affinity-down`, `failover`, `prefer`, `spill`, `balance`, `summary`, `ctx`, `ctx-retry`, `split`,
  `split-prefill`, `none-healthy`, `loop-stop`, `loop-resume`; gauges `llmt_router_inflight`, `_healthy`, `_drained`,
  `_stalled` per backend; `llmt_router_prefix_entries`; counters `llmt_router_backend_stalls_total`,
  `llmt_router_session_tagged_total`, `llmt_router_split_total{result}`, `llmt_router_split_prefill_seconds_total`,
  `llmt_router_stream_{streams,eligible,tool_calls,eligible_seconds}_total{backend}`.
- `GET /router/state`: JSON (per backend: health, drain/hold, in flight, cap, requests by reason, engine waiting /
  running / KV fill; prefix entries, mean completion length, uptime). No request content.
- Anything else goes to the first healthy backend.

### Drain and hold

- `touch $LLMT_ROUTER_DRAIN_DIR/drain-<name>`: no new requests for that backend; running ones finish. It is still the
  last resort when nothing else is usable. Remove the file to undo.
- `hold-<name>` (or `hold-<name without "rtx">`, e.g. `hold-5090`) **together with** `drain-<name>`: the GPU is out
  of service and may run a test engine on the same port; it gets no traffic at all, not even as the fallback (a fast
  502 beats a silent reroute into an experiment). A held backend whose drain file is removed serves again.

## Configuration

Values are those of `router.example.json` (one Arc Pro B70 and one RTX 5090 serving Qwen3.8-27B EXL3 4.0 bpw); the
rates are hardware measurements and need re-measuring on other hardware.

### Top level

| Key | Meaning | Example |
|---|---|---|
| `listen` | address to listen on | `"127.0.0.1:8010"` |
| `backends` | list of backends (next table) | |
| `externality` | weight of the delay a request causes the requests already running on a GPU (0 = selfish placement) | `1.0` |
| `prefill_slowdown` | default fraction of their rate that running streams lose while a prefill runs (per-backend value wins) | `0.4` |
| `shared_tier` | fraction of a cached prefix the *other* backends can load from a shared KV tier | `0.9` |
| `chars_per_token` | prompt estimate: characters per token before calibration | `3.3` (default) |
| `est_scale` | start value of the learned estimate scale (real / estimated prompt tokens) | `1.0` (default) |
| `gen_default` | expected completion tokens until the running mean has data | `400` |
| `return_window_s` | a conversation that finished within this many seconds counts as returning | `20.0` |
| `returning_weight` | a returning conversation counts as this much of a running request | `0.5` |
| `inject_kv_ns` | `llmt_ns` added to `kv_transfer_params` (shared KV-tier namespace); `""` = off | `"shared"` |
| `split` | disaggregated prefill settings (table below) | `{"on": false, ...}` |
| `trace_dir` | directory for the per-request trace and the saved prefix map; `""` = no trace | `""` |
| `spill_kv` | KV fill (0..1) at which an engine counts as saturated for the spill rule; `0` = off | `0.9` |
| `spill_stale_s` | engine numbers older than this are ignored | `3.0` |
| `stall_s` | flat token counters with requests pending for this long = stalled; `0` = off | `90` (default) |
| `ctx_margin` | safety factor on the prompt estimate against `max_ctx` (exact counts skip it) | `1.1` |
| `exact_band` | estimates within `[lo, hi]` x the smallest `max_ctx` are counted exactly via `/tokenize` | `[0.7, 1.4]` |
| `balance_new` | `{backend: weight}`: new conversations go to the fewest active conversations per weight; `{}` = off | `{"rtx5090": 2, "arc": 1}` |
| `balance_skip_tokens` | requests without tools at least this large skip `balance_new` (placed by finish time) | `8192` |
| `summary_target` | backend that gets those large tool-less requests first; `""` = by finish time | `""` |
| `affinity_min_tokens` | a matched prefix shorter than this does not force affinity | `32768` |
| `affinity_min_frac` | nor one shorter than this fraction of the prompt | `0.5` |
| `affinity_ttl_s` | a placement older than this no longer forces affinity (likely evicted from VRAM); `0` = never expires | `600` |
| `min_out_tokens` | raise a smaller client output cap to this (dropped when it reaches the context room); `0` = off | `196608` |
| `session_tag_ids` | session headers -> `X-Request-Id` tag | `false` |
| `lora_sep` | separator of `<model><sep><adapter>`; `""` = suffix off (body field and header still work) | `":"` |
| `loop_stop` | runaway-repetition stop (table below); `{}` = off | `{"words": 1600, ...}` |

### Per backend

| Key | Meaning | Example (arc / rtx5090) |
|---|---|---|
| `name` | label in metrics, drain/hold file names, `balance_new`, `summary_target` | `"arc"` / `"rtx5090"` |
| `url` | engine base URL (http only) | `"http://127.0.0.1:8000"` / `"http://127.0.0.1:8001"` |
| `cap` | in-flight requests before the 30 s per request over-cap penalty | `16` / `6` |
| `prefill` | prefill tok/s for the uncached part of a prompt | `2800` / `6500` |
| `dec` | per-stream decode tok/s at 1, 2, ... running requests (aggregate flat beyond the list); turns on the finish-time model | `[83, 74, 63, 57, 48, 41]` / `[245, 203, 139, 137, 123, 110, 98, 86]` |
| `stream_cost` | seconds of delay per in-flight request (only without `dec`) | `0.5` (default) |
| `prefill_slowdown` | this GPU's value of the top-level key | `0.65` / `0.65` |
| `load_gbps` | split mode: KV-tier load bandwidth into this GPU, GB/s | `2.5` / `2.0` |
| `max_ctx` | the engine's `max_model_len`; `0` = unknown | `196608` |
| `max_images` | the engine's image limit per prompt; `0` = unknown | `32` |
| `lora` | the engine serves LoRA adapters by name (requests for an adapter go only to such backends) | `false` |
| `prefer_until` | every request goes here while it has fewer than this many in flight; `0` = off | `0` / `2` |

Why `prefer_until`: with the cost model alone, the Arc served 29% of requests but 52% of agent wait time. Per stream
it decodes 90/67/53/50/41/37/32/24 tok/s with 0..7 other streams against the 5090's 219/170/120/98/89/~70, and the
5090's 8-stream aggregate (851 tok/s) is 4.4x the Arc's. The cost model is per request and cannot see that a request
sent to the slower GPU is slowed again by the next arrivals there.

### `split` (disaggregated prefill, off)

When the decode GPU D has requests decoding, a prefill-only copy (`max_tokens` 1, same namespace) goes to another GPU
P, whose KV tier stores the prompt; then D loads it from the tier and recomputes only the tail. Taken when the stall
it saves D's running streams exceeds the extra delay for this request. Needs `inject_kv_ns`; skipped for images.

| Key | Meaning | Example |
|---|---|---|
| `on` | enable | `false` |
| `min_tokens` | minimum uncached tokens on P | `2048` |
| `min_running` | minimum requests decoding on D | `1` |
| `kv_bytes_per_token` | KV + recurrent-state bytes per token moved through the tier | `41000` |
| `tail_tokens` | tokens D recomputes itself | `832` |
| `overhead_s`, `store_wait_s` | fixed overhead; wait for the stores to land before D looks them up | `0.3`, `0.3` |
| `weight` | required ratio of saved stall to added delay | `1.0` |
| `max_prefill_running` | P may have at most this many requests running | `99` (default) |

### `prefix_hints` (off by default)

| Key | Meaning | Default |
|---|---|---|
| `on` | ask the broker (WHERE) and inject `llmt_anchor` | `false` |
| `socket` | the broker's Unix socket | `"/run/llmt-kv/broker.sock"` |
| `engines` | backend name -> the engine's broker HELLO name (string) or engine id (int); unlisted = the backend name | `{}` |
| `timeout_ms` | per WHERE; a miss routes without hints | `5` |
| `max_anchors` | prefix hashes asked per request (newest first) | `64` |
| `inject_anchor` | add `llmt_anchor` to `kv_transfer_params` (a client's own wins) | `true` |
| `min_tokens` | RAM-resident tokens that make a forgotten conversation "not new" (also `>= affinity_min_tokens` and `affinity_min_frac` x prompt) | `4096` |
| `vram_ttl_s` | a hint younger than this from the engine that just served it counts as VRAM-recent (logged only) | `120` |

### `loop_stop`

| Key | Meaning | Example |
|---|---|---|
| `words` | window of generated words (reasoning, content, tool arguments) | `1600` |
| `ngram` | phrase length in words | `6` |
| `min_distinct` | distinct-phrase share below which a check counts as looping | `0.15` |
| `every` | words between checks | `200` |
| `confirm` | consecutive low checks before acting | `2` |
| `resume` | continue the turn once at a higher temperature instead of ending it | `true` |
| `resume_temperature` | temperature of that continuation | `1.3` |
| `test_force_after_words` | tests only: treat as a loop after N words | `0` |

### `session_policy` and the policy file (off by default)

Clients that tag their requests get session-aware placement and cache classes; untagged requests are routed exactly
as without the policy, and nothing new is sent downstream while `session_policy.on` is false.

Request headers (all optional): `X-Session-Id` (one per conversation), `X-Parent-Session-Id` (a child conversation,
e.g. a delegated task), `X-Agent` (role: `main`, `summary`, `compaction`, ...), `X-Agent-Name` (a slug
`[a-z0-9][a-z0-9_.:-]*`, at most 64 characters) and `X-Priority` (a tier name).

What it does with them:
- **Tier**: `X-Priority`, else `agent_priority[name]`, else the class learned from that agent's traffic (single
  stream, no follow-ups, little prefix reuse = `low`), else `default_tier`. A tier has a `rank` (0 = most urgent), a
  `vllm_priority` (sent as vLLM's `priority` field only to backends with `"priority_sched": true`, i.e. engines run
  with `--scheduling-policy priority`), a `cache` class for the RAM tier (`normal` / `low` / `lowest`) and
  `background` (never takes a GPU that carries more agent work than another; waits up to `low_wait_s` while every GPU
  is saturated).
- **Rule** (first match on agent-name globs and roles): `retention` (`session`: the conversation's chain stays
  `normal` until a `terminal_roles` request; `group`; `oneshot`: its new blocks are stored `low`; `lowest`), `place`
  (`session` affinity always first, then `group` or `parent` replica) and `group` (`name` or `name+session`).
- **Downstream**: `kv_transfer_params` gets `llmt_cls`, `llmt_sess` / `llmt_parent` (64-bit keyed hashes, never the
  raw ids), `llmt_agent` (role letter) and `llmt_term`. AllBroker's `llmt_kv` connector turns these into the broker's
  value classes and session tags (`ENTRY_LOW`, `TOUCH_TAGGED`).

Where the policy lives: router.json `session_policy` (`{"on": true}` plus any keys below) and a separate policy file,
re-read when it changes, whose keys win: `--policy-file` / `LLMT_ROUTER_POLICY`, else router.json `policy_file`. Start
from [router-policy.example.json](router-policy.example.json) (fictional agent names). Keep your real policy outside
the repository: agent names can be business data.

| Key | Meaning | Default |
|---|---|---|
| `on` | enable (router.json `session_policy` only) | `false` |
| `tiers` | tier name -> `{rank, vllm_priority, cache, background}` | `high` / `normal` / `low` / `background` |
| `default_tier` | tier of a request without `X-Priority`, map entry or learned class | `normal` |
| `agent_priority` | agent name or `role:<X-Agent>` -> tier | `{}` |
| `rules`, `default_rule` | see above | `[]`, `{"retention": "session", "place": "parent"}` |
| `learn`, `learn_tier`, `learn_min`, `learn_window`, `learn_follow_max`, `learn_reuse_max`, `learn_conc_max` | learned classes | `true`, `low`, 20, 200, 0.1, 0.6, 0.5 |
| `session_ttl_s`, `group_ttl_s` | how long a session / group placement is remembered | 1800, 600 |
| `parent_affinity` | children go to the parent's replica | `true` |
| `low_wait_s`, `low_kv`, `low_target` | background wait limit, KV share that counts as busy (default `spill_kv` or 0.9), preferred GPU for background work | 0, -, `""` |
| `agents_log_s` | period of the per-agent summary log line | 600 |

Metrics: `llmt_router_class_requests_total{tier,cls,source}`, `llmt_router_background_waits_total`,
`llmt_router_background_inflight`, `llmt_router_sessions`, `llmt_router_groups` and per agent key
`llmt_router_agent_{requests_total,followup_ratio,reuse_ratio,concurrency,learned_low}` (labels sanitized, at most 64
keys, the rest `other`).

Power boost for priority tiers: the router does not change GPU power itself. On an Intel Arc Pro B70,
[echowarp/b70-linux](https://github.com/echowarp/b70-linux) `docs/priority-boost.md` describes a trigger that runs
the card at full power while interactive requests are in flight and at its tokens-per-joule level otherwise; map
your top tiers (e.g. `high` / `normal`) to its interactive class, or put its ASGI middleware in front of the engine.

## Privacy

Request content (prompts, completions, tool arguments) is never logged and never written to disk.

- Kept in memory beyond a request: 16-byte blake2b hashes of message prefixes, the backend each was sent to, token
  estimates, timestamps, and counters.
- Kept in memory only while a request runs: the request body (to parse and forward it) and, with `loop_stop`, the
  last `words` words of that stream (with `resume`, the reply text so far, to continue from it).
- Written to disk: the decision log (backend, reason, cost estimates, token counts, health), `prefix-map.txt` every
  30 s (hash, backend, token count, time for placements younger than `affinity_ttl_s`, else 1 h; reloaded at start;
  in `trace_dir`, else `/var/lib/llmt-router`), and the trace if `trace_dir` is set.

**Trace** (`trace-YYYYMMDD.jsonl`, one JSON line per routed request, for replaying a load's pacing and order):
arrival wall-clock time `t`, `path`, route `reason`, `prompt_est` (estimated prompt tokens), `msgs` (message count),
`chain` (prefix-chain length), `conv` (the first three prefix hashes, 12 hex each, to group the requests of one
conversation), `max_tokens`, `stream`, `backend`, HTTP `code`, `first_s` / `done_s` (first byte and completion,
seconds after arrival), `failover` (backend index), `completion_tokens`, `prompt_tokens` (from the response's usage),
`est_scale`, `hint` (with `prefix_hints`: RAM-resident prompt tokens per backend), `tool_calls` (count) and `finish` (finish reason, or `loop` / `loop-resumed:*`). No text.

With `prefix_hints`, the same 16-byte prefix hashes also go to the local KV broker (WHERE) and to the engine
(`llmt_anchor` in `kv_transfer_params`), which hands them back to the broker; no text.

With `session_policy`, agent names and the tier counts appear in local metrics, `/router/state` and the log
(sanitized, capped); raw session ids never leave the router, and engines see only keyed hashes.

The prefix hashes are one-way but unkeyed: someone who holds a trace or prefix-map file *and* a candidate prompt can
check whether that exact prompt was routed. Treat these files as operational logs (root-only, short retention).

## Tests

No GPU and no captured data: the tests start fake backends on 127.0.0.1 (fixed ports 18301-18341, `test_router.py`
uses ephemeral ports) with synthetic prompts.

```
for t in tests/test_*.py; do python3 "$t" || echo "FAILED: $t"; done
```

| Test | Covers |
|---|---|
| `test_router.py` | affinity, drain, spread under load, steal, failover, SSE passthrough, `prefer_until`, KV namespace injection, split mode, the stream tracker, metrics, hold flags, trace fields, no content in state or trace |
| `test_estimate.py` | prompt estimate (messages + tools), calibration, exact count via `/tokenize` |
| `test_maxctx.py` | `max_ctx` / `max_images`, context-length retry, `balance_new`, `affinity_min_*`, `affinity_ttl_s`, `summary_target` |
| `test_minout.py` | `min_out_tokens` raise and cap drop |
| `test_sesstag.py` | session tags |
| `test_spill.py` | spill rule |
| `test_prefix_hints.py` | prefix hints against a fake broker: off by default, broker missing / hung / old = old routing, RAM holder placement, drains and caps, anchor injection, WHERE decoding |
| `test_stalldetect.py` | stall detection |
| `test_lora.py` | LoRA adapter selection (suffix / field / header), routing to `lora` backends only, 400 / 503 / 404 |
| `test_sessprio.py` | session + priority policy: off = old routing, session / group / parent placement, tiers, background placement and wait, learned classes, policy file reload, what goes downstream (hashes and classes only) |

Loop stop, by hand (`fake_loop_backend.py` streams a repetition when the prompt contains "loop"):

```
python3 tests/fake_loop_backend.py 18401 /tmp/fake.log &
cat > /tmp/loop-router.json <<'EOF'
{"listen": "127.0.0.1:18010", "trace_dir": "/tmp/loop-trace",
 "backends": [{"name": "fake", "url": "http://127.0.0.1:18401"}],
 "loop_stop": {"words": 1600, "ngram": 6, "min_distinct": 0.15, "every": 200, "confirm": 2}}
EOF
LLMT_ROUTER_DRAIN_DIR=/tmp python3 llmt-router.py --config /tmp/loop-router.json --decision-log '' &
python3 tests/loop_stream_client.py '{"model":"m","stream":true,"messages":[{"role":"user","content":"loop"}]}'
# -> finish ["length"] after ~500 events, and /tmp/fake.log shows client_disconnected: true
```

## License

MIT, see [../LICENSE](../LICENSE).
