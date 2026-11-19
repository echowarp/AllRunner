# Launchers

Scripts that start the AllRunner serving stack: one Qwen3.8-27B engine per GPU, a shared host-RAM KV tier, and the
prefix-affinity router in front of them. Each script reproduces the configuration that served production traffic
(agent workloads: tool calls, 20k+ token prompts, several agents at once).

> **Status: drafts.** The flags and environment values are the production ones, but the production engines were
> started by site-specific scripts; the launchers have only been syntax-checked (`bash -n`). Both images in
> [`../docker/`](../docker/) build as written and import cleanly, but no engine has been started from them yet. See
> [Untested](#untested) before relying on them.

## How the pieces fit

```
clients --(API key)--> gateway (optional, e.g. llama-swap) :1234
                          |
                          v
                     llmt-router 127.0.0.1:8010 ----> Arc B70 engine   127.0.0.1:8000  (launchers/arc-b70.sh)
                     (router.sh)               \---> RTX 5090 engine  127.0.0.1:8001  (launchers/rtx5090.sh)
                                                         \                 /
                                                          llmt-kvbroker (kvbroker.sh)
                                                          16 GiB pool in /dev/shm, socket /run/llmt-kv-i4/broker.sock
```

- Both engines serve the same model under the same name (`qwen3.8-27b`): EXL3 4.0 bpw weights, int4 KV cache,
  MTP speculative decoding with k = 7 and a per-batch-size depth schedule, 196,608-token context, up to 32 images per
  prompt.
- The broker holds KV blocks and GDN states that fell out of VRAM. Each engine has its own namespace; requests that
  carry `llmt_ns: "shared"` (the router adds it, `inject_kv_ns`) read and write one namespace that both GPUs use, so a
  conversation can move between GPUs without re-prefilling. Engines also start and serve without a broker (the tier
  turns itself off after 30 s).
- Nothing in this stack authenticates clients. Engines, broker and router listen on 127.0.0.1 only; put a gateway
  that holds the API key in front if anything beyond localhost needs access.

| Script | What | Implemented in |
|---|---|---|
| `kvbroker.sh` | host-RAM KV tier, production settings | [echowarp/AllBroker](https://github.com/echowarp/AllBroker) (`llmt-kvbroker`) |
| `rtx5090.sh` | RTX 5090 engine (container, port 8001) | [echowarp/vllm](https://github.com/echowarp/vllm) `main`, [echowarp/exllamav3](https://github.com/echowarp/exllamav3) `master` (`exl3cuda/`), AllBroker `llmt_kv` |
| `arc-b70.sh` | Arc Pro B70 engine (container, port 8000) | echowarp/vllm `xpu`, [echowarp/exl3xpu](https://github.com/echowarp/exl3xpu) `main`, AllBroker `llmt_kv` |
| `llmt-warmup.py` | started by both engine launchers (`WARMUP=1`, default): warms the first-use shapes after a start, router drain gate (section "Warm-up") | |
| `router.sh` | llmt-router on 127.0.0.1:8010 | [`../router/`](../router/) |
| `common.sh` | hash-seed check and helpers (sourced) | |

Every site-specific value (model directory, image name, ports, broker socket and pool, cache volume, data files) is
a variable at the top of each script with a comment; override it from the environment, e.g.
`MODEL_DIR=/data/models/qwen KSMOOTH_TABLE=/etc/allrunner/ksmooth.pt launchers/arc-b70.sh`.

## Before the first start

1. **Weights:** `turboderp/Qwen3.8-27B-exl3`, branch `4.00bpw`, in `MODEL_DIR` (both engines mount it at `/models`;
   the KV tier's compatibility tag includes a hash of the model's config files, so both must use the same files).
2. **Images:** `docker build -f docker/Dockerfile.rtx5090 -t allrunner-rtx5090 .` and
   `docker build -f docker/Dockerfile.arc -t allrunner-arc .` (from the repo root; see the files for memory needs).
3. **Broker:** `cargo build --release` in AllBroker, install `target/release/llmt-kvbroker` (default path in
   `kvbroker.sh`: `/usr/local/bin/llmt-kvbroker`).
4. **PYTHONHASHSEED** (required): see below.
5. **Router config:** `cp router/router.example.json /etc/allrunner/router.json` and edit it (`router/README.md`).
6. Optional data files (K-smoothing table, draft vocabularies, draft head): see [Data files](#data-files-you-generate).

### PYTHONHASHSEED

vLLM's prefix-cache block hashes depend on Python's hash seed. Every engine that shares a broker must use the same
value, and it must stay the same across restarts: otherwise blocks in the RAM tier are never found again (no errors,
only misses). Production treats it as a secret (whoever knows it can compute the block keys of a prompt). The engine
launchers read it from `HASHSEED_FILE` (default `/etc/allrunner/hashseed.env`) and **refuse to start** without it;
there is no default value anywhere in this repository. Create it once:

```sh
sudo install -d -m 0755 /etc/allrunner
echo "PYTHONHASHSEED=$(python3 -c 'import secrets; print(secrets.randbelow(2**32 - 1) + 1)')" \
  | sudo install -m 0600 /dev/stdin /etc/allrunner/hashseed.env
```

The launcher passes it to docker by name (`-e PYTHONHASHSEED`), so the value does not show up in `ps` or
`docker inspect`'s command line (it is in the container's environment).

## Start order

1. `kvbroker.sh` (as root; under systemd: AllBroker's `contrib/systemd/llmt-kvbroker.service` with `ExecStart`
   pointing at this script, `Restart=on-failure`, `Before=` your engine units). Start it before the engines: an engine
   that finds no broker within 30 s serves without the RAM tier until it is restarted.
2. One engine, then the other: wait for `curl -s 127.0.0.1:8000/health` (Arc) or `:8001` (5090) before starting the
   second. Each start loads weights, compiles and captures graphs (minutes); staggering also keeps the two cards
   from peaking together (see [Hardware notes](#hardware-notes)). Before restarting an engine, stop the old container
   first (`docker stop -t 30 allrunner-rtx5090`); the 5090 launcher waits up to 30 s for the GPU memory to be free,
   because vLLM sizes the KV cache from the free memory at start.
3. `router.sh`.
4. The gateway (optional).

Example llama-swap entries (llama-swap holds the API key; see its documentation for the key setting):

```yaml
models:
  "qwen3.8-27b":
    cmd: /opt/allrunner/launchers/router.sh
    proxy: http://127.0.0.1:8010
    checkEndpoint: /health
    ttl: 0
  "qwen3.8-27b-arc":                      # direct access to one engine (optional)
    cmd: /opt/allrunner/launchers/arc-b70.sh
    cmdStop: docker stop -t 30 allrunner-arc
    proxy: http://127.0.0.1:8000
    checkEndpoint: /health
    useModelName: "qwen3.8-27b"
    ttl: 0
  "qwen3.8-27b-5090":
    cmd: /opt/allrunner/launchers/rtx5090.sh
    cmdStop: docker stop -t 30 allrunner-rtx5090
    proxy: http://127.0.0.1:8001
    checkEndpoint: /health
    useModelName: "qwen3.8-27b"
    ttl: 0
```

Production keeps all three loaded at once (a llama-swap group with `persistent: true`, `swap: false`,
`exclusive: false`). Any other OpenAI-compatible gateway or reverse proxy works the same way: send the public model
name to the router.

## RTX 5090 engine (`rtx5090.sh`)

Repo column: **vLLM** = echowarp/vllm `main`, **exl3cuda** = echowarp/exllamav3 `exl3cuda/`,
**llmt_kv** = AllBroker's Python client, **stock** = upstream vLLM 0.26.

### Engine flags

| Flag | Value | What it does | Repo |
|---|---|---|---|
| `--dtype` | float16 | activations | stock |
| `--max-model-len` | 196608 | context limit (`MAX_MODEL_LEN`) | stock |
| `--max-num-seqs` | 32 (`MAX_NUM_SEQS`) | concurrent requests; above 16 the depth schedule drops to k = 1 so a 32-stream step stays at 64 rows (c32 1,325 vs 1,237 tok/s with 16 seqs, TTFT 0.88 vs 1.26 s) | stock |
| `--max-num-batched-tokens` | 8192 | tokens per step (prefill chunk) | stock |
| `--gpu-memory-utilization` | 0.86 (`GPU_MEM_UTIL`) | VRAM share; the GDN-recompute pools and the CUDA graphs (1.33 GiB at <= 1,024 tokens) live outside vLLM's profile: 0.88 ran out of memory in a burst of 16 concurrent ~6k-token prompts, 0.86 passed 2 x 32 | stock |
| `--enable-prefix-caching`, `--mamba-cache-mode align` | | prefix caching for the hybrid (GDN + attention) model; vLLM leaves it off for hybrids by default | stock |
| `--kv-cache-dtype` | int4_per_token_head | int4 KV, per token and head scales (about 2x the context of int8 KV) | stock (kernels: exl3cuda VA4, KVW4, I4FA) |
| `--attention-backend` | TRITON_ATTN | backend whose calls exl3cuda replaces with its own kernels | stock |
| `--speculative-config` | `qwen3_5_mtp`, k = 7, `num_speculative_tokens_per_batch_size [[1,2,7],[3,16,3],[17,32,1]]`, TRITON_ATTN, `draft_sample_method probabilistic`, `rejection_sample_method standard` | MTP drafting from the checkpoint's MTP layer: 7 drafts with 1-2 requests running, 3 with 3-16, 1 with 17-32 | stock fields |
| `--compilation-config` | `max_cudagraph_capture_size` 1024 | CUDA graphs up to 1,024 tokens per step; vLLM's derived default (256 = 16 seqs x 8 x 2) left multi-stream mixed steps (~280 tokens) and prefills eager (~1,900 launches per step). Capture 36 s, 1.33 GiB, KV cache unchanged | stock |
| `--limit-mm-per-prompt` | image 32, video 0 | | stock |
| `--structured-outputs-config` | xgrammar, `disable_any_whitespace` | compact JSON (avoids whitespace loops) | stock |
| `--reasoning-parser`, `--tool-call-parser`, `--enable-auto-tool-choice` | qwen3, qwen3_coder | | stock |
| `--kv-transfer-config` | OffloadingConnector + `BrokerOffloadingSpec` from `llmt_kv.cuda_tier`, socket `BROKER_SOCK`, namespace `rtx5090-private`, `broker_engine_id` 2, `broker_name` rtx5090, `broker_timeout_ms` 5, `broker_ns_allow ["shared"]`, `broker_lease_frac` 0.75 (`BROKER_LEASE_FRAC`; llmt_kv default 0.25), `cpu_bytes_to_use` 1 (size comes from the broker) | shared host-RAM KV tier; the lease share caps how many tier slots this engine may be loading from at once (0.25 let ~4 agents resume 22k-token contexts from RAM at a time) | llmt_kv + vLLM |
| docker | `--shm-size 16g`, `--ulimit memlock=-1:-1` (slot pinning), CDI GPU device, port `127.0.0.1:8001` | | |

### Environment

| Variable | Value | What it does | Repo |
|---|---|---|---|
| `PYTHONHASHSEED` | from `HASHSEED_FILE` | stable block hashes (see above) | |
| `HF_HUB_OFFLINE` | 1 | never contact the Hub | stock |
| `TRITON_CACHE_DIR` | /root/.cache/triton | Triton kernels survive restarts (else they JIT-compile under traffic for minutes) | stock |
| `VLLM_CACHE_ROOT` | /root/.cache/vllm-cc/<key> | torch.compile cache keyed by image id + flags + environment | stock |
| `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS` | 0 | vLLM's graph-memory estimate was 6.1 GiB vs 0.76 actual: +12 % KV | stock |
| `VLLM_KV_CACHE_LAYOUT` | NHD | the Arc's page layout, so RAM-tier entries are interchangeable | stock |
| `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` | 12800 | keep GDN states for the prefix cache every 4 blocks (3,200 tokens each) | vLLM |
| `EXL3_INT8_PREFILL` | 1 | int8 tensor-core GEMMs for prefill rows | exl3cuda |
| `EXL3_INT8_L2SPLIT` | 1 | bit-identical trellis decode into L2-sized int8 slices | exl3cuda |
| `EXL3_FUSE_DECODE`, `EXL3_FUSE_MAX_M`, `EXL3_FUSE_SCRATCH` | 1, 64, 1 | fused modules (gate+up, q+k+v, ...) as one mgemm up to 64 rows; one shared scratch | exl3cuda |
| `EXL3_GDN_RECOMPUTE`, `EXL3_RC_FUSED_IN`, `EXL3_RC_SLICE` | 1, 1, 1 | MTP with one committed GDN state per request (more KV, no per-draft state pages); fused recompute input; views instead of gathers in mixed steps | exl3cuda (+ vLLM hooks) |
| `EXL3_RC_PINNED` | 1 | recompute metadata through pinned buffers (no host stalls) | vLLM |
| `EXL3_GDN_PORT`, `EXL3_GDN_FAST` | 1, 1 | GDN prefill via the Magnitude WY kernel; one launch for decode GDN metadata | exl3cuda |
| `EXL3_FA16` | f | FA2 prefill build with fp16-accumulate MMAs | exl3cuda |
| `EXL3_I8FA`, `EXL3_I8FA_PAGED_MAXQ` | 1, auto | int8-KV prefill kernels (loaded; unused with int4 KV) | exl3cuda |
| `EXL3_VA`, `EXL3_VA4_OP`, `EXL3_VA4_SO` | 1, va4c, /opt/exl3cuda-lib/va4c.so | custom verify/decode attention; VA4 = the int4-KV kernel (the code default names an older variant, so both are set) | exl3cuda |
| `EXL3_I4PLAN` | default (1) | int4 mixed steps build the attention row plan once per step without host syncs (was ~96 syncs per mixed step) | exl3cuda |
| `EXL3_VERIFY_Q` | 8 | query rows per request handled as decode/verify (k + 1) | exl3cuda |
| `EXL3_TAIL_FUSE` | 1 | bit-exact fusion of the decode step's small-kernel tail | exl3cuda |
| `EXL3_DYN_SD_FULL` | 2 | FULL CUDA graphs for every depth of the MTP schedule | exl3cuda |
| `EXL3_LAZY_VISION`, `EXL3_LAZY_VISION_NS` | 1, `weights:qwen3.8-27b-visual` | vision tower loaded only for image requests (more KV); its host copy is a broker entry shared by both GPUs | llmt_kv |
| `EXL3_LEND_ONLY` | 1 (`VISION_LEND=1`) | KV-block lending for the vision tower (no LoRA on this GPU): the tower lives in lent KV blocks while images are in flight | llmt_kv |
| `enable_cross_layers_blocks` (kv_connector_extra_config) | `"true"` (`PACKED_KV=1`, with the RAM tier) | packed KV layout: one page per block across layers. With lazy vision and lending: KV 436,585 -> 482,846 tokens | echowarp/vllm |
| `EXL3_WHX`, `EXL3_WHX_DIR`, `EXL3_WHX_EXCLUSIVE` | keeper socket, side-file dir, 1 (only when `KEEPER_SOCK` exists) | weight keeper attach: weights stay on the GPU across engine restarts | llmt_kv |
| `EXL3_KSMOOTH` | /data/ksmooth.pt (default `data/ksmooth/public_r.pt`; `KSMOOTH_TABLE=none`: off) | K smoothing (store K - mean) in the quantized KV cache | exl3cuda |
| `EXL3_DRAFT_VOCAB`, `EXL3_DRAFT_VOCAB_FAR`, `EXL3_DRAFT_FAR_FROM` | /data/draft-vocab.json (both; default `data/draft-vocab/public-512.json`), 2 | pruned MTP draft head (512 blocks of 128 tokens); same file near and far = one head copy. `DRAFT_VOCAB=none`: full-head drafting | exl3cuda |
| `EXL3_DRAFT_DYN`, `EXL3_DRAFT_DYN_NEAR`, `EXL3_DRAFT_BIAS` | 256, 128, 0 (with a draft vocab, i.e. by default) | per-request extra draft-head rows from the prompt's frequent tokens | exl3cuda |
| `LLMT_DYNROW` | default (1) | with DRAFT-DYN, the draft row is written by TAIL-FUSE's scatter plus one Triton kernel for the extra rows (was ~8 small ops per draft pass) | exl3cuda |
| `LLMT_FASTLAUNCH` | unset (0) | optional, not in the production configuration: cached Triton launches for the draft-loop kernels (bit-identical; single stream +0-1.9 % steps/s) | exl3cuda |
| `EXL3_DRAFT_WEIGHTS` | /data/draft-head.safetensors if `DRAFT_WEIGHTS` is set | 4-bit re-quantized draft head rows | exl3cuda |
| `EXL3_DRAFT_GRAPH` | 1 | draft passes 2..k as one captured CUDA graph (eager above 4 streams or on any capture failure); greedy output bit-identical to eager. Measured on an RTX 5090 against the same build without it: step rate +14 % at 1 stream, +10 % at 2, +1 % at 4, +11 % at 1 stream with 20k context | exl3cuda |
| `EXL3_KV_DTYPE_TAG` | int4_per_token_head | KV dtype in the RAM tier's compatibility tag | llmt_kv |
| `EXL3_ATTN_VERIFY_BM`, `_WARPS`, `EXL3_ATTN_SEGMENTS`, `_MANY` | 32, 2, 64, 32 | Triton verify-attention tiling (fallback path) | vLLM |
| `EXL3_FI_TOPKP` | 1 | FlashInfer top-k/top-p for sampling and verification | vLLM |
| `EXL3_ALIGN_NOSYNC` | 1 | no accepted-count host sync on quiet align-mode steps | vLLM |
| `EXL3_ALIGN_NOSYNC2` | default (1) | also no accepted-count sync on steps that add, finish or reorder requests, while every continuing speculative request stays inside its Mamba block | vLLM |
| `EXL3_SPEC_STOP_TRUNC` | 1 | drop draft tokens after a stop token on the GPU | vLLM |
| `EXL3_EAGLE_KEEP` | 2 | keep the full prefix hit with MTP drafters | vLLM |
| `EXL3_TAIL_HOLD` (+ `_S` 60, `_TOOL_S` 300, `_FREE` 0.03, `_MAX` 16, `_DECAY_S` 60) | 1 | keep a finished turn's partial tail block for the agent's next turn (60 s, 300 s after a tool call; released below 3 % free KV; at most 16 holds) | vLLM |
| `EXL3_SCHED_DECODE_STEPS` | 1 | one decode-only step after each mixed prefill step | vLLM |
| `EXL3_SCHED_MIXED_K` | -1 | no cap on drafts in mixed steps | vLLM |
| `EXL3_DYN_SD_DECODE_COUNT` | 1 | key the depth schedule on decoding requests only | vLLM |
| `EXL3_GPU_EVICT_ORDER` | chain | evict a finished session deepest position first across KV groups | vLLM |
| `EXL3_MAMBA_KEEP_LAST` | 2 | keep the last 2 state boundaries under sparse retention | vLLM |
| `EXL3_ENCODER_CACHE_TOKENS` | 16384 | own encoder cache budget (multi-image livelock fix) | vLLM |
| `EXL3_STATE_STORE` | prompt_end | store GDN states in the RAM tier only at the prompt end (43 % fewer tier writes, same hit rate) | vLLM |
| `EXL3_KV_COPY_PACE_MB` | 8 | RAM-tier copies in <= 8 MB sub-batches so decode steps don't wait behind them | vLLM |
| `EXL3_KV_NANCHECK` | 1 | scan tier stores for NaN/inf before they can serve hits | vLLM |
| `EXL3_DYN_PIN`, `LLMT_KV_PIN_NAME`, `LLMT_KV_ENGINE_ID` | 1, rtx5090-pin, 2 | pin the broker pool per slot and follow its resizes | vLLM + llmt_kv |

Defaults that production relies on without setting them: `EXL3_NOMMAP_LOAD` (weights read without mmap, via
`llmt_kv.nommap_loader`), `EXL3_KVW4`, `EXL3_VA4`, `EXL3_I4FA`, `EXL3_KSMOOTH_INT4`, `EXL3_UNIFORM_SPEC`, `EXL3_MIX3D`
(exl3cuda), and in vLLM the stall guard, NaN recompute, length cap and structured-output fixes (all default on).

## Arc Pro B70 engine (`arc-b70.sh`)

The image's entrypoint is exl3xpu's `scripts/serve.py`, which turns the checkpoint config
`models/qwen3.8-27b-exl3-4.00bpw/model.yaml` into a `vllm serve` command; the launcher overrides it with
`--set vllm.<field>=` and `--set env.<VAR>=` (serve.py applies model.yaml's `env:` over the process environment, so
engine variables must go through `--set env.`). Repo column: **vLLM** = echowarp/vllm `xpu`,
**exl3xpu** = echowarp/exl3xpu `main`.

### Engine settings

| Setting | Value | What it does | Repo |
|---|---|---|---|
| `max_model_len` | 196608 | context limit | stock |
| `gpu_memory_utilization` | 0.94 | OOR fix part 1: headroom for the Level Zero driver's own allocations | stock |
| `max_num_batched_tokens` | 8192 | step budget (= encoder cache budget; 4096 livelocked a 4-image prompt) | stock |
| `kv_cache_dtype` | int4_per_token_head | int4 KV | stock (kernels: exl3xpu) |
| `attention_backend`, `speculative_config.attention_backend` | TRITON_ATTN | backend whose forward exl3xpu replaces | stock |
| `speculative_config` | k = 7, `num_speculative_tokens_per_batch_size [[1,4,7],[5,5,5],[6,6,4],[7,8,3],[9,256,2]]`, `draft_sample_method probabilistic` | MTP depth by running requests (row budget ~32 per step) | stock fields |
| `structured_outputs_config`, `limit_mm_per_prompt`, parsers | as on the 5090 | | stock |
| `kv_transfer_config` | OffloadingConnector + `BrokerOffloadingSpec` from `llmt_kv.xpu_tier`, namespace `arc-b70-i4`, `broker_lease_frac` 0.75 (`BROKER_LEASE_FRAC`; code default 0.25), `broker_ns_allow ["shared"]` | shared host-RAM KV tier (pageable copies; `LLMT_KV_PIN=1` would import the pool into Level Zero, off in production) | llmt_kv + vLLM |
| from the image's model.yaml | dtype float16, max_num_seqs 16, prefix caching, mamba_cache_mode align, cudagraph_mode FULL_DECODE_ONLY, capture sizes 1-64, image size cap; env `VLLM_XPU_ENABLE_XPU_GRAPH=1`, `EXL3_KV_BLOCK_EXACT=1`, `EXL3_ONEDNN_ATTN=1`, spawn workers | | exl3xpu |
| docker | `--device /dev/dri`, `/dev/dri/by-path` (ro), `--shm-size 32g`, port `127.0.0.1:8000` | | |

### Environment

| Variable | Value | What it does | Repo |
|---|---|---|---|
| `PYTHONHASHSEED` | from `HASHSEED_FILE` | stable block hashes | |
| `HF_HUB_OFFLINE`, `ZES_ENABLE_SYSMAN` | 1, 1 | offline; Level Zero sysman (device memory queries) | stock / Intel |
| `TORCH_CAP_MIB` | 29100 | OOR fix part 2: cap torch's XPU caching allocator below the card's 32 GB | exl3xpu `tools/oor` |
| `MEMPROBE_S` | 30 | allocator numbers to stderr every 30 s (numbers only) | exl3xpu `tools/oor` |
| `PYTORCH_ALLOC_CONF` | max_split_size_mb:512,garbage_collection_threshold:0.85 | OOR fix part 3 (all three parts are needed) | torch |
| `VLLM_KV_CACHE_LAYOUT` | NHD | page layout of the Arc kernels (and of the shared tier) | stock |
| `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` | 12800 | as on the 5090 | vLLM |
| `VLLM_XPU_USE_SAMPLER_KERNEL` | 0 (image) | vLLM's XPU sampler kernel off (exl3xpu's fused top-k/top-p is used) | Intel vLLM |
| `EXL3_SHAPE_P_FILE`, `EXL3_MAX_MB` | the fork's B70 split-K table, 128 | decode-GEMM split-K per shape; MB96/128 tiles | exl3xpu |
| `EXL3_INT8_PREFILL` | 1 | W8A8 prefill linears | exl3xpu |
| `EXL3_SPEC_HP`, `EXL3_SPEC_HP_BMAX` | 1, 8 | HEAD-PACK for the drafter's 1-row attention calls, up to 8 sequences | exl3xpu |
| `EXL3_SPEC_VARR_MAXL` | 84000 | VARR verify-kernel variant only up to 84k-token sequences (without the gate, a GPU fault on a ~120k request with other streams decoding was traced to it as the suspect) | exl3xpu |
| `EXL3_I8_SPLITS` | 80,40,64,40 | decode-attention split count by request count | exl3xpu |
| `EXL3_I8_ONEDNN_MAXLEN` | 60000 | above 60k keys, prefill attention in bounded chunks instead of one oneDNN SDPA call (needed for 192k) | exl3xpu |
| `EXL3_GDN_RECOMPUTE`, `EXL3_RC_FUSED_IN`, `EXL3_GDN_RPS` | 1, 1, 4 | GDN recompute (one committed state per request), fused input, rows per sub-group | exl3xpu (+ vLLM hooks) |
| `EXL3_GDN_PORT`, `EXL3_GDN_PORT_TOKINDX` | /opt/exl3xpu/exl3xpu/gdnchunk.so, 1 | deterministic chunked GDN prefill; mixed steps in place | exl3xpu (+ vLLM `pos0`) |
| `EXL3_DYN_SD_FULL`, `EXL3_UNIFORM_SPEC` | 2, 1 | FULL decode graphs for every schedule depth; uniform-decode dispatch only for all-spec steps | exl3xpu |
| `EXL3_DET_LINEAR` | 1 | deterministic oneDNN primitives for unquantized linears | exl3xpu |
| `EXL3_FAST_NAN_GUARD` | 1 | fused NaN guard for the runner's NaN check | exl3xpu (+ vLLM) |
| `EXL3_LAZY_VISION`, `EXL3_LAZY_VISION_NS`, `EXL3_LAZY_VISION_ENGINE_ID` | 1, `weights:qwen3.8-27b-visual`, 1 | lazy vision tower (KV +22k tokens); host copy shared with the 5090 through the broker | llmt_kv + vLLM |
| `EXL3_LORA_HOT`, `VLLM_ALLOW_RUNTIME_LORA_UPDATING`, `EXL3_LORA_DIR`, `EXL3_LORA_MAX_RANK`, `EXL3_LORA_MARGIN_MB` + `--enable-lora --max-loras 4 --max-lora-rank 16 --max-cpu-loras 8 --lora-target-modules ...` | 1, 1, /loras, 16, 8192 (`LORA=1`; `LORA_DIR` mounted at /loras if set) | hot-loaded LoRA adapters in lent KV blocks (vision tower lent too) | llmt_kv + exl3xpu LoRA kernels |
| `EXL3_WHX`, `EXL3_WHX_DIR`, `EXL3_WHX_EXCLUSIVE`, `EXL3_WHX_MARGIN_MB` | keeper socket, side-file dir, 1, 768 (only when `KEEPER_SOCK` exists) | weight keeper attach; the 768 MiB margin keeps the allocator headroom with held weights | llmt_kv |
| `EXL3_KSMOOTH` | /data/ksmooth.pt (default `data/ksmooth/public_r.pt`; `KSMOOTH_TABLE=none`: `null`) | K smoothing; also enables the fused int4 KV write (I4-KVW) | exl3xpu |
| `EXL3_DRAFT_VOCAB`, `EXL3_DRAFT_VOCAB_FAR`, `EXL3_DRAFT_FAR_FROM` | /data/draft-vocab.json (512 blocks), /data/draft-vocab-far.json (128 blocks), 2; default `data/draft-vocab/public-512.json` / `public-128.json` | graded pruned draft heads: positions >= 2 use the smaller far head. `DRAFT_VOCAB=none`: `null` (full-head drafting) | exl3xpu |
| `EXL3_DRAFT_Q4`, `EXL3_DS_SUB`, `EXL3_DRAFT_DYN`, `_NEAR`, `EXL3_DRAFT_BIAS` | 64, 1, 256, 128, 0 (with a draft vocab, i.e. by default) | int4 draft heads (group 64); draft sampling from the compact logits; per-request extra draft rows | exl3xpu |
| `EXL3_KV_DTYPE_TAG` | int4_per_token_head | KV dtype in the tier's compatibility tag | llmt_kv |
| `EXL3_RC_PINNED`, `EXL3_ALIGN_NOSYNC`, `EXL3_SPEC_STOP_TRUNC`, `EXL3_EAGLE_KEEP` | 1, 1, 1, 2 | as on the 5090 | vLLM |
| `EXL3_TAIL_HOLD` (+ `_S` 60, `_TOOL_S` 300, `_FREE` 0.15, `_MAX` 16) | 1 | tail hold; the Arc ran the code defaults (released below 15 % free KV) | vLLM |
| `EXL3_SCHED_DECODE_STEPS`, `EXL3_SCHED_MIXED_K`, `EXL3_SCHED_MIXED_CHUNK` | 0, 4, 0 | at most 4 drafts in a step that mixes a prefill chunk with decodes; no decode-only steps | vLLM |
| `EXL3_DYN_SD_DECODE_COUNT` | 1 | depth schedule keyed on decoding requests | vLLM |
| `EXL3_GPU_EVICT_ORDER`, `EXL3_MAMBA_KEEP_LAST`, `EXL3_ENCODER_CACHE_TOKENS`, `EXL3_STATE_STORE` | chain, 2, 16384, prompt_end | as on the 5090 | vLLM |
| `EXL3_KV_COPY_PACE_MB`, `EXL3_KV_RING_MB`, `EXL3_KV_RING_LOAD_MB`, `EXL3_KV_RING_LOAD_IDLE_S` | 8, 32, 256, 86400 | paced tier copies through a 32 MB pinned staging ring + a 256 MB load buffer kept resident (48k restore with 2 decoders 10.5 s -> 1.6 s) | vLLM |
| `NEOReadDebugKeys`, `OverrideEnableKmdNotify`, `OverrideKmdNotifyDelayMicroseconds`, `EnableUserFenceForCompletionWait`, `EventHostSynchronizeLinuxUserFenceKmdWait`, `OverrideUseKmdWaitFunction` | 1, 1, 200, 1, 1, 1 | Intel compute runtime: wait for GPU completion in the kernel driver instead of spinning a CPU core | Intel NEO |

Defaults that production relies on without setting them: `EXL3_FUSED_MLP`, `EXL3_FAST_NORM`, `EXL3_FAST_TOPKP`,
`EXL3_FAST_DRAFT_SAMPLE`, `EXL3_INT8KV_FAST`, `EXL3_I4_ROT`, `EXL3_I4_PREROT`, `EXL3_I4_KVW`, `EXL3_SPEC_VARR`,
`EXL3_NOMMAP_LOAD`, `EXL3_KV_NANCHECK` (all 1).

## Warm-up (`llmt-warmup.py`)

Both engine launchers start `launchers/llmt-warmup.py` detached, at idle CPU priority, right before the engine
(`start_warmup` in `common.sh`). It waits for the engine's `/health`. Then it sends a few synthetic requests over the
shapes that pay a one-time per-process cost, so real requests don't. All requests use a unique `cache_salt` and
`llmt_nostore`, so nothing enters the prefix cache or the RAM tier. Logs (counts and timings only):
`/var/log/allrunner/warmup-<profile>.log`.

Measured on fresh engine processes with persistent compile caches (synthetic prompts; first use vs second use):

| | Arc B70 | RTX 5090 |
|---|---|---|
| first request | 1.55 vs 0.17 s | 0.92 vs 0.04 s |
| first `json_object` request (xgrammar) | 1.75 vs 0.34 s | 1.08 vs 0.16 s |
| 1-2 concurrent streams | TTFT +0.26-0.33 s, one 178 ms step | TTFT +0.42-0.45 s, one 198 ms step |
| short prompts | 100 tokens +0.25 s, 2k +0.16 s | free |
| 3+ streams (every draft-depth band), prompts of 3k-16k tokens, mixed prefill+decode steps | free | free |

The profiles warm exactly the shapes with a cost: Arc 10 s, 5090 5 s after health. Run `llmt-warmup.py --profile
arc --plan` to see the steps. A cold compile cache (new image or flags) costs far more than this, and a warm-up
cannot hide it: keep `CACHE_VOLUME`.

**Router gate.** With `ROUTER_NAME` set (default `arc` / `rtx5090`, the names in `router.example.json`), the
warm-up drains that backend in the router while it runs. It only drains when another backend is healthy, undrained
and not held per `GET /router/state`; otherwise the engine warms while serving.
- **The drain file:** `$LLMT_ROUTER_DRAIN_DIR/drain-<name>`, created with an `llmt-warmup` marker.
- **Lifted:**
  - when the warm-up ends.
  - on SIGTERM/INT/HUP.
  - `WARMUP_HARD_TIMEOUT` (300) s after engine health.
  - as soon as the other backend stops being usable.
  - by `llmt-warmup.py --cleanup <name>`, which runs after the warm-up process exits by any means.
  - by the next start.
- **Never removed:** a drain file without the marker, or one touched after creation (an operator draining by hand).

Settings: `WARMUP=0` turns the warm-up off, `ROUTER_NAME=` warms without draining, plus `ROUTER_URL`,
`WARMUP_LOG`, `WARMUP_STATE_DIR` and `WARMUP_HARD_TIMEOUT`. Router and launchers must agree on
`LLMT_ROUTER_DRAIN_DIR`. `--lora auto` (off by default) also sends one short request per LoRA module the engine lists
in `/v1/models`.

## Broker (`kvbroker.sh`)

| Flag | Value | Why |
|---|---|---|
| `--pool-gib` | 16 | host RAM for both GPUs' KV blocks and GDN states (tmpfs in `/dev/shm`) |
| `--slot-kib` | 56100 | one slot = one vLLM CPU block: 17 x 3,379,200 B for a 3,200-token int4 block incl. GDN state. Engines with int8 KV need 57,460 KiB slots and a broker of their own |
| `--pool-dir`, `--socket` | /dev/shm/llmt-kv-i4, /run/llmt-kv-i4/broker.sock | both mounted into the engine containers at the same paths |
| `--metrics` | 127.0.0.1:9113 | Prometheus metrics |
| `--no-hugepages-ctl`, `--no-mlock` | | plain shared-memory pages |
| `--new-pool` | | empty pool at start |
| `--hotset-dir off --events-dir off --disk-dir off --chains off` | | RAM only: no hot-set snapshot, no event log, no SSD tier, no lookup-chain index |
| `--mem-reserve-gib 0`, `--stats-log-s 600` | | no extra reserve; one statistics line per 10 min |

Production ran it under systemd with `Restart=on-failure`, `RestartSec=1`, `TimeoutStopSec=180`, `Nice=-5`.
AllBroker's `contrib/systemd/llmt-kvbroker.service` is a generic example (2 MiB slots, SSD tier): use the flags
above for these engines. The n-gram tables in AllBroker are not used by this configuration. The weight keeper
(`llmt-keeper@arc`, `llmt-keeper@rtx5090`, AllBroker `contrib/systemd/`) is optional: an engine launcher attaches to it
when its socket (`KEEPER_SOCK`) exists.

## Router (`router.sh`)

`python3 router/llmt-router.py --config /etc/allrunner/router.json --decision-log /var/log/llmt-router/decisions.log`
(the production command line). Start from `router/router.example.json`: it is the production configuration of one
Arc B70 (`arc`, :8000, cap 16) and one RTX 5090 (`rtx5090`, :8001, cap 32 = its `--max-num-seqs`, `prefer_until 2`), with measured prefill and
per-stream decode rates, `shared_tier 0.9` and `inject_kv_ns "shared"`. Re-measure the rates for other hardware;
every key is documented in `router/README.md`. Drain a backend with `touch /run/llmt-router/drain-<name>`
(`LLMT_ROUTER_DRAIN_DIR`).
`ROUTER_POLICY=<file>` adds `--policy-file` (session + priority policy, `router/router-policy.example.json`; it takes
effect with `"session_policy": {"on": true}` in router.json). Keep the real policy file outside any repository.

## Data files you generate

AllRunner ships no file derived from production traffic. The draft vocabularies default to the public files in
[`../data/draft-vocab/`](../data/draft-vocab/) (`public-512.json`, `public-128.json`, built by exl3xpu
`scripts/public_draft_vocab.py` from public text only; every input and its sha256 is listed inside each file), the
K-smoothing table to [`../data/ksmooth/public_r.pt`](../data/ksmooth/) (exl3xpu `scripts/public_ksmooth.py`, public
text only; same quality and speed as production's traffic-built table). A vocabulary ranked on your own traffic
usually accepts more draft tokens. The 5090's 4-bit draft head has no public default (empty = off); `none` turns the
others off:

| Variable | File | How to make it | Without it |
|---|---|---|---|
| `KSMOOTH_TABLE` (both) | K-mean table, `{layer: mean[H, D]}` torch file | default: the shipped `data/ksmooth/public_r.pt`. Your own: run the engine eager with `EXL3_KSMOOTH_CALIB=<out.pt>` over representative prompts (exl3xpu `scripts/public_ksmooth.py` does this with public text; exl3xpu README-ARC.md "Data files") | `KSMOOTH_TABLE=none`: K smoothing off. Arc, int4 KV, held-out GSM8K, vs production's table: public table NLL +0.037 %, none +0.43 %; speed: the table plus the fused int4 KV write it enables add about +6-7 % on a fast host (Arc B70: 1-stream aggregate 111.8 -> 119.7 tok/s, per-stream decode 139.2 -> 148.5) and up to +35 % per-stream decode on a slow, host-bound CPU (142 -> 192 tok/s on HumanEval prompts). Measured with int8 KV on real agent prompts: +1.00 % NLL without vs +0.51 % with the robust table; int4 KV (KS4) uses the same correction. On the Arc the fused int4 KV write is also off (slower KV writes). Both engines must use the same file to share RAM-tier entries |
| `DRAFT_VOCAB`, `DRAFT_VOCAB_FAR` | draft vocabulary json (128-token blocks): 512 blocks near, 128 blocks far (Arc), the same 512-block file for both on the 5090. Default: the shipped public files | exl3xpu `scripts/draft_vocab.py --blocks 512 ...` on text like your traffic, or `scripts/public_draft_vocab.py --blocks 512 128` (the shipped files); the Arc also accepts `DRAFT_VOCAB=generic` (upstream's 512-block file in the image) | `DRAFT_VOCAB=none`: full-head drafting: every draft pass reads the whole 6-bpw lm_head; DRAFT-DYN, DRAFT-Q4 and DS-SUB are inert. For scale: on the 5090 a 512-block vocab ranked on production outputs gave 4.18 vs 3.74 tokens/step over a 128-block one (+9 % decode) |
| `DRAFT_WEIGHTS` (5090) | 4-bit draft head for that vocab | `exl3cuda/tools/quantize_draft.py --bf16 <BF16 checkpoint> --vocab <vocab> --no-mtp --k-head 4 --out head4.safetensors` | the checkpoint's 6-bit head rows: about -3 % decode, same acceptance (drafts only: output unchanged) |

## Hardware notes

- **Power.** Both GPUs running flat out on one circuit tripped a breaker once (an RTX 5090 alone can draw ~575 W).
  Before loading both cards fully at the same time (benchmarks, simultaneous starts, long prefill bursts), check the
  PSU's and the circuit's headroom, or stagger the load (start and benchmark one GPU at a time; `nvidia-smi -pl`
  lowers the 5090's limit).
- The production Arc ran a 330 W power limit set through a patched kernel driver (275 W is the stock limit), and the
  production 5090 ran a memory/core overclock set by a separate service. Neither is part of AllRunner; the launchers
  run the cards at their stock limits, so expect somewhat lower numbers than the published ones.
- RAM-tier loads cross PCIe. On narrow links (the production 5090 sat on an x4 link) `EXL3_KV_COPY_PACE_MB` keeps
  tier copies from stalling decode steps.
- Host RAM: 16 GiB broker pool in `/dev/shm`, plus each engine's pinned vision tower (lazy vision) and container
  shared memory (`--shm-size` is a cap, used on demand). Make sure `/dev/shm` can hold the pool with room to spare.

## Left out on purpose

Production settings that the launchers do not reproduce:

| Setting | Engine | Why |
|---|---|---|
| request sampling (`EXL3_SAMPLE_REQUESTS=1`) | both | writes users' prompts and outputs to disk; not in the published vLLM branches |
| prefix-cache trace (`EXL3_CACHE_TRACE_KEY` + trace files) | both | traces of users' sessions with a secret key; not published |
| step statistics (`EXL3_STEP_STATS=1`), lookup statistics (`EXL3_LOOKUP_STATS=1`), offload debug log (`EXL3_OFFLOAD_DEBUG=1`), tail-hold log (`EXL3_TAIL_HOLD_LOG=1`), torch profiler config | both | diagnostics; the first three are not in the published branches |
| NaN dump files, GDN activation capture | both | write request-derived data; not published (NaN recompute itself is kept) |
| lookup chains (`EXL3_LOOKUP*`, off), tail spill (`EXL3_TAIL_SPILL=0`), draft trainer capture (off) | 5090 | off in production; not in the published branches |
| `EXL3_SUBBLOCK_CHUNK=1` | Arc | a no-op with `EXL3_SCHED_MIXED_CHUNK=0`; dropped from the published branch |
| `"xgpu-test"` in `broker_ns_allow` | both | cross-GPU test namespace |
| hold/drain files checked by the launchers, watchdog pause files, render-node and disk fallback logic, page-cache prewarm and memory guards | both | specific to the production box (the router's own drain/hold files are kept) |
| power-limit write (330 W) and GPU overclock | both | host kernel / site tuning, see above |
| a hugepage pool mount | Arc | left over from an earlier int8 broker |
| the fp16-KV spec-attention route patched into Intel's `vllm_xpu_kernels` | Arc | never taken in production (needs an fp16 KV cache and a paged call; the Arc runs int4 KV and the plugin's only call into that module passes no block table); not published |
| the production compile-cache key over every mounted file | 5090 | replaced by a key over image id + flags + environment |
| production's draft vocabularies, draft head, K-smoothing table | both | derived from production traffic; the launchers use the public draft vocabularies and K-smoothing table instead (see [Data files](#data-files-you-generate)) |
| KV-RING staging ring (`EXL3_KV_RING_MB` 32, `_LOAD_MB` 256, `_LOAD_IDLE_S` 86400) and `LLMT_KV_SLOTPIN=0` | 5090 | the CUDA ring is not in the published vllm CUDA branch (the Arc branch has it); RAM-tier restores under decode load are slower without it |

## Production parity

Every production engine setting is a launcher default except the ones below (checked 2026-11-16 against the live
production launchers of both GPUs, including the scripts they call and the image's `model.yaml`).

| Setting | Production | Launcher default | Why different |
|---|---|---|---|
| draft vocabulary (both) | 512 + 128 blocks (Arc), 512 near = far (5090), ranked on production traffic | shipped public files (`data/draft-vocab/public-512.json`, `public-128.json`) | traffic-derived; public equivalent shipped. Acceptance (and so decode speed) can differ: re-measure |
| `EXL3_DRAFT_WEIGHTS` (5090) | 4-bit head of the production vocabulary | none (6-bit checkpoint rows) | traffic-derived (rows of that vocabulary). Measured: -3 % decode, acceptance unchanged. A public one is `quantize_draft.py --vocab data/draft-vocab/public-512.json` (needs the BF16 checkpoint; ~170 MB, not shipped) |
| `EXL3_KSMOOTH` (both) | K-mean table from production prompts | shipped public table `data/ksmooth/public_r.pt` | traffic-derived; public equivalent shipped and matched on the Arc (int4 KV, held-out GSM8K): NLL +0.037 % vs the production table, +0.43 % without a table; speed: the table plus the fused int4 KV write it enables add about +6-7 % on a fast host (Arc B70: 1-stream aggregate 111.8 -> 119.7 tok/s, per-stream decode 139.2 -> 148.5) and up to +35 % per-stream decode on a slow, host-bound CPU (142 -> 192 tok/s on HumanEval prompts). RTX 5090 load of this table with these launchers not checked yet |
| KV-RING (5090) | 32 MB ring, 256 MB load buffer, `LLMT_KV_SLOTPIN=0` | not available | not in the published vllm CUDA branch ([Left out](#left-out-on-purpose)) |
| power limit / overclock | Arc 330 W (patched xe driver), 5090 memory/core overclock | stock | host tuning, not AllRunner ([Hardware notes](#hardware-notes)) |
| `broker_ns_allow` | `["xgpu-test","shared"]` | `["shared"]` | test namespace |
| diagnostics (`EXL3_STEP_STATS`, `EXL3_LOOKUP_STATS`, `EXL3_OFFLOAD_DEBUG`, `EXL3_TAIL_HOLD_LOG`, request sampling, cache trace, profiler config) | on | off / absent | write logs or user data; not in the published branches |
| `EXL3_SUBBLOCK_CHUNK=1` (Arc) | set | absent | no-op with `EXL3_SCHED_MIXED_CHUNK=0` |
| lookup chains (5090: `EXL3_LOOKUP=0`, `_GATE`, `_LEN`), tail spill (`EXL3_TAIL_SPILL=0`) | off | absent | off in production; not published |
| paths, ports, sockets, image names, engine ids | site values | variables at the top of each launcher | site-specific |

Same values, different spelling: `EXL3_SHAPE_P_FILE` and `EXL3_GDN_PORT` point at the image's copies (the shape table
is byte-identical); the tail-hold Arc values and `EXL3_DYN_SD_DECODE_COUNT=1` are the code defaults that production
relies on, written out. Priority scheduling (`--scheduling-policy priority`) is off in both.

## Tuning for your own traffic

- **Out of the box** the public draft vocabularies (`data/draft-vocab/`, 512 + 128 blocks) and the public K-smoothing
  table (`data/ksmooth/public_r.pt`) are the launchers' defaults (`DRAFT_VOCAB`, `DRAFT_VOCAB_FAR`, `KSMOOTH_TABLE`). Measured against the tables production builds from its own agent traffic: the public K-smoothing
  table gives NLL +0.037 % and top-5 KL 4.2e-3 (Arc B70, int4 KV, 126,099 held-out GSM8K tokens; no table: NLL +0.43 %)
  at the production table's speed (versus no table: the table plus the fused int4 KV write it enables add about +6-7 % on a fast host (Arc B70: 1-stream aggregate 111.8 -> 119.7 tok/s, per-stream decode 139.2 -> 148.5) and up to +35 % per-stream decode on a slow, host-bound CPU (142 -> 192 tok/s on HumanEval prompts)); the public vocabularies cover the same share of held-out
  model outputs (512 / 128 blocks: 99.3 / 94.8 % on GSM8K, 96.5 / 81.6 % on IFBench). Decode speed with the public
  vocabularies has not been measured yet.
- **Draft vocabulary from your traffic:** rank on text like your model's outputs with exl3xpu
  `scripts/draft_vocab.py --model <checkpoint> --blocks 512 --text '<your outputs>/*.txt' --out draft-512.json` (and
  `--blocks 128` for the far head). `scripts/public_draft_vocab.py` is the same ranking over its fixed public input set.
  Drafts only: the output distribution does not change, only how many draft tokens are accepted.
- **K-smoothing table from your traffic:** run ONE engine eager (`enforce_eager`) with `EXL3_KSMOOTH_CALIB=/path/ksmooth.pt`
  and without `EXL3_KSMOOTH`, on the KV dtype you serve with, over representative traffic or a replay of it. The table is
  rewritten as it goes (exl3xpu: every `EXL3_KSMOOTH_CALIB_EVERY` batches, default 1; exl3cuda: every 64 calls). Then
  restart serving with `EXL3_KSMOOTH=/path/ksmooth.pt` (and without `EXL3_KSMOOTH_CALIB`). exl3xpu
  `scripts/public_ksmooth.py` is this procedure fed with public text. The file format is the same on both GPUs.
- **Caveats:** calibration needs an eager engine (no graphs, so it is slower while it runs): use a separate engine or a
  quiet period. A new table invalidates cached KV and the RAM-tier entries (the table's checksum is part of the tier's
  compatibility tag), so switch at a planned restart; engines that share a RAM tier must all use the same table. This
  is how production does it.

## Untested

- Both images were built from these Dockerfiles on 2026-11-11, from pinned commits, and every component imports
  (vLLM, exllamav3 / exl3cuda or exl3xpu, llmt_kv with its tier and loader modules). No engine has been started from
  them and neither launcher has been run as written; the production engines ran the same code (component branches
  made from the production files) with the same flags, mounted into older images.
- `docker/Dockerfile.arc`: the GPU runtime packages and the three kernel libraries are byte-identical to production's.
  The base image is referenced by digest and must still be available; the runtime packages come from Launchpad's
  archive of superseded PPA builds (sha256-checked).
- `docker/Dockerfile.rtx5090`: all eight exl3cuda kernel libraries build; they have not been compared with the
  production binaries byte for byte (the FA16 kernel code was, separately: identical).
- `llmt_kv.cuda_tier` / `llmt_kv.xpu_tier` (AllBroker) replace the production engines' older tier modules; the
  published versions are newer than the copies production ran.
- The Docker build args follow each repo's default branch (echowarp/vllm `main` / `xpu`, echowarp/exllamav3
  `master`, echowarp/exl3xpu `main`, AllBroker `main`); images built from these
  pins have not been started yet (the 2026-11-11 builds above used older commits).
- Weight keeper attach, LoRA lending (Arc) and vision-tower lending with the packed KV layout (5090) are on by
  default (keeper only when its socket exists) as in production; production ran the same settings, these launchers
  have not started them.
- `router.sh` with `ROUTER_POLICY` and `"session_policy": {"on": true}` has only run under the CPU tests.
