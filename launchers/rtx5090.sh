#!/usr/bin/env bash
# AllRunner: Qwen3.8-27B (EXL3 4.0 bpw) on one NVIDIA RTX 5090 with vLLM 0.26 CUDA + exl3cuda, production configuration.
#
# STATUS: DRAFT, NOT RUN AS WRITTEN. It reproduces the flags and environment of the production 5090 engine, which was
# started by a different (site-specific) script; this file has only been syntax-checked. See launchers/README.md.
#
# Image: docker/Dockerfile.rtx5090 (vllm/vllm-openai:v0.26.0 + echowarp/vllm publish-20261113-cuda + echowarp/exllamav3
# publish-20261113 incl. exl3cuda + llmt_kv from AllBroker).
# Runs in the foreground (for a supervisor such as systemd or llama-swap); DETACH=1 runs it detached.
# Stop: docker stop -t 30 "$NAME".
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=launchers/common.sh
. "$HERE/common.sh"
ROOT=$(cd "$HERE/.." && pwd)                           # AllRunner checkout (data/ holds the shipped public files)

# ---- site settings: every value can be overridden from the environment -------------------------------------------
MODEL_DIR=${MODEL_DIR:-/models/Qwen3.8-27B-exl3-4.00bpw}  # turboderp/Qwen3.8-27B-exl3, branch 4.00bpw (EXL3 4.0 bpw)
IMAGE=${IMAGE:-allrunner-rtx5090:latest}              # built from docker/Dockerfile.rtx5090
NAME=${NAME:-allrunner-rtx5090}                       # container name
BIND=${BIND:-127.0.0.1}                               # host address; keep it on localhost (no auth on the engine)
PORT=${PORT:-8001}                                    # host port (the router expects the 5090 on 8001)
GPU_INDEX=${GPU_INDEX:-0}                             # nvidia-smi index, for the free-memory wait below
DOCKER_GPU_ARGS=${DOCKER_GPU_ARGS:-"--device nvidia.com/gpu=$GPU_INDEX"}   # CDI; or "--gpus device=$GPU_INDEX"
CACHE_VOLUME=${CACHE_VOLUME:-allrunner-rtx5090-cache} # docker volume (or host dir) for /root/.cache: Triton + compile caches
SHM_SIZE=${SHM_SIZE:-16g}
HASHSEED_FILE=${HASHSEED_FILE:-/etc/allrunner/hashseed.env}  # PYTHONHASHSEED=..., mode 0600, same file for every engine
# Shared host-RAM KV tier (launchers/kvbroker.sh). The pool directory is mounted at the same path inside the
# container (the broker hands out its host path).
KV_TIER=${KV_TIER:-1}                                 # 0 = no RAM tier (no kv_transfer_config)
BROKER_SOCK=${BROKER_SOCK:-/run/llmt-kv-i4/broker.sock}
BROKER_POOL_DIR=${BROKER_POOL_DIR:-/dev/shm/llmt-kv-i4}
BROKER_NAMESPACE=${BROKER_NAMESPACE:-rtx5090-private} # this engine's own namespace in the broker
BROKER_NS_ALLOW=${BROKER_NS_ALLOW:-'["shared"]'}      # namespaces a request may pick (the router injects "shared")
# Share of the broker's slots this engine may hold leases on at once (llmt_kv default 0.25). A lease pins a slot only
# while its load is pending; 0.25 of a 186-slot pool (46 leases) let only ~4 agents resume 22k-token contexts from RAM at
# once. Measured: 3 concurrent 22.4k-token RAM hits, 41,600 of 60,800 block tokens hit at 0.25 (4.5 s each), all of
# them at 0.75 (1.8-2.4 s).
BROKER_LEASE_FRAC=${BROKER_LEASE_FRAC:-0.75}
# Data files: the defaults are public builds shipped in data/ (no production traffic in them); "none" = feature off.
#   KSMOOTH_TABLE: K-smoothing table. Default data/ksmooth/public_r.pt (exl3xpu scripts/public_ksmooth.py, public text
#     only; inputs and sha256 in public_r.pt.inputs.json). Measured on the Arc (int4 KV, held-out GSM8K) vs a table built
#     on production traffic: NLL +0.037 %; without a table (KSMOOTH_TABLE=none) NLL +0.43 %. Speed: the table enables
#     the fused int4 KV write: about +6-7 % on a fast host, up to +35 % per-stream decode on a slow, host-bound CPU.
#     Same file format on this GPU (its load with these launchers is not checked yet).
#     Engines that share the RAM tier must use the SAME table file (its hash is part of the tier's compatibility tag).
KSMOOTH_TABLE=${KSMOOTH_TABLE:-$ROOT/data/ksmooth/public_r.pt}
[ "$KSMOOTH_TABLE" = none ] && KSMOOTH_TABLE=""
#   DRAFT_VOCAB: 512-block draft vocabulary json (pruned MTP draft head). Default: the public 512-block file shipped in
#     data/draft-vocab/ (exl3xpu scripts/public_draft_vocab.py, public inputs with sha256s in the file); your own,
#     ranked on text like your traffic (exl3xpu scripts/draft_vocab.py), can replace it. DRAFT_VOCAB=none = full
#     lm_head drafting: each draft pass reads the whole 6-bpw head instead of 512 of its 128-token blocks, and
#     DRAFT-DYN is inert. (On this GPU a production-ranked 512-block vocab gave 4.18 vs 3.74 tokens/step over a
#     128-block one, +9 % t/s.)
DRAFT_VOCAB=${DRAFT_VOCAB:-$ROOT/data/draft-vocab/public-512.json}
[ "$DRAFT_VOCAB" = none ] && DRAFT_VOCAB=""
DRAFT_VOCAB_FAR=${DRAFT_VOCAB_FAR:-}                  # empty = DRAFT_VOCAB (production: same file = one head copy)
#   DRAFT_WEIGHTS: 4-bit re-quantized draft head for DRAFT_VOCAB (exl3cuda tools/quantize_draft.py). Without it the
#     checkpoint's 6-bit head rows are used (measured: -3 % decode, acceptance unchanged). Drafts only: output exact.
DRAFT_WEIGHTS=${DRAFT_WEIGHTS:-}
# Packed (cross-layer) KV layout, needs the RAM tier (KV_TIER=1) and echowarp/vllm: one page per block for all layers.
PACKED_KV=${PACKED_KV:-1}
# KV-block lending for the vision tower (AllBroker llmt_kv.lend, EXL3_LEND_ONLY: no LoRA on this GPU): the tower's
# memory is lent KV blocks while no image is in flight. Measured with packed KV + lazy vision: KV 436,585 -> 482,846.
VISION_LEND=${VISION_LEND:-1}
# Weight keeper (AllBroker llmt-keeper@rtx5090): used when its socket exists (weights stay on the GPU across restarts).
KEEPER_SOCK=${KEEPER_SOCK:-/run/llmt-keeper-rtx5090/keeper.sock}
KEEPER_DIR=${KEEPER_DIR:-/var/lib/llmt/keeper/rtx5090/side}   # keeper side files (DRAFT-DYN table ~0.7 GB) on disk
MAX_MODEL_LEN=${MAX_MODEL_LEN:-196608}
# 0.86: CUDA graphs up to 1,024 tokens take 1.33 GiB that vLLM does not count (VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0);
# at 0.88 a burst of 16 concurrent ~6k-token prompts ran out of memory (272 MiB short), 0.86 passed 2 x 32 of them.
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.86}
# 32 concurrent requests with MTP depth 1 above 16 (the schedule below): a 32-stream decode step stays at 64 rows, the
# top of the fused-GEMM band. Measured c32: 1,325 vs 1,237 tok/s with 16 seqs, TTFT 0.88 vs 1.26 s; c1-c16 unchanged.
MAX_NUM_SEQS=${MAX_NUM_SEQS:-32}
DETACH=${DETACH:-0}
# Post-start warm-up (launchers/llmt-warmup.py, README "Warm-up"): synthetic requests over the shapes with a first-use
# cost, router backend ROUTER_NAME drained meanwhile if another backend is healthy. WARMUP=0 = off; ROUTER_NAME='' = no drain.
WARMUP=${WARMUP:-1}
ROUTER_NAME=${ROUTER_NAME-rtx5090}                        # this engine's backend name in the router config
ROUTER_URL=${ROUTER_URL:-http://127.0.0.1:8010}
# ----------------------------------------------------------------------------------------------------------------------

load_hashseed "$HASHSEED_FILE"
[ -d "$MODEL_DIR" ] || die "MODEL_DIR $MODEL_DIR not found"
check_file "$KSMOOTH_TABLE" KSMOOTH_TABLE
check_file "$DRAFT_VOCAB" DRAFT_VOCAB
check_file "$DRAFT_VOCAB_FAR" DRAFT_VOCAB_FAR
check_file "$DRAFT_WEIGHTS" DRAFT_WEIGHTS
[ -z "$DRAFT_WEIGHTS" ] || [ -n "$DRAFT_VOCAB" ] || die "DRAFT_WEIGHTS needs DRAFT_VOCAB (the head rows it re-quantizes)"

# vLLM sizes the KV cache from the free memory at start: right after a stop, wait (up to 30 s) until the previous
# engine has released the GPU (a start right after a stop measured 203k instead of 245k KV tokens).
# Weights held by the keeper stay allocated: they are added to the threshold.
KEEP_MIB=0
if [ -S "$KEEPER_SOCK" ] && command -v llmt-keeper-ctl >/dev/null 2>&1; then
  KEEP_MIB=$(llmt-keeper-ctl "$KEEPER_SOCK" held-mib 2>/dev/null || echo 0)
fi
if command -v nvidia-smi >/dev/null 2>&1; then
  for _ in $(seq 1 30); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU_INDEX" 2>/dev/null | head -1)
    [ "${used:-0}" -lt $((1000 + ${KEEP_MIB:-0})) ] && break
    sleep 1
  done
fi

ENV=(
  -e PYTHONHASHSEED                       # by name: the value comes from $HASHSEED_FILE
  -e HF_HUB_OFFLINE=1
  -e TRITON_CACHE_DIR=/root/.cache/triton # keep Triton JIT kernels across restarts (else they compile under traffic)
  -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0   # vLLM's graph estimate was 6.1 GiB vs 0.76 actual: +12 % KV
  -e VLLM_KV_CACHE_LAYOUT=NHD             # same page layout as the Arc, so RAM-tier entries are interchangeable
  -e VLLM_PREFIX_CACHE_RETENTION_INTERVAL=12800   # sparse GDN-state retention: 4 x 3,200-token blocks
  # exl3cuda (echowarp/exllamav3 exl3cuda/README.md)
  -e EXL3_INT8_PREFILL=1 -e EXL3_INT8_L2SPLIT=1
  -e EXL3_FUSE_DECODE=1 -e EXL3_FUSE_MAX_M=64 -e EXL3_FUSE_SCRATCH=1 -e EXL3_DRAFT_GRAPH=${EXL3_DRAFT_GRAPH:-1}
  -e EXL3_GDN_RECOMPUTE=1 -e EXL3_RC_FUSED_IN=1 -e EXL3_RC_SLICE=1
  -e EXL3_GDN_PORT=1 -e EXL3_GDN_FAST=1
  -e EXL3_FA16=f
  -e EXL3_I8FA=1 -e EXL3_I8FA_PAGED_MAXQ=auto     # int8-KV prefill kernels: loaded, unused with int4 KV
  -e EXL3_VA=1 -e EXL3_VA4_OP=va4c -e EXL3_VA4_SO=/opt/exl3cuda-lib/va4c.so   # VA4 code default names va4b: set both
  -e EXL3_VERIFY_Q=8                              # k = 7 verify rows + 1
  -e EXL3_TAIL_FUSE=1
  -e EXL3_DYN_SD_FULL=2
  -e EXL3_LAZY_VISION=1 -e EXL3_LAZY_VISION_NS=weights:qwen3.8-27b-visual   # tower entry shared by both GPUs
  -e EXL3_KV_DTYPE_TAG=int4_per_token_head        # KV dtype in the RAM-tier compatibility tag (llmt_kv)
  # vLLM branch publish-20261113-cuda (echowarp/vllm)
  -e EXL3_ATTN_VERIFY_BM=32 -e EXL3_ATTN_VERIFY_WARPS=2 -e EXL3_ATTN_SEGMENTS=64 -e EXL3_ATTN_SEGMENTS_MANY=32
  -e EXL3_FI_TOPKP=1
  -e EXL3_ALIGN_NOSYNC=1 -e EXL3_RC_PINNED=1
  -e EXL3_SPEC_STOP_TRUNC=1
  -e EXL3_EAGLE_KEEP=2
  -e EXL3_TAIL_HOLD=1 -e EXL3_TAIL_HOLD_S=60 -e EXL3_TAIL_HOLD_TOOL_S=300 -e EXL3_TAIL_HOLD_FREE=0.03
  -e EXL3_TAIL_HOLD_MAX=16 -e EXL3_TAIL_HOLD_DECAY_S=60
  -e EXL3_SCHED_DECODE_STEPS=1 -e EXL3_SCHED_MIXED_K=-1 -e EXL3_DYN_SD_DECODE_COUNT=1
  -e EXL3_GPU_EVICT_ORDER=chain -e EXL3_MAMBA_KEEP_LAST=2 -e EXL3_ENCODER_CACHE_TOKENS=16384
  -e EXL3_STATE_STORE=prompt_end
  -e EXL3_KV_COPY_PACE_MB=8 -e EXL3_KV_NANCHECK=1
  -e EXL3_DYN_PIN=1 -e LLMT_KV_PIN_NAME=rtx5090-pin -e LLMT_KV_ENGINE_ID=2
)
MOUNTS=(-v "$MODEL_DIR:/models:ro" -v "$CACHE_VOLUME:/root/.cache")
if [ -n "$KSMOOTH_TABLE" ]; then
  MOUNTS+=(-v "$KSMOOTH_TABLE:/data/ksmooth.pt:ro"); ENV+=(-e EXL3_KSMOOTH=/data/ksmooth.pt)
fi
if [ -n "$DRAFT_VOCAB" ]; then
  MOUNTS+=(-v "$DRAFT_VOCAB:/data/draft-vocab.json:ro")
  ENV+=(-e EXL3_DRAFT_VOCAB=/data/draft-vocab.json -e EXL3_DRAFT_FAR_FROM=2
        -e EXL3_DRAFT_DYN=256 -e EXL3_DRAFT_DYN_NEAR=128 -e EXL3_DRAFT_BIAS=0)
  if [ -n "$DRAFT_VOCAB_FAR" ]; then
    MOUNTS+=(-v "$DRAFT_VOCAB_FAR:/data/draft-vocab-far.json:ro"); ENV+=(-e EXL3_DRAFT_VOCAB_FAR=/data/draft-vocab-far.json)
  else
    ENV+=(-e EXL3_DRAFT_VOCAB_FAR=/data/draft-vocab.json)
  fi
fi
if [ "$VISION_LEND" = 1 ]; then ENV+=(-e EXL3_LEND_ONLY=1); fi
if [ -S "$KEEPER_SOCK" ]; then
  mkdir -p "$KEEPER_DIR"
  KDIR=$(dirname "$KEEPER_SOCK")
  MOUNTS+=(-v "$KDIR:$KDIR" -v "$KEEPER_DIR:$KEEPER_DIR")
  ENV+=(-e "EXL3_WHX=$KEEPER_SOCK" -e "EXL3_WHX_DIR=$KEEPER_DIR" -e EXL3_WHX_EXCLUSIVE=1)
fi
if [ -n "$DRAFT_WEIGHTS" ]; then
  MOUNTS+=(-v "$DRAFT_WEIGHTS:/data/draft-head.safetensors:ro"); ENV+=(-e EXL3_DRAFT_WEIGHTS=/data/draft-head.safetensors)
fi

SPEC='{"method":"qwen3_5_mtp","num_speculative_tokens":7,"attention_backend":"TRITON_ATTN",'
SPEC+='"draft_sample_method":"probabilistic","rejection_sample_method":"standard",'
SPEC+='"num_speculative_tokens_per_batch_size":[[1,2,7],[3,16,3],[17,32,1]]}'
ARGS=(
  /models --served-model-name qwen3.8-27b --dtype float16 --trust-remote-code
  --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_NUM_SEQS" --max-num-batched-tokens 8192
  --gpu-memory-utilization "$GPU_MEM_UTIL"
  --enable-prefix-caching --mamba-cache-mode align
  --kv-cache-dtype int4_per_token_head --attention-backend TRITON_ATTN
  --speculative-config "$SPEC"
  --compilation-config '{"max_cudagraph_capture_size":1024}'   # graphs for mixed steps / prefills up to 1,024 tokens (derived default 256)
  --limit-mm-per-prompt '{"image":32,"video":0}'
  --structured-outputs-config '{"backend":"xgrammar","disable_any_whitespace":true}'
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder
  --port 8000
)
if [ "$KV_TIER" = 1 ]; then
  PK=""; [ "$PACKED_KV" = 1 ] && PK=',"enable_cross_layers_blocks":"true"'
  KVT=$(printf '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"spec_name":"BrokerOffloadingSpec","spec_module_path":"llmt_kv.cuda_tier","cpu_bytes_to_use":1,"broker_socket":"%s","broker_namespace":"%s","broker_engine_id":2,"broker_name":"rtx5090","broker_timeout_ms":5,"broker_ns_allow":%s,"broker_lease_frac":%s%s}}' \
    "$BROKER_SOCK" "$BROKER_NAMESPACE" "$BROKER_NS_ALLOW" "$BROKER_LEASE_FRAC" "$PK")
  ARGS+=(--kv-transfer-config "$KVT")
  SOCK_DIR=$(dirname "$BROKER_SOCK")
  MOUNTS+=(-v "$SOCK_DIR:$SOCK_DIR" -v "$BROKER_POOL_DIR:$BROKER_POOL_DIR")
  wait_broker "$BROKER_SOCK"
fi

# torch.compile cache keyed by the image and the engine configuration (a cache made for another configuration broke
# a start once); old keys can be deleted from the cache volume at any time.
IMG_ID=$(docker image inspect --format '{{.Id}}' "$IMAGE" 2>/dev/null) || die "image $IMAGE not found (build docker/Dockerfile.rtx5090)"
CC_KEY=$(printf '%s\n' "$IMG_ID" "${ARGS[@]}" "${ENV[@]}" | md5sum | cut -c1-12)
ENV+=(-e VLLM_CACHE_ROOT="/root/.cache/vllm-cc/$CC_KEY")

docker rm -f "$NAME" >/dev/null 2>&1 || true
start_warmup rtx5090 "http://127.0.0.1:$PORT" "$ROUTER_NAME" --max-seqs 16
if [ "$DETACH" = 1 ]; then RUNMODE=(-d); else RUNMODE=(--rm); fi
# shellcheck disable=SC2086  # DOCKER_GPU_ARGS is a list of words
exec docker run "${RUNMODE[@]}" --name "$NAME" $DOCKER_GPU_ARGS --shm-size "$SHM_SIZE" --ulimit memlock=-1:-1 \
  --log-opt max-size=50m --log-opt max-file=2 -p "$BIND:$PORT:8000" \
  "${ENV[@]}" "${MOUNTS[@]}" "$IMAGE" "${ARGS[@]}"
