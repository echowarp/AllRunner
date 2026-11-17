#!/usr/bin/env bash
# AllRunner: Qwen3.8-27B (EXL3 4.0 bpw) on one Intel Arc Pro B70 with vLLM 0.26 XPU + exl3xpu, production configuration.
#
# STATUS: DRAFT, NOT RUN AS WRITTEN. It reproduces the flags and environment of the production Arc engine, which was
# started by a different (site-specific) script; this file has only been syntax-checked. See launchers/README.md.
#
# Image: docker/Dockerfile.arc (exl3xpu base image + echowarp/exl3xpu publish-20261113 + echowarp/vllm publish-20261113-xpu +
# llmt_kv from AllBroker). The image's entrypoint is exl3xpu's scripts/serve.py: settings are passed as
# --set vllm.<field>=... and --set env.<VAR>=... (serve.py applies the model.yaml env over the process env, so engine
# variables go through --set env., not docker -e).
# Runs in the foreground (for a supervisor such as systemd or llama-swap); DETACH=1 runs it detached.
# Stop: docker stop -t 30 "$NAME".
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=launchers/common.sh
. "$HERE/common.sh"
ROOT=$(cd "$HERE/.." && pwd)                           # AllRunner checkout (data/ holds the shipped public files)

# ---- site settings: every value can be overridden from the environment -------------------------------------------
MODEL_DIR=${MODEL_DIR:-/models/Qwen3.8-27B-exl3-4.00bpw}  # turboderp/Qwen3.8-27B-exl3, branch 4.00bpw (EXL3 4.0 bpw)
IMAGE=${IMAGE:-allrunner-arc:latest}                  # built from docker/Dockerfile.arc
NAME=${NAME:-allrunner-arc}                           # container name
BIND=${BIND:-127.0.0.1}                               # host address; keep it on localhost (no auth on the engine)
PORT=${PORT:-8000}                                    # host port (the router expects the Arc on 8000)
GPU_INDEX=${GPU_INDEX:-0}                             # Level Zero device index (serve.py --gpu -> ZE_AFFINITY_MASK)
CACHE_VOLUME=${CACHE_VOLUME:-allrunner-arc-cache}     # docker volume (or host dir) for /root/.cache (IGC/NEO caches)
SHM_SIZE=${SHM_SIZE:-32g}
HASHSEED_FILE=${HASHSEED_FILE:-/etc/allrunner/hashseed.env}  # PYTHONHASHSEED=..., mode 0600, same file for every engine
# Shared host-RAM KV tier (launchers/kvbroker.sh). The pool directory is mounted at the same path inside the
# container (the broker hands out its host path).
KV_TIER=${KV_TIER:-1}                                 # 0 = no RAM tier (no kv_transfer_config)
BROKER_SOCK=${BROKER_SOCK:-/run/llmt-kv-i4/broker.sock}
BROKER_POOL_DIR=${BROKER_POOL_DIR:-/dev/shm/llmt-kv-i4}
BROKER_NAMESPACE=${BROKER_NAMESPACE:-arc-b70-i4}      # this engine's own namespace in the broker
BROKER_NS_ALLOW=${BROKER_NS_ALLOW:-'["shared"]'}      # namespaces a request may pick (the router injects "shared")
BROKER_LEASE_FRAC=${BROKER_LEASE_FRAC:-0.75}          # share of the pool's slots this engine may hold leased at once
# Data files: the defaults are public builds shipped in data/ (no production traffic in them); "none" = feature off.
#   KSMOOTH_TABLE: K-smoothing table. Default data/ksmooth/public_r.pt (exl3xpu scripts/public_ksmooth.py, public text
#     only; inputs and sha256 in public_r.pt.inputs.json). Measured on the Arc (int4 KV, held-out GSM8K) vs a table built
#     on production traffic: NLL +0.037 %; without a table (KSMOOTH_TABLE=none) NLL +0.43 %. Speed: the table enables
#     the fused int4 KV write: about +6-7 % on a fast host, up to +35 % per-stream decode on a slow, host-bound CPU.
#     Engines that share the RAM tier must use the SAME table file (its hash is part of the tier's compatibility tag).
KSMOOTH_TABLE=${KSMOOTH_TABLE:-$ROOT/data/ksmooth/public_r.pt}
[ "$KSMOOTH_TABLE" = none ] && KSMOOTH_TABLE=""
#   DRAFT_VOCAB / DRAFT_VOCAB_FAR: 512-block near and 128-block far draft vocabularies. Default: the public pair shipped
#     in data/draft-vocab/ (exl3xpu scripts/public_draft_vocab.py, public inputs with sha256s in the files). Your own
#     pair ranked on text like your traffic (exl3xpu scripts/draft_vocab.py) can replace them. DRAFT_VOCAB=generic =
#     upstream's 512-block draft_vocab.json in the image; DRAFT_VOCAB=none = full lm_head drafting (much more weight
#     traffic per draft pass; DRAFT-DYN, DRAFT-Q4 and DS-SUB, which work on the pruned heads, are inert).
#     DRAFT_VOCAB_FAR=none: every draft position uses the near head.
DRAFT_VOCAB=${DRAFT_VOCAB:-$ROOT/data/draft-vocab/public-512.json}
DRAFT_VOCAB_FAR=${DRAFT_VOCAB_FAR:-$ROOT/data/draft-vocab/public-128.json}
[ "$DRAFT_VOCAB" = none ] && DRAFT_VOCAB=""
[ "$DRAFT_VOCAB_FAR" = none ] && DRAFT_VOCAB_FAR=""
# LoRA adapters (production: on). LORA=1: hot-loaded adapters in lent KV blocks (AllBroker llmt_kv.lend), up to 4 on
# the GPU, rank <= 16; LORA_DIR = host directory with adapters (mounted at /loras; empty = load by path at runtime only).
LORA=${LORA:-1}
LORA_DIR=${LORA_DIR:-}
# Weight keeper (AllBroker llmt-keeper@arc): used when its socket exists (weights stay on the GPU across restarts).
KEEPER_SOCK=${KEEPER_SOCK:-/run/llmt-keeper-arc/keeper.sock}
KEEPER_DIR=${KEEPER_DIR:-/var/lib/llmt/keeper/arc/side}   # keeper side files (DRAFT-DYN tables) on disk
MAX_MODEL_LEN=${MAX_MODEL_LEN:-196608}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.94}                    # OOR fix, part 1 of 3 (with TORCH_CAP_MIB and PYTORCH_ALLOC_CONF)
TORCH_CAP_MIB=${TORCH_CAP_MIB:-29100}                 # torch XPU allocator cap (exl3xpu tools/oor), 32 GB card
DETACH=${DETACH:-0}
# Post-start warm-up (launchers/llmt-warmup.py, README "Warm-up"): synthetic requests over the shapes with a first-use
# cost, router backend ROUTER_NAME drained meanwhile if another backend is healthy. WARMUP=0 = off; ROUTER_NAME='' = no drain.
WARMUP=${WARMUP:-1}
ROUTER_NAME=${ROUTER_NAME-arc}                        # this engine's backend name in the router config
ROUTER_URL=${ROUTER_URL:-http://127.0.0.1:8010}
# ----------------------------------------------------------------------------------------------------------------------

load_hashseed "$HASHSEED_FILE"
[ -d "$MODEL_DIR" ] || die "MODEL_DIR $MODEL_DIR not found"
check_file "$KSMOOTH_TABLE" KSMOOTH_TABLE
[ "$DRAFT_VOCAB" = generic ] || check_file "$DRAFT_VOCAB" DRAFT_VOCAB
check_file "$DRAFT_VOCAB_FAR" DRAFT_VOCAB_FAR
[ -z "$LORA_DIR" ] || [ -d "$LORA_DIR" ] || die "LORA_DIR $LORA_DIR not found"

# At boot the xe driver may bind the GPU after docker starts: wait (up to 3 min) for a render node.
for i in $(seq 1 180); do
  compgen -G '/dev/dri/renderD*' >/dev/null && break
  [ "$i" = 180 ] && note "warning: no /dev/dri/renderD* after 180 s, starting anyway"
  sleep 1
done

# docker -e: read by the process before serve.py (allocator, memprobe .pth hook, hash seed)
DENV=(
  -e PYTHONHASHSEED                       # by name: the value comes from $HASHSEED_FILE
  -e HF_HUB_OFFLINE=1 -e ZES_ENABLE_SYSMAN=1
  -e TORCH_CAP_MIB="$TORCH_CAP_MIB" -e MEMPROBE_S=30            # OOR fix, part 2 (llmt_memprobe, logs numbers only)
  -e PYTORCH_ALLOC_CONF=max_split_size_mb:512,garbage_collection_threshold:0.85   # OOR fix, part 3
)
MOUNTS=(--device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path:ro -v "$MODEL_DIR:/models:ro" -v "$CACHE_VOLUME:/root/.cache")

SETS=(
  --set served_model_name=qwen3.8-27b
  # vLLM fields (stock vLLM 0.26 unless noted); the image's model.yaml supplies dtype float16, max_num_seqs 16,
  # prefix caching, mamba_cache_mode align, FULL_DECODE_ONLY graphs and the image-size caps
  --set vllm.max_model_len="$MAX_MODEL_LEN"
  --set vllm.gpu_memory_utilization="$GPU_MEM_UTIL"
  --set vllm.max_num_batched_tokens=8192
  --set vllm.kv_cache_dtype=int4_per_token_head
  --set vllm.attention_backend=TRITON_ATTN
  --set vllm.speculative_config.attention_backend=TRITON_ATTN
  --set vllm.speculative_config.num_speculative_tokens=7
  --set 'vllm.speculative_config.num_speculative_tokens_per_batch_size=[[1,4,7],[5,5,5],[6,6,4],[7,8,3],[9,256,2]]'
  --set vllm.speculative_config.draft_sample_method=probabilistic
  --set 'vllm.structured_outputs_config={"backend":"xgrammar","disable_any_whitespace":true}'
  --set 'vllm.limit_mm_per_prompt={"image":32,"video":0}'
  --set vllm.reasoning_parser=qwen3 --set vllm.enable_auto_tool_choice=true --set vllm.tool_call_parser=qwen3_coder
  # stock vLLM environment
  --set env.VLLM_KV_CACHE_LAYOUT=NHD
  --set env.VLLM_PREFIX_CACHE_RETENTION_INTERVAL=12800
  # exl3xpu (echowarp/exl3xpu README-ARC.md)
  --set env.EXL3_SHAPE_P_FILE=/opt/exl3xpu/models/qwen3.8-27b-exl3-4.00bpw/shape_p_c.json --set env.EXL3_MAX_MB=128
  --set env.EXL3_INT8_PREFILL=1
  --set env.EXL3_SPEC_HP=1 --set env.EXL3_SPEC_HP_BMAX=8 --set env.EXL3_SPEC_VARR_MAXL=84000
  --set env.EXL3_I8_SPLITS=80,40,64,40 --set env.EXL3_I8_ONEDNN_MAXLEN=60000
  --set env.EXL3_GDN_RECOMPUTE=1 --set env.EXL3_RC_FUSED_IN=1 --set env.EXL3_GDN_RPS=4
  --set env.EXL3_GDN_PORT=/opt/exl3xpu/exl3xpu/gdnchunk.so --set env.EXL3_GDN_PORT_TOKINDX=1
  --set env.EXL3_DYN_SD_FULL=2 --set env.EXL3_UNIFORM_SPEC=1
  --set env.EXL3_DET_LINEAR=1 --set env.EXL3_FAST_NAN_GUARD=1
  --set env.EXL3_LAZY_VISION=1 --set env.EXL3_LAZY_VISION_NS=weights:qwen3.8-27b-visual   # tower entry shared by both GPUs
  --set env.EXL3_LAZY_VISION_ENGINE_ID=1
  --set env.EXL3_KV_DTYPE_TAG=int4_per_token_head   # KV dtype in the RAM-tier compatibility tag (llmt_kv)
  # vLLM branch publish-20261113-xpu (echowarp/vllm)
  --set env.EXL3_RC_PINNED=1 --set env.EXL3_ALIGN_NOSYNC=1 --set env.EXL3_SPEC_STOP_TRUNC=1
  --set env.EXL3_EAGLE_KEEP=2
  --set env.EXL3_TAIL_HOLD=1 --set env.EXL3_TAIL_HOLD_S=60 --set env.EXL3_TAIL_HOLD_TOOL_S=300
  --set env.EXL3_TAIL_HOLD_FREE=0.15 --set env.EXL3_TAIL_HOLD_MAX=16
  --set env.EXL3_SCHED_DECODE_STEPS=0 --set env.EXL3_SCHED_MIXED_K=4 --set env.EXL3_SCHED_MIXED_CHUNK=0
  --set env.EXL3_DYN_SD_DECODE_COUNT=1
  --set env.EXL3_GPU_EVICT_ORDER=chain --set env.EXL3_MAMBA_KEEP_LAST=2 --set env.EXL3_ENCODER_CACHE_TOKENS=16384
  --set env.EXL3_STATE_STORE=prompt_end
  --set env.EXL3_KV_COPY_PACE_MB=8 --set env.EXL3_KV_RING_MB=32 --set env.EXL3_KV_RING_LOAD_MB=256
  --set env.EXL3_KV_RING_LOAD_IDLE_S=86400
  # Intel compute runtime: wait for GPU completion in the kernel driver instead of busy-polling (frees CPU cores)
  --set env.NEOReadDebugKeys=1 --set env.OverrideEnableKmdNotify=1 --set env.OverrideKmdNotifyDelayMicroseconds=200
  --set env.EnableUserFenceForCompletionWait=1 --set env.EventHostSynchronizeLinuxUserFenceKmdWait=1
  --set env.OverrideUseKmdWaitFunction=1
)
if [ -n "$KSMOOTH_TABLE" ]; then
  MOUNTS+=(-v "$KSMOOTH_TABLE:/data/ksmooth.pt:ro"); SETS+=(--set env.EXL3_KSMOOTH=/data/ksmooth.pt)
else
  SETS+=(--set env.EXL3_KSMOOTH=null)   # drop the model.yaml default (the image's copy of the public table)
fi
if [ -z "$DRAFT_VOCAB" ]; then
  SETS+=(--set env.EXL3_DRAFT_VOCAB=null --set env.EXL3_DRAFT_VOCAB_FAR=null)   # null drops the model.yaml defaults: full-head drafting
else
  if [ "$DRAFT_VOCAB" = generic ]; then
    SETS+=(--set 'env.EXL3_DRAFT_VOCAB={model_dir}/draft_vocab.json')
  else
    MOUNTS+=(-v "$DRAFT_VOCAB:/data/draft-vocab.json:ro"); SETS+=(--set env.EXL3_DRAFT_VOCAB=/data/draft-vocab.json)
  fi
  if [ -n "$DRAFT_VOCAB_FAR" ]; then
    MOUNTS+=(-v "$DRAFT_VOCAB_FAR:/data/draft-vocab-far.json:ro")
    SETS+=(--set env.EXL3_DRAFT_VOCAB_FAR=/data/draft-vocab-far.json --set env.EXL3_DRAFT_FAR_FROM=2)
  else
    SETS+=(--set env.EXL3_DRAFT_VOCAB_FAR=null)   # drop the model.yaml default: the near head everywhere
  fi
  SETS+=(--set env.EXL3_DRAFT_Q4=64 --set env.EXL3_DS_SUB=1
         --set env.EXL3_DRAFT_DYN=256 --set env.EXL3_DRAFT_DYN_NEAR=128 --set env.EXL3_DRAFT_BIAS=0)
fi
XARGS=()
if [ "$LORA" = 1 ]; then
  SETS+=(--set env.EXL3_LORA_HOT=1 --set env.VLLM_ALLOW_RUNTIME_LORA_UPDATING=1 --set env.EXL3_LORA_DIR=/loras
         --set env.EXL3_LORA_MAX_RANK=16 --set env.EXL3_LORA_MARGIN_MB=8192)
  [ -z "$LORA_DIR" ] || MOUNTS+=(-v "$LORA_DIR:/loras:ro")
  XARGS=(-- --enable-lora --max-loras 4 --max-lora-rank 16 --max-cpu-loras 8 --lora-target-modules q_proj k_proj v_proj
         o_proj gate_proj up_proj down_proj in_proj_a in_proj_b in_proj_qkv in_proj_z out_proj)
fi
if [ -S "$KEEPER_SOCK" ]; then
  mkdir -p "$KEEPER_DIR"
  KDIR=$(dirname "$KEEPER_SOCK")
  MOUNTS+=(-v "$KDIR:$KDIR" -v "$KEEPER_DIR:$KEEPER_DIR")
  SETS+=(--set "env.EXL3_WHX=$KEEPER_SOCK" --set "env.EXL3_WHX_DIR=$KEEPER_DIR" --set env.EXL3_WHX_EXCLUSIVE=1
         --set env.EXL3_WHX_MARGIN_MB=768)   # attach margin: keeps the OOR headroom with held weights
fi
if [ "$KV_TIER" = 1 ]; then
  KVT=$(printf '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"spec_name":"BrokerOffloadingSpec","spec_module_path":"llmt_kv.xpu_tier","cpu_bytes_to_use":1,"broker_socket":"%s","broker_namespace":"%s","broker_lease_frac":%s,"broker_ns_allow":%s}}' \
    "$BROKER_SOCK" "$BROKER_NAMESPACE" "$BROKER_LEASE_FRAC" "$BROKER_NS_ALLOW")
  SETS+=(--set "vllm.kv_transfer_config=$KVT")
  SOCK_DIR=$(dirname "$BROKER_SOCK")
  MOUNTS+=(-v "$SOCK_DIR:$SOCK_DIR" -v "$BROKER_POOL_DIR:$BROKER_POOL_DIR")
  wait_broker "$BROKER_SOCK"
fi

docker image inspect "$IMAGE" >/dev/null 2>&1 || die "image $IMAGE not found (build docker/Dockerfile.arc)"
docker rm -f "$NAME" >/dev/null 2>&1 || true
start_warmup arc "http://127.0.0.1:$PORT" "$ROUTER_NAME"
if [ "$DETACH" = 1 ]; then RUNMODE=(-d); else RUNMODE=(--rm); fi
exec docker run "${RUNMODE[@]}" --name "$NAME" --shm-size "$SHM_SIZE" \
  --log-opt max-size=50m --log-opt max-file=2 -p "$BIND:$PORT:8000" \
  "${DENV[@]}" "${MOUNTS[@]}" "$IMAGE" \
  models/qwen3.8-27b-exl3-4.00bpw --gpu "$GPU_INDEX" --port 8000 --model-path /models "${SETS[@]}" "${XARGS[@]}"
