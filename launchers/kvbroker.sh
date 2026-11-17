#!/usr/bin/env bash
# AllRunner: the shared host-RAM KV tier (AllBroker's llmt-kvbroker) with the production settings for int4 KV.
#
# STATUS: DRAFT. The flags are the production broker's; this wrapper itself has only been syntax-checked.
#
# Build the broker from echowarp/AllBroker (cargo build --release -> target/release/llmt-kvbroker) and run this as
# root (it creates the socket and pool directories) under a supervisor; AllBroker's contrib/systemd/llmt-kvbroker.service
# is an example unit: point its ExecStart at this script, or copy the flags below into it (that unit's own defaults,
# 2 MiB slots and an SSD tier, are NOT the production settings). Start it before the engines.
#
# One slot = one vLLM CPU block of the int4 KV cache incl. GDN state: 17 x 3,379,200 B = 56,100 KiB for a
# 3,200-token block of Qwen3.8-27B with int4_per_token_head KV. The connector requires slot == block size, so a broker
# serves engines of one KV format only (int8 KV would need 57,460 KiB slots and its own broker).
set -euo pipefail

# ---- site settings ---------------------------------------------------------------------------------------------------
BROKER_BIN=${BROKER_BIN:-/usr/local/bin/llmt-kvbroker}
BROKER_SOCK=${BROKER_SOCK:-/run/llmt-kv-i4/broker.sock}   # engines mount its directory (same path inside)
BROKER_POOL_DIR=${BROKER_POOL_DIR:-/dev/shm/llmt-kv-i4}   # tmpfs; engines mount it at the same path
POOL_GIB=${POOL_GIB:-16}                                  # production: 16 GiB of host RAM for both GPUs
SLOT_KIB=${SLOT_KIB:-56100}
METRICS=${METRICS:-127.0.0.1:9113}                        # Prometheus metrics; keep on localhost
# ----------------------------------------------------------------------------------------------------------------------

[ -x "$BROKER_BIN" ] || { echo "kvbroker.sh: $BROKER_BIN not found (cargo build --release in AllBroker)" >&2; exit 1; }
# The pool lives in /dev/shm: it must fit next to everything else that uses /dev/shm (an engine that writes a page
# tmpfs cannot allocate gets SIGBUS; the broker refuses to start if the ceiling cannot fit).
mkdir -p "$(dirname "$BROKER_SOCK")" "$BROKER_POOL_DIR"

# --no-hugepages-ctl / --no-mlock: plain /dev/shm pages (the Arc connector copies through pageable memory; the 5090
#   connector pins per slot itself)
# --new-pool: start with an empty pool (entries do not survive a broker restart)
# --hotset-dir / --events-dir / --disk-dir / --chains off: no hot-set snapshot, no event log, no SSD tier, no lookup
#   chains (production runs RAM only)
# --mem-reserve-gib 0: no extra host-RAM reserve; --stats-log-s 600: one statistics line every 10 min (numbers only)
exec "$BROKER_BIN" --pool-gib "$POOL_GIB" --slot-kib "$SLOT_KIB" --pool-dir "$BROKER_POOL_DIR" \
  --socket "$BROKER_SOCK" --metrics "$METRICS" \
  --no-hugepages-ctl --no-mlock --new-pool \
  --hotset-dir off --events-dir off --disk-dir off --chains off \
  --mem-reserve-gib 0 --stats-log-s 600
