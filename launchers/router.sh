#!/usr/bin/env bash
# AllRunner: llmt-router (router/llmt-router.py), the prefix-affinity router in front of the engines.
#
# STATUS: DRAFT. Same command line as production; this wrapper itself has only been syntax-checked.
#
# The router does not authenticate anyone: keep it on localhost behind an OpenAI-compatible gateway that holds the
# API key (production uses llama-swap: its public model name proxies to http://127.0.0.1:8010, checkEndpoint /health).
# Any gateway works; without one, only local clients can reach the router.
#
# Config: copy router/router.example.json to $ROUTER_CONFIG and edit the backends (URLs, max_ctx, max_images) and the
# measured rates (prefill, dec) for your hardware; the file is re-read when it changes. Its "inject_kv_ns": "shared"
# makes both engines use the broker's shared namespace (it must be in each engine's BROKER_NS_ALLOW).
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# ---- site settings ---------------------------------------------------------------------------------------------------
PYTHON=${PYTHON:-python3}                                  # Python 3.10+, standard library only
ROUTER=${ROUTER:-$HERE/../router/llmt-router.py}
ROUTER_CONFIG=${ROUTER_CONFIG:-/etc/allrunner/router.json} # from router/router.example.json; its "listen" wins
DECISION_LOG=${DECISION_LOG:-/var/log/llmt-router/decisions.log}   # one line per routing decision, no content; '' = stderr
ROUTER_POLICY=${ROUTER_POLICY:-}                         # optional session + priority policy file (router/router-policy.example.json)
export LLMT_ROUTER_DRAIN_DIR=${LLMT_ROUTER_DRAIN_DIR:-/run/llmt-router}   # drain-<name> / hold-<name> files
# ----------------------------------------------------------------------------------------------------------------------

[ -f "$ROUTER_CONFIG" ] || { echo "router.sh: $ROUTER_CONFIG missing: cp router/router.example.json $ROUTER_CONFIG and edit it" >&2; exit 1; }
mkdir -p "$LLMT_ROUTER_DRAIN_DIR"
[ -z "$DECISION_LOG" ] || mkdir -p "$(dirname "$DECISION_LOG")"
POLICY_ARGS=()
if [ -n "$ROUTER_POLICY" ]; then POLICY_ARGS=(--policy-file "$ROUTER_POLICY"); fi
exec "$PYTHON" "$ROUTER" --config "$ROUTER_CONFIG" --decision-log "$DECISION_LOG" "${POLICY_ARGS[@]}"
