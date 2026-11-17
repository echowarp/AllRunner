# shellcheck shell=bash
# Shared helpers for the AllRunner launchers (sourced, not executed).

die() { echo "${0##*/}: $*" >&2; exit 1; }
note() { echo "${0##*/}: $*" >&2; }

# PYTHONHASHSEED: vLLM's prefix-cache block hashes depend on it. Every engine that shares one KV broker must use the
# same value, and it must stay the same across restarts, or entries in the host-RAM tier are never found again
# (the cache still works, it just misses). Production treats the value as a secret: anyone who knows it can compute
# the block keys of a given prompt. The operator creates the file once, e.g.
#   sudo install -d -m 0755 /etc/allrunner
#   echo "PYTHONHASHSEED=$(python3 -c 'import secrets; print(secrets.randbelow(2**32 - 1) + 1)')" \
#     | sudo install -m 0600 /dev/stdin /etc/allrunner/hashseed.env
# The value is passed to docker by name (-e PYTHONHASHSEED), so it never appears on a command line.
load_hashseed() {
  local f=$1
  [ -f "$f" ] || die "hash seed file $f is missing: create it (see launchers/README.md, 'PYTHONHASHSEED'); refusing to start"
  [ -r "$f" ] || die "hash seed file $f is not readable by $(id -un) (run the launcher as root or give this user read access)"
  local mode
  mode=$(stat -c %a "$f")
  case "$mode" in
    600|400) ;;
    *) note "warning: $f has mode $mode; 0600 recommended (the seed is a secret)" ;;
  esac
  unset PYTHONHASHSEED
  # shellcheck disable=SC1090
  set -a; . "$f"; set +a
  case "${PYTHONHASHSEED:-}" in
    ''|*[!0-9]*) die "$f must set PYTHONHASHSEED to an integer in 1..4294967295" ;;
  esac
  if [ "$PYTHONHASHSEED" -lt 1 ] || [ "$PYTHONHASHSEED" -gt 4294967295 ]; then
    die "$f: PYTHONHASHSEED out of range (1..4294967295)"
  fi
  export PYTHONHASHSEED
}

# check_file <path or ""> <variable name>: an optional data file must exist when it is set.
check_file() {
  [ -z "$1" ] || [ -f "$1" ] || die "$2: file $1 not found"
}

# Wait until the broker socket exists (the engine itself waits 30 s more and then starts with the RAM tier off).
wait_broker() {
  local sock=$1
  for _ in $(seq 1 30); do
    [ -S "$sock" ] && return 0
    sleep 1
  done
  note "warning: no broker socket at $sock after 30 s; the engine will start with the RAM tier OFF"
}

# start_warmup <profile> <engine url> <router backend name or ""> [extra llmt-warmup.py args]: post-start warm-up
# (launchers/llmt-warmup.py), detached and at idle CPU priority, so the first real requests of each shape do not pay
# one-time per-process costs. With a backend name it drains that backend in the router while it runs, but only when
# another backend is healthy and undrained (launchers/README.md, "Warm-up"). WARMUP=0 turns it off. Call it right
# before the engine's exec: it waits for the engine's /health itself.
start_warmup() {
  [ "${WARMUP:-1}" = 1 ] || return 0
  local profile=$1 url=$2 gate=$3; shift 3
  local py=${PYTHON:-python3} log=${WARMUP_LOG:-/var/log/allrunner/warmup-$profile.log}
  command -v "$py" >/dev/null 2>&1 || { note "warning: no $py: warm-up skipped"; return 0; }
  mkdir -p "$(dirname "$log")" 2>/dev/null || log=/dev/null
  local drain=${LLMT_ROUTER_DRAIN_DIR:-/run/llmt-router} state=${WARMUP_STATE_DIR:-/run/allrunner-warmup}
  local args=(--profile "$profile" --url "$url" --router "${ROUTER_URL:-http://127.0.0.1:8010}" --drain-dir "$drain"
              --state-dir "$state" --health-timeout 1500 --hard-timeout "${WARMUP_HARD_TIMEOUT:-300}" "$@")
  [ -z "$gate" ] || args+=(--gate "$gate")
  # after the warm-up exits (however it exits: SIGKILL, OOM), --cleanup removes a drain it left behind
  ( W_PY=$py W_TOOL=$HERE/llmt-warmup.py W_GATE=$gate W_DRAIN=$drain W_STATE=$state setsid nice -n 19 bash -c '
      timeout -k 10 1800 "$W_PY" "$W_TOOL" "$@"
      [ -z "$W_GATE" ] || "$W_PY" "$W_TOOL" --cleanup "$W_GATE" --drain-dir "$W_DRAIN" --state-dir "$W_STATE"' \
      _ "${args[@]}" >> "$log" 2>&1 < /dev/null & )
}
