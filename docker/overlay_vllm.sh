#!/usr/bin/env bash
# overlay_vllm.sh <vllm git checkout> <base commit> [<branch commit, default HEAD>]
#
# Install an echowarp/vllm branch over the vLLM that the base image already has, by copying only the Python files the
# branch changes relative to <base>. Used by Dockerfile.rtx5090 and Dockerfile.arc.
#
# Why not `pip install` the branch: that rebuilds vLLM's CUDA/SYCL extensions (hours, and for XPU a toolchain the
# base images do not carry), or depends on precompiled-wheel downloads keyed by upstream commits. Our branches change
# Python files only, so copying them is exact as long as the installed files are the ones the branch started from.
# This script checks that first: every file it replaces must be byte-identical to <base>'s version, otherwise the
# build stops (set OVERLAY_FORCE=1 to overwrite anyway). Files the branch adds are copied; deleted or renamed files
# stop the build.
set -euo pipefail
SRC=$1
BASE=$2
REF=${3:-HEAD}
git -C "$SRC" cat-file -e "$BASE^{commit}" || { echo "overlay: base $BASE not in $SRC" >&2; exit 1; }
SITE=$(python3 -c 'import importlib.util, os; print(os.path.dirname(importlib.util.find_spec("vllm").origin))')
echo "overlay: installed vLLM at $SITE; branch $(git -C "$SRC" rev-parse --short "$REF") over base $(git -C "$SRC" rev-parse --short "$BASE")"

other=$(git -C "$SRC" diff --name-only --diff-filter=DRCTUXB "$BASE" "$REF" -- vllm/)
[ -z "$other" ] || { echo "overlay: deleted/renamed files not supported:" >&2; echo "$other" >&2; exit 1; }

n=0 bad=0
changed=()
while IFS= read -r f; do
  [ -n "$f" ] || continue
  dst="$SITE/${f#vllm/}"
  if [ -e "$dst" ]; then
    want=$(git -C "$SRC" show "$BASE:$f" | sha256sum | cut -d' ' -f1)
    have=$(sha256sum "$dst" | cut -d' ' -f1)
    if [ "$want" != "$have" ]; then
      echo "overlay: $f: installed file differs from the base version" >&2
      bad=$((bad + 1))
    fi
  elif git -C "$SRC" cat-file -e "$BASE:$f" 2>/dev/null; then
    echo "overlay: $f: in the base commit but not installed" >&2
    bad=$((bad + 1))
  fi
  changed+=("$f")
done < <(git -C "$SRC" diff --name-only --diff-filter=AM "$BASE" "$REF" -- vllm/)

if [ "$bad" -gt 0 ] && [ "${OVERLAY_FORCE:-0}" != 1 ]; then
  echo "overlay: $bad file(s) do not match the base; the installed vLLM is not the one this branch was made for" >&2
  exit 1
fi
for f in "${changed[@]}"; do
  dst="$SITE/${f#vllm/}"
  mkdir -p "$(dirname "$dst")"
  git -C "$SRC" show "$REF:$f" > "$dst"
  python3 -m py_compile "$dst"
  n=$((n + 1))
done
echo "overlay: $n file(s) installed"
