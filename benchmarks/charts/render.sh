#!/usr/bin/env bash
# Render the charts. Needs Python 3.12 with matplotlib 3.9 and numpy 2.1 (matplotlib 3.9/3.10 hit a deepcopy recursion
# on Python 3.14). Either point CHART_PY at such an interpreter, or let this script build a venv in a python:3.12
# container (network for pip; nothing else is mounted).   ./render.sh [--only name,...]
set -euo pipefail
D=$(cd "$(dirname "$0")" && pwd)
if [ -n "${CHART_PY:-}" ]; then
  cd "$D" && exec "$CHART_PY" make_charts.py "$@"
fi
IMG=${IMG:-python:3.12-slim}
docker run --rm -m 3g -v "$D":/charts -w /charts "$IMG" bash -c \
  'pip install -q --no-cache-dir matplotlib==3.9.2 numpy==2.1.3 && python make_charts.py "$@" 2>&1 | grep -v -i findfont' \
  render "$@"
