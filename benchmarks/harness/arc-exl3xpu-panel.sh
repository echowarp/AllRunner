#!/usr/bin/env bash
# Upstream exl3xpu's own benchmark (bench/sweep.py) against a running Arc engine, with the settings of the published
# panel: real-text corpus, thinking on, T 0.7, decode at 1/2/4/8 concurrent streams (prose + code), prefill at
# 2k / 8k / 32k tokens. Start the engine first (launchers/arc-b70.sh) and keep the router away from it (drain it).
#   EXL3XPU_UPSTREAM=<checkout of the upstream exl3xpu repo> benchmarks/harness/arc-exl3xpu-panel.sh [label]
# Fill the upstream corpus once with its bench/fetch_corpus.sh (public Gutenberg books, HumanEval, Python source).
set -euo pipefail
X=${EXL3XPU_UPSTREAM:?set EXL3XPU_UPSTREAM to a checkout of the upstream exl3xpu repository}
BASE=${BASE:-http://127.0.0.1:8000}
MODEL=${MODEL:-qwen3.8-27b}
OUT=${OUT:-$PWD/x3-results.jsonl}
LABEL=${1:-arc-panel-$(date +%Y%m%d)}
PY=${PYTHON:-python3}                                # needs the upstream bench requirements
cd "$X"
timeout 3600 "$PY" bench/sweep.py --base "$BASE" --model "$MODEL" --corpus real --thinking \
  --temperature 0.7 --max-tokens 16384 --decode-c 1,2,4,8 --classes prose,code --prefill-c 1 \
  --prefill-ctx 2048,8192,32768 --label "$LABEL" --out "$OUT"
