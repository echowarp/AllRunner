#!/bin/bash
# Download the public prompt sets used by the replica benchmarks (no private data). Usage: fetch_data.sh <dir>
set -eu
O=${1:-./data}; mkdir -p "$O"; cd "$O"
curl -sfL -o HumanEval.jsonl.gz https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz && gunzip -f HumanEval.jsonl.gz
curl -sfL -o gsm8k_test.jsonl https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl
curl -sfL -o mt_bench_question.jsonl https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl
curl -sfL -o pg1342.txt https://www.gutenberg.org/cache/epub/1342/pg1342.txt
for b in 2701 2600; do curl -sfL -o pg$b.txt https://www.gutenberg.org/cache/epub/$b/pg$b.txt; done
# NousResearch/json-mode-eval (parquet) -> jsonl (needs pyarrow; any python with pyarrow works)
curl -sfL -o json-mode-eval.parquet "https://huggingface.co/datasets/NousResearch/json-mode-eval/resolve/main/data/train-00000-of-00001.parquet"
${PYARROW_PY:-python3} - <<'PY'
import json, pyarrow.parquet as pq
t = pq.read_table("json-mode-eval.parquet").to_pylist()
with open("json_mode_eval.jsonl", "w") as f:
    for r in t:
        f.write(json.dumps(r, default=str) + "\n")
print("json-mode-eval rows:", len(t))
PY
sha256sum HumanEval.jsonl gsm8k_test.jsonl mt_bench_question.jsonl pg1342.txt json_mode_eval.jsonl > SHA256SUMS
wc -l HumanEval.jsonl gsm8k_test.jsonl mt_bench_question.jsonl json_mode_eval.jsonl
