# Benchmark charts

Marketing-styled charts for the public benchmark comparisons. One look everywhere: white spec-sheet layout, our runtime
in signal orange, everyone else in slate, Barlow / Barlow Condensed (SIL OFL, bundled in `fonts/`).

```
./render.sh                         # all charts -> out/*.png (2x) + out/*.svg
./render.sh --only 5090_aggregate   # one chart (function name without "chart_")
```

`render.sh` runs `make_charts.py` with Python 3.12 + matplotlib 3.9.2 (`CHART_PY=<python>`, else in a `python:3.12-slim` container; matplotlib hits a deepcopy
recursion on the host's Python 3.14). Any Python 3.12 with `matplotlib==3.9.2 numpy==2.1.3` works:
`python3 make_charts.py --out out`.

## Data

- `data/published.json`: other runtimes' numbers exactly as published; each block names its source key in
  `../SOURCES.md`.
- `data/ours.json`: our numbers; each block names the run it came from (raw files in `../results/`).

New results: edit the JSON (e.g. append a GPQA repeat to `official.GPQA Diamond.repeats`) and re-render.

## Charts

| File | Shows |
|---|---|
| `5090-single-stream` | SparkInfer's harness shape per workload, ours vs SparkInfer + DSpark v2 |
| `5090-aggregate` | total tok/s vs concurrent requests, ours vs vLLM 0.28 and SparkInfer |
| `arc-upstream-panel` | upstream exl3xpu's own benchmark (their script and prompts), ours vs published |
| `arc-prefill` | Arc B70 cold prefill of a 32k-token prompt: ours vs upstream exl3xpu and llama.cpp SYCL (published) |
| `quality` | the quality card: same-harness accuracy (BF16 / Q4_K_M / NVFP4 / ours), KL vs BF16 (Q4_K_M vs ours), Qwen's official GPQA Diamond + IFBench |
| `headline` | Hero graphic: Arc B70 vs published RTX 5090 runtimes (1-4 users), RTX 5090 at 16 users, GPQA vs BF16, production cache savings |
| `system` | Two-GPU system graphic: per-GPU KV of today's production engines (RTX 5090 436,585 / Arc 497,302 int4 tokens (keeper-attach starts, 2026-11-13), `kv_capacity.*.int4_production`), the shared 10 GiB RAM tier (`ram_tier`: 186 slots x 3,200 tokens, ~0.60M; GDN-state entries share the slots, so an upper bound), ~1.5M combined, highlights |
| `on-demand` | Reserve nothing, borrow everything: a typical static layout (vision + 3 LoRA loaded, prefill buffer + margin held) vs one KV pool where vision and adapters borrow the coldest KV blocks (RAM tier keeps the copies); RTX 5090 budget to scale (`memory_pool`); gains from measured memory budgets; status in the footnote |
| `tail-reuse` | Explainer: what a tool-result turn recomputes under stock vLLM block-rounded caching vs our token-exact tail hold (illustrative turn + modeled medians on real tool turns) |
| `task-time` | Modeled replay of 340 multi-agent production tasks on each engine (both GPUs vs published alternatives), calibrated against the measured 5090 |
| `quality-32k-vs-nvfp4` | GPQA Diamond and AIME 2025 at the 32k thinking budget of the published RTX 5090 NVFP4 runs |
| `cache-production` | real 5090 agent traffic: where prompt tokens came from, prefill time saved |

## Add a chart

Write `chart_<name>(P, O, out)` in `make_charts.py`: build it with `style.frame(title, subtitle, source=...)`, color our
series `style.OURS` and others `style.OTHERS[i]`, label values with `style.bar_labels`, and return
`style.save(fig, out, "<file-name>")`. It is picked up automatically.

All throughput and acceptance numbers in these charts were measured with production's traffic-ranked draft
vocabularies and K-smoothing table, not with the public files the launchers ship (`data/`); see `benchmarks/README.md`.
