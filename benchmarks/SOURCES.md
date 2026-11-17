# Published numbers we compare against

Every external figure used in this directory, with its source. Retrieved 2026-11-10 unless noted. Trust: **V** =
vendor or project self-report, **R** = reviewer with a stated method, **F** = single community post. Numbers are copied
as published; where a page left a setting out, it says "not stated".

## Qwen3.8-27B, RTX 5090 (32 GB)

**Headline / single-stream / aggregate charts (RTX 5090, ours):** `chart5090-20261116-shipped` (single stream, median of 3 reps) and `chart5090-20261116-shipped` + `chart5090-20261116-head` (aggregate, pooled median of 6 reps): RTX 5090 on shipped defaults (public draft vocabulary, public K-smoothing table, DYN / Q4, draft graphs, no traffic-derived draft head), `harness/yieldbench.py`, held out of traffic by the router; aggregate 293.4 / 485.4 / 650.5 / 869.7 / 1,262.5 / 1,337.6 tok/s at 1 / 2 / 4 / 8 / 16 / 32 users; single stream chat 238.7, code 458.5, math 319.0, JSON 426.7. Workload-tuned tables (our production configuration) are within ~1-3 % higher.

| Runtime | Setup | Published result | Source | Trust |
|---|---|---|---|---|
| SparkInfer v0.5.5 + DSpark v2 (NVFP4) | own bench harness: one prompt per workload, 256 tokens, greedy, a few hundred tokens of context | 264.8 tok/s overall; chat 174.4, code 420.2, math 276.5, JSON 296.4 (accept 2.32 / 6.78 / 4.36 / 4.60) | [gittensor-model-hub/Qwen3.8-27B-NVFP4](https://huggingface.co/gittensor-model-hub/Qwen3.8-27B-NVFP4) | V |
| SGLang 0.5.18 + DSpark (NVFP4) | cross-engine harness: 24 prompts (chat / code / math / JSON), 256 max tokens, T=0, thinking off, sequential, `--ctx 262144` | 161.7 tok/s | same card | V |
| SparkInfer / SGLang / vLLM, no speculation | same harness | 92.9 / 85.8 / 86.8 tok/s | same card | V |
| SparkInfer, 240 held-out prompts | longer outputs, own settings | no spec 88.45; MTP 136.90 (accept 2.758); DSpark NVFP4 v2 180.30 (2.904) | same card | V |
| SparkInfer v0.5.5 concurrency | distinct prompts, 128 output tokens, `--ctx 40960`, 32 requests per level | aggregate 88.9 / 162.1 / 249.6 / 344.3 / 345.1 / 318.2 tok/s at 1 / 2 / 4 / 8 / 16 / 32 | same card | V |
| vLLM 0.28, same harness | as above | aggregate 134.5 / 266.6 / 545.8 / 546.6 at 2 / 4 / 8 / 16 | same card | V |
| SparkInfer long context | 244K context | 90.3 tok/s (vLLM 63.5); cold TTFT at 250K ~121 s | same card | V |
| SGLang 0.5.19 + DFlash2 (NVFP4, fp8 KV) | ISL 8192 / OSL 1024, concurrency 1 | median TPOT 4.92 ms (accept length 4.29); reported as 206.1 tok/s | [SGLang cookbook: Qwen3.8-27B](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B); [runtimewire summary](https://runtimewire.com/article/sglang-qwen3-8-27b-nvfp4-dflash2-cookbook) | V |
| SGLang + EAGLE/MTP (NVFP4) | cookbook, GPU not named explicitly | 152.9 tok/s/user | SGLang cookbook | V |
| llama.cpp, NVFP4 + MTP | tutorial | ~142 tok/s average, peaks ~170 | [DataCamp tutorial](https://www.datacamp.com/de/tutorial/how-to-run-qwen3-8-27b-locally) (search snippet) | F |

## Qwen3.8-27B, Intel Arc Pro B70 (32 GB)

**Headline / aggregate chart (Arc B70 multi-user):** shipped defaults: Arc Pro B70 at 300 W, publish code + public draft vocabulary + public K-smoothing table + DYN / Q4 draft head, util 0.94, `harness/yieldbench.py` agg cells (same shape as the published RTX 5090 numbers), median of 3 reps: 119.7 / 193.4 / 280.2 / 390.1 / 479.0 / 477.5 tok/s at 1 / 2 / 4 / 8 / 16 / 32 users. Arc numbers measured on a desktop host; a slower host CPU lowers small-batch decode.

| Runtime | Setup | Published result | Source | Trust |
|---|---|---|---|---|
| exl3xpu upstream (EXL3 4.0bpw, k=3, fp8 KV) | `bench/sweep.py --corpus real` (Gutenberg passage + essay; HumanEval code), thinking on, T=0.7 | decode C1 91.2 / 66.1, C2 151.4 / 125.3, C4 262.6 / 212.9, C8 357 / 322 tok/s (prose / code); prefill 4K 2,421, 32K 2,124 tok/s | [0xSero/exl3xpu README](https://github.com/0xSero/exl3xpu), [PROGRESS.md](https://github.com/0xSero/exl3xpu/blob/main/docs/PROGRESS.md) | V |
| vLLM XPU, GPTQ Int4 + MTP4 | p512 / g128, T=0 (~100 % acceptance) | 84.6 tok/s; MTP2 52.2 (35.8 at 98K) | [jonathanmann.tech](https://jonathanmann.tech/blog/qwen38-intel-arc-b70), [MikeCaldera/intel-arc-pro-b70-qwen38-vllm](https://github.com/MikeCaldera/intel-arc-pro-b70-qwen38-vllm) | F |
| vLLM XPU, INT4 AutoRound, **2x B70 TP=2** | custom image, greedy | 57 tok/s single stream, 199 at 32 concurrent (stock vLLM 0.21: 16.3 / 76.9) | [CySpiegel/Qwen3.8-27B-Int4-AutoRound](https://huggingface.co/CySpiegel/Qwen3.8-27B-Int4-AutoRound) | F |
| llama.cpp SYCL, Q4_K_M | | 20-25 tok/s (MTP2 28.4), C8 57; prefill 32K 629 tok/s | sources in the exl3xpu row's survey: [PMZFX benchmarks](https://github.com/PMZFX/intel-arc-pro-b70-benchmarks), [Thotheris/exl3xpu-b70-lab](https://github.com/Thotheris/exl3xpu-b70-lab) | F |
| llama.cpp SYCL fork | quant not stated | peaks 37 tok/s generation, 900 prefill | [cantosun99/llama.cpp-sycl](https://github.com/cantosun99/llama.cpp-sycl) | F |

Older models on the B70, for context only: Qwen3.5-27B llama.cpp SYCL Q4_K_M pp512 718 / tg128 20.4 tok/s
([PMZFX](https://github.com/PMZFX/intel-arc-pro-b70-benchmarks)); Qwen3.6-27B on **4x** B70 with Intel LLM Scaler
(FP16, TP4) 13.1 tok/s at 1 user, 95.9 at 8 users ([Puget Systems via Gigazine](https://www.gigazine.net/gsc_news/en/20260619-intel-arc-pro-b70-local-ai)).

## Quality (accuracy vs BF16)

**Headline accuracy tile:** the 5-task mean (MMLU, ARC-C, HellaSwag, GSM8K, HumanEval; same harness for every row) as % of BF16: BF16 93.5 (93.54, = 100), Q4_K_M GGUF 93.2 (99.6 %), NVFP4 93.0 (99.4 %), ours on the RTX 5090 with int4 KV 92.9 (99.3 %), and the 3-bit reference UD-IQ3_XXS 92.7 (99.1 %). Percentages are computed from the unrounded per-task values in the table below; the bars start at 0.

From [acyildirimer/Qwen3.8-27B-NVFP4](https://huggingface.co/acyildirimer/Qwen3.8-27B-NVFP4) (vLLM 0.27.1 for the
NVFP4 row; BF16 and GGUF rows taken by the author from a public dataset, so the page calls them indicative; thinking
off, temperature 0; MMLU 5-shot 50 % sample, ARC-C 25-shot, HellaSwag 10-shot 50 % sample, GSM8K 5-shot CoT full,
HumanEval 0-shot pass@1):

| Checkpoint | MMLU | ARC-C | HellaSwag | GSM8K | HumanEval | 4-task mean |
|---|---|---|---|---|---|---|
| BF16 | 85.3 | 96.8 | 94.3 | 97.4 | 93.9 | 93.5 |
| Q8_0 GGUF | 85.2 | 96.8 | 94.4 | 97.4 | 94.5 | 93.5 |
| Q6_K GGUF | 85.3 | 96.7 | 94.3 | 97.5 | 94.5 | 93.5 |
| UD-Q4_K_XL GGUF | 85.1 | 96.6 | 94.4 | 97.3 | 93.9 | 93.4 |
| Q4_K_M GGUF | 85.0 | 96.8 | 94.3 | 97.1 | 92.7 | 93.3 |
| acyildirimer NVFP4 (mixed FP8) | 85.1 | 96.8 | 94.1 | 96.9 | 92.1 | 93.2 |
| unsloth Dynamic V3 NVFP4 | 84.3 | 96.9 | 94.3 | 97.1 | 89.6 | 93.2 |

KL divergence vs BF16, Qwen3.6-27B (previous generation), 100 agentic tool-use conversations, top-200 logprobs
([erli.xyz](https://erli.xyz/posts/qwen36-27b-quant-comparison/),
[r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/comments/1vksqju/i_compared_gguf_quants_of_qwen36_27b_to_nvfp4_awq/)):
Q4_K_L 0.2218, UD_Q4_K_XL 0.2273, AWQ 0.2776, NVIDIA mixed NVFP4 0.2807. Different metric from ours (full-distribution
KL on our set), so it is cited for the ordering only.

EXL3 4.0bpw weights vs BF16: a third-party KLD of 0.016 (turboderp / jedi95, different method). Our own same-set measurement (24 production-shaped chats, teacher-forced KL vs BF16): EXL3 4.0bpw + int8 KV 0.0060, + int4 KV 0.0080, Q4_K_M 0.0140, so both of our configurations are closer to BF16 than Q4_K_M.

## Fidelity (KL vs BF16) of NVFP4 builds

[adrienbrault/qwen3.8-27b-rtx5090 docs/FIDELITY.md](https://github.com/adrienbrault/qwen3.8-27b-rtx5090/blob/main/docs/FIDELITY.md)
(teacher-forced against BF16 dumps; dense ruler: 693 documents of text and code, 555,549 positions; agentic ruler: 72
tool-calling turns, 57,972 positions; truncated KL over the BF16 top-k):

| Configuration | Dense top-1 | Dense ppl Δ | Dense KL | Agentic top-1 |
|---|---|---|---|---|
| RedHatAI NVFP4, FP8 KV | 93.07 % | +0.38 % | 0.0129 | 95.95 % |
| RedHatAI NVFP4, NVFP4 KV | 92.77 % | +0.74 % | 0.0141 | 95.58 % |
| NVIDIA NVFP4 | 90.67 % | +1.83 % | 0.0226 | – |
| gittensor NVFP4, FP8 KV | 88.56 % | +4.46 % | 0.0349 | 92.60 % |

## Card prices (headline graphic)

| Card | List | Street (retrieved 2026-11-11) | Source |
|---|---|---|---|
| Intel Arc Pro B70 32 GB | $949 (unchanged per Intel) | store-sold floor ~$1,280 (2026-10-24), Newegg up to $2,399 (late Oct) | [Notebookcheck](https://www.notebookcheck.net/Intel-s-Arc-Pro-B70-has-climbed-faster-than-AMD-s-32GB-Radeon-AI-Pro-R9700-and-memory-costs-alone-can-t-explain-why.1435374.0.html), [VideoCardz](https://videocardz.com/newz/intel-arc-pro-b70-prices-more-than-double-in-us-now-listed-for-up-to-2400), [Tom's Hardware](https://www.tomshardware.com/pc-components/gpus/intels-arc-pro-b70-workstation-gpu-is-now-up-to-48-percent-more-expensive-than-it-was-just-a-month-ago-32gb-battlemage-workstation-card-climbs-toward-usd2-000) |
| NVIDIA RTX 5090 32 GB | $1,999 | best US price $6,919 (Tom's Hardware tracker, 2026-11-03); trackers $5,500-9,400 | [tech-insider summary of the Tom's Hardware tracker](https://tech-insider.org/gpu-price-tracking-2026-rtx-5090-6919-tomshardware-lowest-prices/), [videocardprices.com](https://videocardprices.com/card/nvidia-rtx-5090/) |

Retail check by the author, 2026-11-11: Arc Pro B70 $1,700; RTX 5090 $7,000-11,900. The headline says "~20 % of the
RTX 5090's price": the mean of $1,700 / $7,000 = 0.24 and $1,700 / $11,900 = 0.14. Prices move daily; the trackers above are the
citable references. List prices: $949 vs $1,999.
The "matches the RTX 5090" tile is limited to 1-4 concurrent users (SparkInfer's concurrency test shape, best published
RTX 5090 result per level: SparkInfer v0.5.5 at 1, 2; vLLM 0.28 at 4); at 8+ users vLLM on the RTX 5090 is faster.
