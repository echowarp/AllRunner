# AllRunner 2026-11-16: first public release

## What is in it

- `launchers/`: one engine per GPU (Arc Pro B70: `arc-b70.sh`, RTX 5090: `rtx5090.sh`), the shared KV broker
  (`kvbroker.sh`), the router (`router.sh`) and a post-start warm-up with a router drain gate (`llmt-warmup.py`).
  Defaults are the production settings; the remaining differences are listed in `launchers/README.md`
  ("Production parity").
- `data/draft-vocab/`, `data/ksmooth/`: public draft vocabularies (512 and 128 blocks) and a public K-smoothing table,
  built from public text only, the launchers' defaults. K-smoothing table (Arc, int4 KV, held-out GSM8K): quality
  NLL +0.037 % vs the production table, +0.43 % without a table; speed: the table plus the fused int4 KV write it enables add about +6-7 % on a fast host (Arc B70: 1-stream aggregate 111.8 -> 119.7 tok/s, per-stream decode 139.2 -> 148.5) and up to +35 % per-stream decode on a slow, host-bound CPU (142 -> 192 tok/s on HumanEval prompts).
- `router/`: prefix-affinity router with RAM-tier namespace injection, LoRA adapters by model suffix, prefix hints
  from the broker (off by default), session / priority policy (off by default).
- `docker/`: Dockerfiles for both images. `benchmarks/`: results, harness, charts.

## How it was validated

- Router: all 10 CPU test scripts pass. Launchers: `bash -n`, plus a dry run with a stub `docker` checking the
  assembled command line (draft vocabularies, keeper, LoRA, packed KV).
- The production engines ran these flags (from site-specific scripts) when the benchmark numbers were measured, e.g.
  RTX 5090 aggregate decode 300.4 / 492.8 / 684.8 / 926.7 / 1,285.2 / 1,359.9 tok/s and single-stream 241.0 (chat),
  470.1 (code), 322.0 (math), 442.3 (JSON) tok/s (2026-11-16, KV 482,846 tokens). Those runs used production's
  traffic-ranked draft vocabularies, not the shipped public ones.

## Known limitations

- The Docker images have not been rebuilt from the pinned release commits yet; no engine has been started from these
  launchers as written.
- Benchmark numbers with the shipped public draft vocabularies are not measured yet.
- RTX 5090: two starts of the same engine are not bit-identical (4/6 greedy prompts identical, max |dlp| 0.091);
  compare runs against that noise floor.
- vLLM logs `MAMBA-ACC` warnings (a request's accepted-token count reset to 1): a corrected race, harmless.
- LoRA on the RTX 5090 is off by default (vision-tower lending only); LoRA is on on the Arc.
- Not reproducible from public files: the RTX 5090's 4-bit draft head (about 3 % decode); the 5090 KV staging ring is
  not in the published vLLM CUDA branch. The public K-smoothing table was measured on the Arc only.
