# Public draft vocabularies

Pruned MTP draft-head vocabularies for Qwen3.8-27B (128-token blocks of the 248,077-token lm_head), the launchers'
default `DRAFT_VOCAB` / `DRAFT_VOCAB_FAR`:

| File | Blocks | Tokens | Coverage of the input text | Used as |
|---|---|---|---|---|
| `public-512.json` | 512 | 65,536 | 98.5 % | near head (both GPUs), far head (RTX 5090) |
| `public-128.json` | 128 | 16,384 | 89.2 % | far head, draft positions >= 2 (Arc B70) |

Built by echowarp/exl3xpu `scripts/public_draft_vocab.py --blocks 512 128` from public text only: Python standard
library source, PyPI package descriptions, json-mode-eval tool calls rendered through the model's chat template,
MT-Bench, HumanEval and three Project Gutenberg books (fixed group shares). Each file lists every input with its
sha256 (`inputs`), so it can be rebuilt and audited. No production traffic went into them. Drafts only: the output
distribution is unchanged; a vocabulary ranked on your own traffic (exl3xpu `scripts/draft_vocab.py`) usually accepts
more draft tokens.
