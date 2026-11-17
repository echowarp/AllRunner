# Public K-smoothing table

`public_r.pt` is the launchers' default `KSMOOTH_TABLE` (`EXL3_KSMOOTH`) for both GPUs: the robust mean key per layer,
KV head and channel of Qwen3.8-27B (`{layer: float32 [H, D]}`, 17 attention layers), which int4 / int8 KV caches
subtract before quantizing. sha256 `9c7765bc8707254624f805993744264541e4c6b5ff09bfef5901860f206e646d`.

Built by echowarp/exl3xpu `scripts/public_ksmooth.py`: an eager engine with `EXL3_KSMOOTH_CALIB` (robust) fed about
380k prompt tokens of public text only, with the same groups and shares as the public draft vocabularies (Python
standard library, PyPI package descriptions, json-mode-eval incl. rendered tool calls, MT-Bench, HumanEval, three
Project Gutenberg books). `public_r.pt.inputs.json` lists every input with its sha256 and the tokens per group. No
production traffic went into it.

Measured on an Arc Pro B70 (int4 KV, 126,099 held-out GSM8K tokens teacher-forced, against a table built the same way
on production agent traffic):

| Table | NLL vs production table | top-5 KL |
|---|---|---|
| production (traffic) | reference | |
| `public_r.pt` | +0.037 % | 4.2e-3 |
| none | +0.43 % | 4.5e-3 |

Speed: with a table the fused int4 KV write is on. Table + fused write add about +6-7 % on a fast host (Arc B70:
1-stream aggregate 111.8 -> 119.7 tok/s, per-stream decode 139.2 -> 148.5) and up to +35 % per-stream decode on a
slow, host-bound CPU (142 -> 192 tok/s on HumanEval prompts); the public and the production table run at the same speed.

Engines that share the RAM tier must use the same table (its hash is part of the tier's compatibility tag).
