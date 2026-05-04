# CLI reference

Eleven commands. Pipeline shape:

`collect → prune | merge → smoke-test`, plus `serve` (online collection +
inference + steering API), `ui` (Gradio dashboard), `safety-scan / steer /
abliterate` (safety analysis), `domain-scan / amplify` (domain
specialization), `convert-nvfp4` (NVIDIA modelopt → MLX-native NVFP4), and
`stats-{diff,merge,purge}` (saliency operations).

---

## `collect` — offline saliency calibration

Run calibration to measure expert importance:

```bash
mlx-fun collect \
    --model mlx-community/MiniMax-M1-40k-4bit \
    --dataset ./data/solidity_calibration.jsonl \
    --output ./saliency.npz \
    --max-samples 128 --seed 42
```

| Flag | Default | Description |
|---|---|---|
| `--model` | *(required)* | Model path or HuggingFace repo ID |
| `--dataset` | *(required)* | Path to JSONL file or directory |
| `--output` | *(required)* | Output `.npz` file for saliency stats |
| `--max-samples` | 128 | Maximum calibration samples (random subset) |
| `--max-tokens` | 2048 | Maximum tokens per sample |
| `--text-key` | `content` | JSON key for text in JSONL files |
| `--seed` | *(none)* | Random seed for reproducible sample selection |

## `prune` — remove low-saliency experts

```bash
# Mode 1: from saliency file
mlx-fun prune --model ./model --saliency ./saliency.npz \
    --output ./pruned --n-prune 16 --metric reap --strategy bottom

# Mode 2: from frontend export
mlx-fun prune --model ./model --expert-list filtered_experts.json --output ./pruned
```

| Flag | Default | Description |
|---|---|---|
| `--model` | *(required)* | Same model used for collection |
| `--saliency` | *(required if no --expert-list)* | Path to `.npz` from collect step |
| `--expert-list` | *(none)* | Path to `.json` or `.csv` from frontend export |
| `--output` | *(required)* | Output directory for pruned model |
| `--n-prune` | *(required if no --expert-list)* | Number of experts to remove |
| `--metric` | `reap` | `reap`, `ean`, `freq`, `weighted_freq` |
| `--strategy` | `bottom` | `bottom` or `strided` |
| `--model-wide` | `false` | Select N experts globally across all layers |
| `--min-experts-per-layer` | `1` | Floor for `--model-wide` |
| `--ignore-experts` | *(none)* | Protected indices, format `1,2,250..255`. With `--model-wide` only. |
| `--safety-map` | *(none)* | Path to `safety_report.json` from `safety-scan` |
| `--safety-mode` | *(none)* | `protect` or `target` |
| `--domain-map` | *(none)* | Path to `domain_report.json` from `domain-scan` |
| `--domain-mode` | *(none)* | `protect` |
| `--stream` | `false` | Stream-prune by slicing safetensors shards directly. Bypasses `mlx_lm.load`. Per-layer bottom strategy with saliency input only. |

The pruned model is saved as `safetensors + config.json + tokenizer +
reap_metadata.json` and loads with `mlx_lm.load()`.

## `merge` — REAM expert merging

```bash
mlx-fun merge --model ./model --saliency ./saliency.npz \
    --dataset ./data/calib.jsonl --output ./merged \
    --n-prune 16 --max-group-size 16 --max-samples 64
```

Supports the same `--expert-list`, `--model-wide`, and constraint flags as
`prune`. Adds:

| Flag | Default | Description |
|---|---|---|
| `--dataset` | *(required)* | Calibration data for similarity / alignment |
| `--similarity-mode` | `gated` | `gated` or `average` |
| `--alignment` | `greedy` | `greedy`, `hungarian` (needs scipy), or `none` |
| `--max-group-size` | `16` | Max experts per merge group |
| `--max-samples` | `64` | Calibration samples for similarity |
| `--max-similarity-tokens` | `512` | Tokens for similarity (subsampled) |
| `--max-alignment-tokens` | `256` | Tokens for permutation alignment |

Merged model output adds `ream_metadata.json`.

## `smoke-test` — verify generation

```bash
mlx-fun smoke-test --model ./pruned_model --prompt "pragma solidity ^0.8.0;"
```

| Flag | Default | Description |
|---|---|---|
| `--model` | *(required)* | Model directory |
| `--prompt` | `pragma solidity ^0.8.0;` | Generation prompt |
| `--max-tokens` | 100 | Max tokens to generate |
| `--kv-compress` | *(none)* | `turbo` or `rotor` |
| `--kv-compress-bits` | 4 (turbo), 3 (rotor) | Bits per channel |

## `serve` — OpenAI-compatible API + online expert counting

```bash
mlx-fun serve --model mlx-community/MiniMax-M1-40k-4bit --port 8080 \
    --enable-counting --auto-save ./online_saliency.npz
```

See [`serving.md`](./serving.md) for full endpoint and steering details.

| Flag | Default | Description |
|---|---|---|
| `--model` | *(none)* | Pre-load on startup. If unset, loads on first request. |
| `--host` | `0.0.0.0` | Bind address |
| `--port` | `8080` | Server port |
| `--mode` | `lightweight` | `lightweight` (freq/weighted_freq) or `full` (reap/ean) |
| `--enable-counting` | `false` | Install MoE counting hooks |
| `--prompt-cache-size` | 10 | LRU prompt cache entries |
| `--auto-save` | *(none)* | Save accumulator on shutdown |
| `--max-tokens` | 512 | Default generation cap |
| `--max-kv-size` | *(none)* | Sliding window for KV cache |
| `--idle-timeout` | 1800 | Auto-unload after N seconds; 0 disables |
| `--chat-template` | *(none)* | Override chat template |
| `--trust-remote-code` | `false` | For Kimi-K2 / MiniMax-M2 / etc. with custom tokenizer code |
| `--safety-map` | *(none)* | Path to `safety_report.json` for steering |
| `--steering-mode` | *(none)* | `safe` or `unsafe` |
| `--domain-map` | *(none)* | Path to `domain_report.json` |
| `--domain-steering-mode` | *(none)* | `boost` or `suppress` |
| `--kv-compress` | *(none)* | `turbo` or `rotor` |
| `--kv-compress-bits` | 4 (turbo), 3 (rotor) | Bits per channel |
| `--draft-model` | *(none)* | Draft model for speculative decoding |
| `--num-draft-tokens` | 3 | Drafted tokens per spec step |
| `--default-temperature / --default-top-k / --default-top-p / --default-min-p / --default-repetition-penalty / --default-repetition-context-size` | *(none)* | Server-wide sampling defaults injected when request omits them |
| `--log-level` | `info` | `debug`, `info`, `warning`, `error` |

## `ui` — Gradio dashboard

```bash
mlx-fun ui --server-url http://127.0.0.1:8080
```

| Flag | Default | Description |
|---|---|---|
| `--server-url` | `http://127.0.0.1:8080` | URL of running server |
| `--host` | `127.0.0.1` | Frontend bind address |
| `--port` | `7860` | Frontend port |
| `--share` | *(off)* | Public Gradio share link |

## `safety-scan` — identify safety-critical experts

```bash
mlx-fun safety-scan --model ./model \
    --harmful-dataset ./data/harmful.jsonl \
    --benign-dataset ./data/benign.jsonl \
    --output safety_report.json --max-samples 128
```

Classifies experts into HCDG (harmful-content detection) and HRCG
(harmful-response-control) groups via differential routing analysis. See
[`safety-and-domain.md`](./safety-and-domain.md) for the full method.

## `steer` — inference with expert steering

```bash
mlx-fun steer --model ./model --safety-map safety_report.json \
    --mode safe --prompt "..." --max-tokens 100
```

Adds bias to gate logits before top-k. `safe` boosts HRCG experts; `unsafe`
masks all safety-critical.

## `abliterate` — refusal direction removal

```bash
mlx-fun abliterate --model ./model \
    --harmful-dataset ./data/harmful.jsonl --benign-dataset ./data/benign.jsonl \
    --output ./abliterated_model --layers auto --target all
```

## `domain-scan` — identify domain-specialized experts

Same shape as `safety-scan`, but classifies experts into domain-specialized vs
general groups.

## `amplify` — permanent domain-expert gate boost

```bash
mlx-fun amplify --model ./model --domain-map domain_report.json \
    --output ./amplified_model --scale 1.0 --threshold 0.0
```

Permanently modifies gate weights; loaded model needs no inference-time hooks.

## `convert-nvfp4` — NVIDIA modelopt → MLX-native NVFP4

```bash
mlx-fun convert-nvfp4 --hf-path nvidia/.../NVFP4 --mlx-path ./out
```

See [`convert-nvfp4.md`](./convert-nvfp4.md).

## `stats-diff` — compare two saliency files

```bash
mlx-fun stats-diff --file1 a.npz --file2 b.npz --metric freq --output diff.json
```

## `stats-merge` — rank-based aggregation

```bash
mlx-fun stats-merge --files a.npz --files b.npz --output merged.npz --metric reap
```

Computes per-layer rankings and sums them across files. Each file
contributes equally regardless of sample count.

## `stats-purge` — filter low-activation experts

```bash
mlx-fun stats-purge --input data.npz --output purged.npz --min-freq 100 --min-count 10
```

Zeros out (doesn't remove) entries that don't meet thresholds. At least one
of `--min-freq`, `--min-count`, `--max-norm` is required.
