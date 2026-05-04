# Serving + dashboard

The `mlx-fun serve` command runs an OpenAI/Anthropic-compatible API server
with optional MoE expert-counting hooks, runtime steering, and a Gradio
dashboard (`mlx-fun ui`) for monitoring and controls.

## Online expert counting

Instead of offline calibration, you can collect expert statistics from real
production traffic:

```bash
mlx-fun serve \
    --model mlx-community/MiniMax-M1-40k-4bit \
    --port 8080 \
    --enable-counting \
    --mode lightweight \
    --auto-save ./online_saliency.npz
```

Hooks are **off by default** — the server runs as plain inference unless
`--enable-counting` is set. `--auto-save` writes the accumulator on shutdown.

### Lightweight vs full mode

| Mode | Metrics | Overhead | When |
|---|---|---|---|
| `lightweight` (default) | `freq`, `weighted_freq` | Minimal | Long-running production serving |
| `full` | `freq`, `weighted_freq`, `reap`, `ean` | Per-expert `mx.linalg.norm()` adds ~5–10% to decode | Short collection runs needing all metrics |

In lightweight mode, `reap` and `ean` scores are zero — use `freq` or
`weighted_freq` as the `--metric` when pruning with lightweight-collected
data.

### Note on chat traffic bias

When serving chat conversations where each request includes the full dialog
history, earlier messages are re-processed more often than later ones. This
inflates expert counts for tokens in early messages. Mitigations:

- Prompt cache partially helps — mlx-lm's LRU prompt cache skips re-computation
  for cached prefixes.
- `reap` and `ean` (full mode) are **averages**, so they're count-invariant.
- Use `/v1/reap/reset` between collection windows for stats from specific
  traffic periods.

## Management endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/v1/reap/stats` | Full expert frequency/score data as JSON |
| `GET` | `/v1/reap/info` | Model info: layer/expert counts, request/token totals |
| `POST` | `/v1/reap/save` | Save accumulator. Body: `{"path": "out.npz"}` |
| `POST` | `/v1/reap/reset` | Reset all counters |
| `GET` | `/v1/reap/steer` | Get current steering config |
| `POST` | `/v1/reap/steer` | Update steering config |
| `DELETE` | `/v1/reap/steer` | Remove all steering |
| `GET` | `/v1/reap/gpu_limit` | Read MLX wired-memory limit |
| `POST` | `/v1/reap/gpu_limit` | Set wired-memory limit live |

`/v1/reap/stats` response shape:

```json
{
  "freq":              [[...], [...]],
  "weighted_freq_sum": [[...], [...]],
  "reap_sum":          [[...], [...]],
  "ean_sum":           [[...], [...]],
  "reap_count":        [[...], [...]],
  "num_layers": 62,
  "num_experts": 256,
  "request_count": 150,
  "token_count": 75000,
  "total_samples": 12687240.0,
  "computed_scores": {
    "reap":          [[...], [...]],
    "ean":           [[...], [...]],
    "freq":          [[...], [...]],
    "weighted_freq": [[...], [...]]
  }
}
```

`computed_scores` is ready for `mlx-fun stats-diff` / `stats-merge` /
`prune --metric ...`.

```bash
# Pull stats and feed directly into prune
curl -X POST http://localhost:8080/v1/reap/save -d '{"path": "online.npz"}'
mlx-fun prune --model ./model --saliency online.npz \
    --output ./pruned --n-prune 16 --metric freq
```

## Runtime steering via REST

The steering endpoint allows hot-swapping expert (de)activation without
restarting the server. Updates are applied atomically — no hook
reinstallation.

```bash
# Enable steering from a safety report
curl -X POST http://localhost:8080/v1/reap/steer \
    -H "Content-Type: application/json" \
    -d '{"safety_map": "/path/to/safety_report.json", "mode": "safe"}'

# Enable steering from a domain report
curl -X POST http://localhost:8080/v1/reap/steer \
    -d '{"domain_map": "/path/to/domain_report.json", "mode": "boost"}'

# Specify experts directly
curl -X POST http://localhost:8080/v1/reap/steer \
    -d '{"deactivate": {"0": [3, 17], "5": [42]}, "activate": {"12": [8]}, "mask_value": -1e9}'

# Inspect / remove
curl http://localhost:8080/v1/reap/steer
curl -X DELETE http://localhost:8080/v1/reap/steer
```

## Default sampling parameters

The server can inject sampling defaults for any request that omits them:

```bash
mlx-fun serve \
    --default-top-k 100 \
    --default-repetition-penalty 1.1 \
    --default-temperature 0.7
```

The defaults are merged into the request body before hand-off to mlx-lm —
explicit values in the request always win.

## KV cache compression

Stack with weight quantization for maximum compression. See
[`kv-cache.md`](./kv-cache.md) for TurboQuant (PolarQuant) and RotorQuant
(Clifford rotors) details.

```bash
mlx-fun serve --model ./model --kv-compress turbo --kv-compress-bits 4
mlx-fun serve --model ./model --kv-compress rotor --kv-compress-bits 3 --max-kv-size 4096
```

## Web dashboard (Gradio)

```bash
# Terminal 1
mlx-fun serve --model ./model --port 8080

# Terminal 2
mlx-fun ui --server-url http://127.0.0.1:8080
```

Tabs:

| Tab | Features |
|---|---|
| **Chat** | Streaming chat, configurable system prompt / temperature / max tokens |
| **Dashboard** | Expert activation heatmaps + per-layer bar charts |
| **Steering** | Apply steering from safety report / domain report / custom JSON |
| **Merge Mode Comparison** | Load 1–4 `.npz` files, filter experts by rank, export `filtered_experts.json` for CLI |
| **Diff Analysis** | Compare two saliency files side-by-side with diverging-color heatmap |
| **Controls** | Server info, save accumulator, reset counters, raw stats JSON |

### Merge Mode Comparison workflow

1. Load 1–4 saliency `.npz` files from disk.
2. Pick a metric (REAP, EAN, Frequency, Weighted Frequency).
3. Apply rank-sum filters or N-to-prune threshold.
4. Switch between Per-Layer / Model-Wide selection.
5. Export `filtered_experts.json` for use with `prune --expert-list` or
   `merge --expert-list`.

```bash
mlx-fun prune --model ./model --expert-list filtered_experts.json --output ./pruned
mlx-fun merge --model ./model --expert-list filtered_experts.json --dataset calib.jsonl --output ./merged
```

## Pre-loaded model + auto-unload

If the server is configured with `--model ...`, the model loads at startup;
otherwise on the first request that names it. After `--idle-timeout` seconds
of inactivity, the model is unloaded to free memory. Set `0` to disable
auto-unload (recommended for very large models like Kimi-K2.6 where load is
slow).
