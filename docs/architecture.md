# Architecture

## Layout

```
┌───────────────────────────────────────────────────────────────────────────────────────────────┐
│                                         CLI (cli.py)                                           │
│  collect / prune / merge / smoke-test / serve / ui / safety-scan / steer / abliterate          │
│  domain-scan / amplify / convert-nvfp4 / stats-{diff,merge,purge}                              │
└──┬────────────┬───────────┬──────────┬─────────────┬────────────┬──────────────┬──────────────┘
   │            │           │          │             │            │              │
┌──▼──────┐ ┌──▼─────┐ ┌───▼────┐ ┌───▼──────────┐ │      ┌─────▼──────────┐ ┌─▼────────────┐
│Observer │ │ Pruner │ │ Merger │ │ Server       │ │      │ Abliterate     │ │ Domain       │
│ hooks   │ │ engine │ │ (REAM) │ │ (OpenAI +    │ │      │ residual hooks │ │ scan + gate  │
│(offline)│ └──┬─────┘ └───┬────┘ │ REAP + Steer)│ │      │ orthogonalize  │ │ amplification│
└──┬──────┘    │           │      └───┬──────────┘ │      └──────┬─────────┘ └──────┬───────┘
   │           │           │          │      ┌─────▼────────┐    │                  │
┌──▼──────┐ ┌──▼──────┐ ┌─▼────────┐ │  ┌───▼────────────┐ │    │                  │
│Saliency │ │ Tensor  │ │REAM hooks│ │  │Safety analysis │ │    │                  │
│ accum.  │ │ slicing │ │Similarity│ │  │DifferentialAcc │◄├────┼──────────────────┘
└─────────┘ └─────────┘ │Alignment │ │  │SafetyReport    │ │    │
                        └─┬────────┘ │  └───┬────────────┘ │    │
                          │    ┌──────▼──────▼─┐     ┌─────▼────▼───────┐
                          │    │OnlineAccum.   │     │ Steering hooks   │
                          │    │(thread-safe)  │     │ (gate bias inj.) │
                          │    └──────┬────────┘     └─────┬────────────┘
              ┌───────────▼───────────▼────────────────────▼──┐
              │                Adapter Layer                    │
              │ MiniMax / GLM4 / GLM5 / Qwen / DSv3 / Kimi-K2.6 │
              └──────────────────┬─────────────────────────────┘
                                 │
              ┌──────────────────▼─────────────────────────────┐
              │          Frontend (frontend.py)                  │
              │  Gradio web UI: chat, heatmaps, steering, mgmt  │
              │  Connects to server via HTTP REST API            │
              └────────────────────────────────────────────────┘
```

## Component notes

- **Adapters** abstract model-specific MoE access (layer paths, gate
  structure, config keys). New architectures plug in via `BaseAdapter` ABC.
  See `src/mlx_fun/adapters/base.py`.
- **Observer** installs hooks via **`__class__` swapping** — Python resolves
  `__call__` on the type, not the instance, so `types.MethodType` doesn't
  work for special methods. The observer creates a dynamic subclass with the
  hooked `__call__` and swaps `block.__class__`.
- **Saliency** accumulates statistics in **numpy float64** using vectorized
  `np.add.at()` scatter-adds. After capturing hook data, `mx.eval()` is
  called immediately followed by `_to_numpy()` (casts bf16 to float32 before
  `np.array(..., copy=False)`) to materialize lazy MLX arrays.
- **Pruner** uses `mx.take(tensor, keep_indices, axis=0)` to slice
  expert-axis tensors (weights, scales, biases, gates). Both `nn.Linear` and
  `nn.QuantizedLinear` gates are handled. Supports bottom and strided
  strategies; accepts `protected_experts` / `targeted_experts` from safety
  reports.
- **Merger** (REAM) computes gated expert similarity, groups experts around
  centroids, aligns intermediate neurons via permutation matching (greedy or
  scipy Hungarian), and produces saliency-weighted averages. Processes
  layers sequentially with model re-forwarding so merged weights feed into
  the next layer.
- **Safety** (`safety.py`) tracks gate logit statistics separately for
  harmful/benign datasets in a `DifferentialAccumulator`, computes per-expert
  differential scores, and classifies experts into HCDG/HRCG groups via
  `SafetyReport`.
- **Steering** (`steering.py`) injects pre-computed bias arrays into gate
  logits before top-k. Per-model-type hooks handle different gating
  mechanisms (sigmoid for MiniMax/GLM4, softmax for Qwen3). Biases are
  hot-swappable at runtime.
- **Abliterate** (`abliterate.py`) hooks decoder layers (one level above MoE
  blocks) to capture residual stream, computes refusal directions, and
  orthogonalizes weight matrices. Supports per-expert targeting via
  `_orthogonalize_expert_proj` (uses `mx.concatenate` on slices because MLX
  has no `.at[].set()`).
- **Domain** (`domain.py`) reuses `DifferentialAccumulator` from `safety.py`
  for domain-vs-general analysis, classifies experts via `DomainReport`,
  computes amplification biases, and permanently modifies gate parameters
  (nn.Linear bias or correction_bias) for hook-free inference.
- **KV Compress** (`kv_compress.py`) implements TurboQuant PolarQuant
  rotation-based KV compression. `TurboQuantKVCache` stores rotated-quantized
  K/V in MLX's native packed format. For supported models, patches the
  module's `scaled_dot_product_attention` to rotate queries → enables
  `mx.quantized_matmul` for hardware-accelerated attention. Falls back to
  dequantize+inverse-rotate for unsupported architectures.
- **Server** (`server.py`) composes on mlx-lm's `APIHandler` and
  `ResponseGenerator`. Compound counting+steering hooks accumulate stats AND
  apply steering in a single `__call__` (two `__class__` swaps on the same
  block would conflict). REST endpoints for runtime steering control.
- **Tool parsers** — Kimi-K2.6 ships a permissive replacement parser at
  `src/mlx_fun/kimi_k26_tool_parser.py`, installed automatically for any
  `model_type == "kimi_k25"`. See [`tool-parsers.md`](./tool-parsers.md).
- **Frontend** (`frontend.py`) is a Gradio app that connects to the running
  server via HTTP. Provides chat (streaming via SSE), expert activation
  heatmaps (matplotlib), steering controls, server management. Launched via
  `mlx-fun ui`.

## Python API

The components are usable directly:

```python
import mlx.core as mx
from mlx_lm import load as mlx_load

from mlx_fun.adapters import get_adapter
from mlx_fun.observer import install_hooks, collect_captures, remove_hooks
from mlx_fun.saliency import SaliencyAccumulator
from mlx_fun.pruner import select_experts_to_keep, select_experts_to_keep_strided, prune_model
from mlx_fun.save import save_pruned_model

# Load
model, tokenizer = mlx_load("mlx-community/MiniMax-M1-40k-4bit")
config = {...}  # from config.json
adapter = get_adapter(model, config)

# Calibrate
moe_indices = adapter.moe_layer_indices()
moe_blocks  = [adapter.get_moe_block(i) for i in moe_indices]
install_hooks(moe_blocks, config["model_type"])

acc = SaliencyAccumulator(len(moe_indices), adapter.num_routed_experts())
for tokens in calibration_data:
    model(tokens.reshape(1, -1))
    for idx, captures in enumerate(collect_captures(moe_blocks)):
        for inds, scores, norms in captures:
            acc.update(idx,
                inds.reshape(-1, inds.shape[-1]),
                scores.reshape(-1, scores.shape[-1]),
                norms.reshape(-1, norms.shape[-1]),
            )
remove_hooks(moe_blocks)

# Prune
scores = acc.compute_scores("reap")
keep_map = select_experts_to_keep(scores, n_prune=16)            # bottom
# keep_map = select_experts_to_keep_strided(scores, n_prune=16)  # strided
model_keep_map = {moe_indices[i]: k for i, k in keep_map.items()}
new_config = prune_model(adapter, model_keep_map)

# Save
save_pruned_model(model, tokenizer, new_config, "./pruned",
                  model_keep_map, adapter.num_routed_experts(), "reap")
```

## Output formats

The output directory contains:

```
output_model/
├── model.safetensors           # or sharded model-NNNNN-of-MMMMM.safetensors
├── model.safetensors.index.json
├── config.json                  # quantization metadata + reduced expert count
├── tokenizer.json               # unchanged
├── tokenizer_config.json
├── special_tokens_map.json
└── <method>_metadata.json       # provenance
```

All output models load via `mlx_lm.load()` — no special loaders needed.

### `reap_metadata.json` (pruned)

```json
{
  "original_num_experts": 64,
  "pruned_num_experts": 48,
  "metric": "reap",
  "keep_map": {
    "0": [0, 1, 3, 5, 7, ...],
    "1": [0, 2, 4, 6, 8, ...]
  }
}
```

### `ream_metadata.json` (merged)

```json
{
  "method": "ream",
  "original_num_experts": 64,
  "merged_num_experts": 48,
  "metric": "reap",
  "centroid_map": { "0": [2, 5, 11, ...], ... },
  "group_map": {
    "0": {"2": [2, 7, 14], "5": [5, 3, 9]},
    ...
  }
}
```

### `abliteration_metadata.json`

```json
{
  "method": "abliteration",
  "target": "all",
  "abliterated_layers": [10, 11, 12, 13, 14, 15],
  "direction_norms": {"10": 0.0234, "11": 0.0312, "12": 0.0287}
}
```

### `amplification_metadata.json`

```json
{
  "method": "amplification",
  "domain_name": "solidity",
  "scale": 1.0,
  "threshold": 0.0,
  "amplified_layers": [0, 1, 2, 5, 8],
  "per_layer_bias": {
    "0": [0.0, 0.0, 0.8, 0.0, 0.6, ...],
    "1": [0.3, 0.0, 0.0, 0.7, ...]
  }
}
```

### `conversion_metadata.json` (NVFP4 / streaming converters)

```json
{
  "source": "/path/to/source",
  "converter": "local/scripts/convert_kimi_k2_int4_to_q3.py",
  "stats": {
    "quantized_linears": 69120,
    "passthrough_tensors": 855,
    "dropped_vision_tensors": 335,
    "output_shards": 145,
    "total_bytes": 467728777372
  },
  "quant": {"bits": 3, "group_size": 64, "mode": "affine"}
}
```

## Testing

```bash
uv pip install -e ".[dev]"
pytest tests/ -v
```

Coverage spans adapter detection, saliency math, observer hooks, pruner
slicing, REAM hooks/merger, dataset loading, server (OnlineAccumulator
thread safety, lightweight/full hooks, management endpoints), safety
analysis, steering, abliteration, domain identification, frontend, KV cache
compression, and CLI wiring.
