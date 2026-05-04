# KV cache compression

mlx_fun ships two methods for compressing the KV cache at inference time —
both reduce KV memory 4–8× while keeping near-zero quality loss, enabling
longer contexts and larger models on Apple Silicon. They stack cleanly with
weight quantization (any of the modes in [`quantization.md`](./quantization.md)).

| Method | Flag | Rotation | Quantization |
|---|---|---|---|
| **TurboQuant** (PolarQuant) | `--kv-compress turbo` | Dense d×d orthogonal matrix | MLX native `mx.quantize` |
| **RotorQuant** | `--kv-compress rotor` | Cl(3,0) Clifford rotors (4 params per 3D group) | Lloyd-Max optimal codebook |

Both work with `--max-kv-size` to add a sliding window cap on top of
compression — when the cache exceeds the limit the oldest tokens are dropped
(while preserving the first 4 tokens for BOS/system prompt).

## TurboQuant (PolarQuant)

From [TurboQuant (Google Research, ICLR 2026)](https://arxiv.org/abs/2504.19874).

```bash
# Smoke test
mlx-fun smoke-test --model ./pruned_model --prompt "..." \
    --kv-compress --kv-compress-bits 4

# Serve at 3-bit
mlx-fun serve --model ./model --kv-compress --kv-compress-bits 3

# Combine with sliding window
mlx-fun serve --model ./model --kv-compress --kv-compress-bits 4 --max-kv-size 4096

# Sliding window only (no compression)
mlx-fun serve --model ./model --max-kv-size 8192

# Combine with steering
mlx-fun steer --model ./model --safety-map sr.json --mode safe \
    --prompt "..." --kv-compress --kv-compress-bits 4
```

**How it works:**

1. **Random rotation** — A Haar-distributed orthogonal matrix is generated
   per attention head via QR decomposition. This rotation spreads outlier
   channel values across all dimensions, making the distribution more uniform.
2. **Quantize** — The rotated K/V are quantized using MLX's `mx.quantize()`
   at the configured bit-width. Removed channel outliers → significantly
   lower quantization error at the same bit budget.
3. **Store** — Rotated-quantized K/V land in MLX's native packed format,
   pre-allocated with 256-token step allocation.
4. **Attention** — For supported models (MiniMax, GLM4, Qwen3, Qwen3-Next),
   the SDPA function is patched to rotate queries with the same matrix,
   enabling hardware-accelerated `mx.quantized_matmul`. Since
   `(Q @ R.T) @ (K @ R.T).T = Q @ K.T` for orthogonal R, attention scores are
   mathematically identical. Unsupported models fall back to dequantize +
   inverse-rotate (still saves memory, slower attention).

| Mode | Memory savings | Attention speed | Models |
|---|---|---|---|
| **Quantized SDPA** (default) | 4–6× | hardware-accelerated `quantized_matmul` | MiniMax, GLM4, Qwen3, Qwen3-Next |
| **Plain SDPA** (fallback) | 4–6× | dequantize on read | All models (DeepSeek V3.2, GLM-5, Kimi-K2, etc.) |

| Bits | Quality | Memory | Use |
|---|---|---|---|
| 8 | Near-lossless | ~2× | Quality-critical |
| 4 | Excellent | ~4× | Default |
| 3 | Good | ~5.3× | Memory-constrained |
| 2 | Acceptable | ~8× | Extreme |

## RotorQuant (Clifford rotors)

From [RotorQuant (Scrya, 2026)](https://www.scrya.com/rotorquant/). Replaces
TurboQuant's dense d×d orthogonal matrix with Cl(3,0) rotors — **44× fewer
rotation parameters** (4 per 3D group vs d²), at matching compression
fidelity.

```bash
# 3-bit RotorQuant (default)
mlx-fun serve --model ./model --kv-compress rotor --kv-compress-bits 3

# Combine with sliding window
mlx-fun serve --model ./model --kv-compress rotor --kv-compress-bits 3 --max-kv-size 4096

# Smoke test
mlx-fun smoke-test --model ./pruned --kv-compress rotor --kv-compress-bits 3
```

**How it works:**

1. **Chunk** — K/V vectors (d dimensions) are split into groups of 3.
2. **Embed** — Each group becomes a grade-1 Cl(3,0) multivector `(e1, e2, e3)`.
3. **Rotor sandwich** — Per-group random rotor R decorrelates via `R v R̃`.
   Each rotor has only 4 parameters (scalar + 3 bivector), vs d² for a dense
   matrix.
4. **Lloyd-Max quantize** — Grade-1 components quantized using a precomputed
   optimal codebook for the Gaussian distribution arising from random
   rotation.
5. **Dequantize on read** — Look up centroids, undo rotor rotation,
   reconstruct vectors. Plain SDPA only (no `quantized_matmul`).

## Comparison

| | TurboQuant | RotorQuant |
|---|---|---|
| **Rotation params** | d² (e.g. 16,384 for d=128) | ~4 × d/3 (e.g. 172 for d=128) |
| **Quantization codebook** | MLX affine (uniform) | Lloyd-Max optimal Gaussian |
| **Hardware accel** | `mx.quantized_matmul` for supported models | Plain SDPA only |
| **Best for** | Quantized-SDPA-supported models, bit speed | All models, minimal parameter overhead |

## Stacking with weight quantization

KV compression is orthogonal to weight quantization. Combine for maximum
savings:

```bash
# Q3 weights + 3-bit RotorQuant KV
mlx-fun serve --model ./model-Q3-g64 --kv-compress rotor --kv-compress-bits 3

# NVFP4 weights + 4-bit TurboQuant KV
mlx-fun serve --model ./model-NVFP4 --kv-compress --kv-compress-bits 4
```

For long-context workloads, KV compression typically gives more memory savings
than further weight compression past 4 bits, because KV grows linearly with
sequence length.
