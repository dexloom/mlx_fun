# Quantization on Apple Silicon

This guide covers the quantization techniques mlx_fun supports for MoE models on
Apple Silicon, from one-shot `bf16 → MLX-native` conversions to streaming
re-quantization of already-quantized sources, and the **mixed-precision**
recipe we use to bridge the size/quality gap between MXFP4 and MXFP8.

## Landscape

MLX exposes four quantization modes via `mx.quantize(..., mode=...)`. All four
share the same fused dequant+matmul Metal kernel — within ~5% on decode speed,
the choice is about size and quality, not throughput.

| Mode | Bits | Default group | Effective bpw | Format | When to use |
|---|---|---|---|---|---|
| `affine` | 2/3/4/6/8 | 32 / 64 | bits + 32/group | symmetric INT codes + scale + bias | General-purpose; widest bit range |
| `mxfp4` | 4 | 32 | ~4.25 | e2m1 codes + e4m3 group scale | Standard 4-bit with FP-shaped distribution |
| `mxfp8` | 8 | 32 | ~8.25 | e4m3 codes + e4m3 group scale | Near-lossless 8-bit at minimal overhead |
| `nvfp4` | 4 | 16 | ~4.5 | e2m1 codes + e4m3 group scale, smaller groups | 4-bit with tighter groups → better quality |

For a typical 100-300B MoE model, the size ranking is roughly:
`affine Q3 g64` (3.5 bpw) <
`affine Q4 g64` (4.0) <
`MXFP4 g32` (4.25) <
`NVFP4 g16` (4.5) <
`affine Q4 g32` (4.5) <
`MXFP8 g32` (8.25).

## Path 1: bf16 → MLX-native

For HuggingFace checkpoints stored in bf16/fp16 (most upstream models), the
`mlx_lm.convert` function is the tool of choice. It loads the source, applies
`mx.quantize` per Linear layer, and writes a sharded MLX checkpoint that
`mlx_lm.load()` reads natively.

The driver scripts under `local/scripts/` follow a uniform pattern — a couple
of constants and one call:

- `quant_glm51_q3.py` — affine Q3 g32
- `quant_glm51_mxfp8.py` — MXFP8 g32
- `quant_glm51_nvfp4.py` — NVFP4 g16
- `quant_nvfp4.py` — generic CLI driver, takes `<src>` `<dst>` `[--bits]` `[--group-size]` `[--trust-remote-code]`

```python
from mlx_lm import convert

convert(
    hf_path="/path/to/bf16/source",
    mlx_path="/path/to/output",
    quantize=True,
    q_mode="nvfp4",      # or "mxfp4", "mxfp8", "affine"
    q_bits=4,
    q_group_size=16,     # 16 for NVFP4, 32 for MXFP4/8, 32-128 for affine
)
```

For trust-remote-code models (Kimi-K2, MiniMax-M2, etc.) the generic CLI
forwards `--trust-remote-code` to the convert call.

## Path 2: Mixed precision (NVFP4 bulk + MXFP8 sensitive)

A pure 4-bit conversion gives you the smallest model and the same speed as
8-bit, but typically loses 0.5–1.0 PPL on instruction-following and tool-call
adherence. A pure 8-bit conversion is bigger than necessary because not all
layers are equally sensitive.

The middle ground: keep the **bulk of params** (routed expert weights — usually
85–95% of the total) at NVFP4, and pin the **sensitivity-critical small
layers** at MXFP8. The MXFP8 portion is small enough that decode speed stays
indistinguishable from pure NVFP4 — memory bandwidth dominates and the experts
are still the hot path.

`mlx_lm.convert` supports this via the `quant_predicate` callable. The
predicate receives `(path, module)` and returns either `True/False` (use the
default mode) or a `dict` (per-layer override stored in `config.json`). At
load time, `mlx_lm.load()` reads these per-layer overrides and applies the
right mode per module — no special loader needed.

### Recipe

```python
import re
from mlx_lm import convert

# Anything matching one of these path regexes uses the MXFP8 override;
# everything else gets the default mode passed via convert (NVFP4 g16).
SENSITIVE_PATTERNS = [
    re.compile(r"^lm_head$"),
    re.compile(r"^model\.embed_tokens$"),
    re.compile(r"^model\.layers\.\d+\.self_attn\.[a-z_]+_proj$"),
    re.compile(r"^model\.layers\.\d+\.self_attn\.kv_a_proj_with_mqa$"),
    re.compile(r"^model\.layers\.\d+\.mlp\.gate$"),     # MoE router
    re.compile(r"^model\.layers\.\d+\.mlp\.shared_experts\.(gate|up|down)_proj$"),
]
MXFP8_OVERRIDE = {"group_size": 32, "bits": 8, "mode": "mxfp8"}


def quant_predicate(path, module):
    for pat in SENSITIVE_PATTERNS:
        if pat.match(path):
            return MXFP8_OVERRIDE
    return True  # default: NVFP4 g16 4-bit


convert(
    hf_path=SRC,
    mlx_path=DST,
    quantize=True,
    q_mode="nvfp4",
    q_bits=4,
    q_group_size=16,
    quant_predicate=quant_predicate,
)
```

`local/scripts/quant_glm51_nvfp4_mixed.py` is the GLM-5.1 driver of this
recipe. Drop the same predicate into a different `SRC/DST` and you have a
MiniMax-M2 / Qwen3-MoE / Gemma4 mixed-precision quant.

### What gets which mode

For DeepSeek-V3-style MoE (GLM-5, Kimi-K2, GLM-4.7-MoE) and similar:

| Layer | What it does | Sensitivity | Mode | Reason |
|---|---|---|---|---|
| `model.embed_tokens` | Token → hidden vector | high (rare tokens) | **MXFP8** | one-time per-token, cost is irrelevant; embedding error compounds across the depth |
| `lm_head` | Hidden → logits | high | **MXFP8** | direct effect on sampled tokens |
| `model.layers.*.self_attn.*_proj` | Q / K / V / O linear projections | high | **MXFP8** | attention is the global mixer; small errors cascade across positions |
| `model.layers.*.mlp.gate` | MoE router (Linear) | high | **MXFP8** | a wrong route picks an unrelated expert; one error per token |
| `model.layers.*.mlp.shared_experts.*` | Always-on expert (small share) | medium | **MXFP8** | only ~1–3% of params, applied to every token |
| `model.layers.*.mlp.experts.*` (= `mlp.switch_mlp.*` after sanitize) | Routed experts | low (per-expert) | **NVFP4** | dominant share of params; each expert sees a small fraction of tokens, so small errors don't compound |
| `model.layers.*.mlp.{gate,up,down}_proj` (dense early layers) | Pre-MoE dense MLP | low | **NVFP4** | small in count and intercepted by attention sensitivity |
| `*.norm.weight` / norms | RMSNorm scales | very high | **bf16** (not quantizable) | left untouched by `nn.quantize` |

After conversion, the resulting `config.json` has:

```json
{
  "quantization": {
    "group_size": 16, "bits": 4, "mode": "nvfp4",
    "lm_head": {"group_size": 32, "bits": 8, "mode": "mxfp8"},
    "model.embed_tokens": {"group_size": 32, "bits": 8, "mode": "mxfp8"},
    "model.layers.0.self_attn.q_a_proj": {"group_size": 32, "bits": 8, "mode": "mxfp8"},
    ...
  }
}
```

### Observed result on GLM-5.1

For `zai-org/GLM-5.1` (78 layers, 256 routed experts, ~755B params bf16):

| | Size | Effective bpw | Build time |
|---|---|---|---|
| bf16 source | 1.51 TB | 16 | — |
| affine Q3 g32 | 347 GB | 4.0 | ~25 min |
| MXFP4 g32 | 368 GB | ~4.25 | ~25 min |
| **NVFP4 + MXFP8 mixed** | **397 GB** | **4.585** | **~29 min** |
| MXFP8 g32 | 715 GB | ~9 | ~50 min |

Mixed sat in 539 layers of MXFP8 overrides (312 attn projections + 225 shared
experts + embed + lm_head). The router gate (`mlp.gate`) is a raw `mx.array`
inside the custom `MoEGate` module rather than an `nn.Linear`, so it stays
bf16 — that's fine, it's tiny.

## Path 3: Streaming dequant→requant for already-quantized sources

Some upstream releases ship pre-quantized weights (typically because the model
is too large to distribute in bf16). The HuggingFace `compressed-tensors`
INT4 format is the most common — used by **Kimi-K2.6**, several DeepSeek-V3
variants, and a few large Qwen releases.

`mlx_lm.convert` can't read these directly (the weights aren't bf16). For
these we wrote a **streaming converter** under `local/scripts/` that:

1. Memory-maps the source shards lazily via `mx.load`.
2. For each quantized linear (detected by presence of `.weight_packed`):
   - Views the packed `int32` codes as `uint32` (zero-copy bit reinterpret).
   - Dequantizes to bf16 via MLX's 4-bit affine kernel with
     `biases = -8 * scale` (compressed-tensors stores INT4 as offset-binary,
     so `out = (code - 8) * scale` matches MLX's `out = code * scale + bias`).
   - Re-quantizes the bf16 result to the target mode/bits/group.
3. Passes through bf16 / fp32 weights as-is.
4. Drops vision-tower / multimodal-projector tensors (text-only output).
5. Writes output shards as it goes — peak memory is one expert's worth of bf16
   (~30 MB) plus an output-shard buffer (~3 GB), not the full model.

The reference implementation is `local/scripts/convert_kimi_k2_int4_to_q3.py`.
It supports **resume** — if the process is killed mid-run (memory pressure can
trip macOS into killing the process around 70–80% through), it scans existing
`model-tmp-*.safetensors` shards on restart and continues from where it left
off.

### Format inversion math

compressed-tensors INT4 sym pack-quantized stores codes in **offset-binary**
form (signed value + 8, range [0, 15]) packed 8-per-int32. To dequant via
MLX's 4-bit affine kernel:

```python
w_u32 = packed_int32.view(mx.uint32)              # zero-copy
biases = mx.array(-8.0, dtype=scales.dtype) * scales
bf16 = mx.dequantize(w_u32, scales, biases, group_size=32, bits=4)
```

This gives bf16 weights byte-identical to what compressed-tensors' own
dequantization produces. From there, `mx.quantize(bf16, group_size=64,
bits=3, mode="affine")` (or any other mode) produces MLX-native quantized
output.

### Observed result on Kimi-K2.6

For `moonshotai/Kimi-K2.6` (61 layers, 384 routed experts, ~1T params total,
source: 485 GB compressed-tensors INT4 + bf16 mix):

| | Size | Effective bpw |
|---|---|---|
| Source (compressed-tensors INT4 routed experts + bf16 attn/shared) | 485 GB | ~4.5 |
| **affine Q3 g64** (current production target) | 436 GB | 3.5 |
| affine Q4 g64 | ~498 GB | 4.0 |
| NVFP4 g16 | ~560 GB | 4.5 |

For Kimi specifically, the 512 GB unified-memory budget on a Mac Studio rules
out anything above ~4 bpw. Q3 g64 is the sweet spot. For models that fit at
higher bpw, use Path 1 or Path 2 with the bf16 source instead.

## Path 4: NVIDIA modelopt NVFP4 → MLX-native NVFP4

Some Nemotron releases come in NVIDIA's NVFP4 format with a two-level scale
hierarchy (`fp4_val * e4m3_group_scale * f32_global_scale`). MLX Metal doesn't
support per-tensor global scales, so we fold them into the per-group e4m3
scales via `from_fp8 → multiply → to_fp8` (~1–2% scale rounding, FP4 codes
preserved exactly).

See [`convert-nvfp4.md`](./convert-nvfp4.md) for the dedicated guide. The
implementation is `src/mlx_fun/convert_nvfp4.py`, exposed as
`mlx-fun convert-nvfp4`.

## Choosing the right path

```
Source format?
├─ bf16 / fp16 HuggingFace checkpoint
│  ├─ Want pure 4-bit, fastest quantize → Path 1 with q_mode="nvfp4" or "mxfp4"
│  └─ Want better quality at ~same size → Path 2 (mixed NVFP4 + MXFP8)
├─ compressed-tensors INT4 (Kimi-K2.6, some DSv3) → Path 3 (streaming dequant→requant)
├─ NVIDIA modelopt NVFP4 (Nemotron) → Path 4 (mlx-fun convert-nvfp4)
└─ GGUF (Kimi unsloth, Llama.cpp ecosystem) → mlx_fun does not handle, use llama.cpp
```

## KV cache compression

Quantizing **weights** is one axis. Quantizing the **KV cache** at runtime is
orthogonal — see `docs/kv-cache.md` for TurboQuant (PolarQuant) and RotorQuant
(Clifford rotors). They stack with any of the four weight-quantization paths
above.
