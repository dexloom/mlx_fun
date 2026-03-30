# Converting NVIDIA NVFP4 Checkpoints to MLX

## Background

NVIDIA's Nemotron models (e.g., `NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4`) are
**natively trained in NVFP4** using quantization-aware training (QAT). The NVFP4
weights are the ground truth — not a post-training compression of BF16. This
means re-quantizing from the BF16 checkpoint would produce _different_ (and
worse) FP4 weight codes than what was actually trained.

However, NVIDIA stores these weights in their proprietary `modelopt` format,
which `mlx-lm` cannot load directly. The `convert-nvfp4` command bridges this
gap by repacking the native FP4 weights into MLX's format.

## NVIDIA vs MLX NVFP4 Format

Both formats use the same fundamental representation:

| Component | NVIDIA (modelopt) | MLX |
|---|---|---|
| FP4 encoding | E2M1 (1+2+1 bits) | E2M1 (identical) |
| Byte packing | uint8 — 2 values/byte | uint32 — 8 values/word |
| Group scales | E4M3 (group_size=16) | E4M3 (group_size=16) |
| Global scale | float32 per-tensor (`weight_scale_2`) | **Not supported on Metal** |

The conversion handles two incompatibilities:

1. **Repacking**: `uint8 [M, N/2]` -> `uint32 [M, N/8]` via `numpy.view` — this
   is a zero-cost byte reinterpretation. The nibble ordering is identical.

2. **Global scale folding**: NVIDIA uses a two-level scale hierarchy
   (`value = fp4 * e4m3_group_scale * f32_global_scale`). MLX only supports
   single-level scales. The converter folds `weight_scale_2` into the per-group
   E4M3 scales by decoding to float, multiplying, and re-encoding. This
   introduces ~1-2% mean relative error from E4M3 rounding, but the **FP4
   weight codes themselves are preserved exactly**.

## Mixed-Precision Layers

NVIDIA NVFP4 checkpoints use mixed precision:

| Layer Type | NVIDIA Format | MLX Output |
|---|---|---|
| Routed experts (512 per MoE layer) | NVFP4 (uint8 + e4m3 scales + f32 global) | NVFP4 (uint32 + e4m3 folded scales) |
| Mamba-2 projections | FP8 E4M3 + f32 per-tensor scale | Dequantized to bfloat16 |
| Shared experts | FP8 E4M3 + f32 per-tensor scale | Dequantized to bfloat16 |
| Embeddings, norms, gates | BF16 / F32 | Pass-through |

FP8 layers are dequantized to bfloat16 because MLX does not have native FP8
linear layers. This does not affect quality — FP8 has 256 representable values
and the dequantization to BF16 is lossless.

## Usage

### Prerequisites

```bash
pip install -e ".[convert]"
# or: pip install safetensors huggingface-hub
```

### Convert with native NVFP4 (recommended)

Preserves the trained FP4 weight codes. Expert weights stay in 4-bit, FP8
layers become bfloat16. Output size is approximately the same as the source
(~80 GB for the 120B model).

```bash
mlx-fun convert-nvfp4 \
    --model nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
    --output ./nemotron-120b-mlx-nvfp4
```

### Convert to full bfloat16 (no quantization)

Dequantizes everything to bfloat16. Useful if you want to apply a different
quantization scheme afterward (e.g., MXFP4, affine 4-bit). Output is ~240 GB.

```bash
mlx-fun convert-nvfp4 \
    --model nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
    --output ./nemotron-120b-bf16 \
    --mode dequant
```

### Convert from a local path

If you've already downloaded the checkpoint:

```bash
mlx-fun convert-nvfp4 \
    --model /path/to/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
    --output ./nemotron-120b-mlx-nvfp4
```

## Running After Conversion

The converted checkpoint loads directly with `mlx-lm`:

```python
from mlx_lm import load, generate

model, tokenizer = load("./nemotron-120b-mlx-nvfp4")
response = generate(model, tokenizer, prompt="Hello!", max_tokens=100)
```

Or with mlx-fun for expert analysis:

```bash
# Collect saliency data
mlx-fun collect \
    --model ./nemotron-120b-mlx-nvfp4 \
    --dataset calibration.jsonl \
    --output nemotron_saliency.npz

# Prune experts
mlx-fun prune \
    --model ./nemotron-120b-mlx-nvfp4 \
    --saliency nemotron_saliency.npz \
    --num-experts-to-keep 256 \
    --output ./nemotron-120b-pruned
```

## Output Files

The converter produces:

```
output_dir/
  model-00001-of-NNNNN.safetensors  # Sharded weights (5 GB each)
  ...
  model.safetensors.index.json       # Weight index
  config.json                        # Model config with MLX quantization metadata
  tokenizer.json                     # Tokenizer (copied from source)
  tokenizer_config.json
  special_tokens_map.json
  generation_config.json
  conversion_metadata.json           # Conversion provenance and stats
```

## Memory Requirements

| Model | NVFP4 Output | BF16 Output | RAM Needed |
|---|---|---|---|
| Nemotron-3-Super-120B | ~80 GB | ~240 GB | 128+ GB |

The converter processes all weight shards into memory before saving. For the
120B model, peak memory usage is approximately equal to the output size. A Mac
with 128 GB unified memory should handle the NVFP4 conversion; the BF16
conversion requires 256+ GB.

## Quality Notes

- **FP4 weight codes are preserved exactly** — the trained values are not
  re-quantized. Only the scale representation changes.
- **Scale folding error**: ~1-2% mean relative error from re-encoding
  `group_scale * global_scale` into E4M3. In practice this is within the noise
  floor of FP4 quantization itself.
- **FP8 -> BF16 dequantization is lossless** — every E4M3 value is exactly
  representable in BF16.
- **Do NOT convert from the BF16 checkpoint and re-quantize** if the model was
  natively trained in NVFP4. The BF16 checkpoint is an upcast — converting it
  back to NVFP4 produces different scale partitions and different FP4 codes.

## Supported Models

Any HuggingFace checkpoint using NVIDIA's `modelopt` NVFP4 format with the
`config_groups` or `quantized_layers` quantization config structure. Currently
tested with:

- `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` (`nemotron_h`)
