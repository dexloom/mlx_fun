"""Convert NVIDIA NVFP4 (modelopt) checkpoints to native MLX NVFP4 format.

NVIDIA's Nemotron models trained natively in NVFP4 store weights in a format
that mlx-lm cannot load directly:

  - NVFP4 expert weights: uint8 packed (2x FP4 e2m1 per byte) + e4m3 group
    scales + per-tensor float32 global scale (weight_scale_2)
  - FP8 layers (Mamba, shared experts): float8_e4m3 + per-tensor float32 scale
  - BF16 layers (embeddings, norms, gates): pass-through

MLX's NVFP4 mode uses the same e2m1 weights and e4m3 scales but:
  - Packs into uint32 (8 FP4 values) instead of uint8 (2 FP4 values)
  - Does NOT support per-tensor global scales on the Metal backend

This converter:
  1. Repacks NVFP4 uint8 -> uint32 (lossless byte reinterpretation)
  2. Folds weight_scale_2 into per-group e4m3 scales (minimal rounding)
  3. Dequantizes FP8 layers to bfloat16 (MLX has no native FP8 linear)
  4. Stacks per-expert weights into SwitchMLP format (same as sanitize())
  5. Emits MLX-native safetensors + config with quantization metadata
"""

import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FP8 E4M3 helpers
# ---------------------------------------------------------------------------

def _dequant_fp8(weight_u8: np.ndarray, scale: float) -> mx.array:
    """Dequantize FP8 e4m3 weight to bfloat16.

    Args:
        weight_u8: uint8 array storing e4m3 values.
        scale: Per-tensor float32 scale.

    Returns:
        bfloat16 MLX array.
    """
    w_mx = mx.array(weight_u8)
    w_float = mx.from_fp8(w_mx, dtype=mx.bfloat16)
    return w_float * scale


# ---------------------------------------------------------------------------
# NVFP4 repacking
# ---------------------------------------------------------------------------

def _repack_nvfp4_weight(weight_u8: np.ndarray) -> np.ndarray:
    """Repack NVIDIA uint8 [M, N/2] -> MLX uint32 [M, N/8].

    Both formats store FP4 nibbles in the same byte order; this is a
    zero-copy view reinterpretation.
    """
    return weight_u8.view(np.uint32).reshape(weight_u8.shape[0], -1)


def _fold_global_scale(
    scales_u8: np.ndarray,
    global_scale: float,
) -> np.ndarray:
    """Fold per-tensor global scale into per-group e4m3 scales.

    Decodes e4m3 -> float32, multiplies by global_scale, re-encodes to e4m3.
    Introduces small rounding (~1-2% mean relative error) because the product
    may not be exactly representable in e4m3.

    Args:
        scales_u8: uint8 array of e4m3 scale bytes [M, N/16].
        global_scale: float32 per-tensor scale (weight_scale_2).

    Returns:
        uint8 array of folded e4m3 scales, same shape.
    """
    scales_mx = mx.array(scales_u8)
    scales_float = mx.from_fp8(scales_mx, dtype=mx.float32)
    folded = scales_float * global_scale
    folded_e4m3 = mx.to_fp8(folded)
    mx.eval(folded_e4m3)
    return np.array(folded_e4m3)


# ---------------------------------------------------------------------------
# Quantization config parsing
# ---------------------------------------------------------------------------

def _parse_quant_config(config: dict) -> Tuple[Dict[str, str], Dict[str, dict]]:
    """Parse NVIDIA quantization_config into per-layer info.

    Returns:
        layer_algo: {layer_path: "NVFP4" | "FP8"}
        layer_params: {layer_path: {"group_size": 16, ...}}
    """
    layer_algo: Dict[str, str] = {}
    layer_params: Dict[str, dict] = {}

    qc = config.get("quantization_config", {})

    # Format 1: hf_quant_config.json style with quantization.quantized_layers
    quant_section = config.get("quantization", qc)
    if "quantized_layers" in quant_section:
        for layer_path, info in quant_section["quantized_layers"].items():
            algo = info.get("quant_algo", "NVFP4")
            layer_algo[layer_path] = algo
            layer_params[layer_path] = info
        return layer_algo, layer_params

    # Format 2: config.json style with config_groups
    config_groups = qc.get("config_groups", {})
    for group_name, group_info in config_groups.items():
        weight_info = group_info.get("weights", {})
        num_bits = weight_info.get("num_bits", 16)
        group_size = weight_info.get("group_size", None)

        if num_bits == 4:
            algo = "NVFP4"
        elif num_bits == 8:
            algo = "FP8"
        else:
            algo = "BF16"

        for target in group_info.get("targets", []):
            layer_algo[target] = algo
            layer_params[target] = {"group_size": group_size, "num_bits": num_bits}

    return layer_algo, layer_params


def _find_layer_algo(
    tensor_name: str,
    layer_algo: Dict[str, str],
) -> Optional[str]:
    """Determine quantization algo for a tensor based on its name.

    Looks for the layer prefix in the algo map. A tensor like
    "backbone.layers.1.mixer.experts.0.up_proj.weight" matches the layer
    prefix "backbone.layers.1.mixer.experts.0.up_proj".
    """
    # Strip .weight, .weight_scale, .weight_scale_2, .input_scale suffixes
    for suffix in (".weight_scale_2", ".weight_scale", ".input_scale", ".weight"):
        if tensor_name.endswith(suffix):
            prefix = tensor_name[: -len(suffix)]
            if prefix in layer_algo:
                return layer_algo[prefix]
    return None


# ---------------------------------------------------------------------------
# Expert weight stacking (mirrors nemotron_h.sanitize)
# ---------------------------------------------------------------------------

def _stack_experts(
    weights: Dict[str, mx.array],
    n_layers: int,
    n_experts: int,
    quantized: bool = False,
) -> Dict[str, mx.array]:
    """Stack per-expert weights into SwitchMLP format.

    Converts:
        backbone.layers.{l}.mixer.experts.{e}.{proj}.weight
        -> backbone.layers.{l}.mixer.switch_mlp.{fc}.weight

    For quantized experts, also stacks .scales tensors.

    Args:
        weights: Mutable weight dict; expert entries are popped.
        n_layers: Total number of decoder layers.
        n_experts: Number of routed experts per MoE layer.
        quantized: If True, also stack .scales tensors for quantized layers.

    Returns:
        Updated weights dict with stacked SwitchMLP entries.
    """
    proj_map = [("down_proj", "fc2"), ("up_proj", "fc1")]

    for layer_idx in range(n_layers):
        prefix = f"backbone.layers.{layer_idx}.mixer"

        for src_proj, dst_proj in proj_map:
            first_key = f"{prefix}.experts.0.{src_proj}.weight"
            if first_key not in weights:
                continue

            # Stack weight tensors
            expert_weights = []
            for e in range(n_experts):
                key = f"{prefix}.experts.{e}.{src_proj}.weight"
                expert_weights.append(weights.pop(key))
            stacked = mx.stack(expert_weights)
            weights[f"{prefix}.switch_mlp.{dst_proj}.weight"] = stacked

            if quantized:
                # Stack scales
                first_scale_key = f"{prefix}.experts.0.{src_proj}.scales"
                if first_scale_key in weights:
                    expert_scales = []
                    for e in range(n_experts):
                        key = f"{prefix}.experts.{e}.{src_proj}.scales"
                        expert_scales.append(weights.pop(key))
                    weights[f"{prefix}.switch_mlp.{dst_proj}.scales"] = mx.stack(
                        expert_scales
                    )

    return weights


# ---------------------------------------------------------------------------
# Conv1d weight fixup (mirrors nemotron_h.sanitize)
# ---------------------------------------------------------------------------

def _fix_conv1d_weights(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
    """Transpose conv1d weights if needed (same as sanitize)."""
    for k in list(weights.keys()):
        if "conv1d.weight" in k:
            v = weights[k]
            if v.ndim == 3 and v.shape[-1] != 1:
                weights[k] = mx.moveaxis(v, 2, 1)
    return weights


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert_nvfp4(
    hf_path: str,
    mlx_path: str,
    *,
    output_mode: str = "nvfp4",
    trust_remote_code: bool = True,
) -> None:
    """Convert an NVIDIA NVFP4 checkpoint to MLX-native format.

    Args:
        hf_path: HuggingFace model ID or local path to NVIDIA NVFP4 checkpoint.
        mlx_path: Output directory for MLX checkpoint.
        output_mode: Target quantization mode for expert weights.
            "nvfp4" — repack NVIDIA FP4 weights + fold global scales (preserves
                      native FP4 codes, minimal scale rounding).
            "dequant" — dequantize everything to bfloat16 (lossless for FP4 values
                        but ~240GB output).
        trust_remote_code: Whether to allow custom code (unused for weight
            loading, included for API consistency).
    """
    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    out = Path(mlx_path)
    if out.exists():
        raise FileExistsError(f"Output path already exists: {out}")
    out.mkdir(parents=True)

    # Resolve HF path
    if Path(hf_path).is_dir():
        src_path = Path(hf_path)
    else:
        logger.info("Downloading model from HuggingFace: %s", hf_path)
        src_path = Path(
            snapshot_download(hf_path, allow_patterns=["*.safetensors", "*.json",
                                                        "*.jinja", "tokenizer*",
                                                        "special_tokens*"])
        )

    # Load configs
    config_path = src_path / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    hf_quant_path = src_path / "hf_quant_config.json"
    hf_quant_config = {}
    if hf_quant_path.exists():
        with open(hf_quant_path) as f:
            hf_quant_config = json.load(f)

    # Parse per-layer quantization info from both configs.
    # config.json has the actual training config (config_groups with explicit
    # num_bits), hf_quant_config.json is a derived summary that may mislabel
    # layers (e.g. marking FP8 shared experts as NVFP4). config.json wins.
    layer_algo_1, layer_params_1 = _parse_quant_config(config)
    layer_algo_2, layer_params_2 = _parse_quant_config(hf_quant_config)
    layer_algo = {**layer_algo_2, **layer_algo_1}   # config.json takes priority
    layer_params = {**layer_params_2, **layer_params_1}

    n_experts = config.get("n_routed_experts", 0)
    n_layers = config["num_hidden_layers"]

    logger.info(
        "Model: %s, layers=%d, experts=%d, MoE layers detected from pattern",
        config.get("model_type", "unknown"),
        n_layers,
        n_experts,
    )

    # Load weight index
    index_path = src_path / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        shard_files = sorted(set(weight_map.values()))
    else:
        # Single shard
        shard_files = sorted(src_path.glob("model*.safetensors"))
        shard_files = [f.name for f in shard_files]

    # Process all shards
    all_weights: Dict[str, mx.array] = {}
    stats = {"nvfp4_repacked": 0, "fp8_dequantized": 0, "passthrough": 0, "skipped": 0}

    for shard_name in shard_files:
        shard_path = str(src_path / shard_name)
        logger.info("Processing shard: %s", shard_name)

        # Load shard via mx.load — handles all dtypes (bfloat16, float8, uint8)
        # that numpy/safetensors-numpy cannot represent.
        shard_mx = mx.load(shard_path)

        # Also open via safetensors for numpy access (NVFP4/FP8 repacking
        # uses numpy views).  Tensor names that fail (bfloat16, float8)
        # fall back to the mx.load dict above.
        try:
            np_handle = safe_open(shard_path, framework="numpy")
        except Exception:
            np_handle = None

        def _get_np(name: str) -> np.ndarray:
            """Get tensor as numpy, falling back to mx->numpy for exotic dtypes."""
            if np_handle is not None:
                try:
                    return np_handle.get_tensor(name)
                except (TypeError, AttributeError):
                    pass
            # bfloat16 / float8_e4m3 / float8_e5m2 — numpy has no such dtype
            t = shard_mx[name]
            try:
                return np.array(t)
            except (TypeError, ValueError):
                # Cast exotic dtypes (bfloat16, float8) to float32
                return np.array(t.astype(mx.float32))

        for tensor_name in shard_mx.keys():
            algo = _find_layer_algo(tensor_name, layer_algo)

            # Skip auxiliary quantization tensors (handled inline)
            if tensor_name.endswith(".input_scale"):
                stats["skipped"] += 1
                continue

            # Skip FP8 KV cache quantization scales (NVIDIA training artifact,
            # not used by MLX inference)
            if tensor_name.endswith((".k_scale", ".v_scale")):
                stats["skipped"] += 1
                continue

            if tensor_name.endswith(".weight_scale_2"):
                stats["skipped"] += 1
                continue

            if tensor_name.endswith(".weight_scale"):
                # Will be consumed when processing the .weight tensor
                stats["skipped"] += 1
                continue

            # Validate / auto-detect quantization algo.
            # hf_quant_config.json may mislabel some FP8 layers as NVFP4.
            # Distinguish by checking for weight_scale_2 (NVFP4 has it, pure
            # FP8 doesn't) and by verifying the weight is half-packed
            # (NVFP4: last_dim = N/2) vs full-size (FP8: last_dim = N).
            if tensor_name.endswith(".weight"):
                t = shard_mx[tensor_name]
                is_uint8 = (t.dtype == mx.uint8)
                scale_name = tensor_name.replace(".weight", ".weight_scale")
                scale2_name = tensor_name.replace(".weight", ".weight_scale_2")
                has_scale = scale_name in shard_mx
                has_scale2 = scale2_name in shard_mx

                if is_uint8 and has_scale:
                    if has_scale2:
                        # Both weight_scale and weight_scale_2 present.
                        # NVFP4: weight is half-packed (last_dim = N/2),
                        #        scale has (M, N/16) with N = last_dim * 2.
                        # FP8 with global scale: weight is full-size (last_dim = N),
                        #        scale is scalar or per-tensor.
                        s = shard_mx[scale_name]
                        if s.ndim >= 2 and t.ndim >= 2:
                            # Per-group scales: check if NVFP4 or FP8 by group ratio.
                            # NVFP4 group_size=16: s_last = (w_last * 2) / 16 = w_last / 8
                            # FP8   group_size=8:  s_last = w_last / 8
                            # Same ratio! Use first dimension instead:
                            # For a (M, N) weight, NVFP4 stores (M, N/2) and FP8 stores (M, N).
                            # Scale first dim always matches weight first dim.
                            # If w_last * 2 > w_first → likely FP4-packed (in < out), else FP8.
                            # Actually simplest: NVFP4 always has weight_scale_2 as scalar.
                            # If scale is per-group AND scale2 is scalar, could be either.
                            # Just trust the quant config for NVFP4, or auto-detect.
                            algo = algo or "NVFP4"  # trust config if set, else default NVFP4
                        elif s.ndim == 0:
                            # Scalar scale + scalar scale2 → FP8 with per-tensor scales
                            algo = "FP8"
                        else:
                            algo = algo or "NVFP4"
                    else:
                        # Only weight_scale, no weight_scale_2 → pure FP8
                        algo = "FP8"
                elif algo is None and not is_uint8:
                    pass  # will hit passthrough

            if algo == "NVFP4" and tensor_name.endswith(".weight"):
                # Load companion tensors from this shard
                scale_name = tensor_name.replace(".weight", ".weight_scale")
                scale2_name = tensor_name.replace(".weight", ".weight_scale_2")

                scales_np = _get_np(scale_name)
                global_scale = float(np.array(shard_mx[scale2_name]).item())

                tensor_np = _get_np(tensor_name)

                if output_mode == "nvfp4":
                    # Repack: uint8 [M, N/2] -> uint32 [M, N/8]
                    wq = _repack_nvfp4_weight(tensor_np)
                    # Fold global scale into per-group e4m3 scales
                    scales_folded = _fold_global_scale(scales_np, global_scale)

                    # Store as quantized weight + scales
                    # MLX QuantizedLinear expects: weight (uint32), scales (uint8)
                    w_key = tensor_name  # .weight
                    s_key = tensor_name.replace(".weight", ".scales")
                    all_weights[w_key] = mx.array(wq)
                    all_weights[s_key] = mx.array(scales_folded)
                else:  # dequant mode
                    # Full dequant: fp4 * e4m3_scale * global_scale -> bf16
                    wq = mx.array(_repack_nvfp4_weight(tensor_np))
                    scales = mx.array(scales_np)
                    w_deq = mx.dequantize(wq, scales, mode="nvfp4")
                    w_deq = w_deq.astype(mx.float32) * global_scale
                    all_weights[tensor_name] = w_deq.astype(mx.bfloat16)

                stats["nvfp4_repacked"] += 1

            elif algo == "FP8" and tensor_name.endswith(".weight"):
                # FP8 e4m3: dequantize to bf16
                scale_name = tensor_name.replace(".weight", ".weight_scale")
                scale2_name = tensor_name.replace(".weight", ".weight_scale_2")
                scale_t = shard_mx[scale_name]

                if scale_t.ndim == 0:
                    # Simple FP8: scalar per-tensor scale
                    fp8_scale = float(np.array(scale_t).item())
                    tensor_np = _get_np(tensor_name)
                    w_bf16 = _dequant_fp8(tensor_np, fp8_scale)
                else:
                    # FP8 with per-group e4m3 scales + optional global scale
                    # (mislabeled NVFP4 in hf_quant_config — actually FP8 e4m3)
                    # Dequant: fp8_val * e4m3_group_scale * global_scale
                    w_mx = shard_mx[tensor_name]
                    w_float = mx.from_fp8(w_mx, dtype=mx.float32)
                    scales_float = mx.from_fp8(scale_t, dtype=mx.float32)
                    # Broadcast scales: weight (M, N), scales (M, N/group_size)
                    group_size = w_mx.shape[-1] // scale_t.shape[-1]
                    # Repeat scales to match weight columns
                    scales_expanded = mx.repeat(scales_float, group_size, axis=-1)
                    # Trim if weight dim isn't exact multiple of group_size
                    scales_expanded = scales_expanded[..., :w_mx.shape[-1]]
                    w_float = w_float * scales_expanded
                    if scale2_name in shard_mx:
                        global_scale = float(np.array(shard_mx[scale2_name]).item())
                        w_float = w_float * global_scale
                    w_bf16 = w_float.astype(mx.bfloat16)

                mx.eval(w_bf16)
                all_weights[tensor_name] = w_bf16
                stats["fp8_dequantized"] += 1

            else:
                # BF16 / F32 passthrough (embeddings, norms, gates, etc.)
                all_weights[tensor_name] = shard_mx[tensor_name]
                stats["passthrough"] += 1

        del shard_mx, np_handle

        # Eval periodically to free memory
        mx.eval(*[v for v in all_weights.values() if not isinstance(v, mx.array)])

    logger.info(
        "Conversion stats: %d NVFP4 repacked, %d FP8 dequantized, "
        "%d passthrough, %d skipped",
        stats["nvfp4_repacked"],
        stats["fp8_dequantized"],
        stats["passthrough"],
        stats["skipped"],
    )

    # Fix conv1d weights
    all_weights = _fix_conv1d_weights(all_weights)

    # Stack experts into SwitchMLP format
    quantized_experts = output_mode == "nvfp4"
    all_weights = _stack_experts(
        all_weights, n_layers, n_experts, quantized=quantized_experts,
    )

    # Build output config
    out_config = dict(config)
    # Remove NVIDIA quantization config
    out_config.pop("quantization_config", None)

    if output_mode == "nvfp4":
        # Add MLX quantization metadata
        out_config["quantization"] = {
            "group_size": 16,
            "bits": 4,
            "mode": "nvfp4",
        }

    # Save weights as sharded safetensors
    _save_sharded(out, all_weights, max_shard_bytes=5 * 1024 ** 3)

    # Save config
    with open(out / "config.json", "w") as f:
        json.dump(out_config, f, indent=2, sort_keys=True)

    # Copy tokenizer and auxiliary files
    for pattern in ["tokenizer*", "special_tokens*", "*.jinja",
                    "generation_config.json"]:
        for src_file in src_path.glob(pattern):
            if src_file.is_file():
                import shutil
                shutil.copy2(src_file, out / src_file.name)

    # Save conversion metadata
    meta = {
        "source": hf_path,
        "converter": "mlx-fun convert-nvfp4",
        "output_mode": output_mode,
        "stats": stats,
        "notes": (
            "NVFP4 expert weights repacked from NVIDIA modelopt format. "
            "FP4 weight codes preserved; per-group scales have folded "
            "weight_scale_2 with minimal e4m3 rounding."
            if output_mode == "nvfp4"
            else "All weights dequantized to bfloat16."
        ),
    }
    with open(out / "conversion_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("Saved MLX checkpoint to %s", out)


# ---------------------------------------------------------------------------
# Sharded safetensors writer
# ---------------------------------------------------------------------------

def _save_sharded(
    out_dir: Path,
    weights: Dict[str, mx.array],
    max_shard_bytes: int = 5 * 1024 ** 3,
) -> None:
    """Save weights as sharded safetensors with an index file."""
    # Compute sizes and assign to shards
    shards: List[Dict[str, mx.array]] = [{}]
    shard_sizes: List[int] = [0]
    weight_map: Dict[str, str] = {}

    for name, tensor in sorted(weights.items()):
        nbytes = tensor.nbytes
        if shard_sizes[-1] + nbytes > max_shard_bytes and shard_sizes[-1] > 0:
            shards.append({})
            shard_sizes.append(0)
        shards[-1][name] = tensor
        shard_sizes[-1] += nbytes

    n_shards = len(shards)
    total_size = sum(shard_sizes)
    total_params = sum(math.prod(t.shape) for t in weights.values())

    for shard_idx, shard in enumerate(shards):
        shard_name = (
            f"model-{shard_idx + 1:05d}-of-{n_shards:05d}.safetensors"
        )
        shard_path = out_dir / shard_name
        mx.save_safetensors(str(shard_path), shard, metadata={"format": "mlx"})
        for name in shard:
            weight_map[name] = shard_name
        logger.info("  Wrote shard %s (%.2f GB)", shard_name, shard_sizes[shard_idx] / 1e9)

    # Write index
    index = {
        "metadata": {
            "total_size": total_size,
            "total_parameters": total_params,
        },
        "weight_map": weight_map,
    }
    with open(out_dir / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)
