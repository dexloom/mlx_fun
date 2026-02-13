"""Abliteration: refusal direction orthogonalization for MoE models.

Computes the refusal direction from mean activation differences between
harmful and benign prompts, then projects it out of model weight matrices.

Based on Arditi et al. (NeurIPS 2024), adapted for Mixture of Experts.
"""

from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .adapters.base import BaseAdapter
from .observer import _to_numpy


# ---------------------------------------------------------------------------
# Residual stream hooks (hook decoder layers, not MoE blocks)
# ---------------------------------------------------------------------------

def _layer_capture_call(self, x, *args, **kwargs):
    """Replacement __call__ for a decoder layer that captures input residual stream."""
    mx.eval(x)
    self._abliterate_captures.append(_to_numpy(x))
    # Call original forward via the base class
    return self.__class__.__bases__[0].__call__(self, x, *args, **kwargs)


def install_residual_hooks(model, layer_indices: List[int]) -> List:
    """Install hooks on decoder layers to capture residual stream.

    Args:
        model: The MLX model (model.model.layers accessible).
        layer_indices: Which decoder layers to hook.

    Returns:
        List of hooked layer objects (for later removal).
    """
    hooked_layers = []
    for idx in layer_indices:
        layer = model.model.layers[idx]
        layer._abliterate_captures = []
        original_cls = type(layer)
        layer._abliterate_original_cls = original_cls
        hooked_cls = type(
            f"_AbliterateHooked_{original_cls.__name__}",
            (original_cls,),
            {"__call__": _layer_capture_call},
        )
        layer.__class__ = hooked_cls
        hooked_layers.append(layer)
    return hooked_layers


def remove_residual_hooks(hooked_layers: List) -> None:
    """Remove residual stream hooks."""
    for layer in hooked_layers:
        if hasattr(layer, "_abliterate_original_cls"):
            layer.__class__ = layer._abliterate_original_cls
            delattr(layer, "_abliterate_original_cls")
        if hasattr(layer, "_abliterate_captures"):
            delattr(layer, "_abliterate_captures")


def collect_residual_captures(hooked_layers: List) -> List[List[np.ndarray]]:
    """Collect and clear captured residual stream data.

    Returns:
        List (per layer) of lists of numpy arrays, each (batch, seq, hidden_dim).
    """
    all_captures = []
    for layer in hooked_layers:
        captures = getattr(layer, "_abliterate_captures", [])
        all_captures.append(list(captures))
        layer._abliterate_captures = []
    return all_captures


# ---------------------------------------------------------------------------
# Refusal direction computation
# ---------------------------------------------------------------------------

def compute_refusal_directions(
    model,
    adapter: BaseAdapter,
    harmful_samples: List[mx.array],
    benign_samples: List[mx.array],
    layer_indices: Optional[List[int]] = None,
    extraction_position: str = "last",
) -> Dict[int, np.ndarray]:
    """Compute refusal direction at each layer.

    For each layer:
    1. Forward harmful samples, capture residual stream activations
    2. Forward benign samples, capture residual stream activations
    3. refusal_direction = normalize(mean_harmful - mean_benign)

    Args:
        model: The MLX model.
        adapter: Model adapter.
        harmful_samples: Tokenized harmful prompts as MLX arrays.
        benign_samples: Tokenized benign prompts as MLX arrays.
        layer_indices: Which layers to target. If None, uses all decoder layers.
        extraction_position: "last" (last token) or "mean" (mean pool).

    Returns:
        Dict[layer_idx, unit_direction_vector] where each vector is (hidden_dim,).
    """
    if layer_indices is None:
        n_layers = len(model.model.layers)
        layer_indices = list(range(n_layers))

    directions = {}
    chunk_size = 4  # Hook a few layers at a time to limit memory

    for chunk_start in range(0, len(layer_indices), chunk_size):
        chunk_layers = layer_indices[chunk_start : chunk_start + chunk_size]

        # --- Harmful dataset ---
        harmful_vecs = {idx: [] for idx in chunk_layers}
        hooked = install_residual_hooks(model, chunk_layers)
        for sample in harmful_samples:
            tokens = sample.reshape(1, -1)
            model(tokens)
            captures = collect_residual_captures(hooked)
            for i, layer_idx in enumerate(chunk_layers):
                for cap in captures[i]:
                    vec = _extract_position(cap, extraction_position)
                    harmful_vecs[layer_idx].append(vec)
        remove_residual_hooks(hooked)

        # --- Benign dataset ---
        benign_vecs = {idx: [] for idx in chunk_layers}
        hooked = install_residual_hooks(model, chunk_layers)
        for sample in benign_samples:
            tokens = sample.reshape(1, -1)
            model(tokens)
            captures = collect_residual_captures(hooked)
            for i, layer_idx in enumerate(chunk_layers):
                for cap in captures[i]:
                    vec = _extract_position(cap, extraction_position)
                    benign_vecs[layer_idx].append(vec)
        remove_residual_hooks(hooked)

        # --- Compute direction ---
        for layer_idx in chunk_layers:
            if not harmful_vecs[layer_idx] or not benign_vecs[layer_idx]:
                continue
            h_mean = np.concatenate(harmful_vecs[layer_idx], axis=0).mean(axis=0)
            b_mean = np.concatenate(benign_vecs[layer_idx], axis=0).mean(axis=0)
            direction = (h_mean - b_mean).astype(np.float64)
            norm = np.linalg.norm(direction)
            if norm > 1e-8:
                direction = direction / norm
            directions[layer_idx] = direction

    return directions


def _extract_position(cap: np.ndarray, position: str) -> np.ndarray:
    """Extract activation vector from captured residual stream.

    Args:
        cap: (batch, seq, hidden_dim) captured activations.
        position: "last" or "mean".

    Returns:
        (batch, hidden_dim) float64 array.
    """
    if position == "last":
        return cap[:, -1, :].astype(np.float64)
    elif position == "mean":
        return cap.mean(axis=1).astype(np.float64)
    else:
        raise ValueError(f"Unknown extraction_position '{position}'")


def auto_select_layers(
    refusal_directions: Dict[int, np.ndarray],
    top_fraction: float = 0.5,
) -> List[int]:
    """Select layers with strongest refusal directions.

    Returns the top_fraction of layers ranked by refusal direction norm.
    """
    norms = {idx: np.linalg.norm(d) for idx, d in refusal_directions.items()}
    sorted_layers = sorted(norms, key=lambda k: norms[k], reverse=True)
    n_select = max(1, int(len(sorted_layers) * top_fraction))
    return sorted(sorted_layers[:n_select])


# ---------------------------------------------------------------------------
# Weight orthogonalization
# ---------------------------------------------------------------------------

def orthogonalize_weights(
    model,
    adapter: BaseAdapter,
    refusal_directions: Dict[int, np.ndarray],
    target: str = "all",
    safety_report=None,
) -> None:
    """Project refusal direction out of model weight matrices, in-place.

    For each layer: W' = W - (W @ d) * d^T  (d is unit vector)

    Args:
        model: The MLX model (weights modified in-place).
        adapter: Model adapter.
        refusal_directions: Per-layer refusal direction unit vectors.
        target: "all", "safety-experts", or "dense-only".
        safety_report: Required if target="safety-experts".
    """
    moe_indices = set(adapter.moe_layer_indices())

    for layer_idx, direction in refusal_directions.items():
        d = mx.array(direction.astype(np.float32))
        layer = model.model.layers[layer_idx]

        # Orthogonalize attention output projection
        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "o_proj"):
            _orthogonalize_linear(layer.self_attn.o_proj, d)

        # MoE layer handling
        if layer_idx in moe_indices and target != "dense-only":
            moe_block = adapter.get_moe_block(layer_idx)
            switch_mlp = adapter.get_switch_mlp(moe_block)

            if target == "all":
                _orthogonalize_switch_proj(switch_mlp.down_proj, d)
            elif target == "safety-experts" and safety_report is not None:
                safety_ids = safety_report.safety_critical.get(layer_idx, [])
                for expert_id in safety_ids:
                    _orthogonalize_expert_proj(switch_mlp.down_proj, expert_id, d)

            # Shared experts if present
            if hasattr(moe_block, "shared_experts") and moe_block.shared_experts is not None:
                if hasattr(moe_block.shared_experts, "weight"):
                    _orthogonalize_linear(moe_block.shared_experts, d)
        elif layer_idx not in moe_indices:
            # Dense layer — orthogonalize MLP output projection
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "down_proj"):
                _orthogonalize_linear(layer.mlp.down_proj, d)


def _orthogonalize_linear(linear, d: mx.array) -> None:
    """Project direction d out of a linear layer's weight matrix.

    W' = W - (W @ d)[:, None] * d[None, :]  (d is unit vector)
    """
    w = linear.weight
    # Handle quantized weights: dequantize first
    is_quantized = hasattr(linear, "scales") and linear.scales is not None
    if is_quantized:
        w = mx.dequantize(
            w, linear.scales, linear.biases,
            group_size=linear.group_size, bits=linear.bits,
        )

    # W @ d -> (out_features,), then outer product subtraction
    proj = w @ d
    w_new = w - proj[:, None] * d[None, :]
    mx.eval(w_new)

    linear.weight = w_new
    # Clear quantization params (store as float)
    if is_quantized:
        linear.scales = None
        linear.biases = None


def _orthogonalize_switch_proj(proj, d: mx.array) -> None:
    """Orthogonalize all experts in a SwitchLinear projection.

    For down_proj: weight shape is (n_experts, out_features, in_features).
    d is in the output space (hidden_dim).
    """
    w = proj.weight
    is_quantized = hasattr(proj, "scales") and proj.scales is not None
    if is_quantized:
        w = mx.dequantize(
            w, proj.scales, proj.biases,
            group_size=proj.group_size, bits=proj.bits,
        )

    # Batched projection: for each expert, project out d from output space
    # proj_coeff[e, i] = sum_o(d[o] * w[e, o, i]) = einsum("o, eoi -> ei")
    proj_coeff = mx.einsum("o,eoi->ei", d, w)
    # w_new[e, o, i] = w[e, o, i] - d[o] * proj_coeff[e, i]
    w_new = w - d[None, :, None] * proj_coeff[:, None, :]
    mx.eval(w_new)

    proj.weight = w_new
    if is_quantized:
        proj.scales = None
        proj.biases = None


def _orthogonalize_expert_proj(proj, expert_id: int, d: mx.array) -> None:
    """Orthogonalize a single expert's weights in a SwitchLinear."""
    w = proj.weight
    is_quantized = hasattr(proj, "scales") and proj.scales is not None
    if is_quantized:
        w = mx.dequantize(
            w, proj.scales, proj.biases,
            group_size=proj.group_size, bits=proj.bits,
        )

    expert_w = w[expert_id]  # (out_features, in_features)
    proj_coeff = d @ expert_w  # (in_features,)
    expert_w_new = expert_w - d[:, None] * proj_coeff[None, :]
    mx.eval(expert_w_new)

    # Write back using concatenation (safer than .at[].set())
    parts = []
    if expert_id > 0:
        parts.append(w[:expert_id])
    parts.append(expert_w_new[None, :, :])
    if expert_id < w.shape[0] - 1:
        parts.append(w[expert_id + 1 :])
    w_new = mx.concatenate(parts, axis=0)
    mx.eval(w_new)

    proj.weight = w_new
    if is_quantized:
        proj.scales = None
        proj.biases = None
