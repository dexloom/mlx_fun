"""Expert selection and tensor slicing for MoE pruning."""

import warnings
from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import numpy as np

from .adapters.base import BaseAdapter


def load_safety_constraints(
    safety_map_path: str,
    mode: str,
) -> Tuple[Optional[Dict[int, np.ndarray]], Optional[Dict[int, np.ndarray]]]:
    """Load safety report and return expert constraints for pruning.

    Args:
        safety_map_path: Path to safety_report.json.
        mode: "protect" (never prune safety experts) or "target" (always prune them).

    Returns:
        (protected_experts, targeted_experts) — one will be None.
    """
    from .safety import SafetyReport

    report = SafetyReport.load(safety_map_path)
    constraints = {
        k: np.array(v, dtype=np.intp) for k, v in report.safety_critical.items()
    }
    if mode == "protect":
        return constraints, None
    elif mode == "target":
        return None, constraints
    else:
        raise ValueError(f"Unknown safety_mode '{mode}'. Use: protect, target")


def load_domain_constraints(
    domain_map_path: str,
    mode: str,
) -> Tuple[Optional[Dict[int, np.ndarray]], Optional[Dict[int, np.ndarray]]]:
    """Load domain report and return expert constraints for pruning.

    Args:
        domain_map_path: Path to domain_report.json.
        mode: "protect" (never prune domain experts).

    Returns:
        (protected_experts, targeted_experts) — targeted is always None.
    """
    from .domain import DomainReport

    report = DomainReport.load(domain_map_path)
    constraints = {
        k: np.array(v, dtype=np.intp) for k, v in report.domain_experts.items()
    }
    if mode == "protect":
        return constraints, None
    else:
        raise ValueError(f"Unknown domain_mode '{mode}'. Use: protect")


def select_experts_to_keep(
    scores: np.ndarray,
    n_prune: int,
    protected_experts: Dict[int, np.ndarray] = None,
    targeted_experts: Dict[int, np.ndarray] = None,
) -> Dict[int, np.ndarray]:
    """Select which experts to keep per layer based on saliency scores.

    Args:
        scores: (num_layers, num_experts) saliency scores. Higher = more important.
        n_prune: Number of experts to prune per layer.
        protected_experts: Optional dict mapping layer_idx -> expert IDs to never prune.
        targeted_experts: Optional dict mapping layer_idx -> expert IDs to always prune.

    Returns:
        Dict mapping layer_index -> numpy array of kept expert indices (sorted).
    """
    num_layers, num_experts = scores.shape
    if n_prune <= 0:
        return {i: np.arange(num_experts) for i in range(num_layers)}
    if n_prune >= num_experts:
        raise ValueError(
            f"Cannot prune {n_prune} experts when only {num_experts} exist."
        )

    keep_map = {}
    for layer_idx in range(num_layers):
        layer_scores = scores[layer_idx].copy()
        # Apply safety constraints
        if protected_experts and layer_idx in protected_experts:
            layer_scores[protected_experts[layer_idx]] = np.inf
        if targeted_experts and layer_idx in targeted_experts:
            layer_scores[targeted_experts[layer_idx]] = -np.inf
        # Find the n_prune lowest-scoring experts
        prune_indices = np.argpartition(layer_scores, n_prune)[:n_prune]
        keep_indices = np.setdiff1d(np.arange(num_experts), prune_indices)
        keep_map[layer_idx] = np.sort(keep_indices)

    return keep_map


def _strided_prune_indices(group_size: int, n_prune: int) -> np.ndarray:
    """Pick n_prune evenly-spaced indices within a group to prune.

    E.g. group_size=120, n_prune=20 → stride=6, prune at positions 5,11,17,...
    (every stride-th element, 0-indexed).
    """
    if n_prune <= 0:
        return np.array([], dtype=int)
    if n_prune >= group_size:
        return np.arange(group_size)
    stride = group_size / n_prune
    return np.unique(np.round(np.arange(n_prune) * stride + (stride - 1)).astype(int))


def select_experts_to_keep_strided(
    scores: np.ndarray,
    n_prune: int,
    protected_experts: Dict[int, np.ndarray] = None,
    targeted_experts: Dict[int, np.ndarray] = None,
) -> Dict[int, np.ndarray]:
    """Select experts to keep using strided pruning from two groups.

    Splits experts into an important group (top n_experts - n_prune) and an
    unimportant group (bottom n_prune), sorted by saliency. Then prunes
    n_prune // 2 from each group at regular intervals, distributing removals
    evenly across the saliency spectrum.

    Args:
        scores: (num_layers, num_experts) saliency scores. Higher = more important.
        n_prune: Total number of experts to prune per layer.

    Returns:
        Dict mapping layer_index -> numpy array of kept expert indices (sorted).
    """
    num_layers, num_experts = scores.shape
    if n_prune <= 0:
        return {i: np.arange(num_experts) for i in range(num_layers)}
    if n_prune >= num_experts:
        raise ValueError(
            f"Cannot prune {n_prune} experts when only {num_experts} exist."
        )

    n_prune_important = n_prune // 2
    n_prune_unimportant = n_prune - n_prune_important  # gets extra if odd

    keep_map = {}
    for layer_idx in range(num_layers):
        layer_scores = scores[layer_idx].copy()
        # Apply safety constraints
        if protected_experts and layer_idx in protected_experts:
            layer_scores[protected_experts[layer_idx]] = np.inf
        if targeted_experts and layer_idx in targeted_experts:
            layer_scores[targeted_experts[layer_idx]] = -np.inf

        # Sort expert indices by ascending saliency score
        sorted_experts = np.argsort(layer_scores)

        # Split: bottom n_prune = unimportant, top (n_experts - n_prune) = important
        unimportant = sorted_experts[:n_prune]
        important = sorted_experts[n_prune:]

        # Pick evenly-spaced indices to prune within each group
        prune_pos_unimp = _strided_prune_indices(len(unimportant), n_prune_unimportant)
        prune_pos_imp = _strided_prune_indices(len(important), n_prune_important)

        # Map positions back to original expert indices
        prune_set = set()
        for pos in prune_pos_unimp:
            prune_set.add(unimportant[pos])
        for pos in prune_pos_imp:
            prune_set.add(important[pos])

        keep_indices = np.array(
            [e for e in range(num_experts) if e not in prune_set]
        )
        keep_map[layer_idx] = np.sort(keep_indices)

    return keep_map


def _slice_switch_linear(proj, keep: mx.array):
    """Slice a SwitchLinear or QuantizedSwitchLinear on the expert axis."""
    proj.weight = mx.take(proj.weight, keep, axis=0)

    # QuantizedSwitchLinear has scales and biases
    if hasattr(proj, "scales") and proj.scales is not None:
        proj.scales = mx.take(proj.scales, keep, axis=0)
    if hasattr(proj, "biases") and proj.biases is not None:
        proj.biases = mx.take(proj.biases, keep, axis=0)

    # Optional per-expert bias
    if "bias" in proj:
        proj.bias = mx.take(proj.bias, keep, axis=0)


def _slice_linear(linear, keep: mx.array):
    """Slice a Linear or QuantizedLinear on the output (axis 0) dimension."""
    linear.weight = mx.take(linear.weight, keep, axis=0)

    # QuantizedLinear has scales and biases
    if hasattr(linear, "scales") and linear.scales is not None:
        linear.scales = mx.take(linear.scales, keep, axis=0)
    if hasattr(linear, "biases") and linear.biases is not None:
        linear.biases = mx.take(linear.biases, keep, axis=0)

    # Optional bias
    if "bias" in linear:
        linear.bias = mx.take(linear.bias, keep, axis=0)


def prune_moe_layer(
    adapter: BaseAdapter,
    layer_idx: int,
    keep_indices: np.ndarray,
):
    """Prune a single MoE layer to keep only the specified experts.

    Args:
        adapter: Model adapter.
        layer_idx: Decoder layer index.
        keep_indices: Sorted numpy array of expert indices to keep.
    """
    moe_block = adapter.get_moe_block(layer_idx)
    switch_mlp = adapter.get_switch_mlp(moe_block)
    keep = mx.array(keep_indices)

    # Slice SwitchGLU projections
    for proj_name in ("gate_proj", "up_proj", "down_proj"):
        proj = getattr(switch_mlp, proj_name)
        _slice_switch_linear(proj, keep)

    # Slice gate weights (handles both Linear and QuantizedLinear)
    model_type = adapter.config.get("model_type", "")
    if model_type in ("minimax", "minimax_m2"):
        _slice_linear(moe_block.gate, keep)
        moe_block.e_score_correction_bias = mx.take(
            moe_block.e_score_correction_bias, keep, axis=0
        )
    elif model_type in ("glm4_moe", "glm4_moe_lite", "glm_moe_dsa", "deepseek_v32"):
        # MoEGate: raw weight + bias (not nn.Linear, slice manually)
        moe_block.gate.weight = mx.take(moe_block.gate.weight, keep, axis=0)
        moe_block.gate.e_score_correction_bias = mx.take(
            moe_block.gate.e_score_correction_bias, keep, axis=0
        )
        moe_block.gate.n_routed_experts = len(keep_indices)
    elif model_type in ("qwen3_moe", "qwen3_next"):
        _slice_linear(moe_block.gate, keep)
        moe_block.num_experts = len(keep_indices)


def prune_model(
    adapter: BaseAdapter,
    keep_map: Dict[int, np.ndarray],
) -> dict:
    """Prune all MoE layers in the model.

    Args:
        adapter: Model adapter.
        keep_map: Dict mapping MoE layer index -> kept expert indices.

    Returns:
        Updated config dict.
    """
    moe_indices = adapter.moe_layer_indices()
    n_experts = adapter.num_routed_experts()
    top_k = adapter.num_experts_per_tok()

    for layer_idx in moe_indices:
        if layer_idx not in keep_map:
            continue
        keep = keep_map[layer_idx]
        n_remaining = len(keep)

        if n_remaining < top_k:
            raise ValueError(
                f"Layer {layer_idx}: keeping {n_remaining} experts "
                f"but top_k={top_k}. Must keep at least top_k experts."
            )
        if n_remaining == top_k:
            warnings.warn(
                f"Layer {layer_idx}: keeping exactly {n_remaining} experts "
                f"= top_k. All experts will always be selected."
            )

        prune_moe_layer(adapter, layer_idx, keep)

    # Update config
    config = adapter.config.copy()
    # Use the count from the first layer's keep set
    first_keep = next(iter(keep_map.values()))
    config[adapter.config_expert_count_key()] = len(first_keep)

    return config
