"""REAM: Router-weighted Expert Activation Merging.

Merges experts instead of pruning them, grouping all N experts around k
high-saliency centroids, aligning intermediate neurons, and producing
saliency-weighted averages.
"""

import warnings
from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .adapters.base import BaseAdapter
from .observer import _to_numpy
from .pruner import _slice_linear
from .ream_hooks import install_ream_hooks, remove_ream_hooks, collect_ream_data


# ---------------------------------------------------------------------------
# Expert weight access (handles quantized and non-quantized SwitchLinear)
# ---------------------------------------------------------------------------

def _get_expert_weight(proj, expert_idx: int) -> mx.array:
    """Extract a single expert's weight, dequantizing if needed.

    Args:
        proj: SwitchLinear or QuantizedSwitchLinear projection.
        expert_idx: Index of the expert.

    Returns:
        Dequantized weight as mx.array of shape (out_features, in_features).
    """
    w = proj.weight[int(expert_idx)]
    if hasattr(proj, "scales") and proj.scales is not None:
        s = proj.scales[int(expert_idx)]
        b = proj.biases[int(expert_idx)] if proj.biases is not None else None
        return mx.dequantize(w, s, b, group_size=proj.group_size, bits=proj.bits)
    return w


# ---------------------------------------------------------------------------
# Expert output and intermediate activation computation
# ---------------------------------------------------------------------------

def _compute_single_expert_output(
    switch_mlp, x: mx.array, expert_idx: int,
) -> Tuple[mx.array, mx.array]:
    """Compute the output and intermediate hidden activations for one expert.

    For SwitchGLU: hidden = silu(x @ gate_proj[i].T) * (x @ up_proj[i].T)
                   output = hidden @ down_proj[i].T

    Args:
        switch_mlp: SwitchGLU module.
        x: Input tensor of shape (n_tokens, hidden_dim).
        expert_idx: Which expert to compute.

    Returns:
        (output, hidden) where output is (n_tokens, hidden_dim) and
        hidden is (n_tokens, intermediate_size).
    """
    gate_w = _get_expert_weight(switch_mlp.gate_proj, expert_idx)
    up_w = _get_expert_weight(switch_mlp.up_proj, expert_idx)
    down_w = _get_expert_weight(switch_mlp.down_proj, expert_idx)

    x_gate = x @ gate_w.T   # (n_tokens, intermediate)
    x_up = x @ up_w.T       # (n_tokens, intermediate)
    hidden = nn.silu(x_gate) * x_up  # (n_tokens, intermediate)
    output = hidden @ down_w.T       # (n_tokens, hidden_dim)

    return output, hidden


# ---------------------------------------------------------------------------
# Similarity computation
# ---------------------------------------------------------------------------

def _cosine_similarity_matrix_from_representations(
    representations: np.ndarray,
) -> np.ndarray:
    """Compute pairwise cosine similarity from per-expert mean representations.

    Args:
        representations: (n_experts, dim) array of mean representations.

    Returns:
        (n_experts, n_experts) cosine similarity matrix.
    """
    norms = np.linalg.norm(representations, axis=-1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    normalized = representations / norms
    return normalized @ normalized.T


def compute_similarity_matrix(
    switch_mlp,
    layer_input: np.ndarray,
    gate_logits: np.ndarray,
    n_experts: int,
    mode: str = "gated",
    max_tokens: int = 512,
) -> np.ndarray:
    """Compute pairwise expert similarity matrix.

    Two modes:
    - "gated": cos_sim of (gate_logit_i * output_i) vs (gate_logit_j * output_j)
    - "average": mean of cos_sim(output_i, output_j) and cos_sim(gate_logit_cols_i, gate_logit_cols_j)

    Args:
        switch_mlp: SwitchGLU module.
        layer_input: (n_tokens, hidden_dim) numpy array — MoE block input.
        gate_logits: (n_tokens, n_experts) numpy array — full gate logits.
        n_experts: Number of experts.
        mode: "gated" or "average".
        max_tokens: Max tokens to use (subsampled if layer_input has more).

    Returns:
        (n_experts, n_experts) similarity matrix.
    """
    n_tokens = layer_input.shape[0]
    if n_tokens > max_tokens:
        indices = np.random.choice(n_tokens, max_tokens, replace=False)
        layer_input = layer_input[indices]
        gate_logits = gate_logits[indices]
        n_tokens = max_tokens

    x_mx = mx.array(layer_input)

    if mode == "gated":
        # Compute gated outputs: gate_logit_i * expert_output_i
        # Accumulate mean representation per expert for memory efficiency
        gated_means = np.zeros((n_experts, layer_input.shape[1]), dtype=np.float64)
        for i in range(n_experts):
            output, _ = _compute_single_expert_output(switch_mlp, x_mx, i)
            mx.eval(output)
            out_np = _to_numpy(output).astype(np.float64)
            # gate_logits[:, i] is (n_tokens,) — broadcast multiply
            gated = gate_logits[:, i:i + 1].astype(np.float64) * out_np
            gated_means[i] = gated.mean(axis=0)

        return _cosine_similarity_matrix_from_representations(gated_means)

    elif mode == "average":
        # Output similarity
        output_means = np.zeros(
            (n_experts, layer_input.shape[1]), dtype=np.float64,
        )
        for i in range(n_experts):
            output, _ = _compute_single_expert_output(switch_mlp, x_mx, i)
            mx.eval(output)
            out_np = _to_numpy(output).astype(np.float64)
            output_means[i] = out_np.mean(axis=0)

        output_sim = _cosine_similarity_matrix_from_representations(output_means)

        # Gate logit similarity: each expert has a column of gate logits
        # (n_tokens, 1) per expert — use the full gate logit column as representation
        gate_sim = _cosine_similarity_matrix_from_representations(
            gate_logits.T.astype(np.float64)  # (n_experts, n_tokens)
        )

        return (output_sim + gate_sim) / 2.0

    else:
        raise ValueError(f"Unknown similarity mode: {mode!r}")


# ---------------------------------------------------------------------------
# Centroid selection and expert grouping
# ---------------------------------------------------------------------------

def select_centroids(scores: np.ndarray, n_keep: int) -> np.ndarray:
    """Select k centroid experts with highest saliency scores.

    Args:
        scores: (num_experts,) saliency scores for one layer.
        n_keep: Number of centroids (experts to keep).

    Returns:
        Sorted array of centroid expert indices.
    """
    return np.sort(np.argpartition(scores, -n_keep)[-n_keep:])


def group_experts(
    scores: np.ndarray,
    centroids: np.ndarray,
    similarity: np.ndarray,
    max_group_size: int = 16,
) -> Dict[int, List[int]]:
    """Assign non-centroid experts to centroid groups by similarity.

    Centroids are processed in descending saliency order. Each centroid
    claims up to max_group_size most similar unassigned experts.

    Args:
        scores: (num_experts,) saliency scores.
        centroids: Array of centroid expert indices.
        similarity: (num_experts, num_experts) similarity matrix.
        max_group_size: Maximum experts per group (C parameter).

    Returns:
        Dict mapping centroid_idx -> list of all member indices (including centroid).
    """
    n_experts = len(scores)
    centroid_set = set(centroids.tolist())
    assigned = set(centroid_set)

    # Process centroids by descending saliency
    centroid_order = sorted(centroids, key=lambda c: scores[c], reverse=True)

    groups: Dict[int, List[int]] = {c: [c] for c in centroid_order}

    for centroid in centroid_order:
        unassigned = [e for e in range(n_experts) if e not in assigned]
        if not unassigned:
            break

        # Rank unassigned experts by similarity to this centroid (descending)
        sims = [(e, similarity[centroid, e]) for e in unassigned]
        sims.sort(key=lambda t: t[1], reverse=True)

        # Take up to max_group_size - 1 (centroid already counted)
        n_to_take = min(max_group_size - 1, len(sims))
        for e, _ in sims[:n_to_take]:
            groups[centroid].append(e)
            assigned.add(e)

    # Assign any remaining unassigned experts to their most similar centroid
    remaining = [e for e in range(n_experts) if e not in assigned]
    for e in remaining:
        best_centroid = max(centroid_order, key=lambda c: similarity[c, e])
        groups[best_centroid].append(e)

    return groups


# ---------------------------------------------------------------------------
# Permutation alignment
# ---------------------------------------------------------------------------

def _cosine_distance_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine distance between rows of a and b.

    Args:
        a: (M, D) array — one row per neuron.
        b: (M, D) array — one row per neuron.

    Returns:
        (M, M) distance matrix where dist[i,j] = 1 - cos_sim(a[i], b[j]).
    """
    norm_a = np.linalg.norm(a, axis=1, keepdims=True)
    norm_b = np.linalg.norm(b, axis=1, keepdims=True)
    norm_a = np.maximum(norm_a, 1e-8)
    norm_b = np.maximum(norm_b, 1e-8)
    sim = (a / norm_a) @ (b / norm_b).T
    return 1.0 - sim


def _greedy_alignment(cost_matrix: np.ndarray) -> np.ndarray:
    """O(n^2) greedy matching: assign each row to its cheapest unmatched column.

    Args:
        cost_matrix: (n, n) cost matrix.

    Returns:
        col_ind array where col_ind[i] is the column assigned to row i.
    """
    n = cost_matrix.shape[0]
    col_ind = np.full(n, -1, dtype=np.intp)
    used_cols = set()

    # Process rows in order of their minimum cost (greediest first)
    row_order = np.argsort(cost_matrix.min(axis=1))

    for row in row_order:
        # Find cheapest available column for this row
        sorted_cols = np.argsort(cost_matrix[row])
        for col in sorted_cols:
            if col not in used_cols:
                col_ind[row] = col
                used_cols.add(col)
                break

    return col_ind


def _hungarian_alignment(cost_matrix: np.ndarray) -> np.ndarray:
    """Optimal assignment via Hungarian algorithm (requires scipy).

    Args:
        cost_matrix: (n, n) cost matrix.

    Returns:
        col_ind array where col_ind[i] is the column assigned to row i.
    """
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        raise ImportError(
            "scipy is required for Hungarian alignment. "
            "Install it with: pip install 'mlx-fun[ream]'"
        )

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    # linear_sum_assignment returns sorted row_ind; build full mapping
    result = np.arange(cost_matrix.shape[0])
    result[row_ind] = col_ind
    return result


def compute_alignment(
    centroid_hidden: np.ndarray,
    member_hidden: np.ndarray,
    centroid_weights: np.ndarray,
    member_weights: np.ndarray,
    method: str = "greedy",
) -> np.ndarray:
    """Compute neuron permutation to align member expert to centroid.

    Args:
        centroid_hidden: (n_tokens, intermediate_size) intermediate activations.
        member_hidden: (n_tokens, intermediate_size) intermediate activations.
        centroid_weights: (intermediate_size, weight_dim) concatenated weights.
        member_weights: (intermediate_size, weight_dim) concatenated weights.
        method: "greedy", "hungarian", or "none".

    Returns:
        Permutation array of length intermediate_size.
    """
    n = centroid_hidden.shape[1]

    if method == "none":
        return np.arange(n)

    # Cost = cosine_distance(hidden activations) + cosine_distance(weights)
    # hidden: each neuron's activation across tokens → (intermediate, n_tokens)
    cost_hidden = _cosine_distance_matrix(
        centroid_hidden.T.astype(np.float64),
        member_hidden.T.astype(np.float64),
    )
    cost_weights = _cosine_distance_matrix(
        centroid_weights.astype(np.float64),
        member_weights.astype(np.float64),
    )
    cost = cost_hidden + cost_weights

    if method == "hungarian":
        return _hungarian_alignment(cost)
    elif method == "greedy":
        return _greedy_alignment(cost)
    else:
        raise ValueError(f"Unknown alignment method: {method!r}")


# ---------------------------------------------------------------------------
# Group merging
# ---------------------------------------------------------------------------

def align_and_merge_group(
    adapter: BaseAdapter,
    layer_idx: int,
    centroid_idx: int,
    members: List[int],
    saliencies: np.ndarray,
    layer_input: mx.array,
    alignment_method: str = "greedy",
    max_alignment_tokens: int = 256,
) -> Tuple[mx.array, mx.array, mx.array]:
    """Align member experts to centroid and produce merged weights.

    Args:
        adapter: Model adapter.
        layer_idx: Decoder layer index.
        centroid_idx: Centroid expert index.
        members: All member indices including centroid.
        saliencies: (num_experts,) saliency scores for this layer.
        layer_input: (n_tokens, hidden_dim) input tensor.
        alignment_method: "greedy", "hungarian", or "none".
        max_alignment_tokens: Max tokens for alignment computation.

    Returns:
        (merged_gate_proj, merged_up_proj, merged_down_proj) weight tensors,
        each shaped for a single expert.
    """
    moe_block = adapter.get_moe_block(layer_idx)
    switch_mlp = adapter.get_switch_mlp(moe_block)

    # Subsample tokens for alignment
    n_tokens = layer_input.shape[0]
    if n_tokens > max_alignment_tokens:
        indices = mx.array(
            np.random.choice(n_tokens, max_alignment_tokens, replace=False)
        )
        x_align = layer_input[indices]
    else:
        x_align = layer_input

    # Get centroid weights and intermediate activations
    centroid_gate_w = _get_expert_weight(switch_mlp.gate_proj, centroid_idx)
    centroid_up_w = _get_expert_weight(switch_mlp.up_proj, centroid_idx)
    centroid_down_w = _get_expert_weight(switch_mlp.down_proj, centroid_idx)

    _, centroid_hidden = _compute_single_expert_output(
        switch_mlp, x_align, centroid_idx,
    )
    mx.eval(centroid_gate_w, centroid_up_w, centroid_down_w, centroid_hidden)

    centroid_hidden_np = _to_numpy(centroid_hidden).astype(np.float64)
    # Concatenate gate_proj and up_proj rows as the weight representation
    centroid_w_concat = np.concatenate([
        _to_numpy(centroid_gate_w).astype(np.float64),
        _to_numpy(centroid_up_w).astype(np.float64),
    ], axis=1)  # (intermediate, 2*hidden)

    # Compute saliency-normalized weights for this group
    group_saliencies = np.array([saliencies[e] for e in members], dtype=np.float64)
    sal_sum = group_saliencies.sum()
    if sal_sum < 1e-12:
        sal_norm = np.ones(len(members), dtype=np.float64) / len(members)
    else:
        sal_norm = group_saliencies / sal_sum

    # Initialize merged weights with centroid contribution
    centroid_pos = members.index(centroid_idx)
    merged_gate = _to_numpy(centroid_gate_w).astype(np.float64) * sal_norm[centroid_pos]
    merged_up = _to_numpy(centroid_up_w).astype(np.float64) * sal_norm[centroid_pos]
    merged_down = _to_numpy(centroid_down_w).astype(np.float64) * sal_norm[centroid_pos]

    # Process each non-centroid member
    for i, member_idx in enumerate(members):
        if member_idx == centroid_idx:
            continue

        member_gate_w = _get_expert_weight(switch_mlp.gate_proj, member_idx)
        member_up_w = _get_expert_weight(switch_mlp.up_proj, member_idx)
        member_down_w = _get_expert_weight(switch_mlp.down_proj, member_idx)

        _, member_hidden = _compute_single_expert_output(
            switch_mlp, x_align, member_idx,
        )
        mx.eval(member_gate_w, member_up_w, member_down_w, member_hidden)

        member_hidden_np = _to_numpy(member_hidden).astype(np.float64)
        member_w_concat = np.concatenate([
            _to_numpy(member_gate_w).astype(np.float64),
            _to_numpy(member_up_w).astype(np.float64),
        ], axis=1)

        # Compute permutation alignment
        perm = compute_alignment(
            centroid_hidden_np, member_hidden_np,
            centroid_w_concat, member_w_concat,
            method=alignment_method,
        )

        # Apply permutation and accumulate
        mg = _to_numpy(member_gate_w).astype(np.float64)[perm, :]
        mu = _to_numpy(member_up_w).astype(np.float64)[perm, :]
        md = _to_numpy(member_down_w).astype(np.float64)[:, perm]

        merged_gate += mg * sal_norm[i]
        merged_up += mu * sal_norm[i]
        merged_down += md * sal_norm[i]

    return (
        mx.array(merged_gate.astype(np.float32)),
        mx.array(merged_up.astype(np.float32)),
        mx.array(merged_down.astype(np.float32)),
    )


# ---------------------------------------------------------------------------
# Layer and model merging
# ---------------------------------------------------------------------------

def merge_moe_layer(
    adapter: BaseAdapter,
    layer_idx: int,
    centroids: np.ndarray,
    groups: Dict[int, List[int]],
    saliencies: np.ndarray,
    layer_input: mx.array,
    alignment_method: str = "greedy",
    max_alignment_tokens: int = 256,
) -> None:
    """Merge all expert groups in one MoE layer, updating weights in-place.

    After merging, the SwitchGLU has n_centroids experts and the gate
    weights are sliced to centroid rows only.

    Args:
        adapter: Model adapter.
        layer_idx: Decoder layer index.
        centroids: Sorted array of centroid expert indices.
        groups: {centroid_idx: [member_indices]} from group_experts().
        saliencies: (num_experts,) saliency scores for this layer.
        layer_input: (n_tokens, hidden_dim) input tensor.
        alignment_method: "greedy", "hungarian", or "none".
        max_alignment_tokens: Max tokens for alignment computation.
    """
    moe_block = adapter.get_moe_block(layer_idx)
    switch_mlp = adapter.get_switch_mlp(moe_block)

    n_centroids = len(centroids)
    # Collect merged weights for each centroid (in centroid order)
    all_gate_proj = []
    all_up_proj = []
    all_down_proj = []

    for centroid in centroids:
        members = groups[centroid]
        mg, mu, md = align_and_merge_group(
            adapter, layer_idx, centroid, members, saliencies,
            layer_input, alignment_method, max_alignment_tokens,
        )
        all_gate_proj.append(mg)
        all_up_proj.append(mu)
        all_down_proj.append(md)

    # Stack merged weights: (n_centroids, ...)
    new_gate_proj = mx.stack(all_gate_proj, axis=0)
    new_up_proj = mx.stack(all_up_proj, axis=0)
    new_down_proj = mx.stack(all_down_proj, axis=0)

    # Replace SwitchGLU projection weights
    switch_mlp.gate_proj.weight = new_gate_proj
    switch_mlp.up_proj.weight = new_up_proj
    switch_mlp.down_proj.weight = new_down_proj

    # Clear quantization params if they existed (merged weights are float)
    for proj in (switch_mlp.gate_proj, switch_mlp.up_proj, switch_mlp.down_proj):
        if hasattr(proj, "scales") and proj.scales is not None:
            proj.scales = None
        if hasattr(proj, "biases") and proj.biases is not None:
            proj.biases = None

    # Slice optional per-expert bias
    keep = mx.array(centroids)
    for proj in (switch_mlp.gate_proj, switch_mlp.up_proj, switch_mlp.down_proj):
        if "bias" in proj:
            proj.bias = mx.take(proj.bias, keep, axis=0)

    # Slice gate weights (model-type-specific, same as pruner.py)
    model_type = adapter.config.get("model_type", "")
    if model_type in ("minimax", "minimax_m2"):
        _slice_linear(moe_block.gate, keep)
        moe_block.e_score_correction_bias = mx.take(
            moe_block.e_score_correction_bias, keep, axis=0,
        )
    elif model_type in ("glm4_moe", "glm4_moe_lite", "glm_moe_dsa", "deepseek_v32"):
        moe_block.gate.weight = mx.take(moe_block.gate.weight, keep, axis=0)
        moe_block.gate.e_score_correction_bias = mx.take(
            moe_block.gate.e_score_correction_bias, keep, axis=0,
        )
        moe_block.gate.n_routed_experts = n_centroids
    elif model_type in ("qwen3_moe", "qwen3_next"):
        _slice_linear(moe_block.gate, keep)
        moe_block.num_experts = n_centroids


def merge_model(
    model,
    adapter: BaseAdapter,
    saliency_scores: np.ndarray,
    n_keep: int,
    calibration_samples: list,
    similarity_mode: str = "gated",
    alignment_method: str = "greedy",
    max_group_size: int = 16,
    max_similarity_tokens: int = 512,
    max_alignment_tokens: int = 256,
    progress_callback=None,
) -> Tuple[dict, Dict[int, np.ndarray], Dict[int, Dict[int, List[int]]]]:
    """Full REAM merge pipeline with sequential layer processing.

    For each MoE layer (in decoder order):
    1. Forward calibration data to capture layer input and gate logits
    2. Compute similarity matrix
    3. Select centroids and form groups
    4. Merge each group (align + weighted average)
    5. Update model weights in-place

    Args:
        model: The MLX model.
        adapter: Model adapter.
        saliency_scores: (num_layers, num_experts) from SaliencyAccumulator.
        n_keep: Target number of experts per layer.
        calibration_samples: List[mx.array] tokenized samples.
        similarity_mode: "gated" or "average".
        alignment_method: "greedy", "hungarian", or "none".
        max_group_size: Maximum experts per merge group (C).
        max_similarity_tokens: Max tokens for similarity computation.
        max_alignment_tokens: Max tokens for alignment computation.
        progress_callback: Optional callable(layer_num, total_layers) for progress.

    Returns:
        (updated_config, centroid_map, group_map) tuple.
    """
    moe_indices = adapter.moe_layer_indices()
    n_experts = adapter.num_routed_experts()
    top_k = adapter.num_experts_per_tok()
    model_type = adapter.config.get("model_type", "")

    if n_keep < top_k:
        raise ValueError(
            f"n_keep={n_keep} must be >= top_k={top_k}."
        )
    if n_keep >= n_experts:
        raise ValueError(
            f"n_keep={n_keep} must be < n_experts={n_experts}."
        )
    if n_keep == top_k:
        warnings.warn(
            f"n_keep={n_keep} equals top_k. All experts will always be selected."
        )

    centroid_map: Dict[int, np.ndarray] = {}
    group_map: Dict[int, Dict[int, List[int]]] = {}

    for acc_idx, layer_idx in enumerate(moe_indices):
        if progress_callback:
            progress_callback(acc_idx, len(moe_indices))

        layer_saliency = saliency_scores[acc_idx]

        # --- Step 1: Forward calibration data and capture layer input + gate logits ---
        moe_block = adapter.get_moe_block(layer_idx)
        install_ream_hooks([moe_block], model_type)

        for sample in calibration_samples:
            tokens = sample.reshape(1, -1)
            model(tokens)
            mx.eval(model.parameters())

        captures = collect_ream_data([moe_block])
        remove_ream_hooks([moe_block])

        # Concatenate all captured inputs and gate logits
        all_inputs = []
        all_gates = []
        for inp, gates in captures[0]:
            # Flatten batch and seq dims
            flat_inp = inp.reshape(-1, inp.shape[-1])
            flat_gates = gates.reshape(-1, gates.shape[-1])
            all_inputs.append(flat_inp)
            all_gates.append(flat_gates)

        layer_input_np = np.concatenate(all_inputs, axis=0)
        gate_logits_np = np.concatenate(all_gates, axis=0)

        # --- Step 2: Compute similarity matrix ---
        similarity = compute_similarity_matrix(
            adapter.get_switch_mlp(moe_block),
            layer_input_np,
            gate_logits_np,
            n_experts,
            mode=similarity_mode,
            max_tokens=max_similarity_tokens,
        )

        # --- Step 3: Select centroids and form groups ---
        centroids = select_centroids(layer_saliency, n_keep)
        groups = group_experts(
            layer_saliency, centroids, similarity, max_group_size,
        )

        centroid_map[layer_idx] = centroids
        group_map[layer_idx] = {int(c): members for c, members in groups.items()}

        # --- Step 4: Merge ---
        layer_input_mx = mx.array(layer_input_np)
        merge_moe_layer(
            adapter, layer_idx, centroids, groups, layer_saliency,
            layer_input_mx, alignment_method, max_alignment_tokens,
        )

    # Update config
    config = adapter.config.copy()
    config[adapter.config_expert_count_key()] = n_keep

    return config, centroid_map, group_map


def merge_model_with_keep_map(
    model,
    adapter: BaseAdapter,
    keep_map: Dict[int, np.ndarray],
    saliency_scores: np.ndarray,
    calibration_samples: list,
    similarity_mode: str = "gated",
    alignment_method: str = "greedy",
    max_group_size: int = 16,
    max_similarity_tokens: int = 512,
    max_alignment_tokens: int = 256,
    progress_callback=None,
) -> Tuple[dict, Dict[int, np.ndarray], Dict[int, Dict[int, List[int]]]]:
    """REAM merge pipeline with per-layer expert counts from keep_map.

    This variant supports model-wide pruning where each layer may have
    a different number of experts to keep.

    Args:
        model: The MLX model.
        adapter: Model adapter.
        keep_map: Dict mapping accumulator layer_idx -> array of expert indices to keep.
        saliency_scores: (num_layers, num_experts) from SaliencyAccumulator.
        calibration_samples: List[mx.array] tokenized samples.
        similarity_mode: "gated" or "average".
        alignment_method: "greedy", "hungarian", or "none".
        max_group_size: Maximum experts per merge group (C).
        max_similarity_tokens: Max tokens for similarity computation.
        max_alignment_tokens: Max tokens for alignment computation.
        progress_callback: Optional callable(layer_num, total_layers) for progress.

    Returns:
        (updated_config, centroid_map, group_map) tuple.
    """
    moe_indices = adapter.moe_layer_indices()
    n_experts = adapter.num_routed_experts()
    top_k = adapter.num_experts_per_tok()
    model_type = adapter.config.get("model_type", "")

    # Validate keep_map
    for acc_idx in range(len(moe_indices)):
        n_keep_layer = len(keep_map[acc_idx])
        if n_keep_layer < top_k:
            raise ValueError(
                f"Layer {acc_idx}: n_keep={n_keep_layer} must be >= top_k={top_k}."
            )

    centroid_map: Dict[int, np.ndarray] = {}
    group_map: Dict[int, Dict[int, List[int]]] = {}

    for acc_idx, layer_idx in enumerate(moe_indices):
        if progress_callback:
            progress_callback(acc_idx, len(moe_indices))

        layer_saliency = saliency_scores[acc_idx]
        n_keep_layer = len(keep_map[acc_idx])

        # Skip layers where all experts are kept (no merging needed)
        if n_keep_layer >= n_experts:
            centroid_map[layer_idx] = np.arange(n_experts)
            group_map[layer_idx] = {i: [i] for i in range(n_experts)}
            continue

        # --- Step 1: Forward calibration data and capture layer input + gate logits ---
        moe_block = adapter.get_moe_block(layer_idx)
        install_ream_hooks([moe_block], model_type)

        for sample in calibration_samples:
            tokens = sample.reshape(1, -1)
            model(tokens)
            mx.eval(model.parameters())

        captures = collect_ream_data([moe_block])
        remove_ream_hooks([moe_block])

        # Concatenate all captured inputs and gate logits
        all_inputs = []
        all_gates = []
        for inp, gates in captures[0]:
            # Flatten batch and seq dims
            flat_inp = inp.reshape(-1, inp.shape[-1])
            flat_gates = gates.reshape(-1, gates.shape[-1])
            all_inputs.append(flat_inp)
            all_gates.append(flat_gates)

        layer_input_np = np.concatenate(all_inputs, axis=0)
        gate_logits_np = np.concatenate(all_gates, axis=0)

        # --- Step 2: Compute similarity matrix ---
        similarity = compute_similarity_matrix(
            adapter.get_switch_mlp(moe_block),
            layer_input_np,
            gate_logits_np,
            n_experts,
            mode=similarity_mode,
            max_tokens=max_similarity_tokens,
        )

        # --- Step 3: Select centroids and form groups ---
        # Use the keep_map to determine which experts are centroids
        centroids = keep_map[acc_idx].copy()
        groups = group_experts(
            layer_saliency, centroids, similarity, max_group_size,
        )

        centroid_map[layer_idx] = centroids
        group_map[layer_idx] = {int(c): members for c, members in groups.items()}

        # --- Step 4: Merge ---
        layer_input_mx = mx.array(layer_input_np)
        merge_moe_layer(
            adapter, layer_idx, centroids, groups, layer_saliency,
            layer_input_mx, alignment_method, max_alignment_tokens,
        )

    # Determine the most common n_keep for config (or use max)
    n_keep_values = [len(keep_map[i]) for i in range(len(moe_indices))]
    n_keep_config = max(set(n_keep_values), key=n_keep_values.count)

    # Update config
    config = adapter.config.copy()
    config[adapter.config_expert_count_key()] = n_keep_config

    return config, centroid_map, group_map
