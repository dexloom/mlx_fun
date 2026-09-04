"""REAM-specific hooks for capturing MoE block inputs and full gate logits.

Unlike the REAP observer which captures top-k indices/scores/norms, REAM needs:
- The raw MoE block input x (for similarity and permutation alignment)
- Full gate logits for ALL experts (before top-k selection)

Each capture also carries the **real** selected expert indices the block's own
routing produced, so callers that need selection frequency use the true
selection (grouped top-k, ``e_score_correction_bias``, per-expert scale and all)
rather than reconstructing it from the raw logits. The REAM merger ignores the
third element; the safety and domain scans use it.

Uses the same __class__ swap pattern as observer.py.
"""

from typing import List, Tuple

import mlx.core as mx
import numpy as np

from .observer import _to_numpy


def _minimax_ream_call(self, x: mx.array) -> mx.array:
    """Capture input, full gate logits and real selection; then normal forward."""
    gates = self.gate(x.astype(mx.float32))

    scores = mx.sigmoid(gates)
    orig_scores = scores
    # Real selection is on the correction-bias-adjusted score, not raw sigmoid.
    scores = scores + self.e_score_correction_bias
    k = self.num_experts_per_tok
    inds = mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]

    mx.eval(x, gates, inds)
    self._ream_captures.append((_to_numpy(x), _to_numpy(gates), _to_numpy(inds)))

    scores = mx.take_along_axis(orig_scores, inds, axis=-1)
    scores = scores / (mx.sum(scores, axis=-1, keepdims=True) + 1e-20)
    scores = scores.astype(x.dtype)
    y = self.switch_mlp(x, inds)
    y = (y * scores[..., None]).sum(axis=-2)
    return y


def _glm4_ream_call(self, x: mx.array) -> mx.array:
    """Capture input, full gate logits and real selection; then normal forward."""
    if getattr(self, "sharding_group", None) is not None:
        raise RuntimeError(
            "Merging sharded models not supported. Load without sharding."
        )

    gates = x @ self.gate.weight.T
    # The gate applies grouped top-k selection and the correction bias — capture
    # what it actually selects, not a raw-sigmoid reconstruction.
    inds, scores = self.gate(x)

    mx.eval(x, gates, inds)
    self._ream_captures.append((_to_numpy(x), _to_numpy(gates), _to_numpy(inds)))

    # Latent projection (Nemotron-H): hidden → moe_latent_size before experts
    x_experts = x
    if hasattr(self, "fc1_latent_proj"):
        x_experts = self.fc1_latent_proj(x)
    y = self.switch_mlp(x_experts, inds)
    y = (y * scores[..., None]).sum(axis=-2).astype(y.dtype)
    if hasattr(self, "fc2_latent_proj"):
        y = self.fc2_latent_proj(y)
    if hasattr(self, "shared_experts") and self.shared_experts is not None:
        y = y + self.shared_experts(x)
    return y


def _qwen3_moe_ream_call(self, x: mx.array) -> mx.array:
    """Capture input, full gate logits and real selection; then normal forward."""
    gates_raw = self.gate(x)

    gates = mx.softmax(gates_raw, axis=-1, precise=True)
    k = self.top_k
    inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]

    mx.eval(x, gates_raw, inds)
    self._ream_captures.append((_to_numpy(x), _to_numpy(gates_raw), _to_numpy(inds)))

    scores = mx.take_along_axis(gates, inds, axis=-1)
    if self.norm_topk_prob:
        scores = scores / mx.sum(scores, axis=-1, keepdims=True)
    y = self.switch_mlp(x, inds)
    y = (y * scores[..., None]).sum(axis=-2)
    return y


def _qwen3_next_ream_call(self, x: mx.array) -> mx.array:
    """Capture input, full gate logits and real selection; then normal forward."""
    gates_raw = self.gate(x)

    gates = mx.softmax(gates_raw, axis=-1, precise=True)
    k = self.top_k
    inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]

    mx.eval(x, gates_raw, inds)
    self._ream_captures.append((_to_numpy(x), _to_numpy(gates_raw), _to_numpy(inds)))

    scores = mx.take_along_axis(gates, inds, axis=-1)
    if self.norm_topk_prob:
        scores = scores / mx.sum(scores, axis=-1, keepdims=True)
    y = self.switch_mlp(x, inds)
    y = (y * scores[..., None]).sum(axis=-2)

    # Shared expert (always active, sigmoid-gated)
    shared_y = self.shared_expert(x)
    shared_y = mx.sigmoid(self.shared_expert_gate(x)) * shared_y
    return y + shared_y


def _gemma4_ream_call(self, h: mx.array) -> mx.array:
    """Capture input, full gate logits and real selection; then normal forward."""
    router = self.router

    x_normed = mx.fast.rms_norm(h, router.scale * router._root_size, router.eps)
    gates_raw = router.proj(x_normed)
    # Real selection from the router (softmax + per-expert scale + top-k).
    top_k_indices, top_k_weights = self.router(h)

    mx.eval(h, gates_raw, top_k_indices)
    self._ream_captures.append(
        (_to_numpy(h), _to_numpy(gates_raw), _to_numpy(top_k_indices))
    )

    h2 = self.pre_feedforward_layernorm_2(h)
    return self.experts(h2, top_k_indices, top_k_weights)


def _qwen4_exp_ream_call(self, x: mx.array) -> mx.array:
    """Capture input, full gate logits and real selection; then normal forward."""
    gates_raw = self.gate(x)

    gates = mx.softmax(gates_raw, axis=-1, precise=True)
    k = self.top_k
    inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]

    mx.eval(x, gates_raw, inds)
    self._ream_captures.append((_to_numpy(x), _to_numpy(gates_raw), _to_numpy(inds)))

    scores = mx.take_along_axis(gates, inds, axis=-1)
    scores = scores / mx.sum(scores, axis=-1, keepdims=True)
    y = self.switch_mlp(x, inds)
    y = (y * scores[..., None]).sum(axis=-2)

    # Shared expert (always active, sigmoid-gated)
    shared_y = self.shared_expert(x)
    shared_y = mx.sigmoid(self.shared_expert_gate(x)) * shared_y
    return y + shared_y


_REAM_HOOK_MAP = {
    "minimax": _minimax_ream_call,
    "minimax_m2": _minimax_ream_call,
    "glm4_moe": _glm4_ream_call,
    "glm4_moe_lite": _glm4_ream_call,
    "glm_moe_dsa": _glm4_ream_call,
    "deepseek_v32": _glm4_ream_call,
    "nemotron_h": _glm4_ream_call,
    "glm5_next": _glm4_ream_call,
    "qwen3_moe": _qwen3_moe_ream_call,
    "qwen3_next": _qwen3_next_ream_call,
    "gemma4": _gemma4_ream_call,
    "qwen4_exp": _qwen4_exp_ream_call,
}


def install_ream_hooks(moe_blocks: List, model_type: str) -> None:
    """Install REAM capture hooks on a list of MoE blocks.

    Args:
        moe_blocks: List of MoE nn.Module instances.
        model_type: Model type string.
    """
    hook_fn = _REAM_HOOK_MAP.get(model_type)
    if hook_fn is None:
        raise ValueError(f"No REAM hook for model_type '{model_type}'")

    for block in moe_blocks:
        block._ream_captures = []
        original_cls = type(block)
        block._ream_original_cls = original_cls
        hooked_cls = type(
            f"_ReamHooked_{original_cls.__name__}",
            (original_cls,),
            {"__call__": hook_fn},
        )
        block.__class__ = hooked_cls


def remove_ream_hooks(moe_blocks: List) -> None:
    """Remove REAM capture hooks, restoring the original class."""
    for block in moe_blocks:
        if hasattr(block, "_ream_original_cls"):
            block.__class__ = block._ream_original_cls
            delattr(block, "_ream_original_cls")
        if hasattr(block, "_ream_captures"):
            delattr(block, "_ream_captures")


def collect_ream_data(
    moe_blocks: List,
) -> List[List[Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """Collect and clear captured REAM data.

    Returns:
        List (per block) of lists of ``(layer_input, gate_logits, selected_inds)``
        tuples. ``selected_inds`` is the real top-k selection the block's routing
        produced, shaped ``(..., top_k)``.
    """
    all_captures = []
    for block in moe_blocks:
        captures = getattr(block, "_ream_captures", [])
        all_captures.append(list(captures))
        block._ream_captures = []
    return all_captures
