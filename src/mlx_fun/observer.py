"""Monkey-patch hooks for MoE calibration.

MLX has no register_forward_hook(), so we swap __class__ on instances
to capture routing decisions and activation norms during forward passes.
Python resolves special methods (__call__) on the type, not the instance,
so we dynamically create a subclass with the hooked __call__.
"""

from typing import List, Tuple

import mlx.core as mx
import numpy as np


def _to_numpy(arr: mx.array) -> np.ndarray:
    """Convert MLX array to numpy, casting bf16 to float32 first."""
    if arr.dtype == mx.bfloat16:
        arr = arr.astype(mx.float32)
    return np.array(arr, copy=False)


def _minimax_hooked_call(self, x: mx.array) -> mx.array:
    """Replacement __call__ for MiniMaxSparseMoeBlock that captures metrics."""
    gates = self.gate(x.astype(mx.float32))
    scores = mx.sigmoid(gates)
    orig_scores = scores
    scores = scores + self.e_score_correction_bias

    k = self.num_experts_per_tok
    inds = mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]
    scores = mx.take_along_axis(orig_scores, inds, axis=-1)
    scores = scores / (mx.sum(scores, axis=-1, keepdims=True) + 1e-20)
    scores = scores.astype(x.dtype)

    y = self.switch_mlp(x, inds)
    # y shape: (batch, seq, top_k, hidden)
    activation_norms = mx.linalg.norm(y, axis=-1)

    # Materialize and capture
    mx.eval(inds, scores, activation_norms)
    self._reap_captures.append((
        _to_numpy(inds),
        _to_numpy(scores),
        _to_numpy(activation_norms),
    ))

    y = (y * scores[..., None]).sum(axis=-2)
    return y


def _glm4_hooked_call(self, x: mx.array) -> mx.array:
    """Replacement __call__ for GLM4 MoE block that captures metrics."""
    if self.sharding_group is not None:
        raise RuntimeError(
            "Pruning sharded models not supported in v1. Load without sharding."
        )

    inds, scores = self.gate(x)
    y = self.switch_mlp(x, inds)
    # y shape: (batch, seq, top_k, hidden)
    activation_norms = mx.linalg.norm(y, axis=-1)

    # Materialize and capture
    mx.eval(inds, scores, activation_norms)
    self._reap_captures.append((
        _to_numpy(inds),
        _to_numpy(scores),
        _to_numpy(activation_norms),
    ))

    y = (y * scores[..., None]).sum(axis=-2).astype(y.dtype)
    if hasattr(self, "shared_experts") and self.shared_experts is not None:
        y = y + self.shared_experts(x)

    return y


def _qwen3_moe_hooked_call(self, x: mx.array) -> mx.array:
    """Replacement __call__ for Qwen3MoeSparseMoeBlock that captures metrics."""
    gates = self.gate(x)
    gates = mx.softmax(gates, axis=-1, precise=True)

    k = self.top_k
    inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
    scores = mx.take_along_axis(gates, inds, axis=-1)
    if self.norm_topk_prob:
        scores = scores / mx.sum(scores, axis=-1, keepdims=True)

    y = self.switch_mlp(x, inds)
    # y shape: (batch, seq, top_k, hidden)
    activation_norms = mx.linalg.norm(y, axis=-1)

    # Materialize and capture
    mx.eval(inds, scores, activation_norms)
    self._reap_captures.append((
        _to_numpy(inds),
        _to_numpy(scores),
        _to_numpy(activation_norms),
    ))

    y = (y * scores[..., None]).sum(axis=-2)
    return y


def _qwen3_next_hooked_call(self, x: mx.array) -> mx.array:
    """Replacement __call__ for Qwen3NextSparseMoeBlock that captures metrics."""
    gates = self.gate(x)
    gates = mx.softmax(gates, axis=-1, precise=True)

    k = self.top_k
    inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
    scores = mx.take_along_axis(gates, inds, axis=-1)
    if self.norm_topk_prob:
        scores = scores / mx.sum(scores, axis=-1, keepdims=True)

    y = self.switch_mlp(x, inds)
    # y shape: (batch, seq, top_k, hidden)
    activation_norms = mx.linalg.norm(y, axis=-1)

    # Materialize and capture
    mx.eval(inds, scores, activation_norms)
    self._reap_captures.append((
        _to_numpy(inds),
        _to_numpy(scores),
        _to_numpy(activation_norms),
    ))

    y = (y * scores[..., None]).sum(axis=-2)

    # Shared expert (always active, sigmoid-gated)
    shared_y = self.shared_expert(x)
    shared_y = mx.sigmoid(self.shared_expert_gate(x)) * shared_y

    return y + shared_y


_HOOK_MAP = {
    "minimax": _minimax_hooked_call,
    "minimax_m2": _minimax_hooked_call,
    "glm4_moe": _glm4_hooked_call,
    "glm4_moe_lite": _glm4_hooked_call,
    "glm_moe_dsa": _glm4_hooked_call,
    "deepseek_v32": _glm4_hooked_call,
    "qwen3_moe": _qwen3_moe_hooked_call,
    "qwen3_next": _qwen3_next_hooked_call,
}


def install_hooks(
    moe_blocks: List,
    model_type: str,
) -> None:
    """Install capture hooks on a list of MoE blocks.

    Uses __class__ swapping so Python's special method lookup finds our hook.

    Args:
        moe_blocks: List of MoE nn.Module instances.
        model_type: 'minimax', 'glm4_moe', or 'qwen3_moe'.
    """
    hook_fn = _HOOK_MAP.get(model_type)
    if hook_fn is None:
        raise ValueError(f"No hook for model_type '{model_type}'")

    for block in moe_blocks:
        block._reap_captures = []
        original_cls = type(block)
        block._reap_original_cls = original_cls
        hooked_cls = type(
            f"_Hooked_{original_cls.__name__}",
            (original_cls,),
            {"__call__": hook_fn},
        )
        block.__class__ = hooked_cls


def remove_hooks(moe_blocks: List) -> None:
    """Remove capture hooks, restoring original class."""
    for block in moe_blocks:
        if hasattr(block, "_reap_original_cls"):
            block.__class__ = block._reap_original_cls
            delattr(block, "_reap_original_cls")
        if hasattr(block, "_reap_captures"):
            delattr(block, "_reap_captures")


def collect_captures(moe_blocks: List) -> List[List[Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """Collect and clear all captured data from hooked blocks.

    Returns:
        List (per block) of lists of (inds, scores, norms) tuples.
    """
    all_captures = []
    for block in moe_blocks:
        captures = getattr(block, "_reap_captures", [])
        all_captures.append(list(captures))
        block._reap_captures = []
    return all_captures
