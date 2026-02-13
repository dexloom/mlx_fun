"""REAM-specific hooks for capturing MoE block inputs and full gate logits.

Unlike the REAP observer which captures top-k indices/scores/norms, REAM needs:
- The raw MoE block input x (for similarity and permutation alignment)
- Full gate logits for ALL experts (before top-k selection)

Uses the same __class__ swap pattern as observer.py.
"""

from typing import List, Tuple

import mlx.core as mx
import numpy as np

from .observer import _to_numpy


def _minimax_ream_call(self, x: mx.array) -> mx.array:
    """Capture input + full gate logits, then run normal MiniMax forward."""
    # Capture input and gate logits
    gates = self.gate(x.astype(mx.float32))
    mx.eval(x, gates)
    self._ream_captures.append((_to_numpy(x), _to_numpy(gates)))

    # Normal forward
    scores = mx.sigmoid(gates)
    orig_scores = scores
    scores = scores + self.e_score_correction_bias
    k = self.num_experts_per_tok
    inds = mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]
    scores = mx.take_along_axis(orig_scores, inds, axis=-1)
    scores = scores / (mx.sum(scores, axis=-1, keepdims=True) + 1e-20)
    scores = scores.astype(x.dtype)
    y = self.switch_mlp(x, inds)
    y = (y * scores[..., None]).sum(axis=-2)
    return y


def _glm4_ream_call(self, x: mx.array) -> mx.array:
    """Capture input + full gate logits, then run normal GLM4 forward."""
    if self.sharding_group is not None:
        raise RuntimeError(
            "Merging sharded models not supported. Load without sharding."
        )

    # Capture input and gate logits
    gates = x @ self.gate.weight.T
    mx.eval(x, gates)
    self._ream_captures.append((_to_numpy(x), _to_numpy(gates)))

    # Normal forward
    inds, scores = self.gate(x)
    y = self.switch_mlp(x, inds)
    y = (y * scores[..., None]).sum(axis=-2).astype(y.dtype)
    if hasattr(self, "shared_experts") and self.shared_experts is not None:
        y = y + self.shared_experts(x)
    return y


def _qwen3_moe_ream_call(self, x: mx.array) -> mx.array:
    """Capture input + full gate logits, then run normal Qwen3 forward."""
    # Capture input and raw gate logits (before softmax)
    gates_raw = self.gate(x)
    mx.eval(x, gates_raw)
    self._ream_captures.append((_to_numpy(x), _to_numpy(gates_raw)))

    # Normal forward
    gates = mx.softmax(gates_raw, axis=-1, precise=True)
    k = self.top_k
    inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
    scores = mx.take_along_axis(gates, inds, axis=-1)
    if self.norm_topk_prob:
        scores = scores / mx.sum(scores, axis=-1, keepdims=True)
    y = self.switch_mlp(x, inds)
    y = (y * scores[..., None]).sum(axis=-2)
    return y


def _qwen3_next_ream_call(self, x: mx.array) -> mx.array:
    """Capture input + full gate logits, then run normal Qwen3Next forward."""
    # Capture input and raw gate logits (before softmax)
    gates_raw = self.gate(x)
    mx.eval(x, gates_raw)
    self._ream_captures.append((_to_numpy(x), _to_numpy(gates_raw)))

    # Normal forward
    gates = mx.softmax(gates_raw, axis=-1, precise=True)
    k = self.top_k
    inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
    scores = mx.take_along_axis(gates, inds, axis=-1)
    if self.norm_topk_prob:
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
    "qwen3_moe": _qwen3_moe_ream_call,
    "qwen3_next": _qwen3_next_ream_call,
}


def install_ream_hooks(moe_blocks: List, model_type: str) -> None:
    """Install REAM capture hooks on MoE blocks.

    Args:
        moe_blocks: List of MoE nn.Module instances.
        model_type: Model type string (e.g. 'qwen3_moe').
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
    """Remove REAM hooks, restoring original class."""
    for block in moe_blocks:
        if hasattr(block, "_ream_original_cls"):
            block.__class__ = block._ream_original_cls
            delattr(block, "_ream_original_cls")
        if hasattr(block, "_ream_captures"):
            delattr(block, "_ream_captures")


def collect_ream_data(
    moe_blocks: List,
) -> List[List[Tuple[np.ndarray, np.ndarray]]]:
    """Collect and clear captured (input, gate_logits) from hooked blocks.

    Returns:
        List (per block) of lists of (layer_input, gate_logits) tuples.
        layer_input shape: (batch, seq, hidden_dim)
        gate_logits shape: (batch, seq, num_experts)
    """
    all_captures = []
    for block in moe_blocks:
        captures = getattr(block, "_ream_captures", [])
        all_captures.append(list(captures))
        block._ream_captures = []
    return all_captures
