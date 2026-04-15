"""LLM server with online expert counting and multi-API support.

Composes on top of mlx-lm's server infrastructure. Subclasses APIHandler to add
REAP management endpoints and an Anthropic Messages API endpoint (/v1/messages).
Installs lightweight hooks that accumulate expert statistics into a thread-safe
OnlineAccumulator during every forward pass.

Both OpenAI (/v1/chat/completions) and Anthropic (/v1/messages) APIs share the
same generation pipeline — jinja templates always receive OpenAI-style messages.
"""

import argparse
import gc
import io
import json
import logging
import os
import re
import signal
import threading
import time
import uuid
import warnings
from pathlib import Path
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .observer import _to_numpy
from .saliency import SaliencyAccumulator


# ---------------------------------------------------------------------------
# Thread-safe accumulator
# ---------------------------------------------------------------------------

class OnlineAccumulator:
    """Thread-safe wrapper around SaliencyAccumulator."""

    def __init__(self, num_layers: int, num_experts: int):
        self._lock = threading.Lock()
        self._acc = SaliencyAccumulator(num_layers, num_experts)
        self._request_count = 0
        self._token_count = 0
        self.num_layers = num_layers
        self.num_experts = num_experts

    def update(
        self,
        layer_idx: int,
        expert_indices: np.ndarray,
        router_weights: np.ndarray,
        activation_norms: Optional[np.ndarray] = None,
    ):
        """Accumulate statistics for one forward pass at one layer.

        When activation_norms is None (lightweight mode), zeros are used for
        reap_sum/ean_sum fields — only freq/weighted_freq are meaningful.
        """
        if activation_norms is None:
            activation_norms = np.zeros_like(router_weights)
        with self._lock:
            self._acc.update(layer_idx, expert_indices, router_weights, activation_norms)

    def increment_request(self):
        with self._lock:
            self._request_count += 1

    def add_tokens(self, n: int):
        with self._lock:
            self._token_count += n

    def get_stats(self) -> dict:
        """Return current accumulator state as JSON-serializable dict.
        
        Includes both raw accumulator arrays and computed scores for easy
        comparison with stats-diff, stats-merge, and stats-purge operations.
        """
        with self._lock:
            # Compute scores for all metrics
            reap_scores = self._acc.compute_scores("reap").tolist()
            ean_scores = self._acc.compute_scores("ean").tolist()
            freq_scores = self._acc.compute_scores("freq").tolist()
            weighted_freq_scores = self._acc.compute_scores("weighted_freq").tolist()
            
            # Total samples (important for normalized merge mode)
            total_samples = float(self._acc.freq.sum())
            
            return {
                "freq": self._acc.freq.tolist(),
                "weighted_freq_sum": self._acc.weighted_freq_sum.tolist(),
                "reap_sum": self._acc.reap_sum.tolist(),
                "ean_sum": self._acc.ean_sum.tolist(),
                "reap_count": self._acc.reap_count.tolist(),
                "num_layers": self._acc.num_layers,
                "num_experts": self._acc.num_experts,
                "request_count": self._request_count,
                "token_count": self._token_count,
                "total_samples": total_samples,
                "computed_scores": {
                    "reap": reap_scores,
                    "ean": ean_scores,
                    "freq": freq_scores,
                    "weighted_freq": weighted_freq_scores,
                },
            }

    def save(self, path: str):
        """Save accumulator state to .npz (compatible with SaliencyAccumulator.load)."""
        with self._lock:
            self._acc.save(path)

    def reset(self):
        """Reset all counters to zero."""
        with self._lock:
            n_layers = self._acc.num_layers
            n_experts = self._acc.num_experts
            self._acc = SaliencyAccumulator(n_layers, n_experts)
            self._request_count = 0
            self._token_count = 0


# ---------------------------------------------------------------------------
# Lightweight counting hooks (skip activation norm computation)
# ---------------------------------------------------------------------------

def _minimax_counting_call(self, x: mx.array) -> mx.array:
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

    # Materialize routing decisions and accumulate
    mx.eval(inds, scores)
    np_inds = _to_numpy(inds).reshape(-1, inds.shape[-1])
    np_scores = _to_numpy(scores).reshape(-1, scores.shape[-1])
    self._reap_accumulator.update(self._reap_layer_idx, np_inds, np_scores)

    y = (y * scores[..., None]).sum(axis=-2)
    return y


def _glm4_counting_call(self, x: mx.array) -> mx.array:
    if getattr(self, "sharding_group", None) is not None:
        raise RuntimeError(
            "Pruning sharded models not supported in v1. Load without sharding."
        )

    inds, scores = self.gate(x)
    # Latent projection (Nemotron-H): 4096 → moe_latent_size before experts
    x_experts = x
    if hasattr(self, "fc1_latent_proj"):
        x_experts = self.fc1_latent_proj(x)
    y = self.switch_mlp(x_experts, inds)

    mx.eval(inds, scores)
    np_inds = _to_numpy(inds).reshape(-1, inds.shape[-1])
    np_scores = _to_numpy(scores).reshape(-1, scores.shape[-1])
    self._reap_accumulator.update(self._reap_layer_idx, np_inds, np_scores)

    y = (y * scores[..., None]).sum(axis=-2).astype(y.dtype)
    # Latent back-projection: moe_latent_size → 4096
    if hasattr(self, "fc2_latent_proj"):
        y = self.fc2_latent_proj(y)
    if hasattr(self, "shared_experts") and self.shared_experts is not None:
        y = y + self.shared_experts(x)

    return y


def _qwen3_moe_counting_call(self, x: mx.array) -> mx.array:
    gates = self.gate(x)
    gates = mx.softmax(gates, axis=-1, precise=True)

    k = self.top_k
    inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
    scores = mx.take_along_axis(gates, inds, axis=-1)
    if self.norm_topk_prob:
        scores = scores / mx.sum(scores, axis=-1, keepdims=True)

    y = self.switch_mlp(x, inds)

    mx.eval(inds, scores)
    np_inds = _to_numpy(inds).reshape(-1, inds.shape[-1])
    np_scores = _to_numpy(scores).reshape(-1, scores.shape[-1])
    self._reap_accumulator.update(self._reap_layer_idx, np_inds, np_scores)

    y = (y * scores[..., None]).sum(axis=-2)
    return y


def _qwen3_next_counting_call(self, x: mx.array) -> mx.array:
    gates = self.gate(x)
    gates = mx.softmax(gates, axis=-1, precise=True)

    k = self.top_k
    inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
    scores = mx.take_along_axis(gates, inds, axis=-1)
    if self.norm_topk_prob:
        scores = scores / mx.sum(scores, axis=-1, keepdims=True)

    y = self.switch_mlp(x, inds)

    mx.eval(inds, scores)
    np_inds = _to_numpy(inds).reshape(-1, inds.shape[-1])
    np_scores = _to_numpy(scores).reshape(-1, scores.shape[-1])
    self._reap_accumulator.update(self._reap_layer_idx, np_inds, np_scores)

    y = (y * scores[..., None]).sum(axis=-2)

    shared_y = self.shared_expert(x)
    shared_y = mx.sigmoid(self.shared_expert_gate(x)) * shared_y

    return y + shared_y


# ---------------------------------------------------------------------------
# Full counting hooks (with activation norms — same routing as observer.py
# but accumulate directly instead of appending to list)
# ---------------------------------------------------------------------------

def _minimax_full_counting_call(self, x: mx.array) -> mx.array:
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
    activation_norms = mx.linalg.norm(y, axis=-1)

    mx.eval(inds, scores, activation_norms)
    np_inds = _to_numpy(inds).reshape(-1, inds.shape[-1])
    np_scores = _to_numpy(scores).reshape(-1, scores.shape[-1])
    np_norms = _to_numpy(activation_norms).reshape(-1, activation_norms.shape[-1])
    self._reap_accumulator.update(self._reap_layer_idx, np_inds, np_scores, np_norms)

    y = (y * scores[..., None]).sum(axis=-2)
    return y


def _glm4_full_counting_call(self, x: mx.array) -> mx.array:
    if getattr(self, "sharding_group", None) is not None:
        raise RuntimeError(
            "Pruning sharded models not supported in v1. Load without sharding."
        )

    inds, scores = self.gate(x)
    x_experts = x
    if hasattr(self, "fc1_latent_proj"):
        x_experts = self.fc1_latent_proj(x)
    y = self.switch_mlp(x_experts, inds)
    activation_norms = mx.linalg.norm(y, axis=-1)

    mx.eval(inds, scores, activation_norms)
    np_inds = _to_numpy(inds).reshape(-1, inds.shape[-1])
    np_scores = _to_numpy(scores).reshape(-1, scores.shape[-1])
    np_norms = _to_numpy(activation_norms).reshape(-1, activation_norms.shape[-1])
    self._reap_accumulator.update(self._reap_layer_idx, np_inds, np_scores, np_norms)

    y = (y * scores[..., None]).sum(axis=-2).astype(y.dtype)
    if hasattr(self, "fc2_latent_proj"):
        y = self.fc2_latent_proj(y)
    if hasattr(self, "shared_experts") and self.shared_experts is not None:
        y = y + self.shared_experts(x)

    return y


def _qwen3_moe_full_counting_call(self, x: mx.array) -> mx.array:
    gates = self.gate(x)
    gates = mx.softmax(gates, axis=-1, precise=True)

    k = self.top_k
    inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
    scores = mx.take_along_axis(gates, inds, axis=-1)
    if self.norm_topk_prob:
        scores = scores / mx.sum(scores, axis=-1, keepdims=True)

    y = self.switch_mlp(x, inds)
    activation_norms = mx.linalg.norm(y, axis=-1)

    mx.eval(inds, scores, activation_norms)
    np_inds = _to_numpy(inds).reshape(-1, inds.shape[-1])
    np_scores = _to_numpy(scores).reshape(-1, scores.shape[-1])
    np_norms = _to_numpy(activation_norms).reshape(-1, activation_norms.shape[-1])
    self._reap_accumulator.update(self._reap_layer_idx, np_inds, np_scores, np_norms)

    y = (y * scores[..., None]).sum(axis=-2)
    return y


def _qwen3_next_full_counting_call(self, x: mx.array) -> mx.array:
    gates = self.gate(x)
    gates = mx.softmax(gates, axis=-1, precise=True)

    k = self.top_k
    inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
    scores = mx.take_along_axis(gates, inds, axis=-1)
    if self.norm_topk_prob:
        scores = scores / mx.sum(scores, axis=-1, keepdims=True)

    y = self.switch_mlp(x, inds)
    activation_norms = mx.linalg.norm(y, axis=-1)

    mx.eval(inds, scores, activation_norms)
    np_inds = _to_numpy(inds).reshape(-1, inds.shape[-1])
    np_scores = _to_numpy(scores).reshape(-1, scores.shape[-1])
    np_norms = _to_numpy(activation_norms).reshape(-1, activation_norms.shape[-1])
    self._reap_accumulator.update(self._reap_layer_idx, np_inds, np_scores, np_norms)

    y = (y * scores[..., None]).sum(axis=-2)

    shared_y = self.shared_expert(x)
    shared_y = mx.sigmoid(self.shared_expert_gate(x)) * shared_y

    return y + shared_y


def _gemma4_counting_call(self, h: mx.array) -> mx.array:
    top_k_indices, top_k_weights = self.router(h)
    h2 = self.pre_feedforward_layernorm_2(h)
    result = self.experts(h2, top_k_indices, top_k_weights)

    mx.eval(top_k_indices, top_k_weights)
    np_inds = _to_numpy(top_k_indices).reshape(-1, top_k_indices.shape[-1])
    np_scores = _to_numpy(top_k_weights).reshape(-1, top_k_weights.shape[-1])
    self._reap_accumulator.update(self._reap_layer_idx, np_inds, np_scores)

    return result


_COUNTING_HOOK_MAP = {
    "minimax": _minimax_counting_call,
    "minimax_m2": _minimax_counting_call,
    "glm4_moe": _glm4_counting_call,
    "glm4_moe_lite": _glm4_counting_call,
    "glm_moe_dsa": _glm4_counting_call,
    "deepseek_v32": _glm4_counting_call,
    "nemotron_h": _glm4_counting_call,
    "qwen3_moe": _qwen3_moe_counting_call,
    "qwen3_next": _qwen3_next_counting_call,
    "gemma4": _gemma4_counting_call,
}

def _gemma4_full_counting_call(self, h: mx.array) -> mx.array:
    router = self.router
    x_normed = mx.fast.rms_norm(h, router.scale * router._root_size, router.eps)
    expert_scores = router.proj(x_normed)
    router_probs = mx.softmax(expert_scores, axis=-1)
    k = router.config.top_k_experts
    inds = mx.argpartition(-expert_scores, kth=k - 1, axis=-1)[..., :k]
    scores = mx.take_along_axis(router_probs, inds, axis=-1)
    scores = scores / mx.sum(scores, axis=-1, keepdims=True)
    scores = scores * router.per_expert_scale[inds]

    h2 = self.pre_feedforward_layernorm_2(h)
    B, S, H = h2.shape
    x_flat = h2.reshape(B * S, H)
    indices_flat = inds.reshape(B * S, k)
    expert_out = self.switch_glu(x_flat, indices_flat)
    activation_norms = mx.linalg.norm(expert_out, axis=-1)

    mx.eval(inds, scores, activation_norms)
    np_inds = _to_numpy(inds).reshape(-1, inds.shape[-1])
    np_scores = _to_numpy(scores).reshape(-1, scores.shape[-1])
    np_norms = _to_numpy(activation_norms).reshape(-1, activation_norms.shape[-1])
    self._reap_accumulator.update(self._reap_layer_idx, np_inds, np_scores, np_norms)

    weights = scores.reshape(B * S, k)[..., None]
    return (expert_out * weights).sum(axis=-2).reshape(B, S, H)


_FULL_COUNTING_HOOK_MAP = {
    "minimax": _minimax_full_counting_call,
    "minimax_m2": _minimax_full_counting_call,
    "glm4_moe": _glm4_full_counting_call,
    "glm4_moe_lite": _glm4_full_counting_call,
    "glm_moe_dsa": _glm4_full_counting_call,
    "deepseek_v32": _glm4_full_counting_call,
    "nemotron_h": _glm4_full_counting_call,
    "qwen3_moe": _qwen3_moe_full_counting_call,
    "qwen3_next": _qwen3_next_full_counting_call,
    "gemma4": _gemma4_full_counting_call,
}


# ---------------------------------------------------------------------------
# Compound counting + steering hooks (counting with gate logit bias injection)
# ---------------------------------------------------------------------------

def _minimax_counting_steering_call(self, x: mx.array) -> mx.array:
    gates = self.gate(x.astype(mx.float32))
    if self._steering_bias is not None:
        gates = gates + self._steering_bias
    scores = mx.sigmoid(gates)
    orig_scores = scores
    scores = scores + self.e_score_correction_bias

    k = self.num_experts_per_tok
    inds = mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]
    scores = mx.take_along_axis(orig_scores, inds, axis=-1)
    scores = scores / (mx.sum(scores, axis=-1, keepdims=True) + 1e-20)
    scores = scores.astype(x.dtype)

    y = self.switch_mlp(x, inds)

    mx.eval(inds, scores)
    np_inds = _to_numpy(inds).reshape(-1, inds.shape[-1])
    np_scores = _to_numpy(scores).reshape(-1, scores.shape[-1])
    self._reap_accumulator.update(self._reap_layer_idx, np_inds, np_scores)

    y = (y * scores[..., None]).sum(axis=-2)
    return y


def _glm4_counting_steering_call(self, x: mx.array) -> mx.array:
    if getattr(self, "sharding_group", None) is not None:
        raise RuntimeError(
            "Sharded models not supported. Load without sharding."
        )

    # Inline the gate forward to inject steering bias before routing
    raw_gates = x @ self.gate.weight.T
    if self._steering_bias is not None:
        raw_gates = raw_gates + self._steering_bias

    scores = mx.sigmoid(raw_gates.astype(mx.float32))
    orig_scores = scores
    scores = scores + self.gate.e_score_correction_bias
    k = self.gate.top_k
    inds = mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]
    scores = mx.take_along_axis(orig_scores, inds, axis=-1)
    scores = scores / (mx.sum(scores, axis=-1, keepdims=True) + 1e-20)

    # Latent projection (Nemotron-H): 4096 → moe_latent_size before experts
    x_experts = x
    if hasattr(self, "fc1_latent_proj"):
        x_experts = self.fc1_latent_proj(x)
    y = self.switch_mlp(x_experts, inds)

    mx.eval(inds, scores)
    np_inds = _to_numpy(inds).reshape(-1, inds.shape[-1])
    np_scores = _to_numpy(scores).reshape(-1, scores.shape[-1])
    self._reap_accumulator.update(self._reap_layer_idx, np_inds, np_scores)

    y = (y * scores[..., None]).sum(axis=-2).astype(y.dtype)
    # Latent back-projection: moe_latent_size → 4096
    if hasattr(self, "fc2_latent_proj"):
        y = self.fc2_latent_proj(y)
    if hasattr(self, "shared_experts") and self.shared_experts is not None:
        y = y + self.shared_experts(x)

    return y


def _qwen3_moe_counting_steering_call(self, x: mx.array) -> mx.array:
    gates = self.gate(x)
    if self._steering_bias is not None:
        gates = gates + self._steering_bias
    gates = mx.softmax(gates, axis=-1, precise=True)

    k = self.top_k
    inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
    scores = mx.take_along_axis(gates, inds, axis=-1)
    if self.norm_topk_prob:
        scores = scores / mx.sum(scores, axis=-1, keepdims=True)

    y = self.switch_mlp(x, inds)

    mx.eval(inds, scores)
    np_inds = _to_numpy(inds).reshape(-1, inds.shape[-1])
    np_scores = _to_numpy(scores).reshape(-1, scores.shape[-1])
    self._reap_accumulator.update(self._reap_layer_idx, np_inds, np_scores)

    y = (y * scores[..., None]).sum(axis=-2)
    return y


def _qwen3_next_counting_steering_call(self, x: mx.array) -> mx.array:
    gates = self.gate(x)
    if self._steering_bias is not None:
        gates = gates + self._steering_bias
    gates = mx.softmax(gates, axis=-1, precise=True)

    k = self.top_k
    inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
    scores = mx.take_along_axis(gates, inds, axis=-1)
    if self.norm_topk_prob:
        scores = scores / mx.sum(scores, axis=-1, keepdims=True)

    y = self.switch_mlp(x, inds)

    mx.eval(inds, scores)
    np_inds = _to_numpy(inds).reshape(-1, inds.shape[-1])
    np_scores = _to_numpy(scores).reshape(-1, scores.shape[-1])
    self._reap_accumulator.update(self._reap_layer_idx, np_inds, np_scores)

    y = (y * scores[..., None]).sum(axis=-2)

    shared_y = self.shared_expert(x)
    shared_y = mx.sigmoid(self.shared_expert_gate(x)) * shared_y

    return y + shared_y


def _gemma4_counting_steering_call(self, h: mx.array) -> mx.array:
    router = self.router
    x_normed = mx.fast.rms_norm(h, router.scale * router._root_size, router.eps)
    expert_scores = router.proj(x_normed)

    if self._steering_bias is not None:
        expert_scores = expert_scores + self._steering_bias

    router_probs = mx.softmax(expert_scores, axis=-1)
    k = router.config.top_k_experts
    inds = mx.argpartition(-expert_scores, kth=k - 1, axis=-1)[..., :k]
    scores = mx.take_along_axis(router_probs, inds, axis=-1)
    scores = scores / mx.sum(scores, axis=-1, keepdims=True)
    scores = scores * router.per_expert_scale[inds]

    h2 = self.pre_feedforward_layernorm_2(h)
    result = self.experts(h2, inds, scores)

    mx.eval(inds, scores)
    np_inds = _to_numpy(inds).reshape(-1, inds.shape[-1])
    np_scores = _to_numpy(scores).reshape(-1, scores.shape[-1])
    self._reap_accumulator.update(self._reap_layer_idx, np_inds, np_scores)

    return result


_COUNTING_STEERING_HOOK_MAP = {
    "minimax": _minimax_counting_steering_call,
    "minimax_m2": _minimax_counting_steering_call,
    "glm4_moe": _glm4_counting_steering_call,
    "glm4_moe_lite": _glm4_counting_steering_call,
    "glm_moe_dsa": _glm4_counting_steering_call,
    "deepseek_v32": _glm4_counting_steering_call,
    "nemotron_h": _glm4_counting_steering_call,
    "qwen3_moe": _qwen3_moe_counting_steering_call,
    "qwen3_next": _qwen3_next_counting_steering_call,
    "gemma4": _gemma4_counting_steering_call,
}


# ---------------------------------------------------------------------------
# Steering bias management
# ---------------------------------------------------------------------------

def _update_steering_bias(moe_blocks: List, config, num_experts: int):
    """Update _steering_bias on all blocks without reinstalling hooks.

    Thread-safe: mx.array attribute assignment is atomic at the GIL level.
    """
    from .steering import _compute_bias

    for layer_idx, block in enumerate(moe_blocks):
        block._steering_bias = _compute_bias(layer_idx, num_experts, config)


# ---------------------------------------------------------------------------
# Hook installation / removal
# ---------------------------------------------------------------------------

def install_counting_hooks(
    moe_blocks: List,
    model_type: str,
    accumulator: OnlineAccumulator,
    mode: str = "lightweight",
    steering: bool = False,
) -> None:
    """Install hooks that accumulate directly into OnlineAccumulator.

    Args:
        moe_blocks: List of MoE nn.Module instances.
        model_type: Model type string (e.g. 'minimax', 'qwen3_moe').
        accumulator: Shared OnlineAccumulator instance.
        mode: 'lightweight' (skip norms) or 'full' (compute norms).
        steering: If True, use compound counting+steering hooks.
    """
    if steering:
        hook_map = _COUNTING_STEERING_HOOK_MAP
    elif mode == "full":
        hook_map = _FULL_COUNTING_HOOK_MAP
    else:
        hook_map = _COUNTING_HOOK_MAP

    hook_fn = hook_map.get(model_type)
    if hook_fn is None:
        raise ValueError(f"No counting hook for model_type '{model_type}'")

    for layer_idx, block in enumerate(moe_blocks):
        block._reap_accumulator = accumulator
        block._reap_layer_idx = layer_idx
        if steering:
            block._steering_bias = None  # Will be set by _update_steering_bias
        original_cls = type(block)
        block._reap_original_cls = original_cls
        hooked_cls = type(
            f"_Counting_{original_cls.__name__}",
            (original_cls,),
            {"__call__": hook_fn},
        )
        block.__class__ = hooked_cls


def remove_counting_hooks(moe_blocks: List) -> None:
    """Remove counting hooks, restoring original class."""
    for block in moe_blocks:
        if hasattr(block, "_reap_original_cls"):
            block.__class__ = block._reap_original_cls
            delattr(block, "_reap_original_cls")
        for attr in ("_reap_accumulator", "_reap_layer_idx", "_steering_bias"):
            if hasattr(block, attr):
                delattr(block, attr)


# ---------------------------------------------------------------------------
# Model path resolution
# ---------------------------------------------------------------------------

_LMSTUDIO_ROOT = Path.home() / ".lmstudio" / "models"


def _resolve_model_path(model_id: str) -> str:
    """Resolve a model identifier to a locally available filesystem path.

    The server never downloads models. Resolution order:
      1. Absolute path that exists → use it.
      2. Exact relative path under ~/.lmstudio/models/ → use it.
      3. Partial name match (directory name ends with model_id) → use it.
      4. HuggingFace cache snapshot for a repo ID → use it.

    Raises ValueError if the model cannot be found locally, or on ambiguous
    partial match.
    """
    # 1. Absolute path
    p = Path(model_id)
    if p.is_absolute() and p.exists():
        return str(p)

    # 2-3. Search lmstudio models directory
    if _LMSTUDIO_ROOT.exists():
        # 2. Exact relative match
        candidate = _LMSTUDIO_ROOT / model_id
        if candidate.exists() and (candidate / "config.json").exists():
            return str(candidate)

        # 3. Partial / suffix match
        matches = []
        for config_path in _LMSTUDIO_ROOT.rglob("config.json"):
            model_dir = config_path.parent
            rel = str(model_dir.relative_to(_LMSTUDIO_ROOT))
            if rel.endswith(model_id) or model_id in rel:
                matches.append(str(model_dir))

        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(
                f"Ambiguous model ID '{model_id}' matches {len(matches)} models: "
                + ", ".join(matches[:5])
            )

    # 4. HuggingFace cache lookup (local_files_only — no download).
    # Only attempt for strings that look like repo IDs ("org/name").
    if "/" in model_id and not model_id.startswith("/"):
        try:
            from huggingface_hub import snapshot_download
            from huggingface_hub.errors import LocalEntryNotFoundError

            try:
                cached = snapshot_download(model_id, local_files_only=True)
                if (Path(cached) / "config.json").exists():
                    return cached
            except LocalEntryNotFoundError:
                pass
        except ImportError:
            pass

    raise ValueError(
        f"Model '{model_id}' is not available locally. "
        f"The server is configured to never download models. "
        f"Place the model under {_LMSTUDIO_ROOT} or pre-download it into "
        f"the HuggingFace cache (~/.cache/huggingface/hub)."
    )


# ---------------------------------------------------------------------------
# On-demand model manager
# ---------------------------------------------------------------------------

class ModelManager:
    """Manages single-model lifecycle: on-demand load, hook install, auto-unload."""

    def __init__(
        self,
        mode: str = "lightweight",
        max_tokens: int = 512,
        chat_template: Optional[str] = None,
        idle_timeout: float = 1800.0,
        max_kv_size: Optional[int] = None,
        kv_compress: Optional[str] = None,
        kv_compress_bits: int = 4,
        draft_model_path: Optional[str] = None,
        num_draft_tokens: int = 3,
        capture_layers: Optional[str] = None,
        dflash_block_size: Optional[int] = None,
        dflash_num_layers: int = 5,
        dflash_num_heads: int = 8,
    ):
        # Config (immutable for server lifetime)
        self._mode = mode
        self._max_tokens = max_tokens
        self._chat_template = chat_template
        self._idle_timeout = idle_timeout
        self._max_kv_size = max_kv_size
        self._kv_compress = kv_compress
        self._kv_compress_bits = kv_compress_bits
        self._draft_model_path = draft_model_path
        self._num_draft_tokens = num_draft_tokens
        self._capture_layers = capture_layers
        self._dflash_block_size = dflash_block_size
        self._dflash_num_layers = dflash_num_layers
        self._dflash_num_heads = dflash_num_heads

        # Mutable state (protected by _lock)
        self._lock = threading.RLock()
        self._model = None
        self._tokenizer = None
        self._config: Optional[dict] = None
        self._model_path: Optional[str] = None
        self._model_type: Optional[str] = None
        self._adapter = None
        self._moe_blocks: List = []
        self._accumulator: Optional[OnlineAccumulator] = None
        self._response_generator = None
        self._provider = None
        self._n_experts: int = 0
        self._kv_compress_info: Optional[dict] = None
        self._steering_config = None  # Persists across model swaps

        # Loading state
        self._loading = False
        self._load_condition = threading.Condition(self._lock)

        # Idle timer
        self._last_request_time: float = 0.0
        self._unload_timer: Optional[threading.Timer] = None

    # --- Properties ---

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def loaded_model_path(self) -> Optional[str]:
        return self._model_path

    @property
    def accumulator(self) -> Optional[OnlineAccumulator]:
        return self._accumulator

    @property
    def moe_blocks(self) -> List:
        return self._moe_blocks

    @property
    def n_experts(self) -> int:
        return self._n_experts

    @property
    def kv_compress_info(self) -> Optional[dict]:
        return self._kv_compress_info

    @property
    def max_kv_size(self) -> Optional[int]:
        return self._max_kv_size

    @property
    def steering_config(self):
        return self._steering_config

    @steering_config.setter
    def steering_config(self, value):
        self._steering_config = value

    # --- Core lifecycle ---

    def ensure_loaded(self, model_id: str):
        """Ensure model_id is loaded. Returns the ResponseGenerator.

        If a different model is loaded, unloads it first.
        If the same model is loaded, resets the idle timer.
        Blocks concurrent callers while loading is in progress.
        """
        resolved = _resolve_model_path(model_id)

        with self._lock:
            # Same model already loaded
            if self._model_path == resolved and self._response_generator is not None:
                self._reset_idle_timer()
                return self._response_generator

            # Wait if another thread is loading
            while self._loading:
                self._load_condition.wait()

            # Re-check after wait (other thread may have loaded our model)
            if self._model_path == resolved and self._response_generator is not None:
                self._reset_idle_timer()
                return self._response_generator

            self._loading = True

        # Load outside lock (long operation)
        try:
            self._do_load(resolved)
        finally:
            with self._lock:
                self._loading = False
                self._load_condition.notify_all()

        return self._response_generator

    def _do_load(self, resolved_path: str):
        """Load model, install hooks, create ResponseGenerator."""
        from mlx_lm import load as mlx_load
        from mlx_lm.server import LRUPromptCache, ResponseGenerator

        logging.info(f"Loading model: {resolved_path}")
        model, tokenizer, config = mlx_load(resolved_path, return_config=True)

        model_type = config.get("model_type", "")

        # Set up MoE adapter + hooks
        adapter, moe_blocks, accumulator, n_experts = self._setup_hooks(model, config)

        # Apply KV compression
        kv_compress_info = self._setup_kv_compression(model, model_type)

        # Resolve chat template
        chat_template_content = _resolve_chat_template(self._chat_template, model_type)

        # Build cli_args and provider
        cli_kwargs = dict(
            model=resolved_path,
            max_tokens=self._max_tokens,
            num_draft_tokens=self._num_draft_tokens,
        )
        if chat_template_content:
            cli_kwargs["chat_template"] = chat_template_content
        if self._draft_model_path:
            cli_kwargs["draft_model"] = self._draft_model_path
        cli_args = _make_cli_args(**cli_kwargs)

        provider = ReapModelProvider(model, tokenizer, cli_args)
        if provider.draft_model is not None:
            logging.info(
                f"Speculative decoding enabled: draft_model={self._draft_model_path}, "
                f"num_draft_tokens={self._num_draft_tokens}"
            )

        # Install hidden state capture hooks if requested (Phase 2)
        # When DFlash is enabled, auto-configure capture layers if not set
        capture_layers = self._capture_layers
        if self._dflash_block_size is not None and capture_layers is None:
            from .dflash_draft import build_target_layer_ids
            num_model_layers = len(model.model.layers)
            auto_layer_ids = build_target_layer_ids(num_model_layers, self._dflash_num_layers)
            capture_layers = ",".join(str(i) for i in auto_layer_ids)
            logging.info(
                f"DFlash auto-configured capture layers: {auto_layer_ids}"
            )

        if capture_layers is not None:
            from .hidden_state_capture import HiddenStateCapture, parse_capture_layers

            num_model_layers = len(model.model.layers)
            layer_indices = parse_capture_layers(capture_layers, num_model_layers)
            if layer_indices is not None:
                hsc = HiddenStateCapture(model, layer_indices)
                hsc.install()
                provider.hidden_state_capture = hsc
                logging.info(
                    f"Hidden state capture enabled on {len(layer_indices)} "
                    f"decoder layers: {layer_indices}"
                )

        # Create DFlash block diffusion draft model (Phase 3)
        if self._dflash_block_size is not None:
            from .dflash_draft import create_dflash_draft_model

            dflash_model = create_dflash_draft_model(
                target_model=model,
                num_layers=self._dflash_num_layers,
                num_heads=self._dflash_num_heads,
                block_size=self._dflash_block_size,
            )
            mx.eval(dflash_model.parameters())
            provider.dflash_draft_model = dflash_model
            logging.info(
                f"DFlash draft model enabled: block_size={self._dflash_block_size}, "
                f"layers={self._dflash_num_layers}, heads={self._dflash_num_heads}, "
                f"target_layers={dflash_model.target_layer_ids}"
            )

        prompt_cache = LRUPromptCache()
        response_generator = ResponseGenerator(provider, prompt_cache)

        # Swap state under lock
        with self._lock:
            old_rg = self._response_generator
            old_moe = list(self._moe_blocks)

            self._model = model
            self._tokenizer = tokenizer
            self._config = config
            self._model_path = resolved_path
            self._model_type = model_type
            self._adapter = adapter
            self._moe_blocks = moe_blocks
            self._accumulator = accumulator
            self._n_experts = n_experts
            self._response_generator = response_generator
            self._provider = provider
            self._kv_compress_info = kv_compress_info
            self._reset_idle_timer()

        # Clean up old resources outside lock
        if old_rg is not None:
            old_rg.stop_and_join()
        if old_moe:
            remove_counting_hooks(old_moe)
        gc.collect()

        # Re-apply steering if configured
        if self._steering_config is not None and moe_blocks:
            _update_steering_bias(moe_blocks, self._steering_config, n_experts)

        if accumulator:
            logging.info(
                f"Model loaded: {model_type}, MoE layers: {accumulator.num_layers}, "
                f"Experts: {n_experts}"
            )
        else:
            logging.info(f"Model loaded: {model_type} (plain inference, no MoE)")

    def _setup_hooks(self, model, config):
        """Install MoE hooks. Returns (adapter, moe_blocks, accumulator, n_experts)."""
        from .adapters import get_adapter

        model_type = config.get("model_type", "")
        try:
            adapter = get_adapter(model, config)
            moe_indices = adapter.moe_layer_indices()
        except (ValueError, KeyError, TypeError):
            logging.info(f"Model type '{model_type}' has no MoE adapter — plain inference")
            return None, [], None, 0

        # Dense model (no MoE layers) — skip hook installation
        if not moe_indices:
            logging.info(f"Model type '{model_type}' has no MoE layers — plain inference")
            return None, [], None, 0

        try:
            n_experts = adapter.num_routed_experts()
        except (ValueError, KeyError, TypeError):
            n_experts = None

        if not n_experts:
            logging.info(f"Model type '{model_type}' has no routed experts — plain inference")
            return None, [], None, 0

        accumulator = OnlineAccumulator(len(moe_indices), n_experts)
        moe_blocks = [adapter.get_moe_block(i) for i in moe_indices]
        install_counting_hooks(
            moe_blocks, model_type, accumulator, mode=self._mode, steering=True,
        )
        return adapter, moe_blocks, accumulator, n_experts

    def _setup_kv_compression(self, model, model_type):
        """Apply KV compression if configured. Returns kv_compress_info dict or None."""
        kv_compress_info = None

        if self._max_kv_size is not None and not self._kv_compress:
            from mlx_lm.models.cache import RotatingKVCache
            _num_layers = len(model.layers)
            _max_kv = self._max_kv_size

            def _make_cache():
                return [RotatingKVCache(max_size=_max_kv, keep=4) for _ in range(_num_layers)]

            model.make_cache = _make_cache

        if self._kv_compress == "turbo":
            from .kv_compress import TurboQuantConfig, TurboQuantKVCache, setup_turbo_quant
            cfg = TurboQuantConfig(bits=self._kv_compress_bits, max_size=self._max_kv_size)
            caches, sdpa_patched = setup_turbo_quant(model, model_type, cfg)
            eff_cfg = caches[0].config if caches else cfg
            _num_layers = len(model.layers)

            def _make_cache():
                return [TurboQuantKVCache(config=eff_cfg) for _ in range(_num_layers)]

            model.make_cache = _make_cache
            kv_compress_info = {
                "enabled": True, "bits": self._kv_compress_bits,
                "method": "TurboQuant/PolarQuant", "quantized_sdpa": sdpa_patched,
                "max_size": self._max_kv_size,
            }
        elif self._kv_compress == "rotor":
            from .rotor_quant import RotorQuantConfig, RotorQuantKVCache
            cfg = RotorQuantConfig(bits=self._kv_compress_bits, max_size=self._max_kv_size)
            _num_layers = len(model.layers)

            def _make_cache():
                return [RotorQuantKVCache(config=cfg) for _ in range(_num_layers)]

            model.make_cache = _make_cache
            kv_compress_info = {
                "enabled": True, "bits": self._kv_compress_bits,
                "method": "RotorQuant/Clifford", "quantized_sdpa": False,
                "max_size": self._max_kv_size,
            }

        return kv_compress_info

    def _unload_model(self):
        """Unload current model and free memory. Must be called with _lock held."""
        if self._response_generator is not None:
            self._response_generator.stop_and_join()
        if self._moe_blocks:
            remove_counting_hooks(self._moe_blocks)

        model_path = self._model_path
        self._model = None
        self._tokenizer = None
        self._config = None
        self._model_path = None
        self._model_type = None
        self._adapter = None
        self._moe_blocks = []
        self._accumulator = None
        self._response_generator = None
        self._provider = None
        self._n_experts = 0
        self._kv_compress_info = None

        if self._unload_timer is not None:
            self._unload_timer.cancel()
            self._unload_timer = None

        gc.collect()
        try:
            mx.metal.reset_peak_memory()
        except Exception:
            pass

        logging.info(f"Model unloaded: {model_path}")

    def _reset_idle_timer(self):
        """Cancel existing timer and start a new one. Caller holds _lock."""
        if self._unload_timer is not None:
            self._unload_timer.cancel()
        self._last_request_time = time.time()
        if self._idle_timeout > 0:
            self._unload_timer = threading.Timer(self._idle_timeout, self._on_idle_timeout)
            self._unload_timer.daemon = True
            self._unload_timer.start()

    def _on_idle_timeout(self):
        """Called by Timer thread when idle timeout expires."""
        with self._lock:
            elapsed = time.time() - self._last_request_time
            if elapsed < self._idle_timeout:
                return  # Request came in since timer was set
            if self._loading:
                return  # Load in progress
            if not self.is_loaded:
                return  # Already unloaded
            logging.info(f"Model idle for {elapsed:.0f}s, unloading...")
            self._unload_model()

    def shutdown(self):
        """Clean shutdown: cancel timer, unload model."""
        with self._lock:
            if self._unload_timer is not None:
                self._unload_timer.cancel()
                self._unload_timer = None
            if self.is_loaded:
                self._unload_model()


# ---------------------------------------------------------------------------
# Chat template auto-detection
# ---------------------------------------------------------------------------

# Map model_type to bundled template filename in src/mlx_fun/templates/
_MODEL_TYPE_TEMPLATES = {
    "gemma4": "gemma.jinja",
    "glm4_moe": "glm.jinja",
    "glm4_moe_lite": "glm_flash.jinja",
    "glm_moe_dsa": "glm.jinja",
    "deepseek_v32": "glm.jinja",
    "minimax": "minimax.jinja",
    "minimax_m2": "minimax_25.jinja",
    "qwen3_moe": "qwen35.jinja",
    "qwen3_next": "qwen35.jinja",
}


def _resolve_chat_template(
    chat_template: Optional[str], model_type: str
) -> Optional[str]:
    """Resolve chat template to a Jinja string.

    Priority:
      1. Explicit value — if it's a file path, read it; otherwise use as-is.
      2. Auto-detect from model_type using bundled templates.
      3. None — let the tokenizer's built-in template (if any) handle it.
    """
    if chat_template:
        p = Path(chat_template)
        if p.is_file():
            logging.info(f"Using chat template from file: {p}")
            return p.read_text()
        # Assume it's an inline Jinja string
        return chat_template

    # Auto-detect from model_type
    template_name = _MODEL_TYPE_TEMPLATES.get(model_type)
    if template_name:
        template_dir = Path(__file__).parent / "templates"
        template_path = template_dir / template_name
        if template_path.is_file():
            logging.info(
                f"Auto-selected chat template for {model_type}: {template_name}"
            )
            return template_path.read_text()
        else:
            logging.warning(
                f"Bundled template {template_name} not found at {template_path}"
            )
    return None


# ---------------------------------------------------------------------------
# Subclassed ModelProvider — accepts pre-loaded model
# ---------------------------------------------------------------------------

def _make_cli_args(**kwargs) -> argparse.Namespace:
    """Build a minimal cli_args namespace expected by mlx-lm server."""
    defaults = dict(
        model=None,
        adapter_path=None,
        draft_model=None,
        host="127.0.0.1",
        port=8080,
        trust_remote_code=False,
        chat_template="",
        use_default_chat_template=False,
        temp=0.0,
        top_p=1.0,
        top_k=0,
        min_p=0.0,
        max_tokens=512,
        num_draft_tokens=3,
        chat_template_args={},
        decode_concurrency=32,
        prompt_concurrency=8,
        prefill_step_size=2048,
        prompt_cache_size=10,
        prompt_cache_bytes=None,
        allowed_origins="*",
        pipeline=False,
        log_level="INFO",
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


class ReapModelProvider:
    """ModelProvider that wraps a pre-loaded model+tokenizer.

    This avoids double-loading: the model is loaded once at startup (so we can
    inspect config and install hooks), then wrapped here for mlx-lm's server.

    Supports speculative decoding: when ``cli_args.draft_model`` is set, the
    draft model is loaded once at init and exposed via ``self.draft_model``.
    ResponseGenerator reads this attribute to decide whether to use
    ``speculative_generate_step``.
    """

    def __init__(self, model, tokenizer, cli_args: argparse.Namespace):
        self.cli_args = cli_args
        self.model = model
        self.tokenizer = tokenizer
        self.draft_model = None
        self.hidden_state_capture = None  # HiddenStateCapture (Phase 2)
        self.dflash_draft_model = None   # DFlash draft model (Phase 3)
        self.model_key = ("reap_preloaded", None, None)

        group = mx.distributed.init()
        self.pipeline_group = group if group.size() > 1 and cli_args.pipeline else None
        self.tensor_group = (
            group if group.size() > 1 and not cli_args.pipeline else None
        )
        self.is_distributed = group.size() > 1

        # Load draft model for speculative decoding if specified
        if getattr(cli_args, "draft_model", None) is not None:
            self._load_draft_model(cli_args.draft_model)

        # Check batchability — disabled when draft model is present
        from mlx_lm.server import make_prompt_cache
        if self.draft_model is None:
            self.is_batchable = all(
                hasattr(c, "merge") for c in make_prompt_cache(self.model)
            )
        else:
            self.is_batchable = False

    def _load_draft_model(self, draft_model_path: str):
        """Load and validate a draft model for speculative decoding."""
        from mlx_lm import load as mlx_load

        resolved = _resolve_model_path(draft_model_path)
        logging.info(f"Loading draft model: {resolved}")
        draft_model, draft_tokenizer = mlx_load(resolved)

        if draft_tokenizer.vocab_size != self.tokenizer.vocab_size:
            logging.warning(
                "Draft model tokenizer does not match model tokenizer "
                f"(draft vocab={draft_tokenizer.vocab_size}, "
                f"target vocab={self.tokenizer.vocab_size}). "
                "Speculative decoding may not work as expected."
            )

        self.draft_model = draft_model
        self.model_key = ("reap_preloaded", None, draft_model_path)

    def load(self, model_path=None, adapter_path=None, draft_model_path=None):
        """Return the pre-loaded model — no actual loading occurs.

        Handles draft_model_path for compatibility with mlx-lm's server:
        - ``"default_model"``: use the draft model from CLI args (already loaded)
        - explicit path: load that specific draft model
        - ``None``: no draft model
        """
        if draft_model_path == "default_model":
            pass  # Already loaded in __init__ from cli_args
        elif draft_model_path is not None:
            self._load_draft_model(draft_model_path)

        return self.model, self.tokenizer


# ---------------------------------------------------------------------------
# Performance metrics helper
# ---------------------------------------------------------------------------

def _build_perf_block(
    prompt_tokens: int,
    completion_tokens: int,
    t_generate_start: float,
    t_first_token: Optional[float],
    t_end: float,
) -> dict:
    """Build a perf stats block with TTFT and throughput metrics.

    Args:
        prompt_tokens: Number of prompt tokens processed.
        completion_tokens: Number of tokens generated.
        t_generate_start: time.perf_counter() when generate() returned (prompt done).
        t_first_token: time.perf_counter() when first token was yielded, or None.
        t_end: time.perf_counter() when generation finished.
    """
    total_time = t_end - t_generate_start
    ttft = (t_first_token - t_generate_start) if t_first_token else None

    gen_time = (t_end - t_first_token) if t_first_token else total_time
    gen_tps = (completion_tokens / gen_time) if gen_time > 0 else 0.0

    perf: dict = {
        "time_to_first_token_s": round(ttft, 4) if ttft is not None else None,
        "generation_tokens_per_s": round(gen_tps, 2),
        "generation_time_s": round(total_time, 4),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }
    return perf


# ---------------------------------------------------------------------------
# Subclassed APIHandler — adds /v1/reap/* endpoints
# ---------------------------------------------------------------------------

class ReapAPIHandler:
    """Mixin-style handler factory that creates an APIHandler subclass
    with access to the ModelManager and steering controls."""

    @staticmethod
    def create_handler_class(model_manager: ModelManager):
        """Dynamically create a handler class with ModelManager reference.

        We need to do this because BaseHTTPRequestHandler is instantiated
        per-request and the mlx-lm factory pattern passes response_generator
        to __init__. We attach the model_manager as a class attribute.
        """
        from mlx_lm.server import APIHandler

        class _ReapHandler(APIHandler):
            _model_manager = model_manager

            def _ensure_model(self, model_id: str):
                """Load model on demand and update self.response_generator."""
                rg = self._model_manager.ensure_loaded(model_id)
                self.response_generator = rg
                return rg

            def do_GET(self):
                try:
                    if self.path == "/v1/reap/stats":
                        self._handle_reap_stats()
                    elif self.path == "/v1/reap/info":
                        self._handle_reap_info()
                    elif self.path == "/v1/reap/steer":
                        self._handle_steer_get()
                    else:
                        super().do_GET()
                except BrokenPipeError:
                    logging.debug("Client disconnected (GET %s)", self.path)

            def handle_models_request(self):
                """List models from ~/.lmstudio/models with loaded status."""
                models = []
                loaded_path = self._model_manager.loaded_model_path
                loaded_resolved = str(Path(loaded_path).resolve()) if loaded_path else None

                if _LMSTUDIO_ROOT.exists():
                    for config_path in _LMSTUDIO_ROOT.rglob("config.json"):
                        model_dir = config_path.parent
                        model_id = str(model_dir.relative_to(_LMSTUDIO_ROOT))
                        is_loaded = (
                            loaded_resolved is not None
                            and str(model_dir.resolve()) == loaded_resolved
                        )
                        models.append({
                            "id": model_id,
                            "object": "model",
                            "created": self.created,
                            "loaded": is_loaded,
                        })
                response = {"object": "list", "data": models}
                self._json_response(200, response)

            def do_POST(self):
                try:
                    if self.path == "/v1/reap/save":
                        self._handle_reap_save()
                    elif self.path == "/v1/reap/reset":
                        self._handle_reap_reset()
                    elif self.path == "/v1/reap/steer":
                        self._handle_steer_post()
                    elif self.path == "/v1/messages":
                        self._handle_anthropic_messages()
                    elif self.path in ("/v1/chat/completions", "/v1/completions",
                                       "/chat/completions"):
                        # Pre-read body to peek at model field, load on demand,
                        # then re-inject body for parent's do_POST to parse.
                        content_length = self.headers.get("Content-Length")
                        if content_length is None:
                            self._set_completion_headers(411)
                            self.end_headers()
                            self.wfile.write(
                                json.dumps({"error": "Content-Length required"}).encode()
                            )
                            return
                        raw = self.rfile.read(int(content_length))
                        try:
                            body = json.loads(raw)
                        except json.JSONDecodeError as e:
                            self._set_completion_headers(400)
                            self.end_headers()
                            self.wfile.write(
                                json.dumps({"error": f"Invalid JSON: {e}"}).encode()
                            )
                            return

                        model_id = body.get("model", "default")
                        try:
                            self._ensure_model(model_id)
                        except Exception as e:
                            logging.error(f"Model load failed for '{model_id}': {e}", exc_info=True)
                            self._json_response(503, {"error": f"Model load failed: {e}"})
                            return

                        # Re-stuff body for parent do_POST
                        self.rfile = io.BytesIO(raw)
                        super().do_POST()
                    else:
                        super().do_POST()
                except BrokenPipeError:
                    logging.debug("Client disconnected (POST %s)", self.path)

            def do_DELETE(self):
                try:
                    if self.path == "/v1/reap/steer":
                        self._handle_steer_delete()
                    else:
                        self.send_response(405)
                        self.end_headers()
                except BrokenPipeError:
                    logging.debug("Client disconnected (DELETE %s)", self.path)

            # ---------------------------------------------------------------
            # Override handle_completion to inject perf stats
            # ---------------------------------------------------------------

            def handle_completion(self, request, stop_words):
                """Wraps base handle_completion with timing instrumentation.

                Adds a ``perf`` block to every OpenAI response containing
                time_to_first_token_s, generation_tokens_per_s, etc.
                """
                from mlx_lm.server import (
                    CompletionRequest,
                    GenerationArguments,
                    ModelDescription,
                    SamplingArguments,
                    LogitsProcessorArguments,
                    ToolCallFormatter,
                )

                args = GenerationArguments(
                    model=ModelDescription(
                        model=self.requested_model,
                        draft=self.requested_draft_model,
                        adapter=self.adapter,
                    ),
                    sampling=SamplingArguments(
                        temperature=self.temperature,
                        top_p=self.top_p,
                        top_k=self.top_k,
                        min_p=self.min_p,
                        xtc_probability=self.xtc_probability,
                        xtc_threshold=self.xtc_threshold,
                    ),
                    logits=LogitsProcessorArguments(
                        logit_bias=self.logit_bias,
                        repetition_penalty=self.repetition_penalty,
                        repetition_context_size=self.repetition_context_size,
                        presence_penalty=self.presence_penalty,
                        presence_context_size=self.presence_context_size,
                        frequency_penalty=self.frequency_penalty,
                        frequency_context_size=self.frequency_context_size,
                    ),
                    stop_words=stop_words,
                    max_tokens=self.max_tokens,
                    num_draft_tokens=self.num_draft_tokens,
                    logprobs=self.logprobs,
                    top_logprobs=self.top_logprobs,
                    seed=self.seed,
                    chat_template_kwargs=self.chat_template_kwargs,
                )

                def keepalive_callback(processed, total):
                    logging.info(f"Prompt processing progress: {processed}/{total}")
                    if self.stream:
                        msg = f": keepalive {processed}/{total}\n\n".encode()
                        self.wfile.write(msg)
                        self.wfile.flush()

                try:
                    ctx, response = self.response_generator.generate(
                        request, args, progress_callback=keepalive_callback,
                    )
                except Exception as e:
                    self._set_completion_headers(404)
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": str(e)}).encode())
                    return

                if self.stream:
                    self._set_stream_headers(200)
                    self.end_headers()
                else:
                    self._set_completion_headers(200)

                _raw_formatter = ToolCallFormatter(ctx.tool_parser, request.tools, self.stream)

                def tool_formatter(tc):
                    """Safe wrapper: log raw input and fall back to empty on parse errors."""
                    try:
                        return _raw_formatter(tc)
                    except (ValueError, SyntaxError, KeyError) as e:
                        logging.warning(
                            "Tool call parse FAILED: %s\n"
                            "  Raw tool_text from model:\n%s\n"
                            "  Tools available: %s",
                            e,
                            tc,
                            [t.get("function", {}).get("name", "?")
                             for t in (request.tools or [])],
                        )
                        return []

                prev_state = None
                finish_reason = "stop"
                reasoning_text = ""
                made_tool_call = False
                tool_text = ""
                tool_calls = []
                text = ""
                tokens = []
                token_logprobs = []
                top_tokens = []

                # Timing
                t_generate_start = time.perf_counter()
                t_first_token = None

                try:
                    for gen in response:
                        if t_first_token is None:
                            t_first_token = time.perf_counter()

                        if gen.state == "reasoning":
                            reasoning_text += gen.text
                        elif gen.state == "tool":
                            tool_text += gen.text
                        elif gen.state == "normal":
                            if prev_state == "tool":
                                tool_calls.append(tool_text)
                                tool_text = ""
                                made_tool_call = True
                            text += gen.text

                        tokens.append(gen.token)
                        if args.logprobs:
                            token_logprobs.append(gen.logprob)
                        if args.top_logprobs > 0:
                            top_tokens.append(gen.top_tokens)

                        if (
                            self.stream
                            and gen.state != "tool"
                            and (text or tool_calls or reasoning_text)
                        ):
                            resp = self.generate_response(
                                text, None,
                                tool_calls=tool_formatter(tool_calls),
                                reasoning_text=reasoning_text,
                            )
                            self.wfile.write(f"data: {json.dumps(resp)}\n\n".encode())
                            self.wfile.flush()
                            reasoning_text = ""
                            text = ""
                            tool_calls = []

                        if gen.finish_reason is not None:
                            finish_reason = gen.finish_reason
                        prev_state = gen.state

                    if prev_state == "tool" and tool_text:
                        tool_calls.append(tool_text)
                        made_tool_call = True
                    if finish_reason == "stop" and made_tool_call:
                        finish_reason = "tool_calls"

                    t_end = time.perf_counter()
                    perf = _build_perf_block(
                        len(ctx.prompt), len(tokens),
                        t_generate_start, t_first_token, t_end,
                    )

                    if self.stream:
                        resp = self.generate_response(
                            text, finish_reason,
                            tool_calls=tool_formatter(tool_calls),
                            reasoning_text=reasoning_text,
                        )
                        resp["perf"] = perf
                        self.wfile.write(f"data: {json.dumps(resp)}\n\n".encode())
                        self.wfile.flush()
                        if (
                            self.stream_options is not None
                            and self.stream_options["include_usage"]
                        ):
                            resp = self.completion_usage_response(
                                len(ctx.prompt), len(tokens), ctx.prompt_cache_count,
                            )
                            self.wfile.write(f"data: {json.dumps(resp)}\n\n".encode())
                            self.wfile.flush()
                        self.wfile.write("data: [DONE]\n\n".encode())
                        self.wfile.flush()
                    else:
                        resp = self.generate_response(
                            text, finish_reason,
                            len(ctx.prompt), len(tokens), ctx.prompt_cache_count,
                            token_logprobs=token_logprobs,
                            top_tokens=top_tokens,
                            tokens=tokens,
                            reasoning_text=reasoning_text,
                            tool_calls=tool_formatter(tool_calls),
                        )
                        resp["perf"] = perf
                        response_json = json.dumps(resp).encode()
                        self.send_header("Content-Length", str(len(response_json)))
                        self.end_headers()
                        self.wfile.write(response_json)
                        self.wfile.flush()
                finally:
                    ctx.stop()

            def _json_response(self, status, data):
                body = json.dumps(data).encode()
                self.send_response(status)
                self.send_header("Content-type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(body)

            def _handle_reap_stats(self):
                acc = self._model_manager.accumulator
                if acc is None:
                    self._json_response(200, {"status": "no_model_loaded"})
                    return
                self._json_response(200, acc.get_stats())

            def _handle_reap_info(self):
                mm = self._model_manager
                acc = mm.accumulator
                info = {
                    "model_loaded": mm.is_loaded,
                    "model_path": mm.loaded_model_path,
                    "num_layers": acc.num_layers if acc else 0,
                    "num_experts": acc.num_experts if acc else 0,
                    "request_count": acc._request_count if acc else 0,
                    "token_count": acc._token_count if acc else 0,
                    "steering_active": mm.steering_config is not None,
                    "max_kv_size": mm.max_kv_size,
                    "kv_compress": mm.kv_compress_info,
                }
                self._json_response(200, info)

            def _handle_reap_save(self):
                acc = self._model_manager.accumulator
                if acc is None:
                    self._json_response(400, {"error": "no model loaded"})
                    return
                try:
                    content_length = int(self.headers.get("Content-Length", 0))
                    if content_length > 0:
                        raw = self.rfile.read(content_length)
                        data = json.loads(raw.decode())
                    else:
                        data = {}
                    path = data.get("path", "reap_saliency.npz")
                    acc.save(path)
                    self._json_response(200, {"status": "saved", "path": path})
                except Exception as e:
                    self._json_response(500, {"error": str(e)})

            def _handle_reap_reset(self):
                acc = self._model_manager.accumulator
                if acc is None:
                    self._json_response(400, {"error": "no model loaded"})
                    return
                acc.reset()
                self._json_response(200, {"status": "reset"})

            def _handle_steer_get(self):
                """GET /v1/reap/steer — return current steering config."""
                cfg = self._model_manager.steering_config
                if cfg is None:
                    self._json_response(200, {"active": False})
                else:
                    self._json_response(200, {"active": True, "config": cfg.to_dict()})

            def _handle_steer_post(self):
                """POST /v1/reap/steer — update steering config."""
                from .steering import SteeringConfig

                try:
                    content_length = int(self.headers.get("Content-Length", 0))
                    raw = self.rfile.read(content_length) if content_length > 0 else b"{}"
                    data = json.loads(raw.decode())

                    if "safety_map" in data:
                        config = SteeringConfig.from_safety_report(
                            data["safety_map"], data.get("mode", "safe"),
                        )
                    elif "domain_map" in data:
                        config = SteeringConfig.from_domain_report(
                            data["domain_map"], data.get("mode", "boost"),
                        )
                    else:
                        config = SteeringConfig.from_dict(data)

                    mm = self._model_manager
                    if mm.moe_blocks:
                        _update_steering_bias(mm.moe_blocks, config, mm.n_experts)
                    mm.steering_config = config
                    self._json_response(200, {"status": "steering_updated", "config": config.to_dict()})
                except Exception as e:
                    self._json_response(500, {"error": str(e)})

            def _handle_steer_delete(self):
                """DELETE /v1/reap/steer — remove all steering."""
                from .steering import SteeringConfig

                mm = self._model_manager
                if mm.moe_blocks:
                    empty = SteeringConfig()
                    _update_steering_bias(mm.moe_blocks, empty, mm.n_experts)
                mm.steering_config = None
                self._json_response(200, {"status": "steering_removed"})

            # ---------------------------------------------------------------
            # Anthropic Messages API  (/v1/messages)
            # ---------------------------------------------------------------

            def _handle_anthropic_messages(self):
                """POST /v1/messages — Anthropic Messages API.

                Converts Anthropic format to OpenAI internal format, runs the
                same generation pipeline (jinja templates + ResponseGenerator),
                then converts output back to Anthropic response format.
                """
                from mlx_lm.server import (
                    CompletionRequest,
                    GenerationArguments,
                    ModelDescription,
                    SamplingArguments,
                    LogitsProcessorArguments,
                )
                from .api_compat import (
                    anthropic_to_openai_messages,
                    build_anthropic_response,
                    map_stop_reason,
                    anthropic_stream_message_start,
                    anthropic_stream_content_block_start,
                    anthropic_stream_content_block_delta,
                    anthropic_stream_content_block_stop,
                    anthropic_stream_message_delta,
                    anthropic_stream_message_stop,
                    format_anthropic_sse,
                )

                # Parse request body
                content_length = self.headers.get("Content-Length")
                if content_length is None:
                    self._json_response(400, {
                        "type": "error",
                        "error": {"type": "invalid_request_error", "message": "Content-Length header is required"},
                    })
                    return
                try:
                    raw = self.rfile.read(int(content_length))
                    body = json.loads(raw.decode())
                except (ValueError, json.JSONDecodeError) as e:
                    self._json_response(400, {
                        "type": "error",
                        "error": {"type": "invalid_request_error", "message": str(e)},
                    })
                    return

                if not isinstance(body, dict):
                    self._json_response(400, {
                        "type": "error",
                        "error": {"type": "invalid_request_error", "message": "Request body must be a JSON object"},
                    })
                    return

                # Validate required fields
                if "messages" not in body:
                    self._json_response(400, {
                        "type": "error",
                        "error": {"type": "invalid_request_error", "message": "messages is required"},
                    })
                    return
                if "max_tokens" not in body:
                    self._json_response(400, {
                        "type": "error",
                        "error": {"type": "invalid_request_error", "message": "max_tokens is required"},
                    })
                    return

                # Load model on demand
                model_name = body.get("model", "default")
                try:
                    self._ensure_model(model_name)
                except Exception as e:
                    logging.error(f"Model load failed for '{model_name}': {e}", exc_info=True)
                    self._json_response(503, {
                        "type": "error",
                        "error": {"type": "api_error", "message": f"Model load failed: {e}"},
                    })
                    return

                # Convert Anthropic -> OpenAI internal format
                try:
                    messages, tools, stop_words = anthropic_to_openai_messages(body)
                except Exception as e:
                    self._json_response(400, {
                        "type": "error",
                        "error": {"type": "invalid_request_error", "message": f"Message conversion failed: {e}"},
                    })
                    return

                stream = body.get("stream", False)
                max_tokens = body.get("max_tokens", 1024)
                temperature = body.get("temperature", self.response_generator.cli_args.temp)
                top_p = body.get("top_p", self.response_generator.cli_args.top_p)
                top_k = body.get("top_k", self.response_generator.cli_args.top_k)

                # Build generation arguments
                request = CompletionRequest("chat", "", messages, tools, None)
                args = GenerationArguments(
                    model=ModelDescription(model=model_name, draft=None, adapter=None),
                    sampling=SamplingArguments(
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        min_p=0.0,
                        xtc_probability=0.0,
                        xtc_threshold=0.0,
                    ),
                    logits=LogitsProcessorArguments(
                        logit_bias=None,
                        repetition_penalty=0.0,
                        repetition_context_size=20,
                        presence_penalty=0.0,
                        presence_context_size=20,
                        frequency_penalty=0.0,
                        frequency_context_size=20,
                    ),
                    stop_words=stop_words,
                    max_tokens=max_tokens,
                    num_draft_tokens=self.response_generator.cli_args.num_draft_tokens,
                    logprobs=False,
                    top_logprobs=-1,
                    seed=None,
                    chat_template_kwargs=None,
                )

                # Generate
                try:
                    ctx, response = self.response_generator.generate(request, args)
                except Exception as e:
                    self._json_response(500, {
                        "type": "error",
                        "error": {"type": "api_error", "message": str(e)},
                    })
                    return

                if stream:
                    self._anthropic_stream_response(
                        ctx, response, model_name, format_anthropic_sse,
                        anthropic_stream_message_start,
                        anthropic_stream_content_block_start,
                        anthropic_stream_content_block_delta,
                        anthropic_stream_content_block_stop,
                        anthropic_stream_message_delta,
                        anthropic_stream_message_stop,
                        map_stop_reason, request_tools=tools,
                    )
                else:
                    self._anthropic_batch_response(
                        ctx, response, model_name, build_anthropic_response,
                        map_stop_reason, request_tools=tools,
                    )

            def _anthropic_batch_response(self, ctx, response, model_name,
                                          build_response, map_stop_reason_fn,
                                          request_tools=None):
                """Collect full generation and return Anthropic JSON response."""
                from mlx_lm.server import ToolCallFormatter

                text = ""
                tokens = []
                finish_reason = "stop"
                tool_text = ""
                tool_calls = []
                made_tool_call = False
                prev_state = None
                t_generate_start = time.perf_counter()
                t_first_token = None

                try:
                    for gen in response:
                        if t_first_token is None:
                            t_first_token = time.perf_counter()

                        if gen.state == "tool":
                            tool_text += gen.text
                        elif gen.state == "normal":
                            if prev_state == "tool":
                                tool_calls.append(tool_text)
                                tool_text = ""
                                made_tool_call = True
                            text += gen.text
                        # reasoning state: tokens skipped in Anthropic output

                        tokens.append(gen.token)
                        if gen.finish_reason is not None:
                            finish_reason = gen.finish_reason
                        prev_state = gen.state

                    if prev_state == "tool" and tool_text:
                        tool_calls.append(tool_text)
                        made_tool_call = True
                finally:
                    ctx.stop()

                # Parse tool call XML/text into structured tool_calls
                parsed_tool_calls = []
                if tool_calls:
                    raw_formatter = ToolCallFormatter(
                        ctx.tool_parser, request_tools, False,
                    )
                    try:
                        parsed_tool_calls = raw_formatter(tool_calls)
                    except (ValueError, SyntaxError, KeyError) as e:
                        logging.warning(
                            "Anthropic tool call parse FAILED: %s\n  Raw: %s",
                            e, tool_calls,
                        )
                        parsed_tool_calls = []

                t_end = time.perf_counter()
                resp = build_response(
                    text=text,
                    finish_reason=finish_reason,
                    prompt_tokens=len(ctx.prompt),
                    completion_tokens=len(tokens),
                    model=model_name,
                    tool_calls=parsed_tool_calls or None,
                )
                resp["perf"] = _build_perf_block(
                    len(ctx.prompt), len(tokens),
                    t_generate_start, t_first_token, t_end,
                )

                body = json.dumps(resp).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(body)
                self.wfile.flush()

            def _anthropic_stream_response(self, ctx, response, model_name,
                                           fmt_sse, msg_start, cb_start,
                                           cb_delta, cb_stop, msg_delta,
                                           msg_stop, map_stop_reason_fn,
                                           request_tools=None):
                """Stream generation as Anthropic SSE events."""
                from mlx_lm.server import ToolCallFormatter

                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()

                # 1. message_start
                self.wfile.write(fmt_sse("message_start", msg_start(model_name, len(ctx.prompt))))
                self.wfile.flush()

                # 2. content_block_start (text block at index 0)
                self.wfile.write(fmt_sse("content_block_start", cb_start(0, "text")))
                self.wfile.flush()

                # 3. Stream content_block_delta for each token
                tokens = []
                finish_reason = "stop"
                tool_text = ""
                tool_calls = []
                prev_state = None
                text_block_open = True
                t_generate_start = time.perf_counter()
                t_first_token = None
                try:
                    for gen in response:
                        if t_first_token is None:
                            t_first_token = time.perf_counter()
                        tokens.append(gen.token)

                        if gen.state == "tool":
                            tool_text += gen.text
                        elif gen.state == "normal":
                            if prev_state == "tool" and tool_text:
                                tool_calls.append(tool_text)
                                tool_text = ""
                            if gen.text:
                                self.wfile.write(
                                    fmt_sse("content_block_delta", cb_delta(0, gen.text))
                                )
                                self.wfile.flush()

                        if gen.finish_reason is not None:
                            finish_reason = gen.finish_reason
                        prev_state = gen.state

                    if prev_state == "tool" and tool_text:
                        tool_calls.append(tool_text)
                finally:
                    ctx.stop()

                t_end = time.perf_counter()

                # 4. close text block
                self.wfile.write(fmt_sse("content_block_stop", cb_stop(0)))
                self.wfile.flush()

                # 5. emit tool_use blocks if any
                parsed_tool_calls = []
                if tool_calls:
                    raw_formatter = ToolCallFormatter(
                        ctx.tool_parser, request_tools, False,
                    )
                    try:
                        parsed_tool_calls = raw_formatter(tool_calls)
                    except (ValueError, SyntaxError, KeyError) as e:
                        logging.warning(
                            "Anthropic stream tool parse FAILED: %s\n  Raw: %s",
                            e, tool_calls,
                        )

                for idx, tc in enumerate(parsed_tool_calls, start=1):
                    fn = tc.get("function", tc)
                    args = fn.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except (json.JSONDecodeError, ValueError):
                            args = {"_raw": args}
                    tu_block = {
                        "type": "tool_use",
                        "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:12]}"),
                        "name": fn["name"],
                        "input": {},
                    }
                    self.wfile.write(fmt_sse("content_block_start", {
                        "type": "content_block_start",
                        "index": idx,
                        "content_block": tu_block,
                    }))
                    self.wfile.write(fmt_sse("content_block_delta", {
                        "type": "content_block_delta",
                        "index": idx,
                        "delta": {"type": "input_json_delta",
                                  "partial_json": json.dumps(args)},
                    }))
                    self.wfile.write(fmt_sse("content_block_stop", cb_stop(idx)))
                    self.wfile.flush()

                # 6. message_delta with stop_reason + usage + perf
                stop_reason = "tool_use" if parsed_tool_calls else map_stop_reason_fn(finish_reason)
                delta_data = msg_delta(stop_reason, len(tokens))
                delta_data["perf"] = _build_perf_block(
                    len(ctx.prompt), len(tokens),
                    t_generate_start, t_first_token, t_end,
                )
                self.wfile.write(fmt_sse("message_delta", delta_data))
                self.wfile.flush()

                # 7. message_stop
                self.wfile.write(fmt_sse("message_stop", msg_stop()))
                self.wfile.flush()

        return _ReapHandler


# ---------------------------------------------------------------------------
# Server entry point
# ---------------------------------------------------------------------------

def run_reap_server(
    host: str,
    port: int,
    model_path: Optional[str] = None,
    mode: str = "lightweight",
    auto_save: Optional[str] = None,
    max_tokens: int = 512,
    chat_template: Optional[str] = None,
    safety_map: Optional[str] = None,
    steering_mode: Optional[str] = None,
    max_kv_size: Optional[int] = None,
    domain_map: Optional[str] = None,
    domain_steering_mode: Optional[str] = None,
    kv_compress: Optional[str] = None,
    kv_compress_bits: int = 4,
    idle_timeout: float = 1800.0,
    draft_model_path: Optional[str] = None,
    num_draft_tokens: int = 3,
    capture_layers: Optional[str] = None,
    dflash_block_size: Optional[int] = None,
    dflash_num_layers: int = 5,
    dflash_num_heads: int = 8,
):
    """Start the server with on-demand model loading.

    Models are loaded when the first request specifying a model arrives.
    After ``idle_timeout`` seconds of inactivity, the model is unloaded
    to free memory.

    Args:
        host: Bind address.
        port: Bind port.
        model_path: Optional default model to load eagerly at startup.
            If None, the server starts empty and loads on first request.
        mode: 'lightweight' (freq/weighted_freq only) or 'full' (all metrics).
        auto_save: If set, save accumulator to this path on shutdown.
        max_tokens: Default max tokens for generation.
        max_kv_size: If set, cap KV cache to this many tokens per layer.
        chat_template: Optional chat template override.
        safety_map: Optional path to safety_report.json for steering.
        steering_mode: Optional 'safe' or 'unsafe' steering mode.
        domain_map: Optional path to domain_report.json for domain boosting.
        domain_steering_mode: Optional 'boost' or 'suppress' domain mode.
        kv_compress: KV compression method ('turbo', 'rotor', or None).
        kv_compress_bits: Bits per channel for KV compression (2-8).
        idle_timeout: Seconds of inactivity before auto-unloading model.
            0 disables auto-unload.
        draft_model_path: Optional path/repo ID for a draft model
            (speculative decoding). When set, the server uses mlx-lm's
            speculative decoding loop with the draft model.
        num_draft_tokens: Number of tokens to draft per speculative step.
        capture_layers: Optional comma-separated layer indices or "all".
            When set, installs hidden state capture hooks on the specified
            decoder layers for speculative decoding (Phase 2).
    """
    from mlx_lm.server import (
        LRUPromptCache,
        ResponseGenerator,
        _run_http_server,
    )

    # Force HuggingFace offline mode — the server never downloads models.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    if mx.metal.is_available():
        wired_limit = mx.device_info()["max_recommended_working_set_size"]
        mx.set_wired_limit(wired_limit)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Create model manager
    model_manager = ModelManager(
        mode=mode,
        max_tokens=max_tokens,
        chat_template=chat_template,
        idle_timeout=idle_timeout,
        max_kv_size=max_kv_size,
        kv_compress=kv_compress,
        kv_compress_bits=kv_compress_bits,
        draft_model_path=draft_model_path,
        num_draft_tokens=num_draft_tokens,
        capture_layers=capture_layers,
        dflash_block_size=dflash_block_size,
        dflash_num_layers=dflash_num_layers,
        dflash_num_heads=dflash_num_heads,
    )

    # Apply initial steering if configured
    if safety_map and steering_mode:
        from .steering import SteeringConfig
        model_manager.steering_config = SteeringConfig.from_safety_report(
            safety_map, steering_mode,
        )
        logging.info(f"Steering configured: mode={steering_mode}")

    if domain_map and domain_steering_mode:
        from .steering import SteeringConfig
        model_manager.steering_config = SteeringConfig.from_domain_report(
            domain_map, domain_steering_mode,
        )
        logging.info(f"Domain steering configured: mode={domain_steering_mode}")

    # Eagerly load default model if specified
    if model_path:
        model_manager.ensure_loaded(model_path)

    # Create handler class
    handler_class = ReapAPIHandler.create_handler_class(model_manager)

    # Placeholder ResponseGenerator for _run_http_server's handler factory.
    # The real ResponseGenerator is set per-request via _ensure_model().
    placeholder_args = _make_cli_args(max_tokens=max_tokens)
    placeholder_provider = ReapModelProvider.__new__(ReapModelProvider)
    placeholder_provider.cli_args = placeholder_args
    placeholder_provider.model = None
    placeholder_provider.tokenizer = None
    placeholder_provider.draft_model = None
    placeholder_provider.hidden_state_capture = None
    placeholder_provider.dflash_draft_model = None
    placeholder_provider.model_key = ("placeholder", None, None)
    placeholder_provider.pipeline_group = None
    placeholder_provider.tensor_group = None
    placeholder_provider.is_distributed = False
    placeholder_provider.is_batchable = False
    placeholder_rg = ResponseGenerator(placeholder_provider, LRUPromptCache())

    # Auto-save on shutdown
    def _shutdown_save(signum, frame):
        acc = model_manager.accumulator
        if auto_save and acc is not None:
            logging.info(f"Saving accumulator to {auto_save}")
            acc.save(auto_save)
        raise KeyboardInterrupt

    if auto_save:
        signal.signal(signal.SIGTERM, _shutdown_save)

    # Start server
    if model_path:
        logging.info(f"Server ready at {host}:{port} (model: {model_path})")
    else:
        logging.info(f"Server ready at {host}:{port} (no model loaded — will load on first request)")
    timeout_str = f"{idle_timeout:.0f}s" if idle_timeout > 0 else "disabled"
    logging.info(f"Auto-unload after inactivity: {timeout_str}")
    logging.info(
        "API: POST /v1/chat/completions (OpenAI), /v1/messages (Anthropic)"
    )
    logging.info(
        "REAP: GET /v1/reap/stats, /v1/reap/info, /v1/reap/steer | "
        "POST /v1/reap/save, /v1/reap/reset, /v1/reap/steer | "
        "DELETE /v1/reap/steer"
    )
    try:
        _run_http_server(host, port, placeholder_rg, handler_class=handler_class)
    except KeyboardInterrupt:
        pass
    finally:
        acc = model_manager.accumulator
        if auto_save and acc is not None:
            logging.info(f"Auto-saving accumulator to {auto_save}")
            acc.save(auto_save)
        model_manager.shutdown()
        placeholder_rg.stop_and_join()
        logging.info("Server stopped.")
