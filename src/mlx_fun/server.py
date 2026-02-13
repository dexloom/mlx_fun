"""LLM server with online expert counting.

Composes on top of mlx-lm's server infrastructure. Subclasses APIHandler to add
REAP management endpoints. Installs lightweight hooks that accumulate expert
statistics into a thread-safe OnlineAccumulator during every forward pass.
"""

import argparse
import json
import logging
import signal
import threading
import time
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
        """Return current accumulator state as JSON-serializable dict."""
        with self._lock:
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
    if self.sharding_group is not None:
        raise RuntimeError(
            "Pruning sharded models not supported in v1. Load without sharding."
        )

    inds, scores = self.gate(x)
    y = self.switch_mlp(x, inds)

    mx.eval(inds, scores)
    np_inds = _to_numpy(inds).reshape(-1, inds.shape[-1])
    np_scores = _to_numpy(scores).reshape(-1, scores.shape[-1])
    self._reap_accumulator.update(self._reap_layer_idx, np_inds, np_scores)

    y = (y * scores[..., None]).sum(axis=-2).astype(y.dtype)
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
    if self.sharding_group is not None:
        raise RuntimeError(
            "Pruning sharded models not supported in v1. Load without sharding."
        )

    inds, scores = self.gate(x)
    y = self.switch_mlp(x, inds)
    activation_norms = mx.linalg.norm(y, axis=-1)

    mx.eval(inds, scores, activation_norms)
    np_inds = _to_numpy(inds).reshape(-1, inds.shape[-1])
    np_scores = _to_numpy(scores).reshape(-1, scores.shape[-1])
    np_norms = _to_numpy(activation_norms).reshape(-1, activation_norms.shape[-1])
    self._reap_accumulator.update(self._reap_layer_idx, np_inds, np_scores, np_norms)

    y = (y * scores[..., None]).sum(axis=-2).astype(y.dtype)
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


_COUNTING_HOOK_MAP = {
    "minimax": _minimax_counting_call,
    "minimax_m2": _minimax_counting_call,
    "glm4_moe": _glm4_counting_call,
    "glm4_moe_lite": _glm4_counting_call,
    "glm_moe_dsa": _glm4_counting_call,
    "deepseek_v32": _glm4_counting_call,
    "qwen3_moe": _qwen3_moe_counting_call,
    "qwen3_next": _qwen3_next_counting_call,
}

_FULL_COUNTING_HOOK_MAP = {
    "minimax": _minimax_full_counting_call,
    "minimax_m2": _minimax_full_counting_call,
    "glm4_moe": _glm4_full_counting_call,
    "glm4_moe_lite": _glm4_full_counting_call,
    "glm_moe_dsa": _glm4_full_counting_call,
    "deepseek_v32": _glm4_full_counting_call,
    "qwen3_moe": _qwen3_moe_full_counting_call,
    "qwen3_next": _qwen3_next_full_counting_call,
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
    if self.sharding_group is not None:
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

    y = self.switch_mlp(x, inds)

    mx.eval(inds, scores)
    np_inds = _to_numpy(inds).reshape(-1, inds.shape[-1])
    np_scores = _to_numpy(scores).reshape(-1, scores.shape[-1])
    self._reap_accumulator.update(self._reap_layer_idx, np_inds, np_scores)

    y = (y * scores[..., None]).sum(axis=-2).astype(y.dtype)
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


_COUNTING_STEERING_HOOK_MAP = {
    "minimax": _minimax_counting_steering_call,
    "minimax_m2": _minimax_counting_steering_call,
    "glm4_moe": _glm4_counting_steering_call,
    "glm4_moe_lite": _glm4_counting_steering_call,
    "glm_moe_dsa": _glm4_counting_steering_call,
    "deepseek_v32": _glm4_counting_steering_call,
    "qwen3_moe": _qwen3_moe_counting_steering_call,
    "qwen3_next": _qwen3_next_counting_steering_call,
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
        pipeline=False,
        log_level="INFO",
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


class ReapModelProvider:
    """ModelProvider that wraps a pre-loaded model+tokenizer.

    This avoids double-loading: the model is loaded once at startup (so we can
    inspect config and install hooks), then wrapped here for mlx-lm's server.
    """

    def __init__(self, model, tokenizer, cli_args: argparse.Namespace):
        self.cli_args = cli_args
        self.model = model
        self.tokenizer = tokenizer
        self.draft_model = None
        self.model_key = ("reap_preloaded", None, None)

        group = mx.distributed.init()
        self.pipeline_group = group if group.size() > 1 and cli_args.pipeline else None
        self.tensor_group = (
            group if group.size() > 1 and not cli_args.pipeline else None
        )
        self.is_distributed = group.size() > 1

        # Check batchability
        from mlx_lm.utils import make_prompt_cache
        self.is_batchable = all(
            hasattr(c, "merge") for c in make_prompt_cache(self.model)
        )

    def load(self, model_path=None, adapter_path=None, draft_model_path=None):
        """Return the pre-loaded model — no actual loading occurs."""
        return self.model, self.tokenizer


# ---------------------------------------------------------------------------
# Subclassed APIHandler — adds /v1/reap/* endpoints
# ---------------------------------------------------------------------------

class ReapAPIHandler:
    """Mixin-style handler factory that creates an APIHandler subclass
    with access to the OnlineAccumulator and steering controls."""

    @staticmethod
    def create_handler_class(
        accumulator: OnlineAccumulator,
        moe_blocks: List = None,
        num_experts: int = 0,
    ):
        """Dynamically create a handler class with accumulator reference.

        We need to do this because BaseHTTPRequestHandler is instantiated
        per-request and the mlx-lm factory pattern passes response_generator
        to __init__. We attach the accumulator as a class attribute.
        """
        from mlx_lm.server import APIHandler

        class _ReapHandler(APIHandler):
            _reap_accumulator = accumulator
            _reap_moe_blocks = moe_blocks
            _reap_num_experts = num_experts
            _reap_steering_config = None  # Current SteeringConfig or None

            def do_GET(self):
                if self.path == "/v1/reap/stats":
                    self._handle_reap_stats()
                elif self.path == "/v1/reap/info":
                    self._handle_reap_info()
                elif self.path == "/v1/reap/steer":
                    self._handle_steer_get()
                else:
                    super().do_GET()

            def do_POST(self):
                if self.path == "/v1/reap/save":
                    self._handle_reap_save()
                elif self.path == "/v1/reap/reset":
                    self._handle_reap_reset()
                elif self.path == "/v1/reap/steer":
                    self._handle_steer_post()
                else:
                    super().do_POST()

            def do_DELETE(self):
                if self.path == "/v1/reap/steer":
                    self._handle_steer_delete()
                else:
                    self.send_response(405)
                    self.end_headers()

            def _json_response(self, status, data):
                body = json.dumps(data).encode()
                self.send_response(status)
                self.send_header("Content-type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(body)

            def _handle_reap_stats(self):
                self._json_response(200, self._reap_accumulator.get_stats())

            def _handle_reap_info(self):
                acc = self._reap_accumulator
                info = {
                    "num_layers": acc.num_layers,
                    "num_experts": acc.num_experts,
                    "request_count": acc._request_count,
                    "token_count": acc._token_count,
                    "steering_active": self._reap_steering_config is not None,
                }
                self._json_response(200, info)

            def _handle_reap_save(self):
                try:
                    content_length = int(self.headers.get("Content-Length", 0))
                    if content_length > 0:
                        raw = self.rfile.read(content_length)
                        data = json.loads(raw.decode())
                    else:
                        data = {}
                    path = data.get("path", "reap_saliency.npz")
                    self._reap_accumulator.save(path)
                    self._json_response(200, {"status": "saved", "path": path})
                except Exception as e:
                    self._json_response(500, {"error": str(e)})

            def _handle_reap_reset(self):
                self._reap_accumulator.reset()
                self._json_response(200, {"status": "reset"})

            def _handle_steer_get(self):
                """GET /v1/reap/steer — return current steering config."""
                cfg = _ReapHandler._reap_steering_config
                if cfg is None:
                    self._json_response(200, {"active": False})
                else:
                    self._json_response(200, {"active": True, "config": cfg.to_dict()})

            def _handle_steer_post(self):
                """POST /v1/reap/steer — update steering config.

                Body can be either:
                  {"safety_map": "/path/to/report.json", "mode": "safe"|"unsafe"}
                or direct config:
                  {"deactivate": {"0": [1, 2]}, "activate": {"3": [5]}, ...}
                """
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

                    if self._reap_moe_blocks:
                        _update_steering_bias(
                            self._reap_moe_blocks, config, self._reap_num_experts,
                        )
                    _ReapHandler._reap_steering_config = config
                    self._json_response(200, {"status": "steering_updated", "config": config.to_dict()})
                except Exception as e:
                    self._json_response(500, {"error": str(e)})

            def _handle_steer_delete(self):
                """DELETE /v1/reap/steer — remove all steering."""
                from .steering import SteeringConfig

                if self._reap_moe_blocks:
                    empty = SteeringConfig()
                    _update_steering_bias(
                        self._reap_moe_blocks, empty, self._reap_num_experts,
                    )
                _ReapHandler._reap_steering_config = None
                self._json_response(200, {"status": "steering_removed"})

        return _ReapHandler


# ---------------------------------------------------------------------------
# Server entry point
# ---------------------------------------------------------------------------

def run_reap_server(
    host: str,
    port: int,
    model_path: str,
    mode: str = "lightweight",
    auto_save: Optional[str] = None,
    max_tokens: int = 512,
    chat_template: Optional[str] = None,
    safety_map: Optional[str] = None,
    steering_mode: Optional[str] = None,
    domain_map: Optional[str] = None,
    domain_steering_mode: Optional[str] = None,
):
    """Load model, install counting hooks, and start the server.

    Args:
        host: Bind address.
        port: Bind port.
        model_path: HuggingFace repo ID or local model path.
        mode: 'lightweight' (freq/weighted_freq only) or 'full' (all metrics).
        auto_save: If set, save accumulator to this path on shutdown.
        max_tokens: Default max tokens for generation.
        chat_template: Optional chat template override.
        safety_map: Optional path to safety_report.json for steering.
        steering_mode: Optional 'safe' or 'unsafe' steering mode.
        domain_map: Optional path to domain_report.json for domain boosting.
        domain_steering_mode: Optional 'boost' or 'suppress' domain mode.
    """
    from mlx_lm import load as mlx_load
    from mlx_lm.server import (
        LRUPromptCache,
        ResponseGenerator,
        _run_http_server,
        get_system_fingerprint,
    )

    from .adapters import get_adapter

    if mx.metal.is_available():
        wired_limit = mx.device_info()["max_recommended_working_set_size"]
        mx.set_wired_limit(wired_limit)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Load model
    logging.info(f"Loading model: {model_path}")
    model, tokenizer, config = mlx_load(model_path, return_config=True)

    # Set up adapter and get MoE info
    adapter = get_adapter(model, config)
    moe_indices = adapter.moe_layer_indices()
    n_experts = adapter.num_routed_experts()
    model_type = config.get("model_type", "")

    logging.info(
        f"Model type: {model_type}, MoE layers: {len(moe_indices)}, "
        f"Experts per layer: {n_experts}"
    )

    # Create accumulator and install hooks
    # Always use compound steering hooks — when no steering is active,
    # _steering_bias is None and the if-branch is skipped (negligible overhead)
    accumulator = OnlineAccumulator(num_layers=len(moe_indices), num_experts=n_experts)
    moe_blocks = [adapter.get_moe_block(i) for i in moe_indices]
    install_counting_hooks(
        moe_blocks, model_type, accumulator, mode=mode, steering=True,
    )
    logging.info(f"Installed {mode}+steering counting hooks on {len(moe_blocks)} MoE blocks")

    # Apply initial steering if safety map provided
    if safety_map and steering_mode:
        from .steering import SteeringConfig
        steer_config = SteeringConfig.from_safety_report(safety_map, steering_mode)
        _update_steering_bias(moe_blocks, steer_config, n_experts)
        logging.info(f"Steering enabled: mode={steering_mode}, safety_map={safety_map}")

    # Apply initial domain steering if domain map provided
    if domain_map and domain_steering_mode:
        from .steering import SteeringConfig
        domain_config = SteeringConfig.from_domain_report(domain_map, domain_steering_mode)
        _update_steering_bias(moe_blocks, domain_config, n_experts)
        logging.info(f"Domain steering enabled: mode={domain_steering_mode}, domain_map={domain_map}")

    # Build cli_args namespace for mlx-lm server
    cli_kwargs = dict(
        model=model_path,
        max_tokens=max_tokens,
    )
    if chat_template:
        cli_kwargs["chat_template"] = chat_template
    cli_args = _make_cli_args(**cli_kwargs)

    # Create model provider and response generator
    provider = ReapModelProvider(model, tokenizer, cli_args)
    prompt_cache = LRUPromptCache()
    response_generator = ResponseGenerator(provider, prompt_cache)

    # Create handler class with accumulator and steering access
    handler_class = ReapAPIHandler.create_handler_class(
        accumulator, moe_blocks=moe_blocks, num_experts=n_experts,
    )

    # Auto-save on shutdown
    def _shutdown_save(signum, frame):
        if auto_save:
            logging.info(f"Saving accumulator to {auto_save}")
            accumulator.save(auto_save)
        raise KeyboardInterrupt

    if auto_save:
        signal.signal(signal.SIGTERM, _shutdown_save)

    # Start server
    logging.info(f"Starting REAP server at {host}:{port} (mode={mode})")
    logging.info(
        "REAP endpoints: GET /v1/reap/stats, /v1/reap/info, /v1/reap/steer | "
        "POST /v1/reap/save, /v1/reap/reset, /v1/reap/steer | "
        "DELETE /v1/reap/steer"
    )
    try:
        _run_http_server(host, port, response_generator, handler_class=handler_class)
    except KeyboardInterrupt:
        pass
    finally:
        if auto_save:
            logging.info(f"Auto-saving accumulator to {auto_save}")
            accumulator.save(auto_save)
        response_generator.stop_and_join()
        remove_counting_hooks(moe_blocks)
        logging.info("Server stopped.")
