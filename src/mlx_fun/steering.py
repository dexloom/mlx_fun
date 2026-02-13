"""Expert (de)activation steering for inference-time safety control.

Installs hooks that modify gate logits before top-k selection to
boost or suppress specific experts, based on the SteerMoE approach.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import mlx.core as mx
import numpy as np


# ---------------------------------------------------------------------------
# Steering configuration
# ---------------------------------------------------------------------------

@dataclass
class SteeringConfig:
    """Per-layer expert activation/deactivation specification.

    Attributes:
        deactivate: Dict[layer_idx, List[expert_idx]] — experts to suppress.
        activate: Dict[layer_idx, List[expert_idx]] — experts to boost.
        mask_value: Value to add for deactivation (effectively -inf).
        boost_value: Value to add for activation.
    """

    deactivate: Dict[int, List[int]] = field(default_factory=dict)
    activate: Dict[int, List[int]] = field(default_factory=dict)
    mask_value: float = -1e9
    boost_value: float = 1e4

    @classmethod
    def from_safety_report(cls, report_path: str, mode: str) -> "SteeringConfig":
        """Generate steering config from a safety scan report.

        Args:
            report_path: Path to safety_report.json.
            mode: 'safe' (boost safety-control experts) or
                  'unsafe' (mask all safety-critical experts).
        """
        from .safety import SafetyReport

        report = SafetyReport.load(report_path)
        config = cls()
        if mode == "safe":
            config.activate = {k: list(v) for k, v in report.hrcg_experts.items()}
        elif mode == "unsafe":
            config.deactivate = {k: list(v) for k, v in report.safety_critical.items()}
        else:
            raise ValueError(f"Unknown steering mode '{mode}'. Use: safe, unsafe")
        return config

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "deactivate": {str(k): v for k, v in self.deactivate.items()},
            "activate": {str(k): v for k, v in self.activate.items()},
            "mask_value": self.mask_value,
            "boost_value": self.boost_value,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SteeringConfig":
        """Create from dict (e.g. parsed JSON)."""
        return cls(
            deactivate={int(k): v for k, v in d.get("deactivate", {}).items()},
            activate={int(k): v for k, v in d.get("activate", {}).items()},
            mask_value=d.get("mask_value", -1e9),
            boost_value=d.get("boost_value", 1e4),
        )

    @classmethod
    def from_domain_report(cls, report_path: str, mode: str) -> "SteeringConfig":
        """Generate steering config from a domain scan report.

        Args:
            report_path: Path to domain_report.json.
            mode: 'boost' (activate domain experts) or
                  'suppress' (deactivate general experts).
        """
        from .domain import DomainReport

        report = DomainReport.load(report_path)
        config = cls()
        if mode == "boost":
            config.activate = {k: list(v) for k, v in report.domain_experts.items()}
        elif mode == "suppress":
            config.deactivate = {k: list(v) for k, v in report.general_experts.items()}
        else:
            raise ValueError(f"Unknown domain steering mode '{mode}'. Use: boost, suppress")
        return config


# ---------------------------------------------------------------------------
# Steering bias computation
# ---------------------------------------------------------------------------

def _compute_bias(
    layer_idx: int,
    num_experts: int,
    config: SteeringConfig,
) -> Optional[mx.array]:
    """Pre-compute the gate logit bias array for one layer.

    Returns None if no steering is needed for this layer.
    """
    bias = np.zeros(num_experts, dtype=np.float32)
    if layer_idx in config.deactivate:
        for eid in config.deactivate[layer_idx]:
            if 0 <= eid < num_experts:
                bias[eid] = config.mask_value
    if layer_idx in config.activate:
        for eid in config.activate[layer_idx]:
            if 0 <= eid < num_experts:
                bias[eid] = config.boost_value
    if np.any(bias != 0):
        return mx.array(bias)
    return None


# ---------------------------------------------------------------------------
# Steering hooks — one per model type
# ---------------------------------------------------------------------------

def _minimax_steering_call(self, x: mx.array) -> mx.array:
    """MiniMax forward with gate logit steering."""
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
    y = (y * scores[..., None]).sum(axis=-2)
    return y


def _glm4_steering_call(self, x: mx.array) -> mx.array:
    """GLM4 forward with gate logit steering."""
    if self.sharding_group is not None:
        raise RuntimeError(
            "Steering sharded models not supported. Load without sharding."
        )

    # GLM4 gate computes logits internally; inject bias into raw logits
    raw_gates = x @ self.gate.weight.T
    if self._steering_bias is not None:
        raw_gates = raw_gates + self._steering_bias

    # Replicate MoEGate forward with biased logits
    scores = mx.sigmoid(raw_gates.astype(mx.float32))
    orig_scores = scores
    scores = scores + self.gate.e_score_correction_bias
    k = self.gate.top_k
    inds = mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]
    scores = mx.take_along_axis(orig_scores, inds, axis=-1)
    scores = scores / (mx.sum(scores, axis=-1, keepdims=True) + 1e-20)

    y = self.switch_mlp(x, inds)
    y = (y * scores[..., None]).sum(axis=-2).astype(y.dtype)
    if hasattr(self, "shared_experts") and self.shared_experts is not None:
        y = y + self.shared_experts(x)
    return y


def _qwen3_moe_steering_call(self, x: mx.array) -> mx.array:
    """Qwen3 forward with gate logit steering."""
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
    y = (y * scores[..., None]).sum(axis=-2)
    return y


def _qwen3_next_steering_call(self, x: mx.array) -> mx.array:
    """Qwen3Next forward with gate logit steering."""
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
    y = (y * scores[..., None]).sum(axis=-2)

    shared_y = self.shared_expert(x)
    shared_y = mx.sigmoid(self.shared_expert_gate(x)) * shared_y
    return y + shared_y


_STEERING_HOOK_MAP = {
    "minimax": _minimax_steering_call,
    "minimax_m2": _minimax_steering_call,
    "glm4_moe": _glm4_steering_call,
    "glm4_moe_lite": _glm4_steering_call,
    "glm_moe_dsa": _glm4_steering_call,
    "deepseek_v32": _glm4_steering_call,
    "qwen3_moe": _qwen3_moe_steering_call,
    "qwen3_next": _qwen3_next_steering_call,
}


# ---------------------------------------------------------------------------
# Hook installation / removal / update
# ---------------------------------------------------------------------------

def install_steering_hooks(
    moe_blocks: List,
    model_type: str,
    config: SteeringConfig,
    num_experts: int,
) -> None:
    """Install steering hooks on MoE blocks.

    Pre-computes per-layer bias arrays for efficient in-loop application.
    """
    hook_fn = _STEERING_HOOK_MAP.get(model_type)
    if hook_fn is None:
        raise ValueError(f"No steering hook for model_type '{model_type}'")

    for layer_idx, block in enumerate(moe_blocks):
        block._steering_bias = _compute_bias(layer_idx, num_experts, config)
        original_cls = type(block)
        block._steering_original_cls = original_cls
        hooked_cls = type(
            f"_Steered_{original_cls.__name__}",
            (original_cls,),
            {"__call__": hook_fn},
        )
        block.__class__ = hooked_cls


def remove_steering_hooks(moe_blocks: List) -> None:
    """Remove steering hooks, restoring original class."""
    for block in moe_blocks:
        if hasattr(block, "_steering_original_cls"):
            block.__class__ = block._steering_original_cls
            delattr(block, "_steering_original_cls")
        if hasattr(block, "_steering_bias"):
            delattr(block, "_steering_bias")


def update_steering_config(
    moe_blocks: List,
    config: SteeringConfig,
    num_experts: int,
) -> None:
    """Update steering biases without reinstalling hooks.

    Thread-safe: mx.array attribute assignment is atomic at the GIL level.
    """
    for layer_idx, block in enumerate(moe_blocks):
        block._steering_bias = _compute_bias(layer_idx, num_experts, config)
