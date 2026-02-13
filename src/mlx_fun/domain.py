"""Domain-specific expert identification and gate amplification.

Compares expert routing patterns between domain-specific data and general data
to identify domain-specialized experts. Supports permanent gate amplification
so domain experts are favored natively without runtime hooks.

Reuses differential analysis infrastructure from safety.py with domain-appropriate
naming: "harmful"="domain", "benign"="general", positive diff = domain-preferred.
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np


# ---------------------------------------------------------------------------
# Domain expert classification
# ---------------------------------------------------------------------------

@dataclass
class DomainReport:
    """Results of a differential domain scan."""

    domain_name: str
    num_layers: int
    num_experts: int
    threshold_percentile: float
    differential_freq: np.ndarray
    differential_activation: np.ndarray
    composite_score: np.ndarray
    domain_experts: Dict[int, List[int]] = field(default_factory=dict)
    general_experts: Dict[int, List[int]] = field(default_factory=dict)

    def save(self, path: str):
        """Save report as JSON."""
        data = {
            "domain_name": self.domain_name,
            "num_layers": self.num_layers,
            "num_experts": self.num_experts,
            "threshold_percentile": self.threshold_percentile,
            "differential_freq": self.differential_freq.tolist(),
            "differential_activation": self.differential_activation.tolist(),
            "composite_score": self.composite_score.tolist(),
            "domain_experts": {str(k): v for k, v in self.domain_experts.items()},
            "general_experts": {str(k): v for k, v in self.general_experts.items()},
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "DomainReport":
        """Load report from JSON."""
        with open(path) as f:
            data = json.load(f)
        return cls(
            domain_name=data["domain_name"],
            num_layers=data["num_layers"],
            num_experts=data["num_experts"],
            threshold_percentile=data["threshold_percentile"],
            differential_freq=np.array(data["differential_freq"]),
            differential_activation=np.array(data["differential_activation"]),
            composite_score=np.array(data["composite_score"]),
            domain_experts={int(k): v for k, v in data["domain_experts"].items()},
            general_experts={int(k): v for k, v in data["general_experts"].items()},
        )


def identify_domain_experts(
    diff_freq: np.ndarray,
    diff_activation: np.ndarray,
    composite_score: np.ndarray,
    domain_name: str,
    threshold_percentile: float = 90.0,
) -> DomainReport:
    """Classify experts into domain-specialized and general groups.

    domain_experts: High composite score — experts activated MORE on domain data.
    general_experts: Low composite score — experts activated MORE on general data.

    Args:
        diff_freq: (num_layers, num_experts) frequency differential.
        diff_activation: (num_layers, num_experts) activation differential.
        composite_score: (num_layers, num_experts) composite.
        domain_name: Name of the domain (e.g. "solidity", "medical").
        threshold_percentile: Percentile threshold for classification.

    Returns:
        DomainReport with classified experts.
    """
    num_layers, num_experts = composite_score.shape
    domain = {}
    general = {}

    high_threshold = np.percentile(composite_score, threshold_percentile)
    low_threshold = np.percentile(composite_score, 100.0 - threshold_percentile)

    for layer_idx in range(num_layers):
        layer_scores = composite_score[layer_idx]

        layer_domain = [
            int(e) for e in range(num_experts) if layer_scores[e] >= high_threshold
        ]
        layer_general = [
            int(e) for e in range(num_experts) if layer_scores[e] <= low_threshold
        ]

        if layer_domain:
            domain[layer_idx] = layer_domain
        if layer_general:
            general[layer_idx] = layer_general

    return DomainReport(
        domain_name=domain_name,
        num_layers=num_layers,
        num_experts=num_experts,
        threshold_percentile=threshold_percentile,
        differential_freq=diff_freq,
        differential_activation=diff_activation,
        composite_score=composite_score,
        domain_experts=domain,
        general_experts=general,
    )


# ---------------------------------------------------------------------------
# Amplification bias computation
# ---------------------------------------------------------------------------

def compute_amplification_biases(
    report: DomainReport,
    scale: float = 1.0,
    threshold: float = 0.0,
) -> Dict[int, np.ndarray]:
    """Compute per-layer gate bias arrays for domain expert amplification.

    For each domain expert: boost = scale * max(0, composite_score - threshold).
    Returns only layers with at least one nonzero boost value.

    Args:
        report: DomainReport with domain experts and composite scores.
        scale: Amplification strength multiplier.
        threshold: Minimum composite score to amplify.

    Returns:
        Dict mapping layer_idx -> (num_experts,) float64 bias array.
    """
    biases = {}
    for layer_idx, expert_ids in report.domain_experts.items():
        bias = np.zeros(report.num_experts, dtype=np.float64)
        for eid in expert_ids:
            score = report.composite_score[layer_idx, eid]
            boost = scale * max(0.0, score - threshold)
            bias[eid] = boost
        if np.any(bias != 0):
            biases[layer_idx] = bias
    return biases


# ---------------------------------------------------------------------------
# Gate amplification (permanent weight modification)
# ---------------------------------------------------------------------------

def amplify_gate_weights(
    moe_blocks: list,
    model_type: str,
    biases: Dict[int, np.ndarray],
) -> None:
    """Permanently modify gate parameters to amplify domain experts.

    Per model type:
    - MiniMax/MiniMax-M2: Set gate.bias on nn.Linear (bias=False -> bias=True).
    - GLM4/GLM4-Lite/GLM5/DSv3: Add to gate.e_score_correction_bias.
    - Qwen3/Qwen3-Next: Set gate.bias on nn.Linear.

    Args:
        moe_blocks: List of MoE nn.Module instances (indexed by accumulator layer).
        model_type: Model type string.
        biases: Dict mapping layer_idx -> (num_experts,) bias array.
    """
    for layer_idx, bias_array in biases.items():
        if layer_idx >= len(moe_blocks):
            continue
        block = moe_blocks[layer_idx]
        bias_mx = mx.array(bias_array.astype(np.float32))

        if model_type in ("minimax", "minimax_m2"):
            # nn.Linear gate: set bias attribute (adds to pre-sigmoid logits)
            if "bias" in block.gate:
                block.gate.bias = block.gate.bias + bias_mx
            else:
                block.gate.bias = bias_mx

        elif model_type in ("glm4_moe", "glm4_moe_lite", "glm_moe_dsa", "deepseek_v32"):
            # Custom MoEGate: add to e_score_correction_bias (post-sigmoid)
            block.gate.e_score_correction_bias = (
                block.gate.e_score_correction_bias + bias_mx
            )

        elif model_type in ("qwen3_moe", "qwen3_next"):
            # nn.Linear gate: set bias attribute (adds to pre-softmax logits)
            if "bias" in block.gate:
                block.gate.bias = block.gate.bias + bias_mx
            else:
                block.gate.bias = bias_mx

        else:
            raise ValueError(f"No amplification support for model_type '{model_type}'")
