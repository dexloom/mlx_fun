"""SAFEx-style safety-critical expert analysis.

Compares expert routing patterns between harmful and benign datasets
to identify safety-critical experts (HCDG and HRCG groups).

Based on:
- SAFEx (NeurIPS 2025): Stable safety-critical expert identification
- SteerMoE: Expert (de)activation frequency comparison
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Top-k routing replication (model-type-specific)
# ---------------------------------------------------------------------------

def compute_top_k_from_logits(
    gate_logits: np.ndarray,
    model_type: str,
    top_k: int,
) -> np.ndarray:
    """Compute top-k expert indices from raw gate logits.

    Replicates each model's routing logic in numpy to derive which experts
    would be selected, without running the actual model forward.

    Args:
        gate_logits: (n_tokens, num_experts) raw gate logits.
        model_type: Model type string.
        top_k: Number of experts per token.

    Returns:
        (n_tokens, top_k) int array of selected expert indices.
    """
    if model_type in ("minimax", "minimax_m2", "glm4_moe", "glm4_moe_lite",
                       "glm_moe_dsa", "deepseek_v32"):
        # Sigmoid activation, select top-k by descending score
        scores = 1.0 / (1.0 + np.exp(-gate_logits.astype(np.float64)))
        # argpartition for top-k (highest scores = lowest negated scores)
        inds = np.argpartition(-scores, kth=top_k - 1, axis=-1)[..., :top_k]
    elif model_type in ("qwen3_moe", "qwen3_next"):
        # Softmax activation, select top-k by descending probability
        logits = gate_logits.astype(np.float64)
        logits_max = logits.max(axis=-1, keepdims=True)
        exp_logits = np.exp(logits - logits_max)
        probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
        inds = np.argpartition(-probs, kth=top_k - 1, axis=-1)[..., :top_k]
    else:
        raise ValueError(f"Unknown model_type '{model_type}' for top-k computation")

    return inds.astype(np.intp)


# ---------------------------------------------------------------------------
# Differential accumulator
# ---------------------------------------------------------------------------

class DifferentialAccumulator:
    """Track gate logit statistics separately for harmful and benign datasets.

    Uses numpy float64 for numerical stability, following the
    SaliencyAccumulator pattern.
    """

    def __init__(self, num_layers: int, num_experts: int):
        self.num_layers = num_layers
        self.num_experts = num_experts

        # Gate logit sums: shape (num_layers, num_experts)
        self.harmful_gate_sum = np.zeros((num_layers, num_experts), dtype=np.float64)
        self.harmful_gate_count = np.zeros((num_layers, num_experts), dtype=np.float64)
        self.benign_gate_sum = np.zeros((num_layers, num_experts), dtype=np.float64)
        self.benign_gate_count = np.zeros((num_layers, num_experts), dtype=np.float64)

        # Selection frequency (how often each expert is in top-k)
        self.harmful_freq = np.zeros((num_layers, num_experts), dtype=np.float64)
        self.benign_freq = np.zeros((num_layers, num_experts), dtype=np.float64)

        # Token counts for normalization
        self.harmful_tokens = np.zeros(num_layers, dtype=np.float64)
        self.benign_tokens = np.zeros(num_layers, dtype=np.float64)

    def update_from_gate_logits(
        self,
        layer_idx: int,
        gate_logits: np.ndarray,
        dataset: str,
    ):
        """Accumulate full gate logit statistics for one batch at one layer.

        Args:
            layer_idx: Which accumulator layer index.
            gate_logits: (n_tokens, num_experts) raw gate logits.
            dataset: "harmful" or "benign".
        """
        n_tokens = gate_logits.shape[0]
        logits_f64 = gate_logits.astype(np.float64)

        if dataset == "harmful":
            self.harmful_gate_sum[layer_idx] += logits_f64.sum(axis=0)
            self.harmful_gate_count[layer_idx] += n_tokens
            self.harmful_tokens[layer_idx] += n_tokens
        else:
            self.benign_gate_sum[layer_idx] += logits_f64.sum(axis=0)
            self.benign_gate_count[layer_idx] += n_tokens
            self.benign_tokens[layer_idx] += n_tokens

    def update_from_top_k(
        self,
        layer_idx: int,
        expert_indices: np.ndarray,
        dataset: str,
    ):
        """Accumulate top-k selection frequency.

        Args:
            layer_idx: Which accumulator layer index.
            expert_indices: (n_tokens, top_k) selected expert indices.
            dataset: "harmful" or "benign".
        """
        flat_inds = expert_indices.ravel().astype(np.intp)
        if dataset == "harmful":
            np.add.at(self.harmful_freq[layer_idx], flat_inds, 1.0)
        else:
            np.add.at(self.benign_freq[layer_idx], flat_inds, 1.0)


# ---------------------------------------------------------------------------
# Differential scoring
# ---------------------------------------------------------------------------

def _layer_normalize(arr: np.ndarray) -> np.ndarray:
    """Normalize each layer's scores to [0, 1] range."""
    result = np.zeros_like(arr)
    for i in range(arr.shape[0]):
        rng = arr[i].max() - arr[i].min()
        if rng > 1e-12:
            result[i] = (arr[i] - arr[i].min()) / rng
    return result


def compute_differential_scores(
    acc: DifferentialAccumulator,
    freq_weight: float = 0.5,
    activation_weight: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-expert differential safety scores.

    Args:
        acc: Populated DifferentialAccumulator.
        freq_weight: Weight for frequency-based differential.
        activation_weight: Weight for activation-based differential.

    Returns:
        (diff_freq, diff_activation, composite) each (num_layers, num_experts).
        Positive values = expert activated MORE on harmful data.
    """
    # Normalized frequencies: selections per token
    with np.errstate(divide="ignore", invalid="ignore"):
        harmful_freq_norm = np.where(
            acc.harmful_tokens[:, None] > 0,
            acc.harmful_freq / acc.harmful_tokens[:, None],
            0.0,
        )
        benign_freq_norm = np.where(
            acc.benign_tokens[:, None] > 0,
            acc.benign_freq / acc.benign_tokens[:, None],
            0.0,
        )

    diff_freq = harmful_freq_norm - benign_freq_norm

    # Mean gate logits per expert
    with np.errstate(divide="ignore", invalid="ignore"):
        harmful_mean_gate = np.where(
            acc.harmful_gate_count > 0,
            acc.harmful_gate_sum / acc.harmful_gate_count,
            0.0,
        )
        benign_mean_gate = np.where(
            acc.benign_gate_count > 0,
            acc.benign_gate_sum / acc.benign_gate_count,
            0.0,
        )

    diff_activation = harmful_mean_gate - benign_mean_gate

    # Combine with per-layer normalization
    composite = (
        freq_weight * _layer_normalize(diff_freq)
        + activation_weight * _layer_normalize(diff_activation)
    )

    return diff_freq, diff_activation, composite


# ---------------------------------------------------------------------------
# Safety expert classification
# ---------------------------------------------------------------------------

@dataclass
class SafetyReport:
    """Results of a differential safety scan."""

    num_layers: int
    num_experts: int
    threshold_percentile: float
    differential_freq: np.ndarray
    differential_activation: np.ndarray
    composite_score: np.ndarray
    hcdg_experts: Dict[int, List[int]] = field(default_factory=dict)
    hrcg_experts: Dict[int, List[int]] = field(default_factory=dict)
    safety_critical: Dict[int, List[int]] = field(default_factory=dict)

    def save(self, path: str):
        """Save report as JSON."""
        data = {
            "num_layers": self.num_layers,
            "num_experts": self.num_experts,
            "threshold_percentile": self.threshold_percentile,
            "differential_freq": self.differential_freq.tolist(),
            "differential_activation": self.differential_activation.tolist(),
            "composite_score": self.composite_score.tolist(),
            "hcdg_experts": {str(k): v for k, v in self.hcdg_experts.items()},
            "hrcg_experts": {str(k): v for k, v in self.hrcg_experts.items()},
            "safety_critical": {str(k): v for k, v in self.safety_critical.items()},
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "SafetyReport":
        """Load report from JSON."""
        with open(path) as f:
            data = json.load(f)
        return cls(
            num_layers=data["num_layers"],
            num_experts=data["num_experts"],
            threshold_percentile=data["threshold_percentile"],
            differential_freq=np.array(data["differential_freq"]),
            differential_activation=np.array(data["differential_activation"]),
            composite_score=np.array(data["composite_score"]),
            hcdg_experts={int(k): v for k, v in data["hcdg_experts"].items()},
            hrcg_experts={int(k): v for k, v in data["hrcg_experts"].items()},
            safety_critical={int(k): v for k, v in data["safety_critical"].items()},
        )


def identify_safety_experts(
    diff_freq: np.ndarray,
    diff_activation: np.ndarray,
    composite_score: np.ndarray,
    threshold_percentile: float = 90.0,
) -> SafetyReport:
    """Classify experts into HCDG and HRCG groups.

    HCDG (Harmful Content Detection Group): High composite score —
    experts that activate MORE on harmful content.

    HRCG (Harmful Response Control Group): Strongly negative composite —
    experts that activate MORE on benign content and suppress harmful outputs.

    Args:
        diff_freq: (num_layers, num_experts) frequency differential.
        diff_activation: (num_layers, num_experts) activation differential.
        composite_score: (num_layers, num_experts) composite.
        threshold_percentile: Percentile threshold for classification.

    Returns:
        SafetyReport with classified experts.
    """
    num_layers, num_experts = composite_score.shape
    hcdg = {}
    hrcg = {}
    safety = {}

    high_threshold = np.percentile(composite_score, threshold_percentile)
    low_threshold = np.percentile(composite_score, 100.0 - threshold_percentile)

    for layer_idx in range(num_layers):
        layer_scores = composite_score[layer_idx]

        layer_hcdg = [
            int(e) for e in range(num_experts) if layer_scores[e] >= high_threshold
        ]
        layer_hrcg = [
            int(e) for e in range(num_experts) if layer_scores[e] <= low_threshold
        ]
        layer_safety = sorted(set(layer_hcdg) | set(layer_hrcg))

        if layer_hcdg:
            hcdg[layer_idx] = layer_hcdg
        if layer_hrcg:
            hrcg[layer_idx] = layer_hrcg
        if layer_safety:
            safety[layer_idx] = layer_safety

    return SafetyReport(
        num_layers=num_layers,
        num_experts=num_experts,
        threshold_percentile=threshold_percentile,
        differential_freq=diff_freq,
        differential_activation=diff_activation,
        composite_score=composite_score,
        hcdg_experts=hcdg,
        hrcg_experts=hrcg,
        safety_critical=safety,
    )
