"""REAP saliency accumulator and computation."""

from typing import Dict, List, Tuple

import numpy as np


class SaliencyAccumulator:
    """Per-layer, per-expert accumulator for REAP saliency scores.

    Uses numpy float64 for numerical stability. Vectorized with np.add.at()
    for scatter-add operations.
    """

    def __init__(self, num_layers: int, num_experts: int):
        self.num_layers = num_layers
        self.num_experts = num_experts

        # Per-layer accumulators: shape (num_experts,)
        self.reap_sum = np.zeros((num_layers, num_experts), dtype=np.float64)
        self.reap_count = np.zeros((num_layers, num_experts), dtype=np.float64)
        self.ean_sum = np.zeros((num_layers, num_experts), dtype=np.float64)
        self.freq = np.zeros((num_layers, num_experts), dtype=np.float64)
        self.weighted_freq_sum = np.zeros((num_layers, num_experts), dtype=np.float64)

    def update(
        self,
        layer_idx: int,
        expert_indices: np.ndarray,
        router_weights: np.ndarray,
        activation_norms: np.ndarray,
    ):
        """Accumulate saliency statistics for one batch at one layer.

        Args:
            layer_idx: Which decoder layer.
            expert_indices: (batch*seq, top_k) int array of selected experts.
            router_weights: (batch*seq, top_k) float array of router scores.
            activation_norms: (batch*seq, top_k) float array of ||expert_output||.
        """
        # Flatten to 1D for scatter-add
        flat_inds = expert_indices.ravel().astype(np.intp)
        flat_weights = router_weights.ravel().astype(np.float64)
        flat_norms = activation_norms.ravel().astype(np.float64)

        reap_values = flat_norms * flat_weights

        np.add.at(self.reap_sum[layer_idx], flat_inds, reap_values)
        np.add.at(self.reap_count[layer_idx], flat_inds, 1.0)
        np.add.at(self.ean_sum[layer_idx], flat_inds, flat_norms)
        np.add.at(self.freq[layer_idx], flat_inds, 1.0)
        np.add.at(self.weighted_freq_sum[layer_idx], flat_inds, flat_weights)

    def compute_scores(self, metric: str = "reap") -> np.ndarray:
        """Compute final saliency scores.

        Args:
            metric: One of 'reap', 'ean', 'freq', 'weighted_freq'.

        Returns:
            (num_layers, num_experts) array of scores. Higher = more important.
        """
        if metric == "reap":
            with np.errstate(divide="ignore", invalid="ignore"):
                scores = np.where(
                    self.reap_count > 0,
                    self.reap_sum / self.reap_count,
                    0.0,
                )
        elif metric == "ean":
            with np.errstate(divide="ignore", invalid="ignore"):
                scores = np.where(
                    self.freq > 0,
                    self.ean_sum / self.freq,
                    0.0,
                )
        elif metric == "freq":
            scores = self.freq.copy()
        elif metric == "weighted_freq":
            scores = self.weighted_freq_sum.copy()
        else:
            raise ValueError(f"Unknown metric '{metric}'. Use: reap, ean, freq, weighted_freq")
        return scores

    def save(self, path: str):
        """Save accumulator state to .npz file."""
        np.savez(
            path,
            reap_sum=self.reap_sum,
            reap_count=self.reap_count,
            ean_sum=self.ean_sum,
            freq=self.freq,
            weighted_freq_sum=self.weighted_freq_sum,
            num_layers=np.array(self.num_layers),
            num_experts=np.array(self.num_experts),
        )

    @classmethod
    def load(cls, path: str) -> "SaliencyAccumulator":
        """Load accumulator state from .npz file."""
        data = np.load(path)
        num_layers = int(data["num_layers"])
        num_experts = int(data["num_experts"])
        acc = cls(num_layers, num_experts)
        acc.reap_sum = data["reap_sum"]
        acc.reap_count = data["reap_count"]
        acc.ean_sum = data["ean_sum"]
        acc.freq = data["freq"]
        acc.weighted_freq_sum = data["weighted_freq_sum"]
        return acc
