"""Tiny MoE fixtures for unit tests."""

import pytest
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.switch_layers import SwitchGLU, SwitchLinear


class TinyMiniMaxMoE(nn.Module):
    """Minimal replica of MiniMaxSparseMoeBlock for testing."""

    def __init__(self, hidden=32, intermediate=64, n_experts=4, top_k=2):
        super().__init__()
        self.num_experts_per_tok = top_k
        self.gate = nn.Linear(hidden, n_experts, bias=False)
        self.switch_mlp = SwitchGLU(hidden, intermediate, n_experts)
        self.e_score_correction_bias = mx.zeros((n_experts,))

    def __call__(self, x):
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
        y = (y * scores[..., None]).sum(axis=-2)
        return y


class TinyGLM4Gate(nn.Module):
    """Minimal replica of MoEGate for testing."""

    def __init__(self, hidden=32, n_experts=4, top_k=2):
        super().__init__()
        self.top_k = top_k
        self.n_routed_experts = n_experts
        self.weight = mx.zeros((n_experts, hidden))
        self.e_score_correction_bias = mx.zeros((n_experts,))

    def __call__(self, x):
        gates = x @ self.weight.T
        scores = mx.sigmoid(gates.astype(mx.float32))
        orig_scores = scores
        scores = scores + self.e_score_correction_bias
        k = self.top_k
        inds = mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]
        scores = mx.take_along_axis(orig_scores, inds, axis=-1)
        scores = scores / (mx.sum(scores, axis=-1, keepdims=True) + 1e-20)
        return inds, scores


class TinyGLM4MoE(nn.Module):
    """Minimal replica of GLM4 MoE block for testing."""

    def __init__(self, hidden=32, intermediate=64, n_experts=4, top_k=2):
        super().__init__()
        self.num_experts_per_tok = top_k
        self.gate = TinyGLM4Gate(hidden, n_experts, top_k)
        self.switch_mlp = SwitchGLU(hidden, intermediate, n_experts)
        self.shared_experts = None
        self.sharding_group = None

    def __call__(self, x):
        inds, scores = self.gate(x)
        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2).astype(y.dtype)
        return y


class TinyQwen3MoE(nn.Module):
    """Minimal replica of Qwen3MoeSparseMoeBlock for testing."""

    def __init__(self, hidden=32, intermediate=64, n_experts=4, top_k=2):
        super().__init__()
        self.num_experts = n_experts
        self.top_k = top_k
        self.norm_topk_prob = True
        self.gate = nn.Linear(hidden, n_experts, bias=False)
        self.switch_mlp = SwitchGLU(hidden, intermediate, n_experts)

    def __call__(self, x):
        gates = self.gate(x)
        gates = mx.softmax(gates, axis=-1, precise=True)
        k = self.top_k
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if self.norm_topk_prob:
            scores = scores / mx.sum(scores, axis=-1, keepdims=True)
        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2)
        return y


class TinyQwen3NextMoE(nn.Module):
    """Minimal replica of Qwen3NextSparseMoeBlock for testing."""

    def __init__(self, hidden=32, intermediate=64, n_experts=4, top_k=2):
        super().__init__()
        self.num_experts = n_experts
        self.top_k = top_k
        self.norm_topk_prob = False
        self.gate = nn.Linear(hidden, n_experts, bias=False)
        self.switch_mlp = SwitchGLU(hidden, intermediate, n_experts)
        self.shared_expert = nn.Linear(hidden, hidden)
        self.shared_expert_gate = nn.Linear(hidden, 1, bias=False)

    def __call__(self, x):
        gates = self.gate(x)
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


@pytest.fixture
def tiny_minimax_moe():
    """Create a tiny MiniMax-style MoE block."""
    mx.random.seed(42)
    block = TinyMiniMaxMoE(hidden=32, intermediate=64, n_experts=4, top_k=2)
    mx.eval(block.parameters())
    return block


@pytest.fixture
def tiny_glm4_moe():
    """Create a tiny GLM4-style MoE block."""
    mx.random.seed(42)
    block = TinyGLM4MoE(hidden=32, intermediate=64, n_experts=4, top_k=2)
    # Initialize gate weight with random values
    block.gate.weight = mx.random.normal((4, 32)) * 0.1
    mx.eval(block.parameters())
    return block


@pytest.fixture
def tiny_qwen3_moe():
    """Create a tiny Qwen3-style MoE block."""
    mx.random.seed(42)
    block = TinyQwen3MoE(hidden=32, intermediate=64, n_experts=4, top_k=2)
    mx.eval(block.parameters())
    return block


@pytest.fixture
def tiny_qwen3_next_moe():
    """Create a tiny Qwen3Next-style MoE block with shared expert."""
    mx.random.seed(42)
    block = TinyQwen3NextMoE(hidden=32, intermediate=64, n_experts=4, top_k=2)
    mx.eval(block.parameters())
    return block


@pytest.fixture
def sample_input():
    """Sample input tensor: (batch=1, seq=8, hidden=32)."""
    mx.random.seed(0)
    x = mx.random.normal((1, 8, 32))
    mx.eval(x)
    return x
