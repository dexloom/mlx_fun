"""Tests for adapter factory and model-specific adapters."""

import pytest
import mlx.core as mx
import mlx.nn as nn

from mlx_fun.adapters import get_adapter, MiniMaxAdapter, GLM4MoEAdapter, Qwen3MoEAdapter
from mlx_fun.adapters.glm4_moe_lite import GLM4MoELiteAdapter


class FakeMiniMaxLayer:
    def __init__(self, moe_block):
        self.block_sparse_moe = moe_block


class FakeMiniMaxModel:
    def __init__(self, n_layers, moe_block_factory):
        self.layers = [FakeMiniMaxLayer(moe_block_factory()) for _ in range(n_layers)]


class FakeModel:
    def __init__(self, inner):
        self.model = inner


class FakeGLM4Layer:
    def __init__(self, mlp):
        self.mlp = mlp


class FakeGLM4InnerModel:
    def __init__(self, n_layers, moe_factory, dense_factory, first_k_dense):
        self.layers = []
        for i in range(n_layers):
            if i >= first_k_dense:
                self.layers.append(FakeGLM4Layer(moe_factory()))
            else:
                self.layers.append(FakeGLM4Layer(dense_factory()))


def test_get_adapter_minimax(tiny_minimax_moe):
    config = {"model_type": "minimax", "num_local_experts": 4,
              "num_experts_per_tok": 2, "num_hidden_layers": 2}
    inner = FakeMiniMaxModel(2, lambda: tiny_minimax_moe)
    model = FakeModel(inner)

    adapter = get_adapter(model, config)
    assert isinstance(adapter, MiniMaxAdapter)
    assert adapter.num_routed_experts() == 4
    assert adapter.num_experts_per_tok() == 2
    assert adapter.config_expert_count_key() == "num_local_experts"
    assert adapter.moe_layer_indices() == [0, 1]


def test_get_adapter_minimax_m2(tiny_minimax_moe):
    """minimax_m2 model_type uses MiniMaxAdapter."""
    config = {"model_type": "minimax_m2", "num_local_experts": 4,
              "num_experts_per_tok": 2, "num_hidden_layers": 2}
    inner = FakeMiniMaxModel(2, lambda: tiny_minimax_moe)
    model = FakeModel(inner)

    adapter = get_adapter(model, config)
    assert isinstance(adapter, MiniMaxAdapter)
    assert adapter.num_routed_experts() == 4
    assert adapter.num_experts_per_tok() == 2
    assert adapter.config_expert_count_key() == "num_local_experts"
    assert adapter.moe_layer_indices() == [0, 1]


def test_get_adapter_glm4(tiny_glm4_moe):
    config = {"model_type": "glm4_moe", "n_routed_experts": 4,
              "num_experts_per_tok": 2, "num_hidden_layers": 4,
              "first_k_dense_replace": 2}
    inner = FakeGLM4InnerModel(
        4,
        moe_factory=lambda: tiny_glm4_moe,
        dense_factory=lambda: nn.Linear(32, 32),
        first_k_dense=2,
    )
    model = FakeModel(inner)

    adapter = get_adapter(model, config)
    assert isinstance(adapter, GLM4MoEAdapter)
    assert adapter.num_routed_experts() == 4
    assert adapter.moe_layer_indices() == [2, 3]
    assert adapter.config_expert_count_key() == "n_routed_experts"


class FakeGLM4LiteInnerModel:
    def __init__(self, n_layers, moe_factory, dense_factory, first_k_dense, moe_freq):
        self.layers = []
        for i in range(n_layers):
            if i >= first_k_dense and i % moe_freq == 0:
                self.layers.append(FakeGLM4Layer(moe_factory()))
            else:
                self.layers.append(FakeGLM4Layer(dense_factory()))


def test_get_adapter_glm4_moe_lite(tiny_glm4_moe):
    """glm4_moe_lite adapter respects moe_layer_freq."""
    config = {"model_type": "glm4_moe_lite", "n_routed_experts": 4,
              "num_experts_per_tok": 2, "num_hidden_layers": 8,
              "first_k_dense_replace": 1, "moe_layer_freq": 2}
    inner = FakeGLM4LiteInnerModel(
        8,
        moe_factory=lambda: tiny_glm4_moe,
        dense_factory=lambda: nn.Linear(32, 32),
        first_k_dense=1, moe_freq=2,
    )
    model = FakeModel(inner)

    adapter = get_adapter(model, config)
    assert isinstance(adapter, GLM4MoELiteAdapter)
    assert adapter.num_routed_experts() == 4
    assert adapter.num_experts_per_tok() == 2
    assert adapter.config_expert_count_key() == "n_routed_experts"
    # layer_idx >= 1 AND layer_idx % 2 == 0 → layers 2, 4, 6
    assert adapter.moe_layer_indices() == [2, 4, 6]


def test_glm4_moe_lite_freq_1(tiny_glm4_moe):
    """glm4_moe_lite with moe_layer_freq=1 is every eligible layer."""
    config = {"model_type": "glm4_moe_lite", "n_routed_experts": 4,
              "num_experts_per_tok": 2, "num_hidden_layers": 4,
              "first_k_dense_replace": 1, "moe_layer_freq": 1}
    inner = FakeGLM4LiteInnerModel(
        4,
        moe_factory=lambda: tiny_glm4_moe,
        dense_factory=lambda: nn.Linear(32, 32),
        first_k_dense=1, moe_freq=1,
    )
    model = FakeModel(inner)
    adapter = get_adapter(model, config)
    assert adapter.moe_layer_indices() == [1, 2, 3]


def test_get_adapter_unknown():
    with pytest.raises(ValueError, match="Unsupported model_type"):
        get_adapter(None, {"model_type": "unknown"})


def test_minimax_get_moe_block(tiny_minimax_moe):
    config = {"model_type": "minimax", "num_local_experts": 4,
              "num_experts_per_tok": 2, "num_hidden_layers": 1}
    inner = FakeMiniMaxModel(1, lambda: tiny_minimax_moe)
    model = FakeModel(inner)
    adapter = get_adapter(model, config)

    block = adapter.get_moe_block(0)
    assert block is tiny_minimax_moe
    assert adapter.get_switch_mlp(block) is block.switch_mlp


def test_glm4_get_moe_block(tiny_glm4_moe):
    config = {"model_type": "glm4_moe", "n_routed_experts": 4,
              "num_experts_per_tok": 2, "num_hidden_layers": 2,
              "first_k_dense_replace": 0}
    inner = FakeGLM4InnerModel(
        2,
        moe_factory=lambda: tiny_glm4_moe,
        dense_factory=lambda: nn.Linear(32, 32),
        first_k_dense=0,
    )
    model = FakeModel(inner)
    adapter = get_adapter(model, config)

    block = adapter.get_moe_block(0)
    assert adapter.get_switch_mlp(block) is block.switch_mlp


# --- Qwen3 MoE ---

class FakeQwen3Layer:
    def __init__(self, mlp):
        self.mlp = mlp


class FakeQwen3InnerModel:
    def __init__(self, n_layers, moe_factory, dense_factory, sparse_step, mlp_only):
        self.layers = []
        for i in range(n_layers):
            is_moe = (i not in mlp_only) and (i + 1) % sparse_step == 0
            if is_moe:
                self.layers.append(FakeQwen3Layer(moe_factory()))
            else:
                self.layers.append(FakeQwen3Layer(dense_factory()))


def test_get_adapter_qwen3(tiny_qwen3_moe):
    config = {"model_type": "qwen3_moe", "num_experts": 4,
              "num_experts_per_tok": 2, "num_hidden_layers": 6,
              "decoder_sparse_step": 2, "mlp_only_layers": []}
    inner = FakeQwen3InnerModel(
        6,
        moe_factory=lambda: tiny_qwen3_moe,
        dense_factory=lambda: nn.Linear(32, 32),
        sparse_step=2, mlp_only=[],
    )
    model = FakeModel(inner)

    adapter = get_adapter(model, config)
    assert isinstance(adapter, Qwen3MoEAdapter)
    assert adapter.num_routed_experts() == 4
    assert adapter.num_experts_per_tok() == 2
    assert adapter.config_expert_count_key() == "num_experts"
    # Layers where (i+1) % 2 == 0: indices 1, 3, 5
    assert adapter.moe_layer_indices() == [1, 3, 5]


def test_qwen3_moe_layer_indices_with_mlp_only(tiny_qwen3_moe):
    config = {"model_type": "qwen3_moe", "num_experts": 4,
              "num_experts_per_tok": 2, "num_hidden_layers": 6,
              "decoder_sparse_step": 2, "mlp_only_layers": [1, 3]}
    inner = FakeQwen3InnerModel(
        6,
        moe_factory=lambda: tiny_qwen3_moe,
        dense_factory=lambda: nn.Linear(32, 32),
        sparse_step=2, mlp_only=[1, 3],
    )
    model = FakeModel(inner)
    adapter = get_adapter(model, config)
    # Layer 1 and 3 excluded by mlp_only, only layer 5 remains
    assert adapter.moe_layer_indices() == [5]


def test_qwen3_get_moe_block(tiny_qwen3_moe):
    config = {"model_type": "qwen3_moe", "num_experts": 4,
              "num_experts_per_tok": 2, "num_hidden_layers": 2,
              "decoder_sparse_step": 1, "mlp_only_layers": []}
    inner = FakeQwen3InnerModel(
        2,
        moe_factory=lambda: tiny_qwen3_moe,
        dense_factory=lambda: nn.Linear(32, 32),
        sparse_step=1, mlp_only=[],
    )
    model = FakeModel(inner)
    adapter = get_adapter(model, config)

    block = adapter.get_moe_block(0)
    assert adapter.get_switch_mlp(block) is block.switch_mlp


# --- Qwen3 Next ---

def test_get_adapter_qwen3_next(tiny_qwen3_next_moe):
    """qwen3_next model_type uses Qwen3MoEAdapter."""
    config = {"model_type": "qwen3_next", "num_experts": 4,
              "num_experts_per_tok": 2, "num_hidden_layers": 6,
              "decoder_sparse_step": 2, "mlp_only_layers": []}
    inner = FakeQwen3InnerModel(
        6,
        moe_factory=lambda: tiny_qwen3_next_moe,
        dense_factory=lambda: nn.Linear(32, 32),
        sparse_step=2, mlp_only=[],
    )
    model = FakeModel(inner)

    adapter = get_adapter(model, config)
    assert isinstance(adapter, Qwen3MoEAdapter)
    assert adapter.num_routed_experts() == 4
    assert adapter.num_experts_per_tok() == 2
    assert adapter.config_expert_count_key() == "num_experts"
    assert adapter.moe_layer_indices() == [1, 3, 5]
