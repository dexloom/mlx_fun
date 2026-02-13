"""Tests for pruning engine."""

import mlx.core as mx
import numpy as np
import pytest

from mlx_fun.pruner import (
    select_experts_to_keep,
    select_experts_to_keep_strided,
    _strided_prune_indices,
    prune_moe_layer,
    prune_model,
)
from mlx_fun.adapters.minimax import MiniMaxAdapter
from mlx_fun.adapters.glm4_moe import GLM4MoEAdapter
from mlx_fun.adapters.qwen3_moe import Qwen3MoEAdapter


# --- Fixtures mimicking adapter access ---

class FakeMiniMaxLayer:
    def __init__(self, moe_block):
        self.block_sparse_moe = moe_block

class FakeMiniMaxInner:
    def __init__(self, layers):
        self.layers = layers

class FakeModel:
    def __init__(self, inner):
        self.model = inner


class FakeGLM4Layer:
    def __init__(self, mlp):
        self.mlp = mlp

class FakeGLM4Inner:
    def __init__(self, layers):
        self.layers = layers


# --- Expert selection tests ---

def test_select_keep_all():
    """n_prune=0 keeps all experts."""
    scores = np.array([[1.0, 2.0, 3.0, 4.0]])
    keep_map = select_experts_to_keep(scores, n_prune=0)
    np.testing.assert_array_equal(keep_map[0], [0, 1, 2, 3])


def test_select_prune_lowest():
    """Prune the 2 lowest-scoring experts."""
    scores = np.array([[10.0, 1.0, 5.0, 2.0]])
    keep_map = select_experts_to_keep(scores, n_prune=2)
    # Experts 1 and 3 have lowest scores (1.0 and 2.0)
    np.testing.assert_array_equal(keep_map[0], [0, 2])


def test_select_multi_layer():
    """Each layer gets independent selection."""
    scores = np.array([
        [10.0, 1.0, 5.0, 2.0],
        [2.0, 10.0, 1.0, 5.0],
    ])
    keep_map = select_experts_to_keep(scores, n_prune=1)
    # Layer 0: lowest is expert 1
    np.testing.assert_array_equal(keep_map[0], [0, 2, 3])
    # Layer 1: lowest is expert 2
    np.testing.assert_array_equal(keep_map[1], [0, 1, 3])


def test_select_prune_too_many():
    """Cannot prune all experts."""
    scores = np.array([[1.0, 2.0]])
    with pytest.raises(ValueError, match="Cannot prune"):
        select_experts_to_keep(scores, n_prune=2)


# --- Strided pruning tests ---

def test_strided_prune_indices_exact():
    """Stride divides evenly: 120/20=6, prune at 5,11,17,..."""
    indices = _strided_prune_indices(120, 20)
    assert len(indices) == 20
    # Every 6th position (0-indexed: 5, 11, 17, ...)
    expected = np.arange(20) * 6 + 5
    np.testing.assert_array_equal(indices, expected)


def test_strided_prune_indices_small():
    """Stride 2: group=4, prune=2 → positions 1, 3."""
    indices = _strided_prune_indices(4, 2)
    assert len(indices) == 2
    np.testing.assert_array_equal(indices, [1, 3])


def test_strided_prune_indices_zero():
    """Prune 0 returns empty."""
    indices = _strided_prune_indices(10, 0)
    assert len(indices) == 0


def test_strided_prune_indices_all():
    """Prune all returns all indices."""
    indices = _strided_prune_indices(4, 4)
    np.testing.assert_array_equal(indices, [0, 1, 2, 3])


def test_strided_select_correct_count():
    """Strided pruning removes exactly n_prune experts."""
    # 8 experts, prune 4: split into 4 unimportant + 4 important
    # Prune 2 from each group
    scores = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]])
    keep_map = select_experts_to_keep_strided(scores, n_prune=4)
    assert len(keep_map[0]) == 4  # 8 - 4 = 4 kept


def test_strided_select_distributes_pruning():
    """Strided pruning removes from both important and unimportant groups."""
    # 8 experts with scores [1..8], prune 4
    # Unimportant group (bottom 4): experts 0,1,2,3 (scores 1,2,3,4)
    # Important group (top 4): experts 4,5,6,7 (scores 5,6,7,8)
    # Prune 2 from each group at intervals
    scores = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]])
    keep_map = select_experts_to_keep_strided(scores, n_prune=4)
    kept = set(keep_map[0].tolist())
    # Some experts from both groups should be kept AND some pruned from both
    unimportant = {0, 1, 2, 3}
    important = {4, 5, 6, 7}
    assert len(kept & unimportant) > 0, "Should keep some unimportant experts"
    assert len(kept & important) > 0, "Should keep some important experts"
    assert len(kept & unimportant) < 4, "Should prune some unimportant experts"
    assert len(kept & important) < 4, "Should prune some important experts"


def test_strided_select_zero_prune():
    """n_prune=0 keeps all."""
    scores = np.array([[1.0, 2.0, 3.0, 4.0]])
    keep_map = select_experts_to_keep_strided(scores, n_prune=0)
    np.testing.assert_array_equal(keep_map[0], [0, 1, 2, 3])


def test_strided_select_prune_too_many():
    """Cannot prune all experts."""
    scores = np.array([[1.0, 2.0]])
    with pytest.raises(ValueError, match="Cannot prune"):
        select_experts_to_keep_strided(scores, n_prune=2)


def test_strided_vs_bottom_different():
    """Strided and bottom strategies produce different results for large enough inputs."""
    np.random.seed(42)
    scores = np.random.rand(1, 16)
    keep_bottom = select_experts_to_keep(scores, n_prune=8)
    keep_strided = select_experts_to_keep_strided(scores, n_prune=8)
    # Both keep 8 experts but different ones
    assert len(keep_bottom[0]) == 8
    assert len(keep_strided[0]) == 8
    assert not np.array_equal(keep_bottom[0], keep_strided[0])


# --- Tensor slicing tests (MiniMax) ---

def test_minimax_prune_shapes(tiny_minimax_moe):
    """After pruning, all tensors have correct shapes."""
    config = {"model_type": "minimax", "num_local_experts": 4,
              "num_experts_per_tok": 2, "num_hidden_layers": 1}
    layer = FakeMiniMaxLayer(tiny_minimax_moe)
    model = FakeModel(FakeMiniMaxInner([layer]))
    adapter = MiniMaxAdapter(model, config)

    keep = np.array([0, 2])  # Keep 2 of 4
    prune_moe_layer(adapter, 0, keep)

    # SwitchGLU projections: expert axis should be 2
    for proj_name in ("gate_proj", "up_proj", "down_proj"):
        proj = getattr(tiny_minimax_moe.switch_mlp, proj_name)
        assert proj.weight.shape[0] == 2, f"{proj_name} expert axis wrong"

    # Gate: (n_experts, hidden) -> (2, 32)
    assert tiny_minimax_moe.gate.weight.shape[0] == 2
    # Bias: (n_experts,) -> (2,)
    assert tiny_minimax_moe.e_score_correction_bias.shape[0] == 2


def test_minimax_quantized_gate_prune():
    """Pruning slices QuantizedLinear gate (weight, scales, biases)."""
    import mlx.nn as nn
    from mlx_lm.models.switch_layers import SwitchGLU

    # Need hidden >= 64 for quantization group_size
    hidden, n_experts = 256, 8
    moe = type("FakeMoE", (), {})()
    moe.gate = nn.Linear(hidden, n_experts, bias=False)
    moe.switch_mlp = SwitchGLU(hidden, 128, n_experts)
    moe.e_score_correction_bias = mx.zeros((n_experts,))
    mx.eval(moe.gate.parameters())

    # Quantize the gate
    moe.gate = nn.QuantizedLinear.from_linear(moe.gate)
    mx.eval(moe.gate.parameters())
    orig_scales_shape = moe.gate.scales.shape
    orig_biases_shape = moe.gate.biases.shape

    config = {"model_type": "minimax", "num_local_experts": n_experts,
              "num_experts_per_tok": 2, "num_hidden_layers": 1}
    layer = FakeMiniMaxLayer(moe)
    model = FakeModel(FakeMiniMaxInner([layer]))
    adapter = MiniMaxAdapter(model, config)

    keep = np.array([0, 2, 5, 7])  # Keep 4 of 8
    prune_moe_layer(adapter, 0, keep)

    assert moe.gate.weight.shape[0] == 4
    assert moe.gate.scales.shape[0] == 4
    assert moe.gate.biases.shape[0] == 4
    # Other dims unchanged
    assert moe.gate.scales.shape[1] == orig_scales_shape[1]
    assert moe.gate.biases.shape[1] == orig_biases_shape[1]


def test_minimax_zero_prune_identity(tiny_minimax_moe, sample_input):
    """Pruning 0 experts should not change the output."""
    mx.eval(tiny_minimax_moe.parameters())
    orig_out = tiny_minimax_moe(sample_input)
    mx.eval(orig_out)
    orig_np = np.array(orig_out, copy=True)

    config = {"model_type": "minimax", "num_local_experts": 4,
              "num_experts_per_tok": 2, "num_hidden_layers": 1}
    layer = FakeMiniMaxLayer(tiny_minimax_moe)
    model = FakeModel(FakeMiniMaxInner([layer]))
    adapter = MiniMaxAdapter(model, config)

    keep = np.array([0, 1, 2, 3])
    prune_moe_layer(adapter, 0, keep)
    mx.eval(tiny_minimax_moe.parameters())

    after_out = tiny_minimax_moe(sample_input)
    mx.eval(after_out)

    np.testing.assert_allclose(
        np.array(after_out, copy=False), orig_np, atol=1e-5,
    )


# --- Tensor slicing tests (GLM4) ---

def test_glm4_prune_shapes(tiny_glm4_moe):
    """After pruning, GLM4 tensors have correct shapes."""
    config = {"model_type": "glm4_moe", "n_routed_experts": 4,
              "num_experts_per_tok": 2, "num_hidden_layers": 1,
              "first_k_dense_replace": 0}
    layer = FakeGLM4Layer(tiny_glm4_moe)
    model = FakeModel(FakeGLM4Inner([layer]))
    adapter = GLM4MoEAdapter(model, config)

    keep = np.array([1, 3])  # Keep 2 of 4
    prune_moe_layer(adapter, 0, keep)

    for proj_name in ("gate_proj", "up_proj", "down_proj"):
        proj = getattr(tiny_glm4_moe.switch_mlp, proj_name)
        assert proj.weight.shape[0] == 2

    # Gate weight: (n_routed_experts, hidden) -> (2, 32)
    assert tiny_glm4_moe.gate.weight.shape[0] == 2
    assert tiny_glm4_moe.gate.e_score_correction_bias.shape[0] == 2
    assert tiny_glm4_moe.gate.n_routed_experts == 2


# --- Tensor slicing tests (GLM4 MoE Lite) ---

def test_glm4_moe_lite_prune_shapes(tiny_glm4_moe):
    """glm4_moe_lite prunes identically to glm4_moe."""
    config = {"model_type": "glm4_moe_lite", "n_routed_experts": 4,
              "num_experts_per_tok": 2, "num_hidden_layers": 1,
              "first_k_dense_replace": 0, "moe_layer_freq": 1}
    layer = FakeGLM4Layer(tiny_glm4_moe)
    model = FakeModel(FakeGLM4Inner([layer]))
    from mlx_fun.adapters.glm4_moe_lite import GLM4MoELiteAdapter
    adapter = GLM4MoELiteAdapter(model, config)

    keep = np.array([1, 3])
    prune_moe_layer(adapter, 0, keep)

    for proj_name in ("gate_proj", "up_proj", "down_proj"):
        proj = getattr(tiny_glm4_moe.switch_mlp, proj_name)
        assert proj.weight.shape[0] == 2

    assert tiny_glm4_moe.gate.weight.shape[0] == 2
    assert tiny_glm4_moe.gate.e_score_correction_bias.shape[0] == 2
    assert tiny_glm4_moe.gate.n_routed_experts == 2


# --- Tensor slicing tests (Qwen3) ---

class FakeQwen3Layer:
    def __init__(self, mlp):
        self.mlp = mlp

class FakeQwen3Inner:
    def __init__(self, layers):
        self.layers = layers


def test_qwen3_prune_shapes(tiny_qwen3_moe):
    """After pruning, Qwen3 tensors have correct shapes."""
    config = {"model_type": "qwen3_moe", "num_experts": 4,
              "num_experts_per_tok": 2, "num_hidden_layers": 1,
              "decoder_sparse_step": 1, "mlp_only_layers": []}
    layer = FakeQwen3Layer(tiny_qwen3_moe)
    model = FakeModel(FakeQwen3Inner([layer]))
    adapter = Qwen3MoEAdapter(model, config)

    keep = np.array([0, 3])  # Keep 2 of 4
    prune_moe_layer(adapter, 0, keep)

    for proj_name in ("gate_proj", "up_proj", "down_proj"):
        proj = getattr(tiny_qwen3_moe.switch_mlp, proj_name)
        assert proj.weight.shape[0] == 2, f"{proj_name} expert axis wrong"

    # Gate: nn.Linear weight (num_experts, hidden) -> (2, 32)
    assert tiny_qwen3_moe.gate.weight.shape[0] == 2
    # num_experts attribute updated
    assert tiny_qwen3_moe.num_experts == 2


def test_qwen3_zero_prune_identity(tiny_qwen3_moe, sample_input):
    """Pruning 0 experts should not change Qwen3 output."""
    mx.eval(tiny_qwen3_moe.parameters())
    orig_out = tiny_qwen3_moe(sample_input)
    mx.eval(orig_out)
    orig_np = np.array(orig_out, copy=True)

    config = {"model_type": "qwen3_moe", "num_experts": 4,
              "num_experts_per_tok": 2, "num_hidden_layers": 1,
              "decoder_sparse_step": 1, "mlp_only_layers": []}
    layer = FakeQwen3Layer(tiny_qwen3_moe)
    model = FakeModel(FakeQwen3Inner([layer]))
    adapter = Qwen3MoEAdapter(model, config)

    keep = np.array([0, 1, 2, 3])
    prune_moe_layer(adapter, 0, keep)
    mx.eval(tiny_qwen3_moe.parameters())

    after_out = tiny_qwen3_moe(sample_input)
    mx.eval(after_out)

    np.testing.assert_allclose(
        np.array(after_out, copy=False), orig_np, atol=1e-5,
    )


# --- Full model prune ---

def test_prune_model_config_update(tiny_minimax_moe):
    """prune_model returns updated config with reduced expert count."""
    config = {"model_type": "minimax", "num_local_experts": 4,
              "num_experts_per_tok": 2, "num_hidden_layers": 1}
    layer = FakeMiniMaxLayer(tiny_minimax_moe)
    model = FakeModel(FakeMiniMaxInner([layer]))
    adapter = MiniMaxAdapter(model, config)

    keep_map = {0: np.array([0, 1])}
    new_config = prune_model(adapter, keep_map)
    assert new_config["num_local_experts"] == 2


def test_qwen3_next_prune_shapes(tiny_qwen3_next_moe):
    """After pruning, Qwen3Next tensors have correct shapes and shared expert is preserved."""
    config = {"model_type": "qwen3_next", "num_experts": 4,
              "num_experts_per_tok": 2, "num_hidden_layers": 1,
              "decoder_sparse_step": 1, "mlp_only_layers": []}
    layer = FakeQwen3Layer(tiny_qwen3_next_moe)
    model = FakeModel(FakeQwen3Inner([layer]))
    adapter = Qwen3MoEAdapter(model, config)

    keep = np.array([0, 3])
    prune_moe_layer(adapter, 0, keep)

    for proj_name in ("gate_proj", "up_proj", "down_proj"):
        proj = getattr(tiny_qwen3_next_moe.switch_mlp, proj_name)
        assert proj.weight.shape[0] == 2, f"{proj_name} expert axis wrong"

    # Gate: (num_experts, hidden) -> (2, 32)
    assert tiny_qwen3_next_moe.gate.weight.shape[0] == 2
    assert tiny_qwen3_next_moe.num_experts == 2

    # Shared expert should be unchanged
    assert tiny_qwen3_next_moe.shared_expert.weight.shape == (32, 32)
    assert tiny_qwen3_next_moe.shared_expert_gate.weight.shape == (1, 32)


def test_qwen3_next_zero_prune_identity(tiny_qwen3_next_moe, sample_input):
    """Pruning 0 experts should not change Qwen3Next output."""
    mx.eval(tiny_qwen3_next_moe.parameters())
    orig_out = tiny_qwen3_next_moe(sample_input)
    mx.eval(orig_out)
    orig_np = np.array(orig_out, copy=True)

    config = {"model_type": "qwen3_next", "num_experts": 4,
              "num_experts_per_tok": 2, "num_hidden_layers": 1,
              "decoder_sparse_step": 1, "mlp_only_layers": []}
    layer = FakeQwen3Layer(tiny_qwen3_next_moe)
    model = FakeModel(FakeQwen3Inner([layer]))
    adapter = Qwen3MoEAdapter(model, config)

    keep = np.array([0, 1, 2, 3])
    prune_moe_layer(adapter, 0, keep)
    mx.eval(tiny_qwen3_next_moe.parameters())

    after_out = tiny_qwen3_next_moe(sample_input)
    mx.eval(after_out)

    np.testing.assert_allclose(
        np.array(after_out, copy=False), orig_np, atol=1e-5,
    )


def test_minimax_m2_prune_shapes(tiny_minimax_moe):
    """minimax_m2 model_type prunes identically to minimax."""
    config = {"model_type": "minimax_m2", "num_local_experts": 4,
              "num_experts_per_tok": 2, "num_hidden_layers": 1}
    layer = FakeMiniMaxLayer(tiny_minimax_moe)
    model = FakeModel(FakeMiniMaxInner([layer]))
    adapter = MiniMaxAdapter(model, config)

    keep = np.array([0, 2])
    prune_moe_layer(adapter, 0, keep)

    for proj_name in ("gate_proj", "up_proj", "down_proj"):
        proj = getattr(tiny_minimax_moe.switch_mlp, proj_name)
        assert proj.weight.shape[0] == 2
    assert tiny_minimax_moe.gate.weight.shape[0] == 2
    assert tiny_minimax_moe.e_score_correction_bias.shape[0] == 2


def test_prune_model_too_few_experts(tiny_minimax_moe):
    """Error when keeping fewer experts than top_k."""
    config = {"model_type": "minimax", "num_local_experts": 4,
              "num_experts_per_tok": 2, "num_hidden_layers": 1}
    layer = FakeMiniMaxLayer(tiny_minimax_moe)
    model = FakeModel(FakeMiniMaxInner([layer]))
    adapter = MiniMaxAdapter(model, config)

    keep_map = {0: np.array([0])}  # Only 1, but top_k=2
    with pytest.raises(ValueError, match="Must keep at least"):
        prune_model(adapter, keep_map)


def test_prune_model_warns_exact_topk(tiny_minimax_moe):
    """Warning when keeping exactly top_k experts."""
    config = {"model_type": "minimax", "num_local_experts": 4,
              "num_experts_per_tok": 2, "num_hidden_layers": 1}
    layer = FakeMiniMaxLayer(tiny_minimax_moe)
    model = FakeModel(FakeMiniMaxInner([layer]))
    adapter = MiniMaxAdapter(model, config)

    keep_map = {0: np.array([0, 1])}  # Exactly top_k=2
    with pytest.warns(UserWarning, match="exactly"):
        prune_model(adapter, keep_map)
