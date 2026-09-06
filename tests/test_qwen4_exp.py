"""Tests for the Qwen4-Exp / Qwen3.5-3.6 MoE family.

``qwen4_exp`` (Qwen3.8-Flash-Next) and ``qwen3_5_moe`` (the Qwen3.5/3.6 MoE
line, e.g. Qwen3.6-35B-A3B) share one adapter and one hook set, so every case
here runs against both strings. Both load through mlx-vlm, so the adapter has
to reach through the multimodal wrapper and read hyperparameters from a nested
``text_config``. The hooks mirror mlx-vlm's ``Qwen3_5MoeSparseMoeBlock``.
"""

import mlx.core as mx
import numpy as np
import pytest

from mlx_fun.adapters import get_adapter
from mlx_fun.adapters.qwen4_exp import Qwen4ExpAdapter
from mlx_fun.observer import collect_captures, install_hooks, remove_hooks
from mlx_fun.ream_hooks import collect_ream_data, install_ream_hooks, remove_ream_hooks
from mlx_fun.safety import compute_top_k_from_logits
from mlx_fun.steering import (
    SteeringConfig,
    install_steering_hooks,
    remove_steering_hooks,
)

from .conftest import TinyQwen4ExpMoE


N_LAYERS = 3
N_EXPERTS = 4
TOP_K = 2

# Both model_type strings resolve to the same adapter and the same hooks.
MODEL_TYPES = ["qwen4_exp", "qwen3_5_moe"]


# ---------------------------------------------------------------------------
# Fake model tree mirroring mlx-vlm's layout
# ---------------------------------------------------------------------------

class FakeLayer:
    def __init__(self, mlp):
        self.mlp = mlp


class FakeInnerModel:
    def __init__(self, blocks):
        self.layers = [FakeLayer(b) for b in blocks]


class FakeLanguageModel:
    def __init__(self, blocks):
        self.model = FakeInnerModel(blocks)


class FakeQwen4ExpModel:
    """model.language_model.model.layers[i].mlp — mlx-vlm's Qwen4-Exp layout."""

    def __init__(self, blocks):
        self.vision_tower = object()
        self.language_model = FakeLanguageModel(blocks)


@pytest.fixture(params=MODEL_TYPES)
def model_type(request):
    """Both strings the Qwen MoE VLM family registers under."""
    return request.param


@pytest.fixture(params=MODEL_TYPES)
def qwen4_exp_config(request):
    return {
        "model_type": request.param,
        "image_token_id": 248056,
        "vision_config": {"depth": 27, "hidden_size": 1152},
        "text_config": {
            "model_type": "qwen4_exp_text",
            "num_hidden_layers": N_LAYERS,
            "num_experts": N_EXPERTS,
            "num_experts_per_tok": TOP_K,
            "moe_intermediate_size": 64,
            "shared_expert_intermediate_size": 64,
            "hidden_size": 32,
        },
    }


@pytest.fixture
def qwen4_exp_model():
    mx.random.seed(42)
    blocks = [
        TinyQwen4ExpMoE(hidden=32, intermediate=64, n_experts=N_EXPERTS, top_k=TOP_K)
        for _ in range(N_LAYERS)
    ]
    for b in blocks:
        mx.eval(b.parameters())
    return FakeQwen4ExpModel(blocks)


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class TestQwen4ExpAdapter:
    def test_factory_resolves_by_model_type(self, qwen4_exp_model, qwen4_exp_config):
        adapter = get_adapter(qwen4_exp_model, qwen4_exp_config)
        assert isinstance(adapter, Qwen4ExpAdapter)

    def test_every_layer_is_moe(self, qwen4_exp_model, qwen4_exp_config):
        adapter = Qwen4ExpAdapter(qwen4_exp_model, qwen4_exp_config)
        assert adapter.moe_layer_indices() == list(range(N_LAYERS))

    def test_reaches_through_multimodal_wrapper(self, qwen4_exp_model, qwen4_exp_config):
        adapter = Qwen4ExpAdapter(qwen4_exp_model, qwen4_exp_config)
        for i in range(N_LAYERS):
            expected = qwen4_exp_model.language_model.model.layers[i].mlp
            assert adapter.get_moe_block(i) is expected

    def test_reads_nested_text_config(self, qwen4_exp_model, qwen4_exp_config):
        adapter = Qwen4ExpAdapter(qwen4_exp_model, qwen4_exp_config)
        assert adapter.num_routed_experts() == N_EXPERTS
        assert adapter.num_experts_per_tok() == TOP_K
        assert adapter.intermediate_size() == 64

    def test_full_config_kept_for_save(self, qwen4_exp_model, qwen4_exp_config):
        # prune/save write the whole config back out, vision tower included.
        adapter = Qwen4ExpAdapter(qwen4_exp_model, qwen4_exp_config)
        assert adapter.config is qwen4_exp_config
        assert "vision_config" in adapter.config

    def test_gate_and_switch_accessors(self, qwen4_exp_model, qwen4_exp_config):
        adapter = Qwen4ExpAdapter(qwen4_exp_model, qwen4_exp_config)
        block = adapter.get_moe_block(0)
        assert adapter.get_gate_module(block) is block.gate
        assert adapter.get_switch_mlp(block) is block.switch_mlp

    def test_expert_count_key(self, qwen4_exp_model, qwen4_exp_config):
        adapter = Qwen4ExpAdapter(qwen4_exp_model, qwen4_exp_config)
        assert adapter.config_expert_count_key() == "num_experts"


# ---------------------------------------------------------------------------
# Saliency hooks
# ---------------------------------------------------------------------------

class TestQwen4ExpObserverHook:
    def test_hook_preserves_output(self, tiny_qwen4_exp_moe, sample_input, model_type):
        expected = tiny_qwen4_exp_moe(sample_input)
        mx.eval(expected)

        install_hooks([tiny_qwen4_exp_moe], model_type)
        got = tiny_qwen4_exp_moe(sample_input)
        mx.eval(got)
        remove_hooks([tiny_qwen4_exp_moe])

        assert np.allclose(np.array(expected), np.array(got), atol=1e-5)

    def test_hook_captures_routing(self, tiny_qwen4_exp_moe, sample_input, model_type):
        install_hooks([tiny_qwen4_exp_moe], model_type)
        tiny_qwen4_exp_moe(sample_input)
        captures = collect_captures([tiny_qwen4_exp_moe])
        remove_hooks([tiny_qwen4_exp_moe])

        assert len(captures) == 1 and len(captures[0]) == 1
        inds, scores, norms = captures[0][0]
        assert inds.shape == (1, 8, TOP_K)
        assert scores.shape == (1, 8, TOP_K)
        assert norms.shape == (1, 8, TOP_K)
        assert inds.max() < N_EXPERTS

    def test_scores_are_renormalized(self, tiny_qwen4_exp_moe, sample_input, model_type):
        # Qwen4-Exp renormalizes unconditionally — no norm_topk_prob switch.
        install_hooks([tiny_qwen4_exp_moe], model_type)
        tiny_qwen4_exp_moe(sample_input)
        _, scores, _ = collect_captures([tiny_qwen4_exp_moe])[0][0]
        remove_hooks([tiny_qwen4_exp_moe])

        assert np.allclose(scores.sum(axis=-1), 1.0, atol=1e-5)

    def test_hooks_restore_original_class(self, tiny_qwen4_exp_moe, model_type):
        original = type(tiny_qwen4_exp_moe)
        install_hooks([tiny_qwen4_exp_moe], model_type)
        assert type(tiny_qwen4_exp_moe) is not original
        remove_hooks([tiny_qwen4_exp_moe])
        assert type(tiny_qwen4_exp_moe) is original


# ---------------------------------------------------------------------------
# REAM hooks (used by safety-scan / domain-scan)
# ---------------------------------------------------------------------------

class TestQwen4ExpReamHook:
    def test_ream_hook_preserves_output(self, tiny_qwen4_exp_moe, sample_input, model_type):
        expected = tiny_qwen4_exp_moe(sample_input)
        mx.eval(expected)

        install_ream_hooks([tiny_qwen4_exp_moe], model_type)
        got = tiny_qwen4_exp_moe(sample_input)
        mx.eval(got)
        remove_ream_hooks([tiny_qwen4_exp_moe])

        assert np.allclose(np.array(expected), np.array(got), atol=1e-5)

    def test_ream_hook_captures_full_gate_logits(self, tiny_qwen4_exp_moe, sample_input, model_type):
        install_ream_hooks([tiny_qwen4_exp_moe], model_type)
        tiny_qwen4_exp_moe(sample_input)
        captures = collect_ream_data([tiny_qwen4_exp_moe])
        remove_ream_hooks([tiny_qwen4_exp_moe])

        layer_input, gate_logits, sel_inds = captures[0][0]
        assert layer_input.shape == (1, 8, 32)
        assert gate_logits.shape == (1, 8, N_EXPERTS)


# ---------------------------------------------------------------------------
# Steering
# ---------------------------------------------------------------------------

class TestQwen4ExpSteering:
    def test_deactivation_changes_output(self, tiny_qwen4_exp_moe, sample_input, model_type):
        original = np.array(tiny_qwen4_exp_moe(sample_input), copy=False).copy()

        config = SteeringConfig(deactivate={0: [0, 1]}, mask_value=-1e9)
        install_steering_hooks(
            [tiny_qwen4_exp_moe], model_type, config, num_experts=N_EXPERTS,
        )
        steered = tiny_qwen4_exp_moe(sample_input)
        mx.eval(steered)
        steered_np = np.array(steered, copy=False).copy()
        remove_steering_hooks([tiny_qwen4_exp_moe])

        assert not np.allclose(original, steered_np, atol=1e-5)

    def test_empty_config_preserves_output(self, tiny_qwen4_exp_moe, sample_input, model_type):
        original = np.array(tiny_qwen4_exp_moe(sample_input), copy=False).copy()

        install_steering_hooks(
            [tiny_qwen4_exp_moe], model_type, SteeringConfig(), num_experts=N_EXPERTS,
        )
        unsteered = tiny_qwen4_exp_moe(sample_input)
        mx.eval(unsteered)
        unsteered_np = np.array(unsteered, copy=False).copy()
        remove_steering_hooks([tiny_qwen4_exp_moe])

        assert np.allclose(original, unsteered_np, atol=1e-5)

    def test_install_remove_restores_class(self, tiny_qwen4_exp_moe, sample_input, model_type):
        original_cls = type(tiny_qwen4_exp_moe)
        install_steering_hooks(
            [tiny_qwen4_exp_moe], model_type,
            SteeringConfig(deactivate={0: [0]}), num_experts=N_EXPERTS,
        )
        assert type(tiny_qwen4_exp_moe) is not original_cls
        out = tiny_qwen4_exp_moe(sample_input)
        mx.eval(out)
        remove_steering_hooks([tiny_qwen4_exp_moe])
        assert type(tiny_qwen4_exp_moe) is original_cls


# ---------------------------------------------------------------------------
# Safety / domain scan support
# ---------------------------------------------------------------------------

class TestQwen4ExpAmplifyAndKnockout:
    def test_amplify_sets_gate_bias(self, tiny_qwen4_exp_moe, sample_input, model_type):
        """Pre-softmax gate.bias on an nn.Linear(bias=False), as for Qwen3."""
        from mlx_fun.domain import amplify_gate_weights

        original = np.array(tiny_qwen4_exp_moe(sample_input), copy=False).copy()
        amplify_gate_weights(
            [tiny_qwen4_exp_moe], model_type, {0: np.array([0.0, 0.0, 5.0, 0.0])},
        )
        assert "bias" in tiny_qwen4_exp_moe.gate

        amplified = np.array(tiny_qwen4_exp_moe(sample_input), copy=False).copy()
        assert not np.allclose(original, amplified, atol=1e-5)

    def test_knockout_targets_the_gate_bias(self, tiny_qwen4_exp_moe, model_type):
        from mlx_fun.probe import selection_bias_target

        module, attr = selection_bias_target(tiny_qwen4_exp_moe, model_type)
        assert module is tiny_qwen4_exp_moe.gate
        assert attr == "bias"


class TestQwen4ExpSafetyTopK:
    def test_topk_uses_softmax_family(self, model_type):
        # Softmax is monotonic in the logits, so top-k by logit == top-k by prob.
        rng = np.random.default_rng(0)
        logits = rng.normal(size=(16, N_EXPERTS)).astype(np.float32)

        inds = compute_top_k_from_logits(logits, model_type, TOP_K)

        assert inds.shape == (16, TOP_K)
        expected = np.argsort(-logits, axis=-1)[:, :TOP_K]
        assert set(map(tuple, np.sort(inds, axis=-1))) == set(
            map(tuple, np.sort(expected, axis=-1))
        )
