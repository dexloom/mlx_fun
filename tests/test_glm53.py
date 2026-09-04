"""Tests for GLM-5.3 and GLM-5.3-Flash support.

GLM-5.3 (``zai-org/GLM-5.3``) is `model_type: glm_moe_dsa` — the architecture
mlx_fun already supports — but it is the first config in that family to ship an
explicit per-layer ``mlp_layer_types`` list, which the adapter now honors.

GLM-5.3-Flash (``zai-org/GLM-5.3-Flash``) is `model_type: glm5_next`: a
vision-language model whose MoE block is mlx-vlm's ``DeepseekV32MoE`` — the
same sigmoid-scored / ``noaux_tc`` block as GLM-5, so it reuses the GLM hooks.
"""

import mlx.core as mx
import numpy as np
import pytest

from mlx_fun.adapters import get_adapter
from mlx_fun.adapters.glm5_next import GLM5NextAdapter
from mlx_fun.adapters.glm_moe_dsa import GLMMoeDsaAdapter
from mlx_fun.loader import is_vision_model
from mlx_fun.observer import collect_captures, install_hooks, remove_hooks
from mlx_fun.ream_hooks import collect_ream_data, install_ream_hooks, remove_ream_hooks
from mlx_fun.safety import compute_top_k_from_logits
from mlx_fun.steering import (
    SteeringConfig,
    install_steering_hooks,
    remove_steering_hooks,
)

from .conftest import TinyGLM4MoE


# ---------------------------------------------------------------------------
# GLM-5.3 — text-only, model_type glm_moe_dsa
# ---------------------------------------------------------------------------

class FakeLayer:
    def __init__(self, mlp):
        self.mlp = mlp


class FakeInner:
    def __init__(self, blocks):
        self.layers = [FakeLayer(b) for b in blocks]


class FakeTextModel:
    def __init__(self, blocks):
        self.model = FakeInner(blocks)


class FakeLanguageModel:
    def __init__(self, blocks):
        self.model = FakeInner(blocks)


class FakeGLM5NextModel:
    """model.language_model.model.layers[i].mlp — mlx-vlm's glm5_next layout."""

    def __init__(self, blocks):
        self.vision_tower = object()
        self.language_model = FakeLanguageModel(blocks)


def _blocks(n):
    mx.random.seed(42)
    out = []
    for _ in range(n):
        b = TinyGLM4MoE(hidden=32, intermediate=64, n_experts=4, top_k=2)
        b.gate.weight = mx.random.normal((4, 32)) * 0.1
        mx.eval(b.parameters())
        out.append(b)
    return out


N_LAYERS = 8
FIRST_K_DENSE = 3


@pytest.fixture
def glm53_config():
    """Shape of zai-org/GLM-5.3's config (scaled down)."""
    return {
        "model_type": "glm_moe_dsa",
        "num_hidden_layers": N_LAYERS,
        "first_k_dense_replace": FIRST_K_DENSE,
        "moe_layer_freq": 1,
        "n_routed_experts": 4,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 64,
        "n_shared_experts": 1,
        "scoring_func": "sigmoid",
        "topk_method": "noaux_tc",
        "mlp_layer_types": ["dense"] * FIRST_K_DENSE
        + ["sparse"] * (N_LAYERS - FIRST_K_DENSE),
    }


class TestGLM53Adapter:
    def test_resolves_to_existing_glm_adapter(self, glm53_config):
        model = FakeTextModel(_blocks(N_LAYERS))
        assert isinstance(get_adapter(model, glm53_config), GLMMoeDsaAdapter)

    def test_mlp_layer_types_drives_selection(self, glm53_config):
        model = FakeTextModel(_blocks(N_LAYERS))
        adapter = GLMMoeDsaAdapter(model, glm53_config)
        assert adapter.moe_layer_indices() == list(range(FIRST_K_DENSE, N_LAYERS))

    def test_sparse_hole_is_respected(self, glm53_config):
        """A dense layer past first_k_dense_replace must not be treated as MoE."""
        types = list(glm53_config["mlp_layer_types"])
        types[5] = "dense"
        glm53_config["mlp_layer_types"] = types

        adapter = GLMMoeDsaAdapter(FakeTextModel(_blocks(N_LAYERS)), glm53_config)
        assert 5 not in adapter.moe_layer_indices()
        assert adapter.moe_layer_indices() == [3, 4, 6, 7]

    def test_falls_back_to_stride_without_layer_types(self, glm53_config):
        """GLM-5 / DeepSeek V3.2 configs ship no mlp_layer_types."""
        del glm53_config["mlp_layer_types"]
        adapter = GLMMoeDsaAdapter(FakeTextModel(_blocks(N_LAYERS)), glm53_config)
        assert adapter.moe_layer_indices() == list(range(FIRST_K_DENSE, N_LAYERS))

    def test_expert_counts(self, glm53_config):
        adapter = GLMMoeDsaAdapter(FakeTextModel(_blocks(N_LAYERS)), glm53_config)
        assert adapter.num_routed_experts() == 4
        assert adapter.num_experts_per_tok() == 2
        assert adapter.config_expert_count_key() == "n_routed_experts"


# ---------------------------------------------------------------------------
# GLM-5.3-Flash — vision-language, model_type glm5_next
# ---------------------------------------------------------------------------

@pytest.fixture
def glm53_flash_config():
    """Shape of zai-org/GLM-5.3-Flash's config (scaled down)."""
    return {
        "model_type": "glm5_next",
        "image_token_id": 154854,
        "video_token_id": 154855,
        "vision_config": {"depth": 24, "hidden_size": 1024},
        "text_config": {
            "model_type": "glm5_next_text",
            "num_hidden_layers": N_LAYERS,
            "first_k_dense_replace": FIRST_K_DENSE,
            "n_routed_experts": 4,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 64,
            "n_shared_experts": 1,
            "scoring_func": "sigmoid",
            "topk_method": "noaux_tc",
            "norm_topk_prob": True,
            "mlp_layer_types": ["dense"] * FIRST_K_DENSE
            + ["sparse"] * (N_LAYERS - FIRST_K_DENSE),
        },
    }


class TestGLM53FlashAdapter:
    def test_detected_as_vision_model(self, glm53_flash_config):
        assert is_vision_model(glm53_flash_config)

    def test_factory_resolves_by_model_type(self, glm53_flash_config):
        model = FakeGLM5NextModel(_blocks(N_LAYERS))
        assert isinstance(get_adapter(model, glm53_flash_config), GLM5NextAdapter)

    def test_reaches_through_multimodal_wrapper(self, glm53_flash_config):
        model = FakeGLM5NextModel(_blocks(N_LAYERS))
        adapter = GLM5NextAdapter(model, glm53_flash_config)
        for i in adapter.moe_layer_indices():
            expected = model.language_model.model.layers[i].mlp
            assert adapter.get_moe_block(i) is expected

    def test_dense_prefix_excluded(self, glm53_flash_config):
        model = FakeGLM5NextModel(_blocks(N_LAYERS))
        adapter = GLM5NextAdapter(model, glm53_flash_config)
        assert adapter.moe_layer_indices() == list(range(FIRST_K_DENSE, N_LAYERS))

    def test_reads_nested_text_config(self, glm53_flash_config):
        model = FakeGLM5NextModel(_blocks(N_LAYERS))
        adapter = GLM5NextAdapter(model, glm53_flash_config)
        assert adapter.num_routed_experts() == 4
        assert adapter.num_experts_per_tok() == 2
        assert adapter.intermediate_size() == 64
        assert adapter.config_expert_count_key() == "n_routed_experts"

    def test_full_config_kept_for_save(self, glm53_flash_config):
        model = FakeGLM5NextModel(_blocks(N_LAYERS))
        adapter = GLM5NextAdapter(model, glm53_flash_config)
        assert "vision_config" in adapter.config


# ---------------------------------------------------------------------------
# glm5_next reuses the GLM hook family
# ---------------------------------------------------------------------------

class TestGLM5NextHooks:
    def test_saliency_hook_preserves_output(self, tiny_glm4_moe, sample_input):
        reference = np.array(tiny_glm4_moe(sample_input), copy=False).copy()

        install_hooks([tiny_glm4_moe], "glm5_next")
        hooked = np.array(tiny_glm4_moe(sample_input), copy=False).copy()
        inds, scores, norms = collect_captures([tiny_glm4_moe])[0][0]
        remove_hooks([tiny_glm4_moe])

        assert np.allclose(reference, hooked, atol=1e-5)
        assert inds.shape == (1, 8, 2)
        assert norms.shape == (1, 8, 2)

    def test_ream_hook_captures_gate_logits(self, tiny_glm4_moe, sample_input):
        install_ream_hooks([tiny_glm4_moe], "glm5_next")
        tiny_glm4_moe(sample_input)
        layer_input, gate_logits, sel_inds = collect_ream_data([tiny_glm4_moe])[0][0]
        remove_ream_hooks([tiny_glm4_moe])

        assert layer_input.shape == (1, 8, 32)
        assert gate_logits.shape == (1, 8, 4)

    def test_steering_changes_output(self, tiny_glm4_moe, sample_input):
        reference = np.array(tiny_glm4_moe(sample_input), copy=False).copy()

        install_steering_hooks(
            [tiny_glm4_moe], "glm5_next",
            SteeringConfig(deactivate={0: [0, 1]}, mask_value=-1e9),
            num_experts=4,
        )
        steered = np.array(tiny_glm4_moe(sample_input), copy=False).copy()
        remove_steering_hooks([tiny_glm4_moe])

        assert not np.allclose(reference, steered, atol=1e-5)

    def test_safety_topk_uses_sigmoid_family(self):
        rng = np.random.default_rng(0)
        logits = rng.normal(size=(16, 4)).astype(np.float32)

        inds = compute_top_k_from_logits(logits, "glm5_next", 2)

        # Sigmoid is monotonic, so top-k by score == top-k by logit.
        expected = np.argsort(-logits, axis=-1)[:, :2]
        assert inds.shape == (16, 2)
        assert set(map(tuple, np.sort(inds, axis=-1))) == set(
            map(tuple, np.sort(expected, axis=-1))
        )
