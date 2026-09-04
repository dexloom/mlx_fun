"""Vision-language MoE checks against the real mlx-vlm classes.

The unit tests in ``test_qwen4_exp.py`` / ``test_glm53.py`` run on tiny
replicas so they need no optional dependency. These tests instantiate mlx-vlm's
actual modules, which is what catches an upstream change to a block's routing
math or its attribute names — the two things mlx_fun's hooks reimplement.
Skipped when mlx-vlm is not installed (``pip install "mlx-fun[vlm]"``).

Covers:
  * Qwen4-Exp (``Qwen/Qwen3.8-Flash-Next``) -> ``Qwen3_5MoeSparseMoeBlock``
  * GLM-5.3-Flash (``zai-org/GLM-5.3-Flash``) -> ``DeepseekV32MoE``
"""

import mlx.core as mx
import numpy as np
import pytest

pytest.importorskip("mlx_vlm", reason="mlx-vlm not installed")

from mlx_vlm.models.qwen3_5_moe.language import Qwen3_5MoeSparseMoeBlock  # noqa: E402
from mlx_vlm.models.qwen4_exp.config import TextConfig  # noqa: E402
from mlx_vlm.models.qwen4_exp.language import Qwen4ExpDecoderLayer  # noqa: E402

from mlx_fun.observer import collect_captures, install_hooks, remove_hooks  # noqa: E402
from mlx_fun.ream_hooks import (  # noqa: E402
    collect_ream_data,
    install_ream_hooks,
    remove_ream_hooks,
)
from mlx_fun.steering import (  # noqa: E402
    SteeringConfig,
    install_steering_hooks,
    remove_steering_hooks,
)

HIDDEN = 32
N_EXPERTS = 8
TOP_K = 2


@pytest.fixture
def text_config():
    return TextConfig(
        model_type="qwen4_exp_text",
        hidden_size=HIDDEN,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        num_experts=N_EXPERTS,
        num_experts_per_tok=TOP_K,
        shared_expert_intermediate_size=64,
        moe_intermediate_size=64,
        rms_norm_eps=1e-6,
        vocab_size=128,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        max_position_embeddings=256,
    )


@pytest.fixture
def real_block(text_config):
    mx.random.seed(0)
    block = Qwen3_5MoeSparseMoeBlock(text_config)
    mx.eval(block.parameters())
    return block


@pytest.fixture
def x():
    mx.random.seed(1)
    val = mx.random.normal((1, 6, HIDDEN))
    mx.eval(val)
    return val


class TestRealBlockShape:
    def test_decoder_layer_mlp_is_the_moe_block(self, text_config):
        """The adapter looks for the MoE block at ``layer.mlp``."""
        layer = Qwen4ExpDecoderLayer(text_config, 0)
        assert isinstance(layer.mlp, Qwen3_5MoeSparseMoeBlock)

    def test_attribute_names_the_hooks_rely_on(self, real_block):
        for attr in ("gate", "switch_mlp", "shared_expert", "shared_expert_gate",
                     "top_k", "num_experts"):
            assert hasattr(real_block, attr), f"missing {attr}"

    def test_no_norm_topk_prob_switch(self, real_block):
        """Why qwen4_exp needs its own hook rather than reusing qwen3_next's.

        Qwen3-Next's hook reads ``self.norm_topk_prob``; this block has no such
        attribute and renormalizes unconditionally.
        """
        assert not hasattr(real_block, "norm_topk_prob")


class TestHooksAgainstRealBlock:
    def test_saliency_hook_is_output_identical(self, real_block, x):
        reference = np.array(real_block(x), copy=False).copy()

        install_hooks([real_block], "qwen4_exp")
        hooked = np.array(real_block(x), copy=False).copy()
        inds, scores, norms = collect_captures([real_block])[0][0]
        remove_hooks([real_block])

        assert np.allclose(reference, hooked, atol=1e-6)
        assert inds.shape == (1, 6, TOP_K)
        assert scores.shape == (1, 6, TOP_K)
        assert norms.shape == (1, 6, TOP_K)
        assert np.allclose(scores.sum(axis=-1), 1.0, atol=1e-5)
        assert inds.max() < N_EXPERTS

    def test_ream_hook_is_output_identical(self, real_block, x):
        reference = np.array(real_block(x), copy=False).copy()

        install_ream_hooks([real_block], "qwen4_exp")
        hooked = np.array(real_block(x), copy=False).copy()
        layer_input, gate_logits, sel_inds = collect_ream_data([real_block])[0][0]
        remove_ream_hooks([real_block])

        assert np.allclose(reference, hooked, atol=1e-6)
        assert layer_input.shape == (1, 6, HIDDEN)
        assert gate_logits.shape == (1, 6, N_EXPERTS)

    def test_steering_masks_experts(self, real_block, x):
        reference = np.array(real_block(x), copy=False).copy()

        install_steering_hooks(
            [real_block], "qwen4_exp",
            SteeringConfig(deactivate={0: [0, 1]}, mask_value=-1e9),
            num_experts=N_EXPERTS,
        )
        steered = np.array(real_block(x), copy=False).copy()
        remove_steering_hooks([real_block])

        assert not np.allclose(reference, steered, atol=1e-5)

    def test_steering_without_targets_is_a_no_op(self, real_block, x):
        reference = np.array(real_block(x), copy=False).copy()

        install_steering_hooks(
            [real_block], "qwen4_exp", SteeringConfig(), num_experts=N_EXPERTS,
        )
        unsteered = np.array(real_block(x), copy=False).copy()
        remove_steering_hooks([real_block])

        assert np.allclose(reference, unsteered, atol=1e-6)


# ---------------------------------------------------------------------------
# GLM-5.3-Flash (glm5_next) — reuses the GLM hook family
# ---------------------------------------------------------------------------

from mlx_vlm.models.deepseek_v32.config import ModelConfig as DeepseekConfig  # noqa: E402
from mlx_vlm.models.deepseek_v32.language import DeepseekV32MoE  # noqa: E402
from mlx_vlm.models.glm5_next.config import TextConfig as Glm5NextTextConfig  # noqa: E402
from mlx_vlm.models.glm5_next.language import Glm5NextDecoderLayer  # noqa: E402


@pytest.fixture
def glm5_next_block():
    cfg = DeepseekConfig(
        model_type="deepseek_v32",
        hidden_size=HIDDEN,
        moe_intermediate_size=64,
        n_routed_experts=N_EXPERTS,
        num_experts_per_tok=TOP_K,
        n_shared_experts=1,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=2.5,
        norm_topk_prob=True,
        topk_method="noaux_tc",
        scoring_func="sigmoid",
        num_hidden_layers=4,
        first_k_dense_replace=1,
    )
    mx.random.seed(0)
    block = DeepseekV32MoE(cfg)
    mx.eval(block.parameters())
    return block


class TestGLM5NextRealBlock:
    def test_glm5_next_builds_a_deepseek_v32_moe(self):
        """The adapter and hooks assume GLM-5.3-Flash's .mlp is this block."""
        import inspect
        assert "DeepseekV32MoE" in inspect.getsource(Glm5NextDecoderLayer)

    def test_config_declares_per_layer_sparsity(self):
        """GLM5NextAdapter reads mlp_layer_types to pick MoE layers."""
        fields = set(Glm5NextTextConfig.__dataclass_fields__)
        assert "mlp_layer_types" in fields
        assert "first_k_dense_replace" in fields

    def test_attribute_names_the_glm_hooks_rely_on(self, glm5_next_block):
        for attr in ("gate", "switch_mlp", "shared_experts"):
            assert hasattr(glm5_next_block, attr), f"missing {attr}"

    def test_saliency_hook_is_output_identical(self, glm5_next_block, x):
        reference = np.array(glm5_next_block(x), copy=False).copy()

        install_hooks([glm5_next_block], "glm5_next")
        hooked = np.array(glm5_next_block(x), copy=False).copy()
        inds, scores, norms = collect_captures([glm5_next_block])[0][0]
        remove_hooks([glm5_next_block])

        assert np.allclose(reference, hooked, atol=1e-6)
        assert inds.shape == (1, 6, TOP_K)
        assert norms.shape == (1, 6, TOP_K)
        assert inds.max() < N_EXPERTS

    def test_ream_hook_is_output_identical(self, glm5_next_block, x):
        reference = np.array(glm5_next_block(x), copy=False).copy()

        install_ream_hooks([glm5_next_block], "glm5_next")
        hooked = np.array(glm5_next_block(x), copy=False).copy()
        layer_input, gate_logits, sel_inds = collect_ream_data([glm5_next_block])[0][0]
        remove_ream_hooks([glm5_next_block])

        assert np.allclose(reference, hooked, atol=1e-6)
        assert layer_input.shape == (1, 6, HIDDEN)
        assert gate_logits.shape == (1, 6, N_EXPERTS)

    def test_steering_masks_experts(self, glm5_next_block, x):
        reference = np.array(glm5_next_block(x), copy=False).copy()

        install_steering_hooks(
            [glm5_next_block], "glm5_next",
            SteeringConfig(deactivate={0: [0, 1]}, mask_value=-1e9),
            num_experts=N_EXPERTS,
        )
        steered = np.array(glm5_next_block(x), copy=False).copy()
        remove_steering_hooks([glm5_next_block])

        assert not np.allclose(reference, steered, atol=1e-5)
