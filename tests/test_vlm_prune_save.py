"""Pruning and saving a vision-language MoE checkpoint.

Pruning a VLM used to produce a checkpoint that reported success and could not
be loaded. Three independent faults, all of which these tests pin down:

* ``prune_moe_layer``'s model-type branch had no case for the VLM types and no
  ``else``, so the SwitchGLU experts were sliced while the router gate kept its
  original width.
* ``mlx_lm.utils.save_config`` drops ``vision_config`` as an unused key, which
  is right for a text model and fatal for a VLM — mlx-vlm then rebuilds the
  tower from dataclass defaults and the weights stop matching.
* ``prune_model`` wrote the expert count to the top level of a config that
  keeps it under ``text_config``.

The blocks here are the tiny replicas from ``conftest``; the real mlx-vlm
classes are exercised in ``test_vlm_integration.py``.
"""

import json

import mlx.core as mx
import numpy as np
import pytest

from mlx_fun.adapters import get_adapter
from mlx_fun.pruner import prune_model, prune_moe_layer
from mlx_fun.save import _save_config

from .conftest import TinyQwen4ExpMoE


N_LAYERS = 3
N_EXPERTS = 4
TOP_K = 2
HIDDEN = 32


# ---------------------------------------------------------------------------
# Minimal mlx-vlm-shaped model tree
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


class FakeVLM:
    """model.language_model.model.layers[i].mlp — mlx-vlm's layout."""

    def __init__(self, blocks):
        self.vision_tower = object()
        self.language_model = FakeLanguageModel(blocks)


def _config(model_type):
    """Shaped after Qwen/Qwen3.5-35B-A3B's config.json."""
    return {
        "model_type": model_type,
        "image_token_id": 248056,
        "vision_config": {"depth": 27, "hidden_size": 1152},
        "text_config": {
            "model_type": f"{model_type}_text",
            "num_hidden_layers": N_LAYERS,
            "num_experts": N_EXPERTS,
            "num_experts_per_tok": TOP_K,
            "moe_intermediate_size": 64,
            "shared_expert_intermediate_size": 64,
            "hidden_size": HIDDEN,
            "mlp_only_layers": [],
        },
    }


@pytest.fixture
def vlm():
    mx.random.seed(42)
    blocks = [
        TinyQwen4ExpMoE(hidden=HIDDEN, intermediate=64,
                        n_experts=N_EXPERTS, top_k=TOP_K)
        for _ in range(N_LAYERS)
    ]
    for b in blocks:
        mx.eval(b.parameters())
    return FakeVLM(blocks)


# ---------------------------------------------------------------------------
# Slicing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_type", ["qwen3_5_moe", "qwen4_exp"])
def test_prune_slices_the_router_gate(vlm, model_type):
    # The experts and the router must shrink together. Slicing only the
    # SwitchGLU leaves a checkpoint that saves fine and cannot be loaded.
    adapter = get_adapter(vlm, _config(model_type))
    block = adapter.get_moe_block(0)

    prune_moe_layer(adapter, 0, np.array([0, 2], dtype=np.intp))

    assert block.gate.weight.shape[0] == 2
    assert block.switch_mlp.gate_proj.weight.shape[0] == 2
    assert block.num_experts == 2
    # The shared expert is not routed and keeps its full width.
    assert block.shared_expert.weight.shape[0] == HIDDEN


def test_prune_refuses_a_model_type_with_no_branch(vlm):
    # Falling through the dispatch used to slice the experts, leave the router
    # untouched, and save without complaint.
    from mlx_fun.adapters.qwen4_exp import Qwen4ExpAdapter

    adapter = Qwen4ExpAdapter(vlm, dict(_config("qwen3_5_moe"),
                                        model_type="some_future_moe"))
    with pytest.raises(ValueError, match="router gate would be left unsliced"):
        prune_moe_layer(adapter, 0, np.array([0, 2], dtype=np.intp))


# ---------------------------------------------------------------------------
# Config written back out
# ---------------------------------------------------------------------------

def test_expert_count_is_written_under_text_config_only(vlm):
    config = _config("qwen3_5_moe")
    adapter = get_adapter(vlm, config)
    keep = {i: np.array([0, 2], dtype=np.intp) for i in range(N_LAYERS)}

    new_config = prune_model(adapter, keep)

    assert new_config["text_config"]["num_experts"] == 2
    # The source config had no top-level num_experts; inventing one writes a key
    # the loader never reads and the checkpoint never had.
    assert "num_experts" not in new_config
    # And the caller's config must not be mutated.
    assert config["text_config"]["num_experts"] == N_EXPERTS


def test_saved_config_keeps_the_vision_tower(tmp_path):
    config = _config("qwen3_5_moe")
    path = tmp_path / "config.json"

    _save_config(config, path)

    with open(path) as f:
        written = json.load(f)
    assert written["vision_config"] == config["vision_config"]
    assert written["text_config"]["num_experts"] == N_EXPERTS
    # Upstream mutates the dict it is handed; we must not.
    assert "vision_config" in config


def test_saved_config_is_unchanged_for_text_models(tmp_path):
    config = {"model_type": "qwen3_moe", "num_experts": 8}
    path = tmp_path / "config.json"

    _save_config(config, path)

    with open(path) as f:
        written = json.load(f)
    assert "vision_config" not in written
    assert written["num_experts"] == 8
