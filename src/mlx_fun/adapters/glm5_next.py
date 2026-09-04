"""Adapter for GLM-5.3-Flash (``glm5_next``).

GLM-5.3-Flash is a vision-language model, so it loads through mlx-vlm (see
``mlx_fun.loader``) and its language hyperparameters live under a nested
``text_config``. Its MoE block is mlx-vlm's ``DeepseekV32MoE`` — the same
sigmoid-scored, ``noaux_tc`` block as GLM-5 / DeepSeek V3.2 — so it reuses the
existing GLM hooks rather than getting its own.

Layout::

    model.language_model.model.layers[i].mlp
        .gate                 MoEGate -> (inds, scores)
        .switch_mlp           SwitchGLU(hidden, moe_intermediate_size, n_routed_experts)
        .shared_experts       always-on MLP (not routed)

Unlike GLM-5, which derives sparsity from ``first_k_dense_replace`` plus a
``moe_layer_freq`` stride, GLM-5.3-Flash ships an explicit per-layer
``mlp_layer_types`` list. mlx-vlm builds a MoE block only where that list says
``"sparse"``, so the adapter follows the same rule.
"""

from typing import List

import mlx.nn as nn

from .base import BaseAdapter


class GLM5NextAdapter(BaseAdapter):
    """GLM-5.3-Flash: sparse layers per ``mlp_layer_types``; block at ``.mlp``.

    The vision tower is left untouched — it holds no experts.
    """

    def __init__(self, model: nn.Module, config: dict):
        super().__init__(model, config)  # full config for save/prune
        self._moe_config = config.get("text_config", config)
        # Unwrap the multimodal wrapper: experts live in the language stack.
        self._layers = getattr(model, "language_model", model).model.layers

    def moe_layer_indices(self) -> List[int]:
        cfg = self._moe_config
        n_layers = cfg["num_hidden_layers"]
        first_k = cfg.get("first_k_dense_replace", 0)
        layer_types = cfg.get("mlp_layer_types")

        if layer_types:
            return [
                i for i in range(n_layers)
                if i >= first_k and layer_types[i] == "sparse"
            ]

        freq = cfg.get("moe_layer_freq", 1)
        return [i for i in range(n_layers) if i >= first_k and i % freq == 0]

    def get_moe_block(self, layer_idx: int) -> nn.Module:
        return self._layers[layer_idx].mlp

    def get_switch_mlp(self, moe_block: nn.Module):
        return moe_block.switch_mlp

    def num_routed_experts(self) -> int:
        return self._moe_config["n_routed_experts"]

    def num_experts_per_tok(self) -> int:
        return self._moe_config["num_experts_per_tok"]

    def config_expert_count_key(self) -> str:
        return "n_routed_experts"

    def get_gate_module(self, moe_block: nn.Module):
        return moe_block.gate

    def intermediate_size(self) -> int:
        return self._moe_config["moe_intermediate_size"]
