"""Adapter for Qwen4-Exp (``Qwen/Qwen3.8-Flash-Next``) MoE architecture.

Qwen4-Exp is a vision-language model, so it loads through mlx-vlm rather than
mlx-lm (see ``mlx_fun.loader``) and its language hyperparameters live under a
nested ``text_config``. The MoE stack itself is conventional: every decoder
layer carries an ``mlp`` that is mlx-vlm's ``Qwen3_5MoeSparseMoeBlock`` —
softmax gate → top-k → ``SwitchGLU``, plus a sigmoid-gated shared expert that
is not routed and therefore not a pruning target.

Layout::

    model.language_model.model.layers[i].mlp
        .gate                 nn.Linear(hidden, num_experts, bias=False)
        .switch_mlp           SwitchGLU(hidden, moe_intermediate_size, num_experts)
        .shared_expert        always-on MLP (not routed)
        .shared_expert_gate   nn.Linear(hidden, 1, bias=False)
"""

from typing import List

import mlx.nn as nn

from .base import BaseAdapter


class Qwen4ExpAdapter(BaseAdapter):
    """Qwen4-Exp: all decoder layers are MoE; block at ``.mlp``.

    The vision tower is left untouched — it holds no experts.
    """

    def __init__(self, model: nn.Module, config: dict):
        super().__init__(model, config)  # full config for save/prune
        self._moe_config = config.get("text_config", config)
        # Unwrap the multimodal wrapper: experts live in the language stack.
        self._layers = getattr(model, "language_model", model).model.layers

    def moe_layer_indices(self) -> List[int]:
        # Every layer is MoE — the linear-attention / full-attention split
        # applies to the attention branch only, not the MLP branch.
        return list(range(self._moe_config["num_hidden_layers"]))

    def get_moe_block(self, layer_idx: int) -> nn.Module:
        return self._layers[layer_idx].mlp

    def get_switch_mlp(self, moe_block: nn.Module):
        return moe_block.switch_mlp

    def num_routed_experts(self) -> int:
        return self._moe_config["num_experts"]

    def num_experts_per_tok(self) -> int:
        return self._moe_config["num_experts_per_tok"]

    def config_expert_count_key(self) -> str:
        return "num_experts"

    def get_gate_module(self, moe_block: nn.Module):
        return moe_block.gate

    def intermediate_size(self) -> int:
        return self._moe_config["moe_intermediate_size"]
