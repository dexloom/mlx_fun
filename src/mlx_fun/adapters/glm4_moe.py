"""Adapter for GLM4-MoE architecture."""

from typing import List

import mlx.nn as nn

from .base import BaseAdapter


class GLM4MoEAdapter(BaseAdapter):
    """MoE layers where layer_idx >= first_k_dense_replace.
    Block at model.model.layers[i].mlp."""

    def moe_layer_indices(self) -> List[int]:
        first_k = self.config.get("first_k_dense_replace", 0)
        n_layers = self.config["num_hidden_layers"]
        return [i for i in range(n_layers) if i >= first_k]

    def get_moe_block(self, layer_idx: int) -> nn.Module:
        return self.model.model.layers[layer_idx].mlp

    def get_switch_mlp(self, moe_block: nn.Module):
        return moe_block.switch_mlp

    def num_routed_experts(self) -> int:
        return self.config["n_routed_experts"]

    def num_experts_per_tok(self) -> int:
        return self.config["num_experts_per_tok"]

    def config_expert_count_key(self) -> str:
        return "n_routed_experts"

    def get_gate_module(self, moe_block: nn.Module):
        return moe_block.gate

    def intermediate_size(self) -> int:
        return self.config["moe_intermediate_size"]
