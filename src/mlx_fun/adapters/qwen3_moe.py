"""Adapter for Qwen3-MoE architecture."""

from typing import List

import mlx.nn as nn

from .base import BaseAdapter


class Qwen3MoEAdapter(BaseAdapter):
    """MoE layers determined by decoder_sparse_step and mlp_only_layers.
    Block at model.model.layers[i].mlp."""

    def moe_layer_indices(self) -> List[int]:
        n_layers = self.config["num_hidden_layers"]
        num_experts = self.config.get("num_experts", 0)
        sparse_step = self.config.get("decoder_sparse_step", 1)
        mlp_only = set(self.config.get("mlp_only_layers", []))
        return [
            i for i in range(n_layers)
            if i not in mlp_only
            and num_experts > 0
            and (i + 1) % sparse_step == 0
        ]

    def get_moe_block(self, layer_idx: int) -> nn.Module:
        return self.model.model.layers[layer_idx].mlp

    def get_switch_mlp(self, moe_block: nn.Module):
        return moe_block.switch_mlp

    def num_routed_experts(self) -> int:
        return self.config["num_experts"]

    def num_experts_per_tok(self) -> int:
        return self.config["num_experts_per_tok"]

    def config_expert_count_key(self) -> str:
        return "num_experts"

    def get_gate_module(self, moe_block: nn.Module):
        return moe_block.gate

    def intermediate_size(self) -> int:
        return self.config["moe_intermediate_size"]
