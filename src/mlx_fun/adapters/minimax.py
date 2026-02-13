"""Adapter for MiniMax MoE architecture."""

from typing import List

import mlx.nn as nn

from .base import BaseAdapter


class MiniMaxAdapter(BaseAdapter):
    """All layers are MoE. Block at model.model.layers[i].block_sparse_moe."""

    def moe_layer_indices(self) -> List[int]:
        return list(range(len(self.model.model.layers)))

    def get_moe_block(self, layer_idx: int) -> nn.Module:
        return self.model.model.layers[layer_idx].block_sparse_moe

    def get_switch_mlp(self, moe_block: nn.Module):
        return moe_block.switch_mlp

    def num_routed_experts(self) -> int:
        return self.config["num_local_experts"]

    def num_experts_per_tok(self) -> int:
        return self.config["num_experts_per_tok"]

    def config_expert_count_key(self) -> str:
        return "num_local_experts"

    def get_gate_module(self, moe_block: nn.Module):
        return moe_block.gate

    def intermediate_size(self) -> int:
        return self.config["moe_intermediate_size"]
