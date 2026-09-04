"""Adapter for GLM-5 / GLM-5.3 (GLM-MoE-DSA) and DeepSeek V3.2 architectures.

GLM-5 inherits from DeepSeek V3.2. Both share the same DeepseekV32MoE block
with MoEGate. MoE layer selection: layers >= first_k_dense_replace with
moe_layer_freq stride, or the explicit per-layer ``mlp_layer_types`` list when
the config ships one (GLM-5.3 does; GLM-5 and DeepSeek V3.2 do not).
"""

from typing import List

import mlx.nn as nn

from .base import BaseAdapter


class GLMMoeDsaAdapter(BaseAdapter):
    """MoE layers where layer_idx >= first_k_dense_replace AND layer_idx % moe_layer_freq == 0.
    Block at model.model.layers[i].mlp."""

    def moe_layer_indices(self) -> List[int]:
        first_k = self.config.get("first_k_dense_replace", 0)
        n_layers = self.config["num_hidden_layers"]

        # GLM-5.3 declares sparsity per layer; prefer it over the stride rule.
        layer_types = self.config.get("mlp_layer_types")
        if layer_types:
            return [
                i for i in range(n_layers)
                if i >= first_k and layer_types[i] == "sparse"
            ]

        freq = self.config.get("moe_layer_freq", 1)
        return [i for i in range(n_layers) if i >= first_k and i % freq == 0]

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
