"""Adapter for Nemotron-H (nemotron_h) hybrid Mamba-2/Attention/MoE architecture.

Nemotron-H uses a hybrid_override_pattern to interleave Mamba-2 ('M'),
Attention ('*'), and MoE ('E') layers. MoE blocks live at
model.backbone.layers[i].mixer (NemotronHMoE) and use the same MoEGate +
SwitchMLP pattern as DeepSeek V3.2 / GLM-5.
"""

from typing import List

import mlx.nn as nn

from .base import BaseAdapter


class NemotronHAdapter(BaseAdapter):
    """MoE layers identified by 'E' entries in hybrid_override_pattern.
    Block at model.backbone.layers[i].mixer."""

    def moe_layer_indices(self) -> List[int]:
        pattern = self.config["hybrid_override_pattern"]
        return [i for i, t in enumerate(pattern) if t == "E"]

    def get_moe_block(self, layer_idx: int) -> nn.Module:
        return self.model.backbone.layers[layer_idx].mixer

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
