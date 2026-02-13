"""Base adapter ABC for MoE architectures."""

from abc import ABC, abstractmethod
from typing import List

import mlx.nn as nn


class BaseAdapter(ABC):
    """Abstract interface for model-specific MoE access."""

    def __init__(self, model: nn.Module, config: dict):
        self.model = model
        self.config = config

    @abstractmethod
    def moe_layer_indices(self) -> List[int]:
        """Return indices of decoder layers that contain MoE blocks."""

    @abstractmethod
    def get_moe_block(self, layer_idx: int) -> nn.Module:
        """Return the MoE module for a given decoder layer."""

    @abstractmethod
    def get_switch_mlp(self, moe_block: nn.Module):
        """Return the SwitchGLU expert container from a MoE block."""

    @abstractmethod
    def num_routed_experts(self) -> int:
        """Total number of routed experts per layer."""

    @abstractmethod
    def num_experts_per_tok(self) -> int:
        """Number of experts selected per token (top-k)."""

    @abstractmethod
    def config_expert_count_key(self) -> str:
        """Config key for expert count (e.g. 'num_local_experts')."""

    @abstractmethod
    def get_gate_module(self, moe_block: nn.Module):
        """Return the gate/router module from a MoE block."""

    @abstractmethod
    def intermediate_size(self) -> int:
        """Return the intermediate dimension of expert FFN layers."""
