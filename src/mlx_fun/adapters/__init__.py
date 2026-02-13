"""Model-specific adapters for MoE architectures."""

import mlx.nn as nn

from .base import BaseAdapter
from .minimax import MiniMaxAdapter
from .glm4_moe import GLM4MoEAdapter
from .glm4_moe_lite import GLM4MoELiteAdapter
from .qwen3_moe import Qwen3MoEAdapter
from .glm_moe_dsa import GLMMoeDsaAdapter

_ADAPTER_MAP = {
    "minimax": MiniMaxAdapter,
    "minimax_m2": MiniMaxAdapter,
    "glm4_moe": GLM4MoEAdapter,
    "glm4_moe_lite": GLM4MoELiteAdapter,
    "qwen3_moe": Qwen3MoEAdapter,
    "qwen3_next": Qwen3MoEAdapter,
    "glm_moe_dsa": GLMMoeDsaAdapter,
    "deepseek_v32": GLMMoeDsaAdapter,
}


def get_adapter(model: nn.Module, config: dict) -> BaseAdapter:
    """Detect model type from config and return the appropriate adapter."""
    model_type = config.get("model_type", "")
    if model_type not in _ADAPTER_MAP:
        raise ValueError(
            f"Unsupported model_type '{model_type}'. "
            f"Supported: {list(_ADAPTER_MAP.keys())}"
        )
    return _ADAPTER_MAP[model_type](model, config)
