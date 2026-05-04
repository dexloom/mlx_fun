"""Adapter for Kimi-K2 / Kimi-K2.6 (model_type='kimi_k25').

Kimi-K2.6 wraps a DeepSeek-V3 text core inside a multimodal Model:

    Model.language_model.model.layers[i].mlp     (MoE block)

mlx-lm's `kimi_k25.Model` exposes a ``model`` property that returns
``language_model.model``, so ``model.model.layers[i].mlp`` resolves
correctly without needing custom traversal.

The routing parameters live under ``config['text_config']`` rather than
at the top level, so this adapter merges the nested config into a flat
view before delegating to GLMMoeDsaAdapter (DeepSeek-V3's adapter).
"""

from .glm_moe_dsa import GLMMoeDsaAdapter


class KimiK25Adapter(GLMMoeDsaAdapter):
    def __init__(self, model, config):
        text_config = config.get("text_config") or {}
        merged = {**config, **text_config}
        super().__init__(model, merged)
