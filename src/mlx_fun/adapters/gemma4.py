"""Adapter for Gemma 4 MoE architecture.

Gemma 4 has no dedicated MoE block module — the Router and Experts are separate
attributes on each DecoderLayer.  We create a thin Gemma4MoEBlock wrapper that
combines them into a single callable, then patch each DecoderLayer's __call__
to dispatch through the wrapper.  All hook systems (observer, REAM, steering,
server counting) can then swap the wrapper's __call__ via the standard
__class__-swap pattern.
"""

from typing import List

import mlx.core as mx
import mlx.nn as nn

from .base import BaseAdapter


# ---------------------------------------------------------------------------
# MoE block wrapper
# ---------------------------------------------------------------------------

class Gemma4MoEBlock:
    """Combines Gemma 4's separate Router + pre-norm + Experts into one callable.

    NOT an nn.Module — avoids double-registration of parameters during
    tree_flatten / save.
    """

    def __init__(self, router, pre_norm, experts):
        self.router = router
        self.pre_feedforward_layernorm_2 = pre_norm
        self.experts = experts
        self.switch_glu = experts.switch_glu  # alias for get_switch_mlp

    def __call__(self, h: mx.array) -> mx.array:
        top_k_indices, top_k_weights = self.router(h)
        h2 = self.pre_feedforward_layernorm_2(h)
        return self.experts(h2, top_k_indices, top_k_weights)


# ---------------------------------------------------------------------------
# Patched DecoderLayer __call__  (routes MoE through self.moe_block)
# ---------------------------------------------------------------------------

def _patched_decoder_call(self, x, mask=None, cache=None,
                          per_layer_input=None, shared_kv=None, offset=None):
    """DecoderLayer forward that dispatches MoE through self.moe_block."""
    residual = x
    h = self.input_layernorm(x)
    h, shared_kv, offset = self.self_attn(
        h, mask, cache, shared_kv=shared_kv, offset=offset,
    )
    h = self.post_attention_layernorm(h)
    h = residual + h
    residual = h

    if self.enable_moe:
        h1 = self.pre_feedforward_layernorm(h)
        h1 = self.mlp(h1)
        h1 = self.post_feedforward_layernorm_1(h1)

        h2 = self.moe_block(h)
        h2 = self.post_feedforward_layernorm_2(h2)

        h = h1 + h2
    else:
        h = self.pre_feedforward_layernorm(h)
        h = self.mlp(h)

    h = self.post_feedforward_layernorm(h)
    h = residual + h

    if (
        self.per_layer_input_gate is not None
        and self.per_layer_projection is not None
        and self.post_per_layer_input_norm is not None
        and per_layer_input is not None
    ):
        residual = h
        gate = self.per_layer_input_gate(h)
        gate = nn.gelu_approx(gate)
        gate = mx.multiply(gate, per_layer_input)
        gate = self.per_layer_projection(gate)
        gate = self.post_per_layer_input_norm(gate)
        h = residual + gate

    if self.layer_scalar is not None:
        h = h * self.layer_scalar

    return h, shared_kv, offset


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class Gemma4Adapter(BaseAdapter):
    """Gemma 4 MoE: all layers MoE when enable_moe_block is True.
    Router + Experts are on the DecoderLayer; we wrap them in Gemma4MoEBlock."""

    def __init__(self, model: nn.Module, config: dict):
        super().__init__(model, config)  # full config for save/prune
        self._moe_config = config.get("text_config", config)
        self._patch_layers()

    def _patch_layers(self):
        """Create MoE block wrappers and patch DecoderLayers to use them."""
        for idx in self.moe_layer_indices():
            layer = self.model.layers[idx]
            wrapper = Gemma4MoEBlock(
                layer.router,
                layer.pre_feedforward_layernorm_2,
                layer.experts,
            )
            layer.moe_block = wrapper

            # Swap __class__ so the layer's __call__ routes through moe_block
            orig_cls = type(layer)
            patched_cls = type(
                f"_Gemma4Patched_{orig_cls.__name__}",
                (orig_cls,),
                {"__call__": _patched_decoder_call},
            )
            layer.__class__ = patched_cls

    def moe_layer_indices(self) -> List[int]:
        n_layers = self._moe_config["num_hidden_layers"]
        if self._moe_config.get("enable_moe_block", False):
            return list(range(n_layers))
        return []

    def get_moe_block(self, layer_idx: int):
        return self.model.layers[layer_idx].moe_block

    def get_switch_mlp(self, moe_block):
        return moe_block.switch_glu

    def num_routed_experts(self) -> int:
        return self._moe_config["num_experts"]

    def num_experts_per_tok(self) -> int:
        return self._moe_config["top_k_experts"]

    def config_expert_count_key(self) -> str:
        return "num_experts"

    def get_gate_module(self, moe_block):
        return moe_block.router

    def intermediate_size(self) -> int:
        return self._moe_config["moe_intermediate_size"]
