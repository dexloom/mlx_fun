# Copyright © 2026 dexloom
#
# Vendored into mlx_fun so no fork of mlx-lm is required. Registered as
# ``mlx_lm.models.gemma4_assistant`` by ``mlx_fun.models.register_model_types()``
# so stock ``mlx_lm.load()`` resolves the ``gemma4_assistant`` model_type
# through its normal importlib lookup.
#
# Gemma 4 Assistant — Multi-Token Prediction (MTP) drafter.
#
# This model is a small (4-layer) transformer that runs alongside a Gemma 4
# backbone to speculatively predict several tokens ahead. It is not designed
# to generate text on its own: every layer reuses the backbone's keys/values
# (num_kv_shared_layers == num_hidden_layers), and inputs are the backbone's
# hidden state concatenated with the next-token embedding (in backbone hidden
# space), projected down to the drafter's hidden size.
#
# Tensor layout (per ~/.lmstudio/models/sombra/gemma-4-31B-it-assistant):
#   model.embed_tokens.weight                                       [V, H]
#   model.layers.{i}.input_layernorm.weight                         [H]
#   model.layers.{i}.layer_scalar                                   [1]
#   model.layers.{i}.mlp.{gate,up}_proj.weight                      [I, H]
#   model.layers.{i}.mlp.down_proj.weight                           [H, I]
#   model.layers.{i}.{post_attention,pre_feedforward,
#                     post_feedforward}_layernorm.weight            [H]
#   model.layers.{i}.self_attn.q_proj.weight                        [n_h*d_h, H]
#   model.layers.{i}.self_attn.o_proj.weight                        [H, n_h*d_h]
#   model.layers.{i}.self_attn.q_norm.weight                        [d_h]
#   model.norm.weight                                               [H]
#   pre_projection.weight                                           [H, 2*B]
#   post_projection.weight                                          [B, H]
# where H = drafter hidden_size, B = backbone_hidden_size, V = vocab_size,
# n_h = num_attention_heads, d_h = head_dim (or global_head_dim on full layer).

from dataclasses import dataclass
from typing import Any, Dict, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models import gemma4_text
from mlx_lm.models.base import BaseModelArgs
from mlx_lm.models.cache import KVCache, RotatingKVCache


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "gemma4_assistant"
    text_config: Dict[str, Any] = None
    backbone_hidden_size: int = 5376
    vocab_size: int = 262144
    tie_word_embeddings: bool = True
    num_centroids: Optional[int] = None
    centroid_intermediate_top_k: Optional[int] = None
    use_ordered_embeddings: bool = False

    def __post_init__(self):
        if self.text_config is None:
            self.text_config = {}
        # The HF text_config sets num_kv_shared_layers == num_hidden_layers,
        # which is the right value for the gemma4_text Attention/MLP modules.
        self.text_config.setdefault("vocab_size", self.vocab_size)
        self.text_config.setdefault("tie_word_embeddings", self.tie_word_embeddings)


class _AssistantInner(nn.Module):
    """Mirror of gemma4_text.Gemma4TextModel without per-layer-input or
    full-layer KV plumbing, since every layer of the assistant is KV-shared
    against the backbone."""

    def __init__(self, config: gemma4_text.ModelArgs):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            gemma4_text.DecoderLayer(config, layer_idx=i)
            for i in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.text_args = gemma4_text.ModelArgs.from_dict(args.text_config)
        self.tie_word_embeddings = self.text_args.tie_word_embeddings
        self.backbone_hidden_size = args.backbone_hidden_size

        self.model = _AssistantInner(self.text_args)
        self.pre_projection = nn.Linear(
            2 * args.backbone_hidden_size,
            self.text_args.hidden_size,
            bias=False,
        )
        self.post_projection = nn.Linear(
            self.text_args.hidden_size,
            args.backbone_hidden_size,
            bias=False,
        )
        if not self.tie_word_embeddings:
            self.lm_head = nn.Linear(
                self.text_args.hidden_size, self.text_args.vocab_size, bias=False
            )

    @property
    def layers(self):
        return self.model.layers

    def _make_masks(self, h, cache):
        from mlx_lm.models.base import create_attention_mask

        window = self.text_args.sliding_window
        masks_by_type = {}
        masks = []
        for layer, c in zip(self.model.layers, cache):
            t = layer.layer_type
            if t not in masks_by_type:
                if t == "full_attention":
                    masks_by_type[t] = create_attention_mask(h, c)
                else:
                    masks_by_type[t] = create_attention_mask(h, c, window_size=window)
            masks.append(masks_by_type[t])
        return masks

    def _run_layers(self, h, cache, shared_kv_by_type):
        """Run the 4 drafter layers. ``shared_kv_by_type`` provides keys/values
        from the backbone, keyed by layer_type ("sliding_attention" or
        "full_attention"). When None, every layer must already have its own KV
        in ``cache`` (from a previous step) or this will raise."""
        if cache is None:
            cache = [None] * len(self.model.layers)

        masks = self._make_masks(h, cache)
        for layer, c, mask in zip(self.model.layers, cache, masks):
            shared_kv = (
                shared_kv_by_type.get(layer.layer_type)
                if shared_kv_by_type is not None
                else None
            )
            h, _, _ = layer(h, mask=mask, cache=c, shared_kv=shared_kv, offset=0)
        return self.model.norm(h)

    def __call__(
        self,
        inputs: Optional[mx.array] = None,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
        backbone_hidden_states: Optional[mx.array] = None,
        next_token_embeddings: Optional[mx.array] = None,
        shared_kv_by_type: Optional[Dict[str, tuple]] = None,
    ):
        """Forward pass.

        For drafter usage, callers supply ``backbone_hidden_states`` (last
        backbone layer output, B-dim) and ``next_token_embeddings`` (the
        backbone embed_tokens output for the next-position token, also B-dim).
        These are concatenated and projected to the drafter hidden size, run
        through 4 transformer layers using ``shared_kv_by_type`` for KV, then
        projected back to backbone hidden size.

        Returned tuple: (vocab_logits, projected_hidden_in_backbone_space).

        For loading-only / smoke testing, callers may pass ``inputs`` (token
        ids) or ``input_embeddings`` (already 2*B-dim) — when only token ids
        are given, a zero-padded backbone hidden state is synthesized so the
        shapes match. This standalone path will not produce coherent output
        but lets the model run end-to-end without a backbone."""
        if backbone_hidden_states is not None:
            if next_token_embeddings is None:
                raise ValueError(
                    "next_token_embeddings must accompany backbone_hidden_states"
                )
            x = mx.concatenate(
                [backbone_hidden_states, next_token_embeddings], axis=-1
            )
        elif input_embeddings is not None:
            x = input_embeddings
        else:
            if inputs is None:
                raise ValueError("inputs or embeddings must be provided")
            # Standalone fallback: embed tokens with the drafter's own table,
            # tile to fill the 2*B input slot.
            embed = self.model.embed_tokens(inputs)  # [B, L, H]
            zeros = mx.zeros(
                (*embed.shape[:-1], self.backbone_hidden_size), dtype=embed.dtype
            )
            x = mx.concatenate([zeros, zeros], axis=-1)

        h = self.pre_projection(x)
        h = self._run_layers(h, cache, shared_kv_by_type)

        if self.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(h)
        else:
            logits = self.lm_head(h)

        projected = self.post_projection(h)
        return logits, projected

    def make_cache(self):
        # Drafter layers all reuse backbone KV — no per-layer cache is needed
        # here. We still return per-layer slots so callers that index by layer
        # don't blow up; each slot is None.
        return [None] * self.text_args.num_hidden_layers

    def sanitize(self, weights):
        sanitized = {}
        first_kv_shared = (
            self.text_args.num_hidden_layers - self.text_args.num_kv_shared_layers
        )
        for k, v in weights.items():
            if any(
                s in k
                for s in (
                    "self_attn.rotary_emb",
                    "input_max",
                    "input_min",
                    "output_max",
                    "output_min",
                )
            ):
                continue

            # Drop K/V projections for KV-shared layers, mirroring gemma4_text.
            if any(
                s in k
                for s in (
                    ".self_attn.k_proj",
                    ".self_attn.v_proj",
                    ".self_attn.k_norm",
                )
            ):
                try:
                    layer_idx = int(k.split("layers.")[1].split(".")[0])
                    if layer_idx >= first_kv_shared:
                        continue
                except (IndexError, ValueError):
                    pass

            sanitized[k] = v

        return sanitized

    @property
    def quant_predicate(self):
        def predicate(path, _):
            # Don't quantize the backbone-bridge projections — they're small
            # and on the critical path for every drafted token.
            if path in ("pre_projection", "post_projection"):
                return False
            return True

        return predicate

    @property
    def head_dim(self):
        return self.text_args.head_dim

    @property
    def n_kv_heads(self):
        return self.text_args.num_key_value_heads
