"""DFlash block diffusion draft model for speculative decoding (Phase 3).

A lightweight transformer that generates token BLOCKS in parallel, conditioned
on hidden states captured from the target model.  Reuses the target model's
embeddings and LM head — only the small transformer body has its own weights.

Architecture (ported from https://github.com/z-lab/dflash):

    1.  Target hidden states from selected layers are concatenated along the
        feature dimension and projected to ``embed_dim`` via a linear + RMSNorm.
    2.  Each DFlash decoder layer uses *cross-self attention*: Q comes from the
        draft hidden states while K/V are built from
        ``concat(target_hidden, draft_hidden)`` so the draft attends to both
        the target context and its own tokens in a single attention op.
    3.  The final norm output is passed through the target model's LM head
        to produce logits for an entire block of tokens in parallel.

Block sizes are configurable (``b16``, ``b32`` → 16 or 32 tokens per block).
"""

from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def build_target_layer_ids(
    num_target_layers: int,
    num_draft_layers: int,
) -> List[int]:
    """Select which target-model layers to tap for conditioning.

    Evenly spaces ``num_draft_layers`` indices across the target model,
    skipping the very first and last few layers (following the reference
    implementation).

    >>> build_target_layer_ids(32, 5)
    [1, 8, 15, 22, 28]
    """
    if num_draft_layers == 1:
        return [num_target_layers // 2]
    start = 1
    end = num_target_layers - 3
    span = end - start
    return [
        int(round(start + (i * span) / (num_draft_layers - 1)))
        for i in range(num_draft_layers)
    ]


def extract_context_feature(
    hidden_states: dict,
    layer_ids: List[int],
) -> mx.array:
    """Concatenate hidden states from selected target layers along the feature dim.

    Args:
        hidden_states: ``{layer_idx: mx.array}`` mapping from
            :class:`~mlx_fun.hidden_state_capture.HiddenStateCapture.collect_latest`.
        layer_ids: Which layer indices to select and concatenate.

    Returns:
        ``mx.array`` of shape ``(batch, seq_len, len(layer_ids) * hidden_dim)``.
    """
    selected = [hidden_states[lid] for lid in layer_ids]
    return mx.concatenate(selected, axis=-1)


def parse_block_size(value: str) -> int:
    """Parse a block-size CLI string like ``'b16'`` or ``'32'`` into an int.

    >>> parse_block_size("b16")
    16
    >>> parse_block_size("32")
    32
    """
    v = value.strip().lower()
    if v.startswith("b"):
        v = v[1:]
    n = int(v)
    if n <= 0:
        raise ValueError(f"Block size must be positive, got {n}")
    return n


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------

class DFlashRMSNorm(nn.Module):
    """RMSNorm matching MLX conventions."""

    def __init__(self, dims: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        dtype = x.dtype
        x = x.astype(mx.float32)
        rms = mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps)
        return (x * rms).astype(dtype) * self.weight


def _rotate_half(x: mx.array) -> mx.array:
    """Rotary embedding helper: split in half and rotate."""
    mid = x.shape[-1] // 2
    x1 = x[..., :mid]
    x2 = x[..., mid:]
    return mx.concatenate([-x2, x1], axis=-1)


def _apply_rotary_pos_emb(
    q: mx.array,
    k: mx.array,
    cos: mx.array,
    sin: mx.array,
) -> Tuple[mx.array, mx.array]:
    """Apply rotary position embeddings to Q and K.

    ``cos``/``sin`` have shape ``(1, seq_len, head_dim)`` — we unsqueeze to
    broadcast over heads.  Q may be shorter than K (block tokens only) so we
    slice cos/sin from the end.
    """
    # cos/sin: (1, total_seq, head_dim) → (1, 1, total_seq, head_dim)
    cos = mx.expand_dims(cos, axis=1)
    sin = mx.expand_dims(sin, axis=1)
    q_len = q.shape[2]  # q is (B, heads, q_len, head_dim)
    q_embed = (q * cos[..., -q_len:, :]) + (_rotate_half(q) * sin[..., -q_len:, :])
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


class DFlashRotaryEmbedding(nn.Module):
    """Precomputes rotary cos/sin tables."""

    def __init__(self, head_dim: int, max_seq_len: int = 8192, base: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (
            base ** (mx.arange(0, head_dim, 2).astype(mx.float32) / head_dim)
        )
        self._inv_freq = inv_freq

    def __call__(self, x: mx.array, position_ids: mx.array) -> Tuple[mx.array, mx.array]:
        """Return (cos, sin) each of shape ``(1, seq_len, head_dim)``."""
        # position_ids: (1, seq_len) or (batch, seq_len)
        seq_len = position_ids.shape[-1]
        t = position_ids[0].astype(mx.float32)  # (seq_len,)
        freqs = mx.outer(t, self._inv_freq)  # (seq_len, head_dim/2)
        emb = mx.concatenate([freqs, freqs], axis=-1)  # (seq_len, head_dim)
        cos = mx.cos(emb)
        sin = mx.sin(emb)
        return mx.expand_dims(cos, axis=0), mx.expand_dims(sin, axis=0)


class DFlashMLP(nn.Module):
    """SwiGLU MLP (gate + up → silu → down)."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class DFlashAttention(nn.Module):
    """Cross-self attention: Q from draft, K/V from concat(target, draft).

    This is the core DFlash mechanism — the draft model attends to the target
    model's hidden states (cross-attention) and its own tokens (self-attention)
    in a single fused attention operation.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)

        self.q_norm = DFlashRMSNorm(self.head_dim)
        self.k_norm = DFlashRMSNorm(self.head_dim)

    def __call__(
        self,
        hidden_states: mx.array,
        target_hidden: mx.array,
        position_embeddings: Tuple[mx.array, mx.array],
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        B, q_len, _ = hidden_states.shape
        ctx_len = target_hidden.shape[1]

        # Q from draft tokens
        q = self.q_proj(hidden_states)
        q = q.reshape(B, q_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        q = self.q_norm(q)

        # K/V from concat(target, draft)
        k_ctx = self.k_proj(target_hidden)
        k_draft = self.k_proj(hidden_states)
        v_ctx = self.v_proj(target_hidden)
        v_draft = self.v_proj(hidden_states)

        k = mx.concatenate([k_ctx, k_draft], axis=1)
        v = mx.concatenate([v_ctx, v_draft], axis=1)

        k = k.reshape(B, ctx_len + q_len, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, ctx_len + q_len, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_norm(k)

        # GQA expansion
        if self.num_kv_heads < self.num_heads:
            n_rep = self.num_heads // self.num_kv_heads
            k = mx.repeat(k, n_rep, axis=1)
            v = mx.repeat(v, n_rep, axis=1)

        # Rotary embeddings
        cos, sin = position_embeddings
        q, k = _apply_rotary_pos_emb(q, k, cos, sin)

        # Scaled dot-product attention (non-causal)
        attn_weights = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        if mask is not None:
            attn_weights = attn_weights + mask
        attn_weights = mx.softmax(attn_weights, axis=-1)
        attn_output = attn_weights @ v

        # (B, heads, q_len, head_dim) → (B, q_len, hidden_size)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(B, q_len, -1)
        return self.o_proj(attn_output)


class DFlashDecoderLayer(nn.Module):
    """Single DFlash transformer layer: cross-self attention + SwiGLU MLP."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        num_kv_heads: Optional[int] = None,
    ):
        super().__init__()
        self.self_attn = DFlashAttention(hidden_size, num_heads, num_kv_heads)
        self.mlp = DFlashMLP(hidden_size, intermediate_size)
        self.input_layernorm = DFlashRMSNorm(hidden_size)
        self.post_attention_layernorm = DFlashRMSNorm(hidden_size)

    def __call__(
        self,
        hidden_states: mx.array,
        target_hidden: mx.array,
        position_embeddings: Tuple[mx.array, mx.array],
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        # Pre-norm attention with residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            target_hidden=target_hidden,
            position_embeddings=position_embeddings,
            mask=mask,
        )
        hidden_states = residual + hidden_states

        # Pre-norm MLP with residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


# ---------------------------------------------------------------------------
# Main draft model
# ---------------------------------------------------------------------------

class BlockDiffusionDraftModel(nn.Module):
    """DFlash block diffusion draft model for speculative decoding.

    A small transformer (~5 layers) that generates an entire block of tokens
    in parallel, conditioned on hidden states from the target model.

    The model **reuses** the target model's embeddings (``embed_tokens``) and
    LM head — only the transformer body and the conditioning projection have
    learnable weights.

    Args:
        embed_dim: Hidden dimension (must match target model's hidden size).
        num_layers: Number of DFlash transformer layers (typically 5).
        num_heads: Number of attention heads.
        block_size: Tokens per block (16 or 32).
        num_target_layers: Total layers in the target model (for
            ``build_target_layer_ids``).
        intermediate_size: MLP intermediate dimension.  Defaults to
            ``4 * embed_dim``.
        num_kv_heads: Number of KV heads for GQA.  Defaults to ``num_heads``.
        target_layer_ids: Explicit target layer indices to use for
            conditioning.  If ``None``, computed automatically via
            ``build_target_layer_ids``.
        target_embed_tokens: Reference to the target model's embedding layer.
            Set via :meth:`attach_target` or constructor.
        target_lm_head: Reference to the target model's LM head.
            Set via :meth:`attach_target` or constructor.
    """

    def __init__(
        self,
        embed_dim: int,
        num_layers: int = 5,
        num_heads: int = 8,
        block_size: int = 16,
        num_target_layers: int = 32,
        intermediate_size: Optional[int] = None,
        num_kv_heads: Optional[int] = None,
        target_layer_ids: Optional[List[int]] = None,
        target_embed_tokens: Optional[nn.Module] = None,
        target_lm_head: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers_count = num_layers
        self.block_size = block_size
        self.num_target_layers = num_target_layers

        if intermediate_size is None:
            intermediate_size = 4 * embed_dim

        # Target layer mapping
        if target_layer_ids is not None:
            self.target_layer_ids = list(target_layer_ids)
        else:
            self.target_layer_ids = build_target_layer_ids(
                num_target_layers, num_layers,
            )

        # Conditioning projection: concat of N target hidden states → embed_dim
        num_conditioning_layers = len(self.target_layer_ids)
        self.fc = nn.Linear(
            num_conditioning_layers * embed_dim, embed_dim, bias=False,
        )
        self.hidden_norm = DFlashRMSNorm(embed_dim)

        # Transformer body
        self.layers = [
            DFlashDecoderLayer(
                hidden_size=embed_dim,
                num_heads=num_heads,
                intermediate_size=intermediate_size,
                num_kv_heads=num_kv_heads,
            )
            for _ in range(num_layers)
        ]
        self.norm = DFlashRMSNorm(embed_dim)
        self.rotary_emb = DFlashRotaryEmbedding(
            head_dim=embed_dim // num_heads,
        )

        # Shared target model references (not owned — no parameters)
        self._target_embed_tokens = target_embed_tokens
        self._target_lm_head = target_lm_head

    def attach_target(self, model) -> None:
        """Attach target model's embed_tokens and lm_head.

        Args:
            model: Target model with ``model.model.embed_tokens`` and
                ``model.lm_head`` attributes (standard mlx-lm structure).
        """
        self._target_embed_tokens = model.model.embed_tokens
        self._target_lm_head = model.lm_head

    @property
    def target_layer_ids_list(self) -> List[int]:
        return list(self.target_layer_ids)

    def _project_target_hidden(
        self,
        hidden_states: dict,
    ) -> mx.array:
        """Project captured target hidden states into conditioning vector.

        Args:
            hidden_states: ``{layer_idx: mx.array}`` from HiddenStateCapture.

        Returns:
            Projected conditioning tensor of shape ``(B, seq, embed_dim)``.
        """
        context = extract_context_feature(hidden_states, self.target_layer_ids)
        return self.hidden_norm(self.fc(context))

    def __call__(
        self,
        target_hidden_states: dict,
        block_token_ids: mx.array,
        position_ids: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        """Forward pass: generate logits for an entire block of tokens.

        Args:
            target_hidden_states: ``{layer_idx: mx.array}`` captured from the
                target model via :class:`HiddenStateCapture`.
            block_token_ids: Token IDs for the block, shape ``(B, block_size)``.
                On the first iteration these are mask/pad tokens; on subsequent
                iterations they are the previously drafted tokens.
            position_ids: Position indices, shape ``(B, ctx_len + block_size)``.
                If ``None``, auto-generated as ``arange(ctx_len + block_size)``.
            mask: Optional attention mask.

        Returns:
            Logits tensor of shape ``(B, block_size - 1, vocab_size)`` — one
            logit per block position (excluding the first, which is conditioned
            on the target model's last verified token).
        """
        if self._target_embed_tokens is None or self._target_lm_head is None:
            raise RuntimeError(
                "Target model not attached. Call attach_target(model) first."
            )

        # 1. Project target hidden states → conditioning
        target_hidden = self._project_target_hidden(target_hidden_states)
        ctx_len = target_hidden.shape[1]

        # 2. Embed block tokens via target embeddings
        noise_embedding = self._target_embed_tokens(block_token_ids)
        B, block_len, _ = noise_embedding.shape

        # 3. Compute rotary embeddings for full context + block
        total_len = ctx_len + block_len
        if position_ids is None:
            position_ids = mx.arange(total_len).reshape(1, -1)
        position_embeddings = self.rotary_emb(noise_embedding, position_ids)

        # 4. Run through DFlash transformer layers
        hidden_states = noise_embedding
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                target_hidden=target_hidden,
                position_embeddings=position_embeddings,
                mask=mask,
            )

        # 5. Final norm
        hidden_states = self.norm(hidden_states)

        # 6. Project to logits via target LM head (skip first token)
        draft_logits = self._target_lm_head(hidden_states[:, 1:, :])

        return draft_logits

    def draft_block(
        self,
        target_hidden_states: dict,
        first_token_id: mx.array,
        temperature: float = 0.0,
        mask_token_id: int = 0,
    ) -> mx.array:
        """Draft a full block of tokens given target hidden states.

        Convenience method that creates mask-filled block tokens, runs the
        forward pass, and samples output tokens.

        Args:
            target_hidden_states: Captured hidden states from target model.
            first_token_id: The last verified token, shape ``(B, 1)`` or ``(B,)``.
            temperature: Sampling temperature (0 = greedy argmax).
            mask_token_id: Token ID used for ungenerated positions.

        Returns:
            Drafted token IDs of shape ``(B, block_size)``.  Position 0 is
            ``first_token_id``; positions 1..block_size-1 are sampled from
            the draft model's logits.
        """
        if first_token_id.ndim == 1:
            first_token_id = mx.expand_dims(first_token_id, axis=-1)

        B = first_token_id.shape[0]
        # Build block: [first_token, mask, mask, ..., mask]
        block = mx.full((B, self.block_size), mask_token_id, dtype=mx.int32)
        block = mx.concatenate(
            [first_token_id.astype(mx.int32), block[:, 1:]], axis=1,
        )

        # Forward pass → logits for positions 1..block_size-1
        logits = self(target_hidden_states, block)  # (B, block_size-1, vocab)

        # Sample
        if temperature < 1e-5:
            sampled = mx.argmax(logits, axis=-1)  # (B, block_size-1)
        else:
            B_flat, seq, vocab = logits.shape
            logits_flat = logits.reshape(-1, vocab) / temperature
            probs = mx.softmax(logits_flat, axis=-1)
            sampled = mx.random.categorical(probs).reshape(B_flat, seq)

        # Assemble: first_token + sampled
        result = mx.concatenate(
            [first_token_id.astype(mx.int32), sampled.astype(mx.int32)],
            axis=1,
        )
        return result


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_dflash_draft_model(
    target_model,
    num_layers: int = 5,
    num_heads: int = 8,
    block_size: int = 16,
    intermediate_size: Optional[int] = None,
    num_kv_heads: Optional[int] = None,
    target_layer_ids: Optional[List[int]] = None,
) -> BlockDiffusionDraftModel:
    """Create a DFlash draft model sized to match a target model.

    Reads ``embed_dim`` and ``num_target_layers`` from the target model's
    structure and attaches the shared embeddings/LM head.

    Args:
        target_model: The target MLX model (must have
            ``model.model.layers``, ``model.model.embed_tokens``,
            ``model.lm_head``).
        num_layers: Number of DFlash transformer layers.
        num_heads: Attention heads in the draft model.
        block_size: Tokens per block (16 or 32).
        intermediate_size: MLP hidden dim (default: 4 * embed_dim).
        num_kv_heads: KV heads for GQA (default: same as num_heads).
        target_layer_ids: Override automatic target layer selection.

    Returns:
        Initialized :class:`BlockDiffusionDraftModel` with target refs attached.
    """
    # Infer embed_dim from the target model's embedding layer
    embed_tokens = target_model.model.embed_tokens
    embed_dim = embed_tokens.weight.shape[1]
    num_target_layers = len(target_model.model.layers)

    draft = BlockDiffusionDraftModel(
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        block_size=block_size,
        num_target_layers=num_target_layers,
        intermediate_size=intermediate_size,
        num_kv_heads=num_kv_heads,
        target_layer_ids=target_layer_ids,
    )
    draft.attach_target(target_model)

    return draft
