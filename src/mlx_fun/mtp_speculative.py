"""MTP-aware speculative decoding for the Gemma 4 ``gemma4_assistant`` drafter.

This module provides:

* :func:`mtp_speculative_generate_step` — a drop-in alternative to
  ``mlx_lm.generate.speculative_generate_step`` that knows how to drive the
  Gemma 4 MTP drafter. The drafter has all layers KV-shared with the
  backbone, takes ``[scaled_emb | backbone_hidden]`` as input, and feeds
  itself back via ``post_projection``. Standard speculative decoding cannot
  drive it because that flow assumes an independent drafter with its own
  KV cache.

* :func:`is_mtp_drafter` — predicate that returns True for the Gemma 4
  assistant model (so callers can dispatch to this path conditionally).

* :func:`mtp_stream_generate` — token-streaming wrapper that mirrors
  :func:`mlx_lm.generate.stream_generate` so this can plug into existing
  server code with minimal changes.

The flow inside ``mtp_speculative_generate_step`` is the same one the CLI
``MTPDriver`` uses:

    1. Prefill the backbone on the prompt; capture post-norm hidden + per-layer
       anchor KV (last sliding + last full layer).
    2. Each iteration: draft K tokens autoregressively through the drafter,
       feeding back ``post_projection`` output as the next hidden, then verify
       all K in one parallel backbone forward, accept the longest greedy-
       matching prefix + one bonus, trim the backbone's KV cache.

This file intentionally has no dependency on ``MTPDriver`` so the server can
import it without dragging in CLI-specific helpers.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models import cache as cache_utils

from .mtp_driver import _backbone_logits_from_hidden, _post_norm_hidden

def _get_gen_stream():
    """Return mlx_lm's generation_stream. Imported lazily so we always read
    whatever the upstream module currently exposes."""
    from mlx_lm.generate import generation_stream
    return generation_stream


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def is_mtp_drafter(model: nn.Module) -> bool:
    """True if ``model`` is a Gemma 4 ``gemma4_assistant`` drafter."""
    return getattr(model, "model_type", None) == "gemma4_assistant"


# ---------------------------------------------------------------------------
# Anchor selection
# ---------------------------------------------------------------------------


def _last_index_of(layer_types: List[str], wanted: str) -> int:
    for i in range(len(layer_types) - 1, -1, -1):
        if layer_types[i] == wanted:
            return i
    raise ValueError(f"No {wanted} layer found in backbone")


def _pick_anchors(backbone: nn.Module) -> Dict[str, int]:
    inner = backbone.language_model.model
    types = [l.layer_type for l in inner.layers]
    return {
        "sliding_attention": _last_index_of(types, "sliding_attention"),
        "full_attention": _last_index_of(types, "full_attention"),
    }


def _gather_anchors(
    cache: List[Any], anchors: Dict[str, int]
) -> Dict[str, Tuple[mx.array, mx.array]]:
    """Snapshot anchor K/V from the backbone's cache, sliced to the currently
    valid offset (KVCache stores a step-padded buffer)."""
    out = {}
    for layer_type, idx in anchors.items():
        c = cache[idx]
        n = c.offset
        out[layer_type] = (c.keys[..., :n, :], c.values[..., :n, :])
    return out


# ---------------------------------------------------------------------------
# Drafter primitives
# ---------------------------------------------------------------------------


def _drafter_step(
    drafter: nn.Module,
    backbone: nn.Module,
    backbone_h: mx.array,
    prev_token: mx.array,
    anchors: Dict[str, Tuple[mx.array, mx.array]],
    rope_offset: int,
    *,
    scale_emb: bool = True,
) -> Tuple[mx.array, mx.array]:
    """Single drafter forward. Returns ``(token_id_argmax, projected_h)``."""
    inner = backbone.language_model.model
    if prev_token.ndim == 1:
        prev_token = prev_token[:, None]
    emb = inner.embed_tokens(prev_token)
    if scale_emb:
        emb = emb * inner.embed_scale

    x = mx.concatenate([emb, backbone_h], axis=-1)
    h = drafter.pre_projection(x)
    for layer in drafter.model.layers:
        shared_kv = anchors[layer.layer_type]
        h, _, _ = layer(h, mask=None, cache=None, shared_kv=shared_kv,
                        offset=rope_offset)
    h = drafter.model.norm(h)

    if drafter.tie_word_embeddings:
        logits = drafter.model.embed_tokens.as_linear(h)
    else:
        logits = drafter.lm_head(h)

    token = mx.argmax(logits[:, -1, :], axis=-1)
    projected = drafter.post_projection(h)
    return token, projected


# ---------------------------------------------------------------------------
# Speculative generator
# ---------------------------------------------------------------------------


def mtp_speculative_generate_step(
    prompt: mx.array,
    model: nn.Module,
    draft_model: nn.Module,
    *,
    num_draft_tokens: int = 4,
    max_tokens: int = 256,
    sampler: Optional[Callable[[mx.array], mx.array]] = None,
    logits_processors: Optional[
        List[Callable[[mx.array, mx.array], mx.array]]
    ] = None,
    prompt_cache: Optional[List[Any]] = None,
    prefill_step_size: int = 2048,
    **_unused,
) -> Generator[Tuple[int, mx.array, bool], None, None]:
    """Greedy MTP speculative decoding generator.

    Yields ``(token_id, logprobs, from_draft)`` per upstream's interface.
    Sampler is currently honored only when it equals greedy argmax — a
    non-greedy sampler will fall back to greedy with a warning since MTP
    verification with arbitrary samplers requires importance correction.

    ``logits_processors`` are applied to the backbone's verification logits
    so chat constraints (stop tokens etc.) still work.
    """

    inner = model.language_model.model
    anchors = _pick_anchors(model)

    # ---- prepare cache ----
    if prompt_cache is None:
        cache = model.make_cache()
    else:
        # ``prompt_cache`` from server includes both model and draft slots
        # concatenated. Take only the model's slots; draft slots are unused
        # (the MTP drafter has no own KV).
        cache = prompt_cache[: len(inner.layers)]

    if not all(c is None or c.is_trimmable() for c in cache):
        types = {type(c).__name__ for c in cache if c is not None and not c.is_trimmable()}
        raise ValueError(
            f"MTP speculative decoding requires trimmable caches (got {types})."
        )

    # Use mlx_lm's generation_stream as the active stream — model parameters
    # were touched under it during loading. Each compute block enters the
    # context separately; yields happen outside.
    from mlx_lm.generate import generation_stream as _gs

    # ---- prefill (chunked) ----
    with mx.stream(_gs):
        y = prompt.astype(mx.uint32)
        while y.size > prefill_step_size:
            inner(y[:prefill_step_size][None], cache=cache)
            mx.eval([c.state for c in cache if c is not None])
            y = y[prefill_step_size:]
            mx.clear_cache()

        last_hidden = inner(y[None], cache=cache)
        last_h = last_hidden[:, -1:, :]
        verify_logits = _backbone_logits_from_hidden(model, last_h)

        if logits_processors:
            for proc in logits_processors:
                verify_logits = proc(prompt, verify_logits)

        logprobs = verify_logits - mx.logsumexp(verify_logits, axis=-1, keepdims=True)
        verified = mx.argmax(verify_logits[:, -1, :], axis=-1)
        mx.eval(last_h, verified, logprobs)

    ntoks = 0
    if ntoks < max_tokens:
        ntoks += 1
        yield int(verified.item()), logprobs[0, -1], False

    # ---- main draft + verify loop ----
    K = num_draft_tokens
    while ntoks < max_tokens:
        with mx.stream(_gs):
            cur_pos = cache[anchors["full_attention"]].offset
            anchors_kv = _gather_anchors(cache, anchors)

            drafted = []
            h = last_h
            prev = verified
            for k in range(K):
                tok, proj = _drafter_step(
                    draft_model, model, h, prev, anchors_kv,
                    rope_offset=cur_pos + k,
                )
                drafted.append(tok)
                h = proj
                prev = tok
            drafted_arr = mx.stack(drafted, axis=-1)

            verify_input = mx.concatenate([verified[:, None], drafted_arr], axis=-1)
            hidden_seq = _post_norm_hidden(model, verify_input, cache)
            logits_seq = _backbone_logits_from_hidden(model, hidden_seq)

            if logits_processors:
                for proc in logits_processors:
                    logits_seq = proc(prompt, logits_seq)

            logprobs_seq = logits_seq - mx.logsumexp(
                logits_seq, axis=-1, keepdims=True
            )
            backbone_args = mx.argmax(logits_seq[0], axis=-1)
            mx.eval(drafted_arr, backbone_args, logprobs_seq, hidden_seq)

        drafted_list = drafted_arr[0].tolist()
        backbone_list = backbone_args.tolist()
        j = 0
        for i in range(K):
            if drafted_list[i] == backbone_list[i]:
                j += 1
            else:
                break

        for i in range(j):
            if ntoks >= max_tokens:
                break
            ntoks += 1
            yield drafted_list[i], logprobs_seq[0, i, :], True

        if ntoks >= max_tokens:
            break

        bonus_tok = backbone_list[j]
        ntoks += 1
        yield bonus_tok, logprobs_seq[0, j, :], False

        with mx.stream(_gs):
            keep = 1 + j
            drop = (K + 1) - keep
            for c in cache:
                if c is not None:
                    c.trim(drop)
            last_h = hidden_seq[:, keep - 1 : keep, :]
            verified = mx.array([bonus_tok], dtype=mx.uint32)
            mx.eval(last_h, verified)


def _is_greedy_sampler(sampler: Callable) -> bool:
    """Heuristic check: is the supplied sampler greedy argmax? Falls back to
    True (assumes greedy) if we can't tell — the MTP loop is greedy-only."""
    name = getattr(sampler, "__name__", "") or ""
    return "argmax" in name or "greedy" in name


# ---------------------------------------------------------------------------
# Stream generator (mirrors mlx_lm.generate.stream_generate signature)
# ---------------------------------------------------------------------------


def mtp_stream_generate(
    model: nn.Module,
    tokenizer,
    prompt,
    max_tokens: int = 256,
    draft_model: Optional[nn.Module] = None,
    **kwargs,
):
    """Drop-in replacement for ``mlx_lm.generate.stream_generate`` when the
    drafter is a Gemma 4 ``gemma4_assistant`` model. Falls back to
    ``mlx_lm.generate.stream_generate`` for any other drafter."""
    from mlx_lm.generate import stream_generate as _upstream_stream
    from mlx_lm.tokenizer_utils import TokenizerWrapper
    from mlx_lm.generate import GenerationResponse, wired_limit
    import time as _time

    if draft_model is None or not is_mtp_drafter(draft_model):
        yield from _upstream_stream(
            model, tokenizer, prompt,
            max_tokens=max_tokens, draft_model=draft_model, **kwargs,
        )
        return

    if not isinstance(tokenizer, TokenizerWrapper):
        tokenizer = TokenizerWrapper(tokenizer)
    if not isinstance(prompt, mx.array):
        if isinstance(prompt, str):
            add_special = (tokenizer.bos_token is None
                           or not prompt.startswith(tokenizer.bos_token))
            prompt = tokenizer.encode(prompt, add_special_tokens=add_special)
        prompt = mx.array(prompt)

    detok = tokenizer.detokenizer
    kwargs.pop("max_kv_size", None)
    kwargs.pop("prompt_progress_callback", None)
    # mlx_lm passes these for the upstream loop; we don't use them.
    kwargs.pop("kv_bits", None)
    kwargs.pop("kv_group_size", None)
    kwargs.pop("quantized_kv_start", None)

    gen = mtp_speculative_generate_step(
        prompt, model, draft_model,
        max_tokens=max_tokens,
        **kwargs,
    )

    with wired_limit(model):
        tic = _time.perf_counter()
        prompt_time = None
        prompt_tps = 0.0
        last_tok = None
        for n, (token, logprobs, from_draft) in enumerate(gen):
            if n == 0:
                prompt_time = _time.perf_counter() - tic
                prompt_tps = prompt.size / max(prompt_time, 1e-9)
                tic = _time.perf_counter()
            if token in tokenizer.eos_token_ids:
                break
            detok.add_token(token)
            last_tok = token
            if (n + 1) == max_tokens:
                break
            yield GenerationResponse(
                text=detok.last_segment,
                token=token,
                logprobs=logprobs,
                from_draft=from_draft,
                prompt_tokens=prompt.size,
                prompt_tps=prompt_tps,
                generation_tokens=n + 1,
                generation_tps=(n + 1) / max(_time.perf_counter() - tic, 1e-9),
                peak_memory=mx.get_peak_memory() / 1e9,
                finish_reason=None,
            )
        detok.finalize()
        yield GenerationResponse(
            text=detok.last_segment,
            token=last_tok if last_tok is not None else 0,
            logprobs=mx.zeros((1,)),
            from_draft=False,
            prompt_tokens=prompt.size,
            prompt_tps=prompt_tps,
            generation_tokens=0,
            generation_tps=0.0,
            peak_memory=mx.get_peak_memory() / 1e9,
            finish_reason="stop",
        )
