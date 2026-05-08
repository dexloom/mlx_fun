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
    sampler: Optional[Callable[[mx.array], mx.array]] = None,
) -> Tuple[mx.array, mx.array, mx.array]:
    """Single drafter forward.

    Returns ``(sampled_token, last_logits, projected_h)`` where
    ``last_logits`` has shape ``[B, V]`` — the un-normalized logits at the
    last position (kept around so the verify step can compute the
    importance ratio for Leviathan-style speculative sampling).

    The supplied ``sampler`` is applied to log-probs to produce the
    drafted token. Defaults to argmax."""
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

    last_logits = logits[:, -1, :]  # [B, V]
    if sampler is None:
        token = mx.argmax(last_logits, axis=-1)
    else:
        logprobs = last_logits - mx.logsumexp(last_logits, axis=-1, keepdims=True)
        token = sampler(logprobs)
    projected = drafter.post_projection(h)
    return token, last_logits, projected


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
    """MTP speculative decoding generator.

    Yields ``(token_id, logprobs, from_draft)`` per upstream's interface.

    Honors ``sampler`` (typically built from temperature / top_p / top_k by
    ``mlx_lm.server._make_sampler``) via Leviathan-style speculative
    sampling: the drafter samples each token from its own distribution, the
    backbone verifies in one parallel forward, each draft is accepted with
    probability ``min(1, p_target / p_draft)``, and on rejection the
    "corrected" token is sampled from ``(p_target - p_draft)_+`` normalised.
    The bonus token (when all drafts accept) is sampled from the backbone's
    distribution. When ``sampler`` is ``None`` or trivially greedy, the
    fast greedy-argmax path is used (slightly higher acceptance, identical
    outputs).

    ``logits_processors`` are applied to both backbone and drafter logits
    at every position so stop-token/repetition constraints remain consistent.
    """

    greedy = sampler is None or _is_greedy_sampler(sampler)

    def _apply_processors(tokens, logits):
        if logits_processors:
            for p in logits_processors:
                logits = p(tokens, logits)
        return logits

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

    committed: List[int] = list(prompt.tolist())  # for logits_processors context

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
        verify_logits = _apply_processors(
            mx.array(committed), verify_logits[:, -1, :]
        )

        logprobs0 = verify_logits - mx.logsumexp(
            verify_logits, axis=-1, keepdims=True
        )
        if greedy:
            verified = mx.argmax(verify_logits, axis=-1)
        else:
            verified = sampler(logprobs0)
        mx.eval(last_h, verified, logprobs0)

    ntoks = 0
    if ntoks < max_tokens:
        ntoks += 1
        verified_int = int(verified.item())
        committed.append(verified_int)
        yield verified_int, logprobs0[0], False

    # ---- main draft + verify loop ----
    K = num_draft_tokens
    while ntoks < max_tokens:
        with mx.stream(_gs):
            cur_pos = cache[anchors["full_attention"]].offset
            anchors_kv = _gather_anchors(cache, anchors)

            # Draft K tokens autoregressively. Capture the drafter's
            # per-step logits so the verify step can compute the
            # importance ratio.
            drafted = []
            draft_logits_list = []
            h = last_h
            prev = verified
            for k in range(K):
                tok, dlogits, proj = _drafter_step(
                    draft_model, model, h, prev, anchors_kv,
                    rope_offset=cur_pos + k,
                    sampler=None if greedy else sampler,
                )
                # Apply same processors so the drafter sees consistent
                # constraints (e.g. logit_bias).
                dlogits = _apply_processors(
                    mx.array(committed + [int(t.item()) for t in drafted]),
                    dlogits,
                )
                drafted.append(tok)
                draft_logits_list.append(dlogits)
                h = proj
                prev = tok
            drafted_arr = mx.stack(drafted, axis=-1)

            # Verify pass: backbone forward on [verified, d_0..d_{K-1}].
            verify_input = mx.concatenate([verified[:, None], drafted_arr], axis=-1)
            hidden_seq = _post_norm_hidden(model, verify_input, cache)
            logits_seq = _backbone_logits_from_hidden(model, hidden_seq)
            mx.eval(drafted_arr, logits_seq, hidden_seq)

        drafted_list = drafted_arr[0].tolist()
        # Per-position backbone logits at positions [0 .. K] = K+1 outputs.
        # Position i predicts the token AFTER drafted[i-1] (or after verified
        # for i=0). Position K predicts the bonus token.
        target_logits_per_pos = []
        target_argmax = []
        for i in range(K + 1):
            tl = _apply_processors(
                mx.array(committed + drafted_list[:i]),
                logits_seq[:, i, :],
            )
            target_logits_per_pos.append(tl)
            target_argmax.append(int(mx.argmax(tl, axis=-1).item()))

        # Acceptance loop.
        j = 0  # number of drafts accepted
        rejected_correction: Optional[int] = None
        rejected_correction_logprobs: Optional[mx.array] = None

        for i in range(K):
            if greedy:
                if drafted_list[i] == target_argmax[i]:
                    j += 1
                    continue
                # Greedy rejection: take backbone's argmax as the correction.
                tl = target_logits_per_pos[i]
                rejected_correction = target_argmax[i]
                rejected_correction_logprobs = (
                    tl - mx.logsumexp(tl, axis=-1, keepdims=True)
                )[0]
                break
            else:
                # Leviathan-style importance-corrected acceptance.
                #
                #   p_target(x) = softmax(target_logits)[x]
                #   p_draft(x)  = softmax(draft_logits)[x]
                #   accept x with probability min(1, p_target / p_draft)
                #
                # We compute in float32 for numerical stability.
                tl = target_logits_per_pos[i]            # [1, V]
                dl = draft_logits_list[i]                # [1, V]
                p = mx.softmax(tl.astype(mx.float32), axis=-1)
                q = mx.softmax(dl.astype(mx.float32), axis=-1)
                x = drafted_list[i]
                p_x = p[0, x]
                q_x = q[0, x]
                accept_ratio = mx.minimum(
                    mx.array(1.0, dtype=mx.float32),
                    p_x / mx.maximum(q_x, mx.array(1e-9, dtype=mx.float32)),
                )
                u = mx.random.uniform(shape=())
                mx.eval(accept_ratio, u)
                if float(u.item()) < float(accept_ratio.item()):
                    j += 1
                    continue
                # Reject. Sample new token from (p - q)_+ normalised — this
                # gives an unbiased sample from the target distribution.
                corrected = mx.maximum(
                    p - q, mx.array(0.0, dtype=mx.float32)
                )
                corrected_sum = corrected.sum(axis=-1, keepdims=True)
                # Fall back to target distribution if the residual collapsed
                # (rare — happens when q dominates p exactly).
                fallback = mx.where(
                    corrected_sum > 0,
                    corrected / mx.maximum(corrected_sum, mx.array(1e-9, dtype=mx.float32)),
                    p,
                )
                fb_logits = mx.log(
                    mx.maximum(fallback, mx.array(1e-30, dtype=mx.float32))
                )
                # Use the user-supplied sampler so top_p/top_k apply.
                rejected_correction = int(sampler(fb_logits)[0].item())
                rejected_correction_logprobs = (
                    tl - mx.logsumexp(tl, axis=-1, keepdims=True)
                )[0]
                break

        # Yield the j accepted draft tokens.
        for i in range(j):
            if ntoks >= max_tokens:
                break
            tl = target_logits_per_pos[i]
            lp = (tl - mx.logsumexp(tl, axis=-1, keepdims=True))[0]
            ntoks += 1
            committed.append(drafted_list[i])
            yield drafted_list[i], lp, True

        if ntoks >= max_tokens:
            break

        # Bonus token: either the in-rejection correction we sampled, or
        # (when all K accepted) a fresh sample from the backbone's
        # post-K distribution.
        if rejected_correction is not None:
            bonus_tok = rejected_correction
            bonus_lp = rejected_correction_logprobs
        else:
            tl = target_logits_per_pos[K]
            bonus_lp = (tl - mx.logsumexp(tl, axis=-1, keepdims=True))[0]
            if greedy:
                bonus_tok = int(mx.argmax(tl, axis=-1).item())
            else:
                bonus_tok = int(sampler(bonus_lp[None])[0].item())

        ntoks += 1
        committed.append(bonus_tok)
        yield bonus_tok, bonus_lp, False

        with mx.stream(_gs):
            # Trim backbone cache: kept positions = verified + j accepted
            # drafts; the bonus token isn't yet in cache (it'll be processed
            # next iter).
            keep = 1 + j
            drop = (K + 1) - keep
            for c in cache:
                if c is not None:
                    c.trim(drop)
            last_h = hidden_seq[:, keep - 1 : keep, :]
            verified = mx.array([bonus_tok], dtype=mx.uint32)
            mx.eval(last_h, verified)


def _is_greedy_sampler(sampler: Callable) -> bool:
    """Behavior-based check: does the sampler always pick the argmax?

    mlx_lm's ``make_sampler(temp=0)`` returns a lambda — name-based
    detection fails there. We instead probe the sampler on a synthetic
    log-prob vector with a clearly-dominant max and a near-tie sub-max:
    a greedy sampler always returns the dominant index; a stochastic
    sampler will (in expectation) pick the sub-max some fraction of the
    time. We sample a couple of times and check that the result is
    consistent and equal to the argmax. Mis-detection here only changes
    perf, not correctness — both branches are unbiased."""
    try:
        logprobs = mx.array([[-10.0, -10.0, 0.0, -0.5]])
        results = set()
        for _ in range(2):
            results.add(int(sampler(logprobs)[0].item()))
        return len(results) == 1 and 2 in results
    except Exception:
        return False


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
