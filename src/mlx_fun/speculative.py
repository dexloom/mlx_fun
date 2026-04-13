"""DFlash speculative generation loop (Phase 4).

Implements speculative decoding using the DFlash block diffusion draft model.
The target model generates hidden states, which condition the DFlash draft model
to propose a block of candidate tokens in parallel.  The target model then
verifies candidates and accepts matching prefixes.

Algorithm:
    1. Prefill target model on prompt, capture hidden states
    2. Sample first token from target logits
    3. Feed hidden states + first token to DFlash draft_block -> candidate block
    4. Verify candidates against target model logits
    5. Accept matching prefix, reject from first mismatch
    6. Rewind cache on rejection, continue from accepted tokens
    7. Loop until EOS or max_tokens
"""

from typing import Callable, Dict, Generator, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .dflash_draft import BlockDiffusionDraftModel
from .hidden_state_capture import HiddenStateCapture


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_sampler(logits: mx.array) -> mx.array:
    """Greedy argmax sampler."""
    return mx.argmax(logits, axis=-1)


def _concatenate_captures(
    captures: Dict[int, List[mx.array]],
) -> Dict[int, mx.array]:
    """Concatenate chunked prefill captures per layer along the sequence dim."""
    result: Dict[int, mx.array] = {}
    for layer_idx, arrays in captures.items():
        if len(arrays) == 1:
            result[layer_idx] = arrays[0]
        else:
            result[layer_idx] = mx.concatenate(arrays, axis=1)
    return result


def _append_hidden(
    accumulated: Dict[int, mx.array],
    new_states: Dict[int, mx.array],
    n_keep: int,
) -> Dict[int, mx.array]:
    """Append first *n_keep* positions from *new_states* to *accumulated*.

    Used after verification to grow the hidden-state context that conditions
    the DFlash draft model.  Only the accepted token positions are kept.
    """
    result: Dict[int, mx.array] = {}
    for layer_idx in accumulated:
        kept = new_states[layer_idx][:, :n_keep, :]
        result[layer_idx] = mx.concatenate(
            [accumulated[layer_idx], kept], axis=1,
        )
    return result


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

def dflash_generate(
    prompt: mx.array,
    target_model: nn.Module,
    draft_model: BlockDiffusionDraftModel,
    capture: HiddenStateCapture,
    *,
    max_tokens: int = 256,
    sampler: Optional[Callable[[mx.array], mx.array]] = None,
    prompt_cache: Optional[List] = None,
    prefill_step_size: int = 2048,
    eos_token_id: Optional[int] = None,
) -> Generator[Tuple[mx.array, mx.array], None, None]:
    """Speculative generation using the DFlash block diffusion draft model.

    Yields ``(token, logprobs)`` tuples where *token* is a scalar
    ``mx.array`` and *logprobs* is a ``(vocab_size,)`` log-probability
    vector.

    Args:
        prompt: Token IDs, shape ``(seq_len,)`` or ``(1, seq_len)``.
        target_model: The target language model.
        draft_model: DFlash :class:`BlockDiffusionDraftModel` with target
            model attached.
        capture: :class:`HiddenStateCapture` installed on target-model
            layers matching ``draft_model.target_layer_ids``.
        max_tokens: Maximum number of tokens to generate.
        sampler: ``(1, vocab) -> (1,)`` callable.  Defaults to greedy
            argmax.
        prompt_cache: Pre-created KV cache list.  If ``None``, one is
            created via ``make_prompt_cache``.
        prefill_step_size: Chunk size for prompt processing.
        eos_token_id: Stop when this token is produced.

    Yields:
        ``(token, logprobs)`` pairs.
    """
    from mlx_lm.models.cache import (
        can_trim_prompt_cache,
        make_prompt_cache,
        trim_prompt_cache,
    )

    if sampler is None:
        sampler = _default_sampler

    if prompt_cache is None:
        prompt_cache = make_prompt_cache(target_model)

    if not can_trim_prompt_cache(prompt_cache):
        raise ValueError(
            "DFlash speculative decoding requires a trimmable prompt cache."
        )

    # Ensure 2-D input: (1, seq_len)
    if prompt.ndim == 1:
        prompt = mx.expand_dims(prompt, axis=0)

    # ------------------------------------------------------------------
    # Prefill
    # ------------------------------------------------------------------
    capture.clear()
    prompt_len = prompt.shape[1]

    for chunk_start in range(0, prompt_len, prefill_step_size):
        chunk_end = min(chunk_start + prefill_step_size, prompt_len)
        chunk = prompt[:, chunk_start:chunk_end]
        logits = target_model(chunk, cache=prompt_cache)
        mx.eval(logits)

    # Hidden states may span multiple prefill chunks
    all_hidden = _concatenate_captures(capture.collect())
    capture.clear()

    # Sample first token
    last_logits = logits[:, -1, :]  # (1, vocab)
    token = sampler(last_logits)
    if token.ndim == 0:
        token = mx.expand_dims(token, axis=0)
    logprobs = last_logits - mx.logsumexp(last_logits, axis=-1, keepdims=True)
    mx.eval(token, logprobs)

    yield token, logprobs.squeeze(0)

    ntokens = 1
    if eos_token_id is not None and token.item() == eos_token_id:
        return

    # ------------------------------------------------------------------
    # Speculative generation loop
    # ------------------------------------------------------------------
    while ntokens < max_tokens:
        # 1. Draft a block of candidate tokens
        draft_tokens = draft_model.draft_block(
            all_hidden, token, temperature=0.0,
        )
        mx.eval(draft_tokens)
        # (1, block_size) = [current_token, d1, ..., d_{B-1}]

        block_size = draft_tokens.shape[1]
        candidates = draft_tokens[0, 1:]  # (block_size - 1,)

        # 2. Verify all draft tokens through the target model
        capture.clear()
        verify_logits = target_model(draft_tokens, cache=prompt_cache)
        mx.eval(verify_logits)
        # (1, block_size, vocab)

        new_hidden = capture.collect_latest()

        # 3. Accept / reject candidates
        n_accepted = 0
        done = False

        for i in range(candidates.shape[0]):
            pos_logits = verify_logits[:, i, :]  # (1, vocab)
            target_token = sampler(pos_logits)
            if target_token.ndim == 0:
                target_token = mx.expand_dims(target_token, axis=0)
            pos_logprobs = pos_logits - mx.logsumexp(
                pos_logits, axis=-1, keepdims=True,
            )
            mx.eval(target_token)

            if target_token.item() == candidates[i].item():
                # Draft token accepted
                yield target_token, pos_logprobs.squeeze(0)
                n_accepted += 1
                ntokens += 1
                if ntokens >= max_tokens:
                    done = True
                    break
                if eos_token_id is not None and target_token.item() == eos_token_id:
                    done = True
                    break
            else:
                # Mismatch — emit target model's correction token
                yield target_token, pos_logprobs.squeeze(0)
                n_accepted += 1
                ntokens += 1
                token = target_token
                break
        else:
            # All candidates accepted — take bonus token from last position
            bonus_logits = verify_logits[:, -1, :]  # (1, vocab)
            bonus_token = sampler(bonus_logits)
            if bonus_token.ndim == 0:
                bonus_token = mx.expand_dims(bonus_token, axis=0)
            bonus_logprobs = bonus_logits - mx.logsumexp(
                bonus_logits, axis=-1, keepdims=True,
            )
            mx.eval(bonus_token)

            yield bonus_token, bonus_logprobs.squeeze(0)
            n_accepted += 1
            ntokens += 1
            token = bonus_token

            if eos_token_id is not None and bonus_token.item() == eos_token_id:
                done = True

        if done:
            return

        # 4. Rewind cache past rejected draft tokens
        n_to_trim = block_size - n_accepted
        if n_to_trim > 0:
            trim_prompt_cache(prompt_cache, n_to_trim)

        # 5. Grow hidden-state context for next draft round
        all_hidden = _append_hidden(all_hidden, new_hidden, n_accepted)
