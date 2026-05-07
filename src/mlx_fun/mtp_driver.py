"""Multi-Token Prediction (MTP) speculative-decoding driver for Gemma 4.

Pairs a Gemma 4 backbone (``model_type: gemma4``) with its small
``gemma4_assistant`` drafter and runs greedy speculative decoding:

  1. Backbone prefills the prompt; we capture its post-norm hidden state and
     a per-layer-type KV-cache "anchor" (last sliding + last full layer).
  2. Drafter runs ``K`` sequential single-token forwards. Each step takes the
     previous backbone-space hidden + previous-token embedding (in backbone
     space), pre-projects to the drafter hidden size, runs 4 transformer
     layers using the *backbone's* KV (the drafter holds none of its own),
     and emits a vocab logit + a post-projected backbone-space hidden that
     feeds back into the next draft step.
  3. Backbone verifies the drafted tokens in one parallel forward, and we
     accept the longest matching prefix (greedy argmax) plus one bonus
     correction token from the backbone. The KV cache is trimmed back to the
     accepted length and the loop continues.

Greedy only for v1. Sampling, batched prompts, and stop sequences are out of
scope here. The aim is correctness + a measurable speedup signal versus
running the backbone alone; a number of architectural choices (anchor layer
indices, embed scaling, RoPE offsets) are best-effort guesses informed by
the public ``transformers`` Gemma 4 source plus the assistant config — the
driver exposes them so they can be tuned.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx

from mlx_lm.utils import load


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _last_index_of(layer_types: List[str], wanted: str) -> int:
    for i in range(len(layer_types) - 1, -1, -1):
        if layer_types[i] == wanted:
            return i
    raise ValueError(f"No {wanted} layer found")


def _post_norm_hidden(backbone, inputs: mx.array, cache) -> mx.array:
    """Run the backbone's text-model trunk and return the post-norm hidden
    state (pre-lm_head). Mirrors gemma4_text.Model.__call__ minus the head."""
    inner = backbone.language_model.model
    return inner(inputs, cache=cache)


def _backbone_logits_from_hidden(backbone, hidden: mx.array) -> mx.array:
    inner = backbone.language_model.model
    if backbone.language_model.tie_word_embeddings:
        out = inner.embed_tokens.as_linear(hidden)
    else:
        out = backbone.language_model.lm_head(hidden)
    softcap = backbone.language_model.final_logit_softcapping
    if softcap is not None:
        out = mx.tanh(out / softcap) * softcap
    return out


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


@dataclass
class DriverConfig:
    num_draft_tokens: int = 4
    # Index of the backbone layer whose K/V the drafter borrows for each
    # layer_type. Default: the last layer of each type. ``None`` defers to
    # the auto-pick.
    anchor_layer_by_type: Optional[Dict[str, int]] = None
    # If True, scale the next-token embedding by the backbone's embed_scale
    # before passing it to the drafter's pre_projection. Empirically the
    # drafter expects the *scaled* embedding (matches the backbone's first
    # layer input), in concat order ``[scaled_emb | hidden]``.
    scale_next_token_embed: bool = True
    # Hard ceiling on generated tokens.
    max_new_tokens: int = 256
    # Optional EOS list — generation stops as soon as one is produced.
    eos_token_ids: Tuple[int, ...] = ()
    verbose: bool = False


@dataclass
class GenStats:
    new_tokens: int = 0
    backbone_calls: int = 0
    drafter_calls: int = 0
    accepted: int = 0  # drafted tokens that survived verify
    rejected: int = 0  # drafted tokens overwritten by backbone
    bonus: int = 0     # free correction/bonus tokens from verify
    elapsed_s: float = 0.0
    per_iter_accepts: List[int] = field(default_factory=list)

    @property
    def acceptance_rate(self) -> float:
        denom = self.accepted + self.rejected
        return self.accepted / denom if denom else 0.0


class MTPDriver:
    def __init__(
        self,
        backbone_path: str,
        drafter_path: str,
        config: Optional[DriverConfig] = None,
    ):
        self.config = config or DriverConfig()
        self.backbone, self.tokenizer = load(backbone_path)
        self.drafter, _drafter_tok = load(drafter_path)

        self._validate()
        self._inner = self.backbone.language_model.model  # Gemma4TextModel
        self._layer_types = [l.layer_type for l in self._inner.layers]

        if self.config.anchor_layer_by_type is None:
            self._anchors = {
                "sliding_attention": _last_index_of(
                    self._layer_types, "sliding_attention"
                ),
                "full_attention": _last_index_of(
                    self._layer_types, "full_attention"
                ),
            }
        else:
            self._anchors = dict(self.config.anchor_layer_by_type)

        if self.config.verbose:
            print(
                f"[MTPDriver] backbone hidden {self._inner.config.hidden_size}, "
                f"drafter hidden {self.drafter.text_args.hidden_size}, "
                f"anchors {self._anchors}"
            )

    # ---- compatibility ----

    def _validate(self):
        b = self.backbone
        d = self.drafter
        b_inner = b.language_model.model
        if d.backbone_hidden_size != b_inner.config.hidden_size:
            raise ValueError(
                f"Backbone hidden {b_inner.config.hidden_size} != "
                f"drafter.backbone_hidden_size {d.backbone_hidden_size}"
            )
        if d.text_args.vocab_size != b_inner.config.vocab_size:
            raise ValueError("Vocab size mismatch between backbone and drafter")

    # ---- KV anchor extraction ----

    def _gather_anchors(self, cache) -> Dict[str, Tuple[mx.array, mx.array]]:
        """Snapshot the K/V tensors at the anchor layers, sliced to the
        currently-valid offset (since KVCache stores a step-padded buffer)."""
        out = {}
        for layer_type, idx in self._anchors.items():
            c = cache[idx]
            n = c.offset
            k = c.keys[..., :n, :]
            v = c.values[..., :n, :]
            out[layer_type] = (k, v)
        return out

    # ---- prefill ----

    def _prefill(self, prompt_ids: mx.array):
        cache = self.backbone.make_cache()
        hidden = _post_norm_hidden(self.backbone, prompt_ids, cache)
        last_h = hidden[:, -1:, :]  # [B, 1, B_dim]
        logits = _backbone_logits_from_hidden(
            self.backbone, last_h
        )  # [B, 1, V]
        return cache, last_h, logits

    # ---- drafter ----

    def _next_emb(self, token_id: mx.array) -> mx.array:
        """Embedding of ``token_id`` in backbone hidden space, shape
        ``[B, 1, B_dim]``. Optionally embed-scaled."""
        if token_id.ndim == 1:
            token_id = token_id[:, None]
        emb = self._inner.embed_tokens(token_id)
        if self.config.scale_next_token_embed:
            emb = emb * self._inner.embed_scale
        return emb

    def _drafter_step(
        self,
        backbone_h: mx.array,
        prev_token: mx.array,
        anchors: Dict[str, Tuple[mx.array, mx.array]],
        rope_offset: int,
    ) -> Tuple[mx.array, mx.array]:
        """One drafter forward. Returns (next_token_id, projected_h_in_B_dim)."""
        emb = self._next_emb(prev_token)  # [B, 1, B_dim]
        # Concat order is [scaled_emb | backbone_hidden] — matches the
        # trained pre_projection.
        x = mx.concatenate([emb, backbone_h], axis=-1)  # [B, 1, 2*B_dim]
        h = self.drafter.pre_projection(x)  # [B, 1, H]

        # Walk the four drafter layers manually so we can pin the Q's RoPE
        # offset; the KV is supplied by the backbone anchor for the layer's
        # type.
        for layer in self.drafter.model.layers:
            shared_kv = anchors[layer.layer_type]
            h, _, _ = layer(h, mask=None, cache=None, shared_kv=shared_kv,
                            offset=rope_offset)
        h = self.drafter.model.norm(h)

        if self.drafter.tie_word_embeddings:
            logits = self.drafter.model.embed_tokens.as_linear(h)
        else:
            logits = self.drafter.lm_head(h)
        # Drafter has no final_logit_softcapping in the configs we've seen,
        # so skip it here.

        token = mx.argmax(logits[:, -1, :], axis=-1)  # [B]
        projected = self.drafter.post_projection(h)  # [B, 1, B_dim]
        return token, projected

    def _draft_k(
        self,
        starting_h: mx.array,
        verified_token: mx.array,
        anchors: Dict[str, Tuple[mx.array, mx.array]],
        position: int,
    ) -> mx.array:
        """Generate ``K = num_draft_tokens`` token ids autoregressively.

        ``starting_h`` is the backbone post-norm hidden at the position
        immediately before ``verified_token`` (i.e., the position whose
        argmax produced ``verified_token``). ``position`` is the absolute
        position of ``verified_token`` in the sequence (used for RoPE)."""
        K = self.config.num_draft_tokens
        h = starting_h
        prev = verified_token
        out = []
        for k in range(K):
            tok, proj = self._drafter_step(
                h, prev, anchors, rope_offset=position + k
            )
            out.append(tok)
            h = proj
            prev = tok
        result = mx.stack(out, axis=-1)  # [B, K]
        mx.eval(result)
        return result

    # ---- main loop ----

    def generate(self, prompt: str) -> Tuple[str, GenStats]:
        cfg = self.config
        K = cfg.num_draft_tokens
        eos = set(cfg.eos_token_ids)

        prompt_ids = mx.array(self.tokenizer.encode(prompt))[None]  # [1, L]
        L_prompt = prompt_ids.shape[1]

        stats = GenStats()
        t0 = time.time()

        # ---- prefill ----
        cache, last_h, logits = self._prefill(prompt_ids)
        mx.eval(last_h, logits)
        stats.backbone_calls += 1
        verified = mx.argmax(logits[:, -1, :], axis=-1)  # [B]
        out_tokens: List[int] = [int(verified.item())]
        stats.new_tokens += 1
        if int(verified.item()) in eos:
            stats.elapsed_s = time.time() - t0
            return self.tokenizer.decode(out_tokens), stats

        # Loop. Each iteration:
        #   - draft K tokens off (last_h, verified)
        #   - backbone verifies by forwarding [verified, d_0, ..., d_{K-1}]
        #   - accept matching prefix + bonus correction
        #   - last_h := hidden at last accepted position
        #   - verified := next token (= bonus or correction)
        while stats.new_tokens < cfg.max_new_tokens:
            anchors = self._gather_anchors(cache)
            cur_pos = cache[self._anchors["full_attention"]].offset
            # cur_pos == prefix length so far; ``verified`` will sit at this
            # position when the backbone forwards next.

            drafted = self._draft_k(last_h, verified, anchors, position=cur_pos)
            stats.drafter_calls += K

            # Verify: forward [verified, d_0, ..., d_{K-1}] (K+1 tokens).
            verify_input = mx.concatenate(
                [verified[:, None], drafted], axis=-1
            )  # [B, K+1]
            hidden_seq = _post_norm_hidden(self.backbone, verify_input, cache)
            stats.backbone_calls += 1
            logits_seq = _backbone_logits_from_hidden(
                self.backbone, hidden_seq
            )  # [B, K+1, V]
            backbone_args = mx.argmax(logits_seq, axis=-1)[0]  # [K+1]
            mx.eval(backbone_args)

            drafted_arr = drafted[0]  # [K]
            j = 0  # number of drafts accepted
            for i in range(K):
                if int(drafted_arr[i].item()) == int(backbone_args[i].item()):
                    j += 1
                else:
                    break

            # Tokens to commit this iteration: d_0..d_{j-1} + bonus.
            for i in range(j):
                tok = int(drafted_arr[i].item())
                out_tokens.append(tok)
                stats.new_tokens += 1
                stats.accepted += 1
                if tok in eos or stats.new_tokens >= cfg.max_new_tokens:
                    stats.elapsed_s = time.time() - t0
                    return self.tokenizer.decode(out_tokens), stats

            bonus_tok = int(backbone_args[j].item())
            out_tokens.append(bonus_tok)
            stats.new_tokens += 1
            stats.bonus += 1
            stats.rejected += K - j  # the K-j drafts we threw away
            stats.per_iter_accepts.append(j)

            # KV trim: backbone added K+1 to cache; we keep j+1 (verified +
            # j accepted drafts + the position whose logits gave the bonus
            # — wait, the bonus token itself is NOT in the cache yet).
            #
            # After verify, cache covers positions [..., L_prompt + new_tokens
            # - 1 + (K+1) - 1]. We keep up to and including position of the
            # last accepted draft, drop the rest.
            keep = 1 + j  # ``verified`` + j accepted drafts
            drop = (K + 1) - keep
            for c in cache:
                if c is not None:
                    c.trim(drop)

            # ``last_h`` for next iter is the hidden at the LAST kept position
            # (== position of the bonus's predecessor, which is the last
            # accepted draft if j > 0, else verified itself).
            last_h = hidden_seq[:, keep - 1 : keep, :]
            verified = mx.array([bonus_tok])
            mx.eval(last_h, verified)

            if bonus_tok in eos:
                break

            if cfg.verbose:
                print(
                    f"[iter] accepted {j}/{K}, total={stats.new_tokens}, "
                    f"out_tail={out_tokens[-min(8, len(out_tokens)):]}"
                )

        stats.elapsed_s = time.time() - t0
        return self.tokenizer.decode(out_tokens), stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _main():
    import argparse

    p = argparse.ArgumentParser(description="Gemma 4 MTP speculative decoding")
    p.add_argument("--backbone", required=True, help="Path or HF id of the gemma4 backbone")
    p.add_argument("--drafter", required=True, help="Path or HF id of the gemma4_assistant drafter")
    p.add_argument("--prompt", required=True, help="User message (will be wrapped in chat template)")
    p.add_argument("-k", "--num-draft-tokens", type=int, default=4)
    p.add_argument("-n", "--max-new-tokens", type=int, default=128)
    p.add_argument("--no-chat-template", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--baseline", action="store_true",
                   help="Also run the backbone alone for comparison")
    args = p.parse_args()

    cfg = DriverConfig(
        num_draft_tokens=args.num_draft_tokens,
        max_new_tokens=args.max_new_tokens,
        verbose=args.verbose,
    )
    driver = MTPDriver(args.backbone, args.drafter, cfg)

    if args.no_chat_template:
        prompt = args.prompt
    else:
        prompt = driver.tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    if args.baseline:
        from mlx_lm import generate as _gen
        t0 = time.time()
        baseline_text = _gen(driver.backbone, driver.tokenizer,
                             prompt=prompt, max_tokens=args.max_new_tokens,
                             verbose=False)
        dt = time.time() - t0
        print(f"\n=== BASELINE ({args.max_new_tokens}/{dt:.2f}s = {args.max_new_tokens/dt:.2f} tok/s) ===")
        print(baseline_text)

    text, stats = driver.generate(prompt)
    print(f"\n=== MTP (K={args.num_draft_tokens}) ===")
    print(text)
    print(
        f"\ntokens={stats.new_tokens} time={stats.elapsed_s:.2f}s "
        f"= {stats.new_tokens/stats.elapsed_s:.2f} tok/s, "
        f"accept={stats.acceptance_rate:.1%} ({stats.accepted}/{stats.accepted+stats.rejected}), "
        f"backbone_calls={stats.backbone_calls}"
    )


if __name__ == "__main__":
    _main()
