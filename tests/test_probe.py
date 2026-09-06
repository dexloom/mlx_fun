"""Tests for Q&A-driven expert relevance probing."""

import json

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from unittest.mock import MagicMock

from mlx_fun.probe import (
    DOMAIN,
    GENERAL,
    ProbeExample,
    ProbeQuestion,
    ProbeReport,
    ProbeStats,
    answer_nll,
    apply_coverage_filter,
    build_probe_tokens,
    compute_probe_scores,
    expert_mask,
    load_probe_set,
    masks_from_keep_map,
    paired_delta_stats,
    per_question_nll,
    question_vectors,
    run_knockout,
    run_prune_check,
    select_knockout_candidates,
    selection_bias_target,
    trace_question_set,
)


def _make_tokenizer(has_template=True):
    """Mock tokenizer: token count tracks word count, so lengths are predictable."""
    tok = MagicMock()
    tok.has_chat_template = has_template
    tok.encode = MagicMock(
        side_effect=lambda text, **kw: list(range(len(text.split())))
    )
    tok.apply_chat_template = MagicMock(
        side_effect=lambda msgs, tokenize=True, add_generation_prompt=True, **kw: (
            list(range(sum(len(m["content"].split()) for m in msgs) + len(msgs)))
        )
    )
    return tok


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return str(path)


# ---------------------------------------------------------------------------
# Question sets
# ---------------------------------------------------------------------------

class TestLoadProbeSet:
    def test_parses_fields(self, tmp_path):
        path = _write_jsonl(tmp_path / "q.jsonl", [
            {"question": "What is a modifier?", "answer": "A reusable check.",
             "tags": ["functions"]},
            {"question": "What is storage?", "answer": "Persistent contract state."},
        ])
        questions = load_probe_set(path)
        assert len(questions) == 2
        assert questions[0].question == "What is a modifier?"
        assert questions[0].answer == "A reusable check."
        assert questions[0].tags == ["functions"]
        assert questions[1].tags == []
        assert questions[1].system is None

    def test_skips_blank_lines(self, tmp_path):
        path = tmp_path / "q.jsonl"
        path.write_text('{"question": "a", "answer": "b"}\n\n   \n')
        assert len(load_probe_set(str(path))) == 1

    def test_reads_system_override(self, tmp_path):
        path = _write_jsonl(tmp_path / "q.jsonl", [
            {"question": "a", "answer": "b", "system": "Be terse."},
        ])
        assert load_probe_set(path)[0].system == "Be terse."

    def test_missing_question_raises(self, tmp_path):
        path = _write_jsonl(tmp_path / "q.jsonl", [{"answer": "b"}])
        with pytest.raises(ValueError, match="non-empty string"):
            load_probe_set(path)

    def test_empty_question_raises(self, tmp_path):
        path = _write_jsonl(tmp_path / "q.jsonl", [{"question": "   "}])
        with pytest.raises(ValueError, match="non-empty string"):
            load_probe_set(path)

    def test_bad_json_raises(self, tmp_path):
        path = tmp_path / "q.jsonl"
        path.write_text('{"question": "a"\n')
        with pytest.raises(ValueError, match="invalid JSON"):
            load_probe_set(str(path))

    def test_bad_tags_raises(self, tmp_path):
        path = _write_jsonl(tmp_path / "q.jsonl", [{"question": "a", "tags": "nope"}])
        with pytest.raises(ValueError, match="list of strings"):
            load_probe_set(path)

    def test_seeded_subsample(self, tmp_path):
        import random
        path = _write_jsonl(tmp_path / "q.jsonl", [
            {"question": f"q{i}", "answer": "a"} for i in range(20)
        ])
        random.seed(42)
        first = [q.question for q in load_probe_set(path, max_questions=5)]
        random.seed(42)
        second = [q.question for q in load_probe_set(path, max_questions=5)]
        assert len(first) == 5
        assert first == second

    def test_subsample_noop_when_smaller(self, tmp_path):
        path = _write_jsonl(tmp_path / "q.jsonl", [{"question": "a", "answer": "b"}])
        assert len(load_probe_set(path, max_questions=10)) == 1


class TestBuildProbeTokens:
    def test_teacher_appends_answer(self):
        tok = _make_tokenizer()
        q = ProbeQuestion(question="one two", answer="three four five")
        tokens, prompt_len = build_probe_tokens(tok, q, "teacher")
        # prompt: 2 words + 1 role = 3; answer: 3 words
        assert prompt_len == 3
        assert len(tokens) == 6
        tok.encode.assert_called_once_with("three four five", add_special_tokens=False)

    def test_generate_returns_prompt_only(self):
        tok = _make_tokenizer()
        q = ProbeQuestion(question="one two", answer="ignored here")
        tokens, prompt_len = build_probe_tokens(tok, q, "generate")
        assert len(tokens) == prompt_len == 3
        tok.encode.assert_not_called()

    def test_answer_truncated(self):
        tok = _make_tokenizer()
        q = ProbeQuestion(question="one", answer="a b c d e f")
        tokens, prompt_len = build_probe_tokens(tok, q, "teacher", max_answer_tokens=2)
        assert len(tokens) - prompt_len == 2

    def test_chat_template_args_forwarded(self):
        tok = _make_tokenizer()
        q = ProbeQuestion(question="one", answer="two")
        build_probe_tokens(tok, q, "teacher", chat_template_args={"enable_thinking": False})
        assert tok.apply_chat_template.call_args.kwargs["enable_thinking"] is False
        assert tok.apply_chat_template.call_args.kwargs["add_generation_prompt"] is True

    def test_question_system_overrides_default(self):
        tok = _make_tokenizer()
        q = ProbeQuestion(question="one", answer="two", system="Per question")
        build_probe_tokens(tok, q, "teacher", system="Default")
        messages = tok.apply_chat_template.call_args.args[0]
        assert messages[0] == {"role": "system", "content": "Per question"}

    def test_default_system_used(self):
        tok = _make_tokenizer()
        q = ProbeQuestion(question="one", answer="two")
        build_probe_tokens(tok, q, "teacher", system="Default")
        messages = tok.apply_chat_template.call_args.args[0]
        assert messages[0]["content"] == "Default"

    def test_no_system_message_when_absent(self):
        tok = _make_tokenizer()
        build_probe_tokens(tok, ProbeQuestion(question="one", answer="two"), "teacher")
        messages = tok.apply_chat_template.call_args.args[0]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    def test_falls_back_without_chat_template(self):
        tok = _make_tokenizer(has_template=False)
        q = ProbeQuestion(question="one two", answer="three")
        tokens, prompt_len = build_probe_tokens(tok, q, "teacher")
        tok.apply_chat_template.assert_not_called()
        assert prompt_len == 2  # "one two\n" -> 2 words
        assert len(tokens) - prompt_len == 1

    def test_teacher_without_answer_raises(self):
        tok = _make_tokenizer()
        with pytest.raises(ValueError, match="non-empty 'answer'"):
            build_probe_tokens(tok, ProbeQuestion(question="one"), "teacher")

    def test_unknown_mode_raises(self):
        tok = _make_tokenizer()
        q = ProbeQuestion(question="one", answer="two")
        with pytest.raises(ValueError, match="Unknown answer_mode"):
            build_probe_tokens(tok, q, "nonsense")


# ---------------------------------------------------------------------------
# Thinking mode
# ---------------------------------------------------------------------------

class _ThinkTokenizer:
    """Chat tokenizer whose generation prompt ends at an open think marker.

    Stands in for Qwen/GLM (whose wrapper defaults ``enable_thinking`` on) and
    for MiniMax (whose template ignores the flag entirely).
    """

    has_chat_template = True

    def __init__(self, tail="<|im_start|>assistant\n<think>\n",
                 think_start="<think>", think_end="</think>",
                 thinking_kwarg=None):
        self.think_start = think_start
        self.think_end = think_end
        self._tail = tail
        self.template_kwargs = None
        self.encoded = []
        if thinking_kwarg is not None:
            self._thinking_kwarg = thinking_kwarg

    def apply_chat_template(self, messages, tokenize=True,
                            add_generation_prompt=True, **kwargs):
        self.template_kwargs = kwargs
        return [10, 11, 12]

    def decode(self, ids):
        return self._tail

    def encode(self, text, add_special_tokens=True):
        self.encoded.append(text)
        return [100 + i for i in range(max(len(text.split()), 1))]


class TestThinkMarkers:
    def test_prefers_wrapper_attributes(self):
        from mlx_fun.probe import think_markers

        class Tok:
            think_start = "<|channel>thought"
            think_end = "<channel|>"

        assert think_markers(Tok()) == ("<|channel>thought", "<channel|>")

    def test_falls_back_to_vocab_pair(self):
        from mlx_fun.probe import think_markers

        class Tok:
            def get_vocab(self):
                return {"<think>": 1, "</think>": 2, "hello": 3}

        assert think_markers(Tok()) == ("<think>", "</think>")

    def test_longcat_pair(self):
        from mlx_fun.probe import think_markers

        class Tok:
            def get_vocab(self):
                return {"<longcat_think>": 1, "</longcat_think>": 2}

        assert think_markers(Tok()) == ("<longcat_think>", "</longcat_think>")

    def test_gemma_multi_token_channel(self):
        from mlx_fun.probe import think_markers

        class Tok:
            def get_vocab(self):
                return {"<|channel>": 1, "<channel|>": 2}

        assert think_markers(Tok()) == ("<|channel>thought", "<channel|>")

    def test_no_thinking_returns_none(self):
        from mlx_fun.probe import think_markers

        class Tok:
            def get_vocab(self):
                return {"hello": 1}

        assert think_markers(Tok()) is None
        assert think_markers(object()) is None

    def test_mock_tokenizer_is_not_mistaken_for_thinking(self):
        """A MagicMock answers every attribute; none of them are real markers."""
        from mlx_fun.probe import think_markers

        assert think_markers(_make_tokenizer()) is None


class TestThinkingDefaults:
    def test_disabled_by_default(self):
        tok = _make_tokenizer()
        build_probe_tokens(tok, ProbeQuestion(question="one", answer="two"), "teacher")
        assert tok.apply_chat_template.call_args.kwargs["enable_thinking"] is False

    def test_explicit_true_is_respected(self):
        tok = _make_tokenizer()
        build_probe_tokens(
            tok, ProbeQuestion(question="one", answer="two"), "teacher",
            chat_template_args={"enable_thinking": True},
        )
        assert tok.apply_chat_template.call_args.kwargs["enable_thinking"] is True

    def test_explicit_false_is_respected(self):
        tok = _make_tokenizer()
        build_probe_tokens(
            tok, ProbeQuestion(question="one", answer="two"), "teacher",
            chat_template_args={"enable_thinking": False},
        )
        assert tok.apply_chat_template.call_args.kwargs["enable_thinking"] is False

    def test_model_specific_kwarg_is_respected(self):
        """A tokenizer that spells the flag 'thinking' must not also get
        enable_thinking=False injected behind the caller's back."""
        tok = _ThinkTokenizer(thinking_kwarg="thinking")
        tokens, prompt_len = build_probe_tokens(
            tok, ProbeQuestion(question="one", answer="two"), "generate",
            chat_template_args={"thinking": True},
        )
        assert tok.template_kwargs == {"thinking": True}
        assert "enable_thinking" not in tok.template_kwargs
        # Thinking was explicitly enabled, so the open marker stays open.
        assert tokens == [10, 11, 12] and prompt_len == 3


class TestForceClosedThinking:
    def test_dangling_marker_is_closed(self):
        tok = _ThinkTokenizer()
        tokens, prompt_len = build_probe_tokens(
            tok, ProbeQuestion(question="one", answer="two"), "generate",
        )
        # The rendered prompt plus the encoded "\n</think>\n\n".
        assert tokens == [10, 11, 12, 100]
        assert prompt_len == len(tokens)
        assert "</think>" in tok.encoded[0]

    def test_prompt_len_covers_the_appended_tokens(self):
        tok = _ThinkTokenizer()
        tokens, prompt_len = build_probe_tokens(
            tok, ProbeQuestion(question="one", answer="a b c"), "teacher",
        )
        assert prompt_len == 4                  # 3 rendered + 1 closing
        assert len(tokens) - prompt_len == 3    # the reference answer

    def test_not_closed_when_caller_enabled_thinking(self):
        tok = _ThinkTokenizer()
        tokens, prompt_len = build_probe_tokens(
            tok, ProbeQuestion(question="one", answer="two"), "generate",
            chat_template_args={"enable_thinking": True},
        )
        assert tokens == [10, 11, 12] and prompt_len == 3
        assert tok.encoded == []

    def test_not_closed_when_prompt_does_not_dangle(self):
        tok = _ThinkTokenizer(tail="<|im_start|>assistant\n")
        tokens, prompt_len = build_probe_tokens(
            tok, ProbeQuestion(question="one", answer="two"), "generate",
        )
        assert tokens == [10, 11, 12] and prompt_len == 3

    def test_not_closed_without_think_markers(self):
        """A tokenizer with no thinking channel is never touched."""
        tok = _ThinkTokenizer(think_start=None, think_end=None)
        tokens, prompt_len = build_probe_tokens(
            tok, ProbeQuestion(question="one", answer="two"), "generate",
        )
        assert tokens == [10, 11, 12] and prompt_len == 3

    def test_logs_once_per_tokenizer(self, caplog):
        import logging as _logging

        from mlx_fun.probe import _FORCE_CLOSED_THINK

        tok = _ThinkTokenizer()
        _FORCE_CLOSED_THINK.discard(id(tok))
        with caplog.at_level(_logging.INFO, logger="root"):
            for _ in range(3):
                build_probe_tokens(
                    tok, ProbeQuestion(question="one", answer="two"), "generate",
                )
        notices = [r for r in caplog.records if "force an empty think block" in r.message]
        assert len(notices) == 1


# ---------------------------------------------------------------------------
# Capture slicing
# ---------------------------------------------------------------------------

def _capture(seq_len, k=2, fill=0):
    """One (inds, scores, norms) capture whose inds encode the position index."""
    inds = np.arange(fill, fill + seq_len).reshape(1, seq_len, 1) % 4
    inds = np.repeat(inds, k, axis=2)
    scores = np.full((1, seq_len, k), 0.5)
    norms = np.full((1, seq_len, k), 2.0)
    return inds, scores, norms


class TestSliceAnswerCaptures:
    def test_single_capture(self):
        from mlx_fun.probe import slice_answer_captures
        inds, scores, norms = slice_answer_captures([_capture(10)], prompt_len=6, n_answer=4)
        assert inds.shape == scores.shape == norms.shape == (4, 2)

    def test_producing_positions(self):
        """Rows must be [prompt_len-1 : prompt_len-1+n_answer]."""
        from mlx_fun.probe import slice_answer_captures
        full = _capture(10)
        inds, _, _ = slice_answer_captures([full], prompt_len=6, n_answer=4)
        expected = full[0].reshape(-1, 2)[5:9]
        np.testing.assert_array_equal(inds, expected)

    def test_chunked_generation_layout(self):
        """Prefill chunk + last prompt token + one capture per generated token."""
        from mlx_fun.probe import slice_answer_captures
        prompt_len, n_gen = 7, 3
        # prefill of P-1, the last prompt token, then one forward per generated
        # token: P + n_gen positions in total.
        captures = [_capture(prompt_len - 1)] + [_capture(1) for _ in range(1 + n_gen)]
        inds, _, _ = slice_answer_captures(captures, prompt_len, n_answer=n_gen)
        assert inds.shape == (n_gen, 2)

    def test_multi_chunk_prefill(self):
        from mlx_fun.probe import slice_answer_captures
        captures = [_capture(4), _capture(4), _capture(1), _capture(1)]
        inds, _, _ = slice_answer_captures(captures, prompt_len=9, n_answer=1)
        assert inds.shape == (1, 2)

    def test_mismatch_raises(self):
        from mlx_fun.probe import slice_answer_captures
        with pytest.raises(ValueError, match="expected"):
            slice_answer_captures([_capture(10)], prompt_len=6, n_answer=9)

    def test_empty_raises(self):
        from mlx_fun.probe import slice_answer_captures
        with pytest.raises(ValueError, match="no captures"):
            slice_answer_captures([], prompt_len=1, n_answer=1)


# ---------------------------------------------------------------------------
# Answer NLL
# ---------------------------------------------------------------------------

class TestAnswerNll:
    def test_matches_hand_computation(self):
        # T=3, V=4. prompt_len=2 -> one answer token, predicted by position 1.
        logits = mx.zeros((1, 3, 4))
        tokens = [0, 1, 2]
        value = answer_nll(logits, tokens, prompt_len=2)
        assert value == pytest.approx(np.log(4.0), abs=1e-5)

    def test_confident_prediction_is_low(self):
        logits = np.zeros((1, 3, 4), dtype=np.float32)
        logits[0, 1, 2] = 20.0  # position 1 predicts token 2 with confidence
        value = answer_nll(mx.array(logits), [0, 1, 2], prompt_len=2)
        assert value < 1e-4

    def test_prompt_rows_do_not_matter(self):
        base = np.zeros((1, 4, 4), dtype=np.float32)
        base[0, 2, 3] = 5.0
        tokens = [0, 1, 2, 3]
        a = answer_nll(mx.array(base), tokens, prompt_len=3)
        base[0, 0, :] = 99.0  # a prompt-producing row outside the scored slice
        b = answer_nll(mx.array(base), tokens, prompt_len=3)
        assert a == pytest.approx(b)

    def test_no_answer_tokens_raises(self):
        with pytest.raises(ValueError, match="no answer tokens"):
            answer_nll(mx.zeros((1, 3, 4)), [0, 1, 2], prompt_len=3)


# ---------------------------------------------------------------------------
# Question-weighted statistics
# ---------------------------------------------------------------------------

def _sliced(n_positions, experts, num_layers=1, weight=0.5, norm=2.0):
    """Per-layer sliced captures where every position routes to `experts`."""
    k = len(experts)
    inds = np.tile(np.array(experts, dtype=np.intp), (n_positions, 1))
    scores = np.full((n_positions, k), weight)
    norms = np.full((n_positions, k), norm)
    return [(inds, scores, norms) for _ in range(num_layers)]


class TestQuestionVectors:
    def test_rows_sum_to_top_k(self):
        freq, weight = question_vectors(_sliced(5, [0, 2]), num_experts=4)
        assert freq.shape == (1, 4)
        assert freq[0].sum() == pytest.approx(2.0)
        assert weight[0].sum() == pytest.approx(1.0)

    def test_frequency_is_a_fraction(self):
        inds = np.array([[0, 1], [0, 2]], dtype=np.intp)
        scores = np.full((2, 2), 0.5)
        norms = np.ones((2, 2))
        freq, _ = question_vectors([(inds, scores, norms)], num_experts=4)
        assert freq[0, 0] == pytest.approx(1.0)   # both positions
        assert freq[0, 1] == pytest.approx(0.5)   # one position
        assert freq[0, 3] == 0.0


class TestProbeStats:
    def test_long_and_short_questions_weigh_equally(self):
        stats = ProbeStats(num_layers=1, num_experts=4)
        stats.add_question(DOMAIN, _sliced(100, [0, 1]))
        stats.add_question(DOMAIN, _sliced(2, [2, 3]))
        mean = stats.mean_freq(DOMAIN)
        # Each question contributes 1.0 to its two experts, averaged over 2 questions.
        assert mean[0, 0] == pytest.approx(0.5)
        assert mean[0, 3] == pytest.approx(0.5)

    def test_coverage_counts_questions_not_tokens(self):
        stats = ProbeStats(num_layers=1, num_experts=4)
        stats.add_question(DOMAIN, _sliced(50, [0, 1]))
        stats.add_question(DOMAIN, _sliced(1, [0, 2]))
        coverage = stats.coverage_fraction(DOMAIN)
        assert coverage[0, 0] == pytest.approx(1.0)
        assert coverage[0, 1] == pytest.approx(0.5)
        assert coverage[0, 3] == 0.0

    def test_saliency_only_from_domain(self):
        stats = ProbeStats(num_layers=1, num_experts=4)
        stats.add_question(GENERAL, _sliced(4, [0, 1]))
        assert stats.saliency.freq.sum() == 0.0
        stats.add_question(DOMAIN, _sliced(4, [0, 1]))
        assert stats.saliency.freq.sum() > 0.0

    def test_question_weighting_normalizes_saliency(self):
        long_stats = ProbeStats(1, 4)
        long_stats.add_question(DOMAIN, _sliced(100, [0, 1]))
        short_stats = ProbeStats(1, 4)
        short_stats.add_question(DOMAIN, _sliced(2, [0, 1]))
        np.testing.assert_allclose(
            long_stats.saliency.freq, short_stats.saliency.freq,
        )

    def test_token_weighting_keeps_raw_counts(self):
        stats = ProbeStats(1, 4)
        stats.add_question(DOMAIN, _sliced(10, [0, 1]), saliency_weighting="token")
        assert stats.saliency.freq[0, 0] == pytest.approx(10.0)

    def test_reap_unchanged_by_question_weighting(self):
        weighted = ProbeStats(1, 4)
        weighted.add_question(DOMAIN, _sliced(8, [0, 1], weight=0.5, norm=2.0))
        raw = ProbeStats(1, 4)
        raw.add_question(DOMAIN, _sliced(8, [0, 1], weight=0.5, norm=2.0),
                         saliency_weighting="token")
        np.testing.assert_allclose(
            weighted.saliency.compute_scores("reap"),
            raw.saliency.compute_scores("reap"),
        )

    def test_unknown_label_raises(self):
        with pytest.raises(ValueError, match="Unknown label"):
            ProbeStats(1, 4).add_question("other", _sliced(2, [0, 1]))

    def test_unknown_weighting_raises(self):
        with pytest.raises(ValueError, match="Unknown saliency_weighting"):
            ProbeStats(1, 4).add_question(DOMAIN, _sliced(2, [0, 1]), "nonsense")

    def test_empty_label_means_zero(self):
        stats = ProbeStats(2, 4)
        np.testing.assert_array_equal(stats.mean_freq(DOMAIN), np.zeros((2, 4)))


class TestComputeProbeScores:
    def test_domain_only_expert_scores_highest(self):
        stats = ProbeStats(num_layers=1, num_experts=4)
        stats.add_question(DOMAIN, _sliced(4, [0, 1]))
        stats.add_question(GENERAL, _sliced(4, [2, 3]))
        diff_freq, diff_weight, composite = compute_probe_scores(stats)
        assert diff_freq[0, 0] > 0 and diff_freq[0, 2] < 0
        assert diff_weight[0, 0] > 0
        assert composite[0, 0] == pytest.approx(1.0)
        assert composite[0, 2] == pytest.approx(0.0)


class TestCoverageFilter:
    def test_drops_low_coverage_experts(self):
        coverage = np.array([[1.0, 0.1, 0.8, 0.0]])
        filtered = apply_coverage_filter({0: [0, 1, 2]}, coverage, min_coverage=0.5)
        assert filtered == {0: [0, 2]}

    def test_drops_emptied_layers(self):
        coverage = np.array([[0.1, 0.1, 0.1, 0.1]])
        assert apply_coverage_filter({0: [0, 1]}, coverage, 0.5) == {}

    def test_zero_threshold_is_a_passthrough(self):
        coverage = np.zeros((1, 4))
        assert apply_coverage_filter({0: [0, 1]}, coverage, 0.0) == {0: [0, 1]}


class TestSelectKnockoutCandidates:
    def test_orders_by_composite_descending(self):
        composite = np.array([[0.1, 0.9], [0.5, 0.7]])
        pairs = select_knockout_candidates(composite, {0: [0, 1], 1: [0, 1]}, n=3)
        assert pairs == [(0, 1), (1, 1), (1, 0)]

    def test_clips_to_available(self):
        composite = np.array([[0.1, 0.9]])
        assert len(select_knockout_candidates(composite, {0: [0, 1]}, n=99)) == 2

    def test_zero_returns_empty(self):
        assert select_knockout_candidates(np.zeros((1, 2)), {0: [0]}, n=0) == []


# ---------------------------------------------------------------------------
# Paired statistics
# ---------------------------------------------------------------------------

class TestPairedDeltaStats:
    def test_mean_and_median(self):
        stats = paired_delta_stats(
            np.array([1.0, 1.0, 1.0]), np.array([1.1, 1.2, 1.6]), n_boot=0,
        )
        assert stats["mean_delta"] == pytest.approx(0.3, abs=1e-9)
        assert stats["median_delta"] == pytest.approx(0.2, abs=1e-9)
        assert stats["n_valid"] == 3
        assert stats["valid_fraction"] == 1.0

    def test_bootstrap_ci_brackets_the_mean(self):
        rng = np.random.default_rng(0)
        baseline = np.zeros(50)
        masked = rng.normal(0.5, 0.05, size=50)
        stats = paired_delta_stats(baseline, masked, n_boot=500, seed=7)
        assert stats["ci_low"] < stats["mean_delta"] < stats["ci_high"]

    def test_bootstrap_is_reproducible(self):
        baseline, masked = np.zeros(20), np.linspace(0.1, 0.5, 20)
        a = paired_delta_stats(baseline, masked, n_boot=200, seed=3)
        b = paired_delta_stats(baseline, masked, n_boot=200, seed=3)
        assert a["ci_low"] == b["ci_low"] and a["ci_high"] == b["ci_high"]

    def test_verified_requires_effect_and_positive_ci(self):
        stats = paired_delta_stats(np.zeros(20), np.full(20, 0.5), n_boot=200, seed=1)
        assert stats["status"] == "verified"

    def test_tiny_delta_is_not_verified(self):
        stats = paired_delta_stats(np.zeros(20), np.full(20, 1e-6), n_boot=200, seed=1)
        assert stats["status"] == "not_verified"

    def test_mostly_nonfinite_is_catastrophic(self):
        masked = np.array([1.0, np.nan, np.nan, np.nan])
        stats = paired_delta_stats(np.zeros(4), masked, n_boot=0)
        assert stats["status"] == "catastrophic"
        assert stats["n_nonfinite"] == 3

    def test_partial_nonfinite_is_inconclusive(self):
        masked = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, np.nan, np.nan])
        stats = paired_delta_stats(np.zeros(10), masked, n_boot=0, min_valid_fraction=0.9)
        assert stats["status"] == "inconclusive"
        assert stats["n_nonfinite"] == 2

    def test_nonfinite_is_never_averaged_away(self):
        """A collapse on half the questions must not read as a credible delta."""
        masked = np.array([0.01, np.nan, 0.01, np.nan])
        stats = paired_delta_stats(np.zeros(4), masked, n_boot=0)
        assert stats["status"] == "inconclusive"
        assert stats["n_nonfinite"] == 2
        assert stats["valid_fraction"] == 0.5

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="same shape"):
            paired_delta_stats(np.zeros(3), np.zeros(4))


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _report(**kwargs):
    base = dict(
        domain_name="solidity",
        num_layers=2,
        num_experts=4,
        threshold_percentile=90.0,
        differential_freq=np.zeros((2, 4)),
        differential_activation=np.zeros((2, 4)),
        composite_score=np.array([[0.1, 0.9, 0.2, 0.3], [0.4, 0.5, 0.6, 0.7]]),
        domain_experts={0: [1], 1: [3]},
        general_experts={0: [0]},
    )
    base.update(kwargs)
    return ProbeReport(**base)


class TestProbeReport:
    def test_roundtrip(self, tmp_path):
        report = _report(
            answer_mode="generate",
            num_domain_questions=12,
            skipped_questions={"domain": 1, "general": 0},
            domain_coverage=np.full((2, 4), 0.5),
            knockout={"backend": "gate_selection_mask", "per_expert": []},
            knockout_delta=np.zeros((2, 4)),
            verified_domain_experts={0: [1]},
            prune_check={"masked_pairs": 3},
        )
        path = str(tmp_path / "probe.json")
        report.save(path)
        loaded = ProbeReport.load(path)
        assert loaded.answer_mode == "generate"
        assert loaded.num_domain_questions == 12
        assert loaded.skipped_questions == {"domain": 1, "general": 0}
        assert loaded.verified_domain_experts == {0: [1]}
        assert loaded.prune_check == {"masked_pairs": 3}
        np.testing.assert_array_equal(loaded.domain_coverage, np.full((2, 4), 0.5))
        np.testing.assert_array_equal(loaded.composite_score, report.composite_score)

    def test_roundtrip_with_no_verification(self, tmp_path):
        path = str(tmp_path / "probe.json")
        _report().save(path)
        loaded = ProbeReport.load(path)
        assert loaded.knockout is None
        assert loaded.knockout_delta is None
        assert loaded.prune_check is None

    def test_domain_report_reads_probe_report(self, tmp_path):
        """The whole point: downstream tools take this file unchanged."""
        from mlx_fun.domain import DomainReport
        path = str(tmp_path / "probe.json")
        _report(knockout={"x": 1}).save(path)
        loaded = DomainReport.load(path)
        assert loaded.domain_name == "solidity"
        assert loaded.domain_experts == {0: [1], 1: [3]}

    def test_steering_config_reads_probe_report(self, tmp_path):
        from mlx_fun.steering import SteeringConfig
        path = str(tmp_path / "probe.json")
        _report().save(path)
        assert SteeringConfig.from_domain_report(path, "boost").activate == {0: [1], 1: [3]}
        assert SteeringConfig.from_domain_report(path, "suppress").deactivate == {0: [0]}

    def test_pruner_reads_probe_report(self, tmp_path):
        from mlx_fun.pruner import load_domain_constraints
        path = str(tmp_path / "probe.json")
        _report().save(path)
        protected, targeted = load_domain_constraints(path, "protect")
        assert targeted is None
        np.testing.assert_array_equal(protected[0], np.array([1]))

    def test_amplification_biases_read_probe_report(self, tmp_path):
        from mlx_fun.domain import compute_amplification_biases
        path = str(tmp_path / "probe.json")
        _report().save(path)
        biases = compute_amplification_biases(ProbeReport.load(path), scale=2.0)
        assert biases[0][1] == pytest.approx(1.8)


# ---------------------------------------------------------------------------
# Knockout backend
# ---------------------------------------------------------------------------

class TestSelectionBiasTarget:
    def test_minimax_targets_the_block(self, tiny_minimax_moe):
        module, attr = selection_bias_target(tiny_minimax_moe, "minimax")
        assert module is tiny_minimax_moe
        assert attr == "e_score_correction_bias"

    def test_glm4_targets_the_gate(self, tiny_glm4_moe):
        module, attr = selection_bias_target(tiny_glm4_moe, "glm4_moe")
        assert module is tiny_glm4_moe.gate
        assert attr == "e_score_correction_bias"

    def test_qwen3_targets_the_gate_bias(self, tiny_qwen3_moe):
        module, attr = selection_bias_target(tiny_qwen3_moe, "qwen3_moe")
        assert module is tiny_qwen3_moe.gate
        assert attr == "bias"

    def test_gemma4_is_unsupported(self, tiny_minimax_moe):
        with pytest.raises(ValueError, match="Knockout not supported"):
            selection_bias_target(tiny_minimax_moe, "gemma4")

    def test_unknown_type_is_unsupported(self, tiny_minimax_moe):
        with pytest.raises(ValueError, match="Knockout not supported"):
            selection_bias_target(tiny_minimax_moe, "nonsense")


def _routed_experts(block, model_type, x):
    """Every expert the block routes to, via the observer hooks."""
    from mlx_fun.observer import install_hooks, collect_captures, remove_hooks

    install_hooks([block], model_type)
    try:
        block(x)
        mx.eval(block.parameters())
        captures = collect_captures([block])
    finally:
        remove_hooks([block])
    return set(np.unique(np.concatenate([c[0].ravel() for c in captures[0]])).tolist())


class TestExpertMask:
    """A masked expert must be unselectable in the real router, not merely biased."""

    def test_minimax_mask_beats_a_positive_correction(self, tiny_minimax_moe, sample_input):
        block = tiny_minimax_moe
        # A large positive correction would keep expert 1 selected if the mask
        # were applied before the sigmoid instead of to the selection score.
        block.e_score_correction_bias = mx.array([0.0, 5.0, 0.0, 0.0])
        assert 1 in _routed_experts(block, "minimax", sample_input)

        with expert_mask([block], "minimax", {0: [1]}, num_experts=4):
            routed = _routed_experts(block, "minimax", sample_input)
        assert 1 not in routed

    def test_glm4_mask_beats_a_positive_correction(self, tiny_glm4_moe, sample_input):
        block = tiny_glm4_moe
        block.gate.e_score_correction_bias = mx.array([0.0, 0.0, 5.0, 0.0])
        assert 2 in _routed_experts(block, "glm4_moe", sample_input)

        with expert_mask([block], "glm4_moe", {0: [2]}, num_experts=4):
            routed = _routed_experts(block, "glm4_moe", sample_input)
        assert 2 not in routed

    def test_qwen3_mask_deselects(self, tiny_qwen3_moe, sample_input):
        block = tiny_qwen3_moe
        routed_before = _routed_experts(block, "qwen3_moe", sample_input)
        victim = sorted(routed_before)[0]
        with expert_mask([block], "qwen3_moe", {0: [victim]}, num_experts=4):
            routed = _routed_experts(block, "qwen3_moe", sample_input)
        assert victim not in routed

    def test_restore_is_exact(self, tiny_minimax_moe):
        block = tiny_minimax_moe
        block.e_score_correction_bias = mx.array([0.25, -0.5, 1.0, 0.0])
        before = np.array(block.e_score_correction_bias)
        with expert_mask([block], "minimax", {0: [2]}, num_experts=4):
            pass
        np.testing.assert_array_equal(np.array(block.e_score_correction_bias), before)

    def test_temporary_bias_is_removed(self, tiny_qwen3_moe):
        block = tiny_qwen3_moe
        assert "bias" not in block.gate
        with expert_mask([block], "qwen3_moe", {0: [1]}, num_experts=4):
            assert "bias" in block.gate
        assert "bias" not in block.gate

    def test_restore_runs_on_exception(self, tiny_minimax_moe):
        block = tiny_minimax_moe
        before = np.array(block.e_score_correction_bias)
        with pytest.raises(RuntimeError):
            with expert_mask([block], "minimax", {0: [1]}, num_experts=4):
                raise RuntimeError("boom")
        np.testing.assert_array_equal(np.array(block.e_score_correction_bias), before)

    def test_empty_mask_leaves_routing_unchanged(self, tiny_minimax_moe, sample_input):
        block = tiny_minimax_moe
        before = _routed_experts(block, "minimax", sample_input)
        with expert_mask([block], "minimax", {}, num_experts=4):
            after = _routed_experts(block, "minimax", sample_input)
        assert before == after

    def test_out_of_range_expert_raises_and_restores(self, tiny_minimax_moe):
        block = tiny_minimax_moe
        before = np.array(block.e_score_correction_bias)
        with pytest.raises(ValueError, match="out of range"):
            with expert_mask([block], "minimax", {0: [9]}, num_experts=4):
                pass
        np.testing.assert_array_equal(np.array(block.e_score_correction_bias), before)


class TestMasksFromKeepMap:
    def test_inverts_a_keep_map(self):
        keep_map = {0: np.array([0, 2]), 1: np.array([0, 1, 2, 3])}
        assert masks_from_keep_map(keep_map, num_experts=4) == {0: [1, 3]}

    def test_matches_build_keep_map(self):
        from mlx_fun.pruner import build_keep_map
        scores = np.array([[0.9, 0.1, 0.8, 0.2]])
        keep_map = build_keep_map(scores, n_prune=2)
        assert masks_from_keep_map(keep_map, 4) == {0: [1, 3]}


# ---------------------------------------------------------------------------
# Knockout and prune check on a tiny end-to-end model
# ---------------------------------------------------------------------------

class TinyMoEModel(nn.Module):
    """embedding -> MoE block -> lm_head, enough to score a token sequence."""

    def __init__(self, moe_block, vocab=64, hidden=32):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.moe = moe_block
        self.lm_head = nn.Linear(hidden, vocab)

    def __call__(self, tokens):
        return self.lm_head(self.moe(self.embed(tokens)))


@pytest.fixture
def tiny_model(tiny_minimax_moe):
    mx.random.seed(0)
    return TinyMoEModel(tiny_minimax_moe)


@pytest.fixture
def tiny_examples():
    return [
        ProbeExample(tokens=[1, 2, 3, 4, 5, 6], prompt_len=4, question_index=0),
        ProbeExample(tokens=[7, 8, 9, 10, 11, 12], prompt_len=3, question_index=1),
        ProbeExample(tokens=[2, 4, 6, 8, 10], prompt_len=3, question_index=2),
    ]


class TestPerQuestionNll:
    def test_deterministic(self, tiny_model, tiny_examples):
        a = per_question_nll(tiny_model, tiny_examples)
        b = per_question_nll(tiny_model, tiny_examples)
        np.testing.assert_allclose(a, b)
        assert np.all(np.isfinite(a))


class TestRunKnockout:
    def test_baseline_is_stable(self, tiny_model, tiny_examples):
        result = run_knockout(
            tiny_model, tiny_examples, [tiny_model.moe], "minimax",
            num_experts=4, top_k=2, candidates=[], n_boot=0,
        )
        assert result.baseline_nll == pytest.approx(result.plain_baseline_nll, abs=1e-5)
        assert result.num_questions == 3

    def test_masking_a_routed_expert_changes_nll(self, tiny_model, tiny_examples, sample_input):
        routed = _routed_experts(tiny_model.moe, "minimax", sample_input)
        victim = sorted(routed)[0]
        result = run_knockout(
            tiny_model, tiny_examples, [tiny_model.moe], "minimax",
            num_experts=4, top_k=2, candidates=[(0, victim)], n_boot=50,
        )
        entry = result.per_expert[0]
        assert entry["layer"] == 0 and entry["expert"] == victim
        assert abs(entry["mean_delta"]) > 0
        assert entry["n_valid"] == 3

    def test_never_routed_expert_gives_exactly_zero(self, tiny_minimax_moe, tiny_examples):
        # Force routing away from expert 3 entirely.
        tiny_minimax_moe.e_score_correction_bias = mx.array([5.0, 5.0, 5.0, -100.0])
        model = TinyMoEModel(tiny_minimax_moe)
        result = run_knockout(
            model, tiny_examples, [model.moe], "minimax",
            num_experts=4, top_k=2, candidates=[(0, 3)], n_boot=0,
        )
        assert result.per_expert[0]["mean_delta"] == 0.0

    def test_records_composite_and_coverage(self, tiny_model, tiny_examples):
        composite = np.array([[0.1, 0.2, 0.3, 0.4]])
        coverage = np.array([[1.0, 0.5, 0.25, 0.0]])
        result = run_knockout(
            tiny_model, tiny_examples, [tiny_model.moe], "minimax",
            num_experts=4, top_k=2, candidates=[(0, 1)],
            composite=composite, coverage=coverage, n_boot=0,
        )
        assert result.per_expert[0]["composite"] == pytest.approx(0.2)
        assert result.per_expert[0]["domain_coverage"] == pytest.approx(0.5)

    def test_mask_budget_guard(self, tiny_model, tiny_examples):
        with pytest.raises(ValueError, match="top_k"):
            run_knockout(
                tiny_model, tiny_examples, [tiny_model.moe], "minimax",
                num_experts=4, top_k=4, candidates=[(0, 1)], n_boot=0,
            )

    def test_model_is_unchanged_afterwards(self, tiny_model, tiny_examples):
        before = per_question_nll(tiny_model, tiny_examples)
        run_knockout(
            tiny_model, tiny_examples, [tiny_model.moe], "minimax",
            num_experts=4, top_k=2, candidates=[(0, 0), (0, 1)], n_boot=0,
        )
        np.testing.assert_allclose(before, per_question_nll(tiny_model, tiny_examples))


class TestRunPruneCheck:
    def test_masks_the_keep_map_complement(self, tiny_model, tiny_examples):
        keep_map = {0: np.array([0, 1])}
        out = run_prune_check(
            tiny_model, tiny_examples, tiny_examples, [tiny_model.moe], "minimax",
            num_experts=4, top_k=2, keep_map=keep_map, n_boot=0,
        )
        assert out["masked_pairs"] == 2
        assert out[DOMAIN]["n_total"] == 3
        assert out[GENERAL] is not None

    def test_guard_rejects_over_masking(self, tiny_model, tiny_examples):
        with pytest.raises(ValueError, match="top_k"):
            run_prune_check(
                tiny_model, tiny_examples, tiny_examples, [tiny_model.moe], "minimax",
                num_experts=4, top_k=2, keep_map={0: np.array([0])}, n_boot=0,
            )

    def test_empty_general_set(self, tiny_model, tiny_examples):
        out = run_prune_check(
            tiny_model, tiny_examples, [], [tiny_model.moe], "minimax",
            num_experts=4, top_k=2, keep_map={0: np.array([0, 1, 2])}, n_boot=0,
        )
        assert out[GENERAL] is None


# ---------------------------------------------------------------------------
# Trace pass
# ---------------------------------------------------------------------------

class _IdTokenizer:
    """Tokenizer whose ids are word lengths, so token counts are predictable."""

    has_chat_template = False

    def encode(self, text, **kwargs):
        return [(len(w) % 60) + 1 for w in text.split()]


class TestTraceQuestionSet:
    def _run(self, model, questions, **kwargs):
        from mlx_fun.observer import install_hooks, remove_hooks

        stats = ProbeStats(num_layers=1, num_experts=4)
        install_hooks([model.moe], "minimax")
        try:
            examples, skipped = trace_question_set(
                model, model, _IdTokenizer(), {"model_type": "minimax"},
                questions, DOMAIN, stats,
                [model.moe], num_experts=4, **kwargs,
            )
        finally:
            remove_hooks([model.moe])
        return stats, examples, skipped

    def test_teacher_mode_records_one_vector_per_question(self, tiny_model):
        questions = [
            ProbeQuestion(question="what is a mapping type", answer="a key value store"),
            ProbeQuestion(question="explain storage", answer="persistent state"),
        ]
        stats, examples, skipped = self._run(tiny_model, questions)
        assert skipped == []
        assert stats.n_questions[DOMAIN] == 2
        assert len(examples) == 2
        # Each row of mean_freq sums to top_k regardless of answer length.
        assert stats.mean_freq(DOMAIN)[0].sum() == pytest.approx(2.0)

    def test_answer_positions_match_the_example(self, tiny_model):
        questions = [ProbeQuestion(question="one two three", answer="four five")]
        _, examples, _ = self._run(tiny_model, questions)
        assert examples[0].n_answer == 2
        assert examples[0].prompt_len == 3

    def test_missing_answer_is_skipped(self, tiny_model):
        questions = [
            ProbeQuestion(question="no answer here"),
            ProbeQuestion(question="has one", answer="yes it does"),
        ]
        _, examples, skipped = self._run(tiny_model, questions)
        assert len(examples) == 1
        assert skipped[0]["index"] == 0
        assert "answer" in skipped[0]["reason"]

    def test_hooks_removed_on_exception(self, tiny_model):
        from mlx_fun.observer import install_hooks, remove_hooks

        original_cls = type(tiny_model.moe)
        install_hooks([tiny_model.moe], "minimax")
        try:
            with pytest.raises(RuntimeError):
                trace_question_set(
                    tiny_model, tiny_model, _IdTokenizer(), {"model_type": "minimax"},
                    [ProbeQuestion(question="a b", answer="c")], DOMAIN,
                    ProbeStats(1, 4), [tiny_model.moe], num_experts=4,
                    echo=lambda *a: (_ for _ in ()).throw(RuntimeError("boom")),
                )
        finally:
            remove_hooks([tiny_model.moe])
        assert type(tiny_model.moe) is original_cls

    def test_generate_mode_capture_accounting(self, tiny_model, monkeypatch):
        """Generated ids must equal the decode positions captured."""
        import importlib

        n_generated = 3

        class _Response:
            def __init__(self, token):
                self.token = token
                self.text = "x"

        def fake_stream_generate(model, tokenizer, prompt, max_tokens, **kwargs):
            # Mirror mlx-lm: prefill all but the last token, then the last one,
            # then one forward per generated token, yielded after the forward.
            ids = list(prompt)
            model(mx.array(ids[:-1]).reshape(1, -1))
            model(mx.array(ids[-1:]).reshape(1, -1))
            token = 5
            for _ in range(min(n_generated, max_tokens)):
                model(mx.array([token]).reshape(1, -1))
                yield _Response(token)
                token += 1

        # `mlx_lm.generate` as a package attribute is the re-exported *function*,
        # so resolve the module itself before patching.
        gen_module = importlib.import_module("mlx_lm.generate")
        monkeypatch.setattr(gen_module, "stream_generate", fake_stream_generate)

        questions = [ProbeQuestion(question="one two three")]
        stats, examples, skipped = self._run(
            tiny_model, questions, answer_mode="generate", max_answer_tokens=8,
        )
        assert skipped == []
        assert examples[0].n_answer == n_generated
        assert stats.n_questions[DOMAIN] == 1

    def test_prune_check_relabels_status(self, tiny_model, tiny_examples):
        """A credible delta here means damage, not an endorsement of the prune."""
        out = run_prune_check(
            tiny_model, tiny_examples, tiny_examples, [tiny_model.moe], "minimax",
            num_experts=4, top_k=2, keep_map={0: np.array([0, 1])}, n_boot=0,
        )
        assert out[DOMAIN]["interpretation"] in (
            "degraded", "unchanged", "inconclusive", "catastrophic",
        )
        assert "baseline_nll" in out[DOMAIN]
