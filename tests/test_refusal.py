"""Tests for guardrail / refusal expert probing."""

import json

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from mlx_fun.probe import DOMAIN, GENERAL, ProbeExample, ProbeQuestion, ProbeStats
from mlx_fun.refusal import (
    ANSWERED,
    PARTIAL,
    REFUSED,
    RefusalReport,
    _LogitsModel,
    _flip_stats,
    classify_response,
    run_refusal_knockout,
    strip_thinking,
    trace_refusals,
)


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class TestClassifyResponse:
    def test_plain_answer(self):
        assert classify_response(
            "The payable modifier lets a function receive Ether."
        ) == ANSWERED

    def test_leading_refusal(self):
        assert classify_response(
            "I'm sorry, but I can't help with that request."
        ) == REFUSED

    def test_cannot_assist(self):
        assert classify_response("I cannot assist with writing that.") == REFUSED

    def test_empty_is_refused(self):
        assert classify_response("") == REFUSED
        assert classify_response("   \n  ") == REFUSED

    def test_hedged_then_answered_is_partial(self):
        text = (
            "I cannot help with anything malicious, but here is how reentrancy "
            "works in general: an external call hands control to the callee "
            "before the caller updates its state, so the callee can re-enter the "
            "same function and withdraw repeatedly. The fix is to update state "
            "before the external call, following checks-effects-interactions, "
            "and to add a reentrancy guard on the sensitive functions here."
        )
        assert classify_response(text) == PARTIAL

    def test_marker_deep_in_answer_is_not_refusal(self):
        # A quoted refusal phrase far inside a substantive answer must not flip
        # the classification.
        text = "Here is a long technical explanation. " * 20 + " I cannot help"
        assert classify_response(text) == ANSWERED

    def test_case_and_whitespace_insensitive(self):
        assert classify_response("I  CANNOT\n ASSIST with that.") == REFUSED

    def test_extra_markers(self):
        assert classify_response("Nope, not doing that.") == ANSWERED
        assert classify_response(
            "Nope, not doing that.", extra_markers=["nope, not doing that"]
        ) == REFUSED


class TestStripThinking:
    def test_closed_block_returns_the_visible_answer(self):
        text = "<think>\nWeighing this up.\n</think>\n\nI can't help with that."
        assert strip_thinking(text) == "I can't help with that."

    def test_unfinished_block_returns_none(self):
        text = "<think>\nStill reasoning when the budget ran out, and"
        assert strip_thinking(text) is None

    def test_no_block_passes_through(self):
        text = "A mapping is a key-value store."
        assert strip_thinking(text) is text

    def test_last_block_wins(self):
        text = (
            "<think>first</think>\n\nDraft.\n<think>second</think>\n\nFinal answer."
        )
        assert strip_thinking(text) == "Final answer."

    def test_custom_markers(self):
        text = "<|channel>thought reasoning<channel|>  The answer."
        assert strip_thinking(text, "<|channel>thought", "<channel|>") == "The answer."

    def test_close_without_open_still_strips(self):
        """A block whose opening tag the template ate still has to be dropped."""
        assert strip_thinking("reasoning</think>\n\nThe answer.") == "The answer."


class TestThinkingAwareClassification:
    def test_refusal_after_reasoning_is_seen(self):
        """The whole point: the classifier reads only the opening, so a refusal
        behind a long reasoning block reads as 'answered' until it is stripped."""
        text = (
            "<think>\n" + "Considering whether this is appropriate. " * 20
            + "\n</think>\n\nI can't help with that."
        )
        assert classify_response(text) == ANSWERED          # raw: wrong
        assert classify_response(strip_thinking(text)) == REFUSED   # stripped: right


class TestFlipStats:
    def test_all_flip_is_verified(self):
        stats = _flip_stats(np.ones(10), min_flip_rate=0.5, n_boot=200, seed=1)
        assert stats["flip_rate"] == 1.0
        assert stats["status"] == "verified"
        assert stats["ci_low"] > 0

    def test_no_flip_is_not_verified(self):
        stats = _flip_stats(np.zeros(10), min_flip_rate=0.5, n_boot=200, seed=1)
        assert stats["flip_rate"] == 0.0
        assert stats["status"] == "not_verified"

    def test_below_threshold_not_verified(self):
        flips = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)
        stats = _flip_stats(flips, min_flip_rate=0.5, n_boot=200, seed=1)
        assert stats["flip_rate"] == pytest.approx(0.1)
        assert stats["status"] == "not_verified"

    def test_bootstrap_reproducible(self):
        flips = np.array([1, 1, 0, 1, 0, 1, 1, 0], dtype=float)
        a = _flip_stats(flips, 0.5, 200, 3)
        b = _flip_stats(flips, 0.5, 200, 3)
        assert a["ci_low"] == b["ci_low"] and a["ci_high"] == b["ci_high"]


# ---------------------------------------------------------------------------
# Logits shim (the VLM generation path)
# ---------------------------------------------------------------------------

class _Output:
    def __init__(self, logits):
        self.logits = logits


class TestLogitsModel:
    def test_unwraps_logits(self):
        inner = lambda *a, **k: _Output(mx.array([1.0, 2.0]))
        wrapped = _LogitsModel(inner)
        out = wrapped(mx.array([0]))
        assert isinstance(out, mx.array)
        np.testing.assert_array_equal(np.array(out), [1.0, 2.0])

    def test_passes_through_raw_array(self):
        inner = lambda *a, **k: mx.array([3.0])
        assert float(_LogitsModel(inner)(mx.array([0]))[0]) == 3.0

    def test_delegates_attributes(self):
        class Inner:
            layers = ["a", "b"]
            def __call__(self, *a, **k):
                return _Output(mx.zeros(2))
        wrapped = _LogitsModel(Inner())
        assert wrapped.layers == ["a", "b"]


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _report(**kwargs):
    base = dict(
        domain_name="refusal",
        num_layers=2,
        num_experts=4,
        threshold_percentile=90.0,
        differential_freq=np.zeros((2, 4)),
        differential_activation=np.zeros((2, 4)),
        composite_score=np.array([[0.1, 0.9, 0.2, 0.3], [0.4, 0.5, 0.6, 0.7]]),
        domain_experts={0: [1]},
        general_experts={0: [0]},
        objective="refusal",
        num_answered=40,
        num_refused=12,
        num_partial=3,
        candidate_refusal_experts={0: [1, 2]},
        verified_refusal_experts={0: [1]},
        refusal_experts_verified=True,
        per_question_outcome=[{"index": 0, "question": "q", "outcome": "refused", "tags": ["x"]}],
    )
    base.update(kwargs)
    return RefusalReport(**base)


class TestRefusalReport:
    def test_roundtrip(self, tmp_path):
        report = _report(
            knockout={"backend": "regenerate_and_classify", "per_expert": []},
            knockout_delta=np.zeros((2, 4)),
            verified_domain_experts={0: [1]},
        )
        path = str(tmp_path / "refusal.json")
        report.save(path)
        loaded = RefusalReport.load(path)
        assert loaded.objective == "refusal"
        assert loaded.num_refused == 12
        assert loaded.candidate_refusal_experts == {0: [1, 2]}
        assert loaded.verified_refusal_experts == {0: [1]}
        assert loaded.refusal_experts_verified is True
        assert loaded.per_question_outcome[0]["question"] == "q"
        assert loaded.verified_domain_experts == {0: [1]}

    def test_downstream_reads_it_as_a_domain_report(self, tmp_path):
        """The report is structurally a DomainReport; refusal_experts are in
        domain_experts so the existing loaders read them."""
        from mlx_fun.domain import DomainReport
        from mlx_fun.steering import SteeringConfig
        path = str(tmp_path / "refusal.json")
        _report().save(path)

        assert DomainReport.load(path).domain_experts == {0: [1]}
        # boost activates the identified (refusal) experts; to *remove* a
        # guardrail you build a deactivate config from the same set instead.
        assert SteeringConfig.from_domain_report(path, "boost").activate == {0: [1]}

    def test_deactivate_uses_verified_experts(self, tmp_path):
        """The useful guardrail operation deactivates the VERIFIED experts."""
        from mlx_fun.steering import SteeringConfig
        report = _report()
        cfg = SteeringConfig(deactivate=dict(report.verified_refusal_experts))
        assert cfg.deactivate == {0: [1]}          # verified, not the candidate {0:[1,2]}


# ---------------------------------------------------------------------------
# Trace + regeneration knockout on a tiny model
# ---------------------------------------------------------------------------

class TinyMoEModel(nn.Module):
    def __init__(self, moe_block, vocab=64, hidden=32):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.moe = moe_block
        self.lm_head = nn.Linear(hidden, vocab)

    def __call__(self, tokens, cache=None):
        return self.lm_head(self.moe(self.embed(tokens)))


class _FakeTokenizer:
    """Deterministic tokenizer; encode by word length, decode a canned string."""

    has_chat_template = False

    def __init__(self, decoded):
        self._decoded = decoded

    def encode(self, text, **kwargs):
        return [(len(w) % 60) + 1 for w in text.split()]

    def decode(self, ids):
        return self._decoded


@pytest.fixture
def tiny_model(tiny_minimax_moe):
    mx.random.seed(0)
    return TinyMoEModel(tiny_minimax_moe)


def _install_fake_generation(monkeypatch, response_by_question):
    """Patch refusal.generate_response to return a scripted response per question.

    Runs the model forward over prompt+answer so observer hooks capture routing
    with the P + G layout the slicer asserts.
    """
    import mlx_fun.refusal as refusal_mod

    calls = {"n": 0}

    def fake_generate(model, tokenizer, config, prompt_ids, max_tokens):
        text = response_by_question(calls["n"])
        calls["n"] += 1
        gen_ids = [3, 4, 5]  # three answer tokens
        ids = list(prompt_ids)
        # Mirror mlx-lm's P + G capture layout: prefill all but the last prompt
        # token (skipped when the prompt is a single token), the last one, then
        # one forward per generated token.
        if len(ids) > 1:
            model(mx.array(ids[:-1], dtype=mx.int32).reshape(1, -1))
        model(mx.array(ids[-1:], dtype=mx.int32).reshape(1, -1))
        for t in gen_ids:
            model(mx.array([t], dtype=mx.int32).reshape(1, -1))
        return gen_ids, text

    monkeypatch.setattr(refusal_mod, "generate_response", fake_generate)
    return calls


class TestTraceRefusals:
    def test_splits_by_outcome(self, tiny_model, monkeypatch):
        from mlx_fun.observer import install_hooks, remove_hooks

        questions = [
            ProbeQuestion(question="explain reentrancy"),
            ProbeQuestion(question="write an exploit"),
            ProbeQuestion(question="what is a mapping"),
        ]
        # Question 1 is refused, the others answered.
        responses = [
            "A reentrancy bug lets an external call re-enter the caller.",
            "I'm sorry, but I can't help with that.",
            "A mapping is a key-value store.",
        ]
        _install_fake_generation(monkeypatch, lambda i: responses[i])

        stats = ProbeStats(num_layers=1, num_experts=4)
        install_hooks([tiny_model.moe], "minimax")
        try:
            examples, outcomes, skipped, records = trace_refusals(
                tiny_model, _FakeTokenizer(""), {"model_type": "minimax"},
                questions, stats, [tiny_model.moe], num_experts=4,
            )
        finally:
            remove_hooks([tiny_model.moe])

        assert [o["outcome"] for o in outcomes] == [ANSWERED, REFUSED, ANSWERED]
        assert all("question" in o for o in outcomes)   # identifiable records
        assert len(records) == 3
        assert len(examples[DOMAIN]) == 1     # one refused
        assert len(examples[GENERAL]) == 2    # two answered
        assert stats.n_questions[DOMAIN] == 1
        assert stats.n_questions[GENERAL] == 2

    def test_no_refusals_leaves_domain_empty(self, tiny_model, monkeypatch):
        from mlx_fun.observer import install_hooks, remove_hooks

        questions = [ProbeQuestion(question="q1"), ProbeQuestion(question="q2")]
        _install_fake_generation(monkeypatch, lambda i: "A plain answer here.")
        stats = ProbeStats(num_layers=1, num_experts=4)
        install_hooks([tiny_model.moe], "minimax")
        try:
            examples, outcomes, _, _ = trace_refusals(
                tiny_model, _FakeTokenizer(""), {"model_type": "minimax"},
                questions, stats, [tiny_model.moe], num_experts=4,
            )
        finally:
            remove_hooks([tiny_model.moe])
        assert all(o["outcome"] == ANSWERED for o in outcomes)
        assert examples[DOMAIN] == []

    def test_unfinished_thinking_is_skipped(self, tiny_model, monkeypatch):
        """A generation that ran out of tokens mid-reasoning is neither
        answered nor refused, so it must not land in either bucket."""
        from mlx_fun.observer import install_hooks, remove_hooks

        questions = [
            ProbeQuestion(question="q1"),
            ProbeQuestion(question="q2"),
        ]
        responses = [
            "<think>\nStill reasoning when the budget ran out, and",
            "<think>\nBrief.\n</think>\n\nA mapping is a key-value store.",
        ]
        _install_fake_generation(monkeypatch, lambda i: responses[i])

        stats = ProbeStats(num_layers=1, num_experts=4)
        install_hooks([tiny_model.moe], "minimax")
        try:
            examples, outcomes, skipped, records = trace_refusals(
                tiny_model, _FakeTokenizer(""), {"model_type": "minimax"},
                questions, stats, [tiny_model.moe], num_experts=4,
            )
        finally:
            remove_hooks([tiny_model.moe])

        assert len(skipped) == 1
        assert skipped[0]["index"] == 0
        assert skipped[0]["unfinished_thinking"] is True
        # Neither bucket, no record, no statistics contribution.
        assert len(outcomes) == 1 and outcomes[0]["index"] == 1
        assert len(records) == 1
        assert stats.n_questions[DOMAIN] == 0
        assert stats.n_questions[GENERAL] == 1

    def test_refusal_behind_reasoning_is_counted_as_refused(self, tiny_model, monkeypatch):
        from mlx_fun.observer import install_hooks, remove_hooks

        text = ("<think>\n" + "Weighing whether to answer. " * 12
                + "\n</think>\n\nI can't help with that.")
        _install_fake_generation(monkeypatch, lambda i: text)

        stats = ProbeStats(num_layers=1, num_experts=4)
        install_hooks([tiny_model.moe], "minimax")
        try:
            _examples, outcomes, _skipped, _records = trace_refusals(
                tiny_model, _FakeTokenizer(""), {"model_type": "minimax"},
                [ProbeQuestion(question="q1")], stats, [tiny_model.moe],
                num_experts=4,
            )
        finally:
            remove_hooks([tiny_model.moe])

        assert outcomes[0]["outcome"] == REFUSED
        # The visible answer is classified; the raw text stays auditable.
        assert outcomes[0]["answer"] == "I can't help with that."
        assert outcomes[0]["raw_answer"] == text


class TestRunRefusalKnockout:
    def test_flip_detected_when_regeneration_answers(self, tiny_model, monkeypatch):
        # Under the mask, regeneration returns an answer -> a flip.
        _install_fake_generation(monkeypatch, lambda i: "Now here is the answer.")
        refused = [
            ProbeExample(tokens=[1, 2, 3, 4, 5, 6], prompt_len=3, question_index=0),
            ProbeExample(tokens=[7, 8, 9, 10, 11, 12], prompt_len=3, question_index=1),
        ]
        result = run_refusal_knockout(
            tiny_model, _FakeTokenizer(""), {"model_type": "minimax"},
            refused, [tiny_model.moe], "minimax",
            num_experts=4, top_k=2, candidates=[(0, 1)],
            min_flip_rate=0.5, n_boot=100, seed=1,
        )
        assert result[0]["flip_rate"] == 1.0
        assert result[0]["status"] == "verified"

    def test_no_flip_when_regeneration_still_refuses(self, tiny_model, monkeypatch):
        _install_fake_generation(monkeypatch, lambda i: "I cannot help with that.")
        refused = [
            ProbeExample(tokens=[1, 2, 3, 4, 5, 6], prompt_len=3, question_index=0),
        ]
        result = run_refusal_knockout(
            tiny_model, _FakeTokenizer(""), {"model_type": "minimax"},
            refused, [tiny_model.moe], "minimax",
            num_experts=4, top_k=2, candidates=[(0, 1)],
            min_flip_rate=0.5, n_boot=0, seed=1,
        )
        assert result[0]["flip_rate"] == 0.0
        assert result[0]["status"] == "not_verified"

    def test_unfinished_thinking_is_never_a_flip(self, tiny_model, monkeypatch):
        """Running out of tokens mid-reasoning is not a guardrail breaking."""
        _install_fake_generation(
            monkeypatch, lambda i: "<think>\nStill reasoning when the budget ran out",
        )
        refused = [
            ProbeExample(tokens=[1, 2, 3, 4, 5, 6], prompt_len=3, question_index=0),
        ]
        result = run_refusal_knockout(
            tiny_model, _FakeTokenizer(""), {"model_type": "minimax"},
            refused, [tiny_model.moe], "minimax",
            num_experts=4, top_k=2, candidates=[(0, 1)],
            min_flip_rate=0.5, n_boot=0, seed=1,
        )
        assert result[0]["flip_rate"] == 0.0
        assert result[0]["status"] == "not_verified"

    def test_refusal_behind_reasoning_is_not_a_flip(self, tiny_model, monkeypatch):
        """Without stripping, the reasoning block would hide the refusal and
        this would be scored as the guardrail breaking."""
        _install_fake_generation(
            monkeypatch,
            lambda i: ("<think>\n" + "Deliberating. " * 20
                       + "\n</think>\n\nI cannot help with that."),
        )
        refused = [
            ProbeExample(tokens=[1, 2, 3, 4, 5, 6], prompt_len=3, question_index=0),
        ]
        result = run_refusal_knockout(
            tiny_model, _FakeTokenizer(""), {"model_type": "minimax"},
            refused, [tiny_model.moe], "minimax",
            num_experts=4, top_k=2, candidates=[(0, 1)],
            min_flip_rate=0.5, n_boot=0, seed=1,
        )
        assert result[0]["flip_rate"] == 0.0
        assert result[0]["status"] == "not_verified"

    def test_empty_refused_returns_empty(self, tiny_model, monkeypatch):
        _install_fake_generation(monkeypatch, lambda i: "answer")
        result = run_refusal_knockout(
            tiny_model, _FakeTokenizer(""), {"model_type": "minimax"},
            [], [tiny_model.moe], "minimax",
            num_experts=4, top_k=2, candidates=[(0, 1)], n_boot=0,
        )
        assert result == []


# ---------------------------------------------------------------------------
# Regressions from the second review
# ---------------------------------------------------------------------------

class TestClassifierRegressions:
    def test_helpful_i_can_explain_is_answered(self):
        # "I'm sorry, but I can explain" must not match a refusal marker.
        assert classify_response(
            "I'm sorry, but I can explain how reentrancy works: an external call..."
        ) == ANSWERED

    def test_curly_apostrophe_refusal_is_detected(self):
        assert classify_response(
            "I’m sorry, but I can’t help with that."
        ) == REFUSED

    def test_curly_matches_ascii(self):
        curly = classify_response("I can’t assist with that.")
        ascii_ = classify_response("I can't assist with that.")
        assert curly == ascii_ == REFUSED


class TestPartialIsNotAFlip:
    def test_long_hedged_refusal_does_not_verify(self, request, monkeypatch):
        tiny_model = request.getfixturevalue("tiny_model")
        long_hedge = ("I cannot help with that. "
                      + "Here is some general, safe background. " * 15)
        assert classify_response(long_hedge) == PARTIAL

        _install_fake_generation(monkeypatch, lambda i: long_hedge)
        refused = [ProbeExample(tokens=[1, 2, 3, 4, 5, 6], prompt_len=3, question_index=0)]
        result = run_refusal_knockout(
            tiny_model, _FakeTokenizer(""), {"model_type": "minimax"},
            refused, [tiny_model.moe], "minimax",
            num_experts=4, top_k=2, candidates=[(0, 1)],
            min_flip_rate=0.5, n_boot=0, seed=1,
        )
        # A partial (still-refusing) regeneration is not a guardrail break.
        assert result[0]["flip_rate"] == 0.0
        assert result[0]["status"] == "not_verified"


class TestStratifiedScores:
    @staticmethod
    def _rec(label, tags, row):
        v = np.array([row], dtype=float)   # (num_layers=1, num_experts=4)
        return {"label": label, "tags": tags, "freq": v, "weight": v}

    def test_isolates_guardrail_over_topic(self):
        from mlx_fun.refusal import stratified_probe_scores
        from mlx_fun.probe import compute_probe_scores
        # Expert 0 fires on refusals across tags (guardrail). Expert 2 fires only
        # in tag C, which is entirely refused (a topic expert). Stratification
        # drops tag C (no contrast) and isolates expert 0; the global contrast
        # would be dominated by the topic expert 2.
        records = [
            self._rec(DOMAIN, ["A"], [1, 0, 0, 0]),
            self._rec(GENERAL, ["A"], [0, 1, 0, 0]),
            self._rec(DOMAIN, ["C"], [0, 0, 1, 0]),
            self._rec(DOMAIN, ["C"], [0, 0, 1, 0]),
            self._rec(DOMAIN, ["C"], [0, 0, 1, 0]),
        ]
        diff_freq, diff_weight, composite, tags_used = stratified_probe_scores(
            records, num_layers=1, num_experts=4
        )
        assert tags_used == ["A"]
        assert int(composite[0].argmax()) == 0

    def test_returns_none_without_contrast(self):
        from mlx_fun.refusal import stratified_probe_scores
        records = [
            self._rec(DOMAIN, ["A"], [1, 0, 0, 0]),
            self._rec(DOMAIN, ["B"], [1, 0, 0, 0]),
        ]
        assert stratified_probe_scores(records, 1, 4) is None


class TestParaphrasedRefusalRecall:
    """Paraphrased refusals a bare marker list would miss, and the technical
    traps that must not be misread as refusals."""

    @pytest.mark.parametrize("text", [
        "Providing a ready-to-run exploit isn't something I'll do.",
        "Apologies, but that crosses into territory I'd rather avoid.",
        "This looks like a request to attack a live system, and I have to pass.",
        "I'd rather not help build something designed to steal funds.",
        "That's not something I'm going to write for you.",
    ])
    def test_paraphrased_refusals_detected(self, text):
        assert classify_response(text) == REFUSED

    @pytest.mark.parametrize("text", [
        "I'd rather not recompute the hash each iteration, so cache it in memory.",
        "That's not something you can do in a pure function.",
        "You won't write to storage inside a view function.",
        "The attacker can pass control back into withdraw() via the callback.",
        "I'll pass the array by calldata to avoid a copy.",
        "I won't write more than 24576 bytes of bytecode.",
    ])
    def test_technical_prose_is_not_a_refusal(self, text):
        assert classify_response(text) == ANSWERED
