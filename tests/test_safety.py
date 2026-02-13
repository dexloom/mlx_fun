"""Tests for safety-critical expert analysis."""

import json
import numpy as np
import pytest

from mlx_fun.safety import (
    DifferentialAccumulator,
    SafetyReport,
    compute_differential_scores,
    compute_top_k_from_logits,
    identify_safety_experts,
)


class TestDifferentialAccumulator:
    def test_init_shapes(self):
        acc = DifferentialAccumulator(num_layers=2, num_experts=4)
        assert acc.harmful_gate_sum.shape == (2, 4)
        assert acc.benign_freq.shape == (2, 4)
        assert acc.harmful_tokens.shape == (2,)

    def test_update_from_gate_logits_harmful(self):
        acc = DifferentialAccumulator(num_layers=1, num_experts=4)
        logits = np.array([[2.0, 0.1, 0.5, 0.3], [1.0, 0.2, 0.4, 0.6]])
        acc.update_from_gate_logits(0, logits, "harmful")
        assert acc.harmful_tokens[0] == 2.0
        np.testing.assert_allclose(
            acc.harmful_gate_sum[0], [3.0, 0.3, 0.9, 0.9]
        )

    def test_update_from_gate_logits_benign(self):
        acc = DifferentialAccumulator(num_layers=1, num_experts=4)
        logits = np.array([[0.1, 2.0, 0.5, 0.3]])
        acc.update_from_gate_logits(0, logits, "benign")
        assert acc.benign_tokens[0] == 1.0

    def test_update_from_top_k(self):
        acc = DifferentialAccumulator(num_layers=1, num_experts=4)
        inds = np.array([[0, 2], [1, 3]])
        acc.update_from_top_k(0, inds, "harmful")
        np.testing.assert_array_equal(acc.harmful_freq[0], [1.0, 1.0, 1.0, 1.0])

    def test_datasets_tracked_separately(self):
        acc = DifferentialAccumulator(num_layers=1, num_experts=4)
        logits_h = np.array([[2.0, 0.0, 0.0, 0.0]])
        logits_b = np.array([[0.0, 2.0, 0.0, 0.0]])
        acc.update_from_gate_logits(0, logits_h, "harmful")
        acc.update_from_gate_logits(0, logits_b, "benign")
        assert acc.harmful_gate_sum[0, 0] == 2.0
        assert acc.benign_gate_sum[0, 1] == 2.0
        assert acc.harmful_gate_sum[0, 1] == 0.0
        assert acc.benign_gate_sum[0, 0] == 0.0


class TestComputeDifferentialScores:
    def test_basic_differential(self):
        acc = DifferentialAccumulator(num_layers=1, num_experts=4)
        # Expert 0 activated more on harmful, expert 1 more on benign
        acc.update_from_gate_logits(0, np.array([[2.0, 0.1, 0.5, 0.3]]), "harmful")
        acc.update_from_gate_logits(0, np.array([[0.1, 2.0, 0.5, 0.3]]), "benign")

        diff_freq, diff_act, composite = compute_differential_scores(acc)
        # Expert 0 should have positive activation differential
        assert diff_act[0, 0] > 0
        # Expert 1 should have negative
        assert diff_act[0, 1] < 0

    def test_normalized_range(self):
        acc = DifferentialAccumulator(num_layers=1, num_experts=4)
        acc.update_from_gate_logits(0, np.array([[10.0, -5.0, 0.0, 3.0]]), "harmful")
        acc.update_from_gate_logits(0, np.array([[0.0, 5.0, 0.0, -3.0]]), "benign")
        _, _, composite = compute_differential_scores(acc)
        # Composite should be in [0, 1] after normalization
        assert composite[0].min() >= -0.01
        assert composite[0].max() <= 1.01

    def test_empty_dataset_no_crash(self):
        acc = DifferentialAccumulator(num_layers=1, num_experts=4)
        diff_freq, diff_act, composite = compute_differential_scores(acc)
        # Should all be zeros
        np.testing.assert_array_equal(diff_freq, np.zeros((1, 4)))


class TestIdentifySafetyExperts:
    def test_classifies_hcdg_and_hrcg(self):
        composite = np.array([[0.95, 0.05, 0.5, 0.5]])
        diff_freq = np.zeros((1, 4))
        diff_act = np.zeros((1, 4))
        report = identify_safety_experts(diff_freq, diff_act, composite, threshold_percentile=75.0)
        assert 0 in report.hcdg_experts.get(0, [])  # High positive
        assert 1 in report.hrcg_experts.get(0, [])   # Low

    def test_safety_critical_is_union(self):
        composite = np.array([[0.99, 0.01, 0.5, 0.5]])
        diff_freq = np.zeros((1, 4))
        diff_act = np.zeros((1, 4))
        report = identify_safety_experts(diff_freq, diff_act, composite, threshold_percentile=75.0)
        safety = set(report.safety_critical.get(0, []))
        hcdg = set(report.hcdg_experts.get(0, []))
        hrcg = set(report.hrcg_experts.get(0, []))
        assert safety == hcdg | hrcg

    def test_multiple_layers(self):
        composite = np.array([
            [0.9, 0.1, 0.5, 0.5],
            [0.5, 0.5, 0.9, 0.1],
        ])
        diff_freq = np.zeros((2, 4))
        diff_act = np.zeros((2, 4))
        report = identify_safety_experts(diff_freq, diff_act, composite, threshold_percentile=75.0)
        assert report.num_layers == 2
        assert report.num_experts == 4


class TestSafetyReportSaveLoad:
    def test_roundtrip(self, tmp_path):
        report = SafetyReport(
            num_layers=2,
            num_experts=4,
            threshold_percentile=90.0,
            differential_freq=np.array([[0.1, -0.2, 0.0, 0.05], [0.0, 0.0, 0.1, -0.1]]),
            differential_activation=np.array([[0.5, -0.3, 0.1, 0.0], [0.0, 0.2, -0.1, 0.3]]),
            composite_score=np.array([[0.8, 0.2, 0.5, 0.4], [0.3, 0.6, 0.4, 0.7]]),
            hcdg_experts={0: [0], 1: [3]},
            hrcg_experts={0: [1]},
            safety_critical={0: [0, 1], 1: [3]},
        )
        path = str(tmp_path / "report.json")
        report.save(path)
        loaded = SafetyReport.load(path)

        assert loaded.num_layers == 2
        assert loaded.num_experts == 4
        assert loaded.threshold_percentile == 90.0
        np.testing.assert_allclose(loaded.differential_freq, report.differential_freq)
        np.testing.assert_allclose(loaded.composite_score, report.composite_score)
        assert loaded.hcdg_experts == {0: [0], 1: [3]}
        assert loaded.hrcg_experts == {0: [1]}
        assert loaded.safety_critical == {0: [0, 1], 1: [3]}


class TestComputeTopKFromLogits:
    def test_minimax_sigmoid_routing(self):
        logits = np.array([[10.0, -10.0, 5.0, -5.0]])
        inds = compute_top_k_from_logits(logits, "minimax", top_k=2)
        assert inds.shape == (1, 2)
        # Experts 0 and 2 should have highest sigmoid scores
        assert set(inds[0].tolist()) == {0, 2}

    def test_qwen3_softmax_routing(self):
        logits = np.array([[10.0, -10.0, 5.0, -5.0]])
        inds = compute_top_k_from_logits(logits, "qwen3_moe", top_k=2)
        assert inds.shape == (1, 2)
        assert set(inds[0].tolist()) == {0, 2}

    def test_glm4_uses_sigmoid(self):
        logits = np.array([[1.0, -1.0, 0.5, 0.0]])
        inds = compute_top_k_from_logits(logits, "glm4_moe", top_k=2)
        assert inds.shape == (1, 2)
        # Expert 0 (1.0) and expert 2 (0.5) have highest sigmoid values
        assert set(inds[0].tolist()) == {0, 2}

    def test_batch_tokens(self):
        logits = np.random.randn(16, 8)  # 16 tokens, 8 experts
        inds = compute_top_k_from_logits(logits, "minimax", top_k=3)
        assert inds.shape == (16, 3)
        # All indices in valid range
        assert np.all(inds >= 0) and np.all(inds < 8)

    def test_unknown_model_type_raises(self):
        with pytest.raises(ValueError, match="Unknown model_type"):
            compute_top_k_from_logits(np.zeros((1, 4)), "unknown_model", top_k=2)
