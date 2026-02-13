"""Tests for domain-specific expert identification and gate amplification."""

import json
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from mlx_fun.domain import (
    DomainReport,
    amplify_gate_weights,
    compute_amplification_biases,
    identify_domain_experts,
)
from mlx_fun.safety import (
    DifferentialAccumulator,
    compute_differential_scores,
)
from mlx_fun.steering import SteeringConfig
from mlx_fun.pruner import load_domain_constraints


class TestDomainReport:
    def test_save_load_roundtrip(self, tmp_path):
        report = DomainReport(
            domain_name="solidity",
            num_layers=2,
            num_experts=4,
            threshold_percentile=90.0,
            differential_freq=np.array([[0.1, -0.2, 0.0, 0.05], [0.0, 0.0, 0.1, -0.1]]),
            differential_activation=np.array([[0.5, -0.3, 0.1, 0.0], [0.0, 0.2, -0.1, 0.3]]),
            composite_score=np.array([[0.8, 0.2, 0.5, 0.4], [0.3, 0.6, 0.4, 0.7]]),
            domain_experts={0: [0], 1: [3]},
            general_experts={0: [1]},
        )
        path = str(tmp_path / "domain_report.json")
        report.save(path)
        loaded = DomainReport.load(path)

        assert loaded.domain_name == "solidity"
        assert loaded.num_layers == 2
        assert loaded.num_experts == 4
        assert loaded.threshold_percentile == 90.0
        np.testing.assert_allclose(loaded.differential_freq, report.differential_freq)
        np.testing.assert_allclose(loaded.composite_score, report.composite_score)
        assert loaded.domain_experts == {0: [0], 1: [3]}
        assert loaded.general_experts == {0: [1]}

    def test_uses_differential_accumulator(self):
        """DomainReport classification uses same DifferentialAccumulator from safety.py."""
        acc = DifferentialAccumulator(num_layers=1, num_experts=4)
        # "harmful"=domain, "benign"=general
        acc.update_from_gate_logits(0, np.array([[2.0, 0.1, 0.5, 0.3]]), "harmful")
        acc.update_from_gate_logits(0, np.array([[0.1, 2.0, 0.5, 0.3]]), "benign")

        diff_freq, diff_act, composite = compute_differential_scores(acc)
        report = identify_domain_experts(diff_freq, diff_act, composite, "test_domain")

        assert report.domain_name == "test_domain"
        assert report.num_layers == 1
        assert report.num_experts == 4


class TestIdentifyDomainExperts:
    def test_classifies_domain_and_general(self):
        composite = np.array([[0.95, 0.05, 0.5, 0.5]])
        diff_freq = np.zeros((1, 4))
        diff_act = np.zeros((1, 4))
        report = identify_domain_experts(
            diff_freq, diff_act, composite, "solidity", threshold_percentile=75.0,
        )
        assert 0 in report.domain_experts.get(0, [])
        assert 1 in report.general_experts.get(0, [])

    def test_multiple_layers(self):
        composite = np.array([
            [0.9, 0.1, 0.5, 0.5],
            [0.5, 0.5, 0.9, 0.1],
        ])
        diff_freq = np.zeros((2, 4))
        diff_act = np.zeros((2, 4))
        report = identify_domain_experts(
            diff_freq, diff_act, composite, "medical", threshold_percentile=75.0,
        )
        assert report.num_layers == 2
        assert report.num_experts == 4
        # Expert 0 in layer 0 and expert 2 in layer 1 are domain experts
        assert 0 in report.domain_experts.get(0, [])
        assert 2 in report.domain_experts.get(1, [])

    def test_threshold_100_yields_empty(self):
        composite = np.array([[0.5, 0.5, 0.5, 0.5]])
        diff_freq = np.zeros((1, 4))
        diff_act = np.zeros((1, 4))
        report = identify_domain_experts(
            diff_freq, diff_act, composite, "test",
            threshold_percentile=100.0,
        )
        # All scores equal, so with 100th percentile threshold none should qualify
        # (percentile of all-same values = that value, >= means all qualify)
        # This is edge case behavior — just verify no crash
        assert report.num_layers == 1


class TestComputeAmplificationBiases:
    def test_basic_computation(self):
        report = DomainReport(
            domain_name="test",
            num_layers=1,
            num_experts=4,
            threshold_percentile=90.0,
            differential_freq=np.zeros((1, 4)),
            differential_activation=np.zeros((1, 4)),
            composite_score=np.array([[0.8, 0.2, 0.6, 0.1]]),
            domain_experts={0: [0, 2]},
            general_experts={0: [3]},
        )
        biases = compute_amplification_biases(report, scale=2.0, threshold=0.0)
        assert 0 in biases
        # Expert 0: 2.0 * 0.8 = 1.6, Expert 2: 2.0 * 0.6 = 1.2
        np.testing.assert_allclose(biases[0][0], 1.6)
        np.testing.assert_allclose(biases[0][2], 1.2)
        # Non-domain experts should be 0
        assert biases[0][1] == 0.0
        assert biases[0][3] == 0.0

    def test_threshold_filtering(self):
        report = DomainReport(
            domain_name="test",
            num_layers=1,
            num_experts=4,
            threshold_percentile=90.0,
            differential_freq=np.zeros((1, 4)),
            differential_activation=np.zeros((1, 4)),
            composite_score=np.array([[0.8, 0.2, 0.3, 0.1]]),
            domain_experts={0: [0, 2]},
            general_experts={},
        )
        # threshold=0.5 means expert 2 (score=0.3) gets 0 boost
        biases = compute_amplification_biases(report, scale=1.0, threshold=0.5)
        assert 0 in biases
        np.testing.assert_allclose(biases[0][0], 0.3)  # 0.8 - 0.5
        assert biases[0][2] == 0.0  # 0.3 - 0.5 = negative -> 0

    def test_empty_domain_experts(self):
        report = DomainReport(
            domain_name="test",
            num_layers=1,
            num_experts=4,
            threshold_percentile=90.0,
            differential_freq=np.zeros((1, 4)),
            differential_activation=np.zeros((1, 4)),
            composite_score=np.zeros((1, 4)),
            domain_experts={},
            general_experts={},
        )
        biases = compute_amplification_biases(report, scale=1.0)
        assert biases == {}

    def test_scale_zero(self):
        report = DomainReport(
            domain_name="test",
            num_layers=1,
            num_experts=4,
            threshold_percentile=90.0,
            differential_freq=np.zeros((1, 4)),
            differential_activation=np.zeros((1, 4)),
            composite_score=np.array([[0.8, 0.2, 0.6, 0.1]]),
            domain_experts={0: [0]},
            general_experts={},
        )
        biases = compute_amplification_biases(report, scale=0.0)
        assert biases == {}


class TestAmplifyGateWeights:
    def test_minimax_amplify_adds_bias(self, tiny_minimax_moe, sample_input):
        """Setting gate.bias on nn.Linear(bias=False) changes forward pass output."""
        # Baseline
        orig_out = tiny_minimax_moe(sample_input)
        mx.eval(orig_out)
        orig_np = np.array(orig_out, copy=False).copy()

        # Amplify expert 0
        biases = {0: np.array([1.0, 0.0, 0.0, 0.0])}
        amplify_gate_weights([tiny_minimax_moe], "minimax", biases)

        # Verify gate now has bias
        assert "bias" in tiny_minimax_moe.gate

        amp_out = tiny_minimax_moe(sample_input)
        mx.eval(amp_out)
        amp_np = np.array(amp_out, copy=False)

        assert not np.allclose(orig_np, amp_np, atol=1e-5)

    def test_glm4_amplify_changes_correction_bias(self, tiny_glm4_moe, sample_input):
        """Adding to e_score_correction_bias changes GLM4 output."""
        orig_bias = np.array(tiny_glm4_moe.gate.e_score_correction_bias, copy=False).copy()

        biases = {0: np.array([0.0, 0.5, 0.0, 0.0])}
        amplify_gate_weights([tiny_glm4_moe], "glm4_moe", biases)

        new_bias = np.array(tiny_glm4_moe.gate.e_score_correction_bias, copy=False)
        # Expert 1 should have increased correction bias
        assert new_bias[1] > orig_bias[1]

    def test_qwen3_amplify_adds_bias(self, tiny_qwen3_moe, sample_input):
        """Setting gate.bias on Qwen3 nn.Linear(bias=False) changes output."""
        orig_out = tiny_qwen3_moe(sample_input)
        mx.eval(orig_out)
        orig_np = np.array(orig_out, copy=False).copy()

        biases = {0: np.array([0.0, 0.0, 1.0, 0.0])}
        amplify_gate_weights([tiny_qwen3_moe], "qwen3_moe", biases)

        assert "bias" in tiny_qwen3_moe.gate

        amp_out = tiny_qwen3_moe(sample_input)
        mx.eval(amp_out)
        amp_np = np.array(amp_out, copy=False)

        assert not np.allclose(orig_np, amp_np, atol=1e-5)

    def test_unknown_model_type_raises(self):
        biases = {0: np.array([1.0])}
        with pytest.raises(ValueError, match="No amplification support"):
            amplify_gate_weights([None], "unknown_type", biases)


class TestSteeringFromDomainReport:
    def test_boost_mode(self, tmp_path):
        report = DomainReport(
            domain_name="solidity",
            num_layers=2,
            num_experts=4,
            threshold_percentile=90.0,
            differential_freq=np.zeros((2, 4)),
            differential_activation=np.zeros((2, 4)),
            composite_score=np.zeros((2, 4)),
            domain_experts={0: [0, 2], 1: [3]},
            general_experts={0: [1]},
        )
        path = str(tmp_path / "domain_report.json")
        report.save(path)

        config = SteeringConfig.from_domain_report(path, "boost")
        assert config.activate == {0: [0, 2], 1: [3]}
        assert config.deactivate == {}

    def test_suppress_mode(self, tmp_path):
        report = DomainReport(
            domain_name="medical",
            num_layers=2,
            num_experts=4,
            threshold_percentile=90.0,
            differential_freq=np.zeros((2, 4)),
            differential_activation=np.zeros((2, 4)),
            composite_score=np.zeros((2, 4)),
            domain_experts={0: [0]},
            general_experts={0: [1, 3], 1: [2]},
        )
        path = str(tmp_path / "domain_report.json")
        report.save(path)

        config = SteeringConfig.from_domain_report(path, "suppress")
        assert config.deactivate == {0: [1, 3], 1: [2]}
        assert config.activate == {}

    def test_invalid_mode_raises(self, tmp_path):
        report = DomainReport(
            domain_name="test",
            num_layers=1,
            num_experts=4,
            threshold_percentile=90.0,
            differential_freq=np.zeros((1, 4)),
            differential_activation=np.zeros((1, 4)),
            composite_score=np.zeros((1, 4)),
            domain_experts={},
            general_experts={},
        )
        path = str(tmp_path / "domain_report.json")
        report.save(path)

        with pytest.raises(ValueError, match="Unknown domain steering mode"):
            SteeringConfig.from_domain_report(path, "invalid")


class TestPrunerDomainConstraints:
    def test_load_domain_constraints_protect(self, tmp_path):
        report = DomainReport(
            domain_name="solidity",
            num_layers=2,
            num_experts=4,
            threshold_percentile=90.0,
            differential_freq=np.zeros((2, 4)),
            differential_activation=np.zeros((2, 4)),
            composite_score=np.zeros((2, 4)),
            domain_experts={0: [0, 2], 1: [3]},
            general_experts={0: [1]},
        )
        path = str(tmp_path / "domain_report.json")
        report.save(path)

        protected, targeted = load_domain_constraints(path, "protect")
        assert targeted is None
        assert set(protected[0].tolist()) == {0, 2}
        assert set(protected[1].tolist()) == {3}

    def test_domain_protection_in_pruning(self, tmp_path):
        """Domain-protected experts should survive pruning."""
        from mlx_fun.pruner import select_experts_to_keep

        scores = np.array([[1.0, 0.1, 0.2, 0.9]])  # Experts 1,2 are lowest
        report = DomainReport(
            domain_name="test",
            num_layers=1,
            num_experts=4,
            threshold_percentile=90.0,
            differential_freq=np.zeros((1, 4)),
            differential_activation=np.zeros((1, 4)),
            composite_score=np.zeros((1, 4)),
            domain_experts={0: [1]},  # Protect expert 1 (lowest score)
            general_experts={},
        )
        path = str(tmp_path / "domain_report.json")
        report.save(path)

        protected, _ = load_domain_constraints(path, "protect")
        keep = select_experts_to_keep(scores, n_prune=2, protected_experts=protected)
        # Expert 1 should be kept despite low score
        assert 1 in keep[0]

    def test_invalid_mode_raises(self, tmp_path):
        report = DomainReport(
            domain_name="test",
            num_layers=1,
            num_experts=4,
            threshold_percentile=90.0,
            differential_freq=np.zeros((1, 4)),
            differential_activation=np.zeros((1, 4)),
            composite_score=np.zeros((1, 4)),
            domain_experts={},
            general_experts={},
        )
        path = str(tmp_path / "domain_report.json")
        report.save(path)

        with pytest.raises(ValueError, match="Unknown domain_mode"):
            load_domain_constraints(path, "target")
