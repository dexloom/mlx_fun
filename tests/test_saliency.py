"""Tests for saliency accumulator."""

import tempfile
import numpy as np
import pytest

from mlx_fun.saliency import SaliencyAccumulator


def test_basic_accumulation():
    """Known-input formula verification."""
    acc = SaliencyAccumulator(num_layers=1, num_experts=3)

    # 2 tokens, top_k=1
    expert_indices = np.array([[0], [1]])
    router_weights = np.array([[0.8], [0.6]])
    activation_norms = np.array([[2.0], [3.0]])

    acc.update(0, expert_indices, router_weights, activation_norms)

    # Expert 0: reap_sum = 2.0 * 0.8 = 1.6, count = 1
    # Expert 1: reap_sum = 3.0 * 0.6 = 1.8, count = 1
    # Expert 2: never selected
    scores = acc.compute_scores("reap")
    np.testing.assert_allclose(scores[0, 0], 1.6)
    np.testing.assert_allclose(scores[0, 1], 1.8)
    np.testing.assert_allclose(scores[0, 2], 0.0)


def test_multiple_updates():
    """Accumulation across multiple batches."""
    acc = SaliencyAccumulator(num_layers=1, num_experts=2)

    # Batch 1
    acc.update(0, np.array([[0]]), np.array([[1.0]]), np.array([[4.0]]))
    # Batch 2
    acc.update(0, np.array([[0]]), np.array([[1.0]]), np.array([[6.0]]))

    scores = acc.compute_scores("reap")
    # Expert 0: (4.0*1.0 + 6.0*1.0) / 2 = 5.0
    np.testing.assert_allclose(scores[0, 0], 5.0)


def test_top_k_accumulation():
    """Multiple experts per token (top-k > 1)."""
    acc = SaliencyAccumulator(num_layers=1, num_experts=4)

    # 1 token selects experts 0 and 2
    expert_indices = np.array([[0, 2]])
    router_weights = np.array([[0.7, 0.3]])
    activation_norms = np.array([[5.0, 3.0]])

    acc.update(0, expert_indices, router_weights, activation_norms)

    scores = acc.compute_scores("reap")
    np.testing.assert_allclose(scores[0, 0], 5.0 * 0.7)  # 3.5
    np.testing.assert_allclose(scores[0, 2], 3.0 * 0.3)  # 0.9
    np.testing.assert_allclose(scores[0, 1], 0.0)
    np.testing.assert_allclose(scores[0, 3], 0.0)


def test_zero_division_guard():
    """Experts with zero count should get score 0, not NaN."""
    acc = SaliencyAccumulator(num_layers=1, num_experts=4)
    # Only expert 0 gets traffic
    acc.update(0, np.array([[0]]), np.array([[1.0]]), np.array([[2.0]]))

    scores = acc.compute_scores("reap")
    assert not np.any(np.isnan(scores))
    np.testing.assert_allclose(scores[0, 1], 0.0)
    np.testing.assert_allclose(scores[0, 2], 0.0)
    np.testing.assert_allclose(scores[0, 3], 0.0)


def test_ean_metric():
    acc = SaliencyAccumulator(num_layers=1, num_experts=2)
    acc.update(0, np.array([[0], [0]]), np.array([[0.5], [0.8]]), np.array([[3.0], [5.0]]))

    scores = acc.compute_scores("ean")
    # Expert 0: (3.0 + 5.0) / 2 = 4.0
    np.testing.assert_allclose(scores[0, 0], 4.0)


def test_freq_metric():
    acc = SaliencyAccumulator(num_layers=1, num_experts=3)
    acc.update(0, np.array([[0, 1], [0, 2]]), np.array([[0.5, 0.5], [0.5, 0.5]]),
               np.array([[1.0, 1.0], [1.0, 1.0]]))

    scores = acc.compute_scores("freq")
    np.testing.assert_allclose(scores[0, 0], 2.0)
    np.testing.assert_allclose(scores[0, 1], 1.0)
    np.testing.assert_allclose(scores[0, 2], 1.0)


def test_weighted_freq_metric():
    acc = SaliencyAccumulator(num_layers=1, num_experts=2)
    acc.update(0, np.array([[0, 1]]), np.array([[0.7, 0.3]]), np.array([[1.0, 1.0]]))

    scores = acc.compute_scores("weighted_freq")
    np.testing.assert_allclose(scores[0, 0], 0.7)
    np.testing.assert_allclose(scores[0, 1], 0.3)


def test_unknown_metric():
    acc = SaliencyAccumulator(num_layers=1, num_experts=2)
    with pytest.raises(ValueError, match="Unknown metric"):
        acc.compute_scores("bogus")


def test_save_load_roundtrip():
    acc = SaliencyAccumulator(num_layers=2, num_experts=4)
    acc.update(0, np.array([[0, 1]]), np.array([[0.5, 0.5]]), np.array([[2.0, 3.0]]))
    acc.update(1, np.array([[2, 3]]), np.array([[0.6, 0.4]]), np.array([[1.0, 4.0]]))

    with tempfile.NamedTemporaryFile(suffix=".npz") as f:
        acc.save(f.name)
        loaded = SaliencyAccumulator.load(f.name)

    np.testing.assert_array_equal(loaded.reap_sum, acc.reap_sum)
    np.testing.assert_array_equal(loaded.reap_count, acc.reap_count)
    np.testing.assert_array_equal(loaded.ean_sum, acc.ean_sum)
    np.testing.assert_array_equal(loaded.freq, acc.freq)
    np.testing.assert_array_equal(loaded.weighted_freq_sum, acc.weighted_freq_sum)
    assert loaded.num_layers == acc.num_layers
    assert loaded.num_experts == acc.num_experts


def test_multi_layer():
    """Verify layer isolation — updates to layer 0 don't affect layer 1."""
    acc = SaliencyAccumulator(num_layers=2, num_experts=3)
    acc.update(0, np.array([[0]]), np.array([[1.0]]), np.array([[5.0]]))
    acc.update(1, np.array([[1]]), np.array([[1.0]]), np.array([[3.0]]))

    scores = acc.compute_scores("reap")
    np.testing.assert_allclose(scores[0, 0], 5.0)
    np.testing.assert_allclose(scores[0, 1], 0.0)
    np.testing.assert_allclose(scores[1, 0], 0.0)
    np.testing.assert_allclose(scores[1, 1], 3.0)
