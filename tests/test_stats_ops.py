"""Unit tests for stats_ops module - rank-based merge functionality."""

import pytest
import numpy as np
import tempfile
import os

from mlx_fun.saliency import SaliencyAccumulator
from mlx_fun.stats_ops import (
    merge_saliency,
    _compute_ranked_scores,
    diff_saliency,
    compute_diff_stats,
    purge_saliency,
)


class TestComputeRankedScores:
    """Tests for the _compute_ranked_scores helper function."""

    def test_basic_ranking(self):
        """Test that ranking works correctly for simple cases."""
        acc = SaliencyAccumulator(num_layers=2, num_experts=4)
        
        # Layer 0: experts have freq [10, 20, 5, 15]
        acc.freq[0] = [10, 20, 5, 15]
        
        # Layer 1: experts have freq [1, 2, 3, 4]
        acc.freq[1] = [1, 2, 3, 4]
        
        ranks = _compute_ranked_scores(acc, "freq")
        
        # Layer 0: sorted by freq descending: 20, 15, 10, 5
        # Expert 0 (10): rank 3
        # Expert 1 (20): rank 1
        # Expert 2 (5): rank 4
        # Expert 3 (15): rank 2
        assert list(ranks[0]) == [3, 1, 4, 2]
        
        # Layer 1: sorted by freq descending: 4, 3, 2, 1
        # Expert 0 (1): rank 4
        # Expert 1 (2): rank 3
        # Expert 2 (3): rank 2
        # Expert 3 (4): rank 1
        assert list(ranks[1]) == [4, 3, 2, 1]

    def test_rank_sum_per_layer(self):
        """Test that rank sum per layer equals n*(n+1)/2."""
        n_experts = 10
        acc = SaliencyAccumulator(num_layers=3, num_experts=n_experts)
        
        # Set random freq values
        np.random.seed(42)
        acc.freq = np.random.rand(3, n_experts) * 100
        
        ranks = _compute_ranked_scores(acc, "freq")
        
        expected_sum = n_experts * (n_experts + 1) / 2
        for layer_idx in range(3):
            assert abs(ranks[layer_idx].sum() - expected_sum) < 0.01

    def test_ranking_with_ties(self):
        """Test ranking behavior with tied scores."""
        acc = SaliencyAccumulator(num_layers=1, num_experts=4)
        acc.freq[0] = [10, 10, 5, 20]  # Two experts with same score
        
        ranks = _compute_ranked_scores(acc, "freq")
        
        # Expert 3 (20): rank 1
        # Experts 0, 1 (10): ranks 2 and 3 (order depends on argsort stability)
        # Expert 2 (5): rank 4
        assert ranks[0, 3] == 1  # Highest score gets rank 1
        assert ranks[0, 2] == 4  # Lowest score gets rank 4


class TestMergeSaliency:
    """Tests for the merge_saliency function."""

    @pytest.fixture
    def temp_files(self):
        """Create temporary saliency files for testing."""
        acc1 = SaliencyAccumulator(num_layers=2, num_experts=4)
        acc2 = SaliencyAccumulator(num_layers=2, num_experts=4)
        
        # Set different freq patterns
        acc1.freq[0] = [10, 20, 5, 15]
        acc1.freq[1] = [1, 2, 3, 4]
        
        acc2.freq[0] = [15, 10, 20, 5]
        acc2.freq[1] = [4, 3, 2, 1]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path1 = os.path.join(tmpdir, "file1.npz")
            path2 = os.path.join(tmpdir, "file2.npz")
            acc1.save(path1)
            acc2.save(path2)
            yield [path1, path2]

    def test_merge_two_files(self, temp_files):
        """Test merging two files produces correct rank sums."""
        merged = merge_saliency(temp_files, metric="freq")
        
        # Check dimensions
        assert merged.num_layers == 2
        assert merged.num_experts == 4
        
        # Check that rank sums are in valid range
        # Min possible: 2 (rank 1 from each file)
        # Max possible: 8 (rank 4 from each file)
        assert merged.freq.min() >= 2
        assert merged.freq.max() <= 8

    def test_merge_single_file(self, temp_files):
        """Test that merging a single file produces ranks 1 to N."""
        merged = merge_saliency([temp_files[0]], metric="freq")
        
        n = merged.num_experts
        expected_sum = n * (n + 1) / 2
        
        for layer_idx in range(merged.num_layers):
            layer_sum = merged.freq[layer_idx].sum()
            assert abs(layer_sum - expected_sum) < 0.01

    def test_merge_consistency(self, temp_files):
        """Test that same inputs produce same outputs."""
        merged1 = merge_saliency(temp_files, metric="freq")
        merged2 = merge_saliency(temp_files, metric="freq")
        
        assert np.array_equal(merged1.freq, merged2.freq)

    def test_merge_different_metrics(self, temp_files):
        """Test merging with different metrics."""
        for metric in ["reap", "ean", "freq", "weighted_freq"]:
            merged = merge_saliency(temp_files, metric=metric)
            assert merged.num_layers == 2
            assert merged.num_experts == 4

    def test_merge_invalid_metric(self, temp_files):
        """Test that invalid metric raises ValueError."""
        with pytest.raises(ValueError, match="Unknown metric"):
            merge_saliency(temp_files, metric="invalid")

    def test_merge_empty_files(self):
        """Test that empty file list raises ValueError."""
        with pytest.raises(ValueError, match="At least one file"):
            merge_saliency([], metric="freq")

    def test_merge_incompatible_dimensions(self):
        """Test that files with different dimensions raise ValueError."""
        acc1 = SaliencyAccumulator(num_layers=2, num_experts=4)
        acc2 = SaliencyAccumulator(num_layers=3, num_experts=4)  # Different layers
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path1 = os.path.join(tmpdir, "file1.npz")
            path2 = os.path.join(tmpdir, "file2.npz")
            acc1.save(path1)
            acc2.save(path2)
            
            with pytest.raises(ValueError, match="incompatible dimensions"):
                merge_saliency([path1, path2], metric="freq")


class TestDiffSaliency:
    """Tests for diff_saliency function."""

    def test_diff_basic(self):
        """Test basic diff computation."""
        acc1 = SaliencyAccumulator(num_layers=2, num_experts=4)
        acc2 = SaliencyAccumulator(num_layers=2, num_experts=4)
        
        acc1.freq[0] = [10, 20, 30, 40]
        acc2.freq[0] = [5, 25, 30, 35]
        
        diff = diff_saliency(acc1, acc2, "freq")
        
        # diff = acc1 - acc2
        assert diff[0, 0] == 5   # 10 - 5
        assert diff[0, 1] == -5  # 20 - 25
        assert diff[0, 2] == 0   # 30 - 30
        assert diff[0, 3] == 5   # 40 - 35


class TestComputeDiffStats:
    """Tests for compute_diff_stats function."""

    def test_diff_stats(self):
        """Test that diff stats are computed correctly."""
        # Use larger dimensions to avoid argpartition issues with top-k
        acc1 = SaliencyAccumulator(num_layers=4, num_experts=8)
        acc2 = SaliencyAccumulator(num_layers=4, num_experts=8)
        
        # Set different freq patterns
        acc1.freq[0] = [10, 20, 30, 40, 50, 60, 70, 80]
        acc2.freq[0] = [5, 25, 30, 35, 45, 65, 75, 85]
        
        stats = compute_diff_stats(acc1, acc2, "freq")
        
        assert stats["metric"] == "freq"
        assert stats["num_layers"] == 4
        assert stats["num_experts"] == 8
        # Check that positive/negative/zero counts are reasonable
        assert stats["positive_count"] >= 0
        assert stats["negative_count"] >= 0
        assert stats["zero_count"] >= 0
        assert stats["positive_count"] + stats["negative_count"] + stats["zero_count"] == 4 * 8


class TestPurgeSaliency:
    """Tests for purge_saliency function."""

    def test_purge_by_freq(self):
        """Test purging experts by minimum frequency."""
        acc = SaliencyAccumulator(num_layers=1, num_experts=4)
        acc.freq[0] = [10, 20, 5, 15]
        acc.reap_sum[0] = [100, 200, 50, 150]
        
        purged, stats = purge_saliency(acc, min_freq=10, keep_metadata=True)
        
        # Expert 2 (freq=5) should be purged
        assert purged.freq[0, 2] == 0
        assert purged.reap_sum[0, 2] == 0
        
        # Other experts should remain
        assert purged.freq[0, 0] == 10
        assert purged.freq[0, 1] == 20
        assert purged.freq[0, 3] == 15
        
        assert stats["purged_by_freq"] == 1
