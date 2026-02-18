#!/usr/bin/env python3
"""Test script to verify rank-based merge functionality."""

import sys
sys.path.insert(0, 'src')

import numpy as np
from mlx_fun.saliency import SaliencyAccumulator
from mlx_fun.stats_ops import merge_saliency, _compute_ranked_scores


def test_ranked_scores():
    """Test the _compute_ranked_scores function."""
    print("Testing _compute_ranked_scores...")
    
    # Create a simple accumulator with known values
    acc = SaliencyAccumulator(num_layers=2, num_experts=4)
    
    # Layer 0: experts have freq [10, 20, 5, 15]
    acc.freq[0] = [10, 20, 5, 15]
    
    # Layer 1: experts have freq [1, 2, 3, 4]
    acc.freq[1] = [1, 2, 3, 4]
    
    # Compute ranks using freq metric
    ranks = _compute_ranked_scores(acc, "freq")
    
    # Layer 0: sorted by freq descending: 20, 15, 10, 5
    # Expert 0 (10): rank 3
    # Expert 1 (20): rank 1
    # Expert 2 (5): rank 4
    # Expert 3 (15): rank 2
    expected_layer0 = [3, 1, 4, 2]
    
    # Layer 1: sorted by freq descending: 4, 3, 2, 1
    # Expert 0 (1): rank 4
    # Expert 1 (2): rank 3
    # Expert 2 (3): rank 2
    # Expert 3 (4): rank 1
    expected_layer1 = [4, 3, 2, 1]
    
    assert list(ranks[0]) == expected_layer0, f"Layer 0 ranks mismatch: {ranks[0]} vs {expected_layer0}"
    assert list(ranks[1]) == expected_layer1, f"Layer 1 ranks mismatch: {ranks[1]} vs {expected_layer1}"
    
    print("  ✓ Per-layer ranking works correctly")
    print(f"    Layer 0: freq {[10, 20, 5, 15]} -> ranks {list(ranks[0])}")
    print(f"    Layer 1: freq {[1, 2, 3, 4]} -> ranks {list(ranks[1])}")


def test_merge_saliency():
    """Test rank-based merge with sample files."""
    print("\nTesting rank-based merge...")
    
    files = ['data/1.npz', 'data/2.npz']
    
    # Test with different metrics
    for metric in ["reap", "ean", "freq", "weighted_freq"]:
        print(f"\n  Metric: {metric}")
        
        merged = merge_saliency(files, metric=metric)
        
        print(f"    Layers: {merged.num_layers}")
        print(f"    Experts: {merged.num_experts}")
        print(f"    Rank sum range: [{merged.freq.min():.0f}, {merged.freq.max():.0f}]")
        
        # Verify that lower rank sum indicates higher importance
        # The minimum rank sum should be at least 2 (rank 1 from each file)
        assert merged.freq.min() >= 2, f"Minimum rank sum should be >= 2 (got {merged.freq.min()})"
        
        # The maximum rank sum should be at most 2 * num_experts
        max_expected = 2 * merged.num_experts
        assert merged.freq.max() <= max_expected, f"Maximum rank sum should be <= {max_expected}"
    
    print("\n  ✓ Rank-based merge works correctly for all metrics")


def test_merge_single_file():
    """Test that merging a single file still works (returns ranks for that file)."""
    print("\nTesting single file merge...")
    
    merged = merge_saliency(['data/1.npz'], metric="freq")
    
    # With single file, ranks should be 1 to num_experts for each layer
    # Sum should be n*(n+1)/2 for each layer
    n = merged.num_experts
    expected_sum = n * (n + 1) / 2
    
    for layer_idx in range(merged.num_layers):
        layer_sum = merged.freq[layer_idx].sum()
        assert abs(layer_sum - expected_sum) < 0.01, f"Layer {layer_idx} sum mismatch: {layer_sum} vs {expected_sum}"
    
    print(f"  ✓ Single file merge produces valid ranks (sum per layer = {expected_sum})")


def test_merge_consistency():
    """Test that the same files produce consistent results."""
    print("\nTesting merge consistency...")
    
    files = ['data/1.npz', 'data/2.npz']
    
    merged1 = merge_saliency(files, metric="freq")
    merged2 = merge_saliency(files, metric="freq")
    
    assert np.array_equal(merged1.freq, merged2.freq), "Same files should produce same results"
    
    print("  ✓ Merge produces consistent results")


if __name__ == "__main__":
    try:
        test_ranked_scores()
        test_merge_saliency()
        test_merge_single_file()
        test_merge_consistency()
        
        print("\n" + "="*50)
        print("✅ All rank-based merge tests passed!")
        print("="*50)
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
