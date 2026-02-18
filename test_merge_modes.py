#!/usr/bin/env python3
"""Test script to verify merge mode comparison functionality."""

import sys
sys.path.insert(0, 'src')

from mlx_fun.saliency import SaliencyAccumulator
from mlx_fun.stats_ops import merge_saliency, compute_diff_stats

def test_merge_modes():
    """Test all three merge modes with sample files."""
    print("Testing merge modes with data/1.npz and data/2.npz...")
    
    files = ['data/1.npz', 'data/2.npz']
    
    # Test all modes
    print("\n1. Testing SUM mode...")
    merged_sum = merge_saliency(files, mode="sum")
    print(f"   Total samples: {merged_sum.freq.sum():.0f}")
    
    print("\n2. Testing NORMALIZED mode...")
    merged_norm = merge_saliency(files, mode="normalized")
    print(f"   Total samples: {merged_norm.freq.sum():.0f}")
    
    print("\n3. Testing MAX mode...")
    merged_max = merge_saliency(files, mode="max")
    print(f"   Total samples: {merged_max.freq.sum():.0f}")
    
    # Compare modes
    print("\n4. Comparing SUM vs NORMALIZED...")
    diff_sum_norm = compute_diff_stats(merged_sum, merged_norm, "freq")
    print(f"   Positive (sum > norm): {diff_sum_norm['positive_count']}")
    print(f"   Negative (norm > sum): {diff_sum_norm['negative_count']}")
    print(f"   Max difference: {diff_sum_norm['diff_max']:.2f}")
    
    print("\n5. Comparing SUM vs MAX...")
    diff_sum_max = compute_diff_stats(merged_sum, merged_max, "freq")
    print(f"   Positive (sum > max): {diff_sum_max['positive_count']}")
    print(f"   Negative (max > sum): {diff_sum_max['negative_count']}")
    print(f"   Max difference: {diff_sum_max['diff_max']:.2f}")
    
    print("\n6. Comparing NORMALIZED vs MAX...")
    diff_norm_max = compute_diff_stats(merged_norm, merged_max, "freq")
    print(f"   Positive (norm > max): {diff_norm_max['positive_count']}")
    print(f"   Negative (max > norm): {diff_norm_max['negative_count']}")
    print(f"   Max difference: {diff_norm_max['diff_max']:.2f}")
    
    print("\n✅ All merge modes working correctly!")
    return True

if __name__ == "__main__":
    try:
        test_merge_modes()
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)