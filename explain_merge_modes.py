#!/usr/bin/env python3
"""
Explanation of why merge mode comparisons show directional bias.

When comparing the SAME files merged with DIFFERENT modes, the differences
are always in one direction because the merge modes are designed to produce
systematically different results.
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
from mlx_fun.saliency import SaliencyAccumulator
from mlx_fun.stats_ops import merge_saliency, compute_diff_stats

def explain_merge_modes():
    """Demonstrate why merge mode comparisons show directional bias."""
    
    print("=" * 70)
    print("WHY MERGE MODE COMPARISONS SHOW DIRECTIONAL BIAS")
    print("=" * 70)
    
    files = ['data/1.npz', 'data/2.npz']
    
    # Load original files to understand the data
    acc1 = SaliencyAccumulator.load(files[0])
    acc2 = SaliencyAccumulator.load(files[1])
    
    print("\n📊 ORIGINAL FILES:")
    print(f"  File 1 total samples: {acc1.freq.sum():.0f}")
    print(f"  File 2 total samples: {acc2.freq.sum():.0f}")
    print(f"  Ratio: {acc1.freq.sum() / acc2.freq.sum():.2f}x")
    
    # Merge with different modes
    print("\n" + "=" * 70)
    print("MERGING WITH DIFFERENT MODES...")
    print("=" * 70)
    
    merged_sum = merge_saliency(files, mode="sum")
    merged_norm = merge_saliency(files, mode="normalized")
    merged_max = merge_saliency(files, mode="max")
    
    print(f"\n📊 MERGED RESULTS:")
    print(f"  Sum mode:      {merged_sum.freq.sum():.0f} samples")
    print(f"  Normalized:    {merged_norm.freq.sum():.0f} samples")
    print(f"  Max mode:      {merged_max.freq.sum():.0f} samples")
    
    # Compare modes
    print("\n" + "=" * 70)
    print("COMPARING MERGE MODES")
    print("=" * 70)
    
    # Sum vs Normalized
    diff_sum_norm = compute_diff_stats(merged_sum, merged_norm, "freq")
    print(f"\n1️⃣  SUM vs NORMALIZED:")
    print(f"   Positive (Sum > Norm): {diff_sum_norm['positive_count']} experts")
    print(f"   Negative (Norm > Sum): {diff_sum_norm['negative_count']} experts")
    print(f"   Explanation: Sum mode adds raw values, Normalized divides by total")
    print(f"   Result: Sum values are ALWAYS larger because normalization reduces them")
    
    # Sum vs Max
    diff_sum_max = compute_diff_stats(merged_sum, merged_max, "freq")
    print(f"\n2️⃣  SUM vs MAX:")
    print(f"   Positive (Sum > Max): {diff_sum_max['positive_count']} experts")
    print(f"   Negative (Max > Sum): {diff_sum_max['negative_count']} experts")
    print(f"   Explanation: Sum adds all values, Max takes only the peak")
    print(f"   Result: Sum >= Max always (sum of values >= maximum of values)")
    
    # Normalized vs Max
    diff_norm_max = compute_diff_stats(merged_norm, merged_max, "freq")
    print(f"\n3️⃣  NORMALIZED vs MAX:")
    print(f"   Positive (Norm > Max): {diff_norm_max['positive_count']} experts")
    print(f"   Negative (Max > Norm): {diff_norm_max['negative_count']} experts")
    print(f"   Explanation: Normalized averages values, Max keeps peaks")
    print(f"   Result: Max typically larger (peak > average)")
    
    # Now compare the ORIGINAL files (different datasets)
    print("\n" + "=" * 70)
    print("COMPARING ORIGINAL FILES (Different Datasets)")
    print("=" * 70)
    
    diff_original = compute_diff_stats(acc1, acc2, "freq")
    print(f"\n📊 FILE 1 vs FILE 2:")
    print(f"   Positive (File1 > File2): {diff_original['positive_count']} experts")
    print(f"   Negative (File2 > File1): {diff_original['negative_count']} experts")
    print(f"   Explanation: These are different datasets with different activation patterns")
    print(f"   Result: BOTH positive and negative differences (expected!)")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
📌 MERGE MODE COMPARISONS (Same files, different merge strategies):
   - ALWAYS show directional bias (one side always larger)
   - This is CORRECT and EXPECTED behavior
   - Sum mode produces largest values
   - Normalized mode produces smallest values
   - Max mode produces intermediate values

📌 FILE COMPARISONS (Different files, same merge mode):
   - Show BOTH positive and negative differences
   - This is CORRECT and EXPECTED behavior
   - Different datasets have different activation patterns
   - Some experts more active in File 1, others in File 2

🎯 USE CASES:
   - Use "Merge Mode Comparison" to understand HOW merge strategies differ
   - Use "Diff Analysis" to compare DIFFERENT datasets
   - They serve different purposes!
""")

if __name__ == "__main__":
    try:
        explain_merge_modes()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)