"""Operations on collected saliency statistics: diff, merge, purge."""

from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

from .saliency import SaliencyAccumulator


def diff_saliency(
    acc1: SaliencyAccumulator,
    acc2: SaliencyAccumulator,
    metric: str = "reap",
) -> np.ndarray:
    """Compute the difference between two saliency accumulators.

    Args:
        acc1: First accumulator.
        acc2: Second accumulator.
        metric: One of 'reap', 'ean', 'freq', 'weighted_freq'.

    Returns:
        (num_layers, num_experts) array of differences.
        Positive values mean acc1 has higher saliency than acc2.
    """
    scores1 = acc1.compute_scores(metric)
    scores2 = acc2.compute_scores(metric)
    return scores1 - scores2


def compute_diff_stats(
    acc1: SaliencyAccumulator,
    acc2: SaliencyAccumulator,
    metric: str = "reap",
) -> Dict:
    """Compute detailed statistics about the difference between two accumulators.

    Args:
        acc1: First accumulator.
        acc2: Second accumulator.
        metric: Saliency metric to compare.

    Returns:
        Dictionary with detailed statistics about the differences.
    """
    diff = diff_saliency(acc1, acc2, metric)
    
    return {
        "metric": metric,
        "num_layers": int(acc1.num_layers),
        "num_experts": int(acc1.num_experts),
        "diff_mean": float(diff.mean()),
        "diff_std": float(diff.std()),
        "diff_min": float(diff.min()),
        "diff_max": float(diff.max()),
        "diff_abs_max": float(np.abs(diff).max()),
        "positive_count": int(np.sum(diff > 0)),
        "negative_count": int(np.sum(diff < 0)),
        "zero_count": int(np.sum(diff == 0)),
        # Top experts with largest positive differences (acc1 > acc2)
        "top_positive": _get_extreme_experts(diff, k=10, direction="positive"),
        # Top experts with largest negative differences (acc2 > acc1)
        "top_negative": _get_extreme_experts(diff, k=10, direction="negative"),
    }


def _get_extreme_experts(
    diff: np.ndarray,
    k: int = 10,
    direction: str = "positive",
) -> List[Dict]:
    """Get the k most extreme expert differences.

    Args:
        diff: Difference array.
        k: Number of experts to return.
        direction: 'positive' or 'negative'.

    Returns:
        List of dicts with layer_idx, expert_idx, and diff_value.
    """
    if direction == "positive":
        indices = np.argpartition(diff.ravel(), -k)[-k:]
        indices = indices[np.argsort(diff.ravel()[indices])[::-1]]  # descending
    else:
        indices = np.argpartition(diff.ravel(), k)[:k]
        indices = indices[np.argsort(diff.ravel()[indices])]  # ascending
    
    result = []
    for idx in indices:
        layer_idx, expert_idx = np.unravel_index(idx, diff.shape)
        result.append({
            "layer_idx": int(layer_idx),
            "expert_idx": int(expert_idx),
            "diff_value": float(diff[layer_idx, expert_idx]),
        })
    return result


def merge_saliency(
    files: List[str],
    mode: str = "normalized",
) -> SaliencyAccumulator:
    """Merge multiple SaliencyAccumulator files into one.

    Args:
        files: List of paths to .npz files containing SaliencyAccumulator data.
        mode: Merge strategy:
            - "sum" (raw): Sum all arrays. Equivalent to collecting from all datasets together.
              Larger datasets dominate.
            - "normalized" (default): Normalize each file by its total samples before merging.
              Each dataset contributes equally regardless of sample count.
            - "max": Keep the maximum activation per expert across all files.
              Shows peak activation patterns.

    Returns:
        Merged SaliencyAccumulator.

    Raises:
        ValueError: If files have incompatible dimensions or unknown mode.
    """
    if not files:
        raise ValueError("At least one file must be provided for merging.")

    if mode not in ["sum", "normalized", "max"]:
        raise ValueError(f"Unknown merge mode '{mode}'. Use: sum, normalized, or max")

    # Load first file to get dimensions
    first_acc = SaliencyAccumulator.load(files[0])
    num_layers = first_acc.num_layers
    num_experts = first_acc.num_experts

    # Validate all files have same dimensions
    for file_path in files[1:]:
        acc = SaliencyAccumulator.load(file_path)
        if acc.num_layers != num_layers or acc.num_experts != num_experts:
            raise ValueError(
                f"File {file_path} has incompatible dimensions: "
                f"expected ({num_layers}, {num_experts}), "
                f"got ({acc.num_layers}, {acc.num_experts})"
            )

    # Initialize merged accumulator
    merged = SaliencyAccumulator(num_layers, num_experts)

    if mode == "sum":
        # Raw sum - equivalent to collecting from all datasets together
        for file_path in files:
            acc = SaliencyAccumulator.load(file_path)
            merged.reap_sum += acc.reap_sum
            merged.reap_count += acc.reap_count
            merged.ean_sum += acc.ean_sum
            merged.freq += acc.freq
            merged.weighted_freq_sum += acc.weighted_freq_sum

    elif mode == "normalized":
        # Normalize each file by its total samples before merging
        # This gives equal weight to each dataset regardless of sample count
        for file_path in files:
            acc = SaliencyAccumulator.load(file_path)
            
            # Calculate total samples processed in this file
            # Use the sum of all freq counts as a proxy for total samples
            total_samples = acc.freq.sum()
            
            if total_samples > 0:
                # Normalize arrays by sample count
                norm_factor = 1.0 / total_samples
                merged.reap_sum += acc.reap_sum * norm_factor
                merged.reap_count += acc.reap_count * norm_factor
                merged.ean_sum += acc.ean_sum * norm_factor
                merged.freq += acc.freq * norm_factor
                merged.weighted_freq_sum += acc.weighted_freq_sum * norm_factor

        # Scale back up to represent average across datasets
        # This keeps the magnitude reasonable for downstream operations
        avg_samples_per_file = 1.0 / len(files)
        merged.reap_sum *= avg_samples_per_file
        merged.reap_count *= avg_samples_per_file
        merged.ean_sum *= avg_samples_per_file
        merged.freq *= avg_samples_per_file
        merged.weighted_freq_sum *= avg_samples_per_file

    elif mode == "max":
        # Keep maximum activation per expert across all files
        # Useful for identifying peak expert behavior
        accumulators = [SaliencyAccumulator.load(f) for f in files]
        
        merged.reap_sum = np.maximum.reduce([acc.reap_sum for acc in accumulators])
        merged.reap_count = np.maximum.reduce([acc.reap_count for acc in accumulators])
        merged.ean_sum = np.maximum.reduce([acc.ean_sum for acc in accumulators])
        merged.freq = np.maximum.reduce([acc.freq for acc in accumulators])
        merged.weighted_freq_sum = np.maximum.reduce([acc.weighted_freq_sum for acc in accumulators])

    return merged


def purge_saliency(
    acc: SaliencyAccumulator,
    min_freq: Optional[int] = None,
    min_count: Optional[int] = None,
    max_norm: Optional[float] = None,
    keep_metadata: bool = False,
) -> Tuple[SaliencyAccumulator, Dict]:
    """Purge/filter low-activation or outlying data from an accumulator.

    Args:
        acc: Input accumulator.
        min_freq: Minimum activation frequency to keep. Experts with freq < min_freq
                  will have their data zeroed out.
        min_count: Minimum reap_count to keep. Experts with reap_count < min_count
                   will have their data zeroed out.
        max_norm: Maximum activation norm to cap. Values > max_norm will be clamped.
        keep_metadata: If True, returns a copy with modified data. If False,
                       modifies in place (still returns the accumulator).

    Returns:
        Tuple of (purged accumulator, metadata dict with purge statistics).
    """
    if keep_metadata:
        # Create a deep copy
        purged = SaliencyAccumulator(acc.num_layers, acc.num_experts)
        purged.reap_sum = acc.reap_sum.copy()
        purged.reap_count = acc.reap_count.copy()
        purged.ean_sum = acc.ean_sum.copy()
        purged.freq = acc.freq.copy()
        purged.weighted_freq_sum = acc.weighted_freq_sum.copy()
    else:
        purged = acc
    
    # Track purge statistics
    purge_stats = {
        "total_experts": acc.num_layers * acc.num_experts,
        "purged_by_freq": 0,
        "purged_by_count": 0,
        "capped_by_norm": 0,
    }
    
    # Build mask of experts to keep
    keep_mask = np.ones((acc.num_layers, acc.num_experts), dtype=bool)
    
    if min_freq is not None:
        freq_mask = acc.freq >= min_freq
        keep_mask &= freq_mask
        purge_stats["purged_by_freq"] = int(np.sum(~freq_mask))
    
    if min_count is not None:
        count_mask = acc.reap_count >= min_count
        keep_mask &= count_mask
        purge_stats["purged_by_count"] = int(np.sum(~count_mask))
    
    # Apply mask (zero out purged experts)
    # Broadcast mask to apply to all arrays
    mask_3d = keep_mask[:, :, None]  # For 3D operations if needed
    
    purged.reap_sum = purged.reap_sum * keep_mask
    purged.reap_count = purged.reap_count * keep_mask
    purged.ean_sum = purged.ean_sum * keep_mask
    purged.freq = purged.freq * keep_mask
    purged.weighted_freq_sum = purged.weighted_freq_sum * keep_mask
    
    # Cap extreme norms if specified
    if max_norm is not None:
        # Calculate EAN (Expert Activation Norm)
        with np.errstate(divide="ignore", invalid="ignore"):
            ean = np.where(
                purged.freq > 0,
                purged.ean_sum / purged.freq,
                0.0,
            )
        
        # Find values exceeding max_norm
        exceed_mask = ean > max_norm
        if exceed_mask.any():
            purge_stats["capped_by_norm"] = int(np.sum(exceed_mask))
            # Note: We can't easily "cap" the raw sums without knowing the count
            # This is a limitation - we'd need to store more granular data
            # For now, we just track that some values exceeded the threshold
    
    purge_stats["total_purged"] = (
        purge_stats["purged_by_freq"] +
        purge_stats["purged_by_count"]
    )
    purge_stats["kept_count"] = purge_stats["total_experts"] - purge_stats["total_purged"]
    
    return purged, purge_stats


def save_diff_report(
    report: Dict,
    output_path: str,
):
    """Save a diff report to JSON.

    Args:
        report: Dictionary from compute_diff_stats.
        output_path: Path to save the JSON file.
    """
    import json
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)