"""Save pruned, merged, and abliterated models with metadata."""

import json
from pathlib import Path
from typing import Dict, List

import numpy as np


def save_pruned_model(
    model,
    tokenizer,
    config: dict,
    output_path: str,
    keep_map: Dict[int, np.ndarray],
    original_num_experts: int,
    metric: str,
):
    """Save a pruned model with all necessary files.

    Uses mlx_lm utilities for weights/config, adds REAP metadata.

    Args:
        model: The pruned MLX model.
        tokenizer: HuggingFace tokenizer.
        config: Updated config dict.
        output_path: Directory to save to.
        keep_map: Layer -> kept expert indices mapping.
        original_num_experts: Expert count before pruning.
        metric: Saliency metric used for pruning.
    """
    from mlx_lm.utils import save_model, save_config

    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)

    # Save weights
    save_model(out, model)

    # Save config
    save_config(config, out / "config.json")

    # Save tokenizer
    tokenizer.save_pretrained(str(out))

    # Save REAP metadata
    metadata = {
        "original_num_experts": original_num_experts,
        "pruned_num_experts": int(next(iter(keep_map.values())).shape[0]),
        "metric": metric,
        "keep_map": {
            str(k): v.tolist() for k, v in keep_map.items()
        },
    }
    with open(out / "reap_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def save_merged_model(
    model,
    tokenizer,
    config: dict,
    output_path: str,
    centroid_map: Dict[int, "np.ndarray"],
    group_map: Dict[int, Dict[int, list]],
    original_num_experts: int,
    metric: str,
):
    """Save a REAM-merged model with all necessary files.

    Args:
        model: The merged MLX model.
        tokenizer: HuggingFace tokenizer.
        config: Updated config dict.
        output_path: Directory to save to.
        centroid_map: Layer -> centroid expert indices mapping.
        group_map: Layer -> {centroid: [member_indices]} mapping.
        original_num_experts: Expert count before merging.
        metric: Saliency metric used for merging.
    """
    from mlx_lm.utils import save_model, save_config

    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)

    save_model(out, model)
    save_config(config, out / "config.json")
    tokenizer.save_pretrained(str(out))

    # Save REAM metadata
    first_centroids = next(iter(centroid_map.values()))
    metadata = {
        "method": "ream",
        "original_num_experts": original_num_experts,
        "merged_num_experts": int(len(first_centroids)),
        "metric": metric,
        "centroid_map": {
            str(k): v.tolist() for k, v in centroid_map.items()
        },
        "group_map": {
            str(layer): {str(c): members for c, members in groups.items()}
            for layer, groups in group_map.items()
        },
    }
    with open(out / "ream_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def save_abliterated_model(
    model,
    tokenizer,
    config: dict,
    output_path: str,
    refusal_directions: Dict[int, np.ndarray],
    target: str,
    abliterated_layers: List[int],
):
    """Save an abliterated model with metadata.

    Args:
        model: The abliterated MLX model.
        tokenizer: HuggingFace tokenizer.
        config: Model config dict (unchanged).
        output_path: Directory to save to.
        refusal_directions: Per-layer refusal directions used.
        target: What was targeted ("all", "safety-experts", "dense-only").
        abliterated_layers: Which layers were modified.
    """
    from mlx_lm.utils import save_model, save_config

    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)

    save_model(out, model)
    save_config(config, out / "config.json")
    tokenizer.save_pretrained(str(out))

    metadata = {
        "method": "abliteration",
        "target": target,
        "abliterated_layers": abliterated_layers,
        "refusal_direction_norms": {
            str(k): float(np.linalg.norm(v))
            for k, v in refusal_directions.items()
        },
    }
    with open(out / "abliteration_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def save_amplified_model(
    model,
    tokenizer,
    config: dict,
    output_path: str,
    domain_name: str,
    scale: float,
    threshold: float,
    biases: Dict[int, np.ndarray],
):
    """Save an amplified model with metadata.

    Args:
        model: The amplified MLX model.
        tokenizer: HuggingFace tokenizer.
        config: Model config dict (unchanged).
        output_path: Directory to save to.
        domain_name: Name of the domain that was amplified.
        scale: Amplification scale used.
        threshold: Composite score threshold used.
        biases: Per-layer bias arrays that were applied.
    """
    from mlx_lm.utils import save_model, save_config

    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)

    save_model(out, model)
    save_config(config, out / "config.json")
    tokenizer.save_pretrained(str(out))

    metadata = {
        "method": "amplification",
        "domain_name": domain_name,
        "scale": scale,
        "threshold": threshold,
        "amplified_layers": sorted(biases.keys()),
        "per_layer_bias": {
            str(k): v.tolist() for k, v in biases.items()
        },
    }
    with open(out / "amplification_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
