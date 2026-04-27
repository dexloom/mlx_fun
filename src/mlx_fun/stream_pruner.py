"""Streaming MoE prune — slices expert-axis tensors from safetensors shards
without loading the whole model into memory.

Use when the target model is too large for unified memory but you still want
the standard REAP saliency-based pruning. Peak RAM is bounded by the largest
single tensor (a few GB for big experts), not the full param tree.
"""
from __future__ import annotations

import json
import logging
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import numpy as np
from safetensors import safe_open

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per model_type: regexes that match expert-axis tensors. Group "L" captures
# the layer index. axis is always 0 for routed-expert tensors.
# ---------------------------------------------------------------------------
EXPERT_TENSOR_PATTERNS: Dict[str, List[re.Pattern]] = {
    # GLM-5 / DeepSeek V3.2 — routed experts in switch_mlp + gate weights
    "glm_moe_dsa": [
        re.compile(r"^model\.layers\.(?P<L>\d+)\.mlp\.gate\.weight$"),
        re.compile(r"^model\.layers\.(?P<L>\d+)\.mlp\.gate\.e_score_correction_bias$"),
        re.compile(r"^model\.layers\.(?P<L>\d+)\.mlp\.switch_mlp\.(?:gate|up|down)_proj\.(?:weight|scales|biases)$"),
    ],
    "deepseek_v32": [
        re.compile(r"^model\.layers\.(?P<L>\d+)\.mlp\.gate\.weight$"),
        re.compile(r"^model\.layers\.(?P<L>\d+)\.mlp\.gate\.e_score_correction_bias$"),
        re.compile(r"^model\.layers\.(?P<L>\d+)\.mlp\.switch_mlp\.(?:gate|up|down)_proj\.(?:weight|scales|biases)$"),
    ],
    "glm4_moe": [
        re.compile(r"^model\.layers\.(?P<L>\d+)\.mlp\.gate\.weight$"),
        re.compile(r"^model\.layers\.(?P<L>\d+)\.mlp\.gate\.e_score_correction_bias$"),
        re.compile(r"^model\.layers\.(?P<L>\d+)\.mlp\.switch_mlp\.(?:gate|up|down)_proj\.(?:weight|scales|biases)$"),
    ],
    "glm4_moe_lite": [
        re.compile(r"^model\.layers\.(?P<L>\d+)\.mlp\.gate\.weight$"),
        re.compile(r"^model\.layers\.(?P<L>\d+)\.mlp\.gate\.e_score_correction_bias$"),
        re.compile(r"^model\.layers\.(?P<L>\d+)\.mlp\.switch_mlp\.(?:gate|up|down)_proj\.(?:weight|scales|biases)$"),
    ],
    # Qwen3 / Qwen3-Next — block_sparse_moe with experts list
    "qwen3_moe": [
        re.compile(r"^model\.layers\.(?P<L>\d+)\.mlp\.gate\.weight$"),
        re.compile(r"^model\.layers\.(?P<L>\d+)\.mlp\.switch_mlp\.(?:gate|up|down)_proj\.(?:weight|scales|biases)$"),
    ],
    "qwen3_next": [
        re.compile(r"^model\.layers\.(?P<L>\d+)\.mlp\.gate\.weight$"),
        re.compile(r"^model\.layers\.(?P<L>\d+)\.mlp\.switch_mlp\.(?:gate|up|down)_proj\.(?:weight|scales|biases)$"),
    ],
    # MiniMax / MiniMax-M2
    "minimax": [
        re.compile(r"^model\.layers\.(?P<L>\d+)\.block_sparse_moe\.gate\.weight$"),
        re.compile(r"^model\.layers\.(?P<L>\d+)\.block_sparse_moe\.switch_mlp\.(?:gate|up|down)_proj\.(?:weight|scales|biases)$"),
    ],
    "minimax_m2": [
        re.compile(r"^model\.layers\.(?P<L>\d+)\.block_sparse_moe\.gate\.weight$"),
        re.compile(r"^model\.layers\.(?P<L>\d+)\.block_sparse_moe\.switch_mlp\.(?:gate|up|down)_proj\.(?:weight|scales|biases)$"),
    ],
}


def _moe_layer_indices(config: dict) -> List[int]:
    """Return absolute model layer indices that contain routed experts."""
    mt = config.get("model_type", "")
    n_layers = config["num_hidden_layers"]
    if mt in {"glm_moe_dsa", "deepseek_v32", "glm4_moe"}:
        first_k = config.get("first_k_dense_replace", 0)
        return list(range(first_k, n_layers))
    if mt == "glm4_moe_lite":
        first_k = config.get("first_k_dense_replace", 0)
        freq = config.get("moe_layer_freq", 1)
        return [i for i in range(first_k, n_layers) if (i - first_k) % freq == 0]
    if mt in {"qwen3_moe", "qwen3_next"}:
        step = config.get("decoder_sparse_step", 1)
        return [i for i in range(n_layers) if i % step == 0]
    if mt in {"minimax", "minimax_m2"}:
        return list(range(n_layers))
    raise ValueError(f"streaming prune: unknown model_type '{mt}'")


def _expert_count_key(model_type: str) -> str:
    if model_type in {"glm_moe_dsa", "deepseek_v32", "glm4_moe", "glm4_moe_lite"}:
        return "n_routed_experts"
    if model_type in {"qwen3_moe", "qwen3_next"}:
        return "num_experts"
    if model_type in {"minimax", "minimax_m2"}:
        return "num_local_experts"
    raise ValueError(f"streaming prune: unknown model_type '{model_type}'")


def _classify_tensor(
    name: str, patterns: List[re.Pattern]
) -> Optional[int]:
    """Return absolute model layer index if name is an expert-axis tensor, else None."""
    for p in patterns:
        m = p.match(name)
        if m:
            return int(m.group("L"))
    return None


def _build_layer_to_tensors(
    weight_map: Dict[str, str], patterns: List[re.Pattern]
) -> Dict[int, List[str]]:
    """Map model layer idx -> list of expert-axis tensor names in that layer."""
    out: Dict[int, List[str]] = defaultdict(list)
    for name in weight_map:
        L = _classify_tensor(name, patterns)
        if L is not None:
            out[L].append(name)
    return out


def stream_prune(
    src_path: str,
    dst_path: str,
    keep_map_model: Dict[int, np.ndarray],
    log_progress: bool = True,
) -> dict:
    """Stream-prune a quantized MoE checkpoint to a new directory.

    Args:
        src_path: Source model directory (HF/MLX format with sharded safetensors).
        dst_path: Output directory. Must not exist.
        keep_map_model: Dict mapping ABSOLUTE model layer index ->
            sorted numpy array of expert indices to keep (in original numbering).
        log_progress: Emit per-shard progress logs.

    Returns:
        Updated config dict (also written to dst/config.json).
    """
    src = Path(src_path)
    dst = Path(dst_path)
    if dst.exists():
        raise FileExistsError(f"output directory already exists: {dst}")
    dst.mkdir(parents=True)

    with open(src / "config.json") as f:
        config = json.load(f)
    model_type = config.get("model_type", "")
    if model_type not in EXPERT_TENSOR_PATTERNS:
        raise ValueError(
            f"stream_prune: no tensor patterns registered for model_type '{model_type}'. "
            f"Add an entry in EXPERT_TENSOR_PATTERNS."
        )
    patterns = EXPERT_TENSOR_PATTERNS[model_type]

    # Validate: all kept-counts must agree (otherwise n_routed_experts is ambiguous)
    keep_counts = {len(v) for v in keep_map_model.values()}
    if len(keep_counts) != 1:
        raise ValueError(
            f"stream_prune: per-layer keep counts differ {sorted(keep_counts)}; "
            f"streaming output requires uniform expert count per layer."
        )
    new_n_experts = keep_counts.pop()

    # Read source index
    index_path = src / "model.safetensors.index.json"
    if not index_path.exists():
        # single-shard model
        single = sorted(src.glob("*.safetensors"))
        if len(single) != 1:
            raise FileNotFoundError(f"no model.safetensors.index.json and no single .safetensors in {src}")
        weight_map = {}
        with safe_open(single[0], framework="numpy") as f:
            for k in f.keys():
                weight_map[k] = single[0].name
        index_meta = {"metadata": {}, "weight_map": weight_map}
    else:
        with open(index_path) as f:
            index_meta = json.load(f)

    weight_map = index_meta["weight_map"]
    expert_tensors_by_layer = _build_layer_to_tensors(weight_map, patterns)

    # Sanity check expert layer coverage
    expected_layers = set(_moe_layer_indices(config))
    missing = expected_layers - set(expert_tensors_by_layer.keys())
    if missing:
        logger.warning(
            f"stream_prune: {len(missing)} expected MoE layer(s) have no expert tensors "
            f"matched: {sorted(missing)[:5]}..."
        )

    # Group source tensors by source shard so we read each shard exactly once
    shard_to_tensors: Dict[str, List[str]] = defaultdict(list)
    for tname, shard in weight_map.items():
        shard_to_tensors[shard].append(tname)

    new_weight_map: Dict[str, str] = {}
    src_shards = sorted(shard_to_tensors.keys())

    # Pre-build mx.array indexers for each layer's keep set (bounded memory).
    keep_mx: Dict[int, mx.array] = {
        L: mx.array(np.asarray(idx, dtype=np.int32)) for L, idx in keep_map_model.items()
    }

    n_shards = len(src_shards)
    sliced_count = 0
    copied_count = 0
    for i, shard in enumerate(src_shards, 1):
        # mx.load handles bf16/uint8/uint32 etc natively, unlike numpy.
        # One shard at a time keeps peak RAM bounded (~few GB per shard).
        loaded = mx.load(str(src / shard))
        out_buf: Dict[str, mx.array] = {}
        for tname in shard_to_tensors[shard]:
            arr = loaded[tname]
            L = _classify_tensor(tname, patterns)
            if L is not None and L in keep_mx:
                sliced = mx.take(arr, keep_mx[L], axis=0)
                mx.eval(sliced)
                out_buf[tname] = sliced
                sliced_count += 1
            else:
                out_buf[tname] = arr
                copied_count += 1

        out_shard = dst / shard
        mx.save_safetensors(str(out_shard), out_buf)
        new_weight_map.update({k: shard for k in out_buf})
        # Free shard memory before next iteration
        del loaded, out_buf
        if log_progress:
            logger.info(f"  [{i}/{n_shards}] {shard}")

    # Write updated index
    if index_path.exists():
        with open(dst / "model.safetensors.index.json", "w") as f:
            json.dump(
                {"metadata": index_meta.get("metadata", {}), "weight_map": new_weight_map},
                f, indent=2,
            )

    # Update + write config.json
    new_config = dict(config)
    key = _expert_count_key(model_type)
    new_config[key] = int(new_n_experts)
    if "text_config" in new_config and key in new_config["text_config"]:
        nested = dict(new_config["text_config"])
        nested[key] = int(new_n_experts)
        new_config["text_config"] = nested
    with open(dst / "config.json", "w") as f:
        json.dump(new_config, f, indent=2)

    # Copy auxiliary files (tokenizer, generation config, chat template, etc.)
    for aux in src.iterdir():
        if aux.is_file() and aux.suffix not in {".safetensors"} and aux.name not in {
            "config.json", "model.safetensors.index.json"
        }:
            shutil.copy2(aux, dst / aux.name)

    logger.info(
        f"stream_prune: done — sliced={sliced_count}, copied={copied_count}, "
        f"experts {config[key]}->{new_n_experts}"
    )
    return new_config
