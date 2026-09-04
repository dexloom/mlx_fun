"""CLI for MLX-FUN: collect saliency, prune experts, smoke-test."""

import os
import shutil
import click
import numpy as np


@click.group()
def main():
    """MLX-FUN: Routing-based Expert Activation Pruning for MoE models."""
    pass


@main.command()
@click.option("--model", required=True, help="Model path or HuggingFace repo ID.")
@click.option("--dataset", required=True, help="Path to JSONL file or directory.")
@click.option("--output", required=True, help="Output path for saliency .npz file.")
@click.option("--max-samples", default=128, help="Maximum calibration samples.")
@click.option("--max-tokens", default=2048, help="Maximum tokens per sample.")
@click.option("--text-key", default="content", help="JSON key for text in JSONL.")
@click.option("--seed", default=None, type=int, help="Random seed for sample selection.")
def collect(model, dataset, output, max_samples, max_tokens, text_key, seed):
    """Collect saliency statistics via calibration."""
    import random
    import mlx.core as mx
    from .loader import load_model, text_forward
    from tqdm import tqdm

    if seed is not None:
        random.seed(seed)

    from .adapters import get_adapter
    from .data import load_dataset
    from .observer import install_hooks, collect_captures, remove_hooks
    from .saliency import SaliencyAccumulator

    # Expand user path and validate if it's a local path
    expanded_model = os.path.expanduser(model)
    if os.path.exists(expanded_model):
        model = expanded_model
        click.echo(f"Loading model from local path: {model}")
    else:
        click.echo(f"Loading model: {model}")
    
    try:
        mlx_model, tokenizer, config = load_model(model)
    except Exception as e:
        if "HFValidationError" in str(type(e).__name__) or "Repo id must be in the form" in str(e):
            click.echo(f"\nError: Model path '{model}' could not be loaded as a local file or HuggingFace repo.", err=True)
            click.echo(f"Please verify:", err=True)
            click.echo(f"  1. The path exists and contains model files (config.json, tokenizer files, etc.)", err=True)
            click.echo(f"  2. If using a HuggingFace repo, ensure the repo ID is correct (format: 'username/repo-name')", err=True)
            raise
        raise

    adapter = get_adapter(mlx_model, config)
    moe_indices = adapter.moe_layer_indices()
    n_experts = adapter.num_routed_experts()

    click.echo(f"Model type: {config.get('model_type')}")
    click.echo(f"MoE layers: {len(moe_indices)}, Experts per layer: {n_experts}")

    click.echo(f"Loading dataset: {dataset}")
    samples = load_dataset(dataset, tokenizer, max_tokens, max_samples, text_key)
    click.echo(f"Loaded {len(samples)} samples")

    # Install hooks
    moe_blocks = [adapter.get_moe_block(i) for i in moe_indices]
    model_type = config.get("model_type", "")
    install_hooks(moe_blocks, model_type)

    acc = SaliencyAccumulator(num_layers=len(moe_indices), num_experts=n_experts)

    click.echo("Running calibration...")
    # Vision checkpoints route token-only passes through the language stack.
    forward = text_forward(mlx_model, config)
    for sample in tqdm(samples, desc="Calibrating"):
        # Run forward pass: (1, seq_len)
        tokens = sample.reshape(1, -1)
        forward(tokens)
        mx.eval(mlx_model.parameters())

        # Collect captures and accumulate
        captures = collect_captures(moe_blocks)
        for block_idx, block_captures in enumerate(captures):
            for inds, scores, norms in block_captures:
                # Flatten batch and seq dims
                flat_inds = inds.reshape(-1, inds.shape[-1])
                flat_scores = scores.reshape(-1, scores.shape[-1])
                flat_norms = norms.reshape(-1, norms.shape[-1])
                acc.update(block_idx, flat_inds, flat_scores, flat_norms)

    remove_hooks(moe_blocks)

    acc.save(output)
    click.echo(f"Saliency stats saved to: {output}")


@main.command()
@click.option("--model", required=True, help="Model path or HuggingFace repo ID.")
@click.option("--saliency", default=None, help="Path to saliency .npz file. Required if --expert-list is not provided.")
@click.option("--expert-list", default=None,
              help="Path to expert list .json or .csv from frontend export. "
                   "Bypasses --n-prune calculation. Use instead of --saliency.")
@click.option("--output", required=True, help="Output directory for pruned model.")
@click.option("--n-prune", default=None, type=int,
              help="Number of experts to prune per layer (or total if --model-wide). "
                   "Required if --expert-list is not provided.")
@click.option("--metric", default="reap", type=click.Choice(["reap", "ean", "freq", "weighted_freq"]))
@click.option("--strategy", default="bottom", type=click.Choice(["bottom", "strided"]),
              help="Pruning strategy: 'bottom' removes lowest-scoring, 'strided' distributes pruning evenly.")
@click.option("--model-wide", is_flag=True, default=False,
              help="Select N experts globally across all layers instead of per-layer.")
@click.option("--min-experts-per-layer", default=1, type=int,
              help="Minimum experts to keep per layer when using --model-wide (default: 1).")
@click.option("--safety-map", default=None, help="Path to safety_report.json from safety-scan.")
@click.option("--safety-mode", default=None, type=click.Choice(["protect", "target"]),
              help="'protect': never prune safety experts; 'target': specifically prune them.")
@click.option("--domain-map", default=None, help="Path to domain_report.json from domain-scan.")
@click.option("--domain-mode", default=None, type=click.Choice(["protect"]),
              help="'protect': never prune domain experts.")
@click.option("--ignore-experts", default=None,
              help="Comma-separated expert indices to protect from model-wide pruning. "
                   "Format: 1,2,250..255 (ranges inclusive). Only valid with --model-wide.")
@click.option("--stream", is_flag=True, default=False,
              help="Stream-prune by slicing safetensors shards directly. Bypasses mlx_lm.load. "
                   "Use for models too large to fit in unified memory. Per-tensor peak RAM (~few GB) "
                   "instead of full-model peak. Currently supports per-layer 'bottom' strategy with "
                   "saliency input; --strided / --model-wide / --expert-list not supported in stream mode.")
def prune(model, saliency, expert_list, output, n_prune, metric, strategy, model_wide, min_experts_per_layer,
          safety_map, safety_mode, domain_map, domain_mode, ignore_experts, stream):
    """Prune experts from model based on saliency statistics or expert list.
    
    Two modes of operation:
    
    1. Using saliency file (traditional):
       mlx-fun prune --model ./model --saliency stats.npz --n-prune 8 --output ./pruned
    
    2. Using expert list from frontend (new):
       mlx-fun prune --model ./model --expert-list filtered_experts.json --output ./pruned
    """
    from .loader import load_model

    from .adapters import get_adapter
    from .pruner import (
        build_keep_map,
        prune_model, load_safety_constraints, load_domain_constraints,
        load_expert_list, parse_expert_list,
    )
    from .saliency import SaliencyAccumulator
    from .save import save_pruned_model

    # Validate inputs
    if expert_list is None and saliency is None:
        raise click.UsageError(
            "Either --expert-list or --saliency must be provided."
        )
    if expert_list is None and n_prune is None:
        raise click.UsageError(
            "--n-prune is required when using --saliency."
        )
    if expert_list is not None and n_prune is not None:
        click.echo("Warning: --n-prune is ignored when --expert-list is provided.")
    
    # Validate --ignore-experts usage
    if ignore_experts and not model_wide:
        raise click.UsageError(
            "--ignore-experts is only valid with --model-wide."
        )

    # Clean output directory if it exists
    expanded_output = os.path.expanduser(output)
    if os.path.exists(expanded_output):
        click.echo(f"Removing existing output directory: {expanded_output}")
        shutil.rmtree(expanded_output)

    # Expand user path and validate if it's a local path
    expanded_model = os.path.expanduser(model)
    if os.path.exists(expanded_model):
        model = expanded_model
        click.echo(f"Loading model from local path: {model}")
    else:
        click.echo(f"Loading model: {model}")

    # Stream-prune path: skip mlx_lm.load entirely; slice safetensors directly.
    if stream:
        if expert_list is not None or model_wide or strategy == "strided":
            raise click.UsageError(
                "--stream currently supports only saliency-based per-layer 'bottom' pruning. "
                "Drop --expert-list / --model-wide / --strategy strided."
            )
        import json as _json
        from .stream_pruner import stream_prune, _moe_layer_indices, _expert_count_key, EXPERT_TENSOR_PATTERNS
        with open(os.path.join(model, "config.json")) as _f:
            cfg = _json.load(_f)
        mt = cfg.get("model_type", "")
        if mt not in EXPERT_TENSOR_PATTERNS:
            raise click.UsageError(
                f"--stream: model_type '{mt}' has no tensor pattern registered. "
                f"Add an entry in src/mlx_fun/stream_pruner.py:EXPERT_TENSOR_PATTERNS."
            )
        click.echo(f"[stream] model_type={mt}")
        click.echo(f"[stream] loading saliency: {saliency}")
        acc = SaliencyAccumulator.load(saliency)
        scores = acc.compute_scores(metric)
        click.echo(f"[stream] scores shape={scores.shape} metric={metric}")

        protected_experts, targeted_experts = None, None
        if safety_map and safety_mode:
            click.echo(f"[stream] safety map: {safety_map} (mode={safety_mode})")
            protected_experts, targeted_experts = load_safety_constraints(safety_map, safety_mode)
        if domain_map and domain_mode:
            click.echo(f"[stream] domain map: {domain_map} (mode={domain_mode})")
            domain_protected, _ = load_domain_constraints(domain_map, domain_mode)
            if domain_protected:
                if protected_experts is None:
                    protected_experts = {}
                for li, ex in domain_protected.items():
                    protected_experts[li] = (
                        np.union1d(protected_experts[li], ex) if li in protected_experts else ex
                    )

        # Stream mode has always used per-layer bottom selection; keep it that way.
        keep_map = build_keep_map(
            scores, n_prune,
            protected_experts=protected_experts,
            targeted_experts=targeted_experts,
        )
        moe_indices = _moe_layer_indices(cfg)
        keep_map_model = {moe_indices[i]: k for i, k in keep_map.items()}
        click.echo(f"[stream] per-layer kept={len(next(iter(keep_map_model.values())))} / {acc.num_experts}")
        click.echo(f"[stream] writing pruned shards to: {expanded_output}")
        new_cfg = stream_prune(model, expanded_output, keep_map_model, log_progress=True)
        # The expert-count key may live in nested text_config (Kimi/multimodal).
        ek = _expert_count_key(mt)
        new_n = new_cfg.get(ek)
        if new_n is None and "text_config" in new_cfg:
            new_n = new_cfg["text_config"].get(ek)
        click.echo(f"[stream] done — new {ek}={new_n}")
        return

    try:
        mlx_model, tokenizer, config = load_model(model)
    except Exception as e:
        if "HFValidationError" in str(type(e).__name__) or "Repo id must be in the form" in str(e):
            click.echo(f"\nError: Model path '{model}' could not be loaded as a local file or HuggingFace repo.", err=True)
            click.echo(f"Please verify:", err=True)
            click.echo(f"  1. The path exists and contains model files (config.json, tokenizer files, etc.)", err=True)
            click.echo(f"  2. If using a HuggingFace repo, ensure the repo ID is correct (format: 'username/repo-name')", err=True)
            raise
        raise

    adapter = get_adapter(mlx_model, config)
    original_n_experts = adapter.num_routed_experts()
    num_moe_layers = len(adapter.moe_layer_indices())

    # Determine keep_map source
    if expert_list:
        click.echo(f"Loading expert list from: {expert_list}")
        keep_map = load_expert_list(expert_list)
        total_kept = sum(len(v) for v in keep_map.values())
        total_original = original_n_experts * num_moe_layers
        click.echo(f"Loaded keep_map with {total_kept} experts to keep ({total_original - total_kept} to prune)")
        
        # Validate keep_map dimensions
        if len(keep_map) != num_moe_layers:
            raise ValueError(
                f"Expert list has {len(keep_map)} layers but model has {num_moe_layers} MoE layers."
            )
    else:
        # Existing saliency-based logic
        click.echo(f"Loading saliency from: {saliency}")
        acc = SaliencyAccumulator.load(saliency)
        scores = acc.compute_scores(metric)

        # Load safety constraints if provided
        protected_experts, targeted_experts = None, None
        if safety_map and safety_mode:
            click.echo(f"Loading safety map: {safety_map} (mode={safety_mode})")
            protected_experts, targeted_experts = load_safety_constraints(safety_map, safety_mode)
        elif safety_map and not safety_mode:
            raise click.UsageError("--safety-mode is required when --safety-map is provided.")

        # Load domain constraints if provided (merge with safety via union)
        if domain_map and domain_mode:
            click.echo(f"Loading domain map: {domain_map} (mode={domain_mode})")
            domain_protected, _ = load_domain_constraints(domain_map, domain_mode)
            if domain_protected:
                if protected_experts is None:
                    protected_experts = {}
                for layer_idx, experts in domain_protected.items():
                    if layer_idx in protected_experts:
                        merged = np.union1d(protected_experts[layer_idx], experts)
                        protected_experts[layer_idx] = merged
                    else:
                        protected_experts[layer_idx] = experts
        elif domain_map and not domain_mode:
            raise click.UsageError("--domain-mode is required when --domain-map is provided.")

        # Select experts to keep based on mode
        ignored_set = None
        if model_wide:
            # Parse ignored experts if provided
            if ignore_experts:
                ignored_set = parse_expert_list(ignore_experts)
                click.echo(f"Ignoring {len(ignored_set)} expert indices: {sorted(ignored_set)}")

            click.echo(f"Selecting experts to prune (model-wide: {n_prune} total, metric={metric})")
        else:
            click.echo(f"Selecting experts to prune (per-layer: {n_prune}/layer, metric={metric}, strategy={strategy})")

        keep_map = build_keep_map(
            scores, n_prune,
            strategy=strategy,
            model_wide=model_wide,
            protected_experts=protected_experts,
            targeted_experts=targeted_experts,
            min_experts_per_layer=min_experts_per_layer,
            ignored_experts=ignored_set,
        )

        if model_wide:
            # Calculate total pruned and per-layer distribution
            total_pruned = sum(original_n_experts - len(keep_map[i]) for i in range(len(keep_map)))
            click.echo(f"Model-wide pruning: {total_pruned} experts removed across {num_moe_layers} layers")

    # Map from accumulator layer indices to actual model layer indices
    moe_indices = adapter.moe_layer_indices()
    model_keep_map = {
        moe_indices[acc_idx]: keep
        for acc_idx, keep in keep_map.items()
    }

    click.echo("Pruning model...")
    new_config = prune_model(adapter, model_keep_map)

    click.echo(f"Saving pruned model to: {output}")
    save_pruned_model(
        mlx_model, tokenizer, new_config, output,
        model_keep_map, original_n_experts, metric,
    )
    
    # Calculate final expert counts
    if expert_list or model_wide:
        total_kept = sum(len(keep) for keep in model_keep_map.values())
        total_original = original_n_experts * num_moe_layers
        click.echo(f"Done! Total experts: {total_original} -> {total_kept}")
    else:
        click.echo(f"Done! Experts per layer: {original_n_experts} -> {original_n_experts - n_prune}")


@main.command()
@click.option("--model", required=True, help="Model path or HuggingFace repo ID.")
@click.option("--saliency", default=None, help="Path to saliency .npz file. Required if --expert-list is not provided.")
@click.option("--expert-list", default=None,
              help="Path to expert list .json or .csv from frontend export. "
                   "Bypasses --n-prune calculation. Use instead of --saliency.")
@click.option("--dataset", required=True, help="Calibration dataset (JSONL or directory).")
@click.option("--output", required=True, help="Output directory for merged model.")
@click.option("--n-prune", default=None, type=int,
              help="Number of experts to prune per layer (or total if --model-wide). "
                   "Required if --expert-list is not provided.")
@click.option("--metric", default="reap", type=click.Choice(["reap", "ean", "freq", "weighted_freq"]))
@click.option("--model-wide", is_flag=True, default=False,
              help="Select N experts globally across all layers instead of per-layer.")
@click.option("--min-experts-per-layer", default=1, type=int,
              help="Minimum experts to keep per layer when using --model-wide (default: 1).")
@click.option("--similarity-mode", default="gated", type=click.Choice(["gated", "average"]),
              help="Similarity metric: 'gated' or 'average'.")
@click.option("--alignment", default="greedy", type=click.Choice(["greedy", "hungarian", "none"]),
              help="Neuron alignment method for permutation.")
@click.option("--max-group-size", default=16, type=int, help="Maximum experts per merge group (C).")
@click.option("--max-samples", default=64, type=int, help="Calibration samples for similarity/alignment.")
@click.option("--max-tokens", default=2048, type=int, help="Max tokens per sample.")
@click.option("--max-similarity-tokens", default=512, type=int,
              help="Max tokens for similarity computation.")
@click.option("--max-alignment-tokens", default=256, type=int,
              help="Max tokens for permutation alignment.")
@click.option("--text-key", default="content", help="JSON key for text in JSONL.")
@click.option("--seed", default=None, type=int, help="Random seed.")
@click.option("--ignore-experts", default=None,
              help="Comma-separated expert indices to protect from model-wide merge. "
                   "Format: 1,2,250..255 (ranges inclusive). Only valid with --model-wide.")
def merge(model, saliency, expert_list, dataset, output, n_prune, metric, model_wide, min_experts_per_layer,
          similarity_mode, alignment, max_group_size, max_samples, max_tokens,
          max_similarity_tokens, max_alignment_tokens, text_key, seed, ignore_experts):
    """Merge experts using REAM (Router-weighted Expert Activation Merging).
    
    Two modes of operation:
    
    1. Using saliency file (traditional):
       mlx-fun merge --model ./model --saliency stats.npz --dataset calib.jsonl --n-prune 8 --output ./merged
    
    2. Using expert list from frontend (new):
       mlx-fun merge --model ./model --expert-list filtered_experts.json --dataset calib.jsonl --output ./merged
    """
    import random
    import mlx.core as mx
    from .loader import load_model
    from tqdm import tqdm

    # Validate inputs
    if expert_list is None and saliency is None:
        raise click.UsageError(
            "Either --expert-list or --saliency must be provided."
        )
    if expert_list is None and n_prune is None:
        raise click.UsageError(
            "--n-prune is required when using --saliency."
        )
    if expert_list is not None and n_prune is not None:
        click.echo("Warning: --n-prune is ignored when --expert-list is provided.")
    
    # Validate --ignore-experts usage
    if ignore_experts and not model_wide:
        raise click.UsageError(
            "--ignore-experts is only valid with --model-wide."
        )

    # Clean output directory if it exists
    expanded_output = os.path.expanduser(output)
    if os.path.exists(expanded_output):
        click.echo(f"Removing existing output directory: {expanded_output}")
        shutil.rmtree(expanded_output)

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    from .adapters import get_adapter
    from .data import load_dataset
    from .merger import merge_model, merge_model_with_keep_map
    from .pruner import select_experts_to_keep_model_wide, load_expert_list, parse_expert_list
    from .saliency import SaliencyAccumulator
    from .save import save_merged_model

    # Load model
    expanded_model = os.path.expanduser(model)
    if os.path.exists(expanded_model):
        model = expanded_model
        click.echo(f"Loading model from local path: {model}")
    else:
        click.echo(f"Loading model: {model}")

    try:
        mlx_model, tokenizer, config = load_model(model)
    except Exception as e:
        if "HFValidationError" in str(type(e).__name__) or "Repo id must be in the form" in str(e):
            click.echo(f"\nError: Model path '{model}' could not be loaded.", err=True)
            raise
        raise

    adapter = get_adapter(mlx_model, config)
    original_n_experts = adapter.num_routed_experts()
    num_moe_layers = len(adapter.moe_layer_indices())

    click.echo(f"Model type: {config.get('model_type')}")

    # Load calibration data (always needed for merge)
    click.echo(f"Loading calibration dataset: {dataset}")
    samples = load_dataset(dataset, tokenizer, max_tokens, max_samples, text_key)
    click.echo(f"Loaded {len(samples)} calibration samples")

    # Determine keep_map source and load scores if needed
    if expert_list:
        click.echo(f"Loading expert list from: {expert_list}")
        keep_map = load_expert_list(expert_list)
        total_kept = sum(len(v) for v in keep_map.values())
        total_original = original_n_experts * num_moe_layers
        click.echo(f"Loaded keep_map with {total_kept} experts to keep ({total_original - total_kept} to merge)")
        
        # Validate keep_map dimensions
        if len(keep_map) != num_moe_layers:
            raise ValueError(
                f"Expert list has {len(keep_map)} layers but model has {num_moe_layers} MoE layers."
            )
        
        # We still need saliency scores for merge ordering
        if saliency:
            click.echo(f"Loading saliency from: {saliency}")
            acc = SaliencyAccumulator.load(saliency)
            scores = acc.compute_scores(metric)
        else:
            # Create dummy scores (all equal) - merge will use similarity only
            click.echo("Warning: No saliency file provided. Merge will rely on similarity only.")
            scores = np.ones((num_moe_layers, original_n_experts), dtype=np.float32)
    else:
        # Existing saliency-based logic
        n_keep = original_n_experts - n_prune
        
        if model_wide:
            click.echo(f"MoE layers: {num_moe_layers}, "
                       f"Total experts: {original_n_experts * num_moe_layers} -> "
                       f"{original_n_experts * num_moe_layers - n_prune} (model-wide)")
        else:
            click.echo(f"MoE layers: {num_moe_layers}, "
                       f"Experts per layer: {original_n_experts} -> {n_keep}")

        click.echo(f"Loading saliency from: {saliency}")
        acc = SaliencyAccumulator.load(saliency)
        scores = acc.compute_scores(metric)

    # Merge
    def progress(layer_num, total):
        click.echo(f"  Processing MoE layer {layer_num + 1}/{total}...")

    if expert_list or model_wide:
        # Use keep_map-based merge
        if not expert_list:
            # Compute keep_map from saliency
            # Parse ignored experts if provided
            ignored_set = None
            if ignore_experts:
                ignored_set = parse_expert_list(ignore_experts)
                click.echo(f"Ignoring {len(ignored_set)} expert indices: {sorted(ignored_set)}")
            
            click.echo(f"Model-wide merge: selecting {n_prune} experts to merge globally...")
            keep_map = select_experts_to_keep_model_wide(
                scores, n_prune,
                min_experts_per_layer=min_experts_per_layer,
                ignored_experts=ignored_set,
            )
        
        # Calculate distribution
        kept_per_layer = [len(keep_map[i]) for i in range(len(keep_map))]
        click.echo(f"Experts per layer after merge: min={min(kept_per_layer)}, max={max(kept_per_layer)}, avg={sum(kept_per_layer)/len(kept_per_layer):.1f}")
        
        click.echo(f"Merging (similarity={similarity_mode}, alignment={alignment}, "
                   f"max_group_size={max_group_size})...")
        
        new_config, centroid_map, group_map = merge_model_with_keep_map(
            mlx_model, adapter, keep_map, scores, samples,
            similarity_mode=similarity_mode,
            alignment_method=alignment,
            max_group_size=max_group_size,
            max_similarity_tokens=max_similarity_tokens,
            max_alignment_tokens=max_alignment_tokens,
            progress_callback=progress,
        )
    else:
        click.echo(f"Merging (similarity={similarity_mode}, alignment={alignment}, "
                   f"max_group_size={max_group_size})...")
        
        new_config, centroid_map, group_map = merge_model(
            mlx_model, adapter, scores, n_keep, samples,
            similarity_mode=similarity_mode,
            alignment_method=alignment,
            max_group_size=max_group_size,
            max_similarity_tokens=max_similarity_tokens,
            max_alignment_tokens=max_alignment_tokens,
            progress_callback=progress,
        )

    # Save
    click.echo(f"Saving merged model to: {output}")
    save_merged_model(
        mlx_model, tokenizer, new_config, output,
        centroid_map, group_map, original_n_experts, metric,
    )
    
    if expert_list or model_wide:
        total_kept = sum(len(keep_map[i]) for i in range(len(keep_map)))
        total_original = original_n_experts * num_moe_layers
        click.echo(f"Done! Total experts: {total_original} -> {total_kept}")
    else:
        click.echo(f"Done! Experts per layer: {original_n_experts} -> {n_keep}")


@main.command("smoke-test")
@click.option("--model", required=True, help="Path to pruned model.")
@click.option("--prompt", default="pragma solidity ^0.8.0;", help="Test prompt.")
@click.option("--max-tokens", default=100, help="Maximum tokens to generate.")
@click.option("--kv-compress", default=None, type=click.Choice(["turbo", "rotor"]),
              help="KV cache compression: 'turbo' (TurboQuant/PolarQuant) or 'rotor' (RotorQuant/Clifford).")
@click.option("--kv-compress-bits", default=4, type=int,
              help="Bits per channel for KV compression (2-8). Default: 4 (turbo), 3 (rotor).")
def smoke_test(model, prompt, max_tokens, kv_compress, kv_compress_bits):
    """Verify generation works with a pruned model."""
    from .loader import load_model
    from mlx_lm.generate import generate_step, stream_generate

    # Expand user path and validate if it's a local path
    expanded_model = os.path.expanduser(model)
    if os.path.exists(expanded_model):
        model = expanded_model
        click.echo(f"Loading model from local path: {model}")
    else:
        click.echo(f"Loading model: {model}")

    try:
        mlx_model, tokenizer, model_config = load_model(model)
    except Exception as e:
        if "HFValidationError" in str(type(e).__name__) or "Repo id must be in the form" in str(e):
            click.echo(f"\nError: Model path '{model}' could not be loaded as a local file or HuggingFace repo.", err=True)
            click.echo(f"Please verify:", err=True)
            click.echo(f"  1. The path exists and contains model files (config.json, tokenizer files, etc.)", err=True)
            click.echo(f"  2. If using a HuggingFace repo, ensure the repo ID is correct (format: 'username/repo-name')", err=True)
            raise
        raise

    from .loader import is_vision_model

    if is_vision_model(model_config):
        raise click.ClickException(
            f"smoke-test drives mlx-lm's text generation loop, which cannot run "
            f"the vision-language model '{model_config.get('model_type')}'. "
            f"Generate with mlx-vlm directly: "
            f"`python -m mlx_vlm.generate --model {model} --prompt ...`"
        )

    prompt_cache = None
    if kv_compress == "turbo":
        from .kv_compress import TurboQuantConfig, setup_turbo_quant

        # Load config to get model_type for SDPA patching
        import json
        config_path = os.path.join(os.path.expanduser(model), "config.json")
        model_type = ""
        if os.path.exists(config_path):
            with open(config_path) as f:
                model_type = json.load(f).get("model_type", "")

        cfg = TurboQuantConfig(bits=kv_compress_bits)
        prompt_cache, sdpa_patched = setup_turbo_quant(mlx_model, model_type, cfg)
        mode_str = "quantized SDPA" if sdpa_patched else "plain SDPA"
        click.echo(f"TurboQuant KV compression enabled ({kv_compress_bits}-bit, {mode_str})")
    elif kv_compress == "rotor":
        from .rotor_quant import RotorQuantConfig, setup_rotor_quant

        cfg = RotorQuantConfig(bits=kv_compress_bits)
        prompt_cache = setup_rotor_quant(mlx_model, cfg)
        click.echo(f"RotorQuant KV compression enabled ({kv_compress_bits}-bit, Clifford rotors)")

    click.echo(f"Generating with prompt: {prompt!r}")
    result = ""
    for response in stream_generate(
        mlx_model, tokenizer, prompt=prompt, max_tokens=max_tokens,
        prompt_cache=prompt_cache,
    ):
        click.echo(response.text, nl=False)
        result += response.text
    click.echo()
    click.echo(f"\nGeneration successful ({len(result)} chars)")


@main.command()
@click.option("--model", required=False, default=None,
              help="Model path or HuggingFace repo ID. If omitted, loads on first request.")
@click.option("--host", default="0.0.0.0",
              help="Server bind address. Default 0.0.0.0 binds all interfaces "
                   "(localhost, LAN, UTM guests). Use 127.0.0.1 for loopback only.")
@click.option("--port", default=8080, type=int, help="Server port.")
@click.option("--mode", default="lightweight", type=click.Choice(["lightweight", "full"]),
              help="Hook mode: 'lightweight' skips activation norms, 'full' computes all metrics.")
@click.option("--auto-save", default=None, help="Path to auto-save stats on shutdown.")
@click.option("--max-tokens", default=512, type=int, help="Default max tokens for generation.")
@click.option("--max-kv-size", default=None, type=int,
              help="Max KV cache size per layer (tokens). Uses RotatingKVCache sliding window "
                   "to cap memory usage for long conversations.")
@click.option("--chat-template", default=None, help="Chat template override.")
@click.option("--chat-template-args", default=None,
              help="JSON string of extra apply_chat_template kwargs applied "
                   "server-wide. Use to disable Gemma 4 thinking by default: "
                   "--chat-template-args '{\"enable_thinking\":false}'.")
@click.option("--safety-map", default=None, help="Path to safety_report.json for steering.")
@click.option("--steering-mode", default=None, type=click.Choice(["safe", "unsafe"]),
              help="Steering mode: 'safe' boosts safety experts, 'unsafe' masks them.")
@click.option("--domain-map", default=None, help="Path to domain_report.json for domain boosting.")
@click.option("--domain-steering-mode", default=None, type=click.Choice(["boost", "suppress"]),
              help="Domain steering: 'boost' activates domain experts, 'suppress' deactivates general.")
@click.option("--kv-compress", default=None, type=click.Choice(["turbo", "rotor"]),
              help="KV cache compression method: 'turbo' (TurboQuant/PolarQuant) or 'rotor' (RotorQuant/Clifford).")
@click.option("--kv-compress-bits", default=4, type=int,
              help="Bits per channel for KV compression (2-8). Default: 4 (turbo), 3 (rotor).")
@click.option("--idle-timeout", default=1800, type=int,
              help="Auto-unload model after N seconds of inactivity. 0 to disable. Default: 1800 (30 min).")
@click.option("--draft-model", default=None,
              help="Path or HuggingFace repo ID for a draft model (speculative decoding).")
@click.option("--num-draft-tokens", default=3, type=int,
              help="Number of tokens to draft per speculative decoding step. Default: 3.")
@click.option("--capture-layers", default=None,
              help="Capture hidden states from these decoder layers during prefill. "
                   "Comma-separated indices (e.g. '0,4,8') or 'all'. For speculative decoding Phase 2.")
@click.option("--dflash-block-size", default=None,
              help="Enable DFlash block diffusion draft model with this block size. "
                   "Accepts 'b16', 'b32', or plain integers like '16'. "
                   "Auto-configures capture layers if --capture-layers is not set.")
@click.option("--dflash-num-layers", default=5, type=int,
              help="Number of transformer layers in the DFlash draft model. Default: 5.")
@click.option("--dflash-num-heads", default=8, type=int,
              help="Number of attention heads in the DFlash draft model. Default: 8.")
@click.option("--log-level",
              type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
              default="INFO",
              help="Log verbosity. DEBUG includes per-request hook + tool-call traces.")
@click.option("--default-temperature", default=None, type=float,
              help="Server-wide default temperature applied when the request omits it.")
@click.option("--default-top-p", default=None, type=float,
              help="Server-wide default top_p applied when the request omits it.")
@click.option("--default-top-k", default=None, type=int,
              help="Server-wide default top_k applied when the request omits it. "
                   "GLM-5.1 Q3 works well at 100.")
@click.option("--default-min-p", default=None, type=float,
              help="Server-wide default min_p applied when the request omits it.")
@click.option("--default-repetition-penalty", default=None, type=float,
              help="Server-wide default repetition_penalty applied when the request omits it. "
                   "GLM-5.1 Q3 works well at 1.1.")
@click.option("--default-repetition-context-size", default=None, type=int,
              help="Token window for the repetition penalty (default upstream: 20).")
@click.option("--default-seed", default=None, type=int,
              help="Server-wide default sampling seed applied when the request "
                   "omits it. Useful for reproducing stalls/loops bit-for-bit. "
                   "Per-request `seed` in the body still wins.")
@click.option("--enable-counting", is_flag=True, default=False,
              help="Install MoE expert-counting hooks so /v1/reap/save and "
                   "/v1/reap/stats return routing data. Off by default — "
                   "plain inference is the common case and the hooks add a "
                   "small per-token overhead.")
@click.option("--prompt-cache-size", default=10, type=int,
              help="Number of prompt-cache entries kept by the LRUPromptCache. "
                   "Default 10. Lower (e.g. 1) when each conversation is large "
                   "and you only need same-thread cache reuse, freeing GPU "
                   "memory that would otherwise be held by stale prefixes.")
@click.option("--trust-remote-code", is_flag=True, default=False,
              help="Allow custom tokenizer/model code from the model repo to be "
                   "executed (passes trust_remote_code=True to the HuggingFace "
                   "AutoTokenizer). Required for models like Kimi-K2 / "
                   "MiniMax-M2 that ship a Python tokenizer alongside the "
                   "weights. Off by default for safety.")
def serve(model, host, port, mode, auto_save, max_tokens, max_kv_size,
          chat_template, chat_template_args, safety_map, steering_mode,
          domain_map, domain_steering_mode, kv_compress, kv_compress_bits,
          idle_timeout, draft_model, num_draft_tokens, capture_layers,
          dflash_block_size, dflash_num_layers, dflash_num_heads, log_level,
          default_temperature, default_top_p, default_top_k, default_min_p,
          default_repetition_penalty, default_repetition_context_size,
          default_seed,
          enable_counting, prompt_cache_size, trust_remote_code):
    """Serve model with on-demand loading and online expert counting.

    Starts an OpenAI and Anthropic compatible server. Models are loaded on
    demand when the first request arrives. After --idle-timeout seconds of
    inactivity, the model is unloaded to free memory.

    \b
    Examples:
      # Start empty, load model on first request
      mlx-fun serve --port 8080

      # Start with a specific model pre-loaded
      mlx-fun serve --model /path/to/model --port 8080

      # Disable auto-unload
      mlx-fun serve --model /path/to/model --idle-timeout 0
    """
    from .server import run_reap_server

    # Parse DFlash block size if provided
    parsed_dflash_block_size = None
    if dflash_block_size is not None:
        from .dflash_draft import parse_block_size
        parsed_dflash_block_size = parse_block_size(dflash_block_size)

    parsed_chat_template_args = {}
    if chat_template_args:
        import json as _json
        try:
            parsed_chat_template_args = _json.loads(chat_template_args)
            if not isinstance(parsed_chat_template_args, dict):
                raise ValueError("expected a JSON object")
        except Exception as e:
            raise click.BadParameter(
                f"--chat-template-args must be a JSON object: {e}"
            )

    run_reap_server(
        host=host,
        port=port,
        model_path=model,
        mode=mode,
        auto_save=auto_save,
        max_tokens=max_tokens,
        max_kv_size=max_kv_size,
        chat_template=chat_template,
        chat_template_args=parsed_chat_template_args,
        safety_map=safety_map,
        steering_mode=steering_mode,
        domain_map=domain_map,
        domain_steering_mode=domain_steering_mode,
        kv_compress=kv_compress,
        kv_compress_bits=kv_compress_bits,
        idle_timeout=float(idle_timeout),
        draft_model_path=draft_model,
        num_draft_tokens=num_draft_tokens,
        capture_layers=capture_layers,
        dflash_block_size=parsed_dflash_block_size,
        dflash_num_layers=dflash_num_layers,
        dflash_num_heads=dflash_num_heads,
        log_level=log_level,
        default_temperature=default_temperature,
        default_top_p=default_top_p,
        default_top_k=default_top_k,
        default_min_p=default_min_p,
        default_repetition_penalty=default_repetition_penalty,
        default_repetition_context_size=default_repetition_context_size,
        default_seed=default_seed,
        enable_counting=enable_counting,
        prompt_cache_size=prompt_cache_size,
        trust_remote_code=trust_remote_code,
    )


@main.command("safety-scan")
@click.option("--model", required=True, help="Model path or HuggingFace repo ID.")
@click.option("--harmful-dataset", required=True, help="Path to harmful prompts (JSONL/directory).")
@click.option("--benign-dataset", required=True, help="Path to benign prompts (JSONL/directory).")
@click.option("--output", required=True, help="Output path for safety_report.json.")
@click.option("--max-samples", default=128, type=int, help="Max samples per dataset.")
@click.option("--max-tokens", default=2048, type=int, help="Max tokens per sample.")
@click.option("--text-key", default="content", help="JSON key for text in JSONL.")
@click.option("--threshold-percentile", default=90.0, type=float,
              help="Percentile threshold for classifying safety-critical experts.")
@click.option("--seed", default=None, type=int, help="Random seed.")
def safety_scan(model, harmful_dataset, benign_dataset, output, max_samples,
                max_tokens, text_key, threshold_percentile, seed):
    """Identify safety-critical experts by comparing routing on harmful vs benign data.

    Implements SAFEx-style differential activation analysis. Produces a safety
    report classifying experts into HCDG (harmful content detection) and
    HRCG (harmful response control) groups.
    """
    import random
    import mlx.core as mx
    from .loader import load_model, text_forward
    from tqdm import tqdm

    if seed is not None:
        random.seed(seed)

    from .adapters import get_adapter
    from .data import load_dataset
    from .ream_hooks import install_ream_hooks, collect_ream_data, remove_ream_hooks
    from .safety import (
        DifferentialAccumulator, compute_differential_scores,
        identify_safety_experts,
    )

    # Load model
    expanded_model = os.path.expanduser(model)
    if os.path.exists(expanded_model):
        model = expanded_model
    click.echo(f"Loading model: {model}")
    mlx_model, tokenizer, config = load_model(model)

    adapter = get_adapter(mlx_model, config)
    moe_indices = adapter.moe_layer_indices()
    n_experts = adapter.num_routed_experts()
    top_k = adapter.num_experts_per_tok()
    model_type = config.get("model_type", "")

    click.echo(f"Model type: {model_type}, MoE layers: {len(moe_indices)}, "
               f"Experts: {n_experts}, top_k: {top_k}")

    acc = DifferentialAccumulator(num_layers=len(moe_indices), num_experts=n_experts)
    moe_blocks = [adapter.get_moe_block(i) for i in moe_indices]

    for dataset_name, dataset_path in [("harmful", harmful_dataset), ("benign", benign_dataset)]:
        click.echo(f"Loading {dataset_name} dataset: {dataset_path}")
        samples = load_dataset(dataset_path, tokenizer, max_tokens, max_samples, text_key)
        click.echo(f"  Loaded {len(samples)} samples")

        install_ream_hooks(moe_blocks, model_type)
        forward = text_forward(mlx_model, config)
        for sample in tqdm(samples, desc=f"Scanning {dataset_name}"):
            tokens = sample.reshape(1, -1)
            forward(tokens)
            mx.eval(mlx_model.parameters())

            captures = collect_ream_data(moe_blocks)
            for block_idx, block_captures in enumerate(captures):
                for layer_input, gate_logits, sel_inds in block_captures:
                    # Flatten batch and seq dims
                    gl_2d = gate_logits.reshape(-1, gate_logits.shape[-1])
                    acc.update_from_gate_logits(block_idx, gl_2d, dataset_name)
                    # Real selection captured from the gate, not reconstructed.
                    inds_2d = sel_inds.reshape(-1, sel_inds.shape[-1])
                    acc.update_from_top_k(block_idx, inds_2d, dataset_name)

        remove_ream_hooks(moe_blocks)

    click.echo("Computing differential scores...")
    diff_freq, diff_act, composite = compute_differential_scores(acc)
    report = identify_safety_experts(diff_freq, diff_act, composite, threshold_percentile)

    report.save(output)
    total_hcdg = sum(len(v) for v in report.hcdg_experts.values())
    total_hrcg = sum(len(v) for v in report.hrcg_experts.values())
    total_safety = sum(len(v) for v in report.safety_critical.values())
    click.echo(f"Safety report saved to: {output}")
    click.echo(f"  HCDG experts: {total_hcdg}, HRCG experts: {total_hrcg}, "
               f"Total safety-critical: {total_safety}")


@main.command()
@click.option("--model", required=True, help="Model path or HuggingFace repo ID.")
@click.option("--safety-map", required=True, help="Path to safety_report.json.")
@click.option("--mode", required=True, type=click.Choice(["safe", "unsafe"]),
              help="'safe' boosts safety experts, 'unsafe' masks them.")
@click.option("--prompt", required=True, help="Input prompt for generation.")
@click.option("--max-tokens", default=100, type=int, help="Max tokens to generate.")
@click.option("--mask-value", default=-1e9, type=float, help="Gate logit bias for deactivation.")
@click.option("--boost-value", default=1e4, type=float, help="Gate logit bias for activation.")
@click.option("--kv-compress", default=None, type=click.Choice(["turbo", "rotor"]),
              help="KV cache compression: 'turbo' (TurboQuant/PolarQuant) or 'rotor' (RotorQuant/Clifford).")
@click.option("--kv-compress-bits", default=4, type=int,
              help="Bits per channel for KV compression (2-8). Default: 4 (turbo), 3 (rotor).")
def steer(model, safety_map, mode, prompt, max_tokens, mask_value, boost_value,
          kv_compress, kv_compress_bits):
    """Generate text with expert steering based on safety analysis.

    Uses SteerMoE-style gate logit injection to selectively activate or
    deactivate safety-critical experts during inference.
    """
    from mlx_lm import generate

    from .loader import load_model

    from .adapters import get_adapter
    from .steering import SteeringConfig, install_steering_hooks, remove_steering_hooks

    expanded_model = os.path.expanduser(model)
    if os.path.exists(expanded_model):
        model = expanded_model
    click.echo(f"Loading model: {model}")
    mlx_model, tokenizer, config = load_model(model)

    adapter = get_adapter(mlx_model, config)
    moe_indices = adapter.moe_layer_indices()
    n_experts = adapter.num_routed_experts()
    model_type = config.get("model_type", "")

    # Build steering config
    steer_config = SteeringConfig.from_safety_report(safety_map, mode)
    steer_config.mask_value = mask_value
    steer_config.boost_value = boost_value

    total_steered = sum(
        len(v) for v in steer_config.deactivate.values()
    ) + sum(
        len(v) for v in steer_config.activate.values()
    )
    click.echo(f"Steering mode: {mode}, affecting {total_steered} expert-layer pairs")

    # Install steering hooks
    moe_blocks = [adapter.get_moe_block(i) for i in moe_indices]
    install_steering_hooks(moe_blocks, model_type, steer_config, n_experts)

    prompt_cache = None
    if kv_compress == "turbo":
        from .kv_compress import TurboQuantConfig, setup_turbo_quant

        cfg = TurboQuantConfig(bits=kv_compress_bits)
        prompt_cache, sdpa_patched = setup_turbo_quant(mlx_model, model_type, cfg)
        mode_str = "quantized SDPA" if sdpa_patched else "plain SDPA"
        click.echo(f"TurboQuant KV compression enabled ({kv_compress_bits}-bit, {mode_str})")
    elif kv_compress == "rotor":
        from .rotor_quant import RotorQuantConfig, setup_rotor_quant

        cfg = RotorQuantConfig(bits=kv_compress_bits)
        prompt_cache = setup_rotor_quant(mlx_model, cfg)
        click.echo(f"RotorQuant KV compression enabled ({kv_compress_bits}-bit, Clifford rotors)")

    click.echo(f"Generating with prompt: {prompt!r}")
    result = generate(
        mlx_model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=True,
        prompt_cache=prompt_cache,
    )
    remove_steering_hooks(moe_blocks)

    if kv_compress == "turbo":
        from .kv_compress import remove_turbo_quant_sdpa
        remove_turbo_quant_sdpa(model_type)

    click.echo(f"\nGeneration successful ({len(result)} chars)")


@main.command()
@click.option("--model", required=True, help="Model path or HuggingFace repo ID.")
@click.option("--harmful-dataset", required=True, help="Path to harmful prompts.")
@click.option("--benign-dataset", required=True, help="Path to benign prompts.")
@click.option("--output", required=True, help="Output directory for abliterated model.")
@click.option("--layers", default="auto", help="Layer range: 'auto', 'all', or 'start-end' (e.g. '10-20').")
@click.option("--target", default="all", type=click.Choice(["all", "safety-experts", "dense-only"]),
              help="Which weights to orthogonalize.")
@click.option("--safety-map", default=None, help="Required if target=safety-experts.")
@click.option("--max-samples", default=64, type=int, help="Max samples per dataset.")
@click.option("--max-tokens", default=2048, type=int, help="Max tokens per sample.")
@click.option("--extraction-position", default="last", type=click.Choice(["last", "mean"]),
              help="Where in the sequence to extract activations.")
@click.option("--text-key", default="content", help="JSON key for text.")
@click.option("--seed", default=None, type=int, help="Random seed.")
def abliterate(model, harmful_dataset, benign_dataset, output, layers, target,
               safety_map, max_samples, max_tokens, extraction_position, text_key, seed):
    """Remove refusal direction from model weights (abliteration).

    Computes the refusal direction from mean activation differences between
    harmful and benign prompts, then projects it out of weight matrices.
    Supports MoE-specific targeting of safety-critical experts.
    """
    import random
    import mlx.core as mx
    from .loader import load_model

    if seed is not None:
        random.seed(seed)

    from .adapters import get_adapter
    from .data import load_dataset
    from .abliterate import (
        compute_refusal_directions, orthogonalize_weights, auto_select_layers,
    )
    from .save import save_abliterated_model

    # Load model
    expanded_model = os.path.expanduser(model)
    if os.path.exists(expanded_model):
        model = expanded_model
    click.echo(f"Loading model: {model}")
    mlx_model, tokenizer, config = load_model(model)
    adapter = get_adapter(mlx_model, config)

    # Load safety report if needed
    safety_report = None
    if target == "safety-experts":
        if not safety_map:
            raise click.UsageError("--safety-map is required when target=safety-experts.")
        from .safety import SafetyReport
        safety_report = SafetyReport.load(safety_map)
        click.echo(f"Loaded safety map: {safety_map}")

    # Parse layer range
    n_layers = len(mlx_model.model.layers)
    if layers == "all":
        layer_indices = list(range(n_layers))
    elif layers == "auto":
        layer_indices = None  # Will auto-select after computing directions
    else:
        # Parse "start-end" format
        parts = layers.split("-")
        if len(parts) == 2:
            layer_indices = list(range(int(parts[0]), int(parts[1]) + 1))
        else:
            layer_indices = [int(layers)]

    # Load datasets
    click.echo(f"Loading harmful dataset: {harmful_dataset}")
    harmful_samples = load_dataset(harmful_dataset, tokenizer, max_tokens, max_samples, text_key)
    click.echo(f"  Loaded {len(harmful_samples)} samples")

    click.echo(f"Loading benign dataset: {benign_dataset}")
    benign_samples = load_dataset(benign_dataset, tokenizer, max_tokens, max_samples, text_key)
    click.echo(f"  Loaded {len(benign_samples)} samples")

    # Compute refusal directions (on all layers first if auto-selecting)
    compute_layers = layer_indices if layer_indices is not None else list(range(n_layers))
    click.echo(f"Computing refusal directions across {len(compute_layers)} layers...")
    refusal_directions = compute_refusal_directions(
        mlx_model, adapter, harmful_samples, benign_samples,
        layer_indices=compute_layers,
        extraction_position=extraction_position,
    )

    # Auto-select layers if needed
    if layers == "auto":
        layer_indices = auto_select_layers(refusal_directions)
        # Filter directions to selected layers only
        refusal_directions = {k: v for k, v in refusal_directions.items() if k in layer_indices}
        click.echo(f"Auto-selected {len(layer_indices)} layers with strongest refusal directions")

    # Orthogonalize weights
    click.echo(f"Orthogonalizing weights (target={target}, {len(refusal_directions)} layers)...")
    orthogonalize_weights(
        mlx_model, adapter, refusal_directions,
        target=target, safety_report=safety_report,
    )

    # Save
    click.echo(f"Saving abliterated model to: {output}")
    save_abliterated_model(
        mlx_model, tokenizer, config, output,
        refusal_directions, target, sorted(refusal_directions.keys()),
    )
    click.echo("Done!")


@main.command("domain-scan")
@click.option("--model", required=True, help="Model path or HuggingFace repo ID.")
@click.option("--domain-dataset", required=True, help="Path to domain data (JSONL/directory).")
@click.option("--general-dataset", required=True, help="Path to general data (JSONL/directory).")
@click.option("--output", required=True, help="Output path for domain_report.json.")
@click.option("--domain-name", required=True, help="Name of the domain (e.g. 'solidity', 'medical').")
@click.option("--max-samples", default=128, type=int, help="Max samples per dataset.")
@click.option("--max-tokens", default=2048, type=int, help="Max tokens per sample.")
@click.option("--text-key", default="content", help="JSON key for text in JSONL.")
@click.option("--threshold-percentile", default=90.0, type=float,
              help="Percentile threshold for classifying domain-specialized experts.")
@click.option("--seed", default=None, type=int, help="Random seed.")
def domain_scan(model, domain_dataset, general_dataset, output, domain_name,
                max_samples, max_tokens, text_key, threshold_percentile, seed):
    """Identify domain-specialized experts by comparing routing on domain vs general data.

    Uses the same differential activation analysis as safety-scan, but classifies
    experts into domain-specialized (activated more on domain data) and general
    (activated more on general data) groups.
    """
    import random
    import mlx.core as mx
    from .loader import load_model, text_forward
    from tqdm import tqdm

    if seed is not None:
        random.seed(seed)

    from .adapters import get_adapter
    from .data import load_dataset
    from .domain import identify_domain_experts
    from .ream_hooks import install_ream_hooks, collect_ream_data, remove_ream_hooks
    from .safety import (
        DifferentialAccumulator, compute_differential_scores,
    )

    # Load model
    expanded_model = os.path.expanduser(model)
    if os.path.exists(expanded_model):
        model = expanded_model
    click.echo(f"Loading model: {model}")
    mlx_model, tokenizer, config = load_model(model)

    adapter = get_adapter(mlx_model, config)
    moe_indices = adapter.moe_layer_indices()
    n_experts = adapter.num_routed_experts()
    top_k = adapter.num_experts_per_tok()
    model_type = config.get("model_type", "")

    click.echo(f"Model type: {model_type}, MoE layers: {len(moe_indices)}, "
               f"Experts: {n_experts}, top_k: {top_k}")

    # Use DifferentialAccumulator with "harmful"=domain, "benign"=general
    acc = DifferentialAccumulator(num_layers=len(moe_indices), num_experts=n_experts)
    moe_blocks = [adapter.get_moe_block(i) for i in moe_indices]

    for dataset_name, dataset_path, acc_label in [
        ("domain", domain_dataset, "harmful"),
        ("general", general_dataset, "benign"),
    ]:
        click.echo(f"Loading {dataset_name} dataset: {dataset_path}")
        samples = load_dataset(dataset_path, tokenizer, max_tokens, max_samples, text_key)
        click.echo(f"  Loaded {len(samples)} samples")

        install_ream_hooks(moe_blocks, model_type)
        forward = text_forward(mlx_model, config)
        for sample in tqdm(samples, desc=f"Scanning {dataset_name}"):
            tokens = sample.reshape(1, -1)
            forward(tokens)
            mx.eval(mlx_model.parameters())

            captures = collect_ream_data(moe_blocks)
            for block_idx, block_captures in enumerate(captures):
                for layer_input, gate_logits, sel_inds in block_captures:
                    gl_2d = gate_logits.reshape(-1, gate_logits.shape[-1])
                    acc.update_from_gate_logits(block_idx, gl_2d, acc_label)
                    # Real selection captured from the gate, not reconstructed.
                    inds_2d = sel_inds.reshape(-1, sel_inds.shape[-1])
                    acc.update_from_top_k(block_idx, inds_2d, acc_label)

        remove_ream_hooks(moe_blocks)

    click.echo("Computing differential scores...")
    diff_freq, diff_act, composite = compute_differential_scores(acc)
    report = identify_domain_experts(
        diff_freq, diff_act, composite, domain_name, threshold_percentile,
    )

    report.save(output)
    total_domain = sum(len(v) for v in report.domain_experts.values())
    total_general = sum(len(v) for v in report.general_experts.values())
    click.echo(f"Domain report saved to: {output}")
    click.echo(f"  Domain '{domain_name}' experts: {total_domain}, "
               f"General experts: {total_general}")


@main.command("domain-probe")
@click.option("--model", required=True, help="Model path or HuggingFace repo ID.")
@click.option("--domain-questions", required=True,
              help="JSONL of domain Q&A (e.g. data/probes/solidity.jsonl).")
@click.option("--general-questions", required=True,
              help="JSONL of contrast Q&A (e.g. data/probes/general.jsonl).")
@click.option("--output", required=True, help="Output path for probe_report.json.")
@click.option("--saliency-output", default=None,
              help="Optional .npz of domain answer routing, for prune --saliency.")
@click.option("--saliency-weighting", default="question",
              type=click.Choice(["question", "token"]),
              help="Weight each question equally, or each answer token equally.")
@click.option("--answers-output", default=None,
              help="Optional JSONL of generated answers (generate mode).")
@click.option("--domain-name", default=None,
              help="Domain label. Default: the stem of --domain-questions.")
@click.option("--answer-mode", default="teacher",
              type=click.Choice(["teacher", "generate"]),
              help="'teacher' scores reference answers; 'generate' has the model answer.")
@click.option("--max-questions", default=0, type=int,
              help="Subsample this many questions per set (0 = all).")
@click.option("--max-answer-tokens", default=128, type=int,
              help="Cap on reference-answer tokens and generated tokens.")
@click.option("--threshold-percentile", default=90.0, type=float,
              help="Percentile threshold for classifying domain experts.")
@click.option("--min-coverage", default=0.0, type=float,
              help="Drop domain experts used in fewer than this fraction of questions.")
@click.option("--verify-top", default=32, type=int,
              help="Knockout-verify this many top experts (0 disables).")
@click.option("--verify-questions", default=0, type=int,
              help="Use this many domain questions for verification (0 = all).")
@click.option("--min-delta", default=0.02, type=float,
              help="Minimum mean NLL increase (nats/token) to call an expert verified.")
@click.option("--min-valid-fraction", default=0.9, type=float,
              help="Below this fraction of usable question pairs, a result is inconclusive.")
@click.option("--bootstrap", default=1000, type=int,
              help="Bootstrap resamples for the delta confidence interval (0 disables).")
@click.option("--verify-prune", default=0, type=int,
              help="Also verify the exact set `prune` would remove at this n_prune.")
@click.option("--verify-metric", default="freq",
              type=click.Choice(["reap", "ean", "freq", "weighted_freq"]),
              help="Saliency metric for the prune-set check.")
@click.option("--verify-strategy", default="bottom",
              type=click.Choice(["bottom", "strided"]),
              help="Selection strategy for the prune-set check.")
@click.option("--verify-model-wide", is_flag=True,
              help="Prune-set check uses model-wide column selection.")
@click.option("--verify-min-experts-per-layer", default=1, type=int,
              help="Minimum experts per layer for a model-wide prune-set check.")
@click.option("--verify-protect-domain/--no-verify-protect-domain", default=True,
              help="Protect domain experts in the prune-set check, as prune --domain-mode protect does.")
@click.option("--mask-value", default=-1e9, type=float,
              help="Additive selection-score mask used for knockouts.")
@click.option("--chat-template-args", default=None,
              help="JSON object of extra chat-template kwargs, e.g. "
                   "--chat-template-args '{\"enable_thinking\":false}'.")
@click.option("--system", default=None, help="Default system prompt for every question.")
@click.option("--seed", default=None, type=int, help="Random seed.")
@click.option("--show-questions", is_flag=True, help="Echo each question and answer.")
def domain_probe(model, domain_questions, general_questions, output, saliency_output,
                 saliency_weighting, answers_output, domain_name, answer_mode,
                 max_questions, max_answer_tokens, threshold_percentile, min_coverage,
                 verify_top, verify_questions, min_delta, min_valid_fraction, bootstrap,
                 verify_prune, verify_metric, verify_strategy, verify_model_wide,
                 verify_min_experts_per_layer, verify_protect_domain, mask_value,
                 chat_template_args, system, seed, show_questions):
    """Score expert relevance to a domain by asking the model questions.

    Traces which experts the model routes to while answering domain questions
    versus general ones, then verifies the top candidates by masking them out of
    the real router and measuring how much the answers degrade. The report feeds
    `prune --domain-map`, `amplify` and `serve --domain-map` unchanged.
    """
    import json as _json
    import random
    from pathlib import Path

    import numpy as np
    from tqdm import tqdm

    from .adapters import get_adapter
    from .domain import identify_domain_experts
    from .loader import load_model, text_forward, is_vision_model
    from .observer import install_hooks, remove_hooks
    from .probe import (
        DOMAIN, GENERAL, ProbeReport, ProbeStats,
        apply_coverage_filter, compute_probe_scores, load_probe_set,
        run_knockout, run_prune_check, select_knockout_candidates,
        trace_question_set,
    )
    from .pruner import build_keep_map

    if seed is not None:
        random.seed(seed)

    parsed_chat_template_args = {}
    if chat_template_args:
        try:
            parsed_chat_template_args = _json.loads(chat_template_args)
            if not isinstance(parsed_chat_template_args, dict):
                raise ValueError("expected a JSON object")
        except Exception as e:
            raise click.BadParameter(
                f"--chat-template-args must be a JSON object: {e}"
            )

    if domain_name is None:
        domain_name = Path(domain_questions).stem

    expanded_model = os.path.expanduser(model)
    if os.path.exists(expanded_model):
        model = expanded_model
    click.echo(f"Loading model: {model}")
    mlx_model, tokenizer, config = load_model(model)

    if answer_mode == "generate" and is_vision_model(config):
        raise click.ClickException(
            f"--answer-mode generate drives mlx-lm's text generation loop, which "
            f"cannot run the vision-language model '{config.get('model_type')}'. "
            f"Use --answer-mode teacher, which scores reference answers with a "
            f"token-only forward pass."
        )

    adapter = get_adapter(mlx_model, config)
    moe_indices = adapter.moe_layer_indices()
    n_experts = adapter.num_routed_experts()
    top_k = adapter.num_experts_per_tok()
    model_type = config.get("model_type", "")
    moe_blocks = [adapter.get_moe_block(i) for i in moe_indices]
    forward = text_forward(mlx_model, config)

    click.echo(f"Model type: {model_type}, MoE layers: {len(moe_indices)}, "
               f"Experts: {n_experts}, top_k: {top_k}")

    stats = ProbeStats(num_layers=len(moe_indices), num_experts=n_experts)
    examples = {}
    skipped = {}
    generated_rows = []

    for label, path in ((DOMAIN, domain_questions), (GENERAL, general_questions)):
        questions = load_probe_set(path, max_questions)
        click.echo(f"Loaded {len(questions)} {label} questions from {path}")

        def _echo(index, question, answer, _label=label):
            if show_questions:
                click.echo(f"  [{_label} {index}] {question}\n    -> {answer}")
            if _label == DOMAIN and answer_mode == "generate":
                generated_rows.append({
                    "question": question,
                    "answer": answer,
                    "tags": questions[index].tags,
                })

        bar = tqdm(total=len(questions), desc=f"Probing {label}")
        install_hooks(moe_blocks, model_type)
        try:
            examples[label], skipped[label] = trace_question_set(
                forward, mlx_model, tokenizer, questions, label, stats, moe_blocks,
                num_experts=n_experts,
                answer_mode=answer_mode,
                max_answer_tokens=max_answer_tokens,
                chat_template_args=parsed_chat_template_args,
                system=system,
                saliency_weighting=saliency_weighting,
                echo=_echo,
                progress=lambda done, total: bar.update(1),
            )
        finally:
            remove_hooks(moe_blocks)
            bar.close()

        if skipped[label]:
            click.echo(f"  Skipped {len(skipped[label])} {label} questions "
                       f"(first: {skipped[label][0]['reason']})")

    if not examples[DOMAIN]:
        raise click.ClickException(
            "No domain questions produced usable answers. In teacher mode every "
            "question needs a non-empty 'answer' field."
        )

    click.echo("Computing differential scores...")
    diff_freq, diff_weight, composite = compute_probe_scores(stats)
    report = identify_domain_experts(
        diff_freq, diff_weight, composite, domain_name, threshold_percentile,
    )
    domain_coverage = stats.coverage_fraction(DOMAIN)
    domain_experts = apply_coverage_filter(
        report.domain_experts, domain_coverage, min_coverage,
    )

    probe = ProbeReport(
        domain_name=domain_name,
        num_layers=report.num_layers,
        num_experts=report.num_experts,
        threshold_percentile=threshold_percentile,
        differential_freq=diff_freq,
        differential_activation=diff_weight,
        composite_score=composite,
        domain_experts=domain_experts,
        general_experts=report.general_experts,
        answer_mode=answer_mode,
        saliency_weighting=saliency_weighting,
        num_domain_questions=stats.n_questions[DOMAIN],
        num_general_questions=stats.n_questions[GENERAL],
        skipped_questions={k: len(v) for k, v in skipped.items()},
        min_coverage=min_coverage,
        domain_mean_freq=stats.mean_freq(DOMAIN),
        general_mean_freq=stats.mean_freq(GENERAL),
        domain_coverage=domain_coverage,
        general_coverage=stats.coverage_fraction(GENERAL),
    )

    verify_examples = examples[DOMAIN]
    verify_general = examples[GENERAL]
    if verify_questions > 0:
        # Seeded sample, not the first N: the shipped sets are ordered by topic,
        # so a prefix would over-test whatever tag happens to come first.
        vrng = random.Random(seed if seed is not None else 0)
        if len(verify_examples) > verify_questions:
            verify_examples = vrng.sample(verify_examples, verify_questions)
        if len(verify_general) > verify_questions:
            verify_general = vrng.sample(verify_general, verify_questions)

    candidates = select_knockout_candidates(composite, domain_experts, verify_top)
    if candidates:
        click.echo(f"Knocking out {len(candidates)} candidate experts over "
                   f"{len(verify_examples)} questions...")
        bar = tqdm(total=len(candidates), desc="Knockout")
        try:
            knockout = run_knockout(
                forward, verify_examples, moe_blocks, model_type,
                num_experts=n_experts, top_k=top_k, candidates=candidates,
                composite=composite, coverage=domain_coverage,
                mask_value=mask_value, min_delta=min_delta,
                min_valid_fraction=min_valid_fraction, n_boot=bootstrap,
                seed=seed or 0,
                progress=lambda done, total: bar.update(1),
            )
        finally:
            bar.close()

        delta = np.zeros_like(composite)
        verified = {}
        for entry in knockout.per_expert:
            delta[entry["layer"], entry["expert"]] = entry["mean_delta"]
            if entry["status"] == "verified":
                verified.setdefault(entry["layer"], []).append(entry["expert"])
        probe.knockout = {
            "backend": "gate_selection_mask",
            "mask_value": mask_value,
            "num_questions": knockout.num_questions,
            "dropped_nonfinite_baseline": knockout.dropped_nonfinite_baseline,
            "plain_baseline_nll": knockout.plain_baseline_nll,
            "baseline_nll": knockout.baseline_nll,
            "min_delta": min_delta,
            "min_valid_fraction": min_valid_fraction,
            "bootstrap": bootstrap,
            "per_expert": knockout.per_expert,
        }
        probe.knockout_delta = delta
        probe.verified_domain_experts = verified

        click.echo(f"  Baseline NLL: {knockout.baseline_nll:.4f} "
                   f"(unmasked {knockout.plain_baseline_nll:.4f})")
        for entry in sorted(knockout.per_expert,
                            key=lambda e: -abs(e["mean_delta"]))[:5]:
            click.echo(f"    L{entry['layer']} E{entry['expert']}: "
                       f"delta={entry['mean_delta']:+.4f} "
                       f"[{entry['ci_low']:+.4f}, {entry['ci_high']:+.4f}] "
                       f"{entry['status']}")
        n_verified = sum(len(v) for v in verified.values())
        click.echo(f"  Verified {n_verified} of {len(candidates)} candidates")

    if verify_prune > 0:
        scores = stats.saliency.compute_scores(verify_metric)
        protected = None
        if verify_protect_domain:
            protected = {
                k: np.array(v, dtype=np.intp) for k, v in domain_experts.items()
            }
        keep_map = build_keep_map(
            scores, verify_prune,
            strategy=verify_strategy,
            model_wide=verify_model_wide,
            protected_experts=protected,
            min_experts_per_layer=verify_min_experts_per_layer,
        )
        click.echo(f"Verifying the prune set (n_prune={verify_prune}, "
                   f"metric={verify_metric}, strategy={verify_strategy})...")
        check = run_prune_check(
            forward, verify_examples, verify_general, moe_blocks, model_type,
            num_experts=n_experts, top_k=top_k, keep_map=keep_map,
            mask_value=mask_value, min_delta=min_delta,
            min_valid_fraction=min_valid_fraction, n_boot=bootstrap, seed=seed or 0,
        )
        check.update({
            "source": "build_keep_map",
            "n_prune": verify_prune,
            "metric": verify_metric,
            "strategy": verify_strategy,
            "model_wide": verify_model_wide,
            "protect_domain": verify_protect_domain,
        })
        probe.prune_check = check
        click.echo(f"  Masked {check['masked_pairs']} expert-layer pairs")
        for label in (DOMAIN, GENERAL):
            if check.get(label):
                click.echo(f"    {label}: delta={check[label]['mean_delta']:+.4f} "
                           f"nats/token ({check[label]['interpretation']})")
        click.echo("  Note: this masks experts in the original router. Confirm the "
                   "real pruned checkpoint with smoke-test.")

    probe.save(output)
    click.echo(f"Probe report saved to: {output}")
    total_domain = sum(len(v) for v in domain_experts.values())
    total_general = sum(len(v) for v in report.general_experts.values())
    click.echo(f"  Domain '{domain_name}' experts: {total_domain}, "
               f"General experts: {total_general}")

    if saliency_output:
        stats.saliency.save(saliency_output)
        click.echo(f"Domain saliency saved to: {saliency_output}")

    if answers_output and generated_rows:
        with open(answers_output, "w") as f:
            for row in generated_rows:
                f.write(_json.dumps(row) + "\n")
        click.echo(f"Generated answers saved to: {answers_output}")


@main.command("refusal-probe")
@click.option("--model", required=True, help="Model path or HuggingFace repo ID.")
@click.option("--questions", required=True,
              help="JSONL of questions to probe for refusals (e.g. data/probes/security.jsonl).")
@click.option("--output", required=True, help="Output path for refusal_report.json.")
@click.option("--saliency-output", default=None,
              help="Optional .npz of refused-answer routing, for prune --saliency.")
@click.option("--answers-output", default=None,
              help="Optional JSONL of generated answers with their outcome classification.")
@click.option("--refusal-markers", default=None,
              help="Optional file of extra refusal phrases, one per line.")
@click.option("--domain-name", default="refusal",
              help="Report label (this set becomes domain_experts for downstream tools).")
@click.option("--max-questions", default=0, type=int,
              help="Subsample this many questions (0 = all).")
@click.option("--max-answer-tokens", default=256, type=int,
              help="Max tokens to generate per question.")
@click.option("--threshold-percentile", default=90.0, type=float,
              help="Percentile threshold for classifying refusal experts.")
@click.option("--min-coverage", default=0.0, type=float,
              help="Drop refusal experts used in fewer than this fraction of refused questions.")
@click.option("--verify-top", default=16, type=int,
              help="Regeneration-verify this many top refusal experts (0 disables).")
@click.option("--verify-questions", default=0, type=int,
              help="Use this many refused questions for verification (0 = all).")
@click.option("--min-flip-rate", default=0.5, type=float,
              help="Minimum refused->answered flip rate to call a refusal expert verified.")
@click.option("--bootstrap", default=1000, type=int,
              help="Bootstrap resamples for the flip-rate confidence interval (0 disables).")
@click.option("--stratify-tags/--no-stratify-tags", default=True,
              help="Score refused-vs-answered within each tag to control topic "
                   "confounding (falls back to global if no tag has both).")
@click.option("--mask-value", default=-1e9, type=float,
              help="Additive selection-score mask used for knockouts.")
@click.option("--chat-template-args", default=None,
              help="JSON object of extra chat-template kwargs, e.g. "
                   "--chat-template-args '{\"enable_thinking\":false}'.")
@click.option("--system", default=None, help="Default system prompt for every question.")
@click.option("--seed", default=None, type=int, help="Random seed.")
@click.option("--show-questions", is_flag=True, help="Echo each question and its outcome.")
def refusal_probe(model, questions, output, saliency_output, answers_output,
                  refusal_markers, domain_name, max_questions, max_answer_tokens,
                  threshold_percentile, min_coverage, verify_top, verify_questions,
                  min_flip_rate, bootstrap, stratify_tags, mask_value,
                  chat_template_args, system, seed, show_questions):
    """Find the experts that implement the model's refusal guardrails.

    Generates an answer to each question, classifies it as answered / refused /
    partial, and contrasts routing on refused questions against answered ones.
    Candidates are verified by masking each expert and regenerating the refused
    questions: a refusal expert is confirmed when removing it turns refusals
    into answers. The report feeds prune / amplify / serve like a domain report.
    """
    import json as _json
    import random
    from pathlib import Path

    import numpy as np
    from tqdm import tqdm

    from .adapters import get_adapter
    from .domain import identify_domain_experts
    from .loader import load_model, is_vision_model
    from .observer import install_hooks, remove_hooks
    from .probe import (
        DOMAIN, GENERAL, ProbeStats,
        apply_coverage_filter, compute_probe_scores, load_probe_set,
        select_knockout_candidates,
    )
    from .refusal import (
        ANSWERED, REFUSED, PARTIAL, RefusalReport,
        run_refusal_knockout, stratified_probe_scores, trace_refusals,
    )

    if seed is not None:
        random.seed(seed)

    parsed_chat_template_args = {}
    if chat_template_args:
        try:
            parsed_chat_template_args = _json.loads(chat_template_args)
            if not isinstance(parsed_chat_template_args, dict):
                raise ValueError("expected a JSON object")
        except Exception as e:
            raise click.BadParameter(f"--chat-template-args must be a JSON object: {e}")

    extra_markers = None
    if refusal_markers:
        with open(refusal_markers) as f:
            extra_markers = [line.strip() for line in f if line.strip()]

    expanded_model = os.path.expanduser(model)
    if os.path.exists(expanded_model):
        model = expanded_model
    click.echo(f"Loading model: {model}")
    mlx_model, tokenizer, config = load_model(model)

    adapter = get_adapter(mlx_model, config)
    moe_indices = adapter.moe_layer_indices()
    n_experts = adapter.num_routed_experts()
    top_k = adapter.num_experts_per_tok()
    model_type = config.get("model_type", "")
    moe_blocks = [adapter.get_moe_block(i) for i in moe_indices]

    click.echo(f"Model type: {model_type}, MoE layers: {len(moe_indices)}, "
               f"Experts: {n_experts}, top_k: {top_k}"
               + (" (VLM: generating on the language stack)" if is_vision_model(config) else ""))

    qs = load_probe_set(questions, max_questions)
    click.echo(f"Loaded {len(qs)} questions from {questions}")

    stats = ProbeStats(num_layers=len(moe_indices), num_experts=n_experts)

    def _echo(index, question, outcome, answer):
        if show_questions:
            click.echo(f"  [{outcome:8}] {question}")

    bar = tqdm(total=len(qs), desc="Probing")
    install_hooks(moe_blocks, model_type)
    try:
        examples, outcomes, skipped, records = trace_refusals(
            mlx_model, tokenizer, config, qs, stats, moe_blocks,
            num_experts=n_experts,
            max_answer_tokens=max_answer_tokens,
            chat_template_args=parsed_chat_template_args,
            system=system,
            extra_markers=extra_markers,
            echo=_echo,
            progress=lambda done, total: bar.update(1),
        )
    finally:
        remove_hooks(moe_blocks)
        bar.close()

    n_answered = sum(1 for o in outcomes if o["outcome"] == ANSWERED)
    n_refused = sum(1 for o in outcomes if o["outcome"] == REFUSED)
    n_partial = sum(1 for o in outcomes if o["outcome"] == PARTIAL)
    click.echo(f"Outcomes: {n_answered} answered, {n_refused} refused, "
               f"{n_partial} partial ({len(skipped)} skipped)")

    if n_refused == 0:
        click.echo("The model refused none of these questions, so there is no "
                   "refusal signal to isolate. Try a set it declines, a stricter "
                   "system prompt, or --refusal-markers to widen detection.")
    if n_refused == 0 or n_answered == 0:
        raise click.ClickException(
            "Need both refused and answered questions to contrast; got "
            f"{n_refused} refused and {n_answered} answered."
        )

    scoring_method = "global"
    stratified_tags = []
    if stratify_tags:
        strat = stratified_probe_scores(records, len(moe_indices), n_experts)
        if strat is not None:
            diff_freq, diff_weight, composite, stratified_tags = strat
            scoring_method = "tag_stratified"
            click.echo(f"Computing tag-stratified scores over {len(stratified_tags)} "
                       f"tags with both outcomes (controls topic confounding)...")
        else:
            click.echo("No tag had both a refused and an answered question; "
                       "falling back to a global contrast (topic-confounded).")
    if scoring_method == "global":
        click.echo("Computing global differential scores (refused vs answered)...")
        diff_freq, diff_weight, composite = compute_probe_scores(stats)

    base = identify_domain_experts(
        diff_freq, diff_weight, composite, domain_name, threshold_percentile,
    )
    refused_coverage = stats.coverage_fraction(DOMAIN)
    candidate_refusal_experts = apply_coverage_filter(
        base.domain_experts, refused_coverage, min_coverage,
    )

    report = RefusalReport(
        domain_name=domain_name,
        num_layers=base.num_layers,
        num_experts=base.num_experts,
        threshold_percentile=threshold_percentile,
        differential_freq=diff_freq,
        differential_activation=diff_weight,
        composite_score=composite,
        # domain_experts defaults to the candidates; replaced by the verified
        # set below when a knockout runs, so downstream never acts on unverified.
        domain_experts=candidate_refusal_experts,
        general_experts=base.general_experts,
        answer_mode="generate",
        min_coverage=min_coverage,
        domain_coverage=refused_coverage,
        general_coverage=stats.coverage_fraction(GENERAL),
        objective="refusal",
        scoring_method=scoring_method,
        num_answered=n_answered,
        num_refused=n_refused,
        num_partial=n_partial,
        candidate_refusal_experts=candidate_refusal_experts,
        stratified_tags=stratified_tags,
        per_question_outcome=outcomes,
        skipped_questions={"probed": len(skipped)},
    )

    refused_examples = examples[DOMAIN]
    if verify_questions > 0 and len(refused_examples) > verify_questions:
        vrng = random.Random(seed if seed is not None else 0)
        refused_examples = vrng.sample(refused_examples, verify_questions)

    candidates = select_knockout_candidates(composite, candidate_refusal_experts, verify_top)
    if candidates and refused_examples:
        click.echo(f"Regeneration-verifying {len(candidates)} candidates over "
                   f"{len(refused_examples)} refused questions...")
        bar = tqdm(total=len(candidates), desc="Knockout")
        try:
            per_expert = run_refusal_knockout(
                mlx_model, tokenizer, config, refused_examples, moe_blocks, model_type,
                num_experts=n_experts, top_k=top_k, candidates=candidates,
                max_answer_tokens=max_answer_tokens,
                chat_template_args=parsed_chat_template_args, system=system,
                extra_markers=extra_markers, mask_value=mask_value,
                min_flip_rate=min_flip_rate, n_boot=bootstrap, seed=seed or 0,
                composite=composite, coverage=refused_coverage,
                progress=lambda done, total: bar.update(1),
            )
        finally:
            bar.close()

        delta = np.zeros_like(composite)
        verified = {}
        for entry in per_expert:
            delta[entry["layer"], entry["expert"]] = entry["flip_rate"]
            if entry["status"] == "verified":
                verified.setdefault(entry["layer"], []).append(entry["expert"])
        report.knockout = {
            "backend": "regenerate_and_classify",
            "mask_value": mask_value,
            "num_refused_questions": len(refused_examples),
            "min_flip_rate": min_flip_rate,
            "bootstrap": bootstrap,
            "per_expert": per_expert,
        }
        report.knockout_delta = delta
        # Downstream reads domain_experts; default it to the VERIFIED set so a
        # prune/steer/deactivate never acts on a merely-correlated expert.
        report.verified_refusal_experts = verified
        report.verified_domain_experts = verified
        report.domain_experts = verified
        report.refusal_experts_verified = True

        for entry in sorted(per_expert, key=lambda e: -e["flip_rate"])[:5]:
            click.echo(f"    L{entry['layer']} E{entry['expert']}: "
                       f"flip_rate={entry['flip_rate']:.2f} "
                       f"[{entry['ci_low']:.2f}, {entry['ci_high']:.2f}] {entry['status']}")
        n_verified = sum(len(v) for v in verified.values())
        click.echo(f"  Verified {n_verified} of {len(candidates)} candidates")
    else:
        click.echo("No knockout ran (--verify-top 0 or no refused questions); "
                   "the report's domain_experts are UNVERIFIED candidates.")

    report.save(output)
    click.echo(f"Refusal report saved to: {output}")
    n_candidates = sum(len(v) for v in candidate_refusal_experts.values())
    n_verified_total = sum(len(v) for v in report.verified_refusal_experts.values())
    click.echo(f"  Candidate refusal experts: {n_candidates}; "
               f"verified: {n_verified_total}"
               + ("" if report.refusal_experts_verified
                  else " (unverified — run with --verify-top to confirm)"))

    if saliency_output:
        stats.saliency.save(saliency_output)
        click.echo(f"Refused-answer saliency saved to: {saliency_output}")

    if answers_output:
        with open(answers_output, "w") as f:
            for o in outcomes:
                f.write(_json.dumps(o) + "\n")
        click.echo(f"Answers and outcomes saved to: {answers_output}")


@main.command()
@click.option("--model", required=True, help="Model path or HuggingFace repo ID.")
@click.option("--domain-map", required=True, help="Path to domain_report.json.")
@click.option("--output", required=True, help="Output directory for amplified model.")
@click.option("--scale", default=1.0, type=float, help="Amplification strength.")
@click.option("--threshold", default=0.0, type=float, help="Min composite score to amplify.")
def amplify(model, domain_map, output, scale, threshold):
    """Permanently amplify domain expert routing by modifying gate weights.

    Loads a domain report from domain-scan, computes amplification biases,
    and modifies gate parameters so domain experts are favored natively.
    The amplified model works with standard mlx_lm.load() — no hooks needed.
    """
    from .loader import load_model

    from .adapters import get_adapter
    from .domain import DomainReport, compute_amplification_biases, amplify_gate_weights
    from .save import save_amplified_model

    expanded_model = os.path.expanduser(model)
    if os.path.exists(expanded_model):
        model = expanded_model
    click.echo(f"Loading model: {model}")
    mlx_model, tokenizer, config = load_model(model)

    adapter = get_adapter(mlx_model, config)
    moe_indices = adapter.moe_layer_indices()
    model_type = config.get("model_type", "")

    click.echo(f"Loading domain map: {domain_map}")
    report = DomainReport.load(domain_map)
    total_domain = sum(len(v) for v in report.domain_experts.values())
    click.echo(f"  Domain '{report.domain_name}': {total_domain} expert-layer pairs")

    biases = compute_amplification_biases(report, scale=scale, threshold=threshold)
    click.echo(f"Computed amplification biases for {len(biases)} layers (scale={scale}, threshold={threshold})")

    moe_blocks = [adapter.get_moe_block(i) for i in moe_indices]
    amplify_gate_weights(moe_blocks, model_type, biases)
    click.echo("Gate weights amplified.")

    click.echo(f"Saving amplified model to: {output}")
    save_amplified_model(
        mlx_model, tokenizer, config, output,
        report.domain_name, scale, threshold, biases,
    )
    click.echo("Done!")


@main.command("stats-diff")
@click.option("--file1", required=True, help="Path to first saliency .npz file.")
@click.option("--file2", required=True, help="Path to second saliency .npz file.")
@click.option("--metric", default="reap", type=click.Choice(["reap", "ean", "freq", "weighted_freq"]),
              help="Saliency metric to compare.")
@click.option("--output", default=None, help="Optional path to save diff report as JSON.")
def stats_diff(file1, file2, metric, output):
    """Compare two collected saliency files and show differences.

    Computes the difference between two SaliencyAccumulator files, showing
    which experts have higher or lower saliency in each file.
    """
    from .saliency import SaliencyAccumulator
    from .stats_ops import compute_diff_stats, save_diff_report

    click.echo(f"Loading file1: {file1}")
    acc1 = SaliencyAccumulator.load(file1)
    click.echo(f"  Layers: {acc1.num_layers}, Experts: {acc1.num_experts}")

    click.echo(f"Loading file2: {file2}")
    acc2 = SaliencyAccumulator.load(file2)
    click.echo(f"  Layers: {acc2.num_layers}, Experts: {acc2.num_experts}")

    if acc1.num_layers != acc2.num_layers or acc1.num_experts != acc2.num_experts:
        click.echo("Error: Files have incompatible dimensions", err=True)
        return

    click.echo(f"\nComputing differences (metric={metric})...")
    report = compute_diff_stats(acc1, acc2, metric)

    # Display summary
    click.echo("\n" + "="*60)
    click.echo("DIFFERENCES SUMMARY")
    click.echo("="*60)
    click.echo(f"Metric: {report['metric']}")
    click.echo(f"Dimensions: {report['num_layers']} layers × {report['num_experts']} experts")
    click.echo(f"\nDifference statistics:")
    click.echo(f"  Mean:   {report['diff_mean']:.4f}")
    click.echo(f"  Std:    {report['diff_std']:.4f}")
    click.echo(f"  Min:    {report['diff_min']:.4f}")
    click.echo(f"  Max:    {report['diff_max']:.4f}")
    click.echo(f"  AbsMax: {report['diff_abs_max']:.4f}")
    click.echo(f"\nDistribution:")
    click.echo(f"  Positive (file1 > file2): {report['positive_count']} experts")
    click.echo(f"  Negative (file2 > file1): {report['negative_count']} experts")
    click.echo(f"  Zero (equal):              {report['zero_count']} experts")

    # Show top differences
    click.echo(f"\nTop 10 experts where file1 > file2:")
    for i, entry in enumerate(report['top_positive'], 1):
        click.echo(f"  {i}. Layer {entry['layer_idx']}, Expert {entry['expert_idx']}: {entry['diff_value']:.4f}")

    click.echo(f"\nTop 10 experts where file2 > file1:")
    for i, entry in enumerate(report['top_negative'], 1):
        click.echo(f"  {i}. Layer {entry['layer_idx']}, Expert {entry['expert_idx']}: {entry['diff_value']:.4f}")

    # Save report if requested
    if output:
        save_diff_report(report, output)
        click.echo(f"\nDiff report saved to: {output}")


@main.command("stats-merge")
@click.option("--files", required=True, multiple=True, help="Paths to saliency .npz files to merge.")
@click.option("--output", required=True, help="Output path for merged .npz file.")
@click.option("--metric", default="reap", type=click.Choice(["reap", "ean", "freq", "weighted_freq"]),
              help="Metric to use for ranking experts (default: reap).")
def stats_merge(files, output, metric):
    """Merge multiple collected saliency files using rank-based aggregation.

    For each input file, experts are ranked per-layer based on the specified
    metric. Ranks are then summed across all files. Lower summed rank indicates
    higher importance (expert consistently ranked high across datasets).

    This approach normalizes data across different datasets, ensuring each
    dataset contributes equally regardless of sample count or scale differences.
    """
    from .saliency import SaliencyAccumulator
    from .stats_ops import merge_saliency

    if len(files) < 2:
        click.echo("Error: At least 2 files are required for merging", err=True)
        return

    click.echo(f"Merging {len(files)} files using rank-based aggregation...")
    click.echo(f"Metric for ranking: {metric}")
    for i, f in enumerate(files, 1):
        click.echo(f"  {i}. {f}")

    try:
        merged = merge_saliency(list(files), metric=metric)
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        return

    click.echo(f"\nMerged accumulator:")
    click.echo(f"  Layers: {merged.num_layers}")
    click.echo(f"  Experts: {merged.num_experts}")
    click.echo(f"  Summed ranks range: [{merged.freq.min():.0f}, {merged.freq.max():.0f}]")
    click.echo(f"  Note: Lower rank sum = higher importance")

    merged.save(output)
    click.echo(f"\nMerged stats saved to: {output}")


@main.command("stats-purge")
@click.option("--input", required=True, help="Path to input saliency .npz file.")
@click.option("--output", required=True, help="Output path for purged .npz file.")
@click.option("--min-freq", default=None, type=int,
              help="Minimum activation frequency to keep (default: no filter).")
@click.option("--min-count", default=None, type=int,
              help="Minimum reap_count to keep (default: no filter).")
@click.option("--max-norm", default=None, type=float,
              help="Maximum activation norm (warning: only reports, doesn't cap).")
def stats_purge(input, output, min_freq, min_count, max_norm):
    """Purge/filter low-activation or outlying data from a saliency file.

    Removes data for experts that don't meet minimum activation criteria.
    This can help focus pruning on experts with meaningful activation patterns.
    """
    from .saliency import SaliencyAccumulator
    from .stats_ops import purge_saliency

    if min_freq is None and min_count is None and max_norm is None:
        click.echo("Error: At least one filter option must be specified", err=True)
        return

    click.echo(f"Loading input: {input}")
    acc = SaliencyAccumulator.load(input)
    click.echo(f"  Layers: {acc.num_layers}, Experts: {acc.num_experts}")

    click.echo(f"\nApplying filters:")
    if min_freq is not None:
        click.echo(f"  min_freq: {min_freq}")
    if min_count is not None:
        click.echo(f"  min_count: {min_count}")
    if max_norm is not None:
        click.echo(f"  max_norm: {max_norm}")

    purged, stats = purge_saliency(
        acc,
        min_freq=min_freq,
        min_count=min_count,
        max_norm=max_norm,
        keep_metadata=True,
    )

    click.echo(f"\nPurge statistics:")
    click.echo(f"  Total expert-layer pairs: {stats['total_experts']}")
    click.echo(f"  Purged by freq < {min_freq}: {stats['purged_by_freq']}")
    click.echo(f"  Purged by count < {min_count}: {stats['purged_by_count']}")
    click.echo(f"  Capped by norm > {max_norm}: {stats['capped_by_norm']}")
    click.echo(f"  Total purged: {stats['total_purged']}")
    click.echo(f"  Kept: {stats['kept_count']}")

    purged.save(output)
    click.echo(f"\nPurged stats saved to: {output}")


@main.command()
@click.option("--server-url", default="http://127.0.0.1:8080",
              help="URL of the running REAP server.")
@click.option("--host", default="127.0.0.1", help="Frontend bind address.")
@click.option("--port", default=7860, type=int, help="Frontend port.")
@click.option("--share", is_flag=True, help="Create a public Gradio share link.")
def ui(server_url, host, port, share):
    """Launch the web dashboard for monitoring and controlling the REAP server.

    Requires a running REAP server (start with `mlx-fun serve`).
    Install the UI dependencies with: pip install 'mlx-fun[ui]'
    """
    from .frontend import launch_frontend

    click.echo(f"Connecting to REAP server at: {server_url}")
    click.echo(f"Starting dashboard at: http://{host}:{port}")
    launch_frontend(
        server_url=server_url,
        host=host,
        port=port,
        share=share,
    )


@main.command("convert-nvfp4")
@click.option("--model", required=True,
              help="HuggingFace repo ID or local path to NVIDIA NVFP4 checkpoint.")
@click.option("--output", required=True,
              help="Output directory for MLX checkpoint.")
@click.option("--mode", default="nvfp4",
              type=click.Choice(["nvfp4", "dequant"]),
              help="Output mode: 'nvfp4' preserves native FP4 codes, "
                   "'dequant' converts everything to bfloat16.")
def convert_nvfp4_cmd(model, output, mode):
    """Convert NVIDIA NVFP4 (modelopt) checkpoint to MLX-native format.

    Repacks natively-trained NVFP4 weights into MLX's NVFP4 format,
    preserving the trained FP4 weight codes. FP8 layers (Mamba, shared
    experts) are dequantized to bfloat16.

    \b
    Examples:
      # Convert keeping native NVFP4 (recommended for QAT models)
      mlx-fun convert-nvfp4 \\
          --model nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \\
          --output ./nemotron-120b-mlx

      # Convert to full bfloat16 (no quantization)
      mlx-fun convert-nvfp4 \\
          --model nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \\
          --output ./nemotron-120b-bf16 --mode dequant
    """
    import logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from .convert_nvfp4 import convert_nvfp4

    click.echo(f"Converting NVIDIA NVFP4 checkpoint: {model}")
    click.echo(f"Output mode: {mode}")
    click.echo(f"Output path: {output}")

    convert_nvfp4(model, output, output_mode=mode)

    click.echo("Conversion complete!")


if __name__ == "__main__":
    main()
