"""CLI for MLX-FUN: collect saliency, prune experts, smoke-test."""

import os
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
    from mlx_lm import load as mlx_load
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
        mlx_model, tokenizer, config = mlx_load(model, return_config=True)
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
    for sample in tqdm(samples, desc="Calibrating"):
        # Run forward pass: (1, seq_len)
        tokens = sample.reshape(1, -1)
        mlx_model(tokens)
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
def prune(model, saliency, expert_list, output, n_prune, metric, strategy, model_wide, min_experts_per_layer,
          safety_map, safety_mode, domain_map, domain_mode):
    """Prune experts from model based on saliency statistics or expert list.
    
    Two modes of operation:
    
    1. Using saliency file (traditional):
       mlx-fun prune --model ./model --saliency stats.npz --n-prune 8 --output ./pruned
    
    2. Using expert list from frontend (new):
       mlx-fun prune --model ./model --expert-list filtered_experts.json --output ./pruned
    """
    from mlx_lm import load as mlx_load

    from .adapters import get_adapter
    from .pruner import (
        select_experts_to_keep, select_experts_to_keep_strided,
        select_experts_to_keep_model_wide,
        prune_model, load_safety_constraints, load_domain_constraints,
        load_expert_list,
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

    # Expand user path and validate if it's a local path
    expanded_model = os.path.expanduser(model)
    if os.path.exists(expanded_model):
        model = expanded_model
        click.echo(f"Loading model from local path: {model}")
    else:
        click.echo(f"Loading model: {model}")

    try:
        mlx_model, tokenizer, config = mlx_load(model, return_config=True)
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
        if model_wide:
            click.echo(f"Selecting experts to prune (model-wide: {n_prune} total, metric={metric})")
            keep_map = select_experts_to_keep_model_wide(
                scores, n_prune,
                protected_experts=protected_experts,
                targeted_experts=targeted_experts,
                min_experts_per_layer=min_experts_per_layer,
            )
            # Calculate total pruned and per-layer distribution
            total_pruned = sum(original_n_experts - len(keep_map[i]) for i in range(len(keep_map)))
            click.echo(f"Model-wide pruning: {total_pruned} experts removed across {num_moe_layers} layers")
        else:
            click.echo(f"Selecting experts to prune (per-layer: {n_prune}/layer, metric={metric}, strategy={strategy})")
            if strategy == "strided":
                keep_map = select_experts_to_keep_strided(
                    scores, n_prune,
                    protected_experts=protected_experts,
                    targeted_experts=targeted_experts,
                )
            else:
                keep_map = select_experts_to_keep(
                    scores, n_prune,
                    protected_experts=protected_experts,
                    targeted_experts=targeted_experts,
                )

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
def merge(model, saliency, expert_list, dataset, output, n_prune, metric, model_wide, min_experts_per_layer,
          similarity_mode, alignment, max_group_size, max_samples, max_tokens,
          max_similarity_tokens, max_alignment_tokens, text_key, seed):
    """Merge experts using REAM (Router-weighted Expert Activation Merging).
    
    Two modes of operation:
    
    1. Using saliency file (traditional):
       mlx-fun merge --model ./model --saliency stats.npz --dataset calib.jsonl --n-prune 8 --output ./merged
    
    2. Using expert list from frontend (new):
       mlx-fun merge --model ./model --expert-list filtered_experts.json --dataset calib.jsonl --output ./merged
    """
    import random
    import mlx.core as mx
    from mlx_lm import load as mlx_load
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

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    from .adapters import get_adapter
    from .data import load_dataset
    from .merger import merge_model, merge_model_with_keep_map
    from .pruner import select_experts_to_keep_model_wide, load_expert_list
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
        mlx_model, tokenizer, config = mlx_load(model, return_config=True)
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
            click.echo(f"Model-wide merge: selecting {n_prune} experts to merge globally...")
            keep_map = select_experts_to_keep_model_wide(
                scores, n_prune,
                min_experts_per_layer=min_experts_per_layer,
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
def smoke_test(model, prompt, max_tokens):
    """Verify generation works with a pruned model."""
    from mlx_lm import load as mlx_load, generate

    # Expand user path and validate if it's a local path
    expanded_model = os.path.expanduser(model)
    if os.path.exists(expanded_model):
        model = expanded_model
        click.echo(f"Loading model from local path: {model}")
    else:
        click.echo(f"Loading model: {model}")
    
    try:
        mlx_model, tokenizer = mlx_load(model)
    except Exception as e:
        if "HFValidationError" in str(type(e).__name__) or "Repo id must be in the form" in str(e):
            click.echo(f"\nError: Model path '{model}' could not be loaded as a local file or HuggingFace repo.", err=True)
            click.echo(f"Please verify:", err=True)
            click.echo(f"  1. The path exists and contains model files (config.json, tokenizer files, etc.)", err=True)
            click.echo(f"  2. If using a HuggingFace repo, ensure the repo ID is correct (format: 'username/repo-name')", err=True)
            raise
        raise

    click.echo(f"Generating with prompt: {prompt!r}")
    result = generate(
        mlx_model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=True,
    )
    click.echo(f"\nGeneration successful ({len(result)} chars)")


@main.command()
@click.option("--model", required=True, help="Model path or HuggingFace repo ID.")
@click.option("--host", default="127.0.0.1", help="Server bind address.")
@click.option("--port", default=8080, type=int, help="Server port.")
@click.option("--mode", default="lightweight", type=click.Choice(["lightweight", "full"]),
              help="Hook mode: 'lightweight' skips activation norms, 'full' computes all metrics.")
@click.option("--auto-save", default=None, help="Path to auto-save stats on shutdown.")
@click.option("--max-tokens", default=512, type=int, help="Default max tokens for generation.")
@click.option("--chat-template", default=None, help="Chat template override.")
@click.option("--safety-map", default=None, help="Path to safety_report.json for steering.")
@click.option("--steering-mode", default=None, type=click.Choice(["safe", "unsafe"]),
              help="Steering mode: 'safe' boosts safety experts, 'unsafe' masks them.")
@click.option("--domain-map", default=None, help="Path to domain_report.json for domain boosting.")
@click.option("--domain-steering-mode", default=None, type=click.Choice(["boost", "suppress"]),
              help="Domain steering: 'boost' activates domain experts, 'suppress' deactivates general.")
def serve(model, host, port, mode, auto_save, max_tokens, chat_template,
          safety_map, steering_mode, domain_map, domain_steering_mode):
    """Serve model with online expert counting and optional steering.

    Starts an OpenAI-compatible server that counts expert activations from
    real traffic. Use /v1/reap/stats to view stats, /v1/reap/save to export.

    With --safety-map, enables SteerMoE-style expert steering. Steering can
    also be configured at runtime via POST /v1/reap/steer.

    With --domain-map, enables domain expert boosting via steering hooks.
    """
    from .server import run_reap_server

    run_reap_server(
        host=host,
        port=port,
        model_path=model,
        mode=mode,
        auto_save=auto_save,
        max_tokens=max_tokens,
        chat_template=chat_template,
        safety_map=safety_map,
        steering_mode=steering_mode,
        domain_map=domain_map,
        domain_steering_mode=domain_steering_mode,
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
    from mlx_lm import load as mlx_load
    from tqdm import tqdm

    if seed is not None:
        random.seed(seed)

    from .adapters import get_adapter
    from .data import load_dataset
    from .ream_hooks import install_ream_hooks, collect_ream_data, remove_ream_hooks
    from .safety import (
        DifferentialAccumulator, compute_differential_scores,
        identify_safety_experts, compute_top_k_from_logits,
    )

    # Load model
    expanded_model = os.path.expanduser(model)
    if os.path.exists(expanded_model):
        model = expanded_model
    click.echo(f"Loading model: {model}")
    mlx_model, tokenizer, config = mlx_load(model, return_config=True)

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
        for sample in tqdm(samples, desc=f"Scanning {dataset_name}"):
            tokens = sample.reshape(1, -1)
            mlx_model(tokens)
            mx.eval(mlx_model.parameters())

            captures = collect_ream_data(moe_blocks)
            for block_idx, block_captures in enumerate(captures):
                for layer_input, gate_logits in block_captures:
                    # Flatten batch and seq dims
                    gl_2d = gate_logits.reshape(-1, gate_logits.shape[-1])
                    acc.update_from_gate_logits(block_idx, gl_2d, dataset_name)
                    # Compute top-k and update frequency
                    top_k_inds = compute_top_k_from_logits(gl_2d, model_type, top_k)
                    acc.update_from_top_k(block_idx, top_k_inds, dataset_name)

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
def steer(model, safety_map, mode, prompt, max_tokens, mask_value, boost_value):
    """Generate text with expert steering based on safety analysis.

    Uses SteerMoE-style gate logit injection to selectively activate or
    deactivate safety-critical experts during inference.
    """
    from mlx_lm import load as mlx_load, generate

    from .adapters import get_adapter
    from .steering import SteeringConfig, install_steering_hooks, remove_steering_hooks

    expanded_model = os.path.expanduser(model)
    if os.path.exists(expanded_model):
        model = expanded_model
    click.echo(f"Loading model: {model}")
    mlx_model, tokenizer, config = mlx_load(model, return_config=True)

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

    click.echo(f"Generating with prompt: {prompt!r}")
    result = generate(
        mlx_model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=True,
    )
    remove_steering_hooks(moe_blocks)
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
    from mlx_lm import load as mlx_load

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
    mlx_model, tokenizer, config = mlx_load(model, return_config=True)
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
    from mlx_lm import load as mlx_load
    from tqdm import tqdm

    if seed is not None:
        random.seed(seed)

    from .adapters import get_adapter
    from .data import load_dataset
    from .domain import identify_domain_experts
    from .ream_hooks import install_ream_hooks, collect_ream_data, remove_ream_hooks
    from .safety import (
        DifferentialAccumulator, compute_differential_scores,
        compute_top_k_from_logits,
    )

    # Load model
    expanded_model = os.path.expanduser(model)
    if os.path.exists(expanded_model):
        model = expanded_model
    click.echo(f"Loading model: {model}")
    mlx_model, tokenizer, config = mlx_load(model, return_config=True)

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
        for sample in tqdm(samples, desc=f"Scanning {dataset_name}"):
            tokens = sample.reshape(1, -1)
            mlx_model(tokens)
            mx.eval(mlx_model.parameters())

            captures = collect_ream_data(moe_blocks)
            for block_idx, block_captures in enumerate(captures):
                for layer_input, gate_logits in block_captures:
                    gl_2d = gate_logits.reshape(-1, gate_logits.shape[-1])
                    acc.update_from_gate_logits(block_idx, gl_2d, acc_label)
                    top_k_inds = compute_top_k_from_logits(gl_2d, model_type, top_k)
                    acc.update_from_top_k(block_idx, top_k_inds, acc_label)

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
    from mlx_lm import load as mlx_load

    from .adapters import get_adapter
    from .domain import DomainReport, compute_amplification_biases, amplify_gate_weights
    from .save import save_amplified_model

    expanded_model = os.path.expanduser(model)
    if os.path.exists(expanded_model):
        model = expanded_model
    click.echo(f"Loading model: {model}")
    mlx_model, tokenizer, config = mlx_load(model, return_config=True)

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


if __name__ == "__main__":
    main()
