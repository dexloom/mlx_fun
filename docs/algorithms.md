# Algorithms

The technical guts: REAP, REAM, saliency math, pruning strategies, and
selection modes.

## The REAP algorithm

Mixture-of-Experts models route each token to a subset of "expert"
sub-networks. Not all experts contribute equally — some are rarely activated
or produce small outputs. REAP quantifies expert importance and prunes the
least useful ones:

1. **Calibrate** — Run domain-specific text through the model. At each MoE
   layer, record which experts are selected, their router weights, and the L2
   norm of each expert's output.

2. **Score** — For each expert, compute a saliency score. The default REAP
   metric is:

   ```
   REAP(expert) = mean( ||expert_output|| * router_weight )
   ```

   averaged over all tokens routed to that expert. Experts with low scores
   contribute little to the final output.

3. **Prune or merge** — Two approaches:
   - **Prune** (`mlx-fun prune`) — Remove the N lowest-scoring experts per
     layer by slicing weight tensors. Two strategies: *bottom* (remove
     lowest) and *strided* (distribute removals evenly).
   - **Merge** (`mlx-fun merge`) — Group all experts around the top-k
     centroids, align neurons via permutation matching, and produce
     saliency-weighted averages. Processes layers sequentially so merged
     weights feed into the next layer.

4. **Save** — Write the compressed model in standard mlx-lm format. It loads
   with `mlx_lm.load()` like any other model.

## Saliency metrics

| Metric | Formula | Description |
|---|---|---|
| `reap` (default) | `mean(activation_norm * router_weight)` | Weighted contribution — the REAP paper metric |
| `ean` | `mean(activation_norm)` | Expert Activation Norm — ignores routing weight |
| `freq` | `count(tokens_routed)` | Raw routing frequency |
| `weighted_freq` | `sum(router_weight)` | Cumulative routing confidence |

Note: `reap` and `ean` require activation-norm capture — only available in
`--mode full`. `freq` and `weighted_freq` work in both `lightweight` and
`full` server modes.

## Pruning strategies

| Strategy | Description |
|---|---|
| `bottom` (default) | Remove the N lowest-scoring experts. Simple and effective — standard REAP behavior. |
| `strided` | Split experts into important (top) and unimportant (bottom) groups, then prune evenly from both using interval-based selection. Preserves more diversity across the saliency spectrum. |

**How strided pruning works:**

Given 160 experts and `--n-prune 40`:

1. Sort experts by saliency. Split into important (top 120) and unimportant
   (bottom 40).
2. Prune 20 from each group (50/50 split).
3. Important group (120 experts): stride = 120/20 = 6, remove every 6th expert.
4. Unimportant group (40 experts): stride = 40/20 = 2, remove every 2nd expert.
5. Result: 120 experts kept, with removals distributed across the full
   saliency range.

This can help preserve routing diversity compared to simply chopping the tail.

## Selection modes: per-layer vs model-wide

By default, pruning and merging select N experts **per layer** independently.
The `--model-wide` flag changes this to select N expert **indices** (columns)
that are removed from **all layers**.

| Mode | Flag | Behavior |
|---|---|---|
| Per-Layer | (default) | Remove N experts from each layer. Total removed = N × num_layers. |
| Model-Wide | `--model-wide` | Remove N expert indices from ALL layers. Same columns removed everywhere. |

**How model-wide selection works:**

1. Sum saliency scores for each expert index across all layers (column-wise sum)
2. Select the N expert indices with lowest total saliency (least important globally)
3. Remove those expert indices from every layer in the model

This results in entire "columns" being blanked out in the expert heatmap
visualization.

**When to use model-wide:**

- When you want consistent expert removal across all layers
- When certain expert indices are consistently less important across the model
- When you want simpler model structure (same experts kept in each layer)

```bash
# Per-layer: removes 16 experts from each of 40 layers = 640 total
mlx-fun prune --model ./model --saliency stats.npz --n-prune 16 --output ./pruned

# Model-wide: removes 40 expert indices from ALL layers
mlx-fun prune --model ./model --saliency stats.npz --n-prune 40 --model-wide --output ./pruned
```

The model-wide mode uses `--min-experts-per-layer` (default: 1) to ensure no
layer loses all its experts.

### Protecting experts with `--ignore-experts`

When using model-wide mode, you can protect specific expert indices from
being pruned using `--ignore-experts`:

```bash
# Model-wide prune 50 experts, but keep experts 0, 1, 2, and 250-255
mlx-fun prune --model ./model --saliency stats.npz --n-prune 50 --model-wide \
    --ignore-experts "0,1,2,250..255" --output ./pruned
```

**Format:**
- Individual indices: `1,2,5`
- Ranges (inclusive): `250..255`
- Combined: `1,2,250..255`

The ignored experts are protected in **all layers** — they will never be
pruned regardless of saliency.

## REAM: expert merging

REAM is an alternative to pruning that **merges** experts instead of removing
them. While pruning discards low-saliency experts entirely, REAM preserves
knowledge from all experts by folding them into fewer, higher-quality
centroids.

**How REAM works:**

1. **Select centroids** — The top-k experts by saliency become centroids (the
   experts that will remain).

2. **Compute similarity** — For each pair of experts, compute gated cosine
   similarity:
   `cos_sim(gate_logit_i * output_i, gate_logit_j * output_j)` averaged over
   calibration tokens.

3. **Group experts** — Each centroid claims the most similar unassigned
   experts (up to `--max-group-size`, default 16). Highest-saliency centroids
   pick first.

4. **Align and merge** — For each group, align member neurons to the centroid
   using a permutation computed from intermediate activation and weight
   similarity, then compute a saliency-weighted average of the aligned
   weights.

5. **Sequential processing** — Layers are processed in order. After merging
   one layer, the model is re-forwarded to get correct inputs for the next
   layer.

**When to use REAM vs REAP:**

| | REAP (prune) | REAM (merge) |
|---|---|---|
| **Speed** | Fast (no calibration data needed at prune time) | Slower (needs calibration forward passes per layer) |
| **Quality** | Good, but discards expert knowledge | Better — preserves all expert knowledge |
| **Memory at prune time** | Low | Higher (computes expert outputs for similarity) |
| **Dependencies** | None extra | `scipy` (optional, for Hungarian alignment) |

## Supported architectures

| Architecture | Config key | MoE location | Notes |
|---|---|---|---|
| **MiniMax** (M1, M2) | `num_local_experts` | All decoder layers | Sigmoid gating + correction bias |
| **GLM4-MoE** | `n_routed_experts` | Layers >= `first_k_dense_replace` | Group-based expert selection, shared experts |
| **GLM4-MoE-Lite** | `n_routed_experts` | `first_k_dense_replace` + `moe_layer_freq` stride | MLA attention, periodic MoE |
| **Qwen3-MoE** | `num_experts` | Every `decoder_sparse_step`-th layer | Softmax gating |
| **Qwen3-Next** | `num_experts` | Every `decoder_sparse_step`-th layer | Sigmoid-gated shared expert |
| **GLM-5 (GLM-MoE-DSA)** | `n_routed_experts` | `first_k_dense_replace` + `moe_layer_freq` | DeepSeek V3.2 + MLA + DSA |
| **DeepSeek V3.2** | `n_routed_experts` | `first_k_dense_replace` + `moe_layer_freq` | Same MoE as GLM-5 |
| **Kimi-K2 / K2.6** (`kimi_k25` wrapper) | `text_config.n_routed_experts` | Same as DeepSeek V3 | Multimodal wrapper, vision dropped at conversion |
| **Nemotron-H** | `n_routed_experts` | hybrid Mamba-2/Attn/MoE pattern | NVFP4 source via `convert-nvfp4` |
| **Gemma4** | varies | per-layer router | Custom routing logic |

Both quantized and unquantized models are supported. Sharded models are not
supported — load without sharding. New architectures plug in by implementing
the `BaseAdapter` interface (`src/mlx_fun/adapters/base.py`).

## Constraints

- You must keep at least `top_k` experts per layer (the number selected per
  token). Pruning below this raises an error.
- Pruning to exactly `top_k` is allowed but warned — every token uses every
  remaining expert, eliminating the MoE routing benefit.
