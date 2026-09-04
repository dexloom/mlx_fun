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

## Domain probing: Q&A scoring + knockout

`domain-probe` answers a different question from REAP saliency. REAP asks "how
much does this expert contribute overall"; the probe asks "does this expert
matter *for this domain*", using questions rather than a corpus.

### 1. Answer-position routing

Each question is chat-templated with `add_generation_prompt=True`, giving a
prompt of length `P`. In teacher mode the reference answer's `A` tokens are
appended and one forward pass scores them; in generate mode the model writes
the answer with the observer hooks live.

Logits at position `t` predict token `t+1`, so the `A` scored predictions are
produced by positions `[P-1, P+A-2]`. Routing is attributed to exactly those
positions. Two consequences:

- The observational score and the knockout delta describe the same predictions.
- The routing decision that produces the *first* answer token is included —
  often the most domain-specific one (`pragma`, `function`, `mapping`).

In generate mode mlx-lm forwards every token before yielding it (prefill of
`P-1`, then the last prompt token, then one forward per generated token), so
captures cover exactly `P + G` positions for `G` generated tokens. The slicer
asserts this rather than assuming it.

### 2. Question-weighted differential

For question `q`, layer `l`, expert `e`, over its `A_q` answer positions:

```
freq_q[l,e]   = (positions routing to e) / A_q      # rows sum to top_k
weight_q[l,e] = (sum of e's routed weight) / A_q
```

These per-question vectors are averaged across questions, so a 200-token answer
and a 10-token answer count the same. This is the substantive difference from
`domain-scan`, which normalizes by total tokens and lets one long sample
dominate.

```
diff_freq   = mean_freq(domain)   - mean_freq(general)
diff_weight = mean_weight(domain) - mean_weight(general)
composite   = 0.5 * layer_norm(diff_freq) + 0.5 * layer_norm(diff_weight)
```

`layer_norm` is the same per-layer min-max used by `safety-scan`. Classification
into domain/general experts then reuses `identify_domain_experts` unchanged.

`coverage[l,e]` — the fraction of questions in which the expert fired at least
once — is tracked separately. `--min-coverage` uses it to drop experts whose
high score rests on one or two questions.

Because the differential is computed from captured routing rather than
recomputed from logits, it reflects the true selection (correction bias,
per-expert scale included) and needs no per-architecture top-k replication.

### 3. Knockout verification

An expert with a high differential is *correlated* with the domain. To test
whether it *matters*, mask it and measure the damage:

```
baseline_i = NLL of answer i with a zero mask installed
masked_i   = NLL of answer i with expert (l,e) unselectable
delta_i    = masked_i - baseline_i
```

The mask is a large negative bias added to the parameter that enters the top-k
**selection score**:

| Architecture family | Parameter | Why |
|---|---|---|
| MiniMax (M1, M2) | `block.e_score_correction_bias` | selection is `argmax(sigmoid(logits) + bias)`, so the mask must land *after* the sigmoid |
| GLM4 / GLM-5 / DeepSeek V3.2 / Nemotron-H | `gate.e_score_correction_bias` | `group_expert_select` adds it before group scoring and top-k |
| Qwen3 / Qwen3-Next / Qwen4-Exp | temporary `gate.bias` | pre-softmax logit; softmax is monotonic and there is no post-softmax correction |

Masking the pre-sigmoid logits instead (what `amplify` biases) would not work
for the sigmoid families: an expert with a large positive correction bias can
survive an arbitrarily negative pre-sigmoid bias. Because only a parameter
changes, the model's own routing runs — grouped selection, `norm_topk_prob`,
`routed_scaling_factor`, latent projections — with no surrogate router and no
hooks. Gemma 4 has no mapping and raises.

The baseline is measured with an all-zero mask installed, on the same code
path, so an expert the router never selects yields a delta of exactly `0.0`.

### 4. Paired statistics

Deltas are paired per question, never pooled means of different question sets.
A masked run that goes non-finite is a *collapse* for that question and is
counted, not dropped — otherwise a mask that destroys half the answers could
average out to "harmless" over the surviving half.

| status | condition |
|---|---|
| `catastrophic` | usable pairs < 50% |
| `inconclusive` | usable pairs < `--min-valid-fraction` (default 0.9) |
| `verified` | `mean_delta >= --min-delta` and the bootstrap CI lower bound > 0 |
| `not_verified` | otherwise |

The CI comes from a seeded bootstrap over the paired deltas (`--bootstrap`,
default 1000 resamples). Only `verified` experts enter
`verified_domain_experts`; the classified `domain_experts` set is never narrowed
by the knockout, since only the top `--verify-top` are checked.

### 5. Prune-set verification

`--verify-prune N` masks the exact set `prune` would remove — the same
`pruner.build_keep_map` call the CLI makes, with the domain experts protected —
and reports the paired delta on both question sets. A healthy result is small
damage on the domain set relative to the general set.

The check masks experts in the *original* router; a real pruned checkpoint has
a smaller expert axis (and a grouped router keeps `n_group` over fewer experts).
It is the closest available stand-in, not proof — confirm with `smoke-test` or
an eval on the pruned model.

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
