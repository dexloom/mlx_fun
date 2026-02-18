# MLX-FUN

**Expert Pruning and Merging for Mixture-of-Experts models on Apple Silicon**

MLX-FUN is an MLX-native toolkit for compressing and analyzing MoE language models on Apple Silicon via [MLX](https://github.com/ml-explore/mlx). It implements seven complementary techniques:

- **REAP** (Routing-based Expert Activation Pruning) — removes the least important experts by measuring saliency from calibration data, then slicing weight tensors. Based on [Cerebras Research's REAP](https://github.com/CerebrasResearch/reap).
- **REAM** (Router-weighted Expert Activation Merging) — instead of discarding experts, groups them around high-saliency centroids and merges via neuron-aligned weighted averaging. Preserves knowledge from all experts while still reducing the model size. Based on [REAM](https://bknyaz.github.io/blog/2026/moe/).
- **SAFEx** (Safety-critical Expert identification) — compares expert routing patterns between harmful and benign datasets to classify experts into HCDG (detect harmful content) and HRCG (control harmful responses) groups. Based on [SAFEx (NeurIPS 2025)](https://arxiv.org/abs/2506.17368).
- **SteerMoE** (Expert Steering) — inference-time expert (de)activation by injecting bias into gate logits before top-k selection. Supports both offline generation and real-time server steering via REST API. Based on [SteerMoE](https://arxiv.org/abs/2509.09660).
- **Abliteration** — removes refusal directions from model weight matrices by orthogonalization, adapted for MoE architectures with per-expert targeting. Based on [Arditi et al. (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/f545448535dfde4f9786555403ab7c49-Paper-Conference.pdf).
- **Domain Scan** — identifies domain-specialized experts by comparing routing patterns on domain-specific data (e.g. Solidity, medical text) vs general data, using the same differential analysis as SAFEx.
- **Amplify** — permanently modifies gate weights/biases so domain-specialized experts are favored natively, producing a model that works with standard `mlx_lm.load()` without runtime hooks.

MLX-FUN supports two collection modes:
- **Offline calibration** (`mlx-fun collect`) — run a dataset through the model in batch
- **Online counting** (`mlx-fun serve`) — serve an OpenAI-compatible API while counting expert activations from real traffic

## How It Works

### The REAP Algorithm

Mixture-of-Experts models route each token to a subset of "expert" sub-networks. Not all experts contribute equally — some are rarely activated or produce small outputs. REAP quantifies expert importance and prunes the least useful ones:

1. **Calibrate** — Run domain-specific text through the model. At each MoE layer, record which experts are selected, their router weights, and the L2 norm of each expert's output.

2. **Score** — For each expert, compute a saliency score. The default REAP metric is:

   ```
   REAP(expert) = mean( ||expert_output|| * router_weight )
   ```

   averaged over all tokens routed to that expert. Experts with low scores contribute little to the final output.

3. **Prune or Merge** — Two approaches:
   - **Prune** (`mlx-fun prune`) — Remove the N lowest-scoring experts per layer by slicing weight tensors. Two strategies: *bottom* (remove lowest) and *strided* (distribute removals evenly).
   - **Merge** (`mlx-fun merge`) — Group all experts around the top-k centroids, align neurons via permutation matching, and produce saliency-weighted averages. Processes layers sequentially so merged weights feed into the next layer.

4. **Save** — Write the compressed model in standard mlx-lm format. It loads with `mlx_lm.load()` like any other model.

### Supported Models

| Architecture | Config key | MoE location | Notes |
|---|---|---|---|
| **MiniMax** (e.g. MiniMax-M1, MiniMax-M2) | `num_local_experts` | All decoder layers | Sigmoid gating + correction bias. Both `minimax` and `minimax_m2` model types supported. |
| **GLM4-MoE** | `n_routed_experts` | Layers >= `first_k_dense_replace` | Group-based expert selection, shared experts preserved |
| **GLM4-MoE-Lite** | `n_routed_experts` | Layers >= `first_k_dense_replace` with `moe_layer_freq` stride | Same as GLM4-MoE + MLA attention, periodic MoE layer frequency |
| **Qwen3-MoE** (e.g. Qwen3-30B-A3B) | `num_experts` | Every `decoder_sparse_step`-th layer, excluding `mlp_only_layers` | Softmax gating, no correction bias |
| **Qwen3-Next** | `num_experts` | Every `decoder_sparse_step`-th layer, excluding `mlp_only_layers` | Softmax gating, sigmoid-gated shared expert preserved |
| **GLM-5** (GLM-MoE-DSA) | `n_routed_experts` | Layers >= `first_k_dense_replace` with `moe_layer_freq` stride | DeepSeek V3.2 architecture, MLA + DSA attention |
| **DeepSeek V3.2** | `n_routed_experts` | Layers >= `first_k_dense_replace` with `moe_layer_freq` stride | Same MoE structure as GLM-5 |

Both quantized (`QuantizedSwitchLinear`) and unquantized models are supported. Sharded models are not supported — load without sharding.

New architectures can be added by implementing the `BaseAdapter` interface.

### Saliency Metrics

| Metric | Formula | Description |
|---|---|---|
| `reap` (default) | `mean(activation_norm * router_weight)` | Weighted contribution — the REAP paper metric |
| `ean` | `mean(activation_norm)` | Expert Activation Norm — ignores routing weight |
| `freq` | `count(tokens_routed)` | Raw routing frequency |
| `weighted_freq` | `sum(router_weight)` | Cumulative routing confidence |

### Pruning Strategies

| Strategy | Description |
|---|---|
| `bottom` (default) | Remove the N lowest-scoring experts. Simple and effective — standard REAP behavior. |
| `strided` | Split experts into important (top) and unimportant (bottom) groups, then prune evenly from both using interval-based selection. Preserves more diversity across the saliency spectrum. |

**How strided pruning works:**

Given 160 experts and `--n-prune 40`:

1. Sort experts by saliency. Split into important (top 120) and unimportant (bottom 40).
2. Prune 20 from each group (50/50 split).
3. Important group (120 experts): stride = 120/20 = 6, remove every 6th expert.
4. Unimportant group (40 experts): stride = 40/20 = 2, remove every 2nd expert.
5. Result: 120 experts kept, with removals distributed across the full saliency range.

This can help preserve routing diversity compared to simply chopping the tail.

### REAM: Expert Merging

REAM is an alternative to pruning that **merges** experts instead of removing them. While pruning discards low-saliency experts entirely, REAM preserves knowledge from all experts by folding them into fewer, higher-quality centroids.

**How REAM works:**

1. **Select centroids** — The top-k experts by saliency become centroids (the experts that will remain).

2. **Compute similarity** — For each pair of experts, compute gated cosine similarity: `cos_sim(gate_logit_i * output_i, gate_logit_j * output_j)` averaged over calibration tokens.

3. **Group experts** — Each centroid claims the most similar unassigned experts (up to `--max-group-size`, default 16). Highest-saliency centroids pick first.

4. **Align and merge** — For each group, align member neurons to the centroid using a permutation computed from intermediate activation and weight similarity, then compute a saliency-weighted average of the aligned weights.

5. **Sequential processing** — Layers are processed in order. After merging one layer, the model is re-forwarded to get correct inputs for the next layer.

**When to use REAM vs REAP:**

| | REAP (prune) | REAM (merge) |
|---|---|---|
| **Speed** | Fast (no calibration data needed at prune time) | Slower (needs calibration forward passes per layer) |
| **Quality** | Good, but discards expert knowledge | Better — preserves all expert knowledge |
| **Memory at prune time** | Low | Higher (computes expert outputs for similarity) |
| **Dependencies** | None extra | `scipy` (optional, for Hungarian alignment) |

## Installation

Requires Python 3.11+ and Apple Silicon (for MLX).

```bash
cd mlx_fun
uv venv && uv pip install -e ".[dev]"
```

Dependencies: `mlx >= 0.30.0`, `mlx-lm >= 0.30.7`, `click`, `tqdm`, `numpy`.

For REAM merging (optional — only needed for `mlx-fun merge`):

```bash
uv pip install -e ".[ream]"
```

For the web dashboard (optional — only needed for `mlx-fun ui`):

```bash
uv pip install -e ".[ui]"
```

For dataset preparation:

```bash
uv pip install -e ".[dataset]"
```

## Dataset Format

MLX-FUN accepts calibration data in two path types (auto-detected) and three JSONL formats (auto-detected per line).

### JSONL File (recommended)

Three JSONL formats are supported. The format is detected automatically per line, so you can even mix formats in one file.

**Chat messages** (compatible with [mlx-lm fine-tuning format](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md)):

```jsonl
{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Hello."}, {"role": "assistant", "content": "How can I help you today?"}]}
{"messages": [{"role": "user", "content": "What is Solidity?"}, {"role": "assistant", "content": "Solidity is a programming language for Ethereum smart contracts."}]}
```

Messages are tokenized via `tokenizer.apply_chat_template()`, so the tokens match exactly what the model sees during chat inference. This is the best format for calibrating chat models.

**Completions** (prompt + completion pairs):

```jsonl
{"prompt": "What is the capital of France?", "completion": "Paris."}
```

**Plain text** (single text field):

```jsonl
{"content": "// SPDX-License-Identifier: MIT\npragma solidity ^0.8.0;\n\ncontract Token {\n    ..."}
```

The text key defaults to `"content"` and can be changed with `--text-key`. Each entry is tokenized and truncated to `--max-tokens` (default 2048).

**Format priority**: `messages` > `prompt`/`completion` > plain text. If a line has a `"messages"` key, it is always treated as chat format regardless of other keys.

### Directory of Source Files

A directory containing raw source files (`.sol`, `.txt` by default). Each file is read, tokenized, and used as one calibration sample:

```
data/solidity/
├── Token.sol
├── Vault.sol
├── Governance.sol
└── ...
```

### Preparing a Solidity Dataset

A preparation script is included to download Solidity code from HuggingFace:

```bash
uv pip install -e ".[dataset]"

python scripts/prepare_dataset.py \
    --source bigcode/the-stack-dedup \
    --output ./data/solidity_calibration.jsonl \
    --max-samples 512
```

Options:

| Flag | Default | Description |
|---|---|---|
| `--source` | `bigcode/the-stack-dedup` | HuggingFace dataset ID |
| `--output` | `./data/solidity_calibration.jsonl` | Output JSONL path |
| `--max-samples` | 512 | Number of samples to collect |
| `--min-tokens` | 64 | Minimum character length filter |
| `--max-chars` | 16384 | Truncate long files |
| `--split` | `train` | Dataset split |

The script streams from HuggingFace, filters for valid Solidity (`pragma solidity` check), and writes JSONL.

### Dataset Guidelines

- **256-512 samples** is a good calibration size — enough to get stable saliency estimates without excessive runtime.
- **Domain matters** — calibrate on the domain you care about. Solidity-calibrated pruning retains better Solidity generation than generic-text calibration at the same prune ratio.
- **Token length** — 2048 tokens per sample captures enough context for routing patterns to stabilize.

## Usage

MLX-FUN provides eleven CLI commands. The main pipeline is **collect** -> **prune** (or **merge**) -> **smoke-test**, with **serve** for online collection, **ui** for a web dashboard, **safety-scan** / **steer** / **abliterate** for safety analysis, and **domain-scan** / **amplify** for domain specialization.

### Step 1: Collect Saliency Statistics

Run calibration to measure expert importance:

```bash
mlx-fun collect \
    --model mlx-community/MiniMax-M1-40k-4bit \
    --dataset ./data/solidity_calibration.jsonl \
    --output ./saliency.npz \
    --max-samples 128 \
    --seed 42
```

| Flag | Default | Description |
|---|---|---|
| `--model` | *(required)* | Model path or HuggingFace repo ID |
| `--dataset` | *(required)* | Path to JSONL file or directory |
| `--output` | *(required)* | Output `.npz` file for saliency stats |
| `--max-samples` | 128 | Maximum calibration samples. If the dataset has more, a random subset is selected. |
| `--max-tokens` | 2048 | Maximum tokens per sample |
| `--text-key` | `content` | JSON key for text in JSONL files |
| `--seed` | *(none)* | Random seed for reproducible sample selection |

This runs each calibration sample through the model with hooks installed on every MoE layer. The hooks capture expert indices, router weights, and activation norms, which are accumulated into per-expert saliency statistics and saved to an `.npz` file.

### Step 2: Prune Experts

Select and remove the least important experts:

```bash
mlx-fun prune \
    --model mlx-community/MiniMax-M1-40k-4bit \
    --saliency ./saliency.npz \
    --output ./pruned_model \
    --n-prune 16 \
    --metric reap \
    --strategy bottom
```

| Flag | Default | Description |
|---|---|---|
| `--model` | *(required)* | Same model used for collection |
| `--saliency` | *(required)* | Path to `.npz` from collect step |
| `--output` | *(required)* | Output directory for pruned model |
| `--n-prune` | *(required)* | Number of experts to remove per layer |
| `--metric` | `reap` | Saliency metric: `reap`, `ean`, `freq`, `weighted_freq` |
| `--strategy` | `bottom` | Pruning strategy: `bottom` (remove lowest) or `strided` (distribute evenly) |
| `--safety-map` | *(none)* | Path to `safety_report.json` from `safety-scan` |
| `--safety-mode` | *(none)* | `protect` (never prune safety experts) or `target` (specifically prune them) |
| `--domain-map` | *(none)* | Path to `domain_report.json` from `domain-scan` |
| `--domain-mode` | *(none)* | `protect` (never prune domain experts) |

The pruned model is saved in standard mlx-lm format (safetensors + config.json + tokenizer) and can be loaded with `mlx_lm.load()`. A `reap_metadata.json` file records the pruning details.

**Safety-aware pruning:**

When a safety report is provided via `--safety-map`, the pruner can either protect or target safety-critical experts:

```bash
# Protect safety experts — never prune them, regardless of saliency score
mlx-fun prune --model ... --saliency ... --output ... --n-prune 16 \
    --safety-map safety_report.json --safety-mode protect

# Target safety experts — always prune them first
mlx-fun prune --model ... --saliency ... --output ... --n-prune 16 \
    --safety-map safety_report.json --safety-mode target
```

Internally, `protect` sets safety-critical experts' saliency scores to `+inf` (never in bottom-n), while `target` sets them to `-inf` (always in bottom-n).

**Domain-aware pruning:**

When a domain report is provided via `--domain-map`, domain-specialized experts are protected from pruning. This can be combined with safety constraints — the protected sets are merged via union:

```bash
# Protect domain experts from pruning
mlx-fun prune --model ... --saliency ... --output ... --n-prune 16 \
    --domain-map domain_report.json --domain-mode protect

# Combine domain + safety protection
mlx-fun prune --model ... --saliency ... --output ... --n-prune 16 \
    --safety-map safety_report.json --safety-mode protect \
    --domain-map domain_report.json --domain-mode protect
```

**Constraints:**
- You must keep at least `top_k` experts per layer (the number selected per token). Pruning below this threshold raises an error.
- Pruning to exactly `top_k` experts is allowed but triggers a warning — it means every token uses every remaining expert, eliminating the MoE routing benefit.

### Step 2b: Merge Experts (REAM Alternative)

Instead of pruning, you can merge experts using REAM. This requires the same saliency `.npz` file from the collect step, plus calibration data for computing expert similarity:

```bash
mlx-fun merge \
    --model mlx-community/MiniMax-M1-40k-4bit \
    --saliency ./saliency.npz \
    --dataset ./data/solidity_calibration.jsonl \
    --output ./merged_model \
    --n-prune 16 \
    --similarity-mode gated \
    --alignment greedy \
    --max-group-size 16 \
    --max-samples 64 \
    --seed 42
```

| Flag | Default | Description |
|---|---|---|
| `--model` | *(required)* | Same model used for collection |
| `--saliency` | *(required)* | Path to `.npz` from collect step |
| `--dataset` | *(required)* | Calibration data for similarity/alignment computation |
| `--output` | *(required)* | Output directory for merged model |
| `--n-prune` | *(required)* | Number of experts to prune (merge reduces by this many per layer) |
| `--metric` | `reap` | Saliency metric: `reap`, `ean`, `freq`, `weighted_freq` |
| `--similarity-mode` | `gated` | Expert similarity: `gated` (gate_logit * output cosine sim) or `average` (mean of output sim + gate logit sim) |
| `--alignment` | `greedy` | Neuron alignment: `greedy` (fast, O(n^2)), `hungarian` (optimal, requires scipy), or `none` (skip alignment) |
| `--max-group-size` | `16` | Maximum experts per merge group (the C parameter from the REAM paper) |
| `--max-samples` | `64` | Calibration samples for similarity computation |
| `--max-tokens` | `2048` | Max tokens per sample |
| `--max-similarity-tokens` | `512` | Max tokens for similarity computation (subsampled if more) |
| `--max-alignment-tokens` | `256` | Max tokens for permutation alignment |
| `--seed` | *(none)* | Random seed |

The merged model is saved with a `ream_metadata.json` recording the centroid map and group assignments.

**Note:** REAM processes layers sequentially — after merging each MoE layer, it re-forwards calibration data through the updated model to get correct inputs for the next layer. This means one full forward pass per MoE layer, which is slower than pruning but produces higher-quality results.

### Step 3: Smoke Test

Verify the pruned model generates text:

```bash
mlx-fun smoke-test \
    --model ./pruned_model \
    --prompt "pragma solidity ^0.8.0;"
```

| Flag | Default | Description |
|---|---|---|
| `--model` | *(required)* | Path to pruned model directory |
| `--prompt` | `pragma solidity ^0.8.0;` | Generation prompt |
| `--max-tokens` | 100 | Maximum tokens to generate |

### Online Collection: Serve with Expert Counting

Instead of offline calibration, you can collect expert statistics from real production traffic. The `serve` command starts an OpenAI-compatible API server with lightweight hooks that count expert activations during inference:

```bash
mlx-fun serve \
    --model mlx-community/MiniMax-M1-40k-4bit \
    --port 8080 \
    --mode lightweight \
    --auto-save ./online_saliency.npz
```

| Flag | Default | Description |
|---|---|---|
| `--model` | *(required)* | Model path or HuggingFace repo ID |
| `--host` | `127.0.0.1` | Server bind address |
| `--port` | `8080` | Server port |
| `--mode` | `lightweight` | `lightweight` (freq/weighted_freq only, fast) or `full` (all metrics incl. activation norms) |
| `--auto-save` | *(none)* | Path to auto-save stats on shutdown (Ctrl+C) |
| `--max-tokens` | `512` | Default max tokens for generation |
| `--chat-template` | *(none)* | Chat template override |
| `--safety-map` | *(none)* | Path to `safety_report.json` for steering at startup |
| `--steering-mode` | *(none)* | `safe` (boost safety experts) or `unsafe` (mask them) |
| `--domain-map` | *(none)* | Path to `domain_report.json` for domain boosting at startup |
| `--domain-steering-mode` | *(none)* | `boost` (activate domain experts) or `suppress` (deactivate general experts) |

The server is fully OpenAI-compatible — use it as a drop-in replacement:

```bash
curl http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":50}'
```

#### Management Endpoints

The server exposes additional REAP endpoints for monitoring and exporting statistics:

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/v1/reap/stats` | Full expert frequency/score data as JSON, including computed scores and total samples |
| `GET` | `/v1/reap/info` | Model info: layer/expert counts, request/token totals, steering status |
| `POST` | `/v1/reap/save` | Save accumulator to `.npz` file. Body: `{"path": "output.npz"}` |
| `POST` | `/v1/reap/reset` | Reset all counters to zero |
| `GET` | `/v1/reap/steer` | Get current steering config |
| `POST` | `/v1/reap/steer` | Update steering config (see below) |
| `DELETE` | `/v1/reap/steer` | Remove all steering (reset biases) |

**Enhanced `/v1/reap/stats` response:**

```json
{
  "freq": [[...], [...]],           // Raw frequency counts (num_layers × num_experts)
  "weighted_freq_sum": [[...], [...]], // Raw weighted frequency sums
  "reap_sum": [[...], [...]],       // Raw REAP sums (activation_norm × router_weight)
  "ean_sum": [[...], [...]],         // Raw Expert Activation Norm sums
  "reap_count": [[...], [...]],     // Sample counts per expert
  "num_layers": 62,
  "num_experts": 256,
  "request_count": 150,
  "token_count": 75000,
  "total_samples": 12687240.0,      // Total samples processed
  "computed_scores": {
    "reap": [[...], [...]],         // Computed REAP scores (divide sum by count)
    "ean": [[...], [...]],          // Computed EAN scores
    "freq": [[...], [...]],         // Computed frequency scores (same as freq array)
    "weighted_freq": [[...], [...]]  // Computed weighted frequency scores
  }
}
```

The `computed_scores` field provides ready-to-use scores for comparison with `mlx-fun stats-diff` and for rank-based merging with `mlx-fun stats-merge`.

```bash
# Check stats after some traffic
curl http://localhost:8080/v1/reap/stats | python -m json.tool

# Save and feed directly into the prune pipeline
curl -X POST http://localhost:8080/v1/reap/save \
    -d '{"path": "online_saliency.npz"}'

mlx-fun prune --model mlx-community/MiniMax-M1-40k-4bit \
    --saliency online_saliency.npz --output ./pruned --n-prune 16 --metric freq
```

#### Runtime Steering via REST

The steering endpoint allows hot-swapping expert (de)activation at runtime without restarting the server. Bias updates are applied atomically — no hook reinstallation needed.

```bash
# Enable steering from a safety report
curl -X POST http://localhost:8080/v1/reap/steer \
    -H "Content-Type: application/json" \
    -d '{"safety_map": "/path/to/safety_report.json", "mode": "safe"}'

# Enable steering from a domain report
curl -X POST http://localhost:8080/v1/reap/steer \
    -H "Content-Type: application/json" \
    -d '{"domain_map": "/path/to/domain_report.json", "mode": "boost"}'

# Or specify experts directly
curl -X POST http://localhost:8080/v1/reap/steer \
    -H "Content-Type: application/json" \
    -d '{"deactivate": {"0": [3, 17], "5": [42]}, "activate": {"12": [8]}, "mask_value": -1e9}'

# Check current steering config
curl http://localhost:8080/v1/reap/steer

# Remove all steering
curl -X DELETE http://localhost:8080/v1/reap/steer
```

#### Lightweight vs Full Mode

| Mode | Metrics available | Overhead | Best for |
|------|-------------------|----------|----------|
| `lightweight` | `freq`, `weighted_freq` | Minimal — no extra computation | Production serving, long-running collection |
| `full` | `reap`, `ean`, `freq`, `weighted_freq` | Computes `mx.linalg.norm()` per expert output | Short collection runs where you need all metrics |

In lightweight mode, `reap` and `ean` scores will be zero since they require activation norms. Use `freq` or `weighted_freq` as the `--metric` when pruning with lightweight-collected data.

#### Note on Chat Traffic Bias

When serving chat conversations where each request includes the full dialogue history, earlier messages are re-processed more often than later ones. This inflates expert counts for tokens in early messages.

Mitigations:
- **KV cache** partially helps — mlx-lm's prompt cache skips re-computation for cached prefixes
- **Use `reap` or `ean` metrics** (full mode) — these are averages, so they're count-invariant
- **Use `/v1/reap/reset`** between collection windows if you want stats from specific traffic periods

### Safety Scan: Identify Safety-Critical Experts

Analyze a model's routing patterns to identify experts that behave differently on harmful vs benign inputs. This implements SAFEx-style differential activation analysis:

```bash
mlx-fun safety-scan \
    --model mlx-community/Qwen3-30B-A3B-4bit \
    --harmful-dataset ./data/harmful_prompts.jsonl \
    --benign-dataset ./data/benign_prompts.jsonl \
    --output safety_report.json \
    --max-samples 128 --seed 42
```

| Flag | Default | Description |
|---|---|---|
| `--model` | *(required)* | Model path or HuggingFace repo ID |
| `--harmful-dataset` | *(required)* | Path to harmful prompts (JSONL or directory) |
| `--benign-dataset` | *(required)* | Path to benign prompts (JSONL or directory) |
| `--output` | *(required)* | Output path for `safety_report.json` |
| `--max-samples` | 128 | Max samples per dataset |
| `--max-tokens` | 2048 | Max tokens per sample |
| `--text-key` | `content` | JSON key for text in JSONL |
| `--threshold-percentile` | 90.0 | Percentile threshold for classifying safety-critical experts |
| `--seed` | *(none)* | Random seed |

The safety report classifies experts into:
- **HCDG** (Harmful Content Detection Group) — experts that activate MORE on harmful content (high composite score)
- **HRCG** (Harmful Response Control Group) — experts that activate MORE on benign content and suppress harmful outputs (low composite score)
- **Safety-critical** — the union of HCDG and HRCG

The report is used by `prune --safety-map`, `steer`, `abliterate --target safety-experts`, and the server's steering API.

### Steer: Inference with Expert Steering

Generate text with SteerMoE-style gate logit injection to selectively activate or deactivate safety-critical experts:

```bash
mlx-fun steer \
    --model mlx-community/Qwen3-30B-A3B-4bit \
    --safety-map safety_report.json \
    --mode safe \
    --prompt "How do I make a bomb?" \
    --max-tokens 100
```

| Flag | Default | Description |
|---|---|---|
| `--model` | *(required)* | Model path or HuggingFace repo ID |
| `--safety-map` | *(required)* | Path to `safety_report.json` |
| `--mode` | *(required)* | `safe` (boost HRCG experts) or `unsafe` (mask all safety-critical experts) |
| `--prompt` | *(required)* | Input prompt for generation |
| `--max-tokens` | 100 | Max tokens to generate |
| `--mask-value` | -1e9 | Gate logit bias for deactivation |
| `--boost-value` | 1e4 | Gate logit bias for activation |

**How it works:** A pre-computed bias array of shape `(num_experts,)` is added to raw gate logits before top-k selection. In `safe` mode, HRCG experts get `+boost_value` bias (ensuring they're selected). In `unsafe` mode, all safety-critical experts get `mask_value` bias (effectively removing them from selection).

### Abliterate: Refusal Direction Removal

Remove the refusal direction from model weights via orthogonalization, adapted for MoE architectures:

```bash
mlx-fun abliterate \
    --model mlx-community/Qwen3-30B-A3B-4bit \
    --harmful-dataset ./data/harmful_prompts.jsonl \
    --benign-dataset ./data/benign_prompts.jsonl \
    --output ./abliterated_model \
    --layers auto \
    --target all \
    --max-samples 64
```

| Flag | Default | Description |
|---|---|---|
| `--model` | *(required)* | Model path or HuggingFace repo ID |
| `--harmful-dataset` | *(required)* | Path to harmful prompts |
| `--benign-dataset` | *(required)* | Path to benign prompts |
| `--output` | *(required)* | Output directory for abliterated model |
| `--layers` | `auto` | `auto` (top 50% by refusal direction norm), `all`, or `start-end` (e.g. `10-20`) |
| `--target` | `all` | `all` (all experts), `safety-experts` (only safety-critical), `dense-only` (skip MoE) |
| `--safety-map` | *(none)* | Required if `--target safety-experts` |
| `--max-samples` | 64 | Max samples per dataset |
| `--max-tokens` | 2048 | Max tokens per sample |
| `--extraction-position` | `last` | `last` (last token) or `mean` (mean pool) |
| `--text-key` | `content` | JSON key for text |
| `--seed` | *(none)* | Random seed |

**How it works:**

1. **Capture** — Hook decoder layers, forward harmful and benign prompts, extract residual stream activations at the last token position
2. **Compute direction** — `refusal_direction[layer] = normalize(mean_harmful - mean_benign)`
3. **Select layers** — Auto-select the top 50% of layers by refusal direction norm (or use specified range)
4. **Orthogonalize** — For each selected layer, project the refusal direction out of weight matrices: `W' = W - (W @ d) * d^T`

**MoE-specific targeting:**
- `--target all` — orthogonalize all expert down_proj weights + attention o_proj
- `--target safety-experts` — only orthogonalize safety-critical experts identified by `safety-scan`
- `--target dense-only` — skip MoE experts, only modify attention o_proj and shared expert weights

The abliterated model is saved with `abliteration_metadata.json` recording the method, target, layers, and direction norms.

### Domain Scan: Identify Domain-Specialized Experts

Analyze a model's routing patterns to identify experts that activate more on domain-specific data (e.g. Solidity code, medical text) compared to general data. Uses the same differential activation analysis as safety-scan:

```bash
mlx-fun domain-scan \
    --model mlx-community/MiniMax-M1-40k-4bit \
    --domain-dataset ./data/solidity_calibration.jsonl \
    --general-dataset ./data/general_text.jsonl \
    --output domain_report.json \
    --domain-name solidity \
    --max-samples 128 --seed 42
```

| Flag | Default | Description |
|---|---|---|
| `--model` | *(required)* | Model path or HuggingFace repo ID |
| `--domain-dataset` | *(required)* | Path to domain data (JSONL or directory) |
| `--general-dataset` | *(required)* | Path to general data (JSONL or directory) |
| `--output` | *(required)* | Output path for `domain_report.json` |
| `--domain-name` | *(required)* | Name of the domain (e.g. `solidity`, `medical`) |
| `--max-samples` | 128 | Max samples per dataset |
| `--max-tokens` | 2048 | Max tokens per sample |
| `--text-key` | `content` | JSON key for text in JSONL |
| `--threshold-percentile` | 90.0 | Percentile threshold for classifying domain-specialized experts |
| `--seed` | *(none)* | Random seed |

The domain report classifies experts into:
- **Domain experts** — experts that activate MORE on domain data (high composite score, above the threshold percentile)
- **General experts** — experts that activate MORE on general data (low composite score, below the inverse threshold)

The report is used by `prune --domain-map`, `amplify`, `serve --domain-map`, and the server's steering API.

### Amplify: Permanent Domain Expert Gate Modification

Permanently modify gate weights so domain-specialized experts are favored natively. The amplified model works with standard `mlx_lm.load()` — no hooks needed at inference time:

```bash
mlx-fun amplify \
    --model mlx-community/MiniMax-M1-40k-4bit \
    --domain-map domain_report.json \
    --output ./amplified_model \
    --scale 1.0 \
    --threshold 0.0
```

| Flag | Default | Description |
|---|---|---|
| `--model` | *(required)* | Model path or HuggingFace repo ID |
| `--domain-map` | *(required)* | Path to `domain_report.json` from `domain-scan` |
| `--output` | *(required)* | Output directory for amplified model |
| `--scale` | 1.0 | Amplification strength multiplier |
| `--threshold` | 0.0 | Minimum composite score to amplify (experts below this get no boost) |

**How it works:** For each domain expert, computes a boost value: `scale * max(0, composite_score - threshold)`. This boost is applied differently per model architecture:

| Model | Gate type | Amplification target | Position |
|-------|-----------|---------------------|----------|
| MiniMax/MiniMax-M2 | `nn.Linear(bias=False)` | Sets `gate.bias` | Pre-sigmoid |
| GLM4/GLM4-Lite/GLM5/DSv3 | Custom `MoEGate` | Adds to `gate.e_score_correction_bias` | Post-sigmoid |
| Qwen3/Qwen3-Next | `nn.Linear(bias=False)` | Sets `gate.bias` | Pre-softmax |

The amplified model is saved with `amplification_metadata.json` recording the domain name, scale, threshold, and per-layer bias arrays.

**Runtime domain boosting via steering** is also available as an alternative to permanent amplification:

```bash
# Via the server
mlx-fun serve --model ... --domain-map domain_report.json --domain-steering-mode boost

# Via the steer command (combine with safety steering)
mlx-fun steer --model ... --safety-map safety_report.json --mode safe --prompt "..."
```

### Statistics Operations: Diff, Merge, Purge

Work with multiple collected saliency files to compare, combine, and filter statistics.

#### Compare Two Saliency Files

Compute differences between two collected saliency files to understand how expert routing varies across datasets:

```bash
mlx-fun stats-diff \
    --file1 data/reap_saliency_agent_minimax_m25.npz \
    --file2 data/reap_saliency_solidity_functions_minimax_m25.npz \
    --metric freq \
    --output diff_report.json
```

| Flag | Default | Description |
|---|---|---|
| `--file1` | *(required)* | Path to first saliency `.npz` file |
| `--file2` | *(required)* | Path to second saliency `.npz` file |
| `--metric` | `reap` | Saliency metric: `reap`, `ean`, `freq`, `weighted_freq` |
| `--output` | *(none)* | Optional path to save diff report as JSON |

The output shows:
- **Difference statistics**: mean, std, min, max of differences
- **Distribution**: count of positive (file1 > file2), negative (file2 > file1), and zero differences
- **Top differences**: 10 experts with largest positive and negative differences

Save the report with `--output` for programmatic analysis or export.

**Use cases:**
- Compare routing patterns between different domains (e.g., code vs general text)
- Identify domain-specific experts (large positive/negative differences)
- Validate that merging preserved the expected statistics
- Debug routing behavior across different calibration datasets

#### Merge Multiple Saliency Files

Combine statistics from multiple datasets using rank-based aggregation. This approach normalizes data across different datasets by computing per-layer rankings and summing them:

```bash
# Rank-based merge (default metric: reap)
mlx-fun stats-merge \
    --files data/reap_saliency_agent_minimax_m25.npz \
    --files data/reap_saliency_solidity_functions_minimax_m25.npz \
    --output data/merged_saliency.npz \
    --metric reap

# Using frequency metric for ranking
mlx-fun stats-merge \
    --files data/run1.npz --files data/run2.npz --files data/run3.npz \
    --output data/merged_ranks.npz \
    --metric freq
```

| Flag | Default | Description |
|---|---|---|
| `--files` | *(required)* | Paths to saliency `.npz` files to merge (repeat for each file) |
| `--output` | *(required)* | Output path for merged `.npz` file |
| `--metric` | `reap` | Metric to use for ranking: `reap`, `ean`, `freq`, or `weighted_freq` |

**How Rank-Based Merge Works:**

1. **Compute scores** — For each input file, compute saliency scores using the specified metric
2. **Rank per-layer** — Within each layer, rank experts from 1 (highest score) to N (lowest score)
3. **Sum ranks** — Add up the ranks from all files for each (layer, expert) pair
4. **Result** — Lower summed rank = more important expert (consistently ranked high across datasets)

This approach ensures each dataset contributes equally regardless of sample count or scale differences.

**Example:**

```bash
# Dataset A has 10,000 samples, Dataset B has 1,000 samples
# Both contribute equally to the final ranking

mlx-fun stats-merge --files A.npz --files B.npz --output merged.npz --metric reap
# Rank sum range: [2, 2*num_experts]
# Lower values = more important (expert ranked high in both datasets)
```

**Metrics for Ranking:**

| Metric | Description |
|--------|-------------|
| `reap` (default) | Router-weighted activation norm — the standard REAP importance score |
| `ean` | Expert Activation Norm — average activation magnitude, ignores routing weight |
| `freq` | Raw routing frequency — how often each expert is selected |
| `weighted_freq` | Cumulative routing confidence — sum of router weights |

**Use cases:**
- Merge statistics from datasets of different sizes (code + natural language) with balanced representation
- Combine multiple calibration runs while normalizing for sample count differences
- Identify experts that are consistently important across different domains

**Validation:**
- All files must have identical dimensions (same num_layers and num_experts)
- Only files from the same model architecture can be merged
- The merge operation is commutative and associative

#### Purge Low-Activation Data

Filter out experts with minimal activation to focus pruning on meaningful patterns:

```bash
mlx-fun stats-purge \
    --input data/merged_saliency.npz \
    --output data/purged_saliency.npz \
    --min-freq 100 \
    --min-count 10
```

| Flag | Default | Description |
|---|---|---|
| `--input` | *(required)* | Path to input saliency `.npz` file |
| `--output` | *(required)* | Output path for purged `.npz` file |
| `--min-freq` | *(none)* | Minimum activation frequency to keep (experts with lower freq are zeroed out) |
| `--min-count` | *(none)* | Minimum reap_count to keep (experts with fewer samples are zeroed out) |
| `--max-norm` | *(none)* | Maximum activation norm (warning: only reports, doesn't cap) |

The purging operation zeros out data for experts that don't meet the specified thresholds. This can help:
- Remove noise from barely-activated experts
- Focus pruning decisions on experts with meaningful activation patterns
- Reduce the influence of statistical outliers
- Create cleaner statistics for downstream analysis

**Output:**
- Reports total expert-layer pairs, number purged, and number kept
- Shows breakdown by filter type (freq, count, norm)
- The purged file maintains the same dimensions (zeroed entries instead of removed)

**Note:** At least one filter option (`--min-freq`, `--min-count`, or `--max-norm`) must be specified.

#### Web Dashboard: Diff Analysis Tab

The Gradio web UI includes a **Diff Analysis** tab for visual comparison of two saliency files:

```bash
# Terminal 1: Start the REAP server (optional - tab works independently)
mlx-fun serve --model mlx-community/MiniMax-M1-40k-4bit --port 8080

# Terminal 2: Launch the dashboard
mlx-fun ui --server-url http://127.0.0.1:8080
```

The Diff Analysis tab provides:
- **File inputs**: Paths to two `.npz` files (with placeholder examples)
- **Metric selector**: Choose Frequency, Weighted Frequency, REAP, or EAN
- **Difference heatmap**: Visual representation using diverging colormap
  - **Red** = positive differences (file1 has higher activation)
  - **Blue** = negative differences (file2 has higher activation)
  - **White** = no difference
- **Statistics summary**: Mean, std, range, and distribution of differences
- **JSON export**: Full difference report for programmatic use

This is useful for quick visual comparisons without needing to run CLI commands.

### Web Dashboard

Launch a Gradio-based web UI to monitor and control a running REAP server:

```bash
# Terminal 1: Start the REAP server
mlx-fun serve --model mlx-community/MiniMax-M1-40k-4bit --port 8080

# Terminal 2: Launch the dashboard
mlx-fun ui --server-url http://127.0.0.1:8080
```

| Flag | Default | Description |
|---|---|---|
| `--server-url` | `http://127.0.0.1:8080` | URL of the running REAP server |
| `--host` | `127.0.0.1` | Frontend bind address |
| `--port` | `7860` | Frontend port |
| `--share` | *(off)* | Create a public Gradio share link |

The dashboard provides four tabs:

| Tab | Features |
|-----|----------|
| **Chat** | Talk to the model via streaming chat, configurable system prompt / temperature / max tokens |
| **Dashboard** | Expert activation heatmaps (frequency or weighted frequency), per-layer bar charts with layer selector |
| **Steering** | Apply steering from a safety report (safe/unsafe) or custom JSON config, view/remove active steering |
| **Controls** | Server info, save saliency data to file, reset counters, raw stats JSON |

### Python API

The components can also be used directly:

```python
import mlx.core as mx
from mlx_lm import load as mlx_load

from mlx_fun.adapters import get_adapter
from mlx_fun.observer import install_hooks, collect_captures, remove_hooks
from mlx_fun.saliency import SaliencyAccumulator
from mlx_fun.pruner import select_experts_to_keep, select_experts_to_keep_strided, prune_model
from mlx_fun.save import save_pruned_model

# Load model
model, tokenizer = mlx_load("mlx-community/MiniMax-M1-40k-4bit")
config = {...}  # from config.json
adapter = get_adapter(model, config)

# Calibrate
moe_indices = adapter.moe_layer_indices()
moe_blocks = [adapter.get_moe_block(i) for i in moe_indices]
install_hooks(moe_blocks, config["model_type"])

acc = SaliencyAccumulator(len(moe_indices), adapter.num_routed_experts())

for tokens in calibration_data:
    model(tokens.reshape(1, -1))
    for idx, captures in enumerate(collect_captures(moe_blocks)):
        for inds, scores, norms in captures:
            acc.update(idx, inds.reshape(-1, inds.shape[-1]),
                       scores.reshape(-1, scores.shape[-1]),
                       norms.reshape(-1, norms.shape[-1]))

remove_hooks(moe_blocks)

# Prune (choose one strategy)
scores = acc.compute_scores("reap")
keep_map = select_experts_to_keep(scores, n_prune=16)           # bottom strategy
# keep_map = select_experts_to_keep_strided(scores, n_prune=16) # strided strategy
model_keep_map = {moe_indices[i]: k for i, k in keep_map.items()}
new_config = prune_model(adapter, model_keep_map)

# Save
save_pruned_model(model, tokenizer, new_config, "./pruned",
                  model_keep_map, adapter.num_routed_experts(), "reap")
```

## Output Format

The pruned model directory contains:

```
pruned_model/
├── model.safetensors          # Pruned weights (or sharded model-*.safetensors)
├── model.safetensors.index.json
├── config.json                # Updated config with reduced expert count
├── tokenizer.json             # Tokenizer files (unchanged)
├── tokenizer_config.json
├── special_tokens_map.json
└── reap_metadata.json         # Pruning provenance
```

For **pruned** models, `reap_metadata.json` records the pruning details:

```json
{
  "original_num_experts": 64,
  "pruned_num_experts": 48,
  "metric": "reap",
  "keep_map": {
    "0": [0, 1, 3, 5, 7, ...],
    "1": [0, 2, 4, 6, 8, ...]
  }
}
```

For **merged** models (REAM), `ream_metadata.json` records the centroid and group assignments:

```json
{
  "method": "ream",
  "original_num_experts": 64,
  "merged_num_experts": 48,
  "metric": "reap",
  "centroid_map": {
    "0": [2, 5, 11, ...],
    "1": [1, 7, 15, ...]
  },
  "group_map": {
    "0": {"2": [2, 7, 14], "5": [5, 3, 9]},
    "1": {"1": [1, 4, 12], "7": [7, 0, 8]}
  }
}
```

For **abliterated** models, `abliteration_metadata.json` records the orthogonalization details:

```json
{
  "method": "abliteration",
  "target": "all",
  "abliterated_layers": [10, 11, 12, 13, 14, 15],
  "direction_norms": {
    "10": 0.0234,
    "11": 0.0312,
    "12": 0.0287
  }
}
```

For **amplified** models, `amplification_metadata.json` records the domain amplification details:

```json
{
  "method": "amplification",
  "domain_name": "solidity",
  "scale": 1.0,
  "threshold": 0.0,
  "amplified_layers": [0, 1, 2, 5, 8],
  "per_layer_bias": {
    "0": [0.0, 0.0, 0.8, 0.0, 0.6, ...],
    "1": [0.3, 0.0, 0.0, 0.7, ...]
  }
}
```

All model types (pruned, merged, abliterated, amplified) load with standard `mlx_lm.load("./output_model")` — no special loader required.

## Architecture

```
┌───────────────────────────────────────────────────────────────────────────────────────────────┐
│                                         CLI (cli.py)                                           │
│  collect / prune / merge / smoke-test / serve / ui / safety-scan / steer / abliterate          │
│  domain-scan / amplify                                                                         │
└──┬────────────┬───────────┬──────────┬─────────────┬────────────┬──────────────┬──────────────┘
   │            │           │          │             │            │              │
┌──▼──────┐ ┌──▼─────┐ ┌───▼────┐ ┌───▼──────────┐ │      ┌─────▼──────────┐ ┌─▼────────────┐
│Observer │ │ Pruner │ │ Merger │ │ Server       │ │      │ Abliterate     │ │ Domain       │
│ hooks   │ │ engine │ │ (REAM) │ │ (OpenAI +    │ │      │ residual hooks │ │ scan + gate  │
│(offline)│ └──┬─────┘ └───┬────┘ │ REAP + Steer)│ │      │ orthogonalize  │ │ amplification│
└──┬──────┘    │           │      └───┬──────────┘ │      └──────┬─────────┘ └──────┬───────┘
   │           │           │          │      ┌─────▼────────┐    │                  │
┌──▼──────┐ ┌──▼──────┐ ┌─▼────────┐ │  ┌───▼────────────┐ │    │                  │
│Saliency │ │ Tensor  │ │REAM hooks│ │  │Safety analysis │ │    │                  │
│ accum.  │ │ slicing │ │Similarity│ │  │DifferentialAcc │◄├────┼──────────────────┘
└─────────┘ └─────────┘ │Alignment │ │  │SafetyReport    │ │    │
                        └─┬────────┘ │  └───┬────────────┘ │    │
                          │    ┌──────▼──────▼─┐     ┌─────▼────▼───────┐
                          │    │OnlineAccum.   │     │ Steering hooks   │
                          │    │(thread-safe)  │     │ (gate bias inj.) │
                          │    └──────┬────────┘     └─────┬────────────┘
              ┌───────────▼───────────▼────────────────────▼──┐
              │                Adapter Layer                    │
              │    MiniMax / GLM4 / GLM5 / Qwen / DSv3         │
              └──────────────────┬─────────────────────────────┘
                                 │
              ┌──────────────────▼─────────────────────────────┐
              │          Frontend (frontend.py)                  │
              │  Gradio web UI: chat, heatmaps, steering, mgmt  │
              │  Connects to server via HTTP REST API            │
              └────────────────────────────────────────────────┘
```

- **Adapters** abstract model-specific MoE access (layer paths, gate structure, config keys)
- **Observer** installs hooks via `__class__` swapping (Python resolves `__call__` on the type, not the instance — `types.MethodType` doesn't work for special methods)
- **Saliency** accumulates statistics in numpy float64 using vectorized `np.add.at()` scatter-adds
- **Pruner** uses `mx.take()` to slice expert-axis tensors (weights, scales, biases, gates). Supports bottom and strided pruning strategies. Accepts `protected_experts` / `targeted_experts` from safety reports.
- **Merger** (REAM) computes gated expert similarity, groups experts around centroids, aligns intermediate neurons via permutation matching, and produces saliency-weighted averages. Processes layers sequentially with model re-forwarding.
- **Safety** (`safety.py`) tracks gate logit statistics separately for harmful/benign datasets using a `DifferentialAccumulator`, computes per-expert differential scores, and classifies experts into HCDG/HRCG groups via `SafetyReport`
- **Steering** (`steering.py`) injects pre-computed bias arrays into gate logits before top-k selection. Per-model-type hooks handle different gating mechanisms (sigmoid for MiniMax/GLM4, softmax for Qwen3). Biases can be hot-swapped at runtime.
- **Abliterate** (`abliterate.py`) hooks decoder layers (not MoE blocks) to capture residual stream, computes refusal directions, and orthogonalizes weight matrices. Supports per-expert targeting via `_orthogonalize_expert_proj`.
- **Domain** (`domain.py`) reuses `DifferentialAccumulator` from `safety.py` for domain-vs-general differential analysis, classifies experts into domain-specialized and general groups via `DomainReport`, computes amplification biases, and permanently modifies gate parameters (nn.Linear bias or correction_bias) for hook-free inference
- **Server** composes on mlx-lm's `APIHandler` and `ResponseGenerator`, uses compound counting+steering hooks that accumulate statistics AND apply steering in a single `__call__`, with REST endpoints for runtime steering control (supports safety_map, domain_map, and direct config)
- **Frontend** (`frontend.py`) is a Gradio web app that connects to the running server via HTTP. Provides chat (streaming via SSE), expert activation heatmaps (matplotlib), steering controls, and server management. Launched via `mlx-fun ui`.

## Testing

```bash
uv pip install -e ".[dev]"
pytest tests/ -v
```

227 tests covering:
- Adapter factory detection and attribute access (MiniMax, MiniMax-M2, GLM4-MoE, GLM4-MoE-Lite, Qwen3-MoE, Qwen3-Next)
- Saliency math (formula verification, zero-division guards, save/load roundtrip)
- Observer hooks (install/remove, capture shapes, numerical equivalence with/without hooks)
- Pruner (bottom + strided expert selection, quantized gate slicing, tensor shapes after slicing, zero-prune identity, edge cases)
- REAM hooks (input/gate-logit capture shapes, numerical equivalence, install/remove cycle for all model types)
- Merger (centroid selection, expert grouping, gated/average similarity, greedy/none alignment, weight shape preservation, single-member groups, gate slicing, config updates, merged model forward pass)
- Dataset loading (chat messages, completions, plain text, mixed formats, subsampling)
- Server (OnlineAccumulator thread safety, lightweight/full counting hooks for all model types, numerical equivalence, .npz compatibility, management endpoints)
- Safety analysis (DifferentialAccumulator stats, differential scoring/normalization, HCDG/HRCG classification, SafetyReport save/load, per-model-type top-k routing replication)
- Steering (SteeringConfig serialization, bias computation, hook install/remove for MiniMax and Qwen3, deactivation output changes, hot-swap config updates)
- Abliteration (linear orthogonalization removes direction component, preserves orthogonal directions, idempotent, batched SwitchLinear orthogonalization, single expert orthogonalization, auto layer selection)
- Domain identification (DomainReport save/load, domain/general classification, amplification bias computation with scale/threshold, gate weight amplification for MiniMax/GLM4/Qwen3, steering from domain report boost/suppress modes, pruner domain constraints and protection)
- Frontend (API client helpers, streaming chat, heatmap/bar chart visualization, error handling with mock HTTP server)
- CLI (command registration, required arguments)

## References

- **REAP paper**: [Routing-based Expert Activation Pruning](https://github.com/CerebrasResearch/reap) — Cerebras Research
- **REAM blog**: [Router-weighted Expert Activation Merging](https://bknyaz.github.io/blog/2026/moe/) — Boris Knyazev
- **SAFEx paper**: [Are Safety Experts Safe? Stable Safety-Critical Expert Identification in MoE](https://arxiv.org/abs/2506.17368) — NeurIPS 2025
- **SteerMoE paper**: [SteerMoE: Adaptive Expert Steering for MoE Safety](https://arxiv.org/abs/2509.09660)
- **Abliteration paper**: [Refusal in Language Models Is Mediated by a Single Direction](https://proceedings.neurips.cc/paper_files/paper/2024/file/f545448535dfde4f9786555403ab7c49-Paper-Conference.pdf) — Arditi et al., NeurIPS 2024
- **MLX**: [ml-explore/mlx](https://github.com/ml-explore/mlx) — Apple's array framework for Apple Silicon
- **mlx-lm**: [ml-explore/mlx-lm](https://github.com/ml-explore/mlx-lm) — Language model tooling for MLX
