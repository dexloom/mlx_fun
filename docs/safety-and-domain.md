# Safety & Domain analysis

Differential routing analysis for two related problems:

- **Safety** (SAFEx): identify experts that activate disproportionately on
  harmful content, then steer / prune / abliterate them.
- **Domain specialization**: identify experts that activate disproportionately
  on a domain (Solidity code, medical text, etc.), then protect / amplify them.

Both reuse the same `DifferentialAccumulator` and scoring math from
`src/mlx_fun/safety.py`.

## Safety scan (SAFEx)

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
| `--model` | *(required)* | Model path or HF repo |
| `--harmful-dataset` | *(required)* | Harmful prompts (JSONL or directory) |
| `--benign-dataset` | *(required)* | Benign prompts (JSONL or directory) |
| `--output` | *(required)* | Output `safety_report.json` |
| `--max-samples` | 128 | Max samples per dataset |
| `--max-tokens` | 2048 | Max tokens per sample |
| `--text-key` | `content` | JSON key |
| `--threshold-percentile` | 90.0 | Classification threshold |
| `--seed` | *(none)* | Random seed |

### Classification

Each expert gets a composite score combining differential frequency,
weighted frequency, and routing entropy. Experts are bucketed:

- **HCDG** (Harmful Content Detection Group) — activate MORE on harmful content
  (high composite score).
- **HRCG** (Harmful Response Control Group) — activate MORE on benign content
  (low composite score) — these are the experts the model uses to refuse.
- **Safety-critical** — union of HCDG and HRCG.

The report is consumed by `prune --safety-map`, `steer`, `abliterate --target
safety-experts`, and the server's steering API.

## SteerMoE (`steer`)

```bash
mlx-fun steer \
    --model mlx-community/Qwen3-30B-A3B-4bit \
    --safety-map safety_report.json \
    --mode safe \
    --prompt "How do I make a bomb?" \
    --max-tokens 100
```

A pre-computed bias array of shape `(num_experts,)` is added to raw gate
logits before top-k selection.

| Mode | What happens |
|---|---|
| `safe` | HRCG experts get `+boost_value` bias → ensured selection |
| `unsafe` | All safety-critical experts get `mask_value` bias → effectively removed |

| Flag | Default | Description |
|---|---|---|
| `--mask-value` | -1e9 | Bias for deactivation |
| `--boost-value` | 1e4 | Bias for activation |
| `--kv-compress` | *(none)* | `turbo` / `rotor` |

Hooks are compound (counting + steering in one `__call__`) to avoid `__class__`
swap conflicts.

## Abliteration (refusal direction removal)

```bash
mlx-fun abliterate \
    --model mlx-community/Qwen3-30B-A3B-4bit \
    --harmful-dataset ./data/harmful.jsonl \
    --benign-dataset ./data/benign.jsonl \
    --output ./abliterated_model \
    --layers auto --target all --max-samples 64
```

| Flag | Default | Description |
|---|---|---|
| `--layers` | `auto` | `auto` (top 50% by refusal direction norm), `all`, or `start-end` (e.g. `10-20`) |
| `--target` | `all` | `all`, `safety-experts`, or `dense-only` |
| `--safety-map` | *(none)* | Required if `--target safety-experts` |
| `--extraction-position` | `last` | `last` (last token) or `mean` (mean pool) |

**How it works:**

1. **Capture** — Hook decoder layers, forward harmful + benign prompts,
   extract residual stream activations at the chosen position.
2. **Compute direction** — `refusal_direction[layer] = normalize(mean_harmful - mean_benign)`.
3. **Select layers** — Auto-select top 50% by refusal direction norm, or use
   the user-specified range.
4. **Orthogonalize** — Project the direction out of weight matrices:
   `W' = W - (W @ d) * d^T`.

**Targets:**

- `all` — orthogonalize all expert `down_proj` weights + attention `o_proj`.
- `safety-experts` — only safety-critical experts identified by `safety-scan`.
- `dense-only` — skip MoE experts; modify only attention and shared experts.

The output writes `abliteration_metadata.json` with method, target, layers,
and direction norms.

## Domain scan

Identifies experts that activate more on domain-specific data vs general
data. Uses the same differential machinery as `safety-scan`:

```bash
mlx-fun domain-scan \
    --model mlx-community/MiniMax-M1-40k-4bit \
    --domain-dataset ./data/solidity_calibration.jsonl \
    --general-dataset ./data/general_text.jsonl \
    --output domain_report.json \
    --domain-name solidity \
    --max-samples 128 --seed 42
```

Classifies experts into:

- **Domain experts** — activate MORE on domain data (high composite, above the
  threshold percentile).
- **General experts** — activate MORE on general data (below inverse threshold).

The report feeds `prune --domain-map`, `amplify`, `serve --domain-map`, and
the server's steering API.

## Amplify — permanent domain gate boost

`amplify` permanently modifies gate weights so domain-specialized experts
are favored natively. The output works with stock `mlx_lm.load()` — no hooks
needed at inference time.

```bash
mlx-fun amplify \
    --model mlx-community/MiniMax-M1-40k-4bit \
    --domain-map domain_report.json \
    --output ./amplified_model \
    --scale 1.0 --threshold 0.0
```

Per domain expert: `boost = scale * max(0, composite_score - threshold)`.
The boost lands differently per architecture:

| Model | Gate type | Target | Position |
|---|---|---|---|
| MiniMax / MiniMax-M2 | `nn.Linear(bias=False)` | `gate.bias` | pre-sigmoid |
| GLM4 / GLM4-Lite / GLM5 / DSv3 | `MoEGate` | `gate.e_score_correction_bias` | post-sigmoid |
| Qwen3 / Qwen3-Next | `nn.Linear(bias=False)` | `gate.bias` | pre-softmax |

Output writes `amplification_metadata.json` with domain name, scale,
threshold, and per-layer biases.

**Runtime alternative** (no permanent modification):

```bash
mlx-fun serve --domain-map domain_report.json --domain-steering-mode boost
```

## Combined safety + domain

The protected sets merge via union — you can protect both safety and domain
experts simultaneously:

```bash
mlx-fun prune \
    --model ./model --saliency ./saliency.npz --output ./pruned --n-prune 16 \
    --safety-map safety_report.json --safety-mode protect \
    --domain-map domain_report.json --domain-mode protect
```

`protect` sets the safety/domain experts' saliency to `+inf` (never bottom-n).
`target` sets them to `-inf` (always bottom-n).
