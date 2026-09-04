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

Each expert gets a composite score, the per-layer-normalized blend of two
differentials: the selection frequency (how often the expert is in the top-k)
and the mean routed weight. Experts are bucketed:

- **HCDG** (Harmful Content Detection Group) — routed MORE by harmful **prompts**
  (high composite score).
- **HRCG** (Harmful Response Control Group) — routed MORE by benign **prompts**
  (low composite score).
- **Safety-critical** — union of HCDG and HRCG.

> **What this does and does not measure.** `safety-scan` forwards prompt tokens
> only; it never observes the model's *response*. So these groups describe which
> experts the two prompt distributions route to differently — not, on their own,
> the machinery the model uses to refuse. Treat HRCG as "benign-prompt-preferred
> experts", and verify any refusal claim causally (e.g. with `domain-probe`'s
> knockout, or by regenerating under a mask and classifying the output).

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

## Domain probe — ask the model questions

`domain-scan` compares routing over raw corpora. `domain-probe` instead asks the
model a curated set of questions and watches which experts it uses while it
*answers* them, then verifies the top candidates causally:

```bash
mlx-fun domain-probe \
    --model ./model \
    --domain-questions ./data/probes/solidity.jsonl \
    --general-questions ./data/probes/general.jsonl \
    --output probe_report.json \
    --saliency-output probe.npz \
    --domain-name solidity \
    --verify-top 32 --verify-prune 8 --seed 42
```

Four question sets ship with the repo — two domain sets and two contrast sets:

| File | Role | Contents |
|---|---|---|
| `data/probes/solidity.jsonl` | domain | General smart-contract development |
| `data/probes/security.jsonl` | domain | Vulnerability discovery, EVM internals, Yul |
| `data/probes/solidity_benign.jsonl` | contrast | Ordinary Solidity, no security content |
| `data/probes/general.jsonl` | contrast | Python, math, trivia |

Pair them deliberately, because the contrast decides what the probe isolates.
`security.jsonl` against `general.jsonl` finds broad smart-contract experts;
`security.jsonl` against `solidity_benign.jsonl` holds Solidity knowledge
constant on both sides, so what survives is the reasoning about what can go
wrong:

```bash
mlx-fun domain-probe \
    --model ./model \
    --domain-questions ./data/probes/security.jsonl \
    --general-questions ./data/probes/solidity_benign.jsonl \
    --output security_report.json --saliency-output security.npz \
    --domain-name security --verify-top 32 --verify-prune 8 --seed 42
```

See [datasets](datasets.md#choosing-a-pair) for the full pairing table and
[Calibration corpora](datasets.md#calibration-corpora) for the matching raw
corpus used by `collect` and `domain-scan`.

### Question format

```json
{"question": "What does the `payable` modifier do?",
 "answer": "It allows a function to receive Ether; calls with value to a non-payable function revert.",
 "tags": ["functions", "ether"],
 "system": null}
```

`question` is required. `answer` is required in teacher mode; `tags` and
`system` are optional (a per-question `system` overrides `--system`).

### Answer modes

- `--answer-mode teacher` (default) — one forward pass over the chat-templated
  question plus its reference answer. Deterministic, fast, and works for vision
  checkpoints through their language stack.
- `--answer-mode generate` — the model writes its own answer with the hooks
  live during decoding. `--answers-output` dumps what it wrote in the probe
  schema, so a generate run can be replayed in teacher mode. Not available for
  vision models.

### How relevance is scored

Routing is attributed to the positions that *produce* the answer tokens —
`[prompt_len-1 : prompt_len-1+n_answer]` — because logits at position `t`
predict token `t+1`. The observational score and the knockout delta therefore
describe exactly the same predictions, and the routing that picks the first
answer token is included.

Each question contributes one vector normalized by its own answer length, so a
verbose answer cannot outvote a terse one. The composite is the same per-layer
min-max blend `safety-scan` uses, over the differential selection frequency and
the differential routed weight. `--min-coverage` drops domain experts that fired
in too few questions, which is the usual source of a high score built on noise.

### Knockout verification

`--verify-top N` masks each of the top N experts out of the router, one at a
time, and measures how much the answer log-likelihood degrades. The mask is
added to the gate parameter that enters the *selection score* — the
post-sigmoid correction bias for MiniMax and the GLM/DeepSeek family, a
temporary pre-softmax `gate.bias` for the Qwen family — so the model's own
routing runs untouched: grouped selection, `norm_topk_prob`,
`routed_scaling_factor` and any latent projections all still apply. There is no
surrogate router and no hook during scoring.

Masking pre-sigmoid would not be enough: MiniMax and GLM select on
`sigmoid(logits) + correction_bias`, so an expert with a large positive
correction can survive a large negative pre-sigmoid bias.

Deltas are paired per question. A masked run that goes non-finite is counted as
a collapse rather than dropped, so a mask that destroys half the answers cannot
average out to "harmless". Each candidate gets a status:

| status | meaning |
|---|---|
| `verified` | mean delta ≥ `--min-delta` and the bootstrap CI excludes zero |
| `not_verified` | no credible degradation |
| `inconclusive` | usable pairs below `--min-valid-fraction` |
| `catastrophic` | fewer than half the questions survived the mask |

Only `verified` entries land in `verified_domain_experts`; `domain_experts` is
never narrowed by the knockout, since only the top N are checked.

The report records both `baseline_nll` (measured with a zero mask installed, so
a never-routed expert gives a delta of exactly zero) and `plain_baseline_nll`
(no mask at all). Small differences between them are numerical, not routing.

### Prune-set verification

`--verify-prune N` masks the exact set `prune` would remove — the same
`build_keep_map` call, with the domain experts protected — and reports the paired
delta on both question sets:

```
  Masked 2496 expert-layer pairs
    domain: delta=+0.0812 nats/token (degraded)
    general: delta=+0.3140 nats/token (degraded)
```

`--verify-metric`, `--verify-strategy`, `--verify-model-wide` and
`--verify-min-experts-per-layer` mirror the corresponding `prune` flags. Note
this masks experts in the *original* router; a real pruned checkpoint has a
smaller expert axis, so confirm the result with `smoke-test` on the pruned
model.

### Outputs

`--output` writes a superset of `domain_report.json`, so `prune --domain-map`,
`amplify`, `serve --domain-map` and the steering API read it unchanged.
`--saliency-output` writes a `SaliencyAccumulator` `.npz` built from the domain
answer tokens, for `prune --saliency`:

```bash
mlx-fun prune --model ./model \
    --saliency probe.npz --metric freq --n-prune 32 \
    --domain-map probe_report.json --domain-mode protect \
    --output ./pruned
```

By default each question contributes equally to that `.npz`
(`--saliency-weighting question`); pass `token` for raw per-token counts.

### Cost

The trace is one forward pass per question. The knockout is
`(2 + N) x questions` forward passes, plus `2 x questions` more for the prune
check. On a 78-layer model with 80 questions and `--verify-top 32` expect tens
of minutes; `--verify-questions`, a smaller `--verify-top`, and omitting
`--verify-prune` are the levers.

Thinking-mode templates are best disabled for probing:
`--chat-template-args '{"enable_thinking": false}'`.

Gemma 4 is not supported for knockout verification; the trace pass works.

## Refusal probe — find the guardrail experts

The domain probe finds experts used to *answer* a domain. `refusal-probe` finds
the experts that implement the model's *refusals* — the guardrails — and is a
distinct objective, so it is a separate command.

It supplies no disallowed-intent dataset. The refusals are detected empirically:
it generates an answer to each question, classifies the response as answered,
refused or partial, and contrasts the routing on the questions the model
**refused** against the ones it **answered**. Experts routed more heavily while
refusing are the refusal machinery.

```bash
mlx-fun refusal-probe \
    --model ./model \
    --questions ./data/probes/security.jsonl \
    --output refusal_report.json \
    --answers-output refusal_answers.jsonl \
    --verify-top 16 --seed 42
```

### Classification

`classify_response` is a conservative heuristic (a phrase list, not a model, so
it is auditable and dependency-free). Text is normalized first — lowercased,
curly apostrophes folded to ASCII — so "I can’t help" is caught, and the
markers are decline-specific ("I can’t", "I cannot", "I won’t"), so a
helpful "I’m sorry, but I can explain…" is *not* a refusal. A marker
near the start means the model declined; a marked response that then continues
well past a length threshold is `partial` (a marker is present, so it is never
treated as compliance). Only marker-free responses are `answered`. Widen
detection with `--refusal-markers <file>` (one phrase per line).

The list includes common paraphrased declines ("I'd rather not help…",
"that's not something I'm going to write…"), kept long and specific so
technical prose ("I'll pass the array by calldata", "I won't write more than
24576 bytes") does not trigger them.

Lexical **similarity** to refusal templates was evaluated on realistic
responses and rejected: paraphrased refusals and technical answers overlap with
no clean threshold (a real "I'd rather not help build something designed to
steal funds" scores the same ~0.42 as an answer explaining a flash loan), and
similarity blurs the decline/offer distinction the markers key on. The marker
list has high precision but a recall ceiling on *novel* paraphrasing that no
lexical method closes — the ceiling-breaker is a semantic classifier (an
embedding judge or an LLM judge). `classify_response` is the single seam where
one would slot in; `--refusal-markers` is the cheap lever until then.

Only refused and answered responses define the signal; partials are recorded
but excluded, and a `partial` never counts as a knockout flip.

If the model refuses none of the questions there is no signal to isolate, and
the command says so and stops — try a set it actually declines, a stricter
`--system` prompt, or wider markers.

### Verification is by regeneration, not likelihood

A log-likelihood delta cannot tell a reworded refusal from real compliance. So
knockout here **regenerates**: it masks a candidate expert out of the real
router, re-runs each refused question, and re-classifies. The **flip rate** is
the fraction that turned from a refusal into an answer. An expert is verified
when the flip rate clears `--min-flip-rate` (default 0.5) and its bootstrap CI
excludes zero. This measures the guardrail directly. Only a fully clean answer
(no refusal marker at all) counts as a flip — a reworded or hedged refusal does
not "verify" an expert the model still refuses with.

The mask is the same real-router selection-score mask the domain knockout uses,
so grouped selection and the correction bias are intact; Gemma 4 is unsupported.

### Candidates vs verified experts

The differential ranks *candidates* — experts correlated with refusing. Only
`--verify-top` of them are knockout-tested, and only those that actually flip
refusals become **verified**. The report keeps the two sets explicit
(`candidate_refusal_experts`, `verified_refusal_experts`), and `domain_experts`
— what the downstream tools read — defaults to the **verified** set when a
knockout ran, so a safety action never disables a merely-correlated expert. With
`--verify-top 0` no verification runs, `domain_experts` falls back to the
candidates, and `refusal_experts_verified` is False (the command warns).

### Controlling topic confounding

Refused and answered questions differ in subject, so the raw contrast can rank a
*topic* expert (security content) above a guardrail expert — and a true
guardrail expert might not reach `--verify-top`. `--stratify-tags` (on by
default) contrasts refused-vs-answered **within each tag** and averages across
the tags that contain both outcomes, holding topic roughly constant. It falls
back to the global contrast, with a warning, when no tag has both. The knockout
is the backstop regardless: a topic expert does not flip refusals, so it never
becomes verified. For sharper isolation, tag the questions finely, or supply
matched allowed/disallowed prompts within each topic.

### Vision models

Unlike the domain probe's generate mode, `refusal-probe` runs on vision models:
it generates on the unwrapped language stack (token ids only, no pixels) through
a logits shim. (This path has unit coverage but has not been validated on real
VLM weights in this repo — no small VLM is available here.)

### Output

The report is a `domain_report.json` superset with `objective: "refusal"`, the
answered/refused/partial counts, the per-question outcomes (each with its
question text, so a record is identifiable despite `--max-questions`
subsampling), the candidate/verified expert sets, and the flip-rate knockout
block. `domain_experts` holds the verified set, so the file loads with the
existing tools — but note the useful guardrail operation is to **deactivate** or
**prune** those experts, not to protect them, and you should act on the
**verified** set:

```bash
# Deactivate the VERIFIED refusal experts at inference via the steering API
curl -X POST http://localhost:8080/v1/reap/steer \
    -d '{"deactivate": <verified_refusal_experts from the report>, "mask_value": -1e9}'
```

`--saliency-output` writes a `SaliencyAccumulator` `.npz` from the refused-answer
routing for `prune --saliency`.

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
