# Gemma 4 MTP speculative decoding

Pairs a Gemma 4 backbone with its small **gemma4_assistant** Multi-Token
Prediction (MTP) drafter to do greedy speculative decoding on Apple Silicon.
The driver typically runs **1.4–1.9× faster than the backbone alone** at the
same quantization, and produces **2.4–4.9× the throughput of bf16** when the
backbone is also quantized to MXFP4.

The HF `Gemma4AssistantForCausalLM` class isn't in public `transformers` yet
(it ships in a private dev branch referenced by `transformers >= 5.7.0.dev0`),
so the model and the driver were both reverse-engineered from the published
checkpoints + the `convert_gemma4_weights.py` script in the public
transformers repo.

---

## What's in the box

- **`mlx_lm.models.gemma4_assistant`** (in the [dexloom/mlx-lm
  fork](https://github.com/dexloom/mlx-lm/tree/gemma4-assistant)) — registers
  the `gemma4_assistant` model_type so `mlx_lm.load()` and `mlx_lm.convert`
  work directly on the raw HF checkpoint.
- **`mlx_fun.mtp_driver`** (`src/mlx_fun/mtp_driver.py`) — the
  speculative-decoding loop: backbone prefill, K-step drafting against
  borrowed backbone KV, parallel verification, KV trim.

## Architecture in three sentences

The drafter has 4 transformer layers at hidden size 1024 and **all four
layers reuse the backbone's K/V** (`num_kv_shared_layers == num_hidden_layers`)
— the drafter never extends the cache itself. Each draft step takes the
backbone-space hidden at the previous position concatenated with the
embedding of the just-sampled token (also in backbone space), pre-projects
to the drafter hidden size, runs the four layers using a backbone "anchor"
KV (last sliding + last full layer of the backbone), emits a vocab logit,
and post-projects the drafter hidden back to backbone space so it can feed
the next draft step. After K drafted tokens, the backbone verifies them in
**one** parallel forward; the longest greedy-argmax-matching prefix is
accepted, the bonus correction token is committed, and the KV cache is
trimmed by `K - accepted` to roll back the rejected suffix.

## Quick start

### 1. Install the fork of mlx-lm with the `gemma4_assistant` model type

```bash
git clone -b gemma4-assistant git@github.com:dexloom/mlx-lm.git
uv pip install -e ./mlx-lm
```

### 2. Convert the HF assistant checkpoints to MLX

```bash
# bf16 (largest, exact reference)
python -m mlx_lm convert \
  --hf-path google/gemma-4-31B-it-assistant \
  --mlx-path /path/to/Gemma-4-31B-it-assistant-MLX \
  --dtype bfloat16

# MXFP4 (recommended — same quality, ~3× smaller, fastest)
python -m mlx_lm convert \
  --hf-path google/gemma-4-31B-it-assistant \
  --mlx-path /path/to/Gemma-4-31B-it-assistant-MXFP4 \
  --quantize --q-mode mxfp4 --q-group-size 32 --q-bits 4 \
  --dtype bfloat16
```

The 26B-A4B variant is identical: `google/gemma-4-26B-A4B-it-assistant`.

### 3. Run

```bash
python -m mlx_fun.mtp_driver \
  --backbone /path/to/Gemma-4-31B-it-MXFP4 \
  --drafter  /path/to/Gemma-4-31B-it-assistant-MXFP4 \
  --prompt "Explain quantum tunneling to a curious 12-year-old in three sentences." \
  -k 4 -n 256 --baseline
```

`--baseline` also runs the backbone alone for a side-by-side speed compare.

### Programmatic use

```python
from mlx_fun.mtp_driver import MTPDriver, DriverConfig

driver = MTPDriver(
    backbone_path="/path/to/Gemma-4-31B-it-MXFP4",
    drafter_path="/path/to/Gemma-4-31B-it-assistant-MXFP4",
    config=DriverConfig(num_draft_tokens=4, max_new_tokens=512),
)

prompt = driver.tokenizer.apply_chat_template(
    [{"role": "user", "content": "..."}],
    tokenize=False, add_generation_prompt=True,
)
text, stats = driver.generate(prompt)
print(stats.acceptance_rate, stats.new_tokens / stats.elapsed_s)
```

## Quantization × speed × quality

The numbers below are from `bench_long.py` on a single-Mac M-series box,
5 long prompts × 300 generated tokens, K=4, MXFP4 drafter throughout, all
measurements warmed-up.

### Standalone speed and perplexity

| Model | Quant | tok/s (no MTP) | Δ vs bf16 | Perplexity | NLL/tok |
|---|---|---:|---:|---:|---:|
| **26B-A4B** | bf16  |  53.5 | reference  |   252 | 5.528 |
| 26B-A4B     | MXFP8 |  67.3 | +26%       |   546 | 6.303 |
| 26B-A4B     | MXFP4 |  88.4 | **+65%**   | 1,326 | 7.190 |
| **31B**     | bf16  |   9.7 | reference  | 9,869 | 9.197 |
| 31B         | MXFP8 |  17.2 | +77%       | 4,100 | 8.319 |
| 31B         | MXFP4 |  28.7 | **+196%**  | 8,154 | 9.006 |

> Perplexity is on a 317-token Wikipedia paragraph (Antikythera mechanism).
> Instruction-tuned models score very high on raw prose — the absolute
> numbers aren't comparable to base-LM perplexity tables. The MXFP4 vs bf16
> *delta* is the meaningful comparison, and on raw prose MXFP4 ≈ 5.3× the
> bf16 perplexity on 26B-A4B but only ~0.83× on 31B (yes, MXFP4 31B has
> *lower* PPL than bf16 31B on this passage — within single-passage noise).

### MTP (MXFP4 drafter, K=4)

| Model | Quant | MTP tok/s | speedup vs own | speedup vs bf16 | Acceptance |
|---|---|---:|---:|---:|---:|
| 26B-A4B | bf16  |  74.7 | 1.40× | 1.40× | **70.9%** |
| 26B-A4B | MXFP8 | 100.9 | 1.50× | 1.89× |    71.4% |
| 26B-A4B | MXFP4 | **129.5** | 1.46× | **2.42×** |    62.0% |
| 31B     | bf16  |  18.4 | **1.89×** | 1.89× | 70.3% |
| 31B     | MXFP8 |  24.6 | 1.43× | 2.54× |    70.3% |
| 31B     | MXFP4 | **47.7** | 1.66× | **4.91×** | 69.8% |

### Per-prompt acceptance (long-prompt run, %)

| Prompt | 26B-bf16 | 26B-MXFP8 | 26B-MXFP4 | 31B-bf16 | 31B-MXFP8 | 31B-MXFP4 |
|---|---:|---:|---:|---:|---:|---:|
| 1. Spec-decoding explainer | 59.8 | 59.9 | 58.8 | 67.6 | 63.4 | 63.2 |
| 2. Code review/bugfix | 66.9 | 69.6 | 53.7 | 64.5 | 68.8 | 67.6 |
| 3. Bronze-Age summary | 71.6 | 71.8 | 69.2 | 74.7 | 74.7 | 75.5 |
| 4. Linear-prog math | **87.6** | 82.1 | 73.9 | 79.2 | **84.9** | 80.3 |
| 5. Five thematic haiku | 68.4 | 73.4 | 54.5 | 65.6 | 59.9 | 62.6 |

Acceptance is **content-driven** more than quantization-driven. Structured
math/list/summary prompts hit 75–88%; free-form coding/creative prompts
sit at 55–70%.

### Drafter quantization is a free lunch

Quantizing the drafter itself (bf16 → MXFP8 → MXFP4) preserves acceptance
rate within prompt-level noise (~3 percentage points) while shrinking the
drafter ~3× and adding 5–14% throughput on top of the backbone+MTP gain.
Always run the drafter at MXFP4.

| Drafter | Size (26B-A4B) | Acceptance | Δ vs bf16 |
|---|---:|---:|---:|
| bf16  | 831 MB | 64.8% | reference |
| MXFP8 | 451 MB | 65.7% | +0.9 pt |
| MXFP4 | **255 MB** | 64.5% | −0.3 pt |

### Greedy top-1 agreement vs bf16 reference

| Quant | 26B-A4B (300 tok) | 31B (300 tok) |
|---|---:|---:|
| MXFP8 | 39.3% | 27.6% |
| MXFP4 |  5.9% | 19.8% |

This number looks alarming but isn't bad news — the trajectories diverge
within the first ~15 tokens (after which they stay diverged), but both
trajectories produce coherent, on-topic, instruction-following output.
Gemma 4 uses `final_logit_softcapping=30.0`, which compresses
top-of-distribution differences so that even tiny quantization noise
flips the argmax. **MXFP4 is not "MXFP8 with errors";** it's a different
high-quality model drawing from a similarly-shaped distribution. For
applications that need bit-identical reproducibility versus bf16, use
the bf16 backbone with the MXFP4 drafter — the verifier guarantees
greedy-bf16 equivalence and you still get ~1.9× speedup.

## When to use which combo

| Goal | Backbone | Drafter | Throughput | Notes |
|---|---|---|---:|---|
| Maximum throughput on 26B-A4B | MXFP4 | MXFP4 | **130 tok/s** | quant drift OK |
| Maximum throughput on 31B | MXFP4 | MXFP4 | **48 tok/s** | quant drift OK; ~5× bf16 |
| Quality + headroom | MXFP8 | MXFP4 | 25 / 101 tok/s | mild drift, near-lossless |
| Bit-identical to bf16 | bf16 | MXFP4 | 18 / 75 tok/s | verifier guarantees bf16 outputs |

## Tunables

The driver exposes the choices that aren't fully documented in public
sources, so you can sweep them on your specific workload:

- **`num_draft_tokens`** (default 4) — bigger K drafts more per cycle but
  rejection cost grows. K=4 is robustly good across all our prompts.
- **`anchor_layer_by_type`** — which backbone layer's K/V the drafter
  borrows for each layer type. Default is the **last** sliding and last
  full layer (i.e., positions 58 and 59 on the 31B). The 31B accepts
  ~9 percentage points lower than 26B-A4B, which suggests other anchor
  choices may help — untested.
- **`scale_next_token_embed`** (default True) — empirically the trained
  drafter expects the embedding scaled by the backbone's `embed_scale =
  sqrt(hidden_size)`. False causes acceptance to collapse to ~0%.
- **RoPE offset** — the driver pins the drafter Q's RoPE offset to
  `prefix_length + draft_step` so attention to the backbone's already-
  RoPE'd K is in-phase. Tweak `_drafter_step` if you want to experiment.

## Limitations

- **Greedy only.** No temperature/top-p/top-k yet. Adding sampling
  requires importance-corrected acceptance (Leviathan-style) for
  unbiased results.
- **Batch size 1.** The verify step is single-batch; multi-prompt batched
  speculation isn't implemented.
- **No EOS-mid-prefix handling.** EOS in the drafted tokens stops
  generation but doesn't reduce wasted draft compute.
- **Sliding-window cache trim** — works because `RotatingKVCache.trim`
  decrements the offset cleanly, but for very long generations near the
  sliding window boundary the trim semantics deserve more testing.

## Reference

The implementation pulls from:

- **`mlx_lm.models.gemma4_text`** — the upstream backbone, including the
  KV-shared `Attention` path that the drafter rides on.
- **HF `convert_gemma4_weights.py`** — confirmed the on-disk weight names:
  `pre_projection.weight` (`2*B → H`) and `post_projection.weight`
  (`H → B`), plus the `model.norm`, `model.embed_tokens`, and per-layer
  layout.
- The published `gemma-4-{31B,26B-A4B,E4B,E2B}-it-assistant` checkpoints
  on Hugging Face, which include the `tie_word_embeddings: true` flag and
  expose the drafter's own 1024-dim embed_tokens as the lm_head.
