# MLX-FUN

**Expert pruning, merging, steering, and inference optimization for
Mixture-of-Experts models on Apple Silicon.**

MLX-FUN is an MLX-native toolkit for compressing, analyzing, and optimizing
MoE language models on Apple Silicon via [MLX](https://github.com/ml-explore/mlx).
It runs alongside [mlx-lm](https://github.com/ml-explore/mlx-lm) and produces
checkpoints that load with stock `mlx_lm.load()`.

---

## Features at a glance

### Model compression

| Feature | What it does | Docs |
|---|---|---|
| **REAP** | Remove least-important experts by saliency scoring + weight slicing | [algorithms](docs/algorithms.md#the-reap-algorithm) |
| **REAM** | Merge experts around centroids via neuron-aligned weighted averaging | [algorithms](docs/algorithms.md#ream-expert-merging) |
| **Quantize** | bf16 → MLX-native (affine / MXFP4 / MXFP8 / NVFP4), mixed-precision NVFP4+MXFP8, INT4-source streaming requantize | [quantization](docs/quantization.md) |

### Safety & steering

| Feature | What it does | Docs |
|---|---|---|
| **SAFEx** | Identify safety-critical experts via differential routing | [safety-and-domain](docs/safety-and-domain.md#safety-scan-safex) |
| **SteerMoE** | Inject bias into gate logits to (de)activate experts at inference time | [safety-and-domain](docs/safety-and-domain.md#steermoe-steer) |
| **Abliteration** | Remove refusal directions from weight matrices via orthogonalization | [safety-and-domain](docs/safety-and-domain.md#abliteration-refusal-direction-removal) |

### Domain specialization

| Feature | What it does | Docs |
|---|---|---|
| **Domain Scan** | Find domain-specialized experts via differential routing | [safety-and-domain](docs/safety-and-domain.md#domain-scan) |
| **Amplify** | Permanently bias gate weights so domain experts are favored natively | [safety-and-domain](docs/safety-and-domain.md#amplify--permanent-domain-gate-boost) |

### Inference optimization

| Feature | What it does | Docs |
|---|---|---|
| **TurboQuant** | KV cache compression via PolarQuant rotation (4–6× memory) | [kv-cache](docs/kv-cache.md#turboquant-polarquant) |
| **RotorQuant** | KV cache compression via Cl(3,0) Clifford rotors (44× fewer rotation params) | [kv-cache](docs/kv-cache.md#rotorquant-clifford-rotors) |
| **Sliding window** | Cap KV cache to N tokens per layer for bounded memory | `--max-kv-size` |
| **Gemma 4 MTP drafter** | Greedy speculative decoding with the `gemma4_assistant` drafter — 1.4–1.9× over backbone, up to 4.9× over bf16 | [mtp-speculative-decoding](docs/mtp-speculative-decoding.md) |

### Tools & dashboard

| Feature | What it does | Docs |
|---|---|---|
| **Online serving** | OpenAI/Anthropic-compatible API with optional live expert counting + steering | [serving](docs/serving.md) |
| **Web dashboard** | Gradio UI: chat, heatmaps, steering, diff analysis, expert filter export | [serving](docs/serving.md#web-dashboard-gradio) |
| **Tool-call patcher** | Permissive Kimi-K2.6 parser for quantized models that emit format quirks | [tool-parsers](docs/tool-parsers.md) |

---

## Install

Requires Python 3.11+ and Apple Silicon.

```bash
cd mlx_fun
uv venv && uv pip install -e ".[dev]"
```

Optional extras: `.[ream]` (scipy for Hungarian alignment), `.[ui]` (Gradio
dashboard), `.[dataset]` (HF dataset prep), `.[convert]` (NVFP4 reader).

---

## Quick start

The main pipeline is **collect → prune (or merge) → smoke-test**, with
**serve** for online collection.

### 1. Calibrate on your domain

```bash
mlx-fun collect \
    --model mlx-community/MiniMax-M1-40k-4bit \
    --dataset ./data/solidity_calibration.jsonl \
    --output ./saliency.npz \
    --max-samples 128 --seed 42
```

Dataset accepts JSONL (chat / prompt-completion / plain text) or a directory
of source files. See [`datasets.md`](docs/datasets.md) for formats and a
prep script for Solidity.

### 2. Prune low-saliency experts

```bash
mlx-fun prune \
    --model mlx-community/MiniMax-M1-40k-4bit \
    --saliency ./saliency.npz \
    --output ./pruned_model \
    --n-prune 16 --metric reap --strategy bottom
```

Or merge instead of pruning (preserves more knowledge, slower):

```bash
mlx-fun merge \
    --model mlx-community/MiniMax-M1-40k-4bit \
    --saliency ./saliency.npz \
    --dataset ./data/solidity_calibration.jsonl \
    --output ./merged_model \
    --n-prune 16
```

See [`algorithms.md`](docs/algorithms.md) for `bottom` vs `strided`,
per-layer vs `--model-wide`, and the REAM algorithm.

### 3. Smoke-test the result

```bash
mlx-fun smoke-test --model ./pruned_model --prompt "pragma solidity ^0.8.0;"
```

The output directory is a stock mlx-lm checkpoint — load it the usual way:

```python
from mlx_lm import load
model, tokenizer = load("./pruned_model")
```

### Online instead of offline calibration

Serve an OpenAI-compatible API with hooks that count expert activations from
real traffic:

```bash
mlx-fun serve \
    --model mlx-community/MiniMax-M1-40k-4bit \
    --port 8080 \
    --enable-counting \
    --auto-save ./online_saliency.npz
```

The accumulator dumps to `--auto-save` on shutdown. Hooks are off by default
— without `--enable-counting` the server is plain inference. See
[`serving.md`](docs/serving.md) for endpoints (`/v1/reap/{stats,save,reset,steer,info,gpu_limit}`),
runtime steering API, and the dashboard.

### Web dashboard

```bash
# Terminal 1
mlx-fun serve --model ./model --port 8080

# Terminal 2
mlx-fun ui --server-url http://127.0.0.1:8080
```

Tabs for chat, expert heatmaps, steering controls, file-based diff analysis,
and rank-based merge filtering with `filtered_experts.json` export for the
CLI.

---

## Quantization paths

For source quantization beyond pruning, mlx_fun supports the full MLX
toolbox:

```bash
# bf16 → MLX-native NVFP4 (4.5 bpw, near-MXFP8 quality at MXFP4 size)
.venv/bin/python local/scripts/quant_nvfp4.py SRC DST

# bf16 → mixed NVFP4 (routed experts) + MXFP8 (sensitive layers)
.venv/bin/python local/scripts/quant_glm51_nvfp4_mixed.py     # template

# compressed-tensors INT4 (Kimi-K2.6) → MLX affine Q3/Q4 streaming
.venv/bin/python local/scripts/convert_kimi_k2_int4_to_q3.py  # template

# NVIDIA modelopt NVFP4 (Nemotron) → MLX-native NVFP4
mlx-fun convert-nvfp4 --hf-path nvidia/.../NVFP4 --mlx-path ./out
```

See [`quantization.md`](docs/quantization.md) for the full landscape, the
mixed-precision recipe (NVFP4 expert bulk + MXFP8 attn/embed/lm_head), and
the streaming dequant→requant pattern for already-quantized sources.

---

## Supported architectures

MiniMax (M1/M2), GLM4-MoE, GLM4-MoE-Lite, Qwen3-MoE, Qwen3-Next, GLM-5
(GLM-MoE-DSA), DeepSeek V3.2, Kimi-K2/K2.6 (`kimi_k25` multimodal wrapper),
Nemotron-H (hybrid Mamba-2/Attn/MoE), Gemma4, GLM-5.3, Qwen4-Exp
(`Qwen/Qwen3.8-Flash-Next`), GLM-5.3-Flash. Both quantized and unquantized
sources work. New architectures plug in via the `BaseAdapter` interface.

**Vision-language models.** Qwen4-Exp (`Qwen/Qwen3.8-Flash-Next`) and
GLM-5.3-Flash (`zai-org/GLM-5.3-Flash`) are VLMs: mlx-lm does not implement
them, so mlx_fun loads them through
[mlx-vlm](https://github.com/Blaizzy/mlx-vlm) and runs the analysis stack
(saliency collection, SAFEx, domain scan, steering) against their language
half. Install the optional extra and point any analysis command at the
checkpoint as usual:

```bash
uv pip install -e ".[vlm]"
mlx-fun collect --model Qwen/Qwen3.8-Flash-Next --dataset ./data.jsonl --output stats.npz
```

`mlx_fun.loader.load_model()` picks the backend from `config.json`, so nothing
else in the workflow changes. Serving VLMs (image inputs on the OpenAI /
Anthropic endpoints) is not wired up — the server remains text-only.

---

## Documentation

| Doc | Topic |
|---|---|
| [`docs/algorithms.md`](docs/algorithms.md) | REAP, REAM, saliency metrics, pruning strategies, model-wide vs per-layer selection |
| [`docs/quantization.md`](docs/quantization.md) | NVFP4 native, mixed NVFP4+MXFP8, INT4-source streaming, NVIDIA NVFP4 reader |
| [`docs/tool-parsers.md`](docs/tool-parsers.md) | Kimi-K2.6 permissive parser, mlx-lm tool-parser dispatch, risk audit per family |
| [`docs/serving.md`](docs/serving.md) | Server, REST endpoints, online expert counting, runtime steering, Gradio dashboard |
| [`docs/safety-and-domain.md`](docs/safety-and-domain.md) | SAFEx, SteerMoE, abliteration, domain scan, gate amplification |
| [`docs/kv-cache.md`](docs/kv-cache.md) | TurboQuant (PolarQuant) and RotorQuant (Clifford rotors) deep dive |
| [`docs/cli-reference.md`](docs/cli-reference.md) | All CLI commands and flags |
| [`docs/datasets.md`](docs/datasets.md) | Dataset formats and Solidity prep script |
| [`docs/architecture.md`](docs/architecture.md) | Architecture diagram, components, Python API, output formats |
| [`docs/convert-nvfp4.md`](docs/convert-nvfp4.md) | NVIDIA modelopt NVFP4 → MLX converter (existing) |

---

## Testing

```bash
uv pip install -e ".[dev]"
pytest tests/ -v
```

322 tests covering adapters, saliency math, observer hooks, pruner, REAM
hooks/merger, dataset loading, server (thread safety, all model types),
safety analysis, steering, abliteration, domain identification, frontend,
KV cache compression, CLI wiring.

---

## References

- **REAP** — [Cerebras Research](https://github.com/CerebrasResearch/reap), routing-based expert activation pruning
- **REAM** — [Boris Knyazev's blog](https://bknyaz.github.io/blog/2026/moe/), router-weighted expert activation merging
- **SAFEx** — [Safety experts in MoE (NeurIPS 2025)](https://arxiv.org/abs/2506.17368)
- **SteerMoE** — [Adaptive expert steering for MoE safety](https://arxiv.org/abs/2509.09660)
- **Abliteration** — [Refusal direction (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/f545448535dfde4f9786555403ab7c49-Paper-Conference.pdf)
- **TurboQuant** — [Online vector quantization (ICLR 2026)](https://arxiv.org/abs/2504.19874)
- **RotorQuant** — [Scrya, 2026](https://www.scrya.com/rotorquant/), Clifford rotors for KV compression
- **MLX** — [ml-explore/mlx](https://github.com/ml-explore/mlx)
- **mlx-lm** — [ml-explore/mlx-lm](https://github.com/ml-explore/mlx-lm)
