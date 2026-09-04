# CLAUDE.md — MLX-FUN Project Guide

## Project Overview

MLX-FUN implements REAP (Routing-based Expert Activation Pruning) for MoE models on Apple Silicon via MLX. It prunes routed experts from MoE language models based on calibration saliency data. It also supports safety-critical expert analysis (SAFEx), inference-time expert steering (SteerMoE), refusal direction removal (abliteration), domain-specific expert identification, and permanent gate amplification.

## Quick Reference

```bash
# Activate venv
source .venv/bin/activate

# Install (editable)
uv pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# CLI entry point
mlx-fun --help
mlx-fun collect --help
mlx-fun prune --help
mlx-fun merge --help
mlx-fun smoke-test --help
mlx-fun safety-scan --help
mlx-fun steer --help
mlx-fun abliterate --help
mlx-fun domain-scan --help
mlx-fun domain-probe --help
mlx-fun amplify --help
mlx-fun convert-nvfp4 --help
mlx-fun ui --help
```

## Project Structure

```
src/mlx_fun/
├── adapters/              # Model-specific MoE access (BaseAdapter ABC)
│   ├── minimax.py         # MiniMax/MiniMax-M2: all layers MoE, block_sparse_moe
│   ├── glm4_moe.py       # GLM4: MoE layers >= first_k_dense_replace, mlp
│   ├── glm4_moe_lite.py  # GLM4-Lite: adds moe_layer_freq stride
│   ├── glm_moe_dsa.py    # GLM-5/DeepSeek V3.2: DeepseekV32MoE with MoEGate
│   ├── qwen3_moe.py      # Qwen3/Qwen3-Next: sparse layers by decoder_sparse_step, mlp
│   ├── nemotron_h.py      # Nemotron-H: hybrid Mamba-2/Attn/MoE via hybrid_override_pattern
│   ├── qwen4_exp.py       # Qwen4-Exp (Qwen3.8-Flash-Next): VLM, MoE under language_model
│   └── glm5_next.py       # GLM-5.3-Flash: VLM, DeepseekV32MoE under language_model
├── models/                # Out-of-tree model types published into mlx_lm.models
│   └── gemma4_assistant.py # Gemma 4 MTP drafter (was a fork of mlx-lm)
├── loader.py              # Backend-aware load: mlx-lm for text, mlx-vlm for vision
├── observer.py            # Hooks via __class__ swap (not MethodType — special methods)
├── ream_hooks.py          # REAM hooks: capture MoE inputs + full gate logits
├── saliency.py            # numpy float64 accumulator with np.add.at() scatter-add
├── pruner.py              # Expert selection (bottom + strided) + mx.take() tensor slicing
├── merger.py              # REAM: expert grouping, permutation alignment, weight merging
├── safety.py              # SAFEx: differential accumulator, safety report, expert classification
├── steering.py            # SteerMoE: gate logit bias injection for expert (de)activation
├── abliterate.py          # Abliteration: residual hooks, refusal direction orthogonalization
├── convert_nvfp4.py       # NVIDIA NVFP4 (modelopt) -> MLX NVFP4 checkpoint converter
├── domain.py              # Domain expert identification, amplification bias computation, gate modification
├── probe.py               # Q&A expert relevance: routing trace over answers + knockout verification
├── frontend.py            # Gradio web dashboard: chat, heatmaps, steering controls, server management
├── data.py                # JSONL + directory dataset loading with random subsampling
├── save.py                # mlx_lm.utils.save_model + reap/ream/abliteration/amplification metadata
└── cli.py                 # Click CLI: collect, prune, merge, smoke-test, serve, ui, safety-scan, steer, abliterate, domain-scan, domain-probe, amplify
```

## Key Design Decisions

- **Observer hooks use `__class__` swapping**, not `types.MethodType`. Python resolves `__call__` on the type, not the instance. The observer creates a dynamic subclass with the hooked `__call__` and swaps `block.__class__`.

- **Saliency uses numpy float64**, not MLX arrays. This avoids lazy evaluation issues and provides numerical stability for accumulation across many batches.

- **After capturing hook data, `mx.eval()` is called immediately** followed by `_to_numpy()` which casts bf16 to float32 before `np.array(..., copy=False)`. This materializes lazy MLX arrays before they can be garbage collected.

- **Tensor slicing uses `mx.take(tensor, keep_indices, axis=0)`** on the expert dimension for SwitchLinear weights, scales, biases, gate weights, and correction biases. Both `nn.Linear` and `nn.QuantizedLinear` gates are handled via `_slice_linear()`.

- **Dataset loading reads all samples first, then randomly subsamples** if `--max-samples` is set and the source has more. Use `--seed` for reproducibility.

- **Two pruning strategies**: `bottom` (remove lowest-scoring, standard REAP) and `strided` (split into important/unimportant groups, prune at regular intervals from both for better diversity).

- **REAM merging** (`merger.py`): Instead of pruning, merges experts around centroids using saliency-weighted averaging with neuron permutation alignment. Processes layers sequentially so merged weights feed into the next layer's similarity computation. Uses gated similarity (gate_logit * expert_output cosine similarity) for grouping.

- **Steering hooks inject bias into gate logits** before top-k selection. A pre-computed `mx.array` of shape `(num_experts,)` is added to raw gate logits — `mask_value` (-1e9) for deactivation, `boost_value` (1e4) for activation. When bias is None, the if-branch is skipped (negligible overhead).

- **Compound hooks** in `server.py` combine counting + steering in a single `__call__` to avoid hook composition issues (two `__class__` swaps on the same block would conflict).

- **Abliteration hooks target decoder layers** (one level above MoE blocks), capturing the residual stream input. Refusal direction = normalize(mean_harmful - mean_benign) at the last token position. Weight orthogonalization: `W' = W - (W @ d) * d^T`.

- **Single expert orthogonalization** uses `mx.concatenate` on slices rather than `.at[].set()` for MLX compatibility.

- **Domain expert identification** (`domain.py`) reuses `DifferentialAccumulator` and `compute_differential_scores` from `safety.py` with domain-appropriate semantics: "harmful"=domain data, "benign"=general data. Positive differential = domain-preferred expert. `DomainReport` mirrors `SafetyReport` structure.

- **Gate amplification** permanently modifies gate parameters so domain experts are favored natively — no hooks needed at inference time. Per model type: MiniMax/Qwen3 set `gate.bias` on `nn.Linear(bias=False)` (MLX's `nn.Linear.__call__` checks `if "bias" in self`), GLM4/GLM5/DSv3 add to `gate.e_score_correction_bias` (post-sigmoid). The amplified model saves/loads with standard `mlx_lm.load()`.

- **Pruner domain constraints** (`load_domain_constraints`) only support `"protect"` mode (never prune domain experts). Domain and safety constraints merge via union of protected sets.

- **`pruner.build_keep_map` is the single entry point** for the three-way
  selection branch (model-wide / strided / bottom). `cli.prune` and the probe's
  prune-set verification both call it, so what the probe masks is exactly what
  `prune` would remove.

- **Probe routing is attributed to answer-*producing* positions**
  (`probe.slice_answer_captures`): logits at position `t` predict token `t+1`,
  so the scored predictions come from `[prompt_len-1 : prompt_len-1+n_answer]`.
  This keeps the observational score and the knockout delta describing the same
  predictions, and includes the routing that picks the first answer token. In
  generate mode mlx-lm forwards every token before yielding it, so captures
  cover exactly `prompt_len + n_generated` positions — asserted, not assumed.

- **Probe scores are question-weighted**, not token-weighted: each question
  contributes one vector normalized by its own answer length (`question_vectors`),
  so a verbose answer cannot outvote a terse one. The `.npz` for `prune
  --saliency` is folded the same way unless `--saliency-weighting token`.

- **Knockout masks the router's selection-score parameter**, never a surrogate
  router (`probe.expert_mask`). MiniMax and the GLM/DeepSeek family select on
  `sigmoid(logits) + e_score_correction_bias`, so the mask must land *after* the
  sigmoid — biasing pre-sigmoid logits (what `amplify` does) cannot reliably
  deselect an expert with a large positive correction. Qwen types get a
  temporary pre-softmax `gate.bias`. The model's own `__call__` runs, so grouped
  selection, `norm_topk_prob`, `routed_scaling_factor` and latent projections all
  still apply. Gemma 4 is unsupported and raises. Parameters are restored in
  `finally`, including on exception.

- **Probe datasets live in `data/probes/`, corpora in `data/corpora/`** — both
  un-ignored in `.gitignore` because they are authored source, unlike the
  downloaded sets in `data/datasets/`. Two domain sets (`solidity`, `security`)
  and two contrast sets (`general`, `solidity_benign`); pairing `security`
  against `solidity_benign` isolates vulnerability reasoning from Solidity
  knowledge, since both sides are Solidity. Every Solidity sample in
  `data/corpora/evm_security.jsonl` compiles under solc 0.8.28 Cancun and the
  Yul object under `--strict-assembly`; keep it that way when adding samples.

- **Knockout deltas are paired per question** with a bootstrap CI
  (`paired_delta_stats`). A masked run that goes non-finite counts as a collapse
  rather than being dropped, so a mask that destroys half the answers cannot
  average out to "harmless".

- **NVFP4 conversion** (`convert_nvfp4.py`): NVIDIA's `modelopt` NVFP4 checkpoints use a two-level scale hierarchy (`fp4_val * e4m3_group_scale * f32_global_scale`) that MLX Metal doesn't support. The converter folds `weight_scale_2` into per-group E4M3 scales via `from_fp8()` → multiply → `to_fp8()`, preserving trained FP4 codes exactly while accepting ~1-2% scale rounding. FP8 layers (Mamba, shared experts) are dequantized to bfloat16. Expert weights are repacked from uint8 `[M, N/2]` to uint32 `[M, N/8]` via `numpy.view` (lossless).

- **No fork of mlx-lm.** `src/mlx_fun/models/` holds model classes upstream
  does not ship (currently `gemma4_assistant`, the Gemma 4 MTP drafter).
  `register_model_types()` — called from `mlx_fun/__init__.py` — inserts them
  into `sys.modules` as `mlx_lm.models.<type>`, which is where mlx-lm's
  `_get_classes` looks via `importlib.import_module`. Upstream always wins: a
  type mlx-lm has since shipped is left alone, so registrations retire
  themselves.

- **Vision models load through mlx-vlm** (`loader.py`). `load_model()` reads
  `config.json` first and routes: a checkpoint with a `vision_config` (or a
  `model_type` in `_VLM_ONLY_MODEL_TYPES`) goes to `mlx_vlm.load()`, everything
  else to `mlx_lm.load()`. It returns the same `(model, tokenizer, config)`
  triple either way, so CLI call sites are backend-agnostic. mlx-vlm is an
  optional extra (`.[vlm]`); text models never import it.

- **Analysis passes on a VLM run against the language stack.**
  `text_forward(model, config)` returns `model.language_model` for vision
  checkpoints — the multimodal wrapper's `__call__` expects pixel values, while
  calibration and routing scans feed token ids only. Serving VLMs is *not*
  wired up; `smoke-test` rejects them with a pointer to `mlx_vlm.generate`.

- **GLM-5.3 needs no new code** — it is `glm_moe_dsa`, the type mlx_fun already
  supports (78 layers, 256 experts). It is the first config in that family to
  ship an explicit per-layer `mlp_layer_types` list, which
  `GLMMoeDsaAdapter.moe_layer_indices` now prefers over the
  `first_k_dense_replace` + `moe_layer_freq` stride rule. Configs without the
  key (GLM-5, DeepSeek V3.2) keep the old path.

- **GLM-5.3-Flash (`glm5_next`) is a VLM whose MoE block is mlx-vlm's
  `DeepseekV32MoE`** — the same sigmoid/`noaux_tc` block as GLM-5, so it maps
  onto the existing `_glm4_*` hooks in `observer.py`, `ream_hooks.py`, and
  `steering.py` rather than getting its own. Only the adapter is new: it reaches
  through `language_model` and honors `mlp_layer_types`.

- **Qwen4-Exp MoE blocks are mlx-vlm's `Qwen3_5MoeSparseMoeBlock`** at
  `model.language_model.model.layers[i].mlp`, on *every* layer (the
  linear/full-attention split applies to the attention branch only). Routing is
  softmax → top-k → unconditional renormalization; unlike Qwen3-Next there is
  no `norm_topk_prob` attribute, which is why it gets its own hooks in
  `observer.py`, `ream_hooks.py`, and `steering.py` rather than reusing
  Qwen3-Next's. The sigmoid-gated shared expert is not routed and is not a
  pruning target.

## Supported Models

| Type | Config `model_type` | Expert count key | MoE block path |
|---|---|---|---|
| MiniMax | `minimax` | `num_local_experts` | `model.model.layers[i].block_sparse_moe` |
| MiniMax-M2 | `minimax_m2` | `num_local_experts` | Same as MiniMax (alias) |
| GLM4-MoE | `glm4_moe` | `n_routed_experts` | `model.model.layers[i].mlp` |
| GLM4-MoE-Lite | `glm4_moe_lite` | `n_routed_experts` | Same as GLM4 + `moe_layer_freq` stride |
| Qwen3-MoE | `qwen3_moe` | `num_experts` | `model.model.layers[i].mlp` (sparse layers only) |
| Qwen3-Next | `qwen3_next` | `num_experts` | Same as Qwen3 + sigmoid-gated shared expert |
| GLM-5 | `glm_moe_dsa` | `n_routed_experts` | `model.model.layers[i].mlp` (DeepSeek V3.2 MoE) |
| DeepSeek V3.2 | `deepseek_v32` | `n_routed_experts` | Same as GLM-5 (shared architecture) |
| Nemotron-H | `nemotron_h` | `n_routed_experts` | `model.backbone.layers[i].mixer` (hybrid Mamba-2/Attn/MoE) |
| Qwen4-Exp | `qwen4_exp` | `num_experts` (in `text_config`) | `model.language_model.model.layers[i].mlp` (VLM — loads via mlx-vlm) |
| GLM-5.3 | `glm_moe_dsa` | `n_routed_experts` | Same as GLM-5 — 78 layers, 256 experts, adds `mlp_layer_types` |
| GLM-5.3-Flash | `glm5_next` | `n_routed_experts` (in `text_config`) | `model.language_model.model.layers[i].mlp` (VLM — loads via mlx-vlm) |

Reference source files (mlx-lm 0.32.0 upstream, mlx-vlm 0.6.17):
- MiniMax: `mlx_lm/models/minimax.py` — `MiniMaxSparseMoeBlock`
- GLM4: `mlx_lm/models/glm4_moe.py` — `MoE`, `MoEGate`
- GLM4-Lite: `mlx_lm/models/glm4_moe_lite.py` — `Glm4MoeLiteMoE`, `MoEGate`
- Qwen3: `mlx_lm/models/qwen3_moe.py` — `Qwen3MoeSparseMoeBlock`
- Qwen3-Next: `mlx_lm/models/qwen3_next.py` — `Qwen3NextSparseMoeBlock`
- GLM-5 / DeepSeek V3.2: `mlx_lm/models/deepseek_v32.py` — `DeepseekV32MoE`, `MoEGate`
- Nemotron-H: `mlx_lm/models/nemotron_h.py` — `NemotronHMoE`, `MoEGate` (hybrid Mamba-2/Attention/MoE)
- Switch layers: `mlx_lm/models/switch_layers.py` — `SwitchGLU`, `SwitchLinear`, `QuantizedSwitchLinear`
- Qwen4-Exp: **mlx-vlm** `mlx_vlm/models/qwen4_exp/language.py` — `Qwen4ExpDecoderLayer`,
  whose `.mlp` is `mlx_vlm/models/qwen3_5_moe/language.py::Qwen3_5MoeSparseMoeBlock`
- GLM-5.3-Flash: **mlx-vlm** `mlx_vlm/models/glm5_next/language.py` — `Glm5NextDecoderLayer`,
  whose `.mlp` is `mlx_vlm/models/deepseek_v32/language.py::DeepseekV32MoE`

## Testing

Tests use tiny MoE fixtures (4 experts, hidden=32) defined in `tests/conftest.py`. No real models are needed for unit tests.

```bash
pytest tests/ -v                    # All 705 tests
pytest tests/test_pruner.py -v      # Just pruner tests
pytest tests/test_safety.py -v      # Safety analysis tests
pytest tests/test_steering.py -v    # Steering hook tests
pytest tests/test_abliterate.py -v  # Abliteration tests
pytest tests/test_domain.py -v      # Domain identification + amplification tests
pytest tests/test_probe.py -v       # Q&A probe: scoring, knockout mask, trace pass
pytest tests/test_frontend.py -v   # Frontend API + visualization tests
pytest tests/test_convert_nvfp4.py -v  # NVFP4 converter tests
pytest tests/test_loader.py -v      # Backend routing (mlx-lm vs mlx-vlm)
pytest tests/test_qwen4_exp.py -v   # Qwen4-Exp adapter + hooks (tiny replicas)
pytest tests/test_glm53.py -v       # GLM-5.3 + GLM-5.3-Flash adapters + hooks
pytest tests/test_vlm_integration.py -v  # Both, against real mlx-vlm classes
```

## Dependencies

Runtime: `mlx >= 0.32.1`, `mlx-lm` (upstream `main`, no fork), `click`, `tqdm`, `numpy`
Dev: `pytest`
REAM merging: `scipy` (optional extra `.[ream]`)
Web dashboard: `gradio`, `matplotlib`, `requests` (optional extra `.[ui]`)
Vision models: `mlx-vlm` (optional extra `.[vlm]`)
NVFP4 conversion: `safetensors`, `huggingface-hub` (optional extra `.[convert]`)
Dataset prep: `datasets`, `huggingface-hub` (optional extra `.[dataset]`)
