# Datasets

mlx_fun accepts calibration data in two path types (auto-detected) and three
JSONL formats (auto-detected per line).

## Path types

### JSONL file (recommended)

A single `.jsonl` file. Each line is one calibration sample. The format is
detected per-line, so you can mix formats in the same file.

### Directory of source files

A directory containing raw source files (`.sol`, `.txt` by default). Each
file becomes one calibration sample:

```
data/solidity/
├── Token.sol
├── Vault.sol
├── Governance.sol
└── ...
```

## JSONL formats

Three are supported. Priority: `messages` > `prompt`/`completion` > plain text.

### Chat messages (best for chat models)

Compatible with [mlx-lm fine-tuning format](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md):

```jsonl
{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Hello."}, {"role": "assistant", "content": "How can I help you today?"}]}
{"messages": [{"role": "user", "content": "What is Solidity?"}, {"role": "assistant", "content": "Solidity is a programming language for Ethereum smart contracts."}]}
```

Tokenized via `tokenizer.apply_chat_template()` so tokens match exactly what
the model sees during chat inference.

### Prompt + completion

```jsonl
{"prompt": "What is the capital of France?", "completion": "Paris."}
```

### Plain text

```jsonl
{"content": "// SPDX-License-Identifier: MIT\npragma solidity ^0.8.0;\n\ncontract Token {\n    ..."}
```

The text key defaults to `"content"` and can be overridden with `--text-key`.
Each entry is tokenized and truncated to `--max-tokens` (default 2048).

## Preparing a Solidity dataset

A preparation script is included to download Solidity code from HuggingFace:

```bash
uv pip install -e ".[dataset]"

python scripts/prepare_dataset.py \
    --source bigcode/the-stack-dedup \
    --output ./data/solidity_calibration.jsonl \
    --max-samples 512
```

| Flag | Default | Description |
|---|---|---|
| `--source` | `bigcode/the-stack-dedup` | HuggingFace dataset ID |
| `--output` | `./data/solidity_calibration.jsonl` | Output JSONL path |
| `--max-samples` | 512 | Number of samples |
| `--min-tokens` | 64 | Minimum character length filter |
| `--max-chars` | 16384 | Truncate long files |
| `--split` | `train` | Dataset split |

The script streams from HuggingFace, filters for valid Solidity (`pragma
solidity` check), and writes JSONL.

## Guidelines

- **256–512 samples** is a good calibration size — enough for stable
  saliency estimates without excessive runtime.
- **Domain matters** — calibrate on the domain you care about.
  Solidity-calibrated pruning retains better Solidity generation than
  generic-text calibration at the same prune ratio.
- **Token length** — 2048 tokens per sample captures enough context for
  routing patterns to stabilize.
- **Subsampling is reproducible** — pass `--seed` to fix the random draw
  when `--max-samples` is less than the dataset size.

## Random seed and `--max-samples`

Mlx_fun reads all samples first, then randomly subsamples down to
`--max-samples` if the source has more. Use `--seed` for reproducibility:

```bash
mlx-fun collect --model ./model --dataset ./data/big.jsonl \
    --output ./saliency.npz --max-samples 128 --seed 42
```

Same `--seed` always picks the same subset. Different seeds give independent
calibration runs you can later combine via `mlx-fun stats-merge`.
