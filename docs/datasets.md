# Datasets

mlx_fun accepts calibration data in two path types (auto-detected) and three
JSONL formats (auto-detected per line).

`domain-probe` takes a different shape — a question set rather than a
calibration corpus. See [Probe question sets](#probe-question-sets) below.

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

## Probe question sets

`domain-probe` consumes its own JSONL shape. These are *not* calibration
samples: each line is a question the model is asked, plus the reference answer
its routing is scored against.

```jsonl
{"question": "What does the `payable` modifier do?", "answer": "It allows the function to receive Ether. Sending value to a non-payable function reverts.", "tags": ["functions", "ether"]}
{"question": "What is a reentrancy attack?", "answer": "An external call lets the callee re-enter the calling contract before its state updates finish.", "tags": ["reentrancy", "security"]}
```

| field | required | meaning |
|---|---|---|
| `question` | yes | Asked as the user turn, through the chat template |
| `answer` | teacher mode | Reference answer; routing is scored over its tokens |
| `tags` | no | Free-form labels, carried through to `--answers-output` |
| `system` | no | Per-question system prompt; overrides `--system` |

With `--answer-mode generate` the model writes its own answer and `answer` may
be omitted. In the default teacher mode a question with no usable answer is
skipped and counted in the report's `skipped_questions`.

### Shipped sets

| File | Role | Contents |
|---|---|---|
| `data/probes/solidity.jsonl` | domain | 86 general smart-contract development questions |
| `data/probes/security.jsonl` | domain | 116 questions on vulnerability discovery, EVM internals and Yul |
| `data/probes/solidity_benign.jsonl` | contrast | 95 ordinary Solidity/EVM questions with no security content |
| `data/probes/general.jsonl` | contrast | 81 questions: Python, algorithms, arithmetic, general trivia |

### Choosing a pair

The contrast set decides what the probe isolates, so pick it deliberately:

| Domain | Contrast | Finds |
|---|---|---|
| `solidity.jsonl` | `general.jsonl` | Experts for smart-contract work in general, against a non-code baseline |
| `security.jsonl` | `general.jsonl` | The same broad set, with a stronger signal — security questions are still Solidity questions |
| `security.jsonl` | `solidity_benign.jsonl` | **Experts used for vulnerability reasoning specifically**, with ordinary Solidity knowledge held constant |

The third pairing is the interesting one for audit work: both sides are
Solidity, so what survives the differential is the reasoning about what can go
wrong, not the language itself.

```bash
mlx-fun domain-probe --model ./model \
    --domain-questions data/probes/security.jsonl \
    --general-questions data/probes/solidity_benign.jsonl \
    --output security_report.json --saliency-output security.npz \
    --domain-name security --verify-top 32 --verify-prune 8 --seed 42
```

`security.jsonl` covers reentrancy in all four flavors, access control,
oracle and price manipulation, flash loans, arithmetic and precision, proxy and
`delegatecall` hazards, signatures and replay, DoS and gas griefing, token
standard pitfalls, MEV, EVM internals (storage slot derivation, memory layout,
opcode semantics, gas rules), and Yul — memory safety against the free memory
pointer, slot arithmetic, sub-word masking, calldata decoding, `verbatim`,
object structure and the `memory-safe` annotation.

### Writing your own

- **The contrast set is what defines the domain.** An expert scores high only
  because it fires more on domain questions than on general ones, so the general
  set must genuinely avoid the domain's vocabulary.
- **Keep answers short but not one-word.** Roughly 10-100 tokens. A one-token
  answer gives a single routing decision, which is noise; the cap is
  `--max-answer-tokens` (default 128) and longer references are truncated.
- **Answers must be correct.** In teacher mode the model is scored on
  predicting them, so a wrong reference measures the model's disagreement rather
  than its domain routing.
- **Size the sets similarly.** Scoring is question-weighted, so unequal answer
  lengths are handled, but very different question counts make the two sides'
  estimates unevenly noisy. 60-100 each is a reasonable starting point.
- **Both sets go through the chat template**, so the tokens match what the model
  sees in real chat use. Disable thinking mode while probing:
  `--chat-template-args '{"enable_thinking": false}'`.

`--max-questions` subsamples each set (seeded by `--seed`), and a generate-mode
run can be captured with `--answers-output` and replayed in teacher mode.

## Calibration corpora

`data/corpora/` holds hand-written calibration corpora in the plain-text format
(`{"content": ...}`), for `collect`, `domain-scan` and `merge`. Unlike
`data/datasets/`, which holds downloaded corpora and is not tracked, these are
authored source and ship with the repo.

| File | Contents |
|---|---|
| `data/corpora/evm_security.jsonl` | 30 samples: vulnerable contracts paired with their fixed counterparts, Yul and assembly libraries, and an audit test harness |

Each vulnerability sample is annotated `VULNERABLE` or `FIXED` in a NatSpec
comment and most appear as a pair, so the corpus carries the distinction rather
than only the flaw. Coverage matches the probe set: reentrancy (including
read-only and cross-function), `tx.origin` auth, unprotected initializers,
storage collision and EIP-1967 slots, spot-price oracles and Chainlink round
validation, ERC-4626 inflation, donation-driven accounting, rounding direction,
unsafe downcasts, return bombs, push-payment DoS, weak and strong EIP-712
signature verification, and flash-loan governance.

The Yul samples are the reason this corpus exists separately from scraped
Solidity, which contains very little assembly: an assembly ERC-20, a
memory-safe concat, a deliberately memory-unsafe counterexample, bounded
calldata decoding, storage slot derivation, an EIP-1167 clone factory,
`mulDiv` with a 512-bit intermediate, a transient-storage guard, and a
standalone Yul object.

Every Solidity sample compiles under solc 0.8.28 (Cancun) and the Yul object
compiles with `--strict-assembly`; selectors, event topics and EIP-1967 slots
were checked against `cast`.

```bash
mlx-fun collect --model ./model \
    --dataset data/corpora/evm_security.jsonl \
    --output security_saliency.npz --max-samples 30
```

At 30 samples this is a seed, not a full calibration set — the guidance below
still asks for 256-512. Combine it with scraped Solidity from
`scripts/prepare_dataset.py`, or point `--dataset` at a directory holding both.

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
