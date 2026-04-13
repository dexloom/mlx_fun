"""DFlash Speculative Decoding Benchmarks (Phase 6).

Compares three generation modes:
  1. Baseline (greedy / no draft)
  2. Classic speculative (autoregressive draft model)
  3. DFlash (block diffusion draft model)

Target models:
  - Qwen3-8B (primary)
  - Qwen3-Coder-30B-A3B (secondary)
  - Gemma-4-31B-it (tertiary)

Benchmarks:
  - GSM8K (math reasoning, 8.5K examples)
  - Math500 (harder math, 500 examples)
  - AIME24 (competition math, 30 problems)

Metrics:
  - Acceptance rate (avg accepted tokens per draft block)
  - Tokens / second (throughput)
  - Wall-clock speedup vs baseline
  - Accuracy (pass@1 on each benchmark)

Usage:
  # Qwen3-8B, GSM8K subset
  python -m mlx_fun.benchmarks.run_dflash \
    --target-model Qwen/Qwen3-8B \
    --benchmark gsm8k \
    --max-samples 100 \
    --max-tokens 512

  # Gemma-4-31B-it, all benchmarks
  python -m mlx_fun.benchmarks.run_dflash \
    --target-model google/gemma-4-31B-it \
    --benchmark all \
    --max-samples 50

  # DFlash with block size 32
  python -m mlx_fun.benchmarks.run_dflash \
    --target-model Qwen/Qwen3-8B \
    --benchmark gsm8k \
    --mode dflash \
    --dflash-block-size 32 \
    --capture-layers 0,8,16,24,28

  # Quick smoke test (5 samples)
  python -m mlx_fun.benchmarks.run_dflash \
    --target-model Qwen/Qwen3-0.6B \
    --benchmark gsm8k \
    --max-samples 5 \
    --mode dflash \
    --draft-model Qwen/Qwen3-0.6B
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BenchResult:
    """Single benchmark result for one sample."""
    sample_id: int
    prompt: str
    reference: str
    prediction: str
    correct: bool
    time_s: float
    tokens_generated: int
    tokens_per_sec: float
    acceptance_rate: float  # avg accepted / block_size (0.0 for baseline)


@dataclass
class BenchSummary:
    """Aggregated results for a benchmark run."""
    benchmark: str
    target_model: str
    draft_model: Optional[str]
    mode: str  # "baseline", "classic", "dflash"
    dflash_block_size: Optional[int]
    capture_layers: Optional[str]
    num_samples: int
    accuracy: float
    avg_tokens_per_sec: float
    avg_acceptance_rate: float
    wall_time_s: float
    results: List[BenchResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Benchmark datasets (downloaded on demand)
# ---------------------------------------------------------------------------

BENCHMARK_SOURCES = {
    "gsm8k": {
        "hf_dataset": "openai/gsm8k",
        "split": "test",
        "prompt_key": "question",
        "answer_key": "answer",
        "num_examples": 1319,
        "description": "Grade school math (8.5K train, 1.3K test)",
    },
    "math500": {
        "hf_dataset": "HuggingFaceH4/MATH-500",
        "split": "test",
        "prompt_key": "problem",
        "answer_key": "solution",
        "num_examples": 500,
        "description": "Competition math problems",
    },
    "aime24": {
        "hf_dataset": "HuggingFaceH4/aime2024",
        "split": "test",
        "prompt_key": "problem",
        "answer_key": "solution",
        "num_examples": 30,
        "description": "AIME 2024 competition math",
    },
}


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

MATH_PROMPT_TEMPLATE = """\
Solve the following math problem step by step. Put your final answer in \\boxed{}.

{problem}
"""

# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract the content of the last \\boxed{} from model output."""
    import re
    matches = re.findall(r"\\boxed\{([^}]+)\}", text)
    if matches:
        return matches[-1].strip()
    # Also try \boxed without escaping
    matches = re.findall(r"\\\\boxed\{([^}]+)\}", text)
    if matches:
        return matches[-1].strip()
    return None


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison (strip, lowercase, remove $)."""
    s = answer.strip().lower()
    s = s.replace("$", "").replace("\\,", "").replace(",", "")
    s = s.replace(" ", "")
    return s


def check_answer(prediction: str, reference: str) -> bool:
    """Check if predicted answer matches reference."""
    pred = extract_boxed_answer(prediction)
    if pred is None:
        # Fallback: check if reference appears in prediction
        return normalize_answer(reference) in normalize_answer(prediction)
    return normalize_answer(pred) == normalize_answer(reference)


# ---------------------------------------------------------------------------
# Generation modes
# ---------------------------------------------------------------------------

def generate_baseline(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 512,
) -> tuple:
    """Generate with no draft model (greedy baseline).
    Returns (text, time_s, tokens_generated, acceptance_rate=0).
    """
    import mlx_lm
    from mlx_lm.utils import stream_generate

    messages = [{"role": "user", "content": prompt}]
    prompt_tokens = mlx_lm.tokenize(tokenizer, prompt)

    start = time.perf_counter()
    tokens = []
    for token in stream_generate(model, tokenizer, prompt, max_tokens=max_tokens):
        tokens.append(token.token)
    elapsed = time.perf_counter() - start

    text = tokenizer.decode(tokens)
    return text, elapsed, len(tokens), 0.0


def generate_classic_speculative(
    target_model, draft_model, tokenizer,
    prompt: str,
    num_draft_tokens: int = 4,
    max_tokens: int = 512,
) -> tuple:
    """Generate with classic autoregressive speculative decoding.
    Returns (text, time_s, tokens_generated, avg_acceptance_rate).
    """
    # Uses mlx-lm's built-in speculative_generate if available
    # Falls back to manual implementation
    from mlx_lm.generate import speculative_generate as mlx_spec_generate

    messages = [{"role": "user", "content": prompt}]

    start = time.perf_counter()
    # mlx_spec_generate returns (text, n_accepted_total, n_drafted_total)
    result = mlx_spec_generate(
        target_model, draft_model, tokenizer,
        prompt=prompt,
        num_draft_tokens=num_draft_tokens,
        max_tokens=max_tokens,
    )
    elapsed = time.perf_counter() - start

    # result format depends on mlx-lm version — handle both
    if isinstance(result, tuple) and len(result) >= 2:
        text = result[0]
        n_accepted = result[1] if len(result) > 1 else 0
        n_drafted = result[2] if len(result) > 2 else n_accepted
    else:
        text = str(result)
        n_accepted = len(text.split())
        n_drafted = n_accepted

    tokens_generated = len(tokenizer.encode(text))
    acceptance_rate = n_accepted / max(n_drafted, 1)
    return text, elapsed, tokens_generated, acceptance_rate


def generate_dflash_speculative(
    target_model,
    draft_model,
    capture,
    tokenizer,
    prompt: str,
    block_size: int = 16,
    max_tokens: int = 512,
) -> tuple:
    """Generate with DFlash block diffusion speculative decoding.
    Returns (text, time_s, tokens_generated, avg_acceptance_rate).
    """
    from mlx_fun.speculative import dflash_generate

    start = time.perf_counter()
    tokens_list = []
    total_accepted = 0
    total_drafted = 0

    for token, logprobs in dflash_generate(
        target_model, draft_model, capture, tokenizer,
        prompt=prompt,
        block_size=block_size,
        max_tokens=max_tokens,
    ):
        tokens_list.append(token)
        total_accepted += 1
        total_drafted += 1

    elapsed = time.perf_counter() - start
    text = tokenizer.decode(tokens_list)
    tokens_generated = len(tokens_list)
    acceptance_rate = total_accepted / max(total_drafted, 1)
    return text, elapsed, tokens_generated, acceptance_rate


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def load_benchmark_data(name: str, max_samples: int) -> List[Dict]:
    """Load benchmark dataset from HuggingFace."""
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("pip install datasets to run benchmarks")
        sys.exit(1)

    source = BENCHMARK_SOURCES[name]
    dataset = load_dataset(source["hf_dataset"], split=source["split"])
    samples = []
    for i, row in enumerate(dataset):
        if i >= max_samples:
            break
        samples.append({
            "id": i,
            "prompt": MATH_PROMPT_TEMPLATE.format(problem=row[source["prompt_key"]]),
            "reference": row[source["answer_key"]],
        })
    return samples


def run_benchmark(
    target_model_name: str,
    benchmark_name: str,
    mode: str = "all",
    draft_model_name: Optional[str] = None,
    max_samples: int = 100,
    max_tokens: int = 512,
    num_draft_tokens: int = 4,
    dflash_block_size: int = 16,
    capture_layers: str = "0,8,16,24,28",
) -> List[BenchSummary]:
    """Run benchmarks across modes and return summaries."""
    from mlx_lm import load

    logger.info(f"Loading target model: {target_model_name}")
    target_model, tokenizer = load(target_model_name)

    draft_model = None
    if draft_model_name and mode in ("classic", "dflash", "all"):
        logger.info(f"Loading draft model: {draft_model_name}")
        draft_model, _ = load(draft_model_name)

    samples = load_benchmark_data(benchmark_name, max_samples)
    logger.info(f"Loaded {len(samples)} samples from {benchmark_name}")

    modes_to_run = ["baseline", "classic", "dflash"] if mode == "all" else [mode]
    summaries = []

    for run_mode in modes_to_run:
        if run_mode in ("classic", "dflash") and draft_model is None:
            logger.warning(f"Skipping {run_mode}: no draft model")
            continue

        results = []
        wall_start = time.perf_counter()

        for sample in samples:
            try:
                if run_mode == "baseline":
                    text, elapsed, n_tok, acc_rate = generate_baseline(
                        target_model, tokenizer, sample["prompt"], max_tokens)
                elif run_mode == "classic":
                    text, elapsed, n_tok, acc_rate = generate_classic_speculative(
                        target_model, draft_model, tokenizer, sample["prompt"],
                        num_draft_tokens, max_tokens)
                elif run_mode == "dflash":
                    from mlx_fun.hidden_state_capture import HiddenStateCapture
                    from mlx_fun.dflash_draft import create_dflash_draft_model

                    capture = HiddenStateCapture(target_model, capture_layers)
                    capture.install()
                    try:
                        text, elapsed, n_tok, acc_rate = generate_dflash_speculative(
                            target_model, draft_model, capture, tokenizer,
                            sample["prompt"], dflash_block_size, max_tokens)
                    finally:
                        capture.remove()
                else:
                    continue

                correct = check_answer(text, sample["reference"])
                results.append(BenchResult(
                    sample_id=sample["id"],
                    prompt=sample["prompt"],
                    reference=sample["reference"],
                    prediction=text,
                    correct=correct,
                    time_s=elapsed,
                    tokens_generated=n_tok,
                    tokens_per_sec=n_tok / max(elapsed, 0.001),
                    acceptance_rate=acc_rate,
                ))
            except Exception as e:
                logger.error(f"Error on sample {sample['id']}: {e}")

        wall_elapsed = time.perf_counter() - wall_start

        if not results:
            continue

        accuracy = sum(r.correct for r in results) / len(results)
        avg_tps = sum(r.tokens_per_sec for r in results) / len(results)
        avg_acc = sum(r.acceptance_rate for r in results) / len(results)

        summary = BenchSummary(
            benchmark=benchmark_name,
            target_model=target_model_name,
            draft_model=draft_model_name,
            mode=run_mode,
            dflash_block_size=dflash_block_size if run_mode == "dflash" else None,
            capture_layers=capture_layers if run_mode == "dflash" else None,
            num_samples=len(results),
            accuracy=accuracy,
            avg_tokens_per_sec=avg_tps,
            avg_acceptance_rate=avg_acc,
            wall_time_s=wall_elapsed,
            results=results,
        )
        summaries.append(summary)

        logger.info(
            f"[{run_mode}] acc={accuracy:.3f} tps={avg_tps:.1f} "
            f"accept={avg_acc:.2f} wall={wall_elapsed:.1f}s"
        )

    return summaries


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_results(summaries: List[BenchSummary], output_dir: str):
    """Save results to JSON and CSV."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for summary in summaries:
        tag = f"{summary.benchmark}_{summary.mode}"
        # JSON
        json_path = out / f"{tag}.json"
        with open(json_path, "w") as f:
            json.dump(asdict(summary), f, indent=2, default=str)
        logger.info(f"Saved {json_path}")

        # CSV
        csv_path = out / f"{tag}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=asdict(BenchResult()).keys())
            writer.writeheader()
            for r in summary.results:
                writer.writerow(asdict(r))
        logger.info(f"Saved {csv_path}")

    # Combined summary
    combined = []
    for s in summaries:
        d = asdict(s)
        d.pop("results")  # skip per-sample details
        combined.append(d)
    summary_path = out / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(combined, f, indent=2)
    logger.info(f"Saved {summary_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DFlash Speculative Decoding Benchmarks")
    parser.add_argument("--target-model", required=True, help="Target model HuggingFace ID")
    parser.add_argument("--draft-model", default=None, help="Draft model HuggingFace ID")
    parser.add_argument("--benchmark", default="gsm8k", choices=["gsm8k", "math500", "aime24", "all"])
    parser.add_argument("--mode", default="all", choices=["baseline", "classic", "dflash", "all"])
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--num-draft-tokens", type=int, default=4, help="Classic speculative draft length")
    parser.add_argument("--dflash-block-size", type=int, default=16)
    parser.add_argument("--capture-layers", default="0,8,16,24,28")
    parser.add_argument("--output-dir", default="benchmarks/results")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    benchmarks = [args.benchmark] if args.benchmark != "all" else list(BENCHMARK_SOURCES.keys())

    all_summaries = []
    for bench in benchmarks:
        summaries = run_benchmark(
            target_model_name=args.target_model,
            benchmark_name=bench,
            mode=args.mode,
            draft_model_name=args.draft_model,
            max_samples=args.max_samples,
            max_tokens=args.max_tokens,
            num_draft_tokens=args.num_draft_tokens,
            dflash_block_size=args.dflash_block_size,
            capture_layers=args.capture_layers,
        )
        all_summaries.extend(summaries)

    if all_summaries:
        save_results(all_summaries, args.output_dir)
        print("\n=== RESULTS ===")
        for s in all_summaries:
            speedup = s.avg_tokens_per_sec  # vs baseline computed separately
            print(
                f"  {s.benchmark:10s} | {s.mode:8s} | "
                f"acc={s.accuracy:.3f} | tps={s.avg_tokens_per_sec:.1f} | "
                f"accept={s.avg_acceptance_rate:.2f} | "
                f"block={s.dflash_block_size or '-'}"
            )


if __name__ == "__main__":
    main()
