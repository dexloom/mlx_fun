#!/usr/bin/env python3
"""Download and prepare Solidity calibration dataset.

Downloads Solidity source files from HuggingFace and converts them
to JSONL format suitable for MLX-FUN calibration.

Usage:
    uv run python scripts/prepare_dataset.py --output ./data/solidity_calibration.jsonl

Requires: pip install datasets huggingface-hub
"""

import json
import re
from pathlib import Path

import click


def has_solidity_pragma(text: str) -> bool:
    """Check if text contains a valid Solidity pragma directive."""
    return bool(re.search(r"pragma\s+solidity", text))


@click.command()
@click.option(
    "--output",
    default="./data/solidity_calibration.jsonl",
    help="Output JSONL file path.",
)
@click.option(
    "--source",
    default="bigcode/the-stack-dedup",
    help="HuggingFace dataset to use.",
)
@click.option("--max-samples", default=512, help="Maximum samples to collect.")
@click.option("--min-tokens", default=64, help="Minimum character count for quality filter.")
@click.option("--max-chars", default=16384, help="Maximum characters per sample.")
@click.option("--split", default="train", help="Dataset split to use.")
def main(output, source, max_samples, min_tokens, max_chars, split):
    """Download and filter Solidity code into calibration JSONL."""
    try:
        from datasets import load_dataset
    except ImportError:
        click.echo(
            "Error: 'datasets' package required. Install with:\n"
            "  uv pip install datasets huggingface-hub",
            err=True,
        )
        raise SystemExit(1)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    click.echo(f"Loading dataset: {source}")

    if "the-stack" in source:
        ds = load_dataset(
            source,
            data_dir="data/solidity",
            split=split,
            streaming=True,
            trust_remote_code=True,
        )
        text_key = "content"
    elif "slither" in source:
        ds = load_dataset(source, split=split, streaming=True, trust_remote_code=True)
        text_key = "source_code"
    else:
        ds = load_dataset(source, split=split, streaming=True, trust_remote_code=True)
        # Try common text keys
        text_key = "content"

    count = 0
    with open(output_path, "w") as f:
        for item in ds:
            text = item.get(text_key, "")
            if not text or len(text) < min_tokens:
                continue
            if not has_solidity_pragma(text):
                continue
            # Truncate very long files
            if len(text) > max_chars:
                text = text[:max_chars]

            f.write(json.dumps({"content": text}) + "\n")
            count += 1

            if count % 50 == 0:
                click.echo(f"  Collected {count}/{max_samples} samples...")

            if count >= max_samples:
                break

    click.echo(f"Done! Wrote {count} samples to {output_path}")


if __name__ == "__main__":
    main()
