"""Calibration dataset loading."""

import json
import random
from pathlib import Path
from typing import List

import mlx.core as mx


def _subsample(samples: List[mx.array], max_samples: int) -> List[mx.array]:
    """Randomly subsample if we have more samples than max_samples."""
    if max_samples > 0 and len(samples) > max_samples:
        samples = random.sample(samples, max_samples)
    return samples


def _tokenize_entry(obj: dict, tokenizer, text_key: str, max_tokens: int) -> list:
    """Tokenize a single JSONL entry, auto-detecting format.

    Supported formats:
      - Chat messages: {"messages": [{"role": "...", "content": "..."}, ...]}
      - Completions:   {"prompt": "...", "completion": "..."}
      - Plain text:    {"<text_key>": "..."}
    """
    if "messages" in obj:
        tokens = tokenizer.apply_chat_template(
            obj["messages"], tokenize=True, add_generation_prompt=False,
        )
    elif "prompt" in obj:
        text = obj["prompt"] + obj.get("completion", "")
        tokens = tokenizer.encode(text)
    else:
        tokens = tokenizer.encode(obj[text_key])

    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    return tokens


def load_jsonl(
    path: str,
    tokenizer,
    max_tokens: int = 2048,
    max_samples: int = 0,
    text_key: str = "content",
) -> List[mx.array]:
    """Load calibration data from a JSONL file.

    Auto-detects format per line. Supported formats:
      - Chat messages: {"messages": [{"role": "...", "content": "..."}, ...]}
      - Completions:   {"prompt": "...", "completion": "..."}
      - Plain text:    {"<text_key>": "..."}

    If max_samples is set and the file contains more entries,
    a random subset is selected.

    Args:
        path: Path to JSONL file.
        tokenizer: HuggingFace tokenizer.
        max_tokens: Maximum tokens per sample.
        max_samples: Maximum number of samples (0 = all).
        text_key: JSON key containing the text.

    Returns:
        List of mx.array token sequences.
    """
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            tokens = _tokenize_entry(obj, tokenizer, text_key, max_tokens)
            if len(tokens) > 0:
                samples.append(mx.array(tokens))
    return _subsample(samples, max_samples)


def load_directory(
    path: str,
    tokenizer,
    max_tokens: int = 2048,
    max_samples: int = 0,
    extensions: tuple = (".sol", ".txt"),
) -> List[mx.array]:
    """Load calibration data from a directory of source files.

    If max_samples is set and the directory contains more files,
    a random subset is selected.

    Args:
        path: Directory path.
        tokenizer: HuggingFace tokenizer.
        max_tokens: Maximum tokens per sample.
        max_samples: Maximum number of samples (0 = all).
        extensions: File extensions to include.

    Returns:
        List of mx.array token sequences.
    """
    samples = []
    dir_path = Path(path)
    files = sorted(f for f in dir_path.rglob("*") if f.suffix in extensions)
    for fpath in files:
        text = fpath.read_text(errors="replace")
        tokens = tokenizer.encode(text)
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
        if len(tokens) > 0:
            samples.append(mx.array(tokens))
    return _subsample(samples, max_samples)


def load_dataset(
    path: str,
    tokenizer,
    max_tokens: int = 2048,
    max_samples: int = 0,
    text_key: str = "content",
) -> List[mx.array]:
    """Auto-detect format and load calibration data.

    If path is a directory, loads files. If a file, loads as JSONL.
    """
    p = Path(path)
    if p.is_dir():
        return load_directory(p, tokenizer, max_tokens, max_samples)
    else:
        return load_jsonl(p, tokenizer, max_tokens, max_samples, text_key)
