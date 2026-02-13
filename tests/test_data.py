"""Tests for dataset loading with multiple JSONL formats."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import mlx.core as mx
import pytest

from mlx_fun.data import load_jsonl, load_dataset, _tokenize_entry


def _make_tokenizer(vocab_size=100):
    """Create a mock tokenizer that returns token IDs based on word count."""
    tok = MagicMock()
    tok.encode = MagicMock(side_effect=lambda text: list(range(len(text.split()))))
    tok.apply_chat_template = MagicMock(
        side_effect=lambda msgs, tokenize=True, add_generation_prompt=False: (
            list(range(sum(len(m["content"].split()) for m in msgs) + len(msgs)))
        )
    )
    return tok


class TestTokenizeEntry:
    def test_messages_format(self):
        tok = _make_tokenizer()
        obj = {
            "messages": [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ]
        }
        tokens = _tokenize_entry(obj, tok, "content", max_tokens=2048)
        tok.apply_chat_template.assert_called_once_with(
            obj["messages"], tokenize=True, add_generation_prompt=False,
        )
        # 3 + 1 + 2 words + 3 roles = 9 tokens from our mock
        assert len(tokens) == 9

    def test_completions_format(self):
        tok = _make_tokenizer()
        obj = {"prompt": "What is AI", "completion": " Artificial Intelligence"}
        tokens = _tokenize_entry(obj, tok, "content", max_tokens=2048)
        tok.encode.assert_called_once_with("What is AI Artificial Intelligence")
        assert len(tokens) > 0

    def test_prompt_only_format(self):
        tok = _make_tokenizer()
        obj = {"prompt": "Hello world"}
        tokens = _tokenize_entry(obj, tok, "content", max_tokens=2048)
        tok.encode.assert_called_once_with("Hello world")

    def test_plain_text_format(self):
        tok = _make_tokenizer()
        obj = {"content": "Some plain text here"}
        tokens = _tokenize_entry(obj, tok, "content", max_tokens=2048)
        tok.encode.assert_called_once_with("Some plain text here")

    def test_custom_text_key(self):
        tok = _make_tokenizer()
        obj = {"text": "Custom key text"}
        tokens = _tokenize_entry(obj, tok, "text", max_tokens=2048)
        tok.encode.assert_called_once_with("Custom key text")

    def test_max_tokens_truncation(self):
        tok = _make_tokenizer()
        obj = {"content": "a b c d e f g h i j"}
        tokens = _tokenize_entry(obj, tok, "content", max_tokens=5)
        assert len(tokens) == 5

    def test_messages_format_truncation(self):
        tok = _make_tokenizer()
        obj = {
            "messages": [
                {"role": "user", "content": "a b c d e f g h i j"},
            ]
        }
        tokens = _tokenize_entry(obj, tok, "content", max_tokens=5)
        assert len(tokens) == 5

    def test_messages_takes_priority_over_text_key(self):
        """If both 'messages' and text_key exist, messages format wins."""
        tok = _make_tokenizer()
        obj = {
            "messages": [{"role": "user", "content": "Hello"}],
            "content": "Ignored text",
        }
        tokens = _tokenize_entry(obj, tok, "content", max_tokens=2048)
        tok.apply_chat_template.assert_called_once()
        tok.encode.assert_not_called()

    def test_prompt_takes_priority_over_text_key(self):
        """If both 'prompt' and text_key exist, prompt format wins."""
        tok = _make_tokenizer()
        obj = {"prompt": "Question", "content": "Ignored"}
        tokens = _tokenize_entry(obj, tok, "content", max_tokens=2048)
        tok.encode.assert_called_once_with("Question")


class TestLoadJsonl:
    def test_messages_format_file(self, tmp_path):
        tok = _make_tokenizer()
        data = [
            {"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]},
            {"messages": [{"role": "user", "content": "Bye"}, {"role": "assistant", "content": "Goodbye friend"}]},
        ]
        path = tmp_path / "chat.jsonl"
        path.write_text("\n".join(json.dumps(d) for d in data))

        samples = load_jsonl(str(path), tok, max_tokens=2048)
        assert len(samples) == 2
        assert tok.apply_chat_template.call_count == 2

    def test_completions_format_file(self, tmp_path):
        tok = _make_tokenizer()
        data = [
            {"prompt": "Q1", "completion": " A1"},
            {"prompt": "Q2", "completion": " A2"},
        ]
        path = tmp_path / "completions.jsonl"
        path.write_text("\n".join(json.dumps(d) for d in data))

        samples = load_jsonl(str(path), tok, max_tokens=2048)
        assert len(samples) == 2

    def test_plain_text_format_file(self, tmp_path):
        tok = _make_tokenizer()
        data = [
            {"content": "First sample text"},
            {"content": "Second sample text here"},
        ]
        path = tmp_path / "plain.jsonl"
        path.write_text("\n".join(json.dumps(d) for d in data))

        samples = load_jsonl(str(path), tok, max_tokens=2048)
        assert len(samples) == 2

    def test_mixed_formats_in_file(self, tmp_path):
        """Different lines can have different formats."""
        tok = _make_tokenizer()
        data = [
            {"messages": [{"role": "user", "content": "Hello"}]},
            {"content": "Plain text"},
            {"prompt": "Question", "completion": " Answer"},
        ]
        path = tmp_path / "mixed.jsonl"
        path.write_text("\n".join(json.dumps(d) for d in data))

        samples = load_jsonl(str(path), tok, max_tokens=2048)
        assert len(samples) == 3

    def test_subsampling(self, tmp_path):
        tok = _make_tokenizer()
        data = [{"content": f"Sample {i}"} for i in range(20)]
        path = tmp_path / "many.jsonl"
        path.write_text("\n".join(json.dumps(d) for d in data))

        samples = load_jsonl(str(path), tok, max_tokens=2048, max_samples=5)
        assert len(samples) == 5

    def test_empty_lines_skipped(self, tmp_path):
        tok = _make_tokenizer()
        path = tmp_path / "with_blanks.jsonl"
        path.write_text('{"content": "hello"}\n\n{"content": "world"}\n')

        samples = load_jsonl(str(path), tok, max_tokens=2048)
        assert len(samples) == 2


class TestLoadDataset:
    def test_auto_detect_jsonl(self, tmp_path):
        tok = _make_tokenizer()
        data = [
            {"messages": [{"role": "user", "content": "Hi"}]},
        ]
        path = tmp_path / "data.jsonl"
        path.write_text(json.dumps(data[0]))

        samples = load_dataset(str(path), tok, max_tokens=2048)
        assert len(samples) == 1

    def test_auto_detect_directory(self, tmp_path):
        tok = _make_tokenizer()
        (tmp_path / "a.txt").write_text("hello world")
        (tmp_path / "b.txt").write_text("foo bar")

        samples = load_dataset(str(tmp_path), tok, max_tokens=2048)
        assert len(samples) == 2
