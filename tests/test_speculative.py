"""Tests for speculative decoding server integration (Phase 1)."""

import argparse
from unittest.mock import patch, MagicMock

import mlx.core as mx
import mlx.nn as nn
import pytest

from click.testing import CliRunner

from mlx_fun.cli import main
from mlx_fun.server import ReapModelProvider, _make_cli_args


# ---------------------------------------------------------------------------
# _make_cli_args tests
# ---------------------------------------------------------------------------

class TestMakeCliArgs:
    def test_defaults_include_draft_fields(self):
        args = _make_cli_args()
        assert args.draft_model is None
        assert args.num_draft_tokens == 3

    def test_draft_model_override(self):
        args = _make_cli_args(draft_model="/tmp/draft", num_draft_tokens=5)
        assert args.draft_model == "/tmp/draft"
        assert args.num_draft_tokens == 5


# ---------------------------------------------------------------------------
# ReapModelProvider draft model tests
# ---------------------------------------------------------------------------

class _TinyModel(nn.Module):
    """Minimal model with a layers attribute for make_prompt_cache."""

    def __init__(self, n_layers=2, hidden=32):
        super().__init__()
        self.layers = [nn.Linear(hidden, hidden) for _ in range(n_layers)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class _FakeTokenizer:
    def __init__(self, vocab_size=32000):
        self.vocab_size = vocab_size


class TestReapModelProviderDraft:
    def _make_provider(self, draft_model=None, num_draft_tokens=3):
        """Create a ReapModelProvider with optional draft model config."""
        model = _TinyModel()
        mx.eval(model.parameters())
        tokenizer = _FakeTokenizer()
        cli_args = _make_cli_args(draft_model=draft_model, num_draft_tokens=num_draft_tokens)

        draft = _TinyModel() if draft_model else None
        if draft:
            mx.eval(draft.parameters())
        draft_tok = _FakeTokenizer() if draft_model else None

        with patch("mlx_lm.load") as mock_load:
            mock_load.return_value = (draft, draft_tok)
            provider = ReapModelProvider(model, tokenizer, cli_args)

        return provider

    def test_no_draft_model(self):
        provider = self._make_provider(draft_model=None)
        assert provider.draft_model is None
        assert provider.model_key == ("reap_preloaded", None, None)

    def test_draft_model_loaded(self):
        provider = self._make_provider(draft_model="/tmp/draft")
        assert provider.draft_model is not None
        assert provider.model_key == ("reap_preloaded", None, "/tmp/draft")

    def test_draft_model_disables_batching(self):
        provider = self._make_provider(draft_model="/tmp/draft")
        assert provider.is_batchable is False

    def test_no_draft_allows_batching(self):
        """Without draft model, batchability depends on cache merge support."""
        provider = self._make_provider(draft_model=None)
        # is_batchable is determined by make_prompt_cache — just check it's set
        assert isinstance(provider.is_batchable, bool)

    def test_load_returns_model_and_tokenizer(self):
        provider = self._make_provider(draft_model="/tmp/draft")
        model, tok = provider.load()
        assert model is provider.model
        assert tok is provider.tokenizer

    def test_load_default_model_keeps_draft(self):
        """load(draft_model_path='default_model') keeps the CLI-loaded draft."""
        provider = self._make_provider(draft_model="/tmp/draft")
        original_draft = provider.draft_model
        provider.load(draft_model_path="default_model")
        assert provider.draft_model is original_draft

    def test_load_explicit_path_reloads_draft(self):
        """load(draft_model_path='/other') loads a new draft model."""
        provider = self._make_provider(draft_model="/tmp/draft")
        original_draft = provider.draft_model

        new_draft = _TinyModel()
        mx.eval(new_draft.parameters())
        new_tok = _FakeTokenizer()

        with patch("mlx_lm.load") as mock_load:
            mock_load.return_value = (new_draft, new_tok)
            provider.load(draft_model_path="/tmp/other_draft")

        assert provider.draft_model is new_draft
        assert provider.model_key == ("reap_preloaded", None, "/tmp/other_draft")

    def test_vocab_mismatch_warns(self):
        """Mismatched vocab sizes should log a warning, not raise."""
        model = _TinyModel()
        mx.eval(model.parameters())
        tokenizer = _FakeTokenizer(vocab_size=32000)
        cli_args = _make_cli_args(draft_model="/tmp/draft")

        draft = _TinyModel()
        mx.eval(draft.parameters())
        draft_tok = _FakeTokenizer(vocab_size=16000)  # mismatch

        import logging as _logging
        with patch("mlx_lm.load") as mock_load, \
             patch.object(_logging, "warning") as mock_warn:
            mock_load.return_value = (draft, draft_tok)
            provider = ReapModelProvider(model, tokenizer, cli_args)

        assert provider.draft_model is draft
        mock_warn.assert_called_once()
        assert "vocab" in mock_warn.call_args[0][0].lower()


# ---------------------------------------------------------------------------
# CLI option tests
# ---------------------------------------------------------------------------

class TestServeCLIOptions:
    def test_serve_help_shows_draft_options(self):
        runner = CliRunner()
        result = runner.invoke(main, ["serve", "--help"])
        assert result.exit_code == 0
        assert "--draft-model" in result.output
        assert "--num-draft-tokens" in result.output

    def test_serve_help_shows_default_draft_tokens(self):
        runner = CliRunner()
        result = runner.invoke(main, ["serve", "--help"])
        assert "Default: 3" in result.output
