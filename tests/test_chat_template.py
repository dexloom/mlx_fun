"""Tests for chat-template resolution.

``serve`` and the probe commands resolve templates with deliberately different
policies, so both resolvers live here.
"""

from pathlib import Path

import pytest

from mlx_fun.chat_template import (
    _MODEL_TYPE_TEMPLATES,
    _resolve_chat_template,
    probe_chat_template_source,
    resolve_probe_chat_template,
)


TEMPLATE_DIR = Path(__file__).resolve().parents[1] / "src" / "mlx_fun" / "templates"


class TestBundledMap:
    def test_every_mapped_template_exists(self):
        missing = [
            name for name in set(_MODEL_TYPE_TEMPLATES.values())
            if not (TEMPLATE_DIR / name).is_file()
        ]
        assert missing == []

    def test_qwen_moe_vlms_map_to_the_qwen_template(self):
        assert _MODEL_TYPE_TEMPLATES["qwen4_exp"] == "qwen35.jinja"
        assert _MODEL_TYPE_TEMPLATES["qwen3_5_moe"] == "qwen35.jinja"


class TestResolveProbeChatTemplate:
    def test_none_keeps_the_checkpoint_template(self):
        assert resolve_probe_chat_template(None, "qwen3_5_moe") is None

    def test_file_path_is_read(self, tmp_path):
        path = tmp_path / "custom.jinja"
        path.write_text("{{ messages }}")
        assert resolve_probe_chat_template(str(path), "qwen3_moe") == "{{ messages }}"

    def test_bundled_reads_the_bundled_template(self):
        content = resolve_probe_chat_template("bundled", "qwen3_5_moe")
        assert content == (TEMPLATE_DIR / "qwen35.jinja").read_text()

    def test_inline_string_passes_through(self):
        inline = "{% for m in messages %}{{ m.content }}{% endfor %}"
        assert resolve_probe_chat_template(inline, "qwen3_moe") == inline

    def test_bundled_with_unknown_type_raises(self):
        with pytest.raises(ValueError, match="No bundled chat template"):
            resolve_probe_chat_template("bundled", "not_a_model_type")

    def test_a_checkpoint_template_is_never_substituted(self):
        """The probes must not silently swap in a bundled template the way
        ``serve`` does — that would change what is being measured."""
        assert resolve_probe_chat_template(None, "minimax") is None


class TestProbeChatTemplateSource:
    def test_labels(self, tmp_path):
        path = tmp_path / "t.jinja"
        path.write_text("x")
        assert probe_chat_template_source(None) == "checkpoint"
        assert probe_chat_template_source("bundled") == "bundled"
        assert probe_chat_template_source(str(path)) == "file"
        assert probe_chat_template_source("{{ messages }}") == "inline"


class TestServeResolver:
    def test_explicit_file_wins(self, tmp_path):
        path = tmp_path / "explicit.jinja"
        path.write_text("EXPLICIT")
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "chat_template.jinja").write_text("OWN")
        assert _resolve_chat_template(str(path), "qwen3_moe", model_dir) == "EXPLICIT"

    def test_model_dir_template_beats_bundled(self, tmp_path):
        (tmp_path / "chat_template.jinja").write_text("OWN")
        assert _resolve_chat_template(None, "qwen3_moe", tmp_path) == "OWN"

    def test_falls_back_to_bundled(self, tmp_path):
        content = _resolve_chat_template(None, "qwen3_moe", tmp_path)
        assert content == (TEMPLATE_DIR / "qwen35.jinja").read_text()

    def test_unknown_type_returns_none(self, tmp_path):
        assert _resolve_chat_template(None, "not_a_model_type", tmp_path) is None


class TestServerReExports:
    """server.py's names stay importable after the move."""

    def test_server_exposes_both(self):
        import mlx_fun.server as server

        assert server._MODEL_TYPE_TEMPLATES is _MODEL_TYPE_TEMPLATES
        assert server._resolve_chat_template is _resolve_chat_template
