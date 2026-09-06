"""Tests for the backend-aware checkpoint loader (mlx-lm vs mlx-vlm)."""

import json

import pytest

from mlx_fun.loader import (
    is_vision_model,
    language_model,
    read_config,
    text_config,
    text_forward,
)


# ---------------------------------------------------------------------------
# Vision-model detection
# ---------------------------------------------------------------------------

# The Qwen MoE VLM family: one adapter and one hook set, two model_type strings.
QWEN_MOE_VLM_TYPES = ["qwen4_exp", "qwen3_5_moe"]


class TestIsVisionModel:
    @pytest.mark.parametrize("model_type", QWEN_MOE_VLM_TYPES)
    def test_qwen_moe_vlm_is_vision(self, model_type):
        # Qwen3.8-Flash-Next / Qwen3.6-35B-A3B: mlx-vlm-only, even before
        # looking at their blocks.
        assert is_vision_model({"model_type": model_type})

    def test_explicit_vision_config(self):
        assert is_vision_model({
            "model_type": "some_vlm",
            "vision_config": {"depth": 27, "hidden_size": 1152},
        })

    def test_empty_vision_config_is_not_vision(self):
        # A stub key with no content must not divert a text model to mlx-vlm.
        assert not is_vision_model({"model_type": "qwen3_moe", "vision_config": {}})

    def test_image_token_plus_text_config(self):
        assert is_vision_model({
            "model_type": "unknown_mm",
            "image_token_id": 248056,
            "text_config": {"num_hidden_layers": 4},
        })

    def test_text_model_is_not_vision(self):
        assert not is_vision_model({"model_type": "qwen3_moe", "num_experts": 128})

    def test_text_config_alone_is_not_vision(self):
        # Kimi-K2.5/2.6 nests a text_config without being multimodal.
        assert not is_vision_model({
            "model_type": "kimi_k25",
            "text_config": {"num_hidden_layers": 4},
        })


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

class TestTextConfig:
    @pytest.mark.parametrize("model_type", QWEN_MOE_VLM_TYPES)
    def test_returns_nested_when_present(self, model_type):
        cfg = {"model_type": model_type, "text_config": {"num_experts": 512}}
        assert text_config(cfg) == {"num_experts": 512}

    def test_returns_flat_when_absent(self):
        cfg = {"model_type": "qwen3_moe", "num_experts": 128}
        assert text_config(cfg) is cfg

    def test_empty_nested_falls_back_to_flat(self):
        cfg = {"model_type": "qwen3_moe", "num_experts": 128, "text_config": {}}
        assert text_config(cfg) is cfg


class TestReadConfig:
    def test_reads_local_config(self, tmp_path):
        (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen4_exp"}))
        assert read_config(str(tmp_path)) == {"model_type": "qwen4_exp"}


# ---------------------------------------------------------------------------
# Model unwrapping
# ---------------------------------------------------------------------------

class FakeLanguageStack:
    def __call__(self, tokens):
        return ("language", tokens)


class FakeVLM:
    def __init__(self):
        self.vision_tower = object()
        self.language_model = FakeLanguageStack()

    def __call__(self, tokens):
        raise AssertionError("multimodal wrapper must not run token-only passes")


class FakeTextModel:
    def __call__(self, tokens):
        return ("text", tokens)


class TestLanguageModel:
    def test_unwraps_multimodal(self):
        model = FakeVLM()
        assert language_model(model) is model.language_model

    def test_passes_through_text_model(self):
        model = FakeTextModel()
        assert language_model(model) is model


class TestTextForward:
    @pytest.mark.parametrize("model_type", QWEN_MOE_VLM_TYPES)
    def test_vision_forward_targets_language_stack(self, model_type):
        model = FakeVLM()
        forward = text_forward(model, {"model_type": model_type})
        assert forward(["tok"]) == ("language", ["tok"])

    def test_text_forward_is_the_model_itself(self):
        model = FakeTextModel()
        forward = text_forward(model, {"model_type": "qwen3_moe"})
        assert forward is model
        assert forward(["tok"]) == ("text", ["tok"])


# ---------------------------------------------------------------------------
# Backend routing
# ---------------------------------------------------------------------------

class TestLoadModelRouting:
    @pytest.mark.parametrize("model_type", QWEN_MOE_VLM_TYPES)
    def test_vision_checkpoint_without_mlx_vlm_explains_itself(
        self, tmp_path, monkeypatch, model_type,
    ):
        """A VLM checkpoint with mlx-vlm absent must say how to install it."""
        (tmp_path / "config.json").write_text(json.dumps({
            "model_type": model_type,
            "vision_config": {"depth": 27},
            "text_config": {"num_hidden_layers": 48},
        }))

        import builtins

        real_import = builtins.__import__

        def no_mlx_vlm(name, *args, **kwargs):
            if name == "mlx_vlm":
                raise ImportError("No module named 'mlx_vlm'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_mlx_vlm)

        from mlx_fun.loader import load_model

        with pytest.raises(ImportError, match="mlx-vlm"):
            load_model(str(tmp_path))

    def test_text_checkpoint_uses_mlx_lm(self, tmp_path, monkeypatch):
        (tmp_path / "config.json").write_text(json.dumps({
            "model_type": "qwen3_moe", "num_experts": 8,
        }))

        called = {}

        def fake_mlx_load(path, tokenizer_config=None, lazy=False, return_config=False):
            called["path"] = path
            called["return_config"] = return_config
            return ("model", "tokenizer", {"model_type": "qwen3_moe"})

        import mlx_lm
        monkeypatch.setattr(mlx_lm, "load", fake_mlx_load)

        from mlx_fun.loader import load_model

        model, tokenizer, config = load_model(str(tmp_path))
        assert (model, tokenizer) == ("model", "tokenizer")
        assert config["model_type"] == "qwen3_moe"
        assert called["return_config"] is True


class TestVLMTokenizerConfig:
    """``mlx_vlm.load`` takes no tokenizer_config, so a chat-template override
    has to be applied to the tokenizer it returns."""

    @staticmethod
    def _fake_vlm(tmp_path, monkeypatch, model_type="qwen3_5_moe"):
        pytest.importorskip("mlx_vlm", reason="mlx-vlm not installed")
        import mlx_vlm

        (tmp_path / "config.json").write_text(json.dumps({
            "model_type": model_type,
            "image_token_id": 151655,
            "text_config": {"num_hidden_layers": 40, "num_experts": 256},
        }))

        class FakeTokenizer:
            chat_template = "ORIGINAL"

        class FakeProcessor:
            def __init__(self):
                self.tokenizer = FakeTokenizer()

        processor = FakeProcessor()
        monkeypatch.setattr(
            mlx_vlm, "load",
            lambda path, lazy=False, trust_remote_code=False: ("model", processor),
        )
        return processor

    def test_chat_template_is_applied(self, tmp_path, monkeypatch):
        processor = self._fake_vlm(tmp_path, monkeypatch)
        from mlx_fun.loader import load_model

        _model, tokenizer, config = load_model(
            str(tmp_path), tokenizer_config={"chat_template": "OVERRIDE"},
        )
        assert tokenizer is processor.tokenizer
        assert tokenizer.chat_template == "OVERRIDE"
        assert config["model_type"] == "qwen3_5_moe"

    def test_no_override_leaves_the_checkpoint_template(self, tmp_path, monkeypatch):
        processor = self._fake_vlm(tmp_path, monkeypatch)
        from mlx_fun.loader import load_model

        _model, tokenizer, _config = load_model(str(tmp_path))
        assert tokenizer.chat_template == "ORIGINAL"
        assert processor.tokenizer.chat_template == "ORIGINAL"


class TestGemma4AssistantImports:
    """Regression: the Gemma 4 MTP drafter's mask helper must import from
    mlx_lm.models.base, not a nonexistent mlx_fun.models.base (which raised
    ModuleNotFoundError only at forward time, inside _make_masks)."""

    def test_no_bad_module_reference(self):
        import inspect
        import mlx_fun.models.gemma4_assistant as g

        src = inspect.getsource(g)
        assert "from mlx_fun.models.base" not in src
        assert "from .base import" not in src

    def test_mask_helper_resolves(self):
        # The symbol _make_masks pulls in must actually exist where it now points.
        from mlx_lm.models.base import create_attention_mask  # noqa: F401
