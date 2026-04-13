"""Tests for hidden state capture hooks (speculative decoding Phase 2)."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from mlx_fun.hidden_state_capture import (
    HiddenStateCapture,
    parse_capture_layers,
)


# ---------------------------------------------------------------------------
# Tiny model fixtures (minimal decoder-layer stack)
# ---------------------------------------------------------------------------

class TinyDecoderLayer(nn.Module):
    """Minimal decoder layer: linear transform + residual."""

    def __init__(self, hidden=32):
        super().__init__()
        self.linear = nn.Linear(hidden, hidden)

    def __call__(self, x, mask=None, cache=None):
        return x + self.linear(x)


class TinyInnerModel(nn.Module):
    """model.model — has .layers list."""

    def __init__(self, n_layers=4, hidden=32):
        super().__init__()
        self.layers = [TinyDecoderLayer(hidden) for _ in range(n_layers)]
        self.norm = nn.RMSNorm(hidden)

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class TinyModel(nn.Module):
    """Top-level model with model.model.layers structure."""

    def __init__(self, n_layers=4, hidden=32, vocab=64):
        super().__init__()
        self.model = TinyInnerModel(n_layers, hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    def __call__(self, x):
        h = self.model(x)
        return self.lm_head(h)


@pytest.fixture
def tiny_model():
    mx.random.seed(42)
    model = TinyModel(n_layers=4, hidden=32, vocab=64)
    mx.eval(model.parameters())
    return model


@pytest.fixture
def sample_hidden_input():
    """Input tensor for the tiny model: (batch=1, seq=8, hidden=32)."""
    mx.random.seed(0)
    x = mx.random.normal((1, 8, 32))
    mx.eval(x)
    return x


# ---------------------------------------------------------------------------
# HiddenStateCapture tests
# ---------------------------------------------------------------------------

class TestHiddenStateCapture:
    def test_install_and_remove(self, tiny_model):
        capture = HiddenStateCapture(tiny_model, layer_indices=[0, 2])
        assert not capture.installed

        capture.install()
        assert capture.installed
        assert capture.num_layers == 2

        # Verify class was swapped
        layer0 = tiny_model.model.layers[0]
        assert "_HiddenCapture_" in type(layer0).__name__

        # Layer 1 should be untouched
        layer1 = tiny_model.model.layers[1]
        assert "_HiddenCapture_" not in type(layer1).__name__

        capture.remove()
        assert not capture.installed
        assert "_HiddenCapture_" not in type(tiny_model.model.layers[0]).__name__

    def test_install_idempotent(self, tiny_model):
        capture = HiddenStateCapture(tiny_model, layer_indices=[0])
        capture.install()
        capture.install()  # Should not raise or double-install
        assert capture.installed
        capture.remove()

    def test_captures_correct_shape(self, tiny_model, sample_hidden_input):
        capture = HiddenStateCapture(tiny_model, layer_indices=[0, 3])
        capture.install()

        tiny_model(sample_hidden_input)

        states = capture.collect()
        assert set(states.keys()) == {0, 3}

        for idx in [0, 3]:
            assert len(states[idx]) == 1
            arr = states[idx][0]
            assert isinstance(arr, mx.array)
            assert arr.shape == (1, 8, 32)  # (batch, seq, hidden)

        capture.remove()

    def test_captures_all_layers(self, tiny_model, sample_hidden_input):
        capture = HiddenStateCapture(tiny_model, layer_indices=None)
        capture.install()
        assert capture.num_layers == 4

        tiny_model(sample_hidden_input)

        states = capture.collect()
        assert set(states.keys()) == {0, 1, 2, 3}
        for idx in range(4):
            assert len(states[idx]) == 1

        capture.remove()

    def test_multiple_forward_passes_accumulate(self, tiny_model, sample_hidden_input):
        capture = HiddenStateCapture(tiny_model, layer_indices=[1])
        capture.install()

        tiny_model(sample_hidden_input)
        tiny_model(sample_hidden_input)

        states = capture.collect()
        assert len(states[1]) == 2

        capture.remove()

    def test_clear_resets_captures(self, tiny_model, sample_hidden_input):
        capture = HiddenStateCapture(tiny_model, layer_indices=[0])
        capture.install()

        tiny_model(sample_hidden_input)
        assert len(capture.collect()[0]) == 1

        capture.clear()
        assert len(capture.collect()[0]) == 0

        # New forward pass works after clear
        tiny_model(sample_hidden_input)
        assert len(capture.collect()[0]) == 1

        capture.remove()

    def test_collect_latest(self, tiny_model, sample_hidden_input):
        capture = HiddenStateCapture(tiny_model, layer_indices=[0, 2])
        capture.install()

        tiny_model(sample_hidden_input)
        tiny_model(sample_hidden_input)

        latest = capture.collect_latest()
        assert set(latest.keys()) == {0, 2}
        for idx in [0, 2]:
            assert isinstance(latest[idx], mx.array)
            assert latest[idx].shape == (1, 8, 32)

        capture.remove()

    def test_collect_latest_empty(self, tiny_model):
        capture = HiddenStateCapture(tiny_model, layer_indices=[0])
        capture.install()

        latest = capture.collect_latest()
        assert latest == {}

        capture.remove()

    def test_output_not_affected_by_hooks(self, tiny_model, sample_hidden_input):
        """Hooks should not change the model's output."""
        out_before = tiny_model(sample_hidden_input)
        mx.eval(out_before)
        np_before = np.array(out_before, copy=False).copy()

        capture = HiddenStateCapture(tiny_model, layer_indices=[0, 1, 2, 3])
        capture.install()

        out_after = tiny_model(sample_hidden_input)
        mx.eval(out_after)
        np_after = np.array(out_after, copy=False)

        np.testing.assert_allclose(np_before, np_after, atol=1e-5)

        capture.remove()

    def test_captured_states_differ_between_layers(self, tiny_model, sample_hidden_input):
        """Different layers should produce different hidden states."""
        capture = HiddenStateCapture(tiny_model, layer_indices=[0, 3])
        capture.install()

        tiny_model(sample_hidden_input)

        states = capture.collect()
        h0 = np.array(states[0][0], copy=False)
        h3 = np.array(states[3][0], copy=False)

        # They should be different (each layer transforms the hidden state)
        assert not np.allclose(h0, h3, atol=1e-5)

        capture.remove()

    def test_repr(self, tiny_model):
        capture = HiddenStateCapture(tiny_model, layer_indices=[0, 2])
        assert "not installed" in repr(capture)
        capture.install()
        assert "installed" in repr(capture)
        assert "[0, 2]" in repr(capture)
        capture.remove()


# ---------------------------------------------------------------------------
# parse_capture_layers tests
# ---------------------------------------------------------------------------

class TestParseCaptureLayers:
    def test_none_returns_none(self):
        assert parse_capture_layers(None, 32) is None

    def test_all_returns_all_indices(self):
        result = parse_capture_layers("all", 4)
        assert result == [0, 1, 2, 3]

    def test_all_case_insensitive(self):
        result = parse_capture_layers("ALL", 4)
        assert result == [0, 1, 2, 3]

    def test_comma_separated(self):
        result = parse_capture_layers("0,2,3", 4)
        assert result == [0, 2, 3]

    def test_whitespace_handling(self):
        result = parse_capture_layers(" 0 , 2 , 3 ", 4)
        assert result == [0, 2, 3]

    def test_deduplicates_and_sorts(self):
        result = parse_capture_layers("3,1,3,0,1", 4)
        assert result == [0, 1, 3]

    def test_single_index(self):
        result = parse_capture_layers("2", 4)
        assert result == [2]

    def test_empty_string_returns_none(self):
        assert parse_capture_layers("", 4) is None

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError, match="out of range"):
            parse_capture_layers("5", 4)

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="out of range"):
            parse_capture_layers("-1", 4)

    def test_invalid_int_raises(self):
        with pytest.raises(ValueError):
            parse_capture_layers("abc", 4)
