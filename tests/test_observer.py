"""Tests for observer hooks."""

import mlx.core as mx
import numpy as np
import pytest

from mlx_fun.observer import install_hooks, remove_hooks, collect_captures


def test_minimax_hook_install_remove(tiny_minimax_moe, sample_input):
    """Hook installs and removes cleanly."""
    # Verify original works
    orig_out = tiny_minimax_moe(sample_input)
    mx.eval(orig_out)

    original_cls = type(tiny_minimax_moe)

    # Install hook
    install_hooks([tiny_minimax_moe], "minimax")
    assert hasattr(tiny_minimax_moe, "_reap_captures")
    assert type(tiny_minimax_moe) is not original_cls

    # Remove hook
    remove_hooks([tiny_minimax_moe])
    assert type(tiny_minimax_moe) is original_cls
    assert not hasattr(tiny_minimax_moe, "_reap_captures")


def test_minimax_hook_captures(tiny_minimax_moe, sample_input):
    """Hook captures inds, scores, norms with correct shapes."""
    install_hooks([tiny_minimax_moe], "minimax")
    out = tiny_minimax_moe(sample_input)
    mx.eval(out)

    captures = collect_captures([tiny_minimax_moe])
    assert len(captures) == 1
    assert len(captures[0]) == 1

    inds, scores, norms = captures[0][0]
    # input shape: (1, 8, 32), top_k=2
    assert inds.shape == (1, 8, 2)
    assert scores.shape == (1, 8, 2)
    assert norms.shape == (1, 8, 2)

    # Expert indices should be in [0, 4)
    assert np.all(inds >= 0) and np.all(inds < 4)

    # Scores should be positive (sigmoid-based)
    assert np.all(scores > 0)

    # Norms should be non-negative
    assert np.all(norms >= 0)

    remove_hooks([tiny_minimax_moe])


def test_minimax_hook_numerical_equivalence(tiny_minimax_moe, sample_input):
    """Output with hook should match output without hook."""
    mx.random.seed(42)
    orig_out = tiny_minimax_moe(sample_input)
    mx.eval(orig_out)

    install_hooks([tiny_minimax_moe], "minimax")
    mx.random.seed(42)
    hooked_out = tiny_minimax_moe(sample_input)
    mx.eval(hooked_out)
    remove_hooks([tiny_minimax_moe])

    np.testing.assert_allclose(
        np.array(orig_out, copy=False),
        np.array(hooked_out, copy=False),
        atol=1e-5,
    )


def test_glm4_hook_install_remove(tiny_glm4_moe, sample_input):
    """GLM4 hook installs and removes cleanly."""
    orig_out = tiny_glm4_moe(sample_input)
    mx.eval(orig_out)

    install_hooks([tiny_glm4_moe], "glm4_moe")
    assert hasattr(tiny_glm4_moe, "_reap_captures")

    remove_hooks([tiny_glm4_moe])
    assert not hasattr(tiny_glm4_moe, "_reap_original_cls")


def test_glm4_hook_captures(tiny_glm4_moe, sample_input):
    """GLM4 hook captures metrics correctly."""
    install_hooks([tiny_glm4_moe], "glm4_moe")
    out = tiny_glm4_moe(sample_input)
    mx.eval(out)

    captures = collect_captures([tiny_glm4_moe])
    assert len(captures) == 1
    assert len(captures[0]) == 1

    inds, scores, norms = captures[0][0]
    assert inds.shape == (1, 8, 2)
    assert scores.shape == (1, 8, 2)
    assert norms.shape == (1, 8, 2)

    remove_hooks([tiny_glm4_moe])


def test_glm4_hook_numerical_equivalence(tiny_glm4_moe, sample_input):
    """GLM4 output with hook should match output without hook."""
    mx.random.seed(42)
    orig_out = tiny_glm4_moe(sample_input)
    mx.eval(orig_out)

    install_hooks([tiny_glm4_moe], "glm4_moe")
    mx.random.seed(42)
    hooked_out = tiny_glm4_moe(sample_input)
    mx.eval(hooked_out)
    remove_hooks([tiny_glm4_moe])

    np.testing.assert_allclose(
        np.array(orig_out, copy=False),
        np.array(hooked_out, copy=False),
        atol=1e-5,
    )


def test_glm4_moe_lite_hook_captures(tiny_glm4_moe, sample_input):
    """glm4_moe_lite uses the same hook as glm4_moe."""
    install_hooks([tiny_glm4_moe], "glm4_moe_lite")
    out = tiny_glm4_moe(sample_input)
    mx.eval(out)

    captures = collect_captures([tiny_glm4_moe])
    assert len(captures) == 1
    assert len(captures[0]) == 1

    inds, scores, norms = captures[0][0]
    assert inds.shape == (1, 8, 2)
    assert scores.shape == (1, 8, 2)
    assert norms.shape == (1, 8, 2)

    remove_hooks([tiny_glm4_moe])


def test_qwen3_hook_install_remove(tiny_qwen3_moe, sample_input):
    """Qwen3 hook installs and removes cleanly."""
    orig_out = tiny_qwen3_moe(sample_input)
    mx.eval(orig_out)

    original_cls = type(tiny_qwen3_moe)
    install_hooks([tiny_qwen3_moe], "qwen3_moe")
    assert hasattr(tiny_qwen3_moe, "_reap_captures")
    assert type(tiny_qwen3_moe) is not original_cls

    remove_hooks([tiny_qwen3_moe])
    assert type(tiny_qwen3_moe) is original_cls
    assert not hasattr(tiny_qwen3_moe, "_reap_captures")


def test_qwen3_hook_captures(tiny_qwen3_moe, sample_input):
    """Qwen3 hook captures metrics correctly."""
    install_hooks([tiny_qwen3_moe], "qwen3_moe")
    out = tiny_qwen3_moe(sample_input)
    mx.eval(out)

    captures = collect_captures([tiny_qwen3_moe])
    assert len(captures) == 1
    assert len(captures[0]) == 1

    inds, scores, norms = captures[0][0]
    assert inds.shape == (1, 8, 2)
    assert scores.shape == (1, 8, 2)
    assert norms.shape == (1, 8, 2)

    # Expert indices should be in [0, 4)
    assert np.all(inds >= 0) and np.all(inds < 4)

    # Softmax scores should be positive and sum <= 1
    assert np.all(scores > 0)

    remove_hooks([tiny_qwen3_moe])


def test_qwen3_hook_numerical_equivalence(tiny_qwen3_moe, sample_input):
    """Qwen3 output with hook should match output without hook."""
    mx.random.seed(42)
    orig_out = tiny_qwen3_moe(sample_input)
    mx.eval(orig_out)

    install_hooks([tiny_qwen3_moe], "qwen3_moe")
    mx.random.seed(42)
    hooked_out = tiny_qwen3_moe(sample_input)
    mx.eval(hooked_out)
    remove_hooks([tiny_qwen3_moe])

    np.testing.assert_allclose(
        np.array(orig_out, copy=False),
        np.array(hooked_out, copy=False),
        atol=1e-5,
    )


def test_qwen3_next_hook_install_remove(tiny_qwen3_next_moe, sample_input):
    """Qwen3Next hook installs and removes cleanly."""
    orig_out = tiny_qwen3_next_moe(sample_input)
    mx.eval(orig_out)

    original_cls = type(tiny_qwen3_next_moe)
    install_hooks([tiny_qwen3_next_moe], "qwen3_next")
    assert hasattr(tiny_qwen3_next_moe, "_reap_captures")
    assert type(tiny_qwen3_next_moe) is not original_cls

    remove_hooks([tiny_qwen3_next_moe])
    assert type(tiny_qwen3_next_moe) is original_cls
    assert not hasattr(tiny_qwen3_next_moe, "_reap_captures")


def test_qwen3_next_hook_captures(tiny_qwen3_next_moe, sample_input):
    """Qwen3Next hook captures metrics correctly."""
    install_hooks([tiny_qwen3_next_moe], "qwen3_next")
    out = tiny_qwen3_next_moe(sample_input)
    mx.eval(out)

    captures = collect_captures([tiny_qwen3_next_moe])
    assert len(captures) == 1
    assert len(captures[0]) == 1

    inds, scores, norms = captures[0][0]
    assert inds.shape == (1, 8, 2)
    assert scores.shape == (1, 8, 2)
    assert norms.shape == (1, 8, 2)

    assert np.all(inds >= 0) and np.all(inds < 4)
    assert np.all(scores > 0)

    remove_hooks([tiny_qwen3_next_moe])


def test_qwen3_next_hook_numerical_equivalence(tiny_qwen3_next_moe, sample_input):
    """Qwen3Next output with hook should match output without hook."""
    mx.random.seed(42)
    orig_out = tiny_qwen3_next_moe(sample_input)
    mx.eval(orig_out)

    install_hooks([tiny_qwen3_next_moe], "qwen3_next")
    mx.random.seed(42)
    hooked_out = tiny_qwen3_next_moe(sample_input)
    mx.eval(hooked_out)
    remove_hooks([tiny_qwen3_next_moe])

    np.testing.assert_allclose(
        np.array(orig_out, copy=False),
        np.array(hooked_out, copy=False),
        atol=1e-5,
    )


def test_minimax_m2_hook_captures(tiny_minimax_moe, sample_input):
    """minimax_m2 model_type uses the same hook as minimax."""
    install_hooks([tiny_minimax_moe], "minimax_m2")
    out = tiny_minimax_moe(sample_input)
    mx.eval(out)

    captures = collect_captures([tiny_minimax_moe])
    assert len(captures) == 1
    assert len(captures[0]) == 1

    inds, scores, norms = captures[0][0]
    assert inds.shape == (1, 8, 2)
    assert scores.shape == (1, 8, 2)
    assert norms.shape == (1, 8, 2)

    remove_hooks([tiny_minimax_moe])


def test_multiple_forward_passes(tiny_minimax_moe, sample_input):
    """Multiple forward passes accumulate captures."""
    install_hooks([tiny_minimax_moe], "minimax")

    tiny_minimax_moe(sample_input)
    tiny_minimax_moe(sample_input)
    mx.eval(tiny_minimax_moe.parameters())

    captures = collect_captures([tiny_minimax_moe])
    assert len(captures[0]) == 2

    # After collect, captures should be cleared
    captures2 = collect_captures([tiny_minimax_moe])
    assert len(captures2[0]) == 0

    remove_hooks([tiny_minimax_moe])


def test_unknown_model_type():
    with pytest.raises(ValueError, match="No hook"):
        install_hooks([], "unknown_model")
