"""Tests for REAM-specific capture hooks."""

import mlx.core as mx
import numpy as np
import pytest

from mlx_fun.ream_hooks import (
    install_ream_hooks,
    remove_ream_hooks,
    collect_ream_data,
)


class TestReamHooksMiniMax:
    def test_captures_input_and_gate_logits(self, tiny_minimax_moe, sample_input):
        install_ream_hooks([tiny_minimax_moe], "minimax")
        tiny_minimax_moe(sample_input)
        captures = collect_ream_data([tiny_minimax_moe])
        remove_ream_hooks([tiny_minimax_moe])

        assert len(captures) == 1
        assert len(captures[0]) == 1
        inp, gates, sel = captures[0][0]
        # input shape: (batch=1, seq=8, hidden=32)
        assert inp.shape == (1, 8, 32)
        # gate logits shape: (batch=1, seq=8, n_experts=4)
        assert gates.shape == (1, 8, 4)
        # Third element is the real top-k selection, (batch, seq, top_k).
        assert sel.shape == (1, 8, 2)

    def test_output_matches_original(self, tiny_minimax_moe, sample_input):
        # Get reference output without hooks
        ref_out = tiny_minimax_moe(sample_input)
        mx.eval(ref_out)
        ref_np = np.array(ref_out, copy=True)

        # Get output with REAM hooks
        install_ream_hooks([tiny_minimax_moe], "minimax")
        hooked_out = tiny_minimax_moe(sample_input)
        mx.eval(hooked_out)
        hooked_np = np.array(hooked_out, copy=True)
        remove_ream_hooks([tiny_minimax_moe])

        np.testing.assert_allclose(ref_np, hooked_np, atol=1e-5)

    def test_install_remove_cycle(self, tiny_minimax_moe):
        original_cls = type(tiny_minimax_moe)
        install_ream_hooks([tiny_minimax_moe], "minimax")
        assert type(tiny_minimax_moe) != original_cls
        assert hasattr(tiny_minimax_moe, "_ream_captures")

        remove_ream_hooks([tiny_minimax_moe])
        assert type(tiny_minimax_moe) == original_cls
        assert not hasattr(tiny_minimax_moe, "_ream_captures")


class TestReamHooksGLM4:
    def test_captures_input_and_gate_logits(self, tiny_glm4_moe, sample_input):
        install_ream_hooks([tiny_glm4_moe], "glm4_moe")
        tiny_glm4_moe(sample_input)
        captures = collect_ream_data([tiny_glm4_moe])
        remove_ream_hooks([tiny_glm4_moe])

        assert len(captures[0]) == 1
        inp, gates, sel = captures[0][0]
        assert inp.shape == (1, 8, 32)
        assert gates.shape == (1, 8, 4)
        # Third element is the real top-k selection, (batch, seq, top_k).
        assert sel.shape == (1, 8, 2)

    def test_output_matches_original(self, tiny_glm4_moe, sample_input):
        ref_out = tiny_glm4_moe(sample_input)
        mx.eval(ref_out)
        ref_np = np.array(ref_out, copy=True)

        install_ream_hooks([tiny_glm4_moe], "glm4_moe")
        hooked_out = tiny_glm4_moe(sample_input)
        mx.eval(hooked_out)
        hooked_np = np.array(hooked_out, copy=True)
        remove_ream_hooks([tiny_glm4_moe])

        np.testing.assert_allclose(ref_np, hooked_np, atol=1e-5)


class TestReamHooksQwen3:
    def test_captures_input_and_gate_logits(self, tiny_qwen3_moe, sample_input):
        install_ream_hooks([tiny_qwen3_moe], "qwen3_moe")
        tiny_qwen3_moe(sample_input)
        captures = collect_ream_data([tiny_qwen3_moe])
        remove_ream_hooks([tiny_qwen3_moe])

        assert len(captures[0]) == 1
        inp, gates, sel = captures[0][0]
        assert inp.shape == (1, 8, 32)
        assert gates.shape == (1, 8, 4)
        # Third element is the real top-k selection, (batch, seq, top_k).
        assert sel.shape == (1, 8, 2)

    def test_output_matches_original(self, tiny_qwen3_moe, sample_input):
        ref_out = tiny_qwen3_moe(sample_input)
        mx.eval(ref_out)
        ref_np = np.array(ref_out, copy=True)

        install_ream_hooks([tiny_qwen3_moe], "qwen3_moe")
        hooked_out = tiny_qwen3_moe(sample_input)
        mx.eval(hooked_out)
        hooked_np = np.array(hooked_out, copy=True)
        remove_ream_hooks([tiny_qwen3_moe])

        np.testing.assert_allclose(ref_np, hooked_np, atol=1e-5)


class TestReamHooksQwen3Next:
    def test_captures_input_and_gate_logits(self, tiny_qwen3_next_moe, sample_input):
        install_ream_hooks([tiny_qwen3_next_moe], "qwen3_next")
        tiny_qwen3_next_moe(sample_input)
        captures = collect_ream_data([tiny_qwen3_next_moe])
        remove_ream_hooks([tiny_qwen3_next_moe])

        assert len(captures[0]) == 1
        inp, gates, sel = captures[0][0]
        assert inp.shape == (1, 8, 32)
        assert gates.shape == (1, 8, 4)
        # Third element is the real top-k selection, (batch, seq, top_k).
        assert sel.shape == (1, 8, 2)

    def test_output_matches_original(self, tiny_qwen3_next_moe, sample_input):
        ref_out = tiny_qwen3_next_moe(sample_input)
        mx.eval(ref_out)
        ref_np = np.array(ref_out, copy=True)

        install_ream_hooks([tiny_qwen3_next_moe], "qwen3_next")
        hooked_out = tiny_qwen3_next_moe(sample_input)
        mx.eval(hooked_out)
        hooked_np = np.array(hooked_out, copy=True)
        remove_ream_hooks([tiny_qwen3_next_moe])

        np.testing.assert_allclose(ref_np, hooked_np, atol=1e-5)


class TestReamHooksMultipleCaptures:
    def test_multiple_forward_passes(self, tiny_qwen3_moe, sample_input):
        install_ream_hooks([tiny_qwen3_moe], "qwen3_moe")

        tiny_qwen3_moe(sample_input)
        tiny_qwen3_moe(sample_input)

        captures = collect_ream_data([tiny_qwen3_moe])
        remove_ream_hooks([tiny_qwen3_moe])

        # Should have 2 captures from 2 forward passes
        assert len(captures[0]) == 2

    def test_collect_clears_captures(self, tiny_qwen3_moe, sample_input):
        install_ream_hooks([tiny_qwen3_moe], "qwen3_moe")

        tiny_qwen3_moe(sample_input)
        captures1 = collect_ream_data([tiny_qwen3_moe])
        assert len(captures1[0]) == 1

        # After collecting, buffer should be empty
        captures2 = collect_ream_data([tiny_qwen3_moe])
        assert len(captures2[0]) == 0

        remove_ream_hooks([tiny_qwen3_moe])


class TestReamHooksRealSelection:
    """The captured selection must be the gate's real top-k, not a raw-logit
    reconstruction — this is what safety-scan and domain-scan consume."""

    def test_minimax_selection_honors_correction_bias(self, tiny_minimax_moe, sample_input):
        from mlx_fun.safety import compute_top_k_from_logits

        block = tiny_minimax_moe
        # Push expert 3 to the top of the corrected score while leaving the raw
        # sigmoid logits unchanged, so the real selection and the raw-logit
        # reconstruction diverge.
        block.e_score_correction_bias = mx.array([0.0, 0.0, 0.0, 10.0])

        install_ream_hooks([block], "minimax")
        block(sample_input)
        inp, gates, sel = collect_ream_data([block])[0][0]
        remove_ream_hooks([block])

        captured = sel.reshape(-1, sel.shape[-1])
        reconstructed = compute_top_k_from_logits(
            gates.reshape(-1, gates.shape[-1]), "minimax", top_k=2
        )

        # Every corrected token must include expert 3 in its real selection.
        assert np.all(np.any(captured == 3, axis=-1))
        # The naive reconstruction ignores the correction bias, so it disagrees.
        assert not np.array_equal(np.sort(captured, axis=-1),
                                  np.sort(reconstructed, axis=-1))

    def test_glm4_selection_matches_the_gate(self, tiny_glm4_moe, sample_input):
        block = tiny_glm4_moe
        real_inds, _ = block.gate(sample_input)
        mx.eval(real_inds)
        real_np = np.array(real_inds)

        install_ream_hooks([block], "glm4_moe")
        block(sample_input)
        _, _, sel = collect_ream_data([block])[0][0]
        remove_ream_hooks([block])

        np.testing.assert_array_equal(
            np.sort(sel, axis=-1), np.sort(real_np, axis=-1)
        )
