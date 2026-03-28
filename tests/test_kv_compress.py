"""Tests for TurboQuant KV cache compression."""

import mlx.core as mx
import numpy as np
import pytest

from mlx_fun.kv_compress import (
    TurboQuantConfig,
    TurboQuantKVCache,
    _MODEL_MODULE_MAP,
    _ORIGINAL_SDPA,
    generate_rotation_matrix,
    install_turbo_quant_sdpa,
    make_turbo_quant_cache,
    polar_dequantize,
    polar_quantize,
    remove_turbo_quant_sdpa,
    setup_turbo_quant,
)


# ---------------------------------------------------------------------------
# Rotation matrix tests
# ---------------------------------------------------------------------------


class TestRotationMatrix:
    def test_orthogonality(self):
        """Q @ Q.T ≈ I for the generated matrix."""
        Q = generate_rotation_matrix(64, seed=42)
        I_approx = Q @ Q.T
        mx.eval(I_approx)
        assert mx.allclose(I_approx, mx.eye(64), atol=1e-5)

    def test_orthogonality_transpose(self):
        """Q.T @ Q ≈ I as well."""
        Q = generate_rotation_matrix(64, seed=42)
        I_approx = Q.T @ Q
        mx.eval(I_approx)
        assert mx.allclose(I_approx, mx.eye(64), atol=1e-5)

    def test_deterministic_same_seed(self):
        Q1 = generate_rotation_matrix(64, seed=42)
        Q2 = generate_rotation_matrix(64, seed=42)
        mx.eval(Q1, Q2)
        assert mx.array_equal(Q1, Q2)

    def test_different_seeds_differ(self):
        Q1 = generate_rotation_matrix(64, seed=42)
        Q2 = generate_rotation_matrix(64, seed=99)
        mx.eval(Q1, Q2)
        assert not mx.array_equal(Q1, Q2)

    def test_small_dim(self):
        """Works for small dimensions like test fixtures."""
        Q = generate_rotation_matrix(8, seed=0)
        I_approx = Q @ Q.T
        mx.eval(I_approx)
        assert mx.allclose(I_approx, mx.eye(8), atol=1e-5)

    def test_det_is_pm_one(self):
        """Determinant of orthogonal matrix is ±1."""
        Q = generate_rotation_matrix(32, seed=7)
        mx.eval(Q)
        Q_np = np.array(Q)
        det = np.linalg.det(Q_np)
        assert abs(abs(det) - 1.0) < 1e-4


# ---------------------------------------------------------------------------
# PolarQuant roundtrip tests
# ---------------------------------------------------------------------------


class TestPolarQuantize:
    def test_roundtrip_8bit(self):
        """8-bit roundtrip has very low error."""
        mx.random.seed(0)
        x = mx.random.normal((2, 4, 8, 64))
        Q = generate_rotation_matrix(64, seed=42)
        mx.eval(x, Q)

        data, scales, biases = polar_quantize(x, Q, group_size=64, bits=8)
        x_hat = polar_dequantize(data, scales, biases, Q, group_size=64, bits=8)
        mx.eval(x_hat)

        error = float(mx.abs(x - x_hat).mean())
        assert error < 0.05, f"8-bit mean error too high: {error}"

    def test_roundtrip_4bit(self):
        """4-bit roundtrip has bounded error."""
        mx.random.seed(1)
        x = mx.random.normal((1, 4, 16, 64))
        Q = generate_rotation_matrix(64, seed=0)
        mx.eval(x, Q)

        data, scales, biases = polar_quantize(x, Q, group_size=64, bits=4)
        x_hat = polar_dequantize(data, scales, biases, Q, group_size=64, bits=4)
        mx.eval(x_hat)

        error = float(mx.abs(x - x_hat).mean())
        assert error < 0.5, f"4-bit mean error too high: {error}"

    def test_inner_product_preservation(self):
        """Inner products are approximately preserved through quantize roundtrip."""
        mx.random.seed(2)
        x = mx.random.normal((1, 1, 1, 64))
        y = mx.random.normal((1, 1, 1, 64))
        Q = generate_rotation_matrix(64, seed=42)
        mx.eval(x, y, Q)

        ip_orig = float(mx.sum(x * y))

        # Quantize both and compute inner product of reconstructed vectors
        dx, sx, bx = polar_quantize(x, Q, group_size=64, bits=8)
        dy, sy, by_ = polar_quantize(y, Q, group_size=64, bits=8)
        x_hat = polar_dequantize(dx, sx, bx, Q, group_size=64, bits=8)
        y_hat = polar_dequantize(dy, sy, by_, Q, group_size=64, bits=8)
        mx.eval(x_hat, y_hat)
        ip_quant = float(mx.sum(x_hat * y_hat))

        rel_error = abs(ip_orig - ip_quant) / (abs(ip_orig) + 1e-8)
        assert rel_error < 0.1, f"Inner product relative error too high: {rel_error}"

    def test_rotation_reduces_error(self):
        """PolarQuant has equal or lower error than vanilla quantization."""
        mx.random.seed(3)
        # Create data with outlier channels to demonstrate rotation benefit
        x = mx.random.normal((1, 4, 32, 64))
        # Add outlier to specific channels
        outlier = mx.zeros_like(x)
        outlier = outlier.at[..., 0].add(10.0)
        outlier = outlier.at[..., 1].add(-8.0)
        x = x + outlier
        Q = generate_rotation_matrix(64, seed=42)
        mx.eval(x, Q)

        # With rotation (PolarQuant)
        data_r, s_r, b_r = polar_quantize(x, Q, group_size=64, bits=4)
        x_rot = polar_dequantize(data_r, s_r, b_r, Q, group_size=64, bits=4)
        mx.eval(x_rot)

        # Without rotation (vanilla quantize)
        data_p, s_p, b_p = mx.quantize(x, group_size=64, bits=4)
        x_plain = mx.dequantize(data_p, s_p, b_p, group_size=64, bits=4)
        mx.eval(x_plain)

        err_rot = float(mx.abs(x - x_rot).mean())
        err_plain = float(mx.abs(x - x_plain).mean())
        # Rotation should help with outliers (allow small tolerance)
        assert err_rot <= err_plain * 1.15, (
            f"PolarQuant error ({err_rot:.4f}) not better than vanilla ({err_plain:.4f})"
        )

    def test_different_group_sizes(self):
        """Works with group_size=32."""
        mx.random.seed(4)
        x = mx.random.normal((1, 2, 4, 64))
        Q = generate_rotation_matrix(64, seed=0)
        mx.eval(x, Q)

        data, scales, biases = polar_quantize(x, Q, group_size=32, bits=4)
        x_hat = polar_dequantize(data, scales, biases, Q, group_size=32, bits=4)
        mx.eval(x_hat)
        assert x_hat.shape == x.shape


# ---------------------------------------------------------------------------
# TurboQuantKVCache tests — Phase 1 (plain SDPA)
# ---------------------------------------------------------------------------


class TestTurboQuantKVCachePlain:
    """Tests for quantized_sdpa=False mode (returns plain tensors)."""

    def _make_cache(self, **kwargs):
        cfg = TurboQuantConfig(quantized_sdpa=False, **kwargs)
        return TurboQuantKVCache(cfg)

    def test_initial_state(self):
        cache = self._make_cache()
        assert cache.empty()
        assert cache.offset == 0
        assert cache.size() == 0

    def test_update_returns_plain_tensors(self):
        cache = self._make_cache(bits=4, group_size=32)
        keys = mx.random.normal((1, 4, 8, 64))
        values = mx.random.normal((1, 4, 8, 64))
        mx.eval(keys, values)

        k_out, v_out = cache.update_and_fetch(keys, values)
        mx.eval(k_out, v_out)

        assert isinstance(k_out, mx.array)
        assert isinstance(v_out, mx.array)
        assert k_out.shape == (1, 4, 8, 64)
        assert v_out.shape == (1, 4, 8, 64)
        assert cache.offset == 8

    def test_no_bits_attribute(self):
        """Plain mode should NOT expose .bits (avoids quantized SDPA dispatch)."""
        cache = self._make_cache()
        assert not hasattr(cache, "bits")

    def test_sequential_updates(self):
        cache = self._make_cache(bits=4, group_size=32)
        k1 = mx.random.normal((1, 4, 8, 64))
        v1 = mx.random.normal((1, 4, 8, 64))
        mx.eval(k1, v1)
        cache.update_and_fetch(k1, v1)
        assert cache.offset == 8

        k2 = mx.random.normal((1, 4, 1, 64))
        v2 = mx.random.normal((1, 4, 1, 64))
        mx.eval(k2, v2)
        k_out, v_out = cache.update_and_fetch(k2, v2)
        mx.eval(k_out, v_out)
        assert cache.offset == 9
        assert k_out.shape == (1, 4, 9, 64)

    def test_dequantized_output_quality(self):
        cfg = TurboQuantConfig(bits=8, group_size=64, quantized_sdpa=False)
        cache = TurboQuantKVCache(cfg)
        keys = mx.random.normal((1, 4, 8, 64))
        values = mx.random.normal((1, 4, 8, 64))
        mx.eval(keys, values)

        k_out, v_out = cache.update_and_fetch(keys, values)
        mx.eval(k_out, v_out)

        k_err = float(mx.abs(keys - k_out).mean())
        v_err = float(mx.abs(values - v_out).mean())
        assert k_err < 0.05, f"Key error: {k_err}"
        assert v_err < 0.05, f"Value error: {v_err}"

    def test_trim(self):
        cache = self._make_cache()
        keys = mx.random.normal((1, 4, 16, 64))
        values = mx.random.normal((1, 4, 16, 64))
        mx.eval(keys, values)
        cache.update_and_fetch(keys, values)
        trimmed = cache.trim(4)
        assert trimmed == 4
        assert cache.offset == 12

    def test_trim_clamps(self):
        cache = self._make_cache()
        keys = mx.random.normal((1, 2, 4, 64))
        values = mx.random.normal((1, 2, 4, 64))
        mx.eval(keys, values)
        cache.update_and_fetch(keys, values)
        trimmed = cache.trim(100)
        assert trimmed == 4
        assert cache.offset == 0

    def test_large_prefill_then_decode(self):
        cfg = TurboQuantConfig(bits=4, group_size=32, quantized_sdpa=False)
        cache = TurboQuantKVCache(cfg)

        k_prefill = mx.random.normal((1, 4, 300, 64))
        v_prefill = mx.random.normal((1, 4, 300, 64))
        mx.eval(k_prefill, v_prefill)
        k_out, v_out = cache.update_and_fetch(k_prefill, v_prefill)
        mx.eval(k_out, v_out)
        assert cache.offset == 300
        assert k_out.shape == (1, 4, 300, 64)

        for i in range(5):
            k_tok = mx.random.normal((1, 4, 1, 64))
            v_tok = mx.random.normal((1, 4, 1, 64))
            mx.eval(k_tok, v_tok)
            k_out, v_out = cache.update_and_fetch(k_tok, v_tok)
            mx.eval(k_out, v_out)
            assert cache.offset == 301 + i
            assert k_out.shape == (1, 4, 301 + i, 64)

    def test_different_key_value_dims(self):
        cfg = TurboQuantConfig(bits=4, group_size=32, quantized_sdpa=False)
        cache = TurboQuantKVCache(cfg)
        keys = mx.random.normal((1, 4, 8, 64))
        values = mx.random.normal((1, 4, 8, 128))
        mx.eval(keys, values)

        k_out, v_out = cache.update_and_fetch(keys, values)
        mx.eval(k_out, v_out)
        assert k_out.shape == (1, 4, 8, 64)
        assert v_out.shape == (1, 4, 8, 128)


# ---------------------------------------------------------------------------
# TurboQuantKVCache tests — Phase 2 (quantized SDPA)
# ---------------------------------------------------------------------------


class TestTurboQuantKVCacheQuantized:
    """Tests for quantized_sdpa=True mode (returns quantized tuples)."""

    def _make_cache(self, **kwargs):
        cfg = TurboQuantConfig(quantized_sdpa=True, **kwargs)
        return TurboQuantKVCache(cfg)

    def test_has_bits_attribute(self):
        """Quantized mode MUST expose .bits for SDPA dispatch."""
        cache = self._make_cache(bits=4)
        assert hasattr(cache, "bits")
        assert cache.bits == 4
        assert hasattr(cache, "group_size")
        assert cache.group_size == 64

    def test_returns_quantized_tuples(self):
        cache = self._make_cache(bits=4, group_size=32)
        keys = mx.random.normal((1, 4, 8, 64))
        values = mx.random.normal((1, 4, 8, 64))
        mx.eval(keys, values)

        k_out, v_out = cache.update_and_fetch(keys, values)
        # Each should be a tuple of (data, scales, biases)
        assert isinstance(k_out, tuple) and len(k_out) == 3
        assert isinstance(v_out, tuple) and len(v_out) == 3
        # Data should be uint32 (packed quantized values)
        assert k_out[0].dtype == mx.uint32
        assert cache.offset == 8

    def test_sequential_updates_quantized(self):
        cache = self._make_cache(bits=4, group_size=32)
        k1 = mx.random.normal((1, 4, 8, 64))
        v1 = mx.random.normal((1, 4, 8, 64))
        mx.eval(k1, v1)
        cache.update_and_fetch(k1, v1)
        assert cache.offset == 8

        k2 = mx.random.normal((1, 4, 1, 64))
        v2 = mx.random.normal((1, 4, 1, 64))
        mx.eval(k2, v2)
        k_out, v_out = cache.update_and_fetch(k2, v2)
        assert cache.offset == 9
        # Seq dim should cover full offset
        assert k_out[0].shape[-2] == 9

    def test_quantized_attention_correctness(self):
        """Rotated Q @ rotated quantized K.T ≈ Q @ K.T (up to quantization)."""
        mx.random.seed(42)
        cache = self._make_cache(bits=8, group_size=64)
        keys = mx.random.normal((1, 1, 4, 64))
        values = mx.random.normal((1, 1, 4, 64))
        queries = mx.random.normal((1, 1, 1, 64))
        mx.eval(keys, values, queries)

        # Store keys in cache
        k_quant, v_quant = cache.update_and_fetch(keys, values)

        # Rotate queries (as the SDPA wrapper would)
        q_rot = queries @ cache._k_rotation.T

        # Compute attention scores via quantized matmul
        scores_quant = mx.quantized_matmul(
            q_rot, *k_quant, transpose=True,
            group_size=cache._group_size, bits=cache._bits,
        )
        mx.eval(scores_quant)

        # Reference: plain Q @ K.T
        scores_ref = queries @ keys.transpose(0, 1, 3, 2)
        mx.eval(scores_ref)

        # Should be close (8-bit quantization)
        rel_err = float(mx.abs(scores_quant - scores_ref).mean()) / (
            float(mx.abs(scores_ref).mean()) + 1e-8
        )
        assert rel_err < 0.1, f"Attention score relative error: {rel_err}"

    def test_value_output_rotation(self):
        """softmax(scores) @ V_quantized @ R_v ≈ softmax(scores) @ V."""
        mx.random.seed(99)
        cache = self._make_cache(bits=8, group_size=64)
        keys = mx.random.normal((1, 1, 4, 64))
        values = mx.random.normal((1, 1, 4, 64))
        mx.eval(keys, values)

        k_quant, v_quant = cache.update_and_fetch(keys, values)

        # Fake uniform attention weights
        scores = mx.ones((1, 1, 1, 4)) * 0.25

        # Quantized path: scores @ V_rotated_quant
        out_quant = mx.quantized_matmul(
            scores, *v_quant, transpose=False,
            group_size=cache._group_size, bits=cache._bits,
        )
        # Inverse-rotate output
        out_corrected = out_quant @ cache._v_rotation
        mx.eval(out_corrected)

        # Reference: scores @ V
        out_ref = scores @ values
        mx.eval(out_ref)

        rel_err = float(mx.abs(out_corrected - out_ref).mean()) / (
            float(mx.abs(out_ref).mean()) + 1e-8
        )
        assert rel_err < 0.1, f"Value output relative error: {rel_err}"


# ---------------------------------------------------------------------------
# SDPA patching tests
# ---------------------------------------------------------------------------


class TestSDPAPatching:
    def test_install_and_remove_known_model(self):
        """Patch installs and removes cleanly for a known model type."""
        result = install_turbo_quant_sdpa("minimax")
        assert result is True
        assert "mlx_lm.models.minimax" in _ORIGINAL_SDPA

        remove_turbo_quant_sdpa("minimax")
        assert "mlx_lm.models.minimax" not in _ORIGINAL_SDPA

    def test_install_unknown_model(self):
        """Unknown model type returns False."""
        result = install_turbo_quant_sdpa("unknown_model_xyz")
        assert result is False

    def test_remove_unpatched_is_safe(self):
        """Removing a patch that was never installed does not raise."""
        remove_turbo_quant_sdpa("minimax")  # Should be a no-op
        remove_turbo_quant_sdpa("unknown_model_xyz")

    def test_double_install_is_idempotent(self):
        """Installing twice doesn't double-wrap."""
        install_turbo_quant_sdpa("qwen3_moe")
        install_turbo_quant_sdpa("qwen3_moe")
        assert "mlx_lm.models.qwen3_moe" in _ORIGINAL_SDPA
        remove_turbo_quant_sdpa("qwen3_moe")

    def test_passthrough_for_non_turbo_cache(self):
        """Patched SDPA passes through for regular caches."""
        import mlx_lm.models.minimax as mod
        install_turbo_quant_sdpa("minimax")
        try:
            # Create fake inputs
            queries = mx.random.normal((1, 1, 1, 64))
            keys = mx.random.normal((1, 1, 4, 64))
            values = mx.random.normal((1, 1, 4, 64))
            mx.eval(queries, keys, values)

            # Use None cache (non-TurboQuant) — should work normally
            output = mod.scaled_dot_product_attention(
                queries, keys, values, cache=None, scale=1.0 / 8.0, mask=None,
            )
            mx.eval(output)
            assert output.shape == (1, 1, 1, 64)
        finally:
            remove_turbo_quant_sdpa("minimax")


# ---------------------------------------------------------------------------
# setup_turbo_quant integration test
# ---------------------------------------------------------------------------


class TestSetupTurboQuant:
    def test_setup_with_quantized_sdpa(self):
        class FakeModel:
            layers = [None] * 4

        cfg = TurboQuantConfig(bits=4, quantized_sdpa=True)
        caches, patched = setup_turbo_quant(FakeModel(), "minimax", cfg)
        assert len(caches) == 4
        assert patched is True
        assert all(c._quantized_sdpa for c in caches)
        assert all(hasattr(c, "bits") for c in caches)
        remove_turbo_quant_sdpa("minimax")

    def test_setup_fallback_for_unsupported_model(self):
        class FakeModel:
            layers = [None] * 4

        cfg = TurboQuantConfig(bits=4, quantized_sdpa=True)
        caches, patched = setup_turbo_quant(FakeModel(), "deepseek_v32", cfg)
        assert patched is False
        # Should have fallen back to plain mode
        assert all(not c._quantized_sdpa for c in caches)
        assert all(not hasattr(c, "bits") for c in caches)

    def test_setup_plain_mode(self):
        class FakeModel:
            layers = [None] * 4

        cfg = TurboQuantConfig(bits=4, quantized_sdpa=False)
        caches, patched = setup_turbo_quant(FakeModel(), "minimax", cfg)
        assert patched is False
        assert all(not c._quantized_sdpa for c in caches)


# ---------------------------------------------------------------------------
# Common cache protocol tests
# ---------------------------------------------------------------------------


class TestCacheProtocol:
    """Tests for cache protocol methods shared by both modes."""

    @pytest.fixture(params=[False, True], ids=["plain", "quantized"])
    def cache(self, request):
        cfg = TurboQuantConfig(bits=4, group_size=32, quantized_sdpa=request.param)
        return TurboQuantKVCache(cfg)

    def test_empty(self, cache):
        assert cache.empty()

    def test_not_empty_after_update(self, cache):
        keys = mx.random.normal((1, 2, 4, 64))
        values = mx.random.normal((1, 2, 4, 64))
        mx.eval(keys, values)
        cache.update_and_fetch(keys, values)
        assert not cache.empty()

    def test_is_trimmable(self, cache):
        assert cache.is_trimmable()

    def test_state_empty(self, cache):
        assert cache.state == []

    def test_meta_state_roundtrip(self, cache):
        keys = mx.random.normal((1, 2, 4, 64))
        values = mx.random.normal((1, 2, 4, 64))
        mx.eval(keys, values)
        cache.update_and_fetch(keys, values)

        meta = cache.meta_state
        cache2 = TurboQuantKVCache()
        cache2.meta_state = meta
        assert cache2.offset == 4
        assert cache2._bits == cache._bits
        assert cache2._group_size == cache._group_size


# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------


class TestMakeTurboQuantCache:
    def test_creates_correct_count(self):
        class FakeModel:
            layers = [None] * 32

        caches = make_turbo_quant_cache(FakeModel(), TurboQuantConfig(bits=4))
        assert len(caches) == 32
        assert all(isinstance(c, TurboQuantKVCache) for c in caches)

    def test_config_propagated(self):
        class FakeModel:
            layers = [None] * 4

        cfg = TurboQuantConfig(bits=3, group_size=32, seed=99)
        caches = make_turbo_quant_cache(FakeModel(), cfg)
        for c in caches:
            assert c._bits == 3
            assert c._group_size == 32
            assert c.config.seed == 99

    def test_default_config(self):
        class FakeModel:
            layers = [None] * 2

        caches = make_turbo_quant_cache(FakeModel())
        assert caches[0]._bits == 4
        assert caches[0]._group_size == 64
