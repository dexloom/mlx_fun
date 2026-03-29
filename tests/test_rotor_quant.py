"""Tests for RotorQuant KV cache compression."""

import mlx.core as mx
import pytest

from mlx_fun.rotor_quant import (
    MV_DIM,
    RotorQuantConfig,
    RotorQuantKVCache,
    _embed_vectors,
    _extract_vectors,
    _geometric_product,
    _make_random_rotor,
    _reverse,
    _rotor_sandwich,
    _solve_lloyd_max,
    make_rotor_quant_cache,
    setup_rotor_quant,
)


# ---------------------------------------------------------------------------
# Clifford algebra tests
# ---------------------------------------------------------------------------


class TestCliffordAlgebra:
    def test_rotor_normalized(self):
        """R R̃ should have scalar part ≈ 1."""
        R = _make_random_rotor(seed=42)
        rev = _reverse(R)
        product = _geometric_product(R, rev)
        mx.eval(product)
        assert abs(product[0].item() - 1.0) < 1e-4

    def test_rotor_deterministic(self):
        R1 = _make_random_rotor(seed=42)
        R2 = _make_random_rotor(seed=42)
        mx.eval(R1, R2)
        assert mx.allclose(R1, R2, atol=1e-5)

    def test_rotor_different_seeds(self):
        R1 = _make_random_rotor(seed=42)
        R2 = _make_random_rotor(seed=99)
        mx.eval(R1, R2)
        assert not mx.array_equal(R1, R2)

    def test_sandwich_preserves_norm(self):
        """Rotor sandwich R v R̃ should preserve vector norm."""
        R = _make_random_rotor(seed=0)
        # Create a grade-1 multivector
        v = mx.zeros((8,))
        v = v.at[1].add(mx.array(3.0))
        v = v.at[2].add(mx.array(4.0))
        v = v.at[3].add(mx.array(0.0))
        mx.eval(v)

        rotated = _rotor_sandwich(R, v)
        mx.eval(rotated)
        # Grade-1 norm: sqrt(e1^2 + e2^2 + e3^2)
        orig_norm = mx.sqrt(v[1]**2 + v[2]**2 + v[3]**2)
        rot_norm = mx.sqrt(rotated[1]**2 + rotated[2]**2 + rotated[3]**2)
        mx.eval(orig_norm, rot_norm)
        assert abs(orig_norm.item() - rot_norm.item()) < 0.1

    def test_embed_extract_roundtrip(self):
        """embed → extract should recover original vectors."""
        v = mx.random.normal((4, 64))
        mx.eval(v)
        mv = _embed_vectors(v)
        v_out = _extract_vectors(mv, 64)
        mx.eval(v_out)
        assert mx.allclose(v, v_out, atol=1e-5)

    def test_embed_extract_non_divisible(self):
        """Works for dims not divisible by 3."""
        v = mx.random.normal((2, 65))
        mx.eval(v)
        mv = _embed_vectors(v)
        v_out = _extract_vectors(mv, 65)
        mx.eval(v_out)
        assert mx.allclose(v, v_out, atol=1e-5)

    def test_geometric_product_shape(self):
        a = mx.random.normal((3, 8))
        b = mx.random.normal((3, 8))
        result = _geometric_product(a, b)
        mx.eval(result)
        assert result.shape == (3, 8)


# ---------------------------------------------------------------------------
# Lloyd-Max codebook tests
# ---------------------------------------------------------------------------


class TestLloydMax:
    def test_correct_levels(self):
        centroids = _solve_lloyd_max(64, bits=3)
        mx.eval(centroids)
        assert centroids.shape == (8,)  # 2^3 = 8

    def test_sorted(self):
        centroids = _solve_lloyd_max(128, bits=3)
        mx.eval(centroids)
        c = centroids.tolist()
        assert c == sorted(c)

    def test_symmetric(self):
        """Centroids should be approximately symmetric around 0."""
        centroids = _solve_lloyd_max(128, bits=3)
        mx.eval(centroids)
        c = centroids.tolist()
        assert abs(c[0] + c[-1]) < 0.01

    def test_2bit(self):
        centroids = _solve_lloyd_max(64, bits=2)
        mx.eval(centroids)
        assert centroids.shape == (4,)

    def test_4bit(self):
        centroids = _solve_lloyd_max(64, bits=4)
        mx.eval(centroids)
        assert centroids.shape == (16,)


# ---------------------------------------------------------------------------
# RotorQuantKVCache tests
# ---------------------------------------------------------------------------


class TestRotorQuantKVCache:
    def _make_cache(self, bits=3, max_size=None, keep=4):
        cfg = RotorQuantConfig(bits=bits, max_size=max_size, keep=keep)
        return RotorQuantKVCache(config=cfg)

    def _random_kv(self, B=1, n_heads=2, steps=1, head_dim=64):
        keys = mx.random.normal((B, n_heads, steps, head_dim))
        values = mx.random.normal((B, n_heads, steps, head_dim))
        mx.eval(keys, values)
        return keys, values

    def test_initial_state(self):
        cache = self._make_cache()
        assert cache.offset == 0
        assert cache.empty()
        assert cache.size() == 0

    def test_update_returns_tensors(self):
        cache = self._make_cache()
        k, v = self._random_kv()
        k_out, v_out = cache.update_and_fetch(k, v)
        mx.eval(k_out, v_out)
        assert k_out.shape == (1, 2, 1, 64)
        assert v_out.shape == (1, 2, 1, 64)

    def test_sequential_updates(self):
        cache = self._make_cache()
        for i in range(5):
            k, v = self._random_kv()
            k_out, v_out = cache.update_and_fetch(k, v)
            mx.eval(k_out, v_out)
        assert cache.offset == 5
        assert k_out.shape == (1, 2, 5, 64)

    def test_not_empty_after_update(self):
        cache = self._make_cache()
        k, v = self._random_kv()
        cache.update_and_fetch(k, v)
        assert not cache.empty()
        assert cache.size() == 1

    def test_reconstruction_quality(self):
        """Dequantized output should be close to original."""
        cache = self._make_cache(bits=4)  # higher bits = better quality
        k, v = self._random_kv(steps=4)
        k_out, v_out = cache.update_and_fetch(k, v)
        mx.eval(k_out, v_out)
        # Cosine similarity should be high
        k_flat = k.reshape(-1)
        k_out_flat = k_out.reshape(-1)
        cos_sim = (mx.sum(k_flat * k_out_flat)
                   / (mx.sqrt(mx.sum(k_flat**2)) * mx.sqrt(mx.sum(k_out_flat**2))))
        mx.eval(cos_sim)
        assert cos_sim.item() > 0.85

    def test_batch_prefill(self):
        """Multi-token insert works."""
        cache = self._make_cache()
        k, v = self._random_kv(steps=10)
        k_out, v_out = cache.update_and_fetch(k, v)
        mx.eval(k_out, v_out)
        assert cache.offset == 10
        assert k_out.shape == (1, 2, 10, 64)

    def test_trim(self):
        cache = self._make_cache()
        k, v = self._random_kv(steps=5)
        cache.update_and_fetch(k, v)
        trimmed = cache.trim(3)
        assert trimmed == 3
        assert cache.offset == 2

    def test_is_trimmable(self):
        cache = self._make_cache()
        assert cache.is_trimmable()

    def test_different_key_value_dims(self):
        """Keys and values can have different head dimensions."""
        cache = self._make_cache()
        keys = mx.random.normal((1, 2, 3, 64))
        values = mx.random.normal((1, 2, 3, 128))
        mx.eval(keys, values)
        k_out, v_out = cache.update_and_fetch(keys, values)
        mx.eval(k_out, v_out)
        assert k_out.shape == (1, 2, 3, 64)
        assert v_out.shape == (1, 2, 3, 128)


# ---------------------------------------------------------------------------
# Sliding window tests
# ---------------------------------------------------------------------------


class TestRotorQuantSlidingWindow:
    def _make_cache(self, max_size, keep=4, bits=3):
        cfg = RotorQuantConfig(bits=bits, max_size=max_size, keep=keep)
        return RotorQuantKVCache(config=cfg)

    def _random_kv(self, B=1, n_heads=2, steps=1, head_dim=64):
        keys = mx.random.normal((B, n_heads, steps, head_dim))
        values = mx.random.normal((B, n_heads, steps, head_dim))
        mx.eval(keys, values)
        return keys, values

    def test_no_trim_under_limit(self):
        cache = self._make_cache(max_size=32)
        for _ in range(16):
            k, v = self._random_kv()
            cache.update_and_fetch(k, v)
        assert cache.offset == 16

    def test_trim_at_limit(self):
        cache = self._make_cache(max_size=16)
        for _ in range(20):
            k, v = self._random_kv()
            cache.update_and_fetch(k, v)
        assert cache.offset == 16

    def test_repeated_trims(self):
        cache = self._make_cache(max_size=8, keep=2)
        for _ in range(50):
            k, v = self._random_kv()
            cache.update_and_fetch(k, v)
        assert cache.offset == 8

    def test_no_window_when_none(self):
        cfg = RotorQuantConfig(bits=3)
        cache = RotorQuantKVCache(config=cfg)
        for _ in range(30):
            k, v = self._random_kv()
            cache.update_and_fetch(k, v)
        assert cache.offset == 30


# ---------------------------------------------------------------------------
# Meta state tests
# ---------------------------------------------------------------------------


class TestRotorQuantMetaState:
    def test_roundtrip(self):
        cfg = RotorQuantConfig(bits=3, max_size=1024, keep=8)
        cache = RotorQuantKVCache(config=cfg)
        # Need to init head dims
        k = mx.random.normal((1, 2, 1, 64))
        v = mx.random.normal((1, 2, 1, 128))
        mx.eval(k, v)
        cache.update_and_fetch(k, v)

        state = cache.meta_state
        cache2 = RotorQuantKVCache()
        cache2.meta_state = state
        assert cache2._bits == 3
        assert cache2._max_size == 1024
        assert cache2._keep == 8
        assert cache2._k_head_dim == 64
        assert cache2._v_head_dim == 128

    def test_none_max_size(self):
        cfg = RotorQuantConfig(bits=3)
        cache = RotorQuantKVCache(config=cfg)
        state = cache.meta_state
        cache2 = RotorQuantKVCache()
        cache2.meta_state = state
        assert cache2._max_size is None


# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------


class TestFactory:
    def test_make_cache_count(self):
        class FakeModel:
            layers = [None] * 16

        caches = make_rotor_quant_cache(FakeModel(), RotorQuantConfig(bits=3))
        assert len(caches) == 16
        assert all(isinstance(c, RotorQuantKVCache) for c in caches)

    def test_setup_returns_list(self):
        class FakeModel:
            layers = [None] * 4

        caches = setup_rotor_quant(FakeModel(), RotorQuantConfig(bits=3))
        assert len(caches) == 4

    def test_config_propagated(self):
        class FakeModel:
            layers = [None] * 2

        cfg = RotorQuantConfig(bits=4, max_size=512, keep=8)
        caches = make_rotor_quant_cache(FakeModel(), cfg)
        assert caches[0]._bits == 4
        assert caches[0]._max_size == 512
        assert caches[0]._keep == 8
