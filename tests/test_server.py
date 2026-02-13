"""Tests for the REAP server module."""

import io
import json
import threading
import tempfile

import mlx.core as mx
import numpy as np
import pytest

from mlx_fun.server import (
    OnlineAccumulator,
    install_counting_hooks,
    remove_counting_hooks,
    ReapAPIHandler,
)
from mlx_fun.saliency import SaliencyAccumulator


# ---------------------------------------------------------------------------
# OnlineAccumulator tests
# ---------------------------------------------------------------------------

class TestOnlineAccumulator:
    def test_basic_update(self):
        acc = OnlineAccumulator(num_layers=2, num_experts=4)
        inds = np.array([[0, 1], [2, 3]])
        weights = np.array([[0.6, 0.4], [0.7, 0.3]])
        acc.update(0, inds, weights)

        stats = acc.get_stats()
        assert stats["num_layers"] == 2
        assert stats["num_experts"] == 4
        # freq for layer 0: experts 0,1,2,3 each selected once
        assert stats["freq"][0][0] == 1.0
        assert stats["freq"][0][1] == 1.0
        assert stats["freq"][0][2] == 1.0
        assert stats["freq"][0][3] == 1.0
        # Layer 1 untouched
        assert stats["freq"][1] == [0.0, 0.0, 0.0, 0.0]

    def test_update_with_norms(self):
        acc = OnlineAccumulator(num_layers=1, num_experts=4)
        inds = np.array([[0, 1]])
        weights = np.array([[0.6, 0.4]])
        norms = np.array([[1.5, 2.0]])
        acc.update(0, inds, weights, norms)

        stats = acc.get_stats()
        # reap_sum = norms * weights: [0.9, 0.8]
        np.testing.assert_allclose(stats["reap_sum"][0][0], 0.9, atol=1e-10)
        np.testing.assert_allclose(stats["reap_sum"][0][1], 0.8, atol=1e-10)
        # ean_sum = norms: [1.5, 2.0]
        np.testing.assert_allclose(stats["ean_sum"][0][0], 1.5, atol=1e-10)
        np.testing.assert_allclose(stats["ean_sum"][0][1], 2.0, atol=1e-10)

    def test_update_lightweight_zeros_norms(self):
        """Without norms, reap_sum and ean_sum stay zero."""
        acc = OnlineAccumulator(num_layers=1, num_experts=4)
        inds = np.array([[0, 1]])
        weights = np.array([[0.6, 0.4]])
        acc.update(0, inds, weights)  # no norms

        stats = acc.get_stats()
        assert stats["reap_sum"][0] == [0.0, 0.0, 0.0, 0.0]
        assert stats["ean_sum"][0] == [0.0, 0.0, 0.0, 0.0]
        # But freq and weighted_freq are populated
        assert stats["freq"][0][0] == 1.0
        assert stats["freq"][0][1] == 1.0
        np.testing.assert_allclose(stats["weighted_freq_sum"][0][0], 0.6, atol=1e-10)
        np.testing.assert_allclose(stats["weighted_freq_sum"][0][1], 0.4, atol=1e-10)

    def test_thread_safety(self):
        """Concurrent updates from multiple threads produce correct totals."""
        acc = OnlineAccumulator(num_layers=1, num_experts=4)
        n_threads = 8
        n_updates = 100

        def worker():
            for _ in range(n_updates):
                inds = np.array([[0, 1]])
                weights = np.array([[1.0, 1.0]])
                acc.update(0, inds, weights)

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        stats = acc.get_stats()
        expected = n_threads * n_updates
        assert stats["freq"][0][0] == expected
        assert stats["freq"][0][1] == expected
        assert stats["freq"][0][2] == 0.0
        assert stats["freq"][0][3] == 0.0

    def test_reset(self):
        acc = OnlineAccumulator(num_layers=1, num_experts=4)
        inds = np.array([[0, 1]])
        weights = np.array([[0.5, 0.5]])
        acc.update(0, inds, weights)
        acc.increment_request()
        acc.add_tokens(10)

        acc.reset()
        stats = acc.get_stats()
        assert stats["freq"][0] == [0.0, 0.0, 0.0, 0.0]
        assert stats["request_count"] == 0
        assert stats["token_count"] == 0

    def test_request_and_token_counts(self):
        acc = OnlineAccumulator(num_layers=1, num_experts=4)
        acc.increment_request()
        acc.increment_request()
        acc.add_tokens(50)
        acc.add_tokens(30)

        stats = acc.get_stats()
        assert stats["request_count"] == 2
        assert stats["token_count"] == 80

    def test_save_load_compatibility(self, tmp_path):
        """Saved .npz is loadable by SaliencyAccumulator.load()."""
        acc = OnlineAccumulator(num_layers=2, num_experts=4)
        inds = np.array([[0, 2], [1, 3]])
        weights = np.array([[0.6, 0.4], [0.7, 0.3]])
        norms = np.array([[1.0, 2.0], [1.5, 0.5]])
        acc.update(0, inds, weights, norms)
        acc.update(1, inds, weights, norms)

        path = str(tmp_path / "test_saliency.npz")
        acc.save(path)

        loaded = SaliencyAccumulator.load(path)
        assert loaded.num_layers == 2
        assert loaded.num_experts == 4
        np.testing.assert_array_equal(loaded.freq[0], acc._acc.freq[0])
        np.testing.assert_array_equal(loaded.freq[1], acc._acc.freq[1])
        np.testing.assert_array_equal(loaded.reap_sum[0], acc._acc.reap_sum[0])
        np.testing.assert_array_equal(loaded.weighted_freq_sum[0], acc._acc.weighted_freq_sum[0])

        # Verify it can compute scores (as used by mlx-fun prune)
        for metric in ("reap", "ean", "freq", "weighted_freq"):
            scores = loaded.compute_scores(metric)
            assert scores.shape == (2, 4)


# ---------------------------------------------------------------------------
# Lightweight counting hook tests
# ---------------------------------------------------------------------------

class TestLightweightCountingHooks:
    def test_minimax_counting_hook(self, tiny_minimax_moe, sample_input):
        acc = OnlineAccumulator(num_layers=1, num_experts=4)
        install_counting_hooks([tiny_minimax_moe], "minimax", acc, mode="lightweight")

        out = tiny_minimax_moe(sample_input)
        mx.eval(out)

        stats = acc.get_stats()
        # 1 batch * 8 seq positions * 2 top_k = 16 total expert selections
        total_freq = sum(stats["freq"][0])
        assert total_freq == 16.0

        # reap_sum should be zero (lightweight mode)
        assert all(v == 0.0 for v in stats["reap_sum"][0])

        remove_counting_hooks([tiny_minimax_moe])

    def test_glm4_counting_hook(self, tiny_glm4_moe, sample_input):
        acc = OnlineAccumulator(num_layers=1, num_experts=4)
        install_counting_hooks([tiny_glm4_moe], "glm4_moe", acc, mode="lightweight")

        out = tiny_glm4_moe(sample_input)
        mx.eval(out)

        stats = acc.get_stats()
        total_freq = sum(stats["freq"][0])
        assert total_freq == 16.0

        remove_counting_hooks([tiny_glm4_moe])

    def test_glm4_moe_lite_counting_hook(self, tiny_glm4_moe, sample_input):
        acc = OnlineAccumulator(num_layers=1, num_experts=4)
        install_counting_hooks([tiny_glm4_moe], "glm4_moe_lite", acc, mode="lightweight")

        out = tiny_glm4_moe(sample_input)
        mx.eval(out)

        stats = acc.get_stats()
        total_freq = sum(stats["freq"][0])
        assert total_freq == 16.0

        remove_counting_hooks([tiny_glm4_moe])

    def test_qwen3_counting_hook(self, tiny_qwen3_moe, sample_input):
        acc = OnlineAccumulator(num_layers=1, num_experts=4)
        install_counting_hooks([tiny_qwen3_moe], "qwen3_moe", acc, mode="lightweight")

        out = tiny_qwen3_moe(sample_input)
        mx.eval(out)

        stats = acc.get_stats()
        total_freq = sum(stats["freq"][0])
        assert total_freq == 16.0

        remove_counting_hooks([tiny_qwen3_moe])

    def test_qwen3_next_counting_hook(self, tiny_qwen3_next_moe, sample_input):
        acc = OnlineAccumulator(num_layers=1, num_experts=4)
        install_counting_hooks([tiny_qwen3_next_moe], "qwen3_next", acc, mode="lightweight")

        out = tiny_qwen3_next_moe(sample_input)
        mx.eval(out)

        stats = acc.get_stats()
        total_freq = sum(stats["freq"][0])
        assert total_freq == 16.0

        remove_counting_hooks([tiny_qwen3_next_moe])

    def test_minimax_m2_counting_hook(self, tiny_minimax_moe, sample_input):
        acc = OnlineAccumulator(num_layers=1, num_experts=4)
        install_counting_hooks([tiny_minimax_moe], "minimax_m2", acc, mode="lightweight")

        out = tiny_minimax_moe(sample_input)
        mx.eval(out)

        stats = acc.get_stats()
        total_freq = sum(stats["freq"][0])
        assert total_freq == 16.0

        remove_counting_hooks([tiny_minimax_moe])


# ---------------------------------------------------------------------------
# Full counting hook tests
# ---------------------------------------------------------------------------

class TestFullCountingHooks:
    def test_minimax_full_hook(self, tiny_minimax_moe, sample_input):
        acc = OnlineAccumulator(num_layers=1, num_experts=4)
        install_counting_hooks([tiny_minimax_moe], "minimax", acc, mode="full")

        out = tiny_minimax_moe(sample_input)
        mx.eval(out)

        stats = acc.get_stats()
        total_freq = sum(stats["freq"][0])
        assert total_freq == 16.0

        # Full mode should have non-zero reap_sum
        assert any(v > 0.0 for v in stats["reap_sum"][0])

        remove_counting_hooks([tiny_minimax_moe])

    def test_glm4_full_hook(self, tiny_glm4_moe, sample_input):
        acc = OnlineAccumulator(num_layers=1, num_experts=4)
        install_counting_hooks([tiny_glm4_moe], "glm4_moe", acc, mode="full")

        out = tiny_glm4_moe(sample_input)
        mx.eval(out)

        stats = acc.get_stats()
        total_freq = sum(stats["freq"][0])
        assert total_freq == 16.0
        assert any(v > 0.0 for v in stats["reap_sum"][0])

        remove_counting_hooks([tiny_glm4_moe])

    def test_qwen3_full_hook(self, tiny_qwen3_moe, sample_input):
        acc = OnlineAccumulator(num_layers=1, num_experts=4)
        install_counting_hooks([tiny_qwen3_moe], "qwen3_moe", acc, mode="full")

        out = tiny_qwen3_moe(sample_input)
        mx.eval(out)

        stats = acc.get_stats()
        total_freq = sum(stats["freq"][0])
        assert total_freq == 16.0
        assert any(v > 0.0 for v in stats["reap_sum"][0])

        remove_counting_hooks([tiny_qwen3_moe])

    def test_qwen3_next_full_hook(self, tiny_qwen3_next_moe, sample_input):
        acc = OnlineAccumulator(num_layers=1, num_experts=4)
        install_counting_hooks([tiny_qwen3_next_moe], "qwen3_next", acc, mode="full")

        out = tiny_qwen3_next_moe(sample_input)
        mx.eval(out)

        stats = acc.get_stats()
        total_freq = sum(stats["freq"][0])
        assert total_freq == 16.0
        assert any(v > 0.0 for v in stats["reap_sum"][0])

        remove_counting_hooks([tiny_qwen3_next_moe])

    def test_full_hook_matches_offline_captures(self, tiny_minimax_moe, sample_input):
        """Full counting hooks should produce same stats as offline collection."""
        from mlx_fun.observer import install_hooks, remove_hooks, collect_captures

        # --- Offline approach ---
        offline_acc = SaliencyAccumulator(num_layers=1, num_experts=4)
        install_hooks([tiny_minimax_moe], "minimax")

        mx.random.seed(42)
        tiny_minimax_moe(sample_input)
        mx.eval(tiny_minimax_moe.parameters())

        captures = collect_captures([tiny_minimax_moe])
        for inds, scores, norms in captures[0]:
            flat_inds = inds.reshape(-1, inds.shape[-1])
            flat_scores = scores.reshape(-1, scores.shape[-1])
            flat_norms = norms.reshape(-1, norms.shape[-1])
            offline_acc.update(0, flat_inds, flat_scores, flat_norms)
        remove_hooks([tiny_minimax_moe])

        # --- Online approach ---
        online_acc = OnlineAccumulator(num_layers=1, num_experts=4)
        install_counting_hooks([tiny_minimax_moe], "minimax", online_acc, mode="full")

        mx.random.seed(42)
        tiny_minimax_moe(sample_input)
        mx.eval(tiny_minimax_moe.parameters())

        remove_counting_hooks([tiny_minimax_moe])

        # Compare
        np.testing.assert_array_equal(offline_acc.freq, online_acc._acc.freq)
        np.testing.assert_allclose(
            offline_acc.weighted_freq_sum, online_acc._acc.weighted_freq_sum, atol=1e-5
        )
        np.testing.assert_allclose(
            offline_acc.reap_sum, online_acc._acc.reap_sum, atol=1e-5
        )
        np.testing.assert_allclose(
            offline_acc.ean_sum, online_acc._acc.ean_sum, atol=1e-5
        )


# ---------------------------------------------------------------------------
# Numerical equivalence: hooked output matches unhooked output
# ---------------------------------------------------------------------------

class TestCountingHookNumericalEquivalence:
    def test_minimax_lightweight(self, tiny_minimax_moe, sample_input):
        mx.random.seed(42)
        orig_out = tiny_minimax_moe(sample_input)
        mx.eval(orig_out)

        acc = OnlineAccumulator(num_layers=1, num_experts=4)
        install_counting_hooks([tiny_minimax_moe], "minimax", acc, mode="lightweight")
        mx.random.seed(42)
        hooked_out = tiny_minimax_moe(sample_input)
        mx.eval(hooked_out)
        remove_counting_hooks([tiny_minimax_moe])

        np.testing.assert_allclose(
            np.array(orig_out, copy=False),
            np.array(hooked_out, copy=False),
            atol=1e-5,
        )

    def test_minimax_full(self, tiny_minimax_moe, sample_input):
        mx.random.seed(42)
        orig_out = tiny_minimax_moe(sample_input)
        mx.eval(orig_out)

        acc = OnlineAccumulator(num_layers=1, num_experts=4)
        install_counting_hooks([tiny_minimax_moe], "minimax", acc, mode="full")
        mx.random.seed(42)
        hooked_out = tiny_minimax_moe(sample_input)
        mx.eval(hooked_out)
        remove_counting_hooks([tiny_minimax_moe])

        np.testing.assert_allclose(
            np.array(orig_out, copy=False),
            np.array(hooked_out, copy=False),
            atol=1e-5,
        )

    def test_glm4(self, tiny_glm4_moe, sample_input):
        mx.random.seed(42)
        orig_out = tiny_glm4_moe(sample_input)
        mx.eval(orig_out)

        acc = OnlineAccumulator(num_layers=1, num_experts=4)
        install_counting_hooks([tiny_glm4_moe], "glm4_moe", acc, mode="lightweight")
        mx.random.seed(42)
        hooked_out = tiny_glm4_moe(sample_input)
        mx.eval(hooked_out)
        remove_counting_hooks([tiny_glm4_moe])

        np.testing.assert_allclose(
            np.array(orig_out, copy=False),
            np.array(hooked_out, copy=False),
            atol=1e-5,
        )

    def test_qwen3(self, tiny_qwen3_moe, sample_input):
        mx.random.seed(42)
        orig_out = tiny_qwen3_moe(sample_input)
        mx.eval(orig_out)

        acc = OnlineAccumulator(num_layers=1, num_experts=4)
        install_counting_hooks([tiny_qwen3_moe], "qwen3_moe", acc, mode="lightweight")
        mx.random.seed(42)
        hooked_out = tiny_qwen3_moe(sample_input)
        mx.eval(hooked_out)
        remove_counting_hooks([tiny_qwen3_moe])

        np.testing.assert_allclose(
            np.array(orig_out, copy=False),
            np.array(hooked_out, copy=False),
            atol=1e-5,
        )

    def test_qwen3_next(self, tiny_qwen3_next_moe, sample_input):
        mx.random.seed(42)
        orig_out = tiny_qwen3_next_moe(sample_input)
        mx.eval(orig_out)

        acc = OnlineAccumulator(num_layers=1, num_experts=4)
        install_counting_hooks([tiny_qwen3_next_moe], "qwen3_next", acc, mode="lightweight")
        mx.random.seed(42)
        hooked_out = tiny_qwen3_next_moe(sample_input)
        mx.eval(hooked_out)
        remove_counting_hooks([tiny_qwen3_next_moe])

        np.testing.assert_allclose(
            np.array(orig_out, copy=False),
            np.array(hooked_out, copy=False),
            atol=1e-5,
        )


# ---------------------------------------------------------------------------
# Hook install / remove tests
# ---------------------------------------------------------------------------

class TestCountingHookInstallRemove:
    def test_install_adds_attributes(self, tiny_minimax_moe):
        acc = OnlineAccumulator(num_layers=1, num_experts=4)
        original_cls = type(tiny_minimax_moe)

        install_counting_hooks([tiny_minimax_moe], "minimax", acc)
        assert hasattr(tiny_minimax_moe, "_reap_accumulator")
        assert hasattr(tiny_minimax_moe, "_reap_layer_idx")
        assert tiny_minimax_moe._reap_layer_idx == 0
        assert type(tiny_minimax_moe) is not original_cls
        assert "_Counting_" in type(tiny_minimax_moe).__name__

        remove_counting_hooks([tiny_minimax_moe])

    def test_remove_restores_class(self, tiny_minimax_moe):
        acc = OnlineAccumulator(num_layers=1, num_experts=4)
        original_cls = type(tiny_minimax_moe)

        install_counting_hooks([tiny_minimax_moe], "minimax", acc)
        remove_counting_hooks([tiny_minimax_moe])

        assert type(tiny_minimax_moe) is original_cls
        assert not hasattr(tiny_minimax_moe, "_reap_accumulator")
        assert not hasattr(tiny_minimax_moe, "_reap_layer_idx")

    def test_unknown_model_type_raises(self):
        acc = OnlineAccumulator(num_layers=1, num_experts=4)
        with pytest.raises(ValueError, match="No counting hook"):
            install_counting_hooks([], "unknown_model", acc)

    def test_multiple_blocks_get_correct_layer_indices(self, tiny_minimax_moe):
        # Create a second block
        mx.random.seed(99)
        from tests.conftest import TinyMiniMaxMoE
        block2 = TinyMiniMaxMoE(hidden=32, intermediate=64, n_experts=4, top_k=2)
        mx.eval(block2.parameters())

        acc = OnlineAccumulator(num_layers=2, num_experts=4)
        blocks = [tiny_minimax_moe, block2]
        install_counting_hooks(blocks, "minimax", acc)

        assert tiny_minimax_moe._reap_layer_idx == 0
        assert block2._reap_layer_idx == 1

        remove_counting_hooks(blocks)


# ---------------------------------------------------------------------------
# Multiple forward passes accumulate
# ---------------------------------------------------------------------------

class TestMultipleForwardPasses:
    def test_accumulation_across_passes(self, tiny_minimax_moe, sample_input):
        acc = OnlineAccumulator(num_layers=1, num_experts=4)
        install_counting_hooks([tiny_minimax_moe], "minimax", acc, mode="lightweight")

        tiny_minimax_moe(sample_input)
        mx.eval(tiny_minimax_moe.parameters())
        tiny_minimax_moe(sample_input)
        mx.eval(tiny_minimax_moe.parameters())

        stats = acc.get_stats()
        total_freq = sum(stats["freq"][0])
        # 2 passes * 8 seq * 2 top_k = 32
        assert total_freq == 32.0

        remove_counting_hooks([tiny_minimax_moe])


# ---------------------------------------------------------------------------
# Management endpoint tests (mock HTTP)
# ---------------------------------------------------------------------------

class TestReapEndpoints:
    """Test the ReapAPIHandler endpoint logic using mock request/response."""

    def _make_handler(self, accumulator, method, path, body=None):
        """Create a handler instance and simulate a request."""
        from http.server import HTTPServer
        from unittest.mock import MagicMock

        handler_class = ReapAPIHandler.create_handler_class(accumulator)

        # Create a mock request
        handler = object.__new__(handler_class)
        handler._reap_accumulator = accumulator
        handler.path = path
        handler.headers = {}
        handler.wfile = io.BytesIO()

        # Mock parent methods
        handler.send_response = MagicMock()
        handler.send_header = MagicMock()
        handler.end_headers = MagicMock()

        if body is not None:
            raw = json.dumps(body).encode()
            handler.rfile = io.BytesIO(raw)
            handler.headers = {"Content-Length": str(len(raw))}

        return handler

    def test_stats_endpoint(self):
        acc = OnlineAccumulator(num_layers=1, num_experts=4)
        inds = np.array([[0, 1]])
        weights = np.array([[0.6, 0.4]])
        acc.update(0, inds, weights)

        handler = self._make_handler(acc, "GET", "/v1/reap/stats")
        handler._handle_reap_stats()

        response = json.loads(handler.wfile.getvalue().decode())
        assert response["freq"][0][0] == 1.0
        assert response["freq"][0][1] == 1.0
        assert response["num_layers"] == 1
        assert response["num_experts"] == 4

    def test_info_endpoint(self):
        acc = OnlineAccumulator(num_layers=2, num_experts=8)
        acc.increment_request()
        acc.add_tokens(100)

        handler = self._make_handler(acc, "GET", "/v1/reap/info")
        handler._handle_reap_info()

        response = json.loads(handler.wfile.getvalue().decode())
        assert response["num_layers"] == 2
        assert response["num_experts"] == 8
        assert response["request_count"] == 1
        assert response["token_count"] == 100

    def test_reset_endpoint(self):
        acc = OnlineAccumulator(num_layers=1, num_experts=4)
        inds = np.array([[0, 1]])
        weights = np.array([[0.6, 0.4]])
        acc.update(0, inds, weights)
        acc.increment_request()

        handler = self._make_handler(acc, "POST", "/v1/reap/reset")
        handler._handle_reap_reset()

        response = json.loads(handler.wfile.getvalue().decode())
        assert response["status"] == "reset"
        assert acc.get_stats()["freq"][0] == [0.0, 0.0, 0.0, 0.0]
        assert acc.get_stats()["request_count"] == 0

    def test_save_endpoint(self, tmp_path):
        acc = OnlineAccumulator(num_layers=1, num_experts=4)
        inds = np.array([[0, 1]])
        weights = np.array([[0.6, 0.4]])
        acc.update(0, inds, weights)

        save_path = str(tmp_path / "test_save.npz")
        handler = self._make_handler(acc, "POST", "/v1/reap/save", body={"path": save_path})
        handler._handle_reap_save()

        response = json.loads(handler.wfile.getvalue().decode())
        assert response["status"] == "saved"
        assert response["path"] == save_path

        # Verify the file is loadable
        loaded = SaliencyAccumulator.load(save_path)
        assert loaded.num_layers == 1
        assert loaded.num_experts == 4
        np.testing.assert_array_equal(loaded.freq[0], acc._acc.freq[0])
