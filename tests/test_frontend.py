"""Tests for the frontend module (API helpers + visualization)."""

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from unittest.mock import patch, MagicMock

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Import frontend helpers (skip if gradio/matplotlib not installed)
# ---------------------------------------------------------------------------

try:
    from mlx_fun.frontend import (
        fetch_server_info,
        fetch_stats,
        post_save,
        post_reset,
        fetch_steering,
        post_steering,
        delete_steering,
        stream_chat,
        make_freq_heatmap,
        make_weighted_freq_heatmap,
        make_per_layer_bar,
        _safe_call,
    )
    HAS_FRONTEND_DEPS = True
except ImportError:
    HAS_FRONTEND_DEPS = False

pytestmark = pytest.mark.skipif(
    not HAS_FRONTEND_DEPS,
    reason="gradio/matplotlib not installed",
)


# ---------------------------------------------------------------------------
# Tiny mock HTTP server
# ---------------------------------------------------------------------------

class _MockHandler(BaseHTTPRequestHandler):
    """Handler that returns canned REAP API responses."""

    _stats = {
        "freq": [[10, 20, 30, 40], [5, 15, 25, 35]],
        "weighted_freq_sum": [[1.0, 2.0, 3.0, 4.0], [0.5, 1.5, 2.5, 3.5]],
        "reap_sum": [[0, 0, 0, 0], [0, 0, 0, 0]],
        "ean_sum": [[0, 0, 0, 0], [0, 0, 0, 0]],
        "reap_count": [[0, 0, 0, 0], [0, 0, 0, 0]],
        "num_layers": 2,
        "num_experts": 4,
        "request_count": 5,
        "token_count": 100,
    }

    _info = {
        "num_layers": 2,
        "num_experts": 4,
        "request_count": 5,
        "token_count": 100,
        "steering_active": False,
    }

    def do_GET(self):
        if self.path == "/v1/reap/stats":
            self._json_response(200, self._stats)
        elif self.path == "/v1/reap/info":
            self._json_response(200, self._info)
        elif self.path == "/v1/reap/steer":
            self._json_response(200, {"active": False})
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b"{}"

        if self.path == "/v1/reap/save":
            data = json.loads(body.decode())
            self._json_response(200, {"status": "saved", "path": data.get("path", "")})
        elif self.path == "/v1/reap/reset":
            self._json_response(200, {"status": "reset"})
        elif self.path == "/v1/reap/steer":
            data = json.loads(body.decode())
            self._json_response(200, {"status": "steering_updated", "config": data})
        elif self.path == "/v1/chat/completions":
            data = json.loads(body.decode())
            if data.get("stream"):
                self._stream_response()
            else:
                self._json_response(200, {
                    "choices": [{"message": {"content": "Hello!"}}],
                })
        else:
            self.send_response(404)
            self.end_headers()

    def do_DELETE(self):
        if self.path == "/v1/reap/steer":
            self._json_response(200, {"status": "steering_removed"})
        else:
            self.send_response(405)
            self.end_headers()

    def _json_response(self, status, data):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def _stream_response(self):
        self.send_response(200)
        self.send_header("Content-type", "text/event-stream")
        self.end_headers()
        chunks = [
            {"choices": [{"delta": {"content": "Hello"}}]},
            {"choices": [{"delta": {"content": " world"}}]},
        ]
        for chunk in chunks:
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def log_message(self, format, *args):
        pass  # Suppress logs during tests


@pytest.fixture(scope="module")
def mock_server():
    """Start a mock REAP server for testing."""
    server = HTTPServer(("127.0.0.1", 0), _MockHandler)
    port = server.server_address[1]
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


# ---------------------------------------------------------------------------
# API client tests
# ---------------------------------------------------------------------------

class TestAPIClients:
    def test_fetch_server_info(self, mock_server):
        info = fetch_server_info(mock_server)
        assert info["num_layers"] == 2
        assert info["num_experts"] == 4
        assert info["request_count"] == 5
        assert info["token_count"] == 100

    def test_fetch_stats(self, mock_server):
        stats = fetch_stats(mock_server)
        assert stats["num_layers"] == 2
        assert stats["num_experts"] == 4
        assert len(stats["freq"]) == 2
        assert len(stats["freq"][0]) == 4

    def test_post_save(self, mock_server):
        result = post_save(mock_server, "test_output.npz")
        assert result["status"] == "saved"
        assert result["path"] == "test_output.npz"

    def test_post_reset(self, mock_server):
        result = post_reset(mock_server)
        assert result["status"] == "reset"

    def test_fetch_steering(self, mock_server):
        result = fetch_steering(mock_server)
        assert result["active"] is False

    def test_post_steering(self, mock_server):
        config = {"deactivate": {"0": [1, 2]}, "activate": {"1": [3]}}
        result = post_steering(mock_server, config)
        assert result["status"] == "steering_updated"
        assert result["config"]["deactivate"] == {"0": [1, 2]}

    def test_delete_steering(self, mock_server):
        result = delete_steering(mock_server)
        assert result["status"] == "steering_removed"

    def test_stream_chat(self, mock_server):
        messages = [{"role": "user", "content": "Hi"}]
        tokens = list(stream_chat(mock_server, messages))
        assert "".join(tokens) == "Hello world"

    def test_connection_error(self):
        with pytest.raises(Exception):
            fetch_server_info("http://127.0.0.1:1")


# ---------------------------------------------------------------------------
# Visualization tests
# ---------------------------------------------------------------------------

class TestVisualization:
    def test_freq_heatmap_shape(self):
        freq = np.array([[10, 20, 30, 40], [5, 15, 25, 35]])
        fig = make_freq_heatmap(freq)
        assert fig is not None
        # Should have one axes with an image
        ax = fig.axes[0]
        assert len(ax.images) == 1
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_freq_heatmap_log_scale(self):
        """Log scale used when range is large."""
        freq = np.array([[0, 0, 0, 10000], [1, 1, 1, 10000]])
        fig = make_freq_heatmap(freq)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_weighted_freq_heatmap(self):
        wfreq = np.array([[1.0, 2.0, 3.0, 4.0], [0.5, 1.5, 2.5, 3.5]])
        fig = make_weighted_freq_heatmap(wfreq)
        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_per_layer_bar(self):
        freq = np.array([[10, 20, 30, 40], [5, 15, 25, 35]])
        fig = make_per_layer_bar(freq, 0)
        assert fig is not None
        # Should have bars
        ax = fig.axes[0]
        assert len(ax.patches) == 4  # 4 experts = 4 bars
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_per_layer_bar_layer1(self):
        freq = np.array([[10, 20, 30, 40], [5, 15, 25, 35]])
        fig = make_per_layer_bar(freq, 1)
        ax = fig.axes[0]
        # Bar heights should match layer 1 values
        heights = [p.get_height() for p in ax.patches]
        assert heights == [5, 15, 25, 35]
        import matplotlib.pyplot as plt
        plt.close(fig)


# ---------------------------------------------------------------------------
# Utility tests
# ---------------------------------------------------------------------------

class TestSafeCall:
    def test_safe_call_success(self):
        result = _safe_call(lambda: {"ok": True})
        assert result == {"ok": True}

    def test_safe_call_connection_error(self):
        import requests
        def raise_conn():
            raise requests.ConnectionError("fail")
        result = _safe_call(raise_conn)
        assert "error" in result
        assert "Cannot connect" in result["error"]

    def test_safe_call_generic_error(self):
        def raise_err():
            raise ValueError("something went wrong")
        result = _safe_call(raise_err)
        assert "error" in result
        assert "something went wrong" in result["error"]
