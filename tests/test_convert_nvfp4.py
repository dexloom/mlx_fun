"""Tests for NVIDIA NVFP4 -> MLX NVFP4 conversion logic."""

import numpy as np
import mlx.core as mx
import pytest

from mlx_fun.convert_nvfp4 import (
    _repack_nvfp4_weight,
    _fold_global_scale,
    _dequant_fp8,
    _stack_experts,
    _fix_conv1d_weights,
    _parse_quant_config,
    _find_layer_algo,
)


class TestRepackNVFP4Weight:
    """Test uint8 -> uint32 repacking."""

    def test_shape(self):
        w = np.random.randint(0, 256, size=(64, 128), dtype=np.uint8)
        result = _repack_nvfp4_weight(w)
        assert result.shape == (64, 32)
        assert result.dtype == np.uint32

    def test_round_trip_bytes(self):
        """uint8 -> uint32 -> uint8 should be identity."""
        w = np.random.randint(0, 256, size=(8, 16), dtype=np.uint8)
        repacked = _repack_nvfp4_weight(w)
        back = repacked.view(np.uint8).reshape(w.shape)
        np.testing.assert_array_equal(w, back)

    def test_compatible_with_mlx_dequantize(self):
        """Repacked weights should be dequantizable by MLX."""
        # Create a weight via MLX quantize, extract uint8 view, repack
        w = mx.random.normal((16, 32))
        wq, scales = mx.quantize(w, mode="nvfp4")
        # MLX uint32 -> uint8 -> repack back to uint32
        wq_u8 = np.array(wq).view(np.uint8)
        repacked = _repack_nvfp4_weight(wq_u8.reshape(16, -1))
        repacked_mx = mx.array(repacked)
        # Should dequantize identically
        deq_original = mx.dequantize(wq, scales, mode="nvfp4").astype(mx.float32)
        deq_repacked = mx.dequantize(repacked_mx, scales, mode="nvfp4").astype(mx.float32)
        np.testing.assert_array_equal(
            np.array(deq_original), np.array(deq_repacked)
        )


class TestFoldGlobalScale:
    """Test global scale folding into e4m3 group scales."""

    def test_identity_scale(self):
        """Global scale of 1.0 should not change scales."""
        scales = np.array([[48, 56], [64, 72]], dtype=np.uint8)
        folded = _fold_global_scale(scales, 1.0)
        np.testing.assert_array_equal(scales, folded)

    def test_power_of_two_scale(self):
        """Power-of-2 global scales should fold exactly in e4m3."""
        # e4m3 value 1.0 * 2.0 = 2.0 (exactly representable)
        scale_1 = mx.to_fp8(mx.array([1.0], dtype=mx.float32))
        scales = np.array(scale_1).reshape(1, 1)
        folded = _fold_global_scale(scales, 2.0)
        result = float(mx.from_fp8(mx.array(folded), dtype=mx.float32).item())
        assert result == 2.0

    def test_error_bounded(self):
        """Folding error should be bounded for reasonable scales."""
        np.random.seed(42)
        # Simulate realistic e4m3 scales
        float_scales = np.random.uniform(0.1, 10.0, size=(64, 16)).astype(np.float32)
        scales_e4m3 = np.array(mx.to_fp8(mx.array(float_scales)))
        global_scale = 0.73

        folded = _fold_global_scale(scales_e4m3, global_scale)

        # Check relative error
        original_float = np.array(mx.from_fp8(mx.array(scales_e4m3), dtype=mx.float32))
        expected = original_float * global_scale
        actual = np.array(mx.from_fp8(mx.array(folded), dtype=mx.float32))

        rel_error = np.abs(expected - actual) / (np.abs(expected) + 1e-10)
        assert rel_error.mean() < 0.10, f"Mean relative error too high: {rel_error.mean()}"


class TestDequantFP8:
    """Test FP8 dequantization."""

    def test_basic(self):
        # Encode some values to e4m3, then dequant
        values = mx.array([1.0, 2.0, -3.0, 0.5], dtype=mx.float32)
        encoded = np.array(mx.to_fp8(values))
        result = _dequant_fp8(encoded, scale=1.0)
        np.testing.assert_allclose(
            np.array(result.astype(mx.float32)),
            np.array(values),
            atol=0.01,
        )

    def test_with_scale(self):
        values = mx.array([1.0, 2.0], dtype=mx.float32)
        encoded = np.array(mx.to_fp8(values))
        result = _dequant_fp8(encoded, scale=0.5)
        expected = np.array(values) * 0.5
        np.testing.assert_allclose(
            np.array(result.astype(mx.float32)),
            expected,
            atol=0.01,
        )


class TestStackExperts:
    """Test expert weight stacking."""

    def test_basic_stacking(self):
        weights = {}
        n_experts = 4
        # Create per-expert weights
        for e in range(n_experts):
            weights[f"backbone.layers.0.mixer.experts.{e}.up_proj.weight"] = (
                mx.ones((8, 16)) * (e + 1)
            )
            weights[f"backbone.layers.0.mixer.experts.{e}.down_proj.weight"] = (
                mx.ones((16, 8)) * (e + 1)
            )

        result = _stack_experts(weights, n_layers=1, n_experts=4)

        # Should have stacked tensors
        assert "backbone.layers.0.mixer.switch_mlp.fc1.weight" in result
        assert "backbone.layers.0.mixer.switch_mlp.fc2.weight" in result

        fc1 = result["backbone.layers.0.mixer.switch_mlp.fc1.weight"]
        assert fc1.shape == (4, 8, 16)  # (n_experts, out, in)

        # Individual expert entries should be removed
        assert "backbone.layers.0.mixer.experts.0.up_proj.weight" not in result

    def test_stacking_with_scales(self):
        weights = {}
        n_experts = 2
        for e in range(n_experts):
            weights[f"backbone.layers.0.mixer.experts.{e}.up_proj.weight"] = (
                mx.zeros((4, 2), dtype=mx.uint32)
            )
            weights[f"backbone.layers.0.mixer.experts.{e}.up_proj.scales"] = (
                mx.zeros((4, 1), dtype=mx.uint8)
            )

        result = _stack_experts(weights, n_layers=1, n_experts=2, quantized=True)

        assert "backbone.layers.0.mixer.switch_mlp.fc1.weight" in result
        assert "backbone.layers.0.mixer.switch_mlp.fc1.scales" in result
        assert result["backbone.layers.0.mixer.switch_mlp.fc1.scales"].shape == (2, 4, 1)

    def test_non_moe_layers_untouched(self):
        weights = {
            "backbone.layers.0.norm.weight": mx.ones((16,)),
            "backbone.layers.0.mixer.gate.weight": mx.ones((4, 16)),
        }
        result = _stack_experts(weights, n_layers=1, n_experts=4)
        assert "backbone.layers.0.norm.weight" in result
        assert "backbone.layers.0.mixer.gate.weight" in result


class TestFixConv1dWeights:
    """Test conv1d weight transposition."""

    def test_transpose_needed(self):
        weights = {"backbone.layers.0.mixer.conv1d.weight": mx.ones((8, 1, 4))}
        result = _fix_conv1d_weights(weights)
        assert result["backbone.layers.0.mixer.conv1d.weight"].shape == (8, 4, 1)

    def test_no_transpose(self):
        weights = {"backbone.layers.0.mixer.conv1d.weight": mx.ones((8, 4, 1))}
        result = _fix_conv1d_weights(weights)
        assert result["backbone.layers.0.mixer.conv1d.weight"].shape == (8, 4, 1)


class TestParseQuantConfig:
    """Test quantization config parsing."""

    def test_config_groups_format(self):
        config = {
            "quantization_config": {
                "config_groups": {
                    "group_0": {
                        "weights": {"num_bits": 8, "type": "float"},
                        "targets": ["backbone.layers.0.mixer.in_proj"],
                    },
                    "group_1": {
                        "weights": {"num_bits": 4, "type": "float", "group_size": 16},
                        "targets": ["backbone.layers.1.mixer.experts.0.up_proj"],
                    },
                }
            }
        }
        algo, params = _parse_quant_config(config)
        assert algo["backbone.layers.0.mixer.in_proj"] == "FP8"
        assert algo["backbone.layers.1.mixer.experts.0.up_proj"] == "NVFP4"

    def test_quantized_layers_format(self):
        config = {
            "quantization": {
                "quantized_layers": {
                    "backbone.layers.0.mixer.in_proj": {"quant_algo": "FP8"},
                    "backbone.layers.1.mixer.experts.0.up_proj": {
                        "quant_algo": "NVFP4",
                        "group_size": 16,
                    },
                }
            }
        }
        algo, _ = _parse_quant_config(config)
        assert algo["backbone.layers.0.mixer.in_proj"] == "FP8"
        assert algo["backbone.layers.1.mixer.experts.0.up_proj"] == "NVFP4"


class TestFindLayerAlgo:
    """Test tensor name -> algo lookup."""

    def test_weight_suffix(self):
        algo_map = {"backbone.layers.0.mixer.in_proj": "FP8"}
        assert _find_layer_algo("backbone.layers.0.mixer.in_proj.weight", algo_map) == "FP8"

    def test_scale_suffix(self):
        algo_map = {"backbone.layers.0.mixer.experts.0.up_proj": "NVFP4"}
        assert _find_layer_algo(
            "backbone.layers.0.mixer.experts.0.up_proj.weight_scale", algo_map
        ) == "NVFP4"

    def test_unknown(self):
        algo_map = {"backbone.layers.0.mixer.in_proj": "FP8"}
        assert _find_layer_algo("backbone.embeddings.weight", algo_map) is None
