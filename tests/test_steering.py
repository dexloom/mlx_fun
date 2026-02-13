"""Tests for expert steering hooks."""

import mlx.core as mx
import numpy as np
import pytest

from mlx_fun.steering import (
    SteeringConfig,
    install_steering_hooks,
    remove_steering_hooks,
    update_steering_config,
    _compute_bias,
)


class TestSteeringConfig:
    def test_to_dict_from_dict_roundtrip(self):
        config = SteeringConfig(
            deactivate={0: [1, 2], 3: [5]},
            activate={1: [0]},
            mask_value=-1e9,
            boost_value=1e4,
        )
        d = config.to_dict()
        loaded = SteeringConfig.from_dict(d)
        assert loaded.deactivate == {0: [1, 2], 3: [5]}
        assert loaded.activate == {1: [0]}
        assert loaded.mask_value == -1e9
        assert loaded.boost_value == 1e4

    def test_from_safety_report(self, tmp_path):
        from mlx_fun.safety import SafetyReport
        report = SafetyReport(
            num_layers=2, num_experts=4, threshold_percentile=90.0,
            differential_freq=np.zeros((2, 4)),
            differential_activation=np.zeros((2, 4)),
            composite_score=np.zeros((2, 4)),
            hcdg_experts={0: [0]},
            hrcg_experts={0: [1], 1: [2]},
            safety_critical={0: [0, 1], 1: [2]},
        )
        path = str(tmp_path / "report.json")
        report.save(path)

        # Safe mode: boost HRCG experts
        safe_config = SteeringConfig.from_safety_report(path, "safe")
        assert safe_config.activate == {0: [1], 1: [2]}
        assert safe_config.deactivate == {}

        # Unsafe mode: mask all safety-critical experts
        unsafe_config = SteeringConfig.from_safety_report(path, "unsafe")
        assert unsafe_config.deactivate == {0: [0, 1], 1: [2]}
        assert unsafe_config.activate == {}


class TestComputeBias:
    def test_deactivation_bias(self):
        config = SteeringConfig(deactivate={0: [1, 3]}, mask_value=-1e9)
        bias = _compute_bias(0, 4, config)
        assert bias is not None
        bias_np = np.array(bias)
        assert bias_np[0] == 0.0
        assert bias_np[1] == pytest.approx(-1e9)
        assert bias_np[2] == 0.0
        assert bias_np[3] == pytest.approx(-1e9)

    def test_activation_bias(self):
        config = SteeringConfig(activate={0: [2]}, boost_value=1e4)
        bias = _compute_bias(0, 4, config)
        bias_np = np.array(bias)
        assert bias_np[2] == pytest.approx(1e4)
        assert bias_np[0] == 0.0

    def test_no_bias_returns_none(self):
        config = SteeringConfig(deactivate={1: [0]})  # layer 1, not 0
        bias = _compute_bias(0, 4, config)
        assert bias is None

    def test_out_of_range_expert_ignored(self):
        config = SteeringConfig(deactivate={0: [99]})
        bias = _compute_bias(0, 4, config)
        assert bias is None  # Expert 99 out of range, no bias applied


class TestSteeringHooksMiniMax:
    def test_install_remove(self, tiny_minimax_moe, sample_input):
        config = SteeringConfig(deactivate={0: [0, 1]})
        original_cls = type(tiny_minimax_moe)
        install_steering_hooks([tiny_minimax_moe], "minimax", config, num_experts=4)
        assert hasattr(tiny_minimax_moe, "_steering_bias")
        assert type(tiny_minimax_moe) is not original_cls

        out = tiny_minimax_moe(sample_input)
        mx.eval(out)

        remove_steering_hooks([tiny_minimax_moe])
        assert type(tiny_minimax_moe) is original_cls
        assert not hasattr(tiny_minimax_moe, "_steering_bias")

    def test_deactivation_changes_output(self, tiny_minimax_moe, sample_input):
        # Baseline output
        orig_out = tiny_minimax_moe(sample_input)
        mx.eval(orig_out)
        orig_np = np.array(orig_out, copy=False).copy()

        # Steered output
        config = SteeringConfig(deactivate={0: [0, 1]}, mask_value=-1e9)
        install_steering_hooks([tiny_minimax_moe], "minimax", config, num_experts=4)
        steered_out = tiny_minimax_moe(sample_input)
        mx.eval(steered_out)
        steered_np = np.array(steered_out, copy=False)
        remove_steering_hooks([tiny_minimax_moe])

        assert not np.allclose(orig_np, steered_np, atol=1e-5)


class TestSteeringHooksQwen3:
    def test_install_remove(self, tiny_qwen3_moe, sample_input):
        config = SteeringConfig(deactivate={0: [0]})
        install_steering_hooks([tiny_qwen3_moe], "qwen3_moe", config, num_experts=4)
        out = tiny_qwen3_moe(sample_input)
        mx.eval(out)
        remove_steering_hooks([tiny_qwen3_moe])

    def test_deactivation_changes_output(self, tiny_qwen3_moe, sample_input):
        orig_out = tiny_qwen3_moe(sample_input)
        mx.eval(orig_out)
        orig_np = np.array(orig_out, copy=False).copy()

        config = SteeringConfig(deactivate={0: [0, 1]}, mask_value=-1e9)
        install_steering_hooks([tiny_qwen3_moe], "qwen3_moe", config, num_experts=4)
        steered_out = tiny_qwen3_moe(sample_input)
        mx.eval(steered_out)
        steered_np = np.array(steered_out, copy=False)
        remove_steering_hooks([tiny_qwen3_moe])

        assert not np.allclose(orig_np, steered_np, atol=1e-5)


class TestUpdateSteeringConfig:
    def test_hot_swap_bias(self, tiny_minimax_moe, sample_input):
        config1 = SteeringConfig(deactivate={0: [0]}, mask_value=-1e9)
        install_steering_hooks([tiny_minimax_moe], "minimax", config1, num_experts=4)

        out1 = tiny_minimax_moe(sample_input)
        mx.eval(out1)
        out1_np = np.array(out1, copy=False).copy()

        # Hot-swap to different config
        config2 = SteeringConfig(deactivate={0: [2, 3]}, mask_value=-1e9)
        update_steering_config([tiny_minimax_moe], config2, num_experts=4)

        out2 = tiny_minimax_moe(sample_input)
        mx.eval(out2)
        out2_np = np.array(out2, copy=False)

        remove_steering_hooks([tiny_minimax_moe])

        # Different configs should give different outputs
        assert not np.allclose(out1_np, out2_np, atol=1e-5)
