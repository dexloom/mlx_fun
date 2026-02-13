"""Tests for abliteration (refusal direction orthogonalization)."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from mlx_fun.abliterate import (
    _orthogonalize_linear,
    _orthogonalize_switch_proj,
    _orthogonalize_expert_proj,
    auto_select_layers,
)


class TestOrthogonalizeLinear:
    def test_removes_direction_component(self):
        mx.random.seed(42)
        linear = nn.Linear(32, 16)
        mx.eval(linear.parameters())

        d = mx.random.normal((32,))
        d = d / mx.linalg.norm(d)
        mx.eval(d)

        _orthogonalize_linear(linear, d)
        mx.eval(linear.weight)

        # W @ d should now be approximately zero
        projection = linear.weight @ d
        mx.eval(projection)
        np.testing.assert_allclose(
            np.array(projection, copy=False), 0.0, atol=1e-5
        )

    def test_preserves_orthogonal_directions(self):
        mx.random.seed(42)
        linear = nn.Linear(32, 16)
        mx.eval(linear.parameters())

        # Use a specific unit vector
        d = mx.zeros(32)
        d = d.at[0].add(1.0)
        mx.eval(d)

        d_perp = mx.zeros(32)
        d_perp = d_perp.at[1].add(1.0)
        mx.eval(d_perp)

        proj_before = linear.weight @ d_perp
        mx.eval(proj_before)
        before_np = np.array(proj_before, copy=False).copy()

        _orthogonalize_linear(linear, d)
        mx.eval(linear.weight)

        proj_after = linear.weight @ d_perp
        mx.eval(proj_after)
        np.testing.assert_allclose(
            before_np,
            np.array(proj_after, copy=False),
            atol=1e-5,
        )

    def test_unit_vector_idempotent(self):
        """Applying orthogonalization twice should give same result."""
        mx.random.seed(42)
        linear = nn.Linear(32, 16)
        mx.eval(linear.parameters())

        d = mx.random.normal((32,))
        d = d / mx.linalg.norm(d)
        mx.eval(d)

        _orthogonalize_linear(linear, d)
        mx.eval(linear.weight)
        w_after_first = np.array(linear.weight, copy=False).copy()

        _orthogonalize_linear(linear, d)
        mx.eval(linear.weight)
        w_after_second = np.array(linear.weight, copy=False)

        np.testing.assert_allclose(w_after_first, w_after_second, atol=1e-5)


class TestOrthogonalizeSwitchProj:
    def test_all_experts_orthogonalized(self):
        from mlx_lm.models.switch_layers import SwitchGLU

        mx.random.seed(42)
        switch = SwitchGLU(32, 64, 4)
        mx.eval(switch.parameters())

        d = mx.random.normal((32,))
        d = d / mx.linalg.norm(d)
        mx.eval(d)

        _orthogonalize_switch_proj(switch.down_proj, d)
        mx.eval(switch.down_proj.weight)

        w = switch.down_proj.weight  # (4, 32, 64)
        for expert_idx in range(4):
            # For each expert: expert_w @ d should be ~0
            # expert_w shape: (32, 64), d shape: (32,)
            # projection = d^T @ expert_w -> (64,)
            proj = mx.einsum("o,oi->i", d, w[expert_idx])
            mx.eval(proj)
            np.testing.assert_allclose(
                np.array(proj, copy=False), 0.0, atol=1e-4
            )


class TestOrthogonalizeExpertProj:
    def test_single_expert_orthogonalized(self):
        from mlx_lm.models.switch_layers import SwitchGLU

        mx.random.seed(42)
        switch = SwitchGLU(32, 64, 4)
        mx.eval(switch.parameters())

        d = mx.random.normal((32,))
        d = d / mx.linalg.norm(d)
        mx.eval(d)

        # Save expert 1's pre-orthogonalization projection
        w_before = switch.down_proj.weight
        mx.eval(w_before)
        expert1_proj_before = mx.einsum("o,oi->i", d, w_before[1])
        expert0_w_before = np.array(w_before[0], copy=False).copy()
        mx.eval(expert1_proj_before)

        # Orthogonalize only expert 1
        _orthogonalize_expert_proj(switch.down_proj, 1, d)
        mx.eval(switch.down_proj.weight)

        w_after = switch.down_proj.weight
        # Expert 1 should be orthogonal to d
        expert1_proj_after = mx.einsum("o,oi->i", d, w_after[1])
        mx.eval(expert1_proj_after)
        np.testing.assert_allclose(
            np.array(expert1_proj_after, copy=False), 0.0, atol=1e-4
        )

        # Expert 0 should be unchanged
        np.testing.assert_allclose(
            np.array(w_after[0], copy=False),
            expert0_w_before,
            atol=1e-6,
        )


class TestAutoSelectLayers:
    def test_selects_top_fraction(self):
        directions = {
            0: np.array([1.0, 0.0, 0.0]),
            1: np.array([0.0, 5.0, 0.0]),
            2: np.array([0.0, 0.0, 0.1]),
            3: np.array([3.0, 0.0, 0.0]),
        }
        selected = auto_select_layers(directions, top_fraction=0.5)
        # Should select layers with norms 5.0 and 3.0
        assert len(selected) == 2
        assert 1 in selected  # norm 5.0
        assert 3 in selected  # norm 3.0

    def test_minimum_one_layer(self):
        directions = {0: np.array([1.0])}
        selected = auto_select_layers(directions, top_fraction=0.1)
        assert len(selected) >= 1

    def test_returns_sorted(self):
        directions = {
            5: np.array([10.0]),
            2: np.array([5.0]),
            8: np.array([3.0]),
        }
        selected = auto_select_layers(directions, top_fraction=1.0)
        assert selected == sorted(selected)
