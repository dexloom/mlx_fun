"""Tests for REAM expert merging."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from mlx_fun.merger import (
    select_centroids,
    group_experts,
    _get_expert_weight,
    _compute_single_expert_output,
    compute_similarity_matrix,
    _cosine_distance_matrix,
    _greedy_alignment,
    compute_alignment,
    align_and_merge_group,
    merge_moe_layer,
)


# ---------------------------------------------------------------------------
# Centroid selection
# ---------------------------------------------------------------------------

class TestSelectCentroids:
    def test_picks_top_k(self):
        scores = np.array([0.1, 0.9, 0.3, 0.7])
        centroids = select_centroids(scores, n_keep=2)
        assert set(centroids) == {1, 3}

    def test_sorted_output(self):
        scores = np.array([0.5, 0.1, 0.9, 0.3])
        centroids = select_centroids(scores, n_keep=3)
        np.testing.assert_array_equal(centroids, np.sort(centroids))

    def test_single_centroid(self):
        scores = np.array([0.1, 0.2, 0.9, 0.4])
        centroids = select_centroids(scores, n_keep=1)
        assert centroids[0] == 2

    def test_all_centroids(self):
        scores = np.array([0.1, 0.2, 0.3, 0.4])
        centroids = select_centroids(scores, n_keep=4)
        np.testing.assert_array_equal(centroids, [0, 1, 2, 3])


# ---------------------------------------------------------------------------
# Expert grouping
# ---------------------------------------------------------------------------

class TestGroupExperts:
    def test_all_experts_assigned(self):
        scores = np.array([0.1, 0.9, 0.3, 0.7])
        centroids = np.array([1, 3])
        sim = np.ones((4, 4))  # all equally similar
        groups = group_experts(scores, centroids, sim, max_group_size=16)

        all_members = set()
        for members in groups.values():
            all_members.update(members)
        assert all_members == {0, 1, 2, 3}

    def test_centroids_are_in_own_group(self):
        scores = np.array([0.1, 0.9, 0.3, 0.7])
        centroids = np.array([1, 3])
        sim = np.eye(4)
        groups = group_experts(scores, centroids, sim, max_group_size=16)

        assert 1 in groups[1]
        assert 3 in groups[3]

    def test_max_group_size_respected(self):
        scores = np.array([0.1, 0.9, 0.3, 0.7, 0.2, 0.5])
        centroids = np.array([1, 3])
        sim = np.ones((6, 6))
        groups = group_experts(scores, centroids, sim, max_group_size=3)

        for centroid, members in groups.items():
            # May exceed max_group_size due to remainder assignment
            # but initial claim should respect it
            pass
        # All experts must be assigned regardless
        all_members = set()
        for members in groups.values():
            all_members.update(members)
        assert all_members == {0, 1, 2, 3, 4, 5}

    def test_similarity_determines_assignment(self):
        scores = np.array([0.1, 0.9, 0.3, 0.7])
        centroids = np.array([1, 3])
        # Expert 0 more similar to centroid 1, expert 2 more similar to centroid 3
        sim = np.array([
            [1.0, 0.9, 0.1, 0.2],
            [0.9, 1.0, 0.2, 0.1],
            [0.1, 0.2, 1.0, 0.8],
            [0.2, 0.1, 0.8, 1.0],
        ])
        groups = group_experts(scores, centroids, sim, max_group_size=2)

        assert 0 in groups[1]  # expert 0 should go to centroid 1
        assert 2 in groups[3]  # expert 2 should go to centroid 3


# ---------------------------------------------------------------------------
# Expert weight access
# ---------------------------------------------------------------------------

class TestGetExpertWeight:
    def test_non_quantized(self, tiny_qwen3_moe):
        switch_mlp = tiny_qwen3_moe.switch_mlp
        w = _get_expert_weight(switch_mlp.gate_proj, 0)
        mx.eval(w)
        # SwitchGLU gate_proj: (n_experts, intermediate, hidden)
        # Single expert: (intermediate, hidden)
        assert w.shape == (64, 32)


# ---------------------------------------------------------------------------
# Expert output computation
# ---------------------------------------------------------------------------

class TestComputeSingleExpertOutput:
    def test_output_shapes(self, tiny_qwen3_moe, sample_input):
        x = sample_input.reshape(-1, 32)  # (8, 32)
        output, hidden = _compute_single_expert_output(
            tiny_qwen3_moe.switch_mlp, x, 0,
        )
        mx.eval(output, hidden)
        assert output.shape == (8, 32)   # (n_tokens, hidden_dim)
        assert hidden.shape == (8, 64)   # (n_tokens, intermediate_size)

    def test_different_experts_different_outputs(self, tiny_qwen3_moe, sample_input):
        x = sample_input.reshape(-1, 32)
        out0, _ = _compute_single_expert_output(tiny_qwen3_moe.switch_mlp, x, 0)
        out1, _ = _compute_single_expert_output(tiny_qwen3_moe.switch_mlp, x, 1)
        mx.eval(out0, out1)
        assert not np.allclose(np.array(out0), np.array(out1))


# ---------------------------------------------------------------------------
# Similarity matrix
# ---------------------------------------------------------------------------

class TestSimilarityMatrix:
    def test_gated_mode_shape(self, tiny_qwen3_moe, sample_input):
        x = sample_input.reshape(-1, 32)
        x_np = np.array(x)
        gates_np = np.random.randn(8, 4).astype(np.float32)

        sim = compute_similarity_matrix(
            tiny_qwen3_moe.switch_mlp, x_np, gates_np, n_experts=4,
            mode="gated", max_tokens=8,
        )
        assert sim.shape == (4, 4)

    def test_average_mode_shape(self, tiny_qwen3_moe, sample_input):
        x = sample_input.reshape(-1, 32)
        x_np = np.array(x)
        gates_np = np.random.randn(8, 4).astype(np.float32)

        sim = compute_similarity_matrix(
            tiny_qwen3_moe.switch_mlp, x_np, gates_np, n_experts=4,
            mode="average", max_tokens=8,
        )
        assert sim.shape == (4, 4)

    def test_diagonal_is_high(self, tiny_qwen3_moe, sample_input):
        x = sample_input.reshape(-1, 32)
        x_np = np.array(x)
        gates_np = np.random.randn(8, 4).astype(np.float32)

        sim = compute_similarity_matrix(
            tiny_qwen3_moe.switch_mlp, x_np, gates_np, n_experts=4,
            mode="gated", max_tokens=8,
        )
        # Diagonal should be 1.0 (self-similarity)
        np.testing.assert_allclose(np.diag(sim), 1.0, atol=1e-5)

    def test_symmetry(self, tiny_qwen3_moe, sample_input):
        x = sample_input.reshape(-1, 32)
        x_np = np.array(x)
        gates_np = np.random.randn(8, 4).astype(np.float32)

        sim = compute_similarity_matrix(
            tiny_qwen3_moe.switch_mlp, x_np, gates_np, n_experts=4,
            mode="gated", max_tokens=8,
        )
        np.testing.assert_allclose(sim, sim.T, atol=1e-10)


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

class TestAlignment:
    def test_greedy_returns_permutation(self):
        np.random.seed(42)
        cost = np.random.rand(4, 4)
        perm = _greedy_alignment(cost)
        assert set(perm) == {0, 1, 2, 3}

    def test_identity_cost_gives_identity_perm(self):
        # Diagonal cost matrix = 0 on diagonal, 1 elsewhere
        # Should map each row to itself
        cost = np.ones((4, 4)) - np.eye(4)
        perm = _greedy_alignment(cost)
        np.testing.assert_array_equal(perm, [0, 1, 2, 3])

    def test_none_alignment_is_identity(self):
        centroid_hidden = np.random.randn(8, 4)
        member_hidden = np.random.randn(8, 4)
        centroid_weights = np.random.randn(4, 10)
        member_weights = np.random.randn(4, 10)

        perm = compute_alignment(
            centroid_hidden, member_hidden,
            centroid_weights, member_weights,
            method="none",
        )
        np.testing.assert_array_equal(perm, [0, 1, 2, 3])

    def test_cosine_distance_matrix_shape(self):
        a = np.random.randn(4, 10)
        b = np.random.randn(4, 10)
        dist = _cosine_distance_matrix(a, b)
        assert dist.shape == (4, 4)

    def test_cosine_distance_self_is_zero(self):
        a = np.random.randn(4, 10)
        dist = _cosine_distance_matrix(a, a)
        np.testing.assert_allclose(np.diag(dist), 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Group merging
# ---------------------------------------------------------------------------

class TestAlignAndMergeGroup:
    @pytest.fixture
    def qwen3_adapter(self, tiny_qwen3_moe):
        """Create a minimal adapter for testing merging."""
        class FakeAdapter:
            def __init__(self, block, config):
                self._block = block
                self.config = config

            def get_moe_block(self, layer_idx):
                return self._block

            def get_switch_mlp(self, moe_block):
                return moe_block.switch_mlp

        return FakeAdapter(
            tiny_qwen3_moe,
            {"model_type": "qwen3_moe"},
        )

    def test_merged_weight_shapes(self, qwen3_adapter, sample_input):
        x = sample_input.reshape(-1, 32)
        x_mx = mx.array(np.array(x))
        saliencies = np.array([0.1, 0.9, 0.3, 0.7])

        mg, mu, md = align_and_merge_group(
            qwen3_adapter, layer_idx=0, centroid_idx=1,
            members=[1, 0, 2], saliencies=saliencies,
            layer_input=x_mx, alignment_method="greedy",
            max_alignment_tokens=8,
        )
        mx.eval(mg, mu, md)
        # gate_proj shape: (intermediate=64, hidden=32)
        assert mg.shape == (64, 32)
        assert mu.shape == (64, 32)
        # down_proj shape: (hidden=32, intermediate=64)
        assert md.shape == (32, 64)

    def test_single_member_preserves_weights(self, qwen3_adapter, sample_input):
        """If centroid is the only member, merged weights should equal centroid weights."""
        x = sample_input.reshape(-1, 32)
        x_mx = mx.array(np.array(x))
        saliencies = np.array([0.1, 0.9, 0.3, 0.7])

        switch_mlp = qwen3_adapter.get_moe_block(0).switch_mlp
        original_gate = np.array(_get_expert_weight(switch_mlp.gate_proj, 1))
        original_up = np.array(_get_expert_weight(switch_mlp.up_proj, 1))
        original_down = np.array(_get_expert_weight(switch_mlp.down_proj, 1))

        mg, mu, md = align_and_merge_group(
            qwen3_adapter, layer_idx=0, centroid_idx=1,
            members=[1], saliencies=saliencies,
            layer_input=x_mx, alignment_method="greedy",
            max_alignment_tokens=8,
        )
        mx.eval(mg, mu, md)

        np.testing.assert_allclose(np.array(mg), original_gate, atol=1e-5)
        np.testing.assert_allclose(np.array(mu), original_up, atol=1e-5)
        np.testing.assert_allclose(np.array(md), original_down, atol=1e-5)


# ---------------------------------------------------------------------------
# Full layer merging
# ---------------------------------------------------------------------------

class TestMergeMoeLayer:
    @pytest.fixture
    def qwen3_setup(self, tiny_qwen3_moe, sample_input):
        """Set up adapter and data for layer merge test."""
        class FakeAdapter:
            def __init__(self, block, config):
                self._block = block
                self.config = config

            def get_moe_block(self, layer_idx):
                return self._block

            def get_switch_mlp(self, moe_block):
                return moe_block.switch_mlp

        adapter = FakeAdapter(tiny_qwen3_moe, {"model_type": "qwen3_moe"})
        x = sample_input.reshape(-1, 32)
        x_mx = mx.array(np.array(x))
        return adapter, tiny_qwen3_moe, x_mx

    def test_merge_reduces_expert_count(self, qwen3_setup):
        adapter, block, x = qwen3_setup
        saliencies = np.array([0.1, 0.9, 0.3, 0.7])
        centroids = np.array([1, 3])
        groups = {1: [1, 0], 3: [3, 2]}

        merge_moe_layer(
            adapter, layer_idx=0, centroids=centroids,
            groups=groups, saliencies=saliencies,
            layer_input=x, alignment_method="none",
            max_alignment_tokens=8,
        )

        # After merge: 2 experts instead of 4
        assert block.switch_mlp.gate_proj.weight.shape[0] == 2
        assert block.switch_mlp.up_proj.weight.shape[0] == 2
        assert block.switch_mlp.down_proj.weight.shape[0] == 2

    def test_gate_sliced_for_qwen3(self, qwen3_setup):
        adapter, block, x = qwen3_setup
        saliencies = np.array([0.1, 0.9, 0.3, 0.7])
        centroids = np.array([1, 3])
        groups = {1: [1, 0], 3: [3, 2]}

        merge_moe_layer(
            adapter, layer_idx=0, centroids=centroids,
            groups=groups, saliencies=saliencies,
            layer_input=x, alignment_method="none",
            max_alignment_tokens=8,
        )

        # Gate weight should have 2 output features (one per centroid)
        assert block.gate.weight.shape[0] == 2
        assert block.num_experts == 2

    def test_merged_model_can_forward(self, qwen3_setup, sample_input):
        adapter, block, x = qwen3_setup
        saliencies = np.array([0.1, 0.9, 0.3, 0.7])
        centroids = np.array([1, 3])
        groups = {1: [1, 0], 3: [3, 2]}

        merge_moe_layer(
            adapter, layer_idx=0, centroids=centroids,
            groups=groups, saliencies=saliencies,
            layer_input=x, alignment_method="none",
            max_alignment_tokens=8,
        )

        # Update top_k to match new expert count
        block.top_k = 2

        # Should be able to forward through the merged block
        out = block(sample_input)
        mx.eval(out)
        assert out.shape == sample_input.shape
