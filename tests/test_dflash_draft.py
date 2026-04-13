"""Tests for DFlash block diffusion draft model (Phase 3)."""

import pytest
import mlx.core as mx
import mlx.nn as nn

from mlx_fun.dflash_draft import (
    BlockDiffusionDraftModel,
    DFlashAttention,
    DFlashDecoderLayer,
    DFlashMLP,
    DFlashRMSNorm,
    DFlashRotaryEmbedding,
    build_target_layer_ids,
    create_dflash_draft_model,
    extract_context_feature,
    parse_block_size,
)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

class TestBuildTargetLayerIds:
    def test_single_layer(self):
        assert build_target_layer_ids(32, 1) == [16]

    def test_five_layers(self):
        ids = build_target_layer_ids(32, 5)
        assert len(ids) == 5
        assert ids[0] >= 1
        assert ids[-1] <= 29
        assert ids == sorted(ids), "IDs should be monotonically increasing"

    def test_all_within_range(self):
        ids = build_target_layer_ids(64, 8)
        assert len(ids) == 8
        for lid in ids:
            assert 1 <= lid <= 61  # end = 64 - 3 = 61

    def test_two_layers(self):
        ids = build_target_layer_ids(32, 2)
        assert len(ids) == 2
        assert ids[0] == 1
        assert ids[1] == 29

    def test_large_target_small_draft(self):
        ids = build_target_layer_ids(128, 3)
        assert len(ids) == 3
        assert ids[0] >= 1


class TestExtractContextFeature:
    def test_basic_concat(self):
        mx.random.seed(42)
        hidden = {
            0: mx.random.normal((1, 4, 32)),
            5: mx.random.normal((1, 4, 32)),
            10: mx.random.normal((1, 4, 32)),
        }
        result = extract_context_feature(hidden, [0, 5, 10])
        assert result.shape == (1, 4, 96)

    def test_subset_of_layers(self):
        mx.random.seed(42)
        hidden = {i: mx.random.normal((2, 3, 16)) for i in range(10)}
        result = extract_context_feature(hidden, [2, 7])
        assert result.shape == (2, 3, 32)

    def test_single_layer(self):
        h = {3: mx.random.normal((1, 5, 64))}
        result = extract_context_feature(h, [3])
        assert result.shape == (1, 5, 64)


class TestParseBlockSize:
    def test_b_prefix(self):
        assert parse_block_size("b16") == 16
        assert parse_block_size("b32") == 32

    def test_no_prefix(self):
        assert parse_block_size("16") == 16
        assert parse_block_size("32") == 32

    def test_whitespace(self):
        assert parse_block_size("  b16  ") == 16

    def test_case_insensitive(self):
        assert parse_block_size("B16") == 16

    def test_zero_raises(self):
        with pytest.raises(ValueError, match="positive"):
            parse_block_size("0")

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="positive"):
            parse_block_size("-1")

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            parse_block_size("abc")


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------

class TestDFlashRMSNorm:
    def test_output_shape(self):
        norm = DFlashRMSNorm(64)
        x = mx.random.normal((2, 8, 64))
        out = norm(x)
        assert out.shape == (2, 8, 64)

    def test_output_dtype_matches_weight(self):
        norm = DFlashRMSNorm(32)
        x = mx.random.normal((1, 4, 32)).astype(mx.bfloat16)
        out = norm(x)
        # Weight is float32, so output is promoted to float32
        assert out.dtype == mx.float32

    def test_normalized_output(self):
        norm = DFlashRMSNorm(32)
        x = mx.ones((1, 1, 32)) * 5.0
        out = norm(x)
        mx.eval(out)
        rms = mx.sqrt(mx.mean(out * out, axis=-1))
        assert abs(rms.item() - 1.0) < 0.01


class TestDFlashRotaryEmbedding:
    def test_output_shapes(self):
        rope = DFlashRotaryEmbedding(head_dim=16)
        x = mx.random.normal((1, 10, 16))
        pos_ids = mx.arange(10).reshape(1, -1)
        cos, sin = rope(x, pos_ids)
        assert cos.shape == (1, 10, 16)
        assert sin.shape == (1, 10, 16)

    def test_different_seq_lens(self):
        rope = DFlashRotaryEmbedding(head_dim=32)
        for seq_len in [1, 8, 64]:
            pos_ids = mx.arange(seq_len).reshape(1, -1)
            x = mx.random.normal((1, seq_len, 32))
            cos, sin = rope(x, pos_ids)
            assert cos.shape == (1, seq_len, 32)


class TestDFlashMLP:
    def test_output_shape(self):
        mlp = DFlashMLP(hidden_size=64, intermediate_size=128)
        mx.eval(mlp.parameters())
        x = mx.random.normal((2, 4, 64))
        out = mlp(x)
        assert out.shape == (2, 4, 64)


class TestDFlashAttention:
    def test_output_shape(self):
        mx.random.seed(42)
        attn = DFlashAttention(hidden_size=64, num_heads=4)
        mx.eval(attn.parameters())
        draft_h = mx.random.normal((1, 8, 64))
        target_h = mx.random.normal((1, 16, 64))
        rope = DFlashRotaryEmbedding(head_dim=16)
        pos_ids = mx.arange(24).reshape(1, -1)  # 16 ctx + 8 block
        pos_emb = rope(draft_h, pos_ids)
        out = attn(draft_h, target_h, pos_emb)
        assert out.shape == (1, 8, 64)

    def test_gqa(self):
        mx.random.seed(42)
        attn = DFlashAttention(hidden_size=64, num_heads=8, num_kv_heads=2)
        mx.eval(attn.parameters())
        draft_h = mx.random.normal((1, 4, 64))
        target_h = mx.random.normal((1, 6, 64))
        rope = DFlashRotaryEmbedding(head_dim=8)
        pos_ids = mx.arange(10).reshape(1, -1)
        pos_emb = rope(draft_h, pos_ids)
        out = attn(draft_h, target_h, pos_emb)
        assert out.shape == (1, 4, 64)

    def test_batch_size(self):
        mx.random.seed(42)
        attn = DFlashAttention(hidden_size=32, num_heads=4)
        mx.eval(attn.parameters())
        draft_h = mx.random.normal((3, 4, 32))
        target_h = mx.random.normal((3, 8, 32))
        rope = DFlashRotaryEmbedding(head_dim=8)
        pos_ids = mx.arange(12).reshape(1, -1)
        pos_emb = rope(draft_h, pos_ids)
        out = attn(draft_h, target_h, pos_emb)
        assert out.shape == (3, 4, 32)


class TestDFlashDecoderLayer:
    def test_output_shape(self):
        mx.random.seed(42)
        layer = DFlashDecoderLayer(
            hidden_size=64, num_heads=4, intermediate_size=128,
        )
        mx.eval(layer.parameters())
        draft_h = mx.random.normal((1, 8, 64))
        target_h = mx.random.normal((1, 12, 64))
        rope = DFlashRotaryEmbedding(head_dim=16)
        pos_ids = mx.arange(20).reshape(1, -1)
        pos_emb = rope(draft_h, pos_ids)
        out = layer(draft_h, target_h, pos_emb)
        assert out.shape == (1, 8, 64)

    def test_residual_connection(self):
        """Output should differ from input (MLP/attn modify it)."""
        mx.random.seed(42)
        layer = DFlashDecoderLayer(
            hidden_size=32, num_heads=4, intermediate_size=64,
        )
        mx.eval(layer.parameters())
        draft_h = mx.random.normal((1, 4, 32))
        target_h = mx.random.normal((1, 4, 32))
        rope = DFlashRotaryEmbedding(head_dim=8)
        pos_ids = mx.arange(8).reshape(1, -1)
        pos_emb = rope(draft_h, pos_ids)
        out = layer(draft_h, target_h, pos_emb)
        mx.eval(out)
        # Should not be identical to input (residual + attn + mlp changes it)
        assert not mx.array_equal(out, draft_h)


# ---------------------------------------------------------------------------
# Fake target model for integration tests
# ---------------------------------------------------------------------------

class _FakeTargetLayers(nn.Module):
    """Fake layers list for target model."""
    def __init__(self, num_layers):
        super().__init__()
        self._layers = [nn.Identity() for _ in range(num_layers)]

    def __len__(self):
        return len(self._layers)

    def __getitem__(self, idx):
        return self._layers[idx]


class _FakeTargetInner(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_layers):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, embed_dim)
        self.layers = _FakeTargetLayers(num_layers)


class FakeTargetModel(nn.Module):
    """Minimal target model with embed_tokens and lm_head."""
    def __init__(self, vocab_size=256, embed_dim=64, num_layers=16):
        super().__init__()
        self.model = _FakeTargetInner(vocab_size, embed_dim, num_layers)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)


@pytest.fixture
def fake_target():
    mx.random.seed(42)
    model = FakeTargetModel(vocab_size=256, embed_dim=64, num_layers=16)
    mx.eval(model.parameters())
    return model


@pytest.fixture
def fake_target_small():
    mx.random.seed(42)
    model = FakeTargetModel(vocab_size=128, embed_dim=32, num_layers=8)
    mx.eval(model.parameters())
    return model


# ---------------------------------------------------------------------------
# BlockDiffusionDraftModel tests
# ---------------------------------------------------------------------------

class TestBlockDiffusionDraftModel:
    def test_instantiation_defaults(self):
        draft = BlockDiffusionDraftModel(embed_dim=64, num_target_layers=16)
        assert draft.embed_dim == 64
        assert draft.block_size == 16
        assert draft.num_layers_count == 5
        assert len(draft.layers) == 5
        assert len(draft.target_layer_ids) == 5

    def test_custom_config(self):
        draft = BlockDiffusionDraftModel(
            embed_dim=128,
            num_layers=3,
            num_heads=4,
            block_size=32,
            num_target_layers=64,
            intermediate_size=256,
            num_kv_heads=2,
        )
        assert draft.block_size == 32
        assert len(draft.layers) == 3
        assert draft.layers[0].self_attn.num_kv_heads == 2

    def test_explicit_target_layer_ids(self):
        draft = BlockDiffusionDraftModel(
            embed_dim=64,
            num_layers=3,
            target_layer_ids=[2, 8, 14],
        )
        assert draft.target_layer_ids == [2, 8, 14]

    def test_attach_target(self, fake_target):
        draft = BlockDiffusionDraftModel(embed_dim=64, num_target_layers=16)
        draft.attach_target(fake_target)
        assert draft._target_embed_tokens is fake_target.model.embed_tokens
        assert draft._target_lm_head is fake_target.lm_head

    def test_no_target_raises(self):
        draft = BlockDiffusionDraftModel(embed_dim=64, num_target_layers=16)
        hidden = {i: mx.random.normal((1, 4, 64)) for i in draft.target_layer_ids}
        tokens = mx.zeros((1, 16), dtype=mx.int32)
        with pytest.raises(RuntimeError, match="Target model not attached"):
            draft(hidden, tokens)

    def test_target_layer_ids_list_property(self):
        draft = BlockDiffusionDraftModel(
            embed_dim=64, target_layer_ids=[1, 5, 10],
        )
        assert draft.target_layer_ids_list == [1, 5, 10]


class TestBlockDiffusionForwardPass:
    """Forward pass shape correctness."""

    def _make_draft(self, fake_target, block_size=16, num_layers=2, num_heads=4):
        draft = BlockDiffusionDraftModel(
            embed_dim=64,
            num_layers=num_layers,
            num_heads=num_heads,
            block_size=block_size,
            num_target_layers=16,
        )
        draft.attach_target(fake_target)
        mx.eval(draft.parameters())
        return draft

    def _make_hidden(self, draft, batch=1, seq=4, dim=64):
        return {lid: mx.random.normal((batch, seq, dim)) for lid in draft.target_layer_ids}

    def test_basic_forward(self, fake_target):
        mx.random.seed(42)
        draft = self._make_draft(fake_target, block_size=16)
        hidden = self._make_hidden(draft)
        tokens = mx.zeros((1, 16), dtype=mx.int32)
        logits = draft(hidden, tokens)
        mx.eval(logits)
        # block_size - 1 = 15 logit positions, vocab = 256
        assert logits.shape == (1, 15, 256)

    def test_b32_block_size(self, fake_target):
        mx.random.seed(42)
        draft = self._make_draft(fake_target, block_size=32)
        hidden = self._make_hidden(draft)
        tokens = mx.zeros((1, 32), dtype=mx.int32)
        logits = draft(hidden, tokens)
        mx.eval(logits)
        assert logits.shape == (1, 31, 256)

    def test_batch_forward(self, fake_target):
        mx.random.seed(42)
        draft = self._make_draft(fake_target, block_size=16)
        hidden = self._make_hidden(draft, batch=4)
        tokens = mx.zeros((4, 16), dtype=mx.int32)
        logits = draft(hidden, tokens)
        mx.eval(logits)
        assert logits.shape == (4, 15, 256)

    def test_custom_position_ids(self, fake_target):
        mx.random.seed(42)
        draft = self._make_draft(fake_target, block_size=16)
        hidden = self._make_hidden(draft, seq=4)
        tokens = mx.zeros((1, 16), dtype=mx.int32)
        # ctx_len=4, block_len=16 → total=20
        pos_ids = mx.arange(20).reshape(1, -1)
        logits = draft(hidden, tokens, position_ids=pos_ids)
        mx.eval(logits)
        assert logits.shape == (1, 15, 256)

    def test_with_attention_mask(self, fake_target):
        mx.random.seed(42)
        draft = self._make_draft(fake_target, block_size=16)
        hidden = self._make_hidden(draft, seq=4)
        tokens = mx.zeros((1, 16), dtype=mx.int32)
        # mask shape: (1, 1, q_len=16, kv_len=4+16=20)
        mask = mx.zeros((1, 1, 16, 20))
        logits = draft(hidden, tokens, mask=mask)
        mx.eval(logits)
        assert logits.shape == (1, 15, 256)

    def test_varying_context_lengths(self, fake_target):
        """Different context sequence lengths should work."""
        mx.random.seed(42)
        draft = self._make_draft(fake_target, block_size=16)
        for ctx_len in [1, 4, 16, 32]:
            hidden = self._make_hidden(draft, seq=ctx_len)
            tokens = mx.zeros((1, 16), dtype=mx.int32)
            logits = draft(hidden, tokens)
            mx.eval(logits)
            assert logits.shape == (1, 15, 256)


class TestDraftBlock:
    """Test the draft_block convenience method."""

    def _make_draft(self, fake_target, block_size=16):
        draft = BlockDiffusionDraftModel(
            embed_dim=64,
            num_layers=2,
            num_heads=4,
            block_size=block_size,
            num_target_layers=16,
        )
        draft.attach_target(fake_target)
        mx.eval(draft.parameters())
        return draft

    def _make_hidden(self, draft, batch=1, seq=4, dim=64):
        return {lid: mx.random.normal((batch, seq, dim)) for lid in draft.target_layer_ids}

    def test_draft_block_greedy(self, fake_target):
        mx.random.seed(42)
        draft = self._make_draft(fake_target, block_size=16)
        hidden = self._make_hidden(draft)
        first_tok = mx.array([5])
        result = draft.draft_block(hidden, first_tok, temperature=0.0)
        mx.eval(result)
        assert result.shape == (1, 16)
        assert result[0, 0].item() == 5

    def test_draft_block_sampling(self, fake_target):
        mx.random.seed(42)
        draft = self._make_draft(fake_target, block_size=16)
        hidden = self._make_hidden(draft)
        first_tok = mx.array([10])
        result = draft.draft_block(hidden, first_tok, temperature=1.0)
        mx.eval(result)
        assert result.shape == (1, 16)
        assert result[0, 0].item() == 10

    def test_draft_block_2d_first_token(self, fake_target):
        """first_token_id can be (B, 1)."""
        mx.random.seed(42)
        draft = self._make_draft(fake_target, block_size=16)
        hidden = self._make_hidden(draft)
        first_tok = mx.array([[7]])
        result = draft.draft_block(hidden, first_tok, temperature=0.0)
        mx.eval(result)
        assert result.shape == (1, 16)
        assert result[0, 0].item() == 7

    def test_draft_block_b32(self, fake_target):
        mx.random.seed(42)
        draft = self._make_draft(fake_target, block_size=32)
        hidden = self._make_hidden(draft)
        first_tok = mx.array([1])
        result = draft.draft_block(hidden, first_tok, temperature=0.0)
        mx.eval(result)
        assert result.shape == (1, 32)

    def test_draft_block_batch(self, fake_target):
        mx.random.seed(42)
        draft = self._make_draft(fake_target, block_size=16)
        hidden = self._make_hidden(draft, batch=2)
        first_tok = mx.array([3, 4])
        result = draft.draft_block(hidden, first_tok, temperature=0.0)
        mx.eval(result)
        assert result.shape == (2, 16)
        assert result[0, 0].item() == 3
        assert result[1, 0].item() == 4

    def test_draft_block_custom_mask_token(self, fake_target):
        mx.random.seed(42)
        draft = self._make_draft(fake_target, block_size=16)
        hidden = self._make_hidden(draft)
        first_tok = mx.array([1])
        result = draft.draft_block(
            hidden, first_tok, temperature=0.0, mask_token_id=99,
        )
        mx.eval(result)
        assert result.shape == (1, 16)


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

class TestCreateDFlashDraftModel:
    def test_basic_factory(self, fake_target):
        draft = create_dflash_draft_model(fake_target)
        assert draft.embed_dim == 64
        assert draft._target_embed_tokens is fake_target.model.embed_tokens
        assert draft._target_lm_head is fake_target.lm_head

    def test_factory_custom_params(self, fake_target):
        draft = create_dflash_draft_model(
            fake_target,
            num_layers=3,
            num_heads=4,
            block_size=32,
            intermediate_size=128,
            num_kv_heads=2,
        )
        assert draft.block_size == 32
        assert len(draft.layers) == 3

    def test_factory_explicit_layer_ids(self, fake_target):
        draft = create_dflash_draft_model(
            fake_target,
            target_layer_ids=[1, 5, 10],
        )
        assert draft.target_layer_ids == [1, 5, 10]

    def test_factory_small_target(self, fake_target_small):
        draft = create_dflash_draft_model(fake_target_small, num_layers=2, num_heads=4)
        assert draft.embed_dim == 32
        assert draft.num_target_layers == 8

    def test_factory_forward_works(self, fake_target):
        """Smoke test: factory model can do a forward pass."""
        mx.random.seed(42)
        draft = create_dflash_draft_model(
            fake_target, num_layers=2, num_heads=4, block_size=16,
        )
        mx.eval(draft.parameters())
        hidden = {
            lid: mx.random.normal((1, 4, 64))
            for lid in draft.target_layer_ids
        }
        tokens = mx.zeros((1, 16), dtype=mx.int32)
        logits = draft(hidden, tokens)
        mx.eval(logits)
        assert logits.shape == (1, 15, 256)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_layer_draft(self, fake_target):
        mx.random.seed(42)
        draft = create_dflash_draft_model(
            fake_target, num_layers=1, num_heads=4, block_size=16,
        )
        mx.eval(draft.parameters())
        hidden = {
            lid: mx.random.normal((1, 4, 64))
            for lid in draft.target_layer_ids
        }
        tokens = mx.zeros((1, 16), dtype=mx.int32)
        logits = draft(hidden, tokens)
        mx.eval(logits)
        assert logits.shape == (1, 15, 256)

    def test_context_length_one(self, fake_target):
        mx.random.seed(42)
        draft = create_dflash_draft_model(
            fake_target, num_layers=2, num_heads=4, block_size=16,
        )
        mx.eval(draft.parameters())
        hidden = {
            lid: mx.random.normal((1, 1, 64))
            for lid in draft.target_layer_ids
        }
        tokens = mx.zeros((1, 16), dtype=mx.int32)
        logits = draft(hidden, tokens)
        mx.eval(logits)
        assert logits.shape == (1, 15, 256)

    def test_block_size_2(self, fake_target):
        """Minimum useful block size."""
        mx.random.seed(42)
        draft = create_dflash_draft_model(
            fake_target, num_layers=2, num_heads=4, block_size=2,
        )
        mx.eval(draft.parameters())
        hidden = {
            lid: mx.random.normal((1, 4, 64))
            for lid in draft.target_layer_ids
        }
        tokens = mx.zeros((1, 2), dtype=mx.int32)
        logits = draft(hidden, tokens)
        mx.eval(logits)
        assert logits.shape == (1, 1, 256)

    def test_greedy_deterministic(self, fake_target):
        """Greedy draft should be deterministic across calls."""
        mx.random.seed(42)
        draft = create_dflash_draft_model(
            fake_target, num_layers=2, num_heads=4, block_size=16,
        )
        mx.eval(draft.parameters())
        hidden = {
            lid: mx.random.normal((1, 4, 64))
            for lid in draft.target_layer_ids
        }
        mx.eval(hidden)
        first_tok = mx.array([1])
        r1 = draft.draft_block(hidden, first_tok, temperature=0.0)
        r2 = draft.draft_block(hidden, first_tok, temperature=0.0)
        mx.eval(r1, r2)
        assert mx.array_equal(r1, r2)

    def test_project_target_hidden(self, fake_target):
        """_project_target_hidden returns correct shape."""
        mx.random.seed(42)
        draft = create_dflash_draft_model(
            fake_target, num_layers=2, num_heads=4,
        )
        mx.eval(draft.parameters())
        hidden = {
            lid: mx.random.normal((1, 8, 64))
            for lid in draft.target_layer_ids
        }
        projected = draft._project_target_hidden(hidden)
        mx.eval(projected)
        assert projected.shape == (1, 8, 64)
