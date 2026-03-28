"""TurboQuant KV cache compression (Google Research, arXiv 2504.19874).

PolarQuant: Haar-distributed random rotation → per-coordinate scalar
quantization exploiting concentrated Beta distribution on rotated
coordinates.  The rotation makes channel distributions more uniform,
reducing quantization error at the same bit budget.

Supports two modes controlled by ``TurboQuantConfig.quantized_sdpa``:

- **False** (Phase 1): stores rotated-quantized K/V, dequantizes +
  inverse-rotates on read, returns plain tensors for standard SDPA.
  Works with any model, no hooks needed.

- **True** (Phase 2): stores rotated-quantized K/V and returns quantized
  tuples.  Exposes ``self.bits`` / ``self.group_size`` so the attention
  dispatcher uses ``mx.quantized_matmul``.  Requires
  :func:`install_turbo_quant_sdpa` to patch the model module so that
  queries are rotated before quantized SDPA — giving both memory savings
  AND hardware-accelerated attention.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import mlx.core as mx
from mlx.utils import tree_map


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TurboQuantConfig:
    """Configuration for TurboQuant KV compression.

    Args:
        bits: Quantization bits for PolarQuant stage (2-8).
        group_size: Group size for ``mx.quantize``.  Must divide head_dim.
        seed: Random seed for rotation matrix generation.
        quantized_sdpa: If True, return quantized tuples and expose
            ``self.bits`` so attention uses ``quantized_matmul``.
            Requires :func:`install_turbo_quant_sdpa` to be called.
        max_size: If set, cap the KV cache to this many tokens using a
            sliding window.  Oldest tokens (except ``keep``) are dropped
            when the cache exceeds this size.
        keep: Number of initial tokens to always preserve when trimming
            via sliding window (e.g. BOS + system prompt start).
    """

    bits: int = 4
    group_size: int = 64
    seed: int = 0
    quantized_sdpa: bool = True
    max_size: Optional[int] = None
    keep: int = 4


# ---------------------------------------------------------------------------
# Core PolarQuant primitives
# ---------------------------------------------------------------------------


def generate_rotation_matrix(dim: int, seed: int = 0) -> mx.array:
    """Generate a Haar-distributed random orthogonal matrix.

    Uses QR decomposition of a Gaussian random matrix and fixes the sign
    convention so the result is a proper Haar sample from O(d).

    Args:
        dim: Matrix dimension (typically head_dim, e.g. 64 or 128).
        seed: Deterministic seed for reproducibility.

    Returns:
        Orthogonal matrix *Q* of shape ``(dim, dim)`` with ``Q @ Q.T ≈ I``.
    """
    key = mx.random.key(seed)
    G = mx.random.normal(shape=(dim, dim), key=key)
    # QR is CPU-only in MLX — run on the default CPU stream.
    cpu = mx.cpu
    Q, R = mx.linalg.qr(G, stream=cpu)
    sign = mx.sign(mx.diag(R, stream=cpu), stream=cpu)
    Q = mx.multiply(Q, sign[None, :], stream=cpu)
    mx.eval(Q)
    return Q


def polar_quantize(
    x: mx.array,
    rotation: mx.array,
    group_size: int = 64,
    bits: int = 4,
) -> Tuple[mx.array, mx.array, mx.array]:
    """Rotate then quantize via MLX built-in affine quantization.

    The rotation transforms the channel distribution towards uniformity,
    which lets ``mx.quantize`` achieve lower per-group error at the same
    bit budget.

    Args:
        x: Tensor of shape ``(..., dim)``.
        rotation: Orthogonal matrix ``(dim, dim)``.
        group_size: Quantization group size.
        bits: Quantization bit-width.

    Returns:
        ``(quantized_data, scales, biases)`` from ``mx.quantize``.
    """
    x_rot = x @ rotation.T
    return mx.quantize(x_rot, group_size=group_size, bits=bits)


def polar_dequantize(
    quantized_data: mx.array,
    scales: mx.array,
    biases: mx.array,
    rotation: mx.array,
    group_size: int = 64,
    bits: int = 4,
) -> mx.array:
    """Dequantize and inverse-rotate to recover the original space.

    Args:
        quantized_data, scales, biases: Output of :func:`polar_quantize`.
        rotation: The same orthogonal matrix used during quantization.
        group_size: Quantization group size.
        bits: Quantization bit-width.

    Returns:
        Reconstructed tensor in the original (unrotated) space.
    """
    x_rot = mx.dequantize(quantized_data, scales, biases, group_size=group_size, bits=bits)
    return x_rot @ rotation  # R^T inverse = R for orthogonal R


# ---------------------------------------------------------------------------
# TurboQuantKVCache
# ---------------------------------------------------------------------------


class TurboQuantKVCache:
    """KV cache with PolarQuant rotation for improved low-bit compression.

    Drop-in replacement for ``KVCache`` / ``QuantizedKVCache``.

    When ``quantized_sdpa=False``: stores rotated-quantized K/V and returns
    plain tensors (dequantized + inverse-rotated).  Compatible with standard
    ``mx.fast.scaled_dot_product_attention``.

    When ``quantized_sdpa=True``: stores rotated-quantized K/V and returns
    quantized tuples ``(data, scales, biases)``.  Exposes ``self.bits`` and
    ``self.group_size`` so the attention dispatcher routes to
    ``quantized_scaled_dot_product_attention`` (using ``mx.quantized_matmul``).
    **Requires** :func:`install_turbo_quant_sdpa` so that queries are rotated
    to match the rotated keys.
    """

    step: int = 256

    def __init__(self, config: Optional[TurboQuantConfig] = None):
        self.config = config or TurboQuantConfig()
        self._bits = self.config.bits
        self._group_size = self.config.group_size
        self._quantized_sdpa = self.config.quantized_sdpa
        self.offset: int = 0
        self._max_size = self.config.max_size
        self._keep = self.config.keep

        # Expose bits/group_size only in quantized SDPA mode so
        # base.scaled_dot_product_attention dispatches to quantized path.
        if self._quantized_sdpa:
            self.bits = self._bits
            self.group_size = self._group_size

        # Quantized storage: each is a tuple (data, scales, biases) or None
        self._keys: Optional[Tuple[mx.array, mx.array, mx.array]] = None
        self._values: Optional[Tuple[mx.array, mx.array, mx.array]] = None

        # Rotation matrices — lazily initialised on the first update.
        self._k_rotation: Optional[mx.array] = None
        self._v_rotation: Optional[mx.array] = None

    # -- helpers ----------------------------------------------------------

    def _ensure_rotations(self, k_head_dim: int, v_head_dim: int) -> None:
        if self._k_rotation is None:
            self._k_rotation = generate_rotation_matrix(k_head_dim, seed=self.config.seed)
            self._v_rotation = generate_rotation_matrix(v_head_dim, seed=self.config.seed + 1)

    @staticmethod
    def _init_quant(
        B: int,
        n_heads: int,
        n_steps: int,
        head_dim: int,
        group_size: int,
        bits: int,
        dtype,
    ) -> Tuple[mx.array, mx.array, mx.array]:
        el_per_int = 8 * mx.uint32.size // bits
        return (
            mx.zeros((B, n_heads, n_steps, head_dim // el_per_int), dtype=mx.uint32),
            mx.zeros((B, n_heads, n_steps, head_dim // group_size), dtype=dtype),
            mx.zeros((B, n_heads, n_steps, head_dim // group_size), dtype=dtype),
        )

    @staticmethod
    def _expand_quant(
        buf: Tuple[mx.array, mx.array, mx.array],
        new_steps: int,
    ) -> Tuple[mx.array, mx.array, mx.array]:
        def _expand(x: mx.array) -> mx.array:
            pad = mx.zeros(
                (*x.shape[:-2], new_steps, x.shape[-1]),
                dtype=x.dtype,
            )
            return mx.concatenate([x, pad], axis=-2)
        return tuple(_expand(t) for t in buf)

    def _sliced(
        self, buf: Tuple[mx.array, mx.array, mx.array],
    ) -> Tuple[mx.array, mx.array, mx.array]:
        return tuple(t[..., : self.offset, :] for t in buf)

    def _trim_to_window(self) -> None:
        """Drop oldest tokens to fit within ``max_size``, preserving
        the first ``keep`` tokens.

        Quantization is per-token along head_dim, so slicing along the
        sequence axis is safe — each token's scales/biases are independent.
        """
        keep = min(self._keep, self._max_size)
        recent_len = self._max_size - keep
        recent_start = self.offset - recent_len
        for label in ("_keys", "_values"):
            buf = getattr(self, label)
            trimmed = tuple(
                mx.concatenate(
                    [t[..., :keep, :], t[..., recent_start : self.offset, :]],
                    axis=-2,
                )
                for t in buf
            )
            setattr(self, label, trimmed)
        self.offset = self._max_size

    # -- core API ---------------------------------------------------------

    def update_and_fetch(
        self,
        keys: mx.array,
        values: mx.array,
    ):
        """Store rotated-quantized K/V and return cached data.

        Args:
            keys: ``(B, n_kv_heads, num_steps, k_head_dim)``
            values: ``(B, n_kv_heads, num_steps, v_head_dim)``

        Returns:
            If ``quantized_sdpa=False``: ``(keys_plain, values_plain)``
            — dequantized + inverse-rotated ``mx.array`` tensors.

            If ``quantized_sdpa=True``: ``(k_quant_tuple, v_quant_tuple)``
            — each a ``(data, scales, biases)`` tuple for ``quantized_matmul``.
        """
        B, n_kv_heads, num_steps, k_head_dim = keys.shape
        v_head_dim = values.shape[-1]
        self._ensure_rotations(k_head_dim, v_head_dim)
        prev = self.offset

        # --- buffer allocation (mirrors QuantizedKVCache) ----------------
        need_alloc = self._keys is None or (prev + num_steps) > self._keys[0].shape[-2]
        if need_alloc:
            new_steps = (self.step + num_steps - 1) // self.step * self.step
            if self._keys is not None:
                # Trim to exact offset before expanding.
                if prev % self.step != 0:
                    self._keys = self._sliced(self._keys)
                    self._values = self._sliced(self._values)
                self._keys = self._expand_quant(self._keys, new_steps)
                self._values = self._expand_quant(self._values, new_steps)
            else:
                self._keys = self._init_quant(
                    B, n_kv_heads, new_steps, k_head_dim,
                    self._group_size, self._bits, keys.dtype,
                )
                self._values = self._init_quant(
                    B, n_kv_heads, new_steps, v_head_dim,
                    self._group_size, self._bits, values.dtype,
                )

        # --- rotate + quantize incoming tokens ---------------------------
        k_rot = keys @ self._k_rotation.T
        v_rot = values @ self._v_rotation.T
        k_q = mx.quantize(k_rot, group_size=self._group_size, bits=self._bits)
        v_q = mx.quantize(v_rot, group_size=self._group_size, bits=self._bits)

        self.offset += num_steps

        # Write into pre-allocated buffers.
        for i in range(3):
            self._keys[i][..., prev : self.offset, :] = k_q[i]
            self._values[i][..., prev : self.offset, :] = v_q[i]

        # --- sliding window trim (if max_size is set) ----------------------
        if self._max_size is not None and self.offset > self._max_size:
            self._trim_to_window()

        k_sliced = self._sliced(self._keys)
        v_sliced = self._sliced(self._values)

        if self._quantized_sdpa:
            # Return quantized tuples — quantized_matmul operates in rotated
            # space.  Queries must be pre-rotated via install_turbo_quant_sdpa.
            return k_sliced, v_sliced

        # --- dequantize all + inverse-rotate for plain SDPA --------------
        k_deq = mx.dequantize(
            k_sliced[0], k_sliced[1], k_sliced[2],
            group_size=self._group_size, bits=self._bits,
        )
        v_deq = mx.dequantize(
            v_sliced[0], v_sliced[1], v_sliced[2],
            group_size=self._group_size, bits=self._bits,
        )
        k_out = k_deq @ self._k_rotation
        v_out = v_deq @ self._v_rotation
        return k_out, v_out

    # -- cache protocol ---------------------------------------------------

    def size(self) -> int:
        return self.offset

    def empty(self) -> bool:
        return self._keys is None

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n: int) -> int:
        n = min(self.offset, n)
        self.offset -= n
        return n

    @property
    def state(self):
        if self._keys is None:
            return []
        if self.offset == self._keys[0].shape[-2]:
            return self._keys, self._values
        return self._sliced(self._keys), self._sliced(self._values)

    @state.setter
    def state(self, v):
        if v is not None and v:
            self._keys, self._values = v
        else:
            self._keys = None
            self._values = None

    @property
    def meta_state(self):
        return tuple(
            map(str, (self.offset, self._bits, self._group_size, self.config.seed,
                       int(self._quantized_sdpa),
                       self._max_size if self._max_size is not None else -1,
                       self._keep))
        )

    @meta_state.setter
    def meta_state(self, v):
        vals = list(v)
        self.offset = int(vals[0])
        self._bits = int(vals[1])
        self._group_size = int(vals[2])
        self.config.seed = int(vals[3])
        if len(vals) > 4:
            self._quantized_sdpa = bool(int(vals[4]))
        if len(vals) > 5:
            ms = int(vals[5])
            self._max_size = ms if ms >= 0 else None
        if len(vals) > 6:
            self._keep = int(vals[6])

    def make_mask(self, *args, **kwargs):
        from mlx_lm.models.cache import create_attention_mask
        offset = min(self.offset, self._max_size - 1) if self._max_size else self.offset
        window = self._max_size if self._max_size else None
        return create_attention_mask(
            *args, offset=offset, window_size=window, **kwargs
        )


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def make_turbo_quant_cache(
    model,
    config: Optional[TurboQuantConfig] = None,
) -> List[TurboQuantKVCache]:
    """Create a per-layer ``TurboQuantKVCache`` list.

    Drop-in replacement for ``mlx_lm.models.cache.make_prompt_cache``.

    Args:
        model: The loaded MLX model (must have ``.layers``).
        config: TurboQuant configuration.  Defaults to 4-bit.

    Returns:
        List of caches, one per decoder layer.
    """
    cfg = config or TurboQuantConfig()
    return [TurboQuantKVCache(config=cfg) for _ in range(len(model.layers))]


# ---------------------------------------------------------------------------
# Phase 2: Quantized SDPA with query rotation
# ---------------------------------------------------------------------------
#
# All supported models (minimax, glm4_moe, qwen3_moe, qwen3_next) import
# ``scaled_dot_product_attention`` from ``mlx_lm.models.base`` as a
# module-level name.  We patch that name in the model module so attention
# transparently rotates queries before ``quantized_matmul``.
#
# Math: rotated keys are stored as K' = K @ R.T.
#   quantized_matmul(Q, K'_quantized) ≈ Q @ K'.T = Q @ R @ K.T  (wrong)
#   quantized_matmul(Q @ R.T, K'_quantized) ≈ (Q @ R.T) @ K'.T
#       = Q @ R.T @ R @ K.T = Q @ K.T  ✓   (orthogonal R)
#
# For values: SDPA computes softmax(scores) @ V.
#   quantized_matmul(scores, V'_quantized) ≈ scores @ V'.T  (transpose=False)
#   V' = V @ R_v.T  →  scores @ V'  (no transpose)
#   = scores @ (V @ R_v.T)  (wrong — we want scores @ V)
#
# So we also need to inverse-rotate the output:
#   output = (scores @ V') @ R_v  →  scores @ V @ R_v.T @ R_v = scores @ V  ✓
#
# The wrapper below handles both rotations.

# Map from model_type → mlx_lm model module name
_MODEL_MODULE_MAP: Dict[str, str] = {
    "minimax": "mlx_lm.models.minimax",
    "minimax_m2": "mlx_lm.models.minimax",
    "glm4_moe": "mlx_lm.models.glm4_moe",
    "glm4_moe_lite": "mlx_lm.models.glm4_moe_lite",
    "qwen3_moe": "mlx_lm.models.qwen3_moe",
    "qwen3_next": "mlx_lm.models.qwen3_next",
    # DeepSeek V3.2 / GLM-5 use latent-compression attention with CacheList —
    # not supported for quantized SDPA yet (Phase 1 fallback works fine).
}

# Per-module storage for the original function reference
_ORIGINAL_SDPA: Dict[str, Callable] = {}


def _make_turbo_sdpa(original_sdpa: Callable) -> Callable:
    """Create a wrapper around ``scaled_dot_product_attention`` that rotates
    queries when a ``TurboQuantKVCache`` is detected.

    For non-TurboQuant caches the wrapper is a transparent pass-through.
    """
    def _turbo_sdpa(
        queries,
        keys,
        values,
        cache,
        scale: float,
        mask,
        sinks=None,
    ):
        if not isinstance(cache, TurboQuantKVCache) or not cache._quantized_sdpa:
            return original_sdpa(queries, keys, values, cache, scale, mask, sinks=sinks)

        # Rotate queries to match rotated keys: Q' = Q @ R_k.T
        queries = queries @ cache._k_rotation.T

        # Call quantized SDPA (dispatched because cache.bits exists)
        output = original_sdpa(queries, keys, values, cache, scale, mask, sinks=sinks)

        # Inverse-rotate output to undo value rotation:
        # output was computed as softmax(Q'K'.T) @ V' where V' = V @ R_v.T
        # We need: softmax(...) @ V = output @ R_v
        output = output @ cache._v_rotation

        return output

    return _turbo_sdpa


def install_turbo_quant_sdpa(model_type: str) -> bool:
    """Patch the model module's ``scaled_dot_product_attention`` to rotate
    queries for TurboQuant quantized SDPA.

    Must be called **once** before generation when using
    ``TurboQuantConfig(quantized_sdpa=True)``.

    Args:
        model_type: The ``config["model_type"]`` string (e.g. ``"minimax"``).

    Returns:
        True if the patch was installed, False if the model type is not
        supported for quantized SDPA (Phase 1 fallback will be used).
    """
    import importlib
    import sys

    module_name = _MODEL_MODULE_MAP.get(model_type)
    if module_name is None:
        return False

    if module_name in _ORIGINAL_SDPA:
        # Already patched
        return True

    mod = sys.modules.get(module_name)
    if mod is None:
        mod = importlib.import_module(module_name)

    original = getattr(mod, "scaled_dot_product_attention")
    _ORIGINAL_SDPA[module_name] = original
    setattr(mod, "scaled_dot_product_attention", _make_turbo_sdpa(original))
    return True


def remove_turbo_quant_sdpa(model_type: str) -> None:
    """Restore the original ``scaled_dot_product_attention`` in the model module.

    Safe to call even if the patch was never installed.
    """
    import sys

    module_name = _MODEL_MODULE_MAP.get(model_type)
    if module_name is None or module_name not in _ORIGINAL_SDPA:
        return

    mod = sys.modules.get(module_name)
    if mod is not None:
        setattr(mod, "scaled_dot_product_attention", _ORIGINAL_SDPA.pop(module_name))


def setup_turbo_quant(
    model,
    model_type: str,
    config: Optional[TurboQuantConfig] = None,
) -> Tuple[List[TurboQuantKVCache], bool]:
    """One-call setup: create caches, install SDPA patch if needed.

    Args:
        model: The loaded MLX model.
        model_type: The ``config["model_type"]`` string.
        config: TurboQuant configuration.

    Returns:
        ``(caches, sdpa_patched)`` — the per-layer cache list and whether
        quantized SDPA was installed.
    """
    cfg = config or TurboQuantConfig()
    sdpa_patched = False

    if cfg.quantized_sdpa:
        sdpa_patched = install_turbo_quant_sdpa(model_type)
        if not sdpa_patched:
            # Fall back to plain SDPA for unsupported model types
            cfg = TurboQuantConfig(
                bits=cfg.bits,
                group_size=cfg.group_size,
                seed=cfg.seed,
                quantized_sdpa=False,
                max_size=cfg.max_size,
                keep=cfg.keep,
            )

    caches = make_turbo_quant_cache(model, cfg)
    return caches, sdpa_patched
