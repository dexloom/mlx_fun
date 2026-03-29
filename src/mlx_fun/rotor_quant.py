"""RotorQuant KV cache compression via Clifford algebra rotors.

Replaces TurboQuant's dense d×d orthogonal rotation matrices with Cl(3,0)
Clifford rotors (4 parameters per 3D group instead of d²), achieving 44×
fewer parameters while matching compression fidelity.

Algorithm:
  1. Chunk d-dimensional K/V vectors into groups of 3
  2. Embed each group as a Cl(3,0) grade-1 multivector
  3. Apply per-group rotor sandwich R v R̃ for decorrelation
  4. Quantize grade-1 components via precomputed Lloyd-Max codebook
  5. On read: look up centroids, undo rotor rotation, reconstruct vectors

Based on `RotorQuant <https://www.scrya.com/rotorquant/>`_ (Scrya, 2026),
which builds on `TurboQuant <https://arxiv.org/abs/2504.19874>`_ (Google
Research, ICLR 2026).

Plain SDPA mode only — codebook indices are not compatible with
``mx.quantized_matmul``, so keys/values are dequantized on every read.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import math

import mlx.core as mx


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class RotorQuantConfig:
    """Configuration for RotorQuant KV compression.

    Args:
        bits: Quantization bits for Lloyd-Max codebook (2-8).
        seed: Random seed for rotor generation.
        max_size: If set, cap the KV cache to this many tokens using a
            sliding window.
        keep: Number of initial tokens to preserve when trimming.
    """

    bits: int = 3
    seed: int = 0
    max_size: Optional[int] = None
    keep: int = 4


# ---------------------------------------------------------------------------
# Cl(3,0) Geometric Algebra primitives
# ---------------------------------------------------------------------------

# Multivector basis: [1, e1, e2, e3, e12, e13, e23, e123]
#                     g0   g1           g2             g3
MV_DIM = 8
S, E1, E2, E3, E12, E13, E23, E123 = range(8)

# Reversion signs: grade 0,1 → +1;  grade 2,3 → -1
_REV_SIGNS = mx.array([1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0])


def _geometric_product(a: mx.array, b: mx.array) -> mx.array:
    """Full Cl(3,0) geometric product  a * b.

    Args:
        a, b: ``(..., 8)`` multivectors.

    Returns:
        Product multivector ``(..., 8)``.
    """
    a0 = a[..., 0]; a1 = a[..., 1]; a2 = a[..., 2]; a3 = a[..., 3]
    a12 = a[..., 4]; a13 = a[..., 5]; a23 = a[..., 6]; a123 = a[..., 7]
    b0 = b[..., 0]; b1 = b[..., 1]; b2 = b[..., 2]; b3 = b[..., 3]
    b12 = b[..., 4]; b13 = b[..., 5]; b23 = b[..., 6]; b123 = b[..., 7]

    # Grade 0 (scalar)
    r0 = (a0*b0 + a1*b1 + a2*b2 + a3*b3
           - a12*b12 - a13*b13 - a23*b23 - a123*b123)
    # Grade 1 (vectors)
    r1 = (a0*b1 + a1*b0 - a2*b12 + a12*b2 - a3*b13 + a13*b3
           + a23*b123 + a123*b23)
    r2 = (a0*b2 + a2*b0 + a1*b12 - a12*b1 - a3*b23 + a23*b3
           - a13*b123 - a123*b13)
    r3 = (a0*b3 + a3*b0 + a1*b13 - a13*b1 + a2*b23 - a23*b2
           + a12*b123 + a123*b12)
    # Grade 2 (bivectors)
    r12 = (a0*b12 + a12*b0 + a1*b2 - a2*b1 + a13*b23 - a23*b13
            + a3*b123 - a123*b3)
    r13 = (a0*b13 + a13*b0 + a1*b3 - a3*b1 - a12*b23 + a23*b12
            - a2*b123 + a123*b2)
    r23 = (a0*b23 + a23*b0 + a2*b3 - a3*b2 + a12*b13 - a13*b12
            + a1*b123 - a123*b1)
    # Grade 3 (pseudoscalar)
    r123 = (a0*b123 + a123*b0 + a1*b23 - a23*b1 - a2*b13 + a13*b2
             + a3*b12 - a12*b3)

    return mx.stack([r0, r1, r2, r3, r12, r13, r23, r123], axis=-1)


def _reverse(x: mx.array) -> mx.array:
    """Clifford reversion x̃ — negate grade 2 and 3 components."""
    return x * _REV_SIGNS


def _rotor_sandwich(rotor: mx.array, x: mx.array) -> mx.array:
    """Apply rotor sandwich R x R̃ — rotate x preserving algebraic structure."""
    return _geometric_product(_geometric_product(rotor, x), _reverse(rotor))


def _make_random_rotor(seed: int = 0) -> mx.array:
    """Generate a single normalized random Cl(3,0) rotor.

    R = cos(θ/2) + sin(θ/2) * B̂  where B̂ is a unit bivector.

    Returns:
        Rotor multivector of shape ``(8,)``.
    """
    key = mx.random.key(seed)
    k1, k2 = mx.random.split(key)
    # Random bivector direction (3 components for e12, e13, e23)
    bv = mx.random.normal(shape=(3,), key=k1)
    bv_norm = mx.maximum(mx.sqrt(mx.sum(bv * bv)), mx.array(1e-8))
    bv_hat = bv / bv_norm
    # Random angle in [0, 2π)
    angle = mx.random.uniform(shape=(), key=k2) * 2.0 * math.pi
    half = angle / 2.0
    cos_ha = mx.cos(half)
    sin_ha = mx.sin(half)

    rotor = mx.zeros((8,))
    rotor = rotor.at[S].add(cos_ha)
    rotor = rotor.at[E12].add(sin_ha * bv_hat[0])
    rotor = rotor.at[E13].add(sin_ha * bv_hat[1])
    rotor = rotor.at[E23].add(sin_ha * bv_hat[2])
    # Normalize
    rev = _reverse(rotor)
    norm_sq = _geometric_product(rotor, rev)[0]  # scalar part
    norm = mx.maximum(mx.sqrt(mx.abs(norm_sq)), mx.array(1e-8))
    rotor = rotor / norm
    mx.eval(rotor)
    return rotor


def _embed_vectors(v: mx.array) -> mx.array:
    """Embed d-dimensional vectors as Cl(3,0) grade-1 multivectors.

    Args:
        v: ``(..., d)`` input vectors.

    Returns:
        ``(..., n_groups, 8)`` multivectors with grade-1 components set.
    """
    d = v.shape[-1]
    pad = (3 - d % 3) % 3
    if pad > 0:
        v = mx.pad(v, [(0, 0)] * (v.ndim - 1) + [(0, pad)])
    n_groups = v.shape[-1] // 3
    grouped = v.reshape(*v.shape[:-1], n_groups, 3)
    mv = mx.zeros((*grouped.shape[:-1], 8), dtype=v.dtype)
    mv = mv.at[..., E1].add(grouped[..., 0])
    mv = mv.at[..., E2].add(grouped[..., 1])
    mv = mv.at[..., E3].add(grouped[..., 2])
    return mv


def _extract_vectors(mv: mx.array, orig_dim: int) -> mx.array:
    """Extract d-dimensional vectors from Cl(3,0) multivectors.

    Args:
        mv: ``(..., n_groups, 8)`` multivectors.
        orig_dim: Original vector dimension.

    Returns:
        ``(..., orig_dim)`` reconstructed vectors.
    """
    v = mx.stack([mv[..., E1], mv[..., E2], mv[..., E3]], axis=-1)
    v = v.reshape(*mv.shape[:-2], -1)
    return v[..., :orig_dim]


# ---------------------------------------------------------------------------
# Lloyd-Max codebook (precomputed via scipy, runtime uses MLX)
# ---------------------------------------------------------------------------


def _gauss_quad(f, a: float, b: float, n: int = 64) -> float:
    """Simple Gauss-Legendre quadrature (no scipy dependency)."""
    # Map [a,b] to [-1,1] and use midpoint composite rule
    h = (b - a) / n
    total = 0.0
    for i in range(n):
        x = a + (i + 0.5) * h
        total += f(x)
    return total * h


def _solve_lloyd_max(d: int, bits: int) -> mx.array:
    """Compute Lloyd-Max optimal centroids for the coordinate distribution
    arising from random rotation of d-dimensional unit vectors.

    Uses Gaussian approximation N(0, 1/d) which is accurate for d >= 64.
    Pure Python — no scipy dependency.

    Returns:
        ``mx.array`` of shape ``(2^bits,)`` with sorted centroids.
    """
    n_levels = 2 ** bits
    sigma = 1.0 / math.sqrt(d)
    inv_2s2 = 1.0 / (2.0 * sigma * sigma)
    coeff = 1.0 / math.sqrt(2.0 * math.pi * sigma * sigma)

    def pdf(x):
        return coeff * math.exp(-x * x * inv_2s2)

    lo, hi = -3.5 * sigma, 3.5 * sigma
    centroids = [lo + (hi - lo) * (i + 0.5) / n_levels for i in range(n_levels)]

    for _ in range(200):
        boundaries = [(centroids[i] + centroids[i + 1]) / 2.0
                      for i in range(n_levels - 1)]
        edges = [lo * 3] + boundaries + [hi * 3]
        new_centroids = []
        for i in range(n_levels):
            a, b = edges[i], edges[i + 1]
            num = _gauss_quad(lambda x: x * pdf(x), a, b)
            den = _gauss_quad(pdf, a, b)
            new_centroids.append(num / den if den > 1e-15 else centroids[i])
        if max(abs(new_centroids[i] - centroids[i]) for i in range(n_levels)) < 1e-10:
            break
        centroids = new_centroids

    return mx.array(centroids, dtype=mx.float32)


# ---------------------------------------------------------------------------
# RotorQuantKVCache
# ---------------------------------------------------------------------------


class RotorQuantKVCache:
    """KV cache with Clifford rotor decorrelation + Lloyd-Max quantization.

    Drop-in replacement for ``TurboQuantKVCache`` / ``KVCache``.

    Always operates in plain SDPA mode (dequantize on read) since codebook
    indices are not compatible with ``mx.quantized_matmul``.
    """

    step: int = 256

    def __init__(self, config: Optional[RotorQuantConfig] = None):
        self.config = config or RotorQuantConfig()
        self._bits = self.config.bits
        self._max_size = self.config.max_size
        self._keep = self.config.keep
        self.offset: int = 0

        # Lazy-initialized per head_dim
        self._k_rotors: Optional[mx.array] = None  # (n_groups, 8)
        self._v_rotors: Optional[mx.array] = None
        self._centroids: Optional[mx.array] = None  # (n_levels,)
        self._k_head_dim: int = 0
        self._v_head_dim: int = 0

        # Quantized storage: indices (uint8) and norms (float16)
        self._k_indices: Optional[mx.array] = None
        self._v_indices: Optional[mx.array] = None
        self._k_norms: Optional[mx.array] = None
        self._v_norms: Optional[mx.array] = None

    # -- lazy init -----------------------------------------------------------

    def _ensure_init(self, k_head_dim: int, v_head_dim: int) -> None:
        if self._k_rotors is not None:
            return
        self._k_head_dim = k_head_dim
        self._v_head_dim = v_head_dim
        k_groups = (k_head_dim + 2) // 3
        v_groups = (v_head_dim + 2) // 3
        # Generate per-group rotors
        self._k_rotors = mx.stack(
            [_make_random_rotor(seed=self.config.seed + i) for i in range(k_groups)]
        )
        self._v_rotors = mx.stack(
            [_make_random_rotor(seed=self.config.seed + k_groups + i)
             for i in range(v_groups)]
        )
        # Precompute Lloyd-Max codebook (uses the larger dim for distribution)
        dim_for_codebook = max(k_head_dim, v_head_dim)
        self._centroids = _solve_lloyd_max(dim_for_codebook, self._bits)
        mx.eval(self._k_rotors, self._v_rotors, self._centroids)

    # -- quantize / dequantize -----------------------------------------------

    def _quantize_vectors(self, x: mx.array, rotors: mx.array,
                          head_dim: int) -> Tuple[mx.array, mx.array]:
        """Quantize vectors via rotor sandwich + Lloyd-Max codebook.

        Args:
            x: ``(B, n_heads, num_steps, head_dim)``
            rotors: ``(n_groups, 8)``
            head_dim: Original head dimension.

        Returns:
            ``(indices, norms)`` where indices is ``uint8`` and norms is
            the original dtype.
        """
        orig_shape = x.shape  # (B, n_heads, S, D)
        # Compute and factor out norms
        norms = mx.maximum(mx.sqrt(mx.sum(x * x, axis=-1, keepdims=True)), mx.array(1e-8))
        x_unit = x / norms

        # Embed as multivectors: (..., n_groups, 8)
        mv = _embed_vectors(x_unit)
        # Apply per-group rotor sandwich: broadcast rotors over batch dims
        mv_rot = _rotor_sandwich(rotors, mv)

        # Extract grade-1 components for quantization: (..., n_groups, 3)
        grade1 = mx.stack([mv_rot[..., E1], mv_rot[..., E2], mv_rot[..., E3]],
                          axis=-1)
        # Flatten groups: (..., n_groups * 3)
        flat = grade1.reshape(*orig_shape[:-1], -1)

        # Nearest-centroid quantization
        # flat: (..., n_comp), centroids: (n_levels,)
        diffs = mx.abs(mx.expand_dims(flat, axis=-1) - self._centroids)
        indices = mx.argmin(diffs, axis=-1).astype(mx.uint8)

        return indices, norms.squeeze(-1)

    def _dequantize_vectors(self, indices: mx.array, norms: mx.array,
                            rotors: mx.array, head_dim: int) -> mx.array:
        """Reconstruct vectors from codebook indices + norms.

        Args:
            indices: ``(B, n_heads, S, n_components)`` uint8
            norms: ``(B, n_heads, S)``
            rotors: ``(n_groups, 8)``
            head_dim: Original head dimension.

        Returns:
            Reconstructed ``(B, n_heads, S, head_dim)`` tensors.
        """
        # Look up centroids
        values = self._centroids[indices.astype(mx.int32)]  # (..., n_comp)
        n_groups = (head_dim + 2) // 3

        # Reshape to groups of 3
        grade1 = values.reshape(*values.shape[:-1], n_groups, 3)

        # Reconstruct multivectors with grade-1 only
        mv_q = mx.zeros((*grade1.shape[:-1], 8), dtype=values.dtype)
        mv_q = mv_q.at[..., E1].add(grade1[..., 0])
        mv_q = mv_q.at[..., E2].add(grade1[..., 1])
        mv_q = mv_q.at[..., E3].add(grade1[..., 2])

        # Undo rotor rotation: R̃ x R
        rotor_rev = _reverse(rotors)
        mv_recon = _rotor_sandwich(rotor_rev, mv_q)

        # Extract and rescale
        x_hat = _extract_vectors(mv_recon, head_dim)
        return x_hat * mx.expand_dims(norms, axis=-1)

    # -- buffer management ---------------------------------------------------

    def _ensure_buffers(self, B: int, n_heads: int, num_steps: int,
                        k_n_comp: int, v_n_comp: int, dtype) -> None:
        """Allocate or expand index/norm buffers."""
        need = self._k_indices is None or (self.offset + num_steps) > self._k_indices.shape[2]
        if not need:
            return
        new_steps = (self.step + num_steps - 1) // self.step * self.step
        if self._k_indices is not None:
            # Trim to exact offset then expand
            if self.offset % self.step != 0:
                self._k_indices = self._k_indices[:, :, :self.offset]
                self._v_indices = self._v_indices[:, :, :self.offset]
                self._k_norms = self._k_norms[:, :, :self.offset]
                self._v_norms = self._v_norms[:, :, :self.offset]
            self._k_indices = mx.concatenate(
                [self._k_indices,
                 mx.zeros((B, n_heads, new_steps, k_n_comp), dtype=mx.uint8)],
                axis=2)
            self._v_indices = mx.concatenate(
                [self._v_indices,
                 mx.zeros((B, n_heads, new_steps, v_n_comp), dtype=mx.uint8)],
                axis=2)
            self._k_norms = mx.concatenate(
                [self._k_norms,
                 mx.zeros((B, n_heads, new_steps), dtype=dtype)], axis=2)
            self._v_norms = mx.concatenate(
                [self._v_norms,
                 mx.zeros((B, n_heads, new_steps), dtype=dtype)], axis=2)
        else:
            self._k_indices = mx.zeros((B, n_heads, new_steps, k_n_comp), dtype=mx.uint8)
            self._v_indices = mx.zeros((B, n_heads, new_steps, v_n_comp), dtype=mx.uint8)
            self._k_norms = mx.zeros((B, n_heads, new_steps), dtype=dtype)
            self._v_norms = mx.zeros((B, n_heads, new_steps), dtype=dtype)

    # -- core API ------------------------------------------------------------

    def update_and_fetch(
        self,
        keys: mx.array,
        values: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        """Store rotor-quantized K/V and return dequantized cached data.

        Args:
            keys: ``(B, n_kv_heads, num_steps, k_head_dim)``
            values: ``(B, n_kv_heads, num_steps, v_head_dim)``

        Returns:
            ``(keys_plain, values_plain)`` — dequantized ``mx.array`` tensors.
        """
        B, n_heads, num_steps, k_head_dim = keys.shape
        v_head_dim = values.shape[-1]
        self._ensure_init(k_head_dim, v_head_dim)

        k_n_comp = ((k_head_dim + 2) // 3) * 3
        v_n_comp = ((v_head_dim + 2) // 3) * 3
        self._ensure_buffers(B, n_heads, num_steps, k_n_comp, v_n_comp, keys.dtype)

        prev = self.offset

        # Quantize incoming tokens
        k_idx, k_norms = self._quantize_vectors(keys, self._k_rotors, k_head_dim)
        v_idx, v_norms = self._quantize_vectors(values, self._v_rotors, v_head_dim)

        self.offset += num_steps

        # Write into buffers
        self._k_indices[:, :, prev:self.offset] = k_idx
        self._v_indices[:, :, prev:self.offset] = v_idx
        self._k_norms[:, :, prev:self.offset] = k_norms
        self._v_norms[:, :, prev:self.offset] = v_norms

        # Sliding window trim
        if self._max_size is not None and self.offset > self._max_size:
            self._trim_to_window()

        # Dequantize all cached tokens
        k_out = self._dequantize_vectors(
            self._k_indices[:, :, :self.offset],
            self._k_norms[:, :, :self.offset],
            self._k_rotors, k_head_dim,
        )
        v_out = self._dequantize_vectors(
            self._v_indices[:, :, :self.offset],
            self._v_norms[:, :, :self.offset],
            self._v_rotors, v_head_dim,
        )
        return k_out, v_out

    # -- sliding window ------------------------------------------------------

    def _trim_to_window(self) -> None:
        """Drop oldest tokens to fit within ``max_size``, preserving ``keep``."""
        keep = min(self._keep, self._max_size)
        recent_start = self.offset - (self._max_size - keep)
        for attr in ("_k_indices", "_v_indices"):
            buf = getattr(self, attr)
            trimmed = mx.concatenate(
                [buf[:, :, :keep], buf[:, :, recent_start:self.offset]],
                axis=2,
            )
            setattr(self, attr, trimmed)
        for attr in ("_k_norms", "_v_norms"):
            buf = getattr(self, attr)
            trimmed = mx.concatenate(
                [buf[:, :, :keep], buf[:, :, recent_start:self.offset]],
                axis=2,
            )
            setattr(self, attr, trimmed)
        self.offset = self._max_size

    # -- cache protocol ------------------------------------------------------

    def size(self) -> int:
        return self.offset

    def empty(self) -> bool:
        return self._k_indices is None

    def is_trimmable(self) -> bool:
        return True

    def trim(self, n: int) -> int:
        n = min(self.offset, n)
        self.offset -= n
        return n

    @property
    def state(self):
        if self._k_indices is None:
            return []
        s = self.offset
        return (
            self._k_indices[:, :, :s],
            self._v_indices[:, :, :s],
            self._k_norms[:, :, :s],
            self._v_norms[:, :, :s],
        )

    @state.setter
    def state(self, v):
        if v is not None and v:
            self._k_indices, self._v_indices, self._k_norms, self._v_norms = v
        else:
            self._k_indices = None
            self._v_indices = None
            self._k_norms = None
            self._v_norms = None

    @property
    def meta_state(self):
        return tuple(
            map(str, (self.offset, self._bits, self.config.seed,
                       self._k_head_dim, self._v_head_dim,
                       self._max_size if self._max_size is not None else -1,
                       self._keep))
        )

    @meta_state.setter
    def meta_state(self, v):
        vals = list(v)
        self.offset = int(vals[0])
        self._bits = int(vals[1])
        self.config.seed = int(vals[2])
        if len(vals) > 3:
            self._k_head_dim = int(vals[3])
            self._v_head_dim = int(vals[4])
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


def make_rotor_quant_cache(
    model,
    config: Optional[RotorQuantConfig] = None,
) -> List[RotorQuantKVCache]:
    """Create a per-layer ``RotorQuantKVCache`` list.

    Drop-in replacement for ``mlx_lm.models.cache.make_prompt_cache``.
    """
    cfg = config or RotorQuantConfig()
    return [RotorQuantKVCache(config=cfg) for _ in range(len(model.layers))]


def setup_rotor_quant(
    model,
    config: Optional[RotorQuantConfig] = None,
) -> List[RotorQuantKVCache]:
    """One-call setup: create caches.

    Unlike TurboQuant, no SDPA patching is needed — RotorQuant always
    operates in plain SDPA mode (dequantize on read).

    Args:
        model: The loaded MLX model.
        config: RotorQuant configuration.

    Returns:
        Per-layer cache list.
    """
    return make_rotor_quant_cache(model, config)
