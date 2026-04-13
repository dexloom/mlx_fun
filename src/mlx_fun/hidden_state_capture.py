"""Hidden state capture hooks for speculative decoding.

Captures intermediate hidden states from decoder layers during forward pass.
Uses the same __class__ swapping pattern as observer.py and abliterate.py.

Phase 2 of speculative decoding: these captured hidden states will be passed
to the draft model in Phase 3 to enable layer-aligned speculative decoding.
"""

from typing import Dict, List, Optional, Sequence, Union

import mlx.core as mx


# ---------------------------------------------------------------------------
# Hook function
# ---------------------------------------------------------------------------

def _capture_output_call(self, x, *args, **kwargs):
    """Replacement __call__ for a decoder layer that captures its output.

    Unlike abliterate hooks (which capture *input*), this captures the
    *output* hidden state — what gets fed into the next layer.
    """
    out = self.__class__.__bases__[0].__call__(self, x, *args, **kwargs)
    mx.eval(out)
    self._hidden_state_capture.append(out)
    return out


# ---------------------------------------------------------------------------
# HiddenStateCapture
# ---------------------------------------------------------------------------

class HiddenStateCapture:
    """Captures hidden states from specified decoder layers during forward pass.

    Installs lightweight hooks on decoder layers that store the output tensor
    after each forward call.  Captures are stored as ``mx.array`` (not numpy)
    because Phase 3 will feed them directly to the draft model.

    Usage::

        capture = HiddenStateCapture(model, layer_indices=[0, 4, 8])
        capture.install()

        model(tokens)                       # forward pass populates captures
        states = capture.collect()          # {0: [mx.array], 4: [...], ...}

        capture.clear()                     # reset for next forward pass
        capture.remove()                    # restore original classes

    Args:
        model: MLX model with ``model.model.layers`` attribute.
        layer_indices: Which decoder layer indices to hook.  Pass ``None``
            to hook **all** layers.
    """

    def __init__(
        self,
        model,
        layer_indices: Optional[Sequence[int]] = None,
    ):
        self.model = model
        layers = model.model.layers
        if layer_indices is None:
            self.layer_indices = list(range(len(layers)))
        else:
            self.layer_indices = list(layer_indices)
        self._hooked_layers: List = []
        self._installed = False

    # -- lifecycle -----------------------------------------------------------

    def install(self) -> None:
        """Install capture hooks on the target decoder layers."""
        if self._installed:
            return
        layers = self.model.model.layers
        for idx in self.layer_indices:
            layer = layers[idx]
            layer._hidden_state_capture = []
            layer._hidden_state_capture_idx = idx
            original_cls = type(layer)
            layer._hidden_state_original_cls = original_cls
            hooked_cls = type(
                f"_HiddenCapture_{original_cls.__name__}",
                (original_cls,),
                {"__call__": _capture_output_call},
            )
            layer.__class__ = hooked_cls
            self._hooked_layers.append(layer)
        self._installed = True

    def remove(self) -> None:
        """Remove hooks and restore original layer classes."""
        for layer in self._hooked_layers:
            if hasattr(layer, "_hidden_state_original_cls"):
                layer.__class__ = layer._hidden_state_original_cls
                delattr(layer, "_hidden_state_original_cls")
            if hasattr(layer, "_hidden_state_capture"):
                delattr(layer, "_hidden_state_capture")
            if hasattr(layer, "_hidden_state_capture_idx"):
                delattr(layer, "_hidden_state_capture_idx")
        self._hooked_layers.clear()
        self._installed = False

    # -- data access ---------------------------------------------------------

    def collect(self) -> Dict[int, List[mx.array]]:
        """Return captured hidden states keyed by layer index.

        Each value is a list of ``mx.array`` tensors with shape
        ``(batch, seq_len, hidden_dim)`` — one per forward call since the
        last :meth:`clear`.
        """
        result: Dict[int, List[mx.array]] = {}
        for layer in self._hooked_layers:
            idx = layer._hidden_state_capture_idx
            result[idx] = list(getattr(layer, "_hidden_state_capture", []))
        return result

    def collect_latest(self) -> Dict[int, mx.array]:
        """Return only the most recent capture per layer.

        Convenience for the common case where a single prefill was run.
        Returns an empty dict for layers with no captures.
        """
        result: Dict[int, mx.array] = {}
        for layer in self._hooked_layers:
            idx = layer._hidden_state_capture_idx
            caps = getattr(layer, "_hidden_state_capture", [])
            if caps:
                result[idx] = caps[-1]
        return result

    def clear(self) -> None:
        """Clear all captured data without removing hooks."""
        for layer in self._hooked_layers:
            layer._hidden_state_capture = []

    # -- introspection -------------------------------------------------------

    @property
    def installed(self) -> bool:
        return self._installed

    @property
    def num_layers(self) -> int:
        return len(self.layer_indices)

    def __repr__(self) -> str:
        status = "installed" if self._installed else "not installed"
        return (
            f"HiddenStateCapture(layers={self.layer_indices}, {status})"
        )


# ---------------------------------------------------------------------------
# Utility: parse --capture-layers CLI arg
# ---------------------------------------------------------------------------

def parse_capture_layers(
    value: Optional[str],
    num_model_layers: int,
) -> Optional[List[int]]:
    """Parse a ``--capture-layers`` CLI string into layer indices.

    Args:
        value: ``None`` (disabled), ``"all"``, or comma-separated ints
            (e.g. ``"0,4,8,12"``).
        num_model_layers: Total number of decoder layers in the model.

    Returns:
        List of layer indices, or ``None`` if capture is disabled.

    Raises:
        ValueError: If any index is out of range.
    """
    if value is None:
        return None
    value = value.strip()
    if value.lower() == "all":
        return list(range(num_model_layers))
    indices = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        idx = int(part)
        if idx < 0 or idx >= num_model_layers:
            raise ValueError(
                f"Layer index {idx} out of range [0, {num_model_layers})"
            )
        indices.append(idx)
    if not indices:
        return None
    return sorted(set(indices))
