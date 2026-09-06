"""Chat-template resolution, shared by ``serve`` and the probe commands.

Two resolvers with deliberately different policies:

* :func:`_resolve_chat_template` is what ``serve`` uses. It *always* produces a
  template when it can — explicit value, then the model directory's own
  ``chat_template.jinja``, then a bundled one by model type — because a broken
  or missing template in a quant makes the server useless.

* :func:`resolve_probe_chat_template` is what ``domain-probe`` and
  ``refusal-probe`` use. It never substitutes a template on its own: the
  default is the checkpoint's own, and a bundled template is only used when the
  operator asks for it by name. A probe measures the model as it is shipped, so
  silently swapping the prompt format would silently change the measurement.

Kept in its own module so the probe commands do not import ``server.py`` (and
with it Flask, the counting hooks and the whole serving stack) for two
functions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

# Map model_type to bundled template filename in src/mlx_fun/templates/
_MODEL_TYPE_TEMPLATES = {
    "gemma4": "gemma.jinja",
    "glm4_moe": "glm.jinja",
    "glm4_moe_lite": "glm_flash.jinja",
    "glm_moe_dsa": "glm51.jinja",
    "deepseek_v32": "glm.jinja",
    "minimax": "minimax.jinja",
    "minimax_m2": "minimax_25.jinja",
    "qwen3_moe": "qwen35.jinja",
    "qwen3_next": "qwen35.jinja",
    "qwen4_exp": "qwen35.jinja",
    "qwen3_5_moe": "qwen35.jinja",
}

# The literal --chat-template value that selects the bundled template.
BUNDLED = "bundled"

_TEMPLATE_DIR = Path(__file__).parent / "templates"


def _bundled_template_path(model_type: str) -> Optional[Path]:
    """Path of the bundled template for a model type, if one is registered."""
    name = _MODEL_TYPE_TEMPLATES.get(model_type)
    return None if name is None else _TEMPLATE_DIR / name


def _resolve_chat_template(
    chat_template: Optional[str],
    model_type: str,
    model_dir: Optional[Path] = None,
) -> Optional[str]:
    """Resolve chat template to a Jinja string.

    Priority:
      1. Explicit value — if it's a file path, read it; otherwise use as-is.
      2. The model directory's own ``chat_template.jinja`` (per-version-accurate
         and tracks upstream HF). Falls through if the file is missing.
      3. Bundled template by model_type — legacy fallback for quants that
         shipped without a chat_template.jinja or with a broken one.
      4. None — let the tokenizer's built-in template (if any) handle it.
    """
    if chat_template:
        p = Path(chat_template)
        if p.is_file():
            logging.info(f"Using chat template from file: {p}")
            return p.read_text()
        # Assume it's an inline Jinja string
        return chat_template

    if model_dir is not None:
        standalone = Path(model_dir) / "chat_template.jinja"
        if standalone.is_file():
            logging.info(
                f"Using model's own chat template: {standalone.name} "
                f"(model_type={model_type})"
            )
            return standalone.read_text()

    # Bundled fallback
    template_path = _bundled_template_path(model_type)
    if template_path is not None:
        if template_path.is_file():
            logging.info(
                f"Falling back to bundled chat template for {model_type}: "
                f"{template_path.name}"
            )
            return template_path.read_text()
        else:
            logging.warning(
                f"Bundled template {template_path.name} not found at {template_path}"
            )
    return None


def resolve_probe_chat_template(
    value: Optional[str], model_type: str,
) -> Optional[str]:
    """Resolve ``--chat-template`` for the probe commands.

    Narrower than :func:`_resolve_chat_template` on purpose: a probe reports on
    the checkpoint as shipped, so its own valid template is never silently
    replaced by a bundled one.

    Args:
        value: ``None`` keeps the checkpoint's own template. The literal
            ``"bundled"`` selects the bundled template for ``model_type``. A
            path to an existing file reads that file. Anything else is used
            verbatim as an inline Jinja template.
        model_type: Checkpoint ``model_type``, used only for ``"bundled"``.

    Returns:
        The Jinja template string, or ``None`` to keep the checkpoint's own.

    Raises:
        ValueError: If ``"bundled"`` is asked for and no bundled template is
            registered for this model type, or the registered file is missing.
    """
    if value is None:
        return None

    if value == BUNDLED:
        template_path = _bundled_template_path(model_type)
        if template_path is None:
            raise ValueError(
                f"No bundled chat template for model_type '{model_type}'. "
                f"Have: {', '.join(sorted(_MODEL_TYPE_TEMPLATES))}. "
                f"Pass a file path or an inline template instead."
            )
        if not template_path.is_file():
            raise ValueError(
                f"Bundled chat template for '{model_type}' is missing: {template_path}"
            )
        return template_path.read_text()

    p = Path(value)
    if p.is_file():
        return p.read_text()

    return value


def probe_chat_template_source(value: Optional[str]) -> str:
    """Name the source :func:`resolve_probe_chat_template` would read from.

    For CLI output only — ``checkpoint``, ``bundled``, ``file`` or ``inline``.
    """
    if value is None:
        return "checkpoint"
    if value == BUNDLED:
        return BUNDLED
    return "file" if Path(value).is_file() else "inline"


__all__ = [
    "BUNDLED",
    "_MODEL_TYPE_TEMPLATES",
    "_resolve_chat_template",
    "probe_chat_template_source",
    "resolve_probe_chat_template",
]
