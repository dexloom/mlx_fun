"""Per-model template dialects for the mlx_fun server.

Each dialect reshapes the canonical OpenAI/Anthropic JSON the server receives
into the shape the model's chat template expects, and parses raw model output
back into normalized tool calls. See ``base.Dialect`` for the protocol and
``base.py`` for shared parsing helpers.
"""
from __future__ import annotations

from typing import Optional

from .base import (
    Dialect,
    ParsedOutput,
    PassThroughDialect,
    ToolCall,
    extract_thinking_blocks,
    parse_json_tool_call,
    parse_json_tool_calls,
    to_tool_call_dict,
    try_parse_json_object_as_tool_call,
)
from .openai import OpenAIDialect


def _build_registry() -> dict[str, Dialect]:
    """Construct the dialect registry.

    Import errors from individual dialect modules are logged at WARNING
    instead of swallowed silently — a dialect that fails to import silently
    falls back to OpenAI passthrough at request time, which masks the real
    failure (and previously hid a dead import in minimax25.py from view).
    """
    import logging

    registry: dict[str, Dialect] = {"openai": OpenAIDialect()}

    _CANDIDATES = (
        ("qwen35", "Qwen35Dialect"),
        ("kimi", "KimiDialect"),
        ("minimax25", "Minimax25Dialect"),
        ("glm", "GLMDialect"),
        ("gemma", "GemmaDialect"),
        ("chatml", "ChatMLDialect"),
        ("minimax", "MinimaxDialect"),
    )
    for module_name, class_name in _CANDIDATES:
        try:
            module = __import__(
                f"mlx_fun.dialect.{module_name}",
                fromlist=[class_name],
            )
            registry[module_name] = getattr(module, class_name)()
        except Exception as e:
            logging.warning(
                f"dialect '{module_name}' failed to load ({type(e).__name__}: {e}); "
                f"falling back to OpenAI passthrough for affected model types"
            )
    return registry


_REGISTRY = _build_registry()


# Map mlx-lm ``model_type`` strings to dialect names. Mirrors
# ``server._MODEL_TYPE_TEMPLATES`` — when a model has a bundled jinja, it also
# gets a dialect.
_MODEL_TYPE_DIALECTS: dict[str, str] = {
    "qwen3_moe": "qwen35",
    "qwen3_next": "qwen35",
    "glm4_moe": "glm",
    "glm4_moe_lite": "glm",
    "glm_moe_dsa": "glm",
    "deepseek_v32": "glm",
    "minimax_m2": "minimax25",
    "minimax": "minimax",
    "gemma4": "gemma",
    "kimi_k25": "kimi",
}


def detect_from_model_type(model_type: Optional[str]) -> Optional[str]:
    if not model_type:
        return None
    return _MODEL_TYPE_DIALECTS.get(model_type)


def detect_from_template_content(template: Optional[str]) -> Optional[str]:
    """Sniff a Jinja template to guess its dialect.

    Ported from ``~/Sombra/sac/src/providers/mlxlm/types.rs:190-227``.
    """
    if not template:
        return None
    t = template

    # Gemma — <|turn> + <tool_call|>
    if "<|turn>" in t and "<tool_call|>" in t:
        return "gemma"

    # GLM — <arg_key>/<arg_value> or [gMASK]/<|observation|>
    if "[gMASK]" in t or "<|observation|>" in t or (
        "<arg_key>" in t and "<arg_value>" in t
    ):
        return "glm"

    # Minimax / Minimax25
    if (
        "<minimax:tool_call>" in t
        or "]~b]" in t
        or "[e~[" in t
    ):
        if "message.tool_calls" in t:
            return "minimax25"
        return "minimax"

    # ChatML / Qwen35 — both share <|im_start|>/<|im_end|>
    if "<|im_start|>" in t or "<|im_end|>" in t:
        if "<function=" in t and "<parameter=" in t:
            return "qwen35"
        return "chatml"

    # Kimi K2.6 markers
    if "<|tool_calls_section_begin|>" in t or "<|tool_call_begin|>" in t:
        return "kimi"

    return None


def resolve_dialect(
    model_type: Optional[str],
    template_content: Optional[str] = None,
) -> Dialect:
    """Resolve the dialect for a model.

    Priority:
      1. Match by ``model_type`` (keys in ``_MODEL_TYPE_DIALECTS``).
      2. Match by fingerprinting the Jinja template source.
      3. Fall back to ``OpenAIDialect`` (pass-through, no tool-text parsing).
    """
    name = detect_from_model_type(model_type)
    if name is None:
        name = detect_from_template_content(template_content)
    if name is None:
        name = "openai"
    return _REGISTRY.get(name, _REGISTRY["openai"])


def get_dialect(name: str) -> Optional[Dialect]:
    return _REGISTRY.get(name)


__all__ = [
    "Dialect",
    "OpenAIDialect",
    "ParsedOutput",
    "PassThroughDialect",
    "ToolCall",
    "detect_from_model_type",
    "detect_from_template_content",
    "extract_thinking_blocks",
    "get_dialect",
    "parse_json_tool_call",
    "parse_json_tool_calls",
    "resolve_dialect",
    "to_tool_call_dict",
    "try_parse_json_object_as_tool_call",
]


# NOTE: ``normalize_tool_call_arguments`` was deliberately removed. mlx-lm's
# ``mlx_lm.server.process_message_content`` (called inside the response
# generator before ``apply_chat_template``) already runs ``json.loads`` on
# every chat ``tool_call.function.arguments``. Decoding a second time in a
# dialect feeds ``json.loads`` an already-decoded dict and crashes the
# request with ``the JSON object must be str, bytes or bytearray, not dict``.
# Dialect ``shape_request`` implementations MUST leave ``arguments`` untouched
# unless the template uses a non-OpenAI field name that mlx-lm doesn't decode.
