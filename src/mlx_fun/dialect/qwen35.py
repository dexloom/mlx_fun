"""Qwen3.5 / Qwen3-Next dialect.

The bundled ``templates/qwen35.jinja`` iterates ``tool_call.arguments | items``,
which needs ``arguments`` as a dict. mlx-lm's
``server.py:process_message_content`` already runs ``json.loads`` on the
incoming JSON-string and converts it to a dict before
``apply_chat_template``, so the dialect's ``shape_request`` is pass-through —
decoding again here would feed ``json.loads`` an already-decoded dict and
crash with ``the JSON object must be str, bytes or bytearray, not dict``.

Output parsing extracts the model's
``<tool_call><function=name><parameter=k>v</parameter></function></tool_call>``
blocks. ``<think>`` extraction is left to ``extract_thinking_blocks``.

Reference: ``~/Sombra/sac/src/providers/mlxlm/client.rs:1718-1955`` (parse).
"""
from __future__ import annotations

import json
from typing import Any, Optional

from .base import (
    PassThroughDialect,
    ToolCall,
    parse_json_tool_call,
    to_tool_call_dict,
)


class Qwen35Dialect(PassThroughDialect):
    name = "qwen35"

    # shape_request is inherited from PassThroughDialect — mlx-lm's
    # process_message_content already JSON-decodes tool_call arguments.

    def parse_output(
        self, text: str, tools: Optional[list] = None
    ) -> list[dict]:
        _, tool_calls = _parse_qwen35_content(text)
        return [to_tool_call_dict(tc) for tc in tool_calls]


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

_TC_OPEN = "<tool_call>"
_TC_CLOSE = "</tool_call>"
_FN_OPEN = "<function="
_PARAM_OPEN = "<parameter="
_PARAM_CLOSE = "</parameter>"


def _parse_qwen35_content(content: str) -> tuple[str, list[ToolCall]]:
    """Extract ``<tool_call>...</tool_call>`` blocks from raw model output."""
    tool_calls: list[ToolCall] = []
    text_parts: list[str] = []
    remaining = content
    counter = 0

    while True:
        start = remaining.find(_TC_OPEN)
        if start < 0:
            break
        before = remaining[:start]
        if before.strip():
            text_parts.append(before.strip())
        end = remaining.find(_TC_CLOSE, start)
        if end < 0:
            # unterminated block — bail
            break
        body = remaining[start + len(_TC_OPEN):end].strip()
        tc = _parse_qwen35_tool_call(body, counter)
        if tc is None:
            # Fall back to JSON-style payload inside <tool_call>...</tool_call>
            tc = parse_json_tool_call(body, counter)
        if tc is not None:
            tool_calls.append(tc)
            counter += 1
        remaining = remaining[end + len(_TC_CLOSE):]

    if remaining.strip():
        text_parts.append(remaining.strip())

    return "\n".join(text_parts), tool_calls


def _parse_qwen35_tool_call(body: str, index: int) -> Optional[ToolCall]:
    """Parse ``<function=name><parameter=k>v</parameter>...</function>``."""
    fn_pos = body.find(_FN_OPEN)
    if fn_pos < 0:
        return None
    name_start = fn_pos + len(_FN_OPEN)
    name_end_rel = body[name_start:].find(">")
    if name_end_rel < 0:
        return None
    name_end = name_start + name_end_rel
    name = body[name_start:name_end].strip()
    if not name:
        return None

    params: dict[str, Any] = {}
    search_start = name_end + 1
    while True:
        p_pos = body.find(_PARAM_OPEN, search_start)
        if p_pos < 0:
            break
        key_start = p_pos + len(_PARAM_OPEN)
        key_end_rel = body[key_start:].find(">")
        if key_end_rel < 0:
            break
        key_end = key_start + key_end_rel
        key = body[key_start:key_end].strip()
        value_start = key_end + 1
        v_end = body.find(_PARAM_CLOSE, value_start)
        if v_end < 0:
            break
        raw_value = body[value_start:v_end]
        # Template emits a leading/trailing newline around the value; strip one
        # of each but preserve interior whitespace.
        if raw_value.startswith("\n"):
            raw_value = raw_value[1:]
        if raw_value.endswith("\n"):
            raw_value = raw_value[:-1]
        # Try JSON-decode (numbers, bools, nested objects); fall back to string.
        try:
            parsed = json.loads(raw_value.strip())
        except (json.JSONDecodeError, ValueError):
            parsed = raw_value
        params[key] = parsed
        search_start = v_end + len(_PARAM_CLOSE)

    return ToolCall(
        id=f"call_{index}",
        name=name,
        arguments=json.dumps(params),
    )
