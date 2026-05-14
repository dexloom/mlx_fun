"""GLM-4.x / GLM-5 / DeepSeek V3.2 dialect.

The bundled ``templates/glm.jinja`` (and ``glm51.jinja``, ``glm_flash.jinja``)
needs ``tool_call.arguments`` as a dict (line 66 of glm.jinja:
``{% set _args = tc.arguments %}{% for k, v in _args.items() %}``). mlx-lm's
``process_message_content`` does that JSON-string→dict decode itself, so
``shape_request`` here must NOT decode again — a second pass feeds
``json.loads`` an already-decoded dict and crashes with
``the JSON object must be str, bytes or bytearray, not dict``.

Model output uses key/value XML:

    <tool_call>{name}<arg_key>{k1}</arg_key><arg_value>{v1}</arg_value>
    <arg_key>{k2}</arg_key><arg_value>{v2}</arg_value>...</tool_call>

Reference: ``~/Sombra/sac/.../client.rs:2223`` (parse), ``:2363`` (tool_call).
"""
from __future__ import annotations

import json
from typing import Any, Optional

from .base import (
    PassThroughDialect,
    ToolCall,
    to_tool_call_dict,
)


_TC_OPEN = "<tool_call>"
_TC_CLOSE = "</tool_call>"
_AK_OPEN = "<arg_key>"
_AK_CLOSE = "</arg_key>"
_AV_OPEN = "<arg_value>"
_AV_CLOSE = "</arg_value>"


class GLMDialect(PassThroughDialect):
    name = "glm"

    # shape_request inherited — mlx-lm decodes tool_call arguments itself.

    def parse_output(
        self, text: str, tools: Optional[list] = None
    ) -> list[dict]:
        _, tool_calls = _parse_glm_content(text)
        return [to_tool_call_dict(tc) for tc in tool_calls]


def _parse_glm_content(content: str) -> tuple[str, list[ToolCall]]:
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
            break
        body = remaining[start + len(_TC_OPEN):end]
        tc = _parse_glm_tool_call(body, counter)
        if tc is not None:
            tool_calls.append(tc)
            counter += 1
        remaining = remaining[end + len(_TC_CLOSE):]

    if remaining.strip():
        text_parts.append(remaining.strip())

    return "\n".join(text_parts), tool_calls


def _parse_glm_tool_call(body: str, index: int) -> Optional[ToolCall]:
    # The function name is at the start, before any <arg_key>.
    name_end = body.find(_AK_OPEN)
    if name_end < 0:
        name_end = len(body)
    name = body[:name_end].strip()
    if not name:
        return None

    params: dict[str, Any] = {}
    cursor = name_end
    while True:
        ks = body.find(_AK_OPEN, cursor)
        if ks < 0:
            break
        ks_inner = ks + len(_AK_OPEN)
        ke = body.find(_AK_CLOSE, ks_inner)
        if ke < 0:
            break
        key = body[ks_inner:ke].strip()

        vs = body.find(_AV_OPEN, ke)
        if vs < 0:
            break
        vs_inner = vs + len(_AV_OPEN)
        ve = body.find(_AV_CLOSE, vs_inner)
        if ve < 0:
            break
        raw = body[vs_inner:ve]
        try:
            parsed = json.loads(raw.strip())
        except (json.JSONDecodeError, ValueError):
            parsed = raw
        params[key] = parsed
        cursor = ve + len(_AV_CLOSE)

    return ToolCall(id=f"call_{index}", name=name, arguments=json.dumps(params))
