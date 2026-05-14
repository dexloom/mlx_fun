"""Gemma 4 dialect.

The bundled ``templates/gemma.jinja`` already handles both dict and string args
(line 186-194), so ``shape_request`` is a pure pass-through. Output parsing
extracts ``<|tool_call>call:name{k:v,...}<tool_call|>`` blocks and
``<|channel>thought\\n...<channel|>`` thinking blocks.
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional

from .base import (
    PassThroughDialect,
    ToolCall,
    extract_thinking_blocks,
    to_tool_call_dict,
)


_TC_RE = re.compile(
    r'<\|tool_call>call:([A-Za-z_][\w]*)\{(.*?)\}<tool_call\|>',
    re.DOTALL,
)
_CHANNEL_OPEN = "<|channel>"
_CHANNEL_CLOSE = "<channel|>"
# Gemma quoted string sentinel: <|"|>value<|"|>
_QUOTED_RE = re.compile(r'<\|"\|>(.*?)<\|"\|>', re.DOTALL)


class GemmaDialect(PassThroughDialect):
    name = "gemma"

    def parse_output(
        self, text: str, tools: Optional[list] = None
    ) -> list[dict]:
        # mlx-lm's state machine strips the `<|tool_call>...<tool_call|>`
        # delimiters before handing the body to the per-dialect parser, so
        # we receive just `call:name{key:val,...}` without the wrappers.
        # The regex below requires both markers, so wrap on the fly when
        # the open marker is missing. Same pattern as Minimax25Dialect.
        if "<|tool_call>" not in text:
            text = f"<|tool_call>{text}<tool_call|>"
        tool_calls: list[ToolCall] = []
        for i, m in enumerate(_TC_RE.finditer(text)):
            tc = _parse_gemma_call(m.group(1), m.group(2), i)
            if tc is not None:
                tool_calls.append(tc)
        return [to_tool_call_dict(tc) for tc in tool_calls]

    def extract_thinking(self, text: str) -> tuple[Optional[str], str]:
        """Gemma uses ``<|channel>thought\\n...<channel|>``. Falls back to
        ``<think>`` blocks if none found."""
        thinking_parts: list[str] = []
        main = []
        remaining = text
        while True:
            start = remaining.find(_CHANNEL_OPEN)
            if start < 0:
                break
            main.append(remaining[:start])
            after = remaining[start + len(_CHANNEL_OPEN):]
            end = after.find(_CHANNEL_CLOSE)
            if end < 0:
                block = after
                tail = ""
            else:
                block = after[:end]
                tail = after[end + len(_CHANNEL_CLOSE):]
            if block.startswith("thought\n"):
                block = block[len("thought\n"):]
            if block.strip():
                thinking_parts.append(block.strip())
            remaining = tail
            if end < 0:
                break
        main.append(remaining)
        main_text = "".join(main)
        std_think, residual = extract_thinking_blocks(main_text)
        if std_think:
            thinking_parts.append(std_think)
        thinking = "\n\n".join(thinking_parts) if thinking_parts else None
        return thinking, residual


def _parse_gemma_call(name: str, body: str, index: int) -> Optional[ToolCall]:
    """Parse Gemma's ``key:value,key:value`` body into a JSON args string.

    Values may be Gemma-quoted strings (``<|"|>...<|"|>``), JSON numbers/bools,
    or nested ``{...}`` / ``[...]`` structures.
    """
    params: dict[str, Any] = {}
    i = 0
    n = len(body)
    while i < n:
        # skip whitespace and commas
        while i < n and body[i] in " ,\n\t":
            i += 1
        if i >= n:
            break
        # key (up to ':')
        colon = body.find(":", i)
        if colon < 0:
            break
        key = body[i:colon].strip()
        i = colon + 1
        # value
        val, j = _read_gemma_value(body, i)
        if val is None:
            break
        params[key] = val
        i = j
    if not name:
        return None
    return ToolCall(id=f"call_{index}", name=name, arguments=json.dumps(params))


def _read_gemma_value(body: str, i: int) -> tuple[Any, int]:
    n = len(body)
    while i < n and body[i] in " \n\t":
        i += 1
    if i >= n:
        return None, i
    # Quoted string
    if body.startswith('<|"|>', i):
        end = body.find('<|"|>', i + len('<|"|>'))
        if end < 0:
            return body[i + len('<|"|>'):], n
        return body[i + len('<|"|>'):end], end + len('<|"|>')
    # Nested object
    if body[i] == "{":
        depth = 1
        j = i + 1
        while j < n and depth > 0:
            if body[j] == "{":
                depth += 1
            elif body[j] == "}":
                depth -= 1
            j += 1
        raw = body[i:j]
        try:
            return json.loads(raw), j
        except (json.JSONDecodeError, ValueError):
            return raw, j
    # Array
    if body[i] == "[":
        depth = 1
        j = i + 1
        while j < n and depth > 0:
            if body[j] == "[":
                depth += 1
            elif body[j] == "]":
                depth -= 1
            j += 1
        raw = body[i:j]
        try:
            return json.loads(raw), j
        except (json.JSONDecodeError, ValueError):
            return raw, j
    # Bare token up to next comma
    end = body.find(",", i)
    if end < 0:
        end = n
    raw = body[i:end].strip()
    try:
        return json.loads(raw), end
    except (json.JSONDecodeError, ValueError):
        return raw, end
