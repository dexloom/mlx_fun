"""Dialect base types and shared parsing helpers.

A "dialect" owns the per-model translation between the canonical OpenAI/Anthropic
JSON shape that mlx_fun exposes externally and the JSON shape that a particular
family's chat template expects internally. Each dialect provides:

- ``shape_request(messages, tools, tool_choice)`` — adjust the message/tool JSON
  *before* it crosses into ``tokenizer.apply_chat_template``.
- ``parse_output(text, tools)`` — extract structured tool calls from raw model
  output. Signature matches ``mlx_lm.server.ToolCallFormatter``'s parser slot,
  so streaming keeps working unchanged.
- ``extract_thinking(text)`` — pull reasoning content out of raw text. Most
  dialects can use ``extract_thinking_blocks`` directly.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable


@dataclass
class ToolCall:
    """Normalized tool call. Matches the OpenAI ``tool_calls[i]`` shape."""

    id: str
    name: str
    arguments: str  # JSON-encoded args; OpenAI wire format is a string


@dataclass
class ParsedOutput:
    content: str
    thinking: Optional[str] = None
    tool_calls: list[ToolCall] = field(default_factory=list)


@runtime_checkable
class Dialect(Protocol):
    name: str

    def shape_request(
        self,
        messages: list[dict],
        tools: Optional[list[dict]],
        tool_choice: Optional[Any],
    ) -> tuple[list[dict], Optional[list[dict]]]: ...

    def parse_output(
        self, text: str, tools: Optional[list] = None
    ) -> list[dict]: ...


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def extract_thinking_blocks(content: str) -> tuple[Optional[str], str]:
    """Extract ``<think>...</think>`` blocks from text.

    Returns ``(reasoning_or_None, content_without_thinking)``. Unclosed
    ``<think>`` tags absorb the rest of the string as thinking content.
    Ported from ``~/Sombra/sac/.../client.rs:2173``.
    """
    thinking_parts: list[str] = []
    text_parts: list[str] = []
    remaining = content

    while True:
        start = remaining.find("<think>")
        if start < 0:
            break
        before = remaining[:start]
        if before.strip():
            text_parts.append(before.strip())
        rest = remaining[start + len("<think>"):]
        end = rest.find("</think>")
        if end < 0:
            if rest.strip():
                thinking_parts.append(rest.strip())
            remaining = ""
            break
        block = rest[:end]
        if block.strip():
            thinking_parts.append(block.strip())
        remaining = rest[end + len("</think>"):]

    if remaining.strip():
        text_parts.append(remaining.strip())

    reasoning = "\n\n".join(thinking_parts) if thinking_parts else None
    return reasoning, "\n".join(text_parts)


def to_tool_call_dict(tc: ToolCall) -> dict:
    """Convert a ToolCall to the flat dict shape ``ToolCallFormatter`` consumes.

    ``ToolCallFormatter._format`` expects ``{"id": str, "name": str, "arguments": <obj>}``
    and JSON-stringifies ``arguments`` itself, so we hand it the decoded Python
    object — not the JSON string that lives on ``ToolCall.arguments``.
    """
    try:
        args_obj = json.loads(tc.arguments) if tc.arguments else {}
    except (json.JSONDecodeError, ValueError):
        args_obj = tc.arguments  # fall back to raw string
    return {"id": tc.id, "name": tc.name, "arguments": args_obj}


def parse_json_tool_call(json_str: str, index: int) -> Optional[ToolCall]:
    """Parse a JSON object as either OpenAI-nested or simple tool-call shape."""
    try:
        obj = json.loads(json_str)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(obj, dict):
        return None

    # OpenAI nested: {"type": "function", "function": {"name": ..., "arguments": ...}}
    func = obj.get("function")
    if isinstance(func, dict):
        name = func.get("name")
        if not isinstance(name, str):
            return None
        args = func.get("arguments", {})
        arguments = args if isinstance(args, str) else json.dumps(args)
        call_id = obj.get("id") if isinstance(obj.get("id"), str) else f"call_{index}"
        return ToolCall(id=call_id, name=name, arguments=arguments)

    # Flat: {"name": ..., "arguments": ...}
    name = obj.get("name")
    if isinstance(name, str):
        args = obj.get("arguments", {})
        arguments = args if isinstance(args, str) else json.dumps(args)
        call_id = obj.get("id") if isinstance(obj.get("id"), str) else f"call_{index}"
        return ToolCall(id=call_id, name=name, arguments=arguments)

    return None


def try_parse_json_object_as_tool_call(content: str, index: int) -> Optional[ToolCall]:
    """Find the first balanced JSON object starting at ``content[0]`` and try
    to parse it as a tool call."""
    depth = 0
    end_pos = 0
    for i, c in enumerate(content):
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                end_pos = i + 1
                break
    if end_pos == 0:
        return None
    return parse_json_tool_call(content[:end_pos], index)


def parse_json_tool_calls(content: str) -> list[ToolCall]:
    """Find tool calls in ``<tool_call>{json}</tool_call>`` blocks AND bare
    ``{"name": ...}`` JSON objects scattered in text.

    Bare-object scanning only inspects regions OUTSIDE wrapper blocks so we
    don't double-count a call that appears inside its own ``<tool_call>`` tag.
    """
    tool_calls: list[ToolCall] = []
    call_index = 0

    # Pass 1: <tool_call>...</tool_call> wrapped JSON. Track the [start, end]
    # spans we consume so pass 2 can skip them.
    consumed_spans: list[tuple[int, int]] = []
    cursor = 0
    while True:
        start = content.find("<tool_call>", cursor)
        if start < 0:
            break
        end = content.find("</tool_call>", start)
        if end < 0:
            break
        body = content[start + len("<tool_call>"):end].strip()
        tc = parse_json_tool_call(body, call_index)
        if tc is not None:
            tool_calls.append(tc)
            call_index += 1
        wrapper_end = end + len("</tool_call>")
        consumed_spans.append((start, wrapper_end))
        cursor = wrapper_end

    def _inside_consumed(pos: int) -> bool:
        return any(s <= pos < e for s, e in consumed_spans)

    # Pass 2: bare {"name": "..."} objects, skipping wrapped regions.
    search_pos = 0
    while search_pos < len(content):
        obj_start = content.find('{"name"', search_pos)
        if obj_start < 0:
            break
        if _inside_consumed(obj_start):
            search_pos = obj_start + 1
            continue
        tc = try_parse_json_object_as_tool_call(content[obj_start:], call_index)
        if tc is not None:
            tool_calls.append(tc)
            call_index += 1
        search_pos = obj_start + 1

    return tool_calls


class PassThroughDialect:
    """Base class for dialects whose shape_request is a no-op. Subclasses
    only need to override ``parse_output`` (and optionally ``extract_thinking``)."""

    name: str = "passthrough"

    def shape_request(
        self,
        messages: list[dict],
        tools: Optional[list[dict]],
        tool_choice: Optional[Any],
    ) -> tuple[list[dict], Optional[list[dict]]]:
        return messages, tools

    def parse_output(self, text: str, tools: Optional[list] = None) -> list[dict]:
        return []

    def extract_thinking(self, text: str) -> tuple[Optional[str], str]:
        return extract_thinking_blocks(text)
