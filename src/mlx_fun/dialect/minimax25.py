"""MiniMax-M2.5+ dialect.

The bundled ``templates/minimax_25.jinja`` expects ``tool_call.arguments`` as a
dict (line 112 of the template: ``{% for k, v in _args.items() %}``), so OpenAI's
JSON-string encoding needs to be parsed. Model output uses the XML form:

    <minimax:tool_call>
    <invoke name="fn">
    <parameter name="key">value</parameter>
    ...
    </invoke>
    </minimax:tool_call>

Reference: ``~/Sombra/sac/.../client.rs:1421`` (parse_minimax_response).
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional

from .base import (
    PassThroughDialect,
    ToolCall,
    to_tool_call_dict,
)


_TC_OPEN = "<minimax:tool_call>"
_TC_CLOSE = "</minimax:tool_call>"
# Match a complete <invoke>…</invoke> block including its body so we can
# iterate all of them inside a single <minimax:tool_call> wrapper. The
# earlier `_INVOKE_RE` only captured the function name on the opening
# tag, and the helper used `.search()` → only the FIRST <invoke> in the
# block survived. Models emit parallel tool calls as multiple <invoke>
# blocks inside one <minimax:tool_call>, so single-match dropped tool
# calls 2..N. Matches upstream mlx_lm/tool_parsers/minimax_m2.py.
_INVOKE_RE = re.compile(r"<invoke\s+name=(.*?)</invoke>", re.DOTALL)
_PARAM_RE = re.compile(
    r'<parameter\s+name="([^"]+)">(.*?)</parameter>',
    re.DOTALL,
)
_INVOKE_NAME_RE = re.compile(r'"([^"]+)"|\'([^\']+)\'|^([^>\s]+)')


class Minimax25Dialect(PassThroughDialect):
    name = "minimax25"

    def shape_request(
        self,
        messages: list[dict],
        tools: Optional[list[dict]],
        tool_choice: Optional[Any],
    ) -> tuple[list[dict], Optional[list[dict]]]:
        # NOTE: do NOT normalize_tool_call_arguments here. mlx-lm's
        # server.py:146-150 already does the string→dict decode on
        # `tool_calls[i].function.arguments` BEFORE apply_chat_template,
        # and it uses a plain json.loads() that throws when the field is
        # already a dict. Doing the normalize twice produces
        # "the JSON object must be str, bytes or bytearray, not dict"
        # and the request crashes with HTTP 404. Pass through and let
        # mlx-lm own the conversion for the template.
        #
        # Bridge `reasoning` → `reasoning_content` on assistant messages.
        # The MiniMax-2.7 chat template (line 89-90) reads
        # `message.reasoning_content`, while canonical clients (SAC,
        # OpenAI o1, OpenRouter, llama.cpp `--jinja`) ship the field
        # under the single name `reasoning`. Without this bridge the
        # template silently renders no prior `<think>` block and the
        # model loses tool-use continuity — verified to freeze the
        # audit at turn 6 instead of turn 10.
        out: list[dict] = []
        for msg in messages:
            if (
                isinstance(msg, dict)
                and msg.get("role") == "assistant"
                and isinstance(msg.get("reasoning"), str)
                and msg.get("reasoning")
                and not msg.get("reasoning_content")
            ):
                new_msg = dict(msg)
                new_msg["reasoning_content"] = msg["reasoning"]
                out.append(new_msg)
            else:
                out.append(msg)
        return out, tools

    def parse_output(
        self, text: str, tools: Optional[list] = None
    ) -> list[dict]:
        # mlx-lm's state machine captures the body BETWEEN
        # <minimax:tool_call> ... </minimax:tool_call> and strips the
        # outer delimiters before calling this parser (see ToolCallFormatter
        # in mlx_lm/server.py and the per-tokenizer
        # tool_call_start/tool_call_end boundary configuration). Our shared
        # _parse_minimax_content expects the full content with delimiters
        # though — and silently returns no tool calls when the open marker
        # is missing, which is the exact failure that makes turn-1 responses
        # look like "empty content, no tool_calls". Wrap on the fly when
        # called with just an inner body so both code paths work.
        if _TC_OPEN not in text:
            text = f"{_TC_OPEN}\n{text}\n{_TC_CLOSE}"
        _, tool_calls = _parse_minimax_content(text)
        return [to_tool_call_dict(tc) for tc in tool_calls]


def _parse_minimax_content(content: str) -> tuple[str, list[ToolCall]]:
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
        block = remaining[start:end + len(_TC_CLOSE)]
        # Iterate ALL <invoke> blocks inside this <minimax:tool_call>
        # wrapper. Parallel tool calls arrive as multiple <invoke> elements
        # inside one wrapper; using `.search()` here drops calls 2..N.
        for tc in _parse_minimax_tool_calls(block, counter):
            tool_calls.append(tc)
            counter += 1
        remaining = remaining[end + len(_TC_CLOSE):]

    if remaining.strip():
        text_parts.append(remaining.strip())

    return "\n".join(text_parts), tool_calls


def _parse_minimax_tool_calls(block: str, start_index: int) -> list[ToolCall]:
    """Extract every <invoke>...</invoke> from `block` as a ToolCall."""
    out: list[ToolCall] = []
    for idx, invoke_text in enumerate(_INVOKE_RE.findall(block)):
        nm = _INVOKE_NAME_RE.match(invoke_text.strip())
        if nm is None:
            continue
        name = nm.group(1) or nm.group(2) or nm.group(3)
        if not name:
            continue
        params: dict[str, Any] = {}
        for pm in _PARAM_RE.finditer(invoke_text):
            key = pm.group(1)
            raw = pm.group(2)
            try:
                parsed = json.loads(raw.strip())
            except (json.JSONDecodeError, ValueError):
                parsed = raw
            params[key] = parsed
        out.append(
            ToolCall(
                id=f"call_{start_index + idx}",
                name=name,
                arguments=json.dumps(params),
            )
        )
    return out
