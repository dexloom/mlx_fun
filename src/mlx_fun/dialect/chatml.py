"""ChatML dialect (Qwen 2.5, IQuest, legacy ChatML).

The bundled ``templates/iquest.jinja`` handles both string and dict args
(line 56-59), so ``shape_request`` is pass-through. Model output uses
``<tool_call>{"name": ..., "arguments": ...}</tool_call>``.
"""
from __future__ import annotations

from typing import Optional

from .base import (
    PassThroughDialect,
    parse_json_tool_calls,
    to_tool_call_dict,
)


class ChatMLDialect(PassThroughDialect):
    name = "chatml"

    def parse_output(
        self, text: str, tools: Optional[list] = None
    ) -> list[dict]:
        return [to_tool_call_dict(tc) for tc in parse_json_tool_calls(text)]
