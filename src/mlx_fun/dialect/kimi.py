"""Kimi K2.6 dialect — wraps the existing ``kimi_k26_tool_parser``.

Kimi's templates accept the plain OpenAI message shape, so ``shape_request``
is a no-op. Output parsing is delegated to the long-standing parser in
``mlx_fun.kimi_k26_tool_parser``, replacing the previous ad-hoc
``tokenizer._tool_parser = ...`` wiring in ``server.py:1065``.
"""
from __future__ import annotations

from typing import Optional

from .base import PassThroughDialect


class KimiDialect(PassThroughDialect):
    name = "kimi"

    def parse_output(
        self, text: str, tools: Optional[list] = None
    ) -> list[dict]:
        from .. import kimi_k26_tool_parser
        result = kimi_k26_tool_parser.parse_tool_call(text, tools)
        return result if isinstance(result, list) else []
