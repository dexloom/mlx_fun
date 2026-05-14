"""Legacy MiniMax (pre-M2.5) dialect.

The bundled ``templates/minimax.jinja`` reads ``tool_call.arguments`` as a
dict (line 114 uses ``_args.items()``). mlx-lm's ``process_message_content``
already converts the OpenAI JSON-string into a dict before
``apply_chat_template``, so ``shape_request`` is a pass-through —
double-decoding here crashes mlx-lm.

Output parsing reuses the M2.5 XML parser; both versions emit the same
``<minimax:tool_call><invoke name=...>...</invoke></minimax:tool_call>``
shape.
"""
from __future__ import annotations

from typing import Optional

from .base import PassThroughDialect, to_tool_call_dict
from .minimax25 import _parse_minimax_content


class MinimaxDialect(PassThroughDialect):
    name = "minimax"

    # shape_request inherited — mlx-lm decodes tool_call arguments itself.

    def parse_output(
        self, text: str, tools: Optional[list] = None
    ) -> list[dict]:
        _, tool_calls = _parse_minimax_content(text)
        return [to_tool_call_dict(tc) for tc in tool_calls]
