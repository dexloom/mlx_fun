"""OpenAI-compatible dialect: pure pass-through.

The chat template handles everything natively. No reshaping, no tool-call
extraction from text — the model is expected to emit tool calls through the
``tool_calls`` field which mlx-lm parses upstream.
"""
from __future__ import annotations

from typing import Any, Optional

from .base import PassThroughDialect


class OpenAIDialect(PassThroughDialect):
    name = "openai"
