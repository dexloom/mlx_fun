"""Anthropic Messages API <-> OpenAI Chat Completions format converters.

Pure conversion functions with no server dependencies. Both APIs share the
same generation pipeline — jinja templates always receive OpenAI-style messages.
"""

import uuid
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Anthropic -> OpenAI message/tool conversion
# ---------------------------------------------------------------------------


def anthropic_to_openai_messages(
    body: dict,
) -> Tuple[List[dict], Optional[List[dict]], List[str]]:
    """Convert an Anthropic Messages request body to OpenAI internal format.

    Returns:
        (messages, tools, stop_words) — ready for CompletionRequest / jinja templates.
    """
    messages: List[dict] = []

    # System prompt -> first message with role "system"
    system = body.get("system")
    if system:
        if isinstance(system, str):
            messages.append({"role": "system", "content": system})
        elif isinstance(system, list):
            # Array of content blocks (with optional cache_control)
            text_parts = [
                block["text"] for block in system if block.get("type") == "text"
            ]
            if text_parts:
                messages.append({"role": "system", "content": "\n".join(text_parts)})

    # Convert each message
    for msg in body.get("messages", []):
        converted = _convert_message(msg)
        if isinstance(converted, list):
            messages.extend(converted)
        else:
            messages.append(converted)

    # Convert tools
    tools = _convert_tools(body.get("tools"))

    # Stop sequences
    stop_words = body.get("stop_sequences", [])

    return messages, tools, stop_words


def _convert_message(msg: dict) -> Any:
    """Convert a single Anthropic message to one or more OpenAI messages."""
    role = msg["role"]
    content = msg.get("content", "")

    if role == "user":
        return _convert_user_message(content)
    elif role == "assistant":
        return _convert_assistant_message(content)
    return {"role": role, "content": content}


def _convert_user_message(content) -> Any:
    """Convert Anthropic user message content to OpenAI format.

    Handles: plain string, text blocks, and tool_result blocks.
    """
    if isinstance(content, str):
        return {"role": "user", "content": content}

    # Array of content blocks — may contain text and/or tool_result
    tool_results = []
    text_parts = []

    for block in content:
        block_type = block.get("type", "")
        if block_type == "tool_result":
            tool_results.append(_convert_tool_result(block))
        elif block_type == "text":
            text_parts.append(block["text"])

    # If only tool results, return them as separate tool messages
    if tool_results and not text_parts:
        return tool_results

    # If only text, return as user message
    if text_parts and not tool_results:
        return {"role": "user", "content": "\n".join(text_parts)}

    # Mixed: user text message + tool result messages
    result = []
    if text_parts:
        result.append({"role": "user", "content": "\n".join(text_parts)})
    result.extend(tool_results)
    return result


def _convert_tool_result(block: dict) -> dict:
    """Convert Anthropic tool_result block to OpenAI tool message."""
    content = block.get("content", "")
    if isinstance(content, list):
        # Extract text from content blocks
        text_parts = [
            b["text"] for b in content if isinstance(b, dict) and b.get("type") == "text"
        ]
        content = "\n".join(text_parts)

    return {
        "role": "tool",
        "content": str(content),
        "tool_call_id": block.get("tool_use_id", ""),
    }


def _convert_assistant_message(content) -> dict:
    """Convert Anthropic assistant message to OpenAI format.

    Handles text blocks and tool_use blocks.
    """
    if isinstance(content, str):
        return {"role": "assistant", "content": content}

    text_parts = []
    tool_calls = []

    for block in content:
        block_type = block.get("type", "")
        if block_type == "text":
            text_parts.append(block["text"])
        elif block_type == "tool_use":
            tool_calls.append({
                "id": block.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                "type": "function",
                "function": {
                    "name": block["name"],
                    "arguments": block.get("input", {}),
                },
            })

    result: dict = {
        "role": "assistant",
        "content": "\n".join(text_parts) if text_parts else None,
    }
    if tool_calls:
        result["tool_calls"] = tool_calls
    return result


def _convert_tools(tools: Optional[List[dict]]) -> Optional[List[dict]]:
    """Convert Anthropic tool definitions to OpenAI format."""
    if not tools:
        return None

    openai_tools = []
    for tool in tools:
        openai_tools.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {}),
            },
        })
    return openai_tools


# ---------------------------------------------------------------------------
# OpenAI response -> Anthropic response conversion
# ---------------------------------------------------------------------------


def map_stop_reason(finish_reason: Optional[str]) -> str:
    """Map OpenAI finish_reason to Anthropic stop_reason."""
    mapping = {
        "stop": "end_turn",
        "length": "max_tokens",
        "tool_calls": "end_turn",
    }
    return mapping.get(finish_reason or "stop", "end_turn")


def build_anthropic_response(
    text: str,
    finish_reason: str,
    prompt_tokens: int,
    completion_tokens: int,
    model: str,
    tool_calls: Optional[List[dict]] = None,
) -> dict:
    """Build a complete Anthropic Messages API response."""
    content: List[dict] = []
    if text:
        content.append({"type": "text", "text": text})

    if tool_calls:
        import json as _json
        for tc in tool_calls:
            fn = tc.get("function", tc)
            args = fn.get("arguments", {})
            # ToolCallFormatter returns arguments as a JSON string —
            # parse it back to a dict for Anthropic format.
            if isinstance(args, str):
                try:
                    args = _json.loads(args)
                except (_json.JSONDecodeError, ValueError):
                    args = {"_raw": args}
            content.append({
                "type": "tool_use",
                "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:12]}"),
                "name": fn["name"],
                "input": args,
            })

    stop_reason = map_stop_reason(finish_reason)
    if tool_calls:
        stop_reason = "tool_use"

    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": model,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
        },
    }


# ---------------------------------------------------------------------------
# Anthropic streaming SSE helpers
# ---------------------------------------------------------------------------


def anthropic_stream_message_start(
    model: str, prompt_tokens: int,
) -> dict:
    """Build the message_start SSE event data."""
    return {
        "type": "message_start",
        "message": {
            "id": f"msg_{uuid.uuid4().hex[:24]}",
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": model,
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {
                "input_tokens": prompt_tokens,
                "output_tokens": 0,
            },
        },
    }


def anthropic_stream_content_block_start(index: int, block_type: str = "text") -> dict:
    """Build the content_block_start SSE event data."""
    block: dict = {"type": block_type}
    if block_type == "text":
        block["text"] = ""
    return {
        "type": "content_block_start",
        "index": index,
        "content_block": block,
    }


def anthropic_stream_content_block_delta(index: int, text: str) -> dict:
    """Build a text content_block_delta SSE event data."""
    return {
        "type": "content_block_delta",
        "index": index,
        "delta": {
            "type": "text_delta",
            "text": text,
        },
    }


def anthropic_stream_content_block_stop(index: int) -> dict:
    """Build the content_block_stop SSE event data."""
    return {
        "type": "content_block_stop",
        "index": index,
    }


def anthropic_stream_message_delta(
    stop_reason: str, output_tokens: int,
) -> dict:
    """Build the message_delta SSE event data."""
    return {
        "type": "message_delta",
        "delta": {
            "stop_reason": stop_reason,
            "stop_sequence": None,
        },
        "usage": {
            "output_tokens": output_tokens,
        },
    }


def anthropic_stream_message_stop() -> dict:
    """Build the message_stop SSE event data."""
    return {"type": "message_stop"}


def format_anthropic_sse(event_type: str, data: dict) -> bytes:
    """Format an Anthropic SSE event as bytes."""
    import json
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n".encode()
