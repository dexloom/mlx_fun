"""Tests for Anthropic <-> OpenAI format conversion (api_compat.py)."""

import json
import pytest

from mlx_fun.api_compat import (
    anthropic_to_openai_messages,
    build_anthropic_response,
    map_stop_reason,
    anthropic_stream_message_start,
    anthropic_stream_content_block_start,
    anthropic_stream_content_block_delta,
    anthropic_stream_content_block_stop,
    anthropic_stream_message_delta,
    anthropic_stream_message_stop,
    format_anthropic_sse,
)


# ---------------------------------------------------------------------------
# anthropic_to_openai_messages
# ---------------------------------------------------------------------------


class TestAnthropicToOpenAIMessages:
    """Test Anthropic request body -> OpenAI messages conversion."""

    def test_simple_text_message(self):
        body = {
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100,
        }
        messages, tools, stop_words = anthropic_to_openai_messages(body)
        assert len(messages) == 1
        assert messages[0] == {"role": "user", "content": "Hello"}
        assert tools is None
        assert stop_words == []

    def test_system_string(self):
        body = {
            "system": "You are helpful.",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 100,
        }
        messages, _, _ = anthropic_to_openai_messages(body)
        assert len(messages) == 2
        assert messages[0] == {"role": "system", "content": "You are helpful."}
        assert messages[1] == {"role": "user", "content": "Hi"}

    def test_system_array_blocks(self):
        body = {
            "system": [
                {"type": "text", "text": "You are helpful."},
                {"type": "text", "text": "Be concise."},
            ],
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 100,
        }
        messages, _, _ = anthropic_to_openai_messages(body)
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are helpful.\nBe concise."

    def test_user_content_blocks(self):
        body = {
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {"type": "text", "text": "Please explain."},
                ],
            }],
            "max_tokens": 100,
        }
        messages, _, _ = anthropic_to_openai_messages(body)
        assert len(messages) == 1
        assert messages[0]["content"] == "What is this?\nPlease explain."

    def test_assistant_text_string(self):
        body = {
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello there!"},
            ],
            "max_tokens": 100,
        }
        messages, _, _ = anthropic_to_openai_messages(body)
        assert messages[1] == {"role": "assistant", "content": "Hello there!"}

    def test_assistant_with_tool_use(self):
        body = {
            "messages": [{
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me check."},
                    {
                        "type": "tool_use",
                        "id": "toolu_abc123",
                        "name": "get_weather",
                        "input": {"location": "NYC"},
                    },
                ],
            }],
            "max_tokens": 100,
        }
        messages, _, _ = anthropic_to_openai_messages(body)
        assert len(messages) == 1
        msg = messages[0]
        assert msg["role"] == "assistant"
        assert msg["content"] == "Let me check."
        assert len(msg["tool_calls"]) == 1
        tc = msg["tool_calls"][0]
        assert tc["id"] == "toolu_abc123"
        assert tc["function"]["name"] == "get_weather"
        assert tc["function"]["arguments"] == {"location": "NYC"}

    def test_tool_result_message(self):
        body = {
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": "toolu_abc123",
                    "content": "Sunny, 72F",
                }],
            }],
            "max_tokens": 100,
        }
        messages, _, _ = anthropic_to_openai_messages(body)
        assert len(messages) == 1
        assert messages[0]["role"] == "tool"
        assert messages[0]["content"] == "Sunny, 72F"
        assert messages[0]["tool_call_id"] == "toolu_abc123"

    def test_tool_result_with_content_blocks(self):
        body = {
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": "toolu_abc",
                    "content": [{"type": "text", "text": "Result data"}],
                }],
            }],
            "max_tokens": 100,
        }
        messages, _, _ = anthropic_to_openai_messages(body)
        assert messages[0]["content"] == "Result data"

    def test_mixed_text_and_tool_result(self):
        body = {
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Here are the results:"},
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_abc",
                        "content": "42",
                    },
                ],
            }],
            "max_tokens": 100,
        }
        messages, _, _ = anthropic_to_openai_messages(body)
        assert len(messages) == 2
        assert messages[0] == {"role": "user", "content": "Here are the results:"}
        assert messages[1]["role"] == "tool"

    def test_tools_conversion(self):
        body = {
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": [{
                "name": "get_weather",
                "description": "Get weather info",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                    },
                    "required": ["location"],
                },
            }],
            "max_tokens": 100,
        }
        _, tools, _ = anthropic_to_openai_messages(body)
        assert tools is not None
        assert len(tools) == 1
        t = tools[0]
        assert t["type"] == "function"
        assert t["function"]["name"] == "get_weather"
        assert t["function"]["description"] == "Get weather info"
        assert t["function"]["parameters"]["type"] == "object"

    def test_stop_sequences(self):
        body = {
            "messages": [{"role": "user", "content": "Hi"}],
            "stop_sequences": ["Human:", "END"],
            "max_tokens": 100,
        }
        _, _, stop_words = anthropic_to_openai_messages(body)
        assert stop_words == ["Human:", "END"]

    def test_no_system(self):
        body = {
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 100,
        }
        messages, _, _ = anthropic_to_openai_messages(body)
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    def test_multi_turn_conversation(self):
        body = {
            "system": "Be helpful.",
            "messages": [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"},
                {"role": "user", "content": "And 3+3?"},
            ],
            "max_tokens": 100,
        }
        messages, _, _ = anthropic_to_openai_messages(body)
        assert len(messages) == 4
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[2]["role"] == "assistant"
        assert messages[3]["role"] == "user"


# ---------------------------------------------------------------------------
# Response conversion
# ---------------------------------------------------------------------------


class TestBuildAnthropicResponse:
    def test_simple_text(self):
        resp = build_anthropic_response(
            text="Hello!",
            finish_reason="stop",
            prompt_tokens=10,
            completion_tokens=5,
            model="test-model",
        )
        assert resp["type"] == "message"
        assert resp["role"] == "assistant"
        assert resp["model"] == "test-model"
        assert resp["stop_reason"] == "end_turn"
        assert resp["stop_sequence"] is None
        assert len(resp["content"]) == 1
        assert resp["content"][0] == {"type": "text", "text": "Hello!"}
        assert resp["usage"] == {"input_tokens": 10, "output_tokens": 5}
        assert resp["id"].startswith("msg_")

    def test_max_tokens_stop(self):
        resp = build_anthropic_response(
            text="partial",
            finish_reason="length",
            prompt_tokens=10,
            completion_tokens=100,
            model="m",
        )
        assert resp["stop_reason"] == "max_tokens"

    def test_with_tool_calls(self):
        resp = build_anthropic_response(
            text="Let me check.",
            finish_reason="stop",
            prompt_tokens=10,
            completion_tokens=20,
            model="m",
            tool_calls=[{
                "id": "toolu_abc",
                "function": {"name": "get_weather", "arguments": {"loc": "NYC"}},
            }],
        )
        assert resp["stop_reason"] == "tool_use"
        assert len(resp["content"]) == 2
        assert resp["content"][0]["type"] == "text"
        assert resp["content"][1]["type"] == "tool_use"
        assert resp["content"][1]["name"] == "get_weather"
        assert resp["content"][1]["input"] == {"loc": "NYC"}

    def test_empty_text_with_tool_calls(self):
        resp = build_anthropic_response(
            text="",
            finish_reason="stop",
            prompt_tokens=5,
            completion_tokens=10,
            model="m",
            tool_calls=[{
                "id": "toolu_abc",
                "function": {"name": "search", "arguments": {"q": "test"}},
            }],
        )
        # Empty text should not produce a text block
        assert len(resp["content"]) == 1
        assert resp["content"][0]["type"] == "tool_use"


# ---------------------------------------------------------------------------
# Stop reason mapping
# ---------------------------------------------------------------------------


class TestMapStopReason:
    def test_stop(self):
        assert map_stop_reason("stop") == "end_turn"

    def test_length(self):
        assert map_stop_reason("length") == "max_tokens"

    def test_tool_calls(self):
        assert map_stop_reason("tool_calls") == "end_turn"

    def test_none(self):
        assert map_stop_reason(None) == "end_turn"

    def test_unknown(self):
        assert map_stop_reason("unknown") == "end_turn"


# ---------------------------------------------------------------------------
# Streaming SSE helpers
# ---------------------------------------------------------------------------


class TestStreamingHelpers:
    def test_message_start(self):
        data = anthropic_stream_message_start("test-model", 42)
        assert data["type"] == "message_start"
        msg = data["message"]
        assert msg["role"] == "assistant"
        assert msg["model"] == "test-model"
        assert msg["content"] == []
        assert msg["stop_reason"] is None
        assert msg["usage"]["input_tokens"] == 42
        assert msg["usage"]["output_tokens"] == 0

    def test_content_block_start_text(self):
        data = anthropic_stream_content_block_start(0, "text")
        assert data["type"] == "content_block_start"
        assert data["index"] == 0
        assert data["content_block"]["type"] == "text"
        assert data["content_block"]["text"] == ""

    def test_content_block_delta(self):
        data = anthropic_stream_content_block_delta(0, "Hello")
        assert data["type"] == "content_block_delta"
        assert data["index"] == 0
        assert data["delta"]["type"] == "text_delta"
        assert data["delta"]["text"] == "Hello"

    def test_content_block_stop(self):
        data = anthropic_stream_content_block_stop(0)
        assert data["type"] == "content_block_stop"
        assert data["index"] == 0

    def test_message_delta(self):
        data = anthropic_stream_message_delta("end_turn", 50)
        assert data["type"] == "message_delta"
        assert data["delta"]["stop_reason"] == "end_turn"
        assert data["delta"]["stop_sequence"] is None
        assert data["usage"]["output_tokens"] == 50

    def test_message_stop(self):
        data = anthropic_stream_message_stop()
        assert data["type"] == "message_stop"

    def test_format_sse(self):
        data = {"type": "message_stop"}
        result = format_anthropic_sse("message_stop", data)
        assert isinstance(result, bytes)
        decoded = result.decode()
        assert decoded.startswith("event: message_stop\n")
        assert "data: " in decoded
        assert decoded.endswith("\n\n")
        # Verify JSON is valid
        json_str = decoded.split("data: ", 1)[1].rstrip("\n")
        parsed = json.loads(json_str)
        assert parsed == data


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_messages(self):
        body = {"messages": [], "max_tokens": 100}
        messages, tools, stop_words = anthropic_to_openai_messages(body)
        assert messages == []
        assert tools is None
        assert stop_words == []

    def test_no_tools(self):
        body = {"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 100}
        _, tools, _ = anthropic_to_openai_messages(body)
        assert tools is None

    def test_empty_tools_list(self):
        body = {
            "messages": [{"role": "user", "content": "Hi"}],
            "tools": [],
            "max_tokens": 100,
        }
        _, tools, _ = anthropic_to_openai_messages(body)
        assert tools is None

    def test_assistant_only_tool_use_no_text(self):
        body = {
            "messages": [{
                "role": "assistant",
                "content": [{
                    "type": "tool_use",
                    "id": "toolu_123",
                    "name": "search",
                    "input": {"q": "test"},
                }],
            }],
            "max_tokens": 100,
        }
        messages, _, _ = anthropic_to_openai_messages(body)
        msg = messages[0]
        assert msg["role"] == "assistant"
        assert msg["content"] is None
        assert len(msg["tool_calls"]) == 1

    def test_multiple_tool_results(self):
        body = {
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "r1"},
                    {"type": "tool_result", "tool_use_id": "t2", "content": "r2"},
                ],
            }],
            "max_tokens": 100,
        }
        messages, _, _ = anthropic_to_openai_messages(body)
        assert len(messages) == 2
        assert messages[0]["tool_call_id"] == "t1"
        assert messages[1]["tool_call_id"] == "t2"

    def test_system_with_cache_control_blocks(self):
        """Anthropic system blocks may have cache_control metadata — we ignore it."""
        body = {
            "system": [
                {
                    "type": "text",
                    "text": "You are a helpful assistant.",
                    "cache_control": {"type": "ephemeral"},
                },
            ],
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 100,
        }
        messages, _, _ = anthropic_to_openai_messages(body)
        assert messages[0]["content"] == "You are a helpful assistant."
