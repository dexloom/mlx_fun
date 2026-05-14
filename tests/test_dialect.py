"""Tests for the per-model template dialect layer.

Each dialect's job is twofold:

1. ``shape_request`` reshapes the OpenAI/Anthropic JSON the server receives
   into the shape its bundled Jinja template expects (most commonly: parse
   ``tool_calls[].function.arguments`` from JSON string to dict so templates
   using ``|items`` / ``.items()`` don't crash).
2. ``parse_output`` extracts structured tool calls from the model's raw text
   output. The return shape matches ``mlx_lm.server.ToolCallFormatter``'s
   parser slot: a list of flat dicts ``{"id", "name", "arguments": <dict>}``.
"""
from __future__ import annotations

import json

import pytest

from mlx_fun.dialect import (
    detect_from_model_type,
    detect_from_template_content,
    extract_thinking_blocks,
    parse_json_tool_calls,
    resolve_dialect,
)


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_type,expected", [
    ("qwen3_moe", "qwen35"),
    ("qwen3_next", "qwen35"),
    ("glm4_moe", "glm"),
    ("glm4_moe_lite", "glm"),
    ("glm_moe_dsa", "glm"),
    ("deepseek_v32", "glm"),
    ("minimax_m2", "minimax25"),
    ("minimax", "minimax"),
    ("gemma4", "gemma"),
    ("kimi_k25", "kimi"),
    ("unknown_xyz", "openai"),
    (None, "openai"),
])
def test_resolve_from_model_type(model_type, expected):
    assert resolve_dialect(model_type).name == expected


@pytest.mark.parametrize("template,expected", [
    ("<|im_start|>system\n<function=foo>\n<parameter=k>v</parameter>", "qwen35"),
    ("<|im_start|>system\nhello<|im_end|>", "chatml"),
    ("[gMASK]<sop>\n<|user|>...", "glm"),
    ("<arg_key>k</arg_key><arg_value>v</arg_value>", "glm"),
    ("<minimax:tool_call>...message.tool_calls...", "minimax25"),
    ("<minimax:tool_call>... ]~b]ai", "minimax"),
    ("<|turn>system\n<|tool_call>call:foo<tool_call|>", "gemma"),
    ("<|tool_call_begin|>functions.foo:0<|tool_call_end|>", "kimi"),
    ("plain text no markers", "openai"),  # falls back
    ("", "openai"),
])
def test_resolve_from_template_content(template, expected):
    # No model_type, fall back to template fingerprinting.
    assert resolve_dialect(None, template).name == expected


def test_resolve_priority_model_type_wins_over_template():
    """When both are given, model_type takes priority."""
    d = resolve_dialect("qwen3_next", "[gMASK]<sop> ...")  # template looks like GLM
    assert d.name == "qwen35"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def test_extract_thinking_blocks_basic():
    text = "before <think>secret reasoning</think> after"
    thinking, body = extract_thinking_blocks(text)
    assert thinking == "secret reasoning"
    assert body == "before\nafter"


def test_extract_thinking_blocks_multiple():
    text = "<think>one</think> mid <think>two</think> end"
    thinking, body = extract_thinking_blocks(text)
    assert thinking == "one\n\ntwo"
    assert "mid" in body
    assert "end" in body


def test_extract_thinking_blocks_unclosed():
    text = "visible <think>tail without close"
    thinking, body = extract_thinking_blocks(text)
    assert thinking == "tail without close"
    assert body == "visible"


def test_extract_thinking_blocks_none():
    thinking, body = extract_thinking_blocks("no thinking here")
    assert thinking is None
    assert body == "no thinking here"


def test_parse_json_tool_calls_wrapped():
    text = '<tool_call>{"name": "foo", "arguments": {"x": 1}}</tool_call>'
    calls = parse_json_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].name == "foo"
    assert json.loads(calls[0].arguments) == {"x": 1}


def test_parse_json_tool_calls_bare_object():
    text = 'some text {"name": "bar", "arguments": {"y": 2}} trailing'
    calls = parse_json_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].name == "bar"


# ---------------------------------------------------------------------------
# Qwen3.5 dialect
# ---------------------------------------------------------------------------


def test_qwen35_shape_is_passthrough_for_args():
    """shape_request MUST NOT decode ``tool_call.arguments`` — mlx-lm's
    ``process_message_content`` does that itself before
    ``apply_chat_template``. Decoding here causes a double-decode crash
    ``the JSON object must be str, bytes or bytearray, not dict``.
    See ``__init__.py`` note for details.
    """
    d = resolve_dialect("qwen3_next")
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "tool_calls": [{
            "id": "call_x",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"location": "Paris", "units": "metric"}',
            },
        }]},
    ]
    out, _ = d.shape_request(messages, None, None)
    args = out[1]["tool_calls"][0]["function"]["arguments"]
    # Args MUST remain a JSON string for mlx-lm to decode itself.
    assert isinstance(args, str)
    assert args == '{"location": "Paris", "units": "metric"}'


def test_qwen35_shape_no_tool_calls_is_noop():
    d = resolve_dialect("qwen3_next")
    messages = [{"role": "user", "content": "hello"}]
    out, tools = d.shape_request(messages, ["x"], None)
    assert out == messages
    assert tools == ["x"]


def test_qwen35_parse_simple_tool_call():
    d = resolve_dialect("qwen3_next")
    raw = (
        "<tool_call>\n<function=get_weather>\n"
        "<parameter=location>\nParis\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    calls = d.parse_output(raw)
    assert len(calls) == 1
    assert calls[0]["name"] == "get_weather"
    assert calls[0]["arguments"] == {"location": "Paris"}


def test_qwen35_parse_json_param_value():
    """A parameter value that looks like JSON should be decoded
    (so numbers don't show up as strings on the wire)."""
    d = resolve_dialect("qwen3_next")
    raw = (
        "<tool_call>\n<function=calc>\n"
        "<parameter=x>\n42\n</parameter>\n"
        "<parameter=flag>\ntrue\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    calls = d.parse_output(raw)
    assert calls[0]["arguments"] == {"x": 42, "flag": True}


def test_qwen35_parse_multiple_tool_calls():
    d = resolve_dialect("qwen3_next")
    raw = (
        "<tool_call><function=a><parameter=p>\n1\n</parameter></function></tool_call>"
        "<tool_call><function=b><parameter=q>\n2\n</parameter></function></tool_call>"
    )
    calls = d.parse_output(raw)
    assert [c["name"] for c in calls] == ["a", "b"]
    assert calls[0]["arguments"] == {"p": 1}
    assert calls[1]["arguments"] == {"q": 2}


def test_qwen35_parse_no_tool_calls():
    d = resolve_dialect("qwen3_next")
    assert d.parse_output("just plain text") == []


# ---------------------------------------------------------------------------
# Minimax25 dialect
# ---------------------------------------------------------------------------


def test_minimax25_shape_passes_args_through():
    """mlx-lm decodes ``arguments`` itself; the dialect must not."""
    d = resolve_dialect("minimax_m2")
    messages = [{"role": "assistant", "tool_calls": [{
        "type": "function",
        "function": {"name": "f", "arguments": '{"a": 1}'},
    }]}]
    out, _ = d.shape_request(messages, None, None)
    assert out[0]["tool_calls"][0]["function"]["arguments"] == '{"a": 1}'


def test_minimax25_shape_bridges_reasoning_to_reasoning_content():
    """The MiniMax-M2.7 chat template reads ``message.reasoning_content``
    while canonical OpenAI/Anthropic clients ship the field as ``reasoning``.
    The dialect bridges the two so prior ``<think>`` blocks survive
    multi-turn tool-use sessions."""
    d = resolve_dialect("minimax_m2")
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "answer", "reasoning": "thought trail"},
    ]
    out, _ = d.shape_request(messages, None, None)
    assert out[1]["reasoning_content"] == "thought trail"
    # original `reasoning` field preserved (mlx-lm tolerates unknown keys)
    assert out[1]["reasoning"] == "thought trail"


def test_minimax25_shape_does_not_overwrite_reasoning_content():
    """If both fields are present, the existing reasoning_content wins."""
    d = resolve_dialect("minimax_m2")
    messages = [{
        "role": "assistant",
        "reasoning": "from-client",
        "reasoning_content": "preferred",
    }]
    out, _ = d.shape_request(messages, None, None)
    assert out[0]["reasoning_content"] == "preferred"


def test_minimax25_parse_invoke_block():
    d = resolve_dialect("minimax_m2")
    raw = (
        '<minimax:tool_call>\n<invoke name="search">\n'
        '<parameter name="q">apples</parameter>\n'
        '<parameter name="limit">5</parameter>\n'
        '</invoke>\n</minimax:tool_call>'
    )
    calls = d.parse_output(raw)
    assert len(calls) == 1
    assert calls[0]["name"] == "search"
    assert calls[0]["arguments"] == {"q": "apples", "limit": 5}


# ---------------------------------------------------------------------------
# GLM dialect
# ---------------------------------------------------------------------------


def test_glm_shape_passes_args_through():
    """mlx-lm decodes args itself; dialect must leave them as JSON string."""
    d = resolve_dialect("glm4_moe")
    messages = [{"role": "assistant", "tool_calls": [{
        "type": "function",
        "function": {"name": "f", "arguments": '{"k": "v"}'},
    }]}]
    out, _ = d.shape_request(messages, None, None)
    assert out[0]["tool_calls"][0]["function"]["arguments"] == '{"k": "v"}'


def test_glm_parse_arg_key_value():
    d = resolve_dialect("glm4_moe")
    raw = (
        "<tool_call>get_weather"
        "<arg_key>city</arg_key><arg_value>Paris</arg_value>"
        "<arg_key>days</arg_key><arg_value>3</arg_value>"
        "</tool_call>"
    )
    calls = d.parse_output(raw)
    assert len(calls) == 1
    assert calls[0]["name"] == "get_weather"
    # "Paris" is not valid JSON, stays as string; 3 parses as int.
    assert calls[0]["arguments"] == {"city": "Paris", "days": 3}


# ---------------------------------------------------------------------------
# Gemma dialect
# ---------------------------------------------------------------------------


def test_gemma_parse_call_block():
    d = resolve_dialect("gemma4")
    raw = '<|tool_call>call:lookup{q:<|"|>apple pie<|"|>,limit:5}<tool_call|>'
    calls = d.parse_output(raw)
    assert len(calls) == 1
    assert calls[0]["name"] == "lookup"
    assert calls[0]["arguments"] == {"q": "apple pie", "limit": 5}


def test_gemma_extract_thinking_channel():
    d = resolve_dialect("gemma4")
    text = "head <|channel>thought\nplanning<channel|> body"
    thinking, body = d.extract_thinking(text)
    assert thinking == "planning"
    assert "body" in body


# ---------------------------------------------------------------------------
# ChatML dialect
# ---------------------------------------------------------------------------


def test_chatml_parse_json_tool_call():
    d = resolve_dialect(None, "<|im_start|>system\nuser content<|im_end|>")
    assert d.name == "chatml"
    raw = '<tool_call>{"name": "ping", "arguments": {"host": "1.1.1.1"}}</tool_call>'
    calls = d.parse_output(raw)
    assert len(calls) == 1
    assert calls[0]["name"] == "ping"
    assert calls[0]["arguments"] == {"host": "1.1.1.1"}


# ---------------------------------------------------------------------------
# OpenAI passthrough
# ---------------------------------------------------------------------------


def test_openai_dialect_is_passthrough():
    d = resolve_dialect("anything_unknown")
    assert d.name == "openai"
    msgs = [{"role": "user", "content": "hi"}]
    out, tools = d.shape_request(msgs, [{"x": 1}], None)
    assert out is msgs
    assert tools == [{"x": 1}]
    assert d.parse_output("model said this") == []


# ---------------------------------------------------------------------------
# Kimi dialect — sanity check that the registry wires up the existing parser
# ---------------------------------------------------------------------------


def test_kimi_dialect_delegates_to_existing_parser():
    d = resolve_dialect("kimi_k25")
    assert d.name == "kimi"
    raw = (
        '<|tool_call_begin|>functions.greet:0'
        '<|tool_call_argument_begin|>{"name": "world"}<|tool_call_end|>'
    )
    calls = d.parse_output(raw)
    assert len(calls) == 1
    assert calls[0]["name"] == "greet"
    assert calls[0]["arguments"] == {"name": "world"}
