"""Permissive Kimi-K2.6 tool-call parser.

The standard Kimi-K2 format the upstream parser expects is:

    <|tool_call_begin|>functions.NAME:0<|tool_call_argument_begin|>{...}<|tool_call_end|>

The Q3-quantized Kimi-K2.6 sometimes emits ids of the form ``tool_call_N``
(sequential index, no function name). This parser accepts either; for the
nameless form, the function name is inferred by matching the emitted
argument keys against each available tool's declared parameter schema.
"""

import ast
import json
from typing import Any, List, Optional

import regex as re


_TOOL_CALL_BLOCK = re.compile(
    r"<\|tool_call_begin\|>(.*?)<\|tool_call_end\|>",
    re.DOTALL,
)
# Lenient block: captures the body after `<|tool_call_begin|>` whose
# closer is either `<|tool_call_end|>`, the next `<|tool_call_begin|>`,
# the section closer, or end-of-string. The Q3-quantized + REAP-pruned
# Kimi-K2.6 variants sometimes drop the trailing `<|tool_call_end|>`
# and/or wrap the whole thing in a `<|tool_calls_section_begin|>` /
# `<|tool_calls_section_end|>` pair instead.
_TOOL_CALL_BLOCK_LENIENT = re.compile(
    r"<\|tool_call_begin\|>(.*?)(?=<\|tool_call_end\|>|<\|tool_call_begin\|>|<\|tool_calls_section_end\|>|$)",
    re.DOTALL,
)
# Strip optional leading <|tool_call_begin|> and trailing <|tool_call_end|>.
# When mlx-lm's streaming state machine flips on the per-call markers,
# the buffer it hands us *starts* with <|tool_call_begin|> but the end
# marker has already been consumed by the state transition.
_BEGIN_MARKER = "<|tool_call_begin|>"
_END_MARKER = "<|tool_call_end|>"
# Section wrappers some variants emit around the whole tool_calls block.
_SECTION_BEGIN_MARKER = "<|tool_calls_section_begin|>"
_SECTION_END_MARKER = "<|tool_calls_section_end|>"
_STANDARD_ID = re.compile(
    r"^\s*(?:functions\.)?(?P<name>[A-Za-z_][\w]*?):(?P<idx>\d+)\s*"
    r"<\|tool_call_argument_begin\|>(?P<args>.*)$",
    re.DOTALL,
)
_ANY_ID = re.compile(
    r"^\s*(?P<id>[^<]*?)\s*<\|tool_call_argument_begin\|>(?P<args>.*)$",
    re.DOTALL,
)
# Inner separator the Q3-quantized + REAP-pruned Kimi variants sometimes
# emit *between* tool calls when they pack multiple calls into a single
# <|tool_call_begin|>…<|tool_call_end|> block instead of one block per call.
# Patterns observed:
#   ``functions.NAME:N<|tool_call_argument_begin|>``  (standard)
#   ``tool_call_N<|tool_call_argument_begin|>``       (Q3 nameless variant)
# Make the ``:N`` suffix optional so both forms split correctly.
_INNER_SEPARATOR = re.compile(
    r"(?:functions\.)?[A-Za-z_][\w]*(?::\d+)?\s*<\|tool_call_argument_begin\|>",
    re.DOTALL,
)


def _strip_markers(text: str) -> str:
    text = text.strip()
    # Section wrappers (some variants emit them, some don't)
    if text.startswith(_SECTION_BEGIN_MARKER):
        text = text[len(_SECTION_BEGIN_MARKER):]
    if text.endswith(_SECTION_END_MARKER):
        text = text[: -len(_SECTION_END_MARKER)]
    text = text.strip()
    # Per-call wrappers
    if text.startswith(_BEGIN_MARKER):
        text = text[len(_BEGIN_MARKER):]
    if text.endswith(_END_MARKER):
        text = text[: -len(_END_MARKER)]
    return text.strip()


def _deserialize(value: str) -> Any:
    value = value.strip()
    try:
        return json.loads(value)
    except Exception:
        pass
    try:
        return ast.literal_eval(value)
    except Exception:
        pass
    return value


def _tool_param_keys(tool: dict) -> tuple:
    """Return (required_keys, all_keys) for a tool's parameter schema."""
    fn = tool.get("function") or tool
    params = fn.get("parameters") or {}
    props = params.get("properties") or {}
    required = set(params.get("required") or [])
    all_keys = set(props.keys()) | required
    return required, all_keys


def _match_tool_by_args(args: Any, tools: List[dict]) -> Optional[str]:
    """Pick the tool whose parameter schema best matches the arg keys.

    Strategy:
      1. Filter to tools where required ⊆ arg_keys AND arg_keys ⊆ all_keys
         (a clean fit on both sides).
      2. If none, relax to require only required ⊆ arg_keys.
      3. Score by (matched_count, -schema_size) — prefer most overlap and
         the tightest schema on ties.
    """
    if not isinstance(args, dict) or not tools:
        return None
    arg_keys = set(args.keys())

    def _candidates(strict: bool):
        out = []
        for tool in tools:
            fn = tool.get("function") or tool
            name = fn.get("name")
            if not name:
                continue
            required, all_keys = _tool_param_keys(tool)
            if required and not required.issubset(arg_keys):
                continue
            if strict and (arg_keys - all_keys):
                continue
            matched = len(arg_keys & all_keys)
            out.append((name, matched, len(all_keys)))
        return out

    cands = _candidates(strict=True) or _candidates(strict=False)
    if not cands:
        return None
    cands.sort(key=lambda c: (-c[1], c[2]))
    return cands[0][0]


def _split_packed_calls(text: str) -> List[str]:
    """Split a body packing multiple tool calls separated by inner
    ``functions.NAME:N<|tool_call_argument_begin|>`` markers.

    The Q3 / REAP-pruned variants emit packed bodies of the form::

        functions.NAME0:0<|tool_call_argument_begin|>{args0}
        functions.NAME1:1<|tool_call_argument_begin|>{args1}
        functions.NAME2:2<|tool_call_argument_begin|>{args2}

    Each separator's *start* offset is the boundary between one call's
    args and the next call's id-prefix. Returns one
    ``"<id><|tool_call_argument_begin|><args>"`` substring per call.
    """
    text = text.strip()
    matches = list(_INNER_SEPARATOR.finditer(text))
    if len(matches) <= 1:
        return [text]
    # Boundaries between calls = the start of each separator after the first.
    boundaries = [m.start() for m in matches[1:]] + [len(text)]
    parts: List[str] = []
    cursor = 0
    for boundary in boundaries:
        parts.append(text[cursor:boundary].rstrip())
        cursor = boundary
    return parts


def _parse_single_tool(text: str, tools: Optional[list]) -> dict:
    """Parse one inner ``<|tool_call_begin|>…<|tool_call_end|>`` body."""
    text = _strip_markers(text)
    m = _STANDARD_ID.match(text)
    if m is not None:
        name = m.group("name")
        idx = m.group("idx")
        args = _deserialize(m.group("args"))
        return dict(id=f"functions.{name}:{idx}", name=name, arguments=args)

    m = _ANY_ID.match(text)
    if m is None:
        raise ValueError("No tool call found.")
    raw_id = m.group("id").strip()
    args = _deserialize(m.group("args"))

    inferred = None
    if isinstance(args, dict):
        inferred = _match_tool_by_args(args, tools or [])
    name = inferred or raw_id
    return dict(id=raw_id or f"functions.{name}:0", name=name, arguments=args)


def _parse_block(text: str, tools: Optional[list]) -> List[dict]:
    """Parse one block, splitting any packed multi-call body first."""
    text = _strip_markers(text)
    if len(_INNER_SEPARATOR.findall(text)) > 1:
        # Multiple tool calls packed into a single block
        try:
            sub_bodies = _split_packed_calls(text)
            out = []
            for body in sub_bodies:
                if not body.strip():
                    continue
                try:
                    out.append(_parse_single_tool(body, tools))
                except Exception:
                    continue
            if out:
                return out
        except Exception:
            pass
    return [_parse_single_tool(text, tools)]


def parse_tool_call(text: str, tools: Optional[list] = None):
    # Strip surrounding `<|tool_calls_section_begin|>…<|tool_calls_section_end|>`
    # wrapper if present, so block-level regexes don't have to deal with it.
    stripped = text.strip()
    if stripped.startswith(_SECTION_BEGIN_MARKER):
        stripped = stripped[len(_SECTION_BEGIN_MARKER):]
    if stripped.endswith(_SECTION_END_MARKER):
        stripped = stripped[: -len(_SECTION_END_MARKER)]
    stripped = stripped.strip()

    # First pass: strict `<|tool_call_begin|>…<|tool_call_end|>` blocks.
    blocks = _TOOL_CALL_BLOCK.findall(stripped)
    if blocks:
        out = []
        for b in blocks:
            out.extend(_parse_block(b, tools))
        return out

    # Second pass: lenient — accept blocks without a trailing
    # `<|tool_call_end|>`. Some Kimi-K2.6 variants drop the closer.
    blocks_lenient = _TOOL_CALL_BLOCK_LENIENT.findall(stripped)
    if blocks_lenient:
        out = []
        for b in blocks_lenient:
            if not b.strip():
                continue
            try:
                out.extend(_parse_block(b, tools))
            except Exception:
                continue
        if out:
            return out

    # Last resort: treat the whole text as one block body.
    return _parse_block(stripped, tools)
