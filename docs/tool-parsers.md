# Tool-call parsers

mlx-lm ships a per-model-family library of tool-call parsers under
`mlx_lm.tool_parsers.*`. The right parser is auto-selected at server load by
matching distinctive marker substrings against the loaded chat template
(`mlx_lm.tokenizer_utils._infer_tool_parser`):

| Marker substring in template | Parser dispatched |
|---|---|
| `<minimax:tool_call>` | `minimax_m2` |
| `<\|tool_call>` and `<tool_call\|>` | `gemma4` |
| `<start_function_call>` | `function_gemma` |
| `<longcat_tool_call>` | `longcat` |
| `<arg_key>` | `glm47` (GLM-4.5/4.7) |
| `<\|tool_list_start\|>` | `pythonic` |
| `<tool_call>\n<function=` | `qwen3_coder` |
| `<\|tool_calls_section_begin\|>` | `kimi_k2` |
| `[TOOL_CALLS]` | `mistral` |
| `<tool_call>` + `tool_call.name` | `json_tools` |

When the model emits its tool call(s), mlx-lm's streaming server runs a
**state machine** keyed off the parser's `tool_call_start` / `tool_call_end`
constants. Tokens arriving while in the "tool" state are buffered and
ultimately handed to `parser.parse_tool_call(text, tools)`, which returns a
list of OpenAI-style `{id, name, arguments}` dicts.

## Why this can fail

Two classes of failure show up in practice:

1. **Parser mismatch** — the model emits a format the parser doesn't accept
   (different id shape, missing function name, wrong argument-begin marker).
   `parse_tool_call` raises and the agent sees no tool calls.
2. **Streaming-boundary mismatch** — the model emits tool-call markers
   different from the ones the state machine listens for, so the buffer never
   flips into "tool" mode and the raw tokens leak through as content text.
   The agent loop terminates because no tool call was extracted.

Aggressive quantization (Q3 g64, INT4-then-Q3 streaming, REAP-pruned variants)
makes both failure modes more frequent: format adherence is the first thing to
go when you cut precision and prune experts.

## Kimi-K2.6 case study

mlx_fun ships a permissive replacement parser at
`src/mlx_fun/kimi_k26_tool_parser.py` and installs it automatically for any
model with `model_type == "kimi_k25"`. It addresses three observed failure
modes layered on top of one another. The wiring lives in
`src/mlx_fun/server.py:_do_load`:

```python
if model_type == "kimi_k25":
    from . import kimi_k26_tool_parser
    tokenizer._tool_parser = kimi_k26_tool_parser.parse_tool_call
    tokenizer._tool_call_start = "<|tool_call_begin|>"
    tokenizer._tool_call_end = "<|tool_call_end|>"
```

### Failure 1 — `tool_call_N` ids without function names

Standard Kimi-K2 format (what the upstream parser expects):

```
<|tool_call_begin|>functions.grep:0<|tool_call_argument_begin|>{"pattern": "x"}<|tool_call_end|>
```

What Q3-g64 quants actually emit:

```
<|tool_call_begin|>tool_call_1<|tool_call_argument_begin|>{"pattern": "x", "glob": "**/*.sol"}<|tool_call_end|>
```

The id lacks the function name — just a sequential index. The upstream regex
`(?:functions\.)?(.+?):\d+` requires `name:N`, fails to match, and raises
"No tool call found."

**Fix:** the permissive parser tries the standard format first, and falls back
to a generic id-then-args regex (`_ANY_ID`). When the id has no function name,
it infers the name by matching the emitted argument keys against each
available tool's parameter schema:

```python
def _match_tool_by_args(args, tools):
    arg_keys = set(args.keys())
    # 1. Strict candidates: required ⊆ arg_keys AND arg_keys ⊆ all_keys
    # 2. Relaxed: only require required ⊆ arg_keys
    # Score by (matched_count, -schema_size); tightest fit wins on ties.
```

For `{"pattern": "x", "glob": "**/*.sol"}` with tools `[read, glob, grep]`:
- `read` has required `{file_path}` — not subset of `{pattern, glob}`, eliminated.
- `glob` has required `{pattern}`, all `{pattern}` — `glob` arg unaccounted, fewer matches.
- `grep` has required `{pattern}`, all `{pattern, glob}` — perfect fit, wins.

### Failure 2 — Multiple calls packed into one block

Worse case observed in the wild:

```
<|tool_call_begin|>functions.glob:0<|tool_call_argument_begin|>{"pattern": "**/*.sol"}functions.glob:1<|tool_call_argument_begin|>{"pattern": "**/POST_MORTEM.md"}functions.grep:2<|tool_call_argument_begin|>{"pattern": "function cook", "glob": "**/*.sol"}<|tool_call_end|>
```

Three tool calls concatenated into a single `<|tool_call_begin|>…<|tool_call_end|>`
block. The next call's id-prefix appears inside what should be the previous
call's args string. SAC-style agents see one malformed args field of length
~407 and reject the whole turn.

**Fix:** detect packed bodies via an inner-separator regex, split on each
separator's start offset, and parse each slice independently:

```python
_INNER_SEPARATOR = re.compile(
    r"(?:functions\.)?[A-Za-z_][\w]*:\d+\s*<\|tool_call_argument_begin\|>",
    re.DOTALL,
)

def _split_packed_calls(text):
    matches = list(_INNER_SEPARATOR.finditer(text))
    if len(matches) <= 1:
        return [text]
    boundaries = [m.start() for m in matches[1:]] + [len(text)]
    parts, cursor = [], 0
    for boundary in boundaries:
        parts.append(text[cursor:boundary].rstrip())
        cursor = boundary
    return parts
```

Each split slice is then routed back through `_parse_single_tool` (which
handles both standard and nameless ids).

### Failure 3 — Streaming boundaries

The upstream `kimi_k2` parser declares its state-machine boundaries as the
**section** markers:

```python
tool_call_start = "<|tool_calls_section_begin|>"
tool_call_end   = "<|tool_calls_section_end|>"
```

But Q3-g64 quants frequently emit the per-call `<|tool_call_begin|>` /
`<|tool_call_end|>` tokens **without** a surrounding section wrapper. The
state machine never flips into "tool" state, the tokens leak through as
content text, and the agent never sees any tool call at all.

**Fix:** retarget the state machine to the per-call markers themselves. Now
each `<|tool_call_begin|>…<|tool_call_end|>` flips state-tool-state; the
buffer captured between flips is the call body. This works whether or not the
section wrapper is present.

There's a side effect: when the state machine flips on the per-call markers,
the captured buffer **starts with `<|tool_call_begin|>`** (the matching token
belongs to the post-transition state) but **ends without `<|tool_call_end|>`**
(that token's match transitioned state to "normal" so it's routed to content
text). The parser's `_strip_markers` accommodates this:

```python
def _strip_markers(text):
    text = text.strip()
    if text.startswith(_SECTION_BEGIN_MARKER): text = text[len(_SECTION_BEGIN_MARKER):]
    if text.endswith(_SECTION_END_MARKER):     text = text[:-len(_SECTION_END_MARKER)]
    text = text.strip()
    if text.startswith(_BEGIN_MARKER):         text = text[len(_BEGIN_MARKER):]
    if text.endswith(_END_MARKER):             text = text[:-len(_END_MARKER)]
    return text.strip()
```

The trailing `<|tool_call_end|>` token does end up in the visible content
stream as a side effect — most agents treat raw `<|...|>` tokens as cosmetic
noise and ignore them. If a downstream agent is strict about content
cleanliness, post-filter `<|tool_call_end|>` from the response text.

### Test coverage

The four streaming-buffer shapes the parser must handle:

| Shape | Source |
|---|---|
| Standard `functions.NAME:N<arg_begin>{...}` block | Properly behaving model |
| Nameless `tool_call_N<arg_begin>{...}` block | Q3-g64 quirk (Failure 1) |
| Inner-only (no markers either side) | streaming state machine slice |
| Begin marker present, no end marker | streaming with retargeted boundaries |
| Multiple back-to-back blocks | normal multi-call response |
| Packed multi-call in one block | Q3-g64 quirk (Failure 2) |

A smoke test covering all six shapes lives at the bottom of
`kimi_k26_tool_parser.py`'s docstring. Re-run it after any parser edit:

```python
from mlx_fun.kimi_k26_tool_parser import parse_tool_call
# (see test cases in src/mlx_fun/kimi_k26_tool_parser.py)
```

## Risk audit across other model families

The "packed multi-call" failure (Failure 2) is the most general — it can
recur on any parser whose format consists of an outer wrapper containing
**repeating inner `id<sep>args` units** without per-unit closing markers. Of
the parsers shipping in mlx-lm:

| Parser | Outer markers | Inner separator | Packing risk |
|---|---|---|---|
| **kimi_k2** | section | `<\|tool_call_begin\|>NAME:N<\|tool_call_argument_begin\|>` | **HIGH** — confirmed; fixed in `kimi_k26_tool_parser` |
| **minimax_m2** | `<minimax:tool_call>` / `</…>` per call | one JSON object per block | **MEDIUM** — would need MiniMax to emit two JSONs back-to-back inside one wrapper; not yet observed |
| **gemma4 / glm47 / qwen3_coder / json_tools** | XML-style `<tool_call>` / `</tool_call>` per call | none — wrapping IS the call boundary | **LOW** — packing requires dropping a closing tag and the next opening tag, very unusual |
| **function_gemma** | `<start_function_call>` / `<end_function_call>` per call | none | **LOW** |
| **longcat** | `<longcat_tool_call>` / `</…>` per call | none | **LOW** |
| **mistral** | `[TOOL_CALLS]` then JSON array | the JSON array is multi-call native | **LOW** |
| **pythonic** | `<\|tool_call_start\|>` / `<\|tool_call_end\|>` enclosing a Python list | the list is multi-call native | **LOW** |

Models worth watching:

- **Kimi-Linear**, **Kimi-VL** — likely share Kimi-K2's tool format.
- A heavily quantized / pruned **MiniMax-M2** could potentially emit two JSON
  objects inside one `<minimax:tool_call>` wrapper if the training corpus had
  any such examples.
- Any future Moonshot follow-on (Kimi-K3 etc.).

## Adding a permissive parser for another model

If you observe similar failures on a different family, the recipe is:

1. **Identify the inner anchor** the model uses to start each call inside a
   block (e.g. for Kimi: `functions.NAME:N<|tool_call_argument_begin|>`).
2. **Write a custom parser** following `kimi_k26_tool_parser.py`'s structure:
   - `_TOOL_CALL_BLOCK` regex over the outer markers.
   - `_INNER_SEPARATOR` regex over the inner anchor.
   - `_split_packed_calls` to slice on each inner anchor's start.
   - `_parse_single_tool` with both a standard regex and a nameless fallback.
   - `_match_tool_by_args` to recover the function name from arg keys.
3. **Wire it in `server.py:_do_load`** by swapping `tokenizer._tool_parser`
   for matching `model_type` and (if the streaming boundaries are wrong)
   overriding `tokenizer._tool_call_start` / `_tool_call_end`.

The arg-shape inference (`_match_tool_by_args`) is the most general piece —
lift it into a helper module if you end up writing more than one of these.
