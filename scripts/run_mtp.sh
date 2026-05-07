#!/bin/zsh
# Run a one-shot MTP speculative-decoding generation. Resolves backbone +
# drafter names from ~/.lmstudio/models/sombra (and converted/). If only a
# backbone name is given, the drafter is auto-derived by inserting
# "-assistant" before the quant suffix.
#
# Usage:
#   ./scripts/run_mtp.sh <backbone-name> "<prompt>"
#   ./scripts/run_mtp.sh <backbone-name> <drafter-name> "<prompt>"
#   ./scripts/run_mtp.sh <backbone-name> "<prompt>" --baseline -k 4 -n 256
#
# Examples:
#   ./scripts/run_mtp.sh Gemma-4-31B-it-NVFP4 "Write a haiku about the moon."
#   ./scripts/run_mtp.sh 26B-A4B-NVFP4 "Hello!" --baseline
set -euo pipefail

ROOT="$HOME/.lmstudio/models/sombra"

resolve() {
    local n="$1"
    if [[ -d "$ROOT/$n" ]]; then echo "$ROOT/$n"; return; fi
    if [[ -d "$ROOT/converted/$n" ]]; then echo "$ROOT/converted/$n"; return; fi
    local matches=()
    # (N) is the zsh "null glob" qualifier — silently yields nothing if
    # there are no matches instead of erroring.
    for d in "$ROOT"/*"$n"*(N) "$ROOT"/converted/*"$n"*(N); do
        [[ -d "$d" ]] && matches+=("$d")
    done
    if [[ ${#matches[@]} -eq 1 ]]; then echo "${matches[1]}"; return; fi
    if [[ ${#matches[@]} -eq 0 ]]; then
        echo "ERR: no model under $ROOT matches '$n'" >&2
        return 1
    fi
    echo "ERR: '$n' is ambiguous; matches:" >&2
    for m in "${matches[@]}"; do echo "  $m" >&2; done
    return 1
}

# Derive drafter name from backbone name: insert "-assistant" before the
# quantization suffix. Recognised suffixes: -bf16, -MXFP8, -MXFP4, -NVFP4.
derive_drafter() {
    local bb="$1"
    local stem="${bb%-bf16}"; [[ "$stem" != "$bb" ]] && { echo "${stem}-assistant-bf16"; return; }
    local stem="${bb%-MXFP8}"; [[ "$stem" != "$bb" ]] && { echo "${stem}-assistant-MXFP8"; return; }
    local stem="${bb%-MXFP4}"; [[ "$stem" != "$bb" ]] && { echo "${stem}-assistant-MXFP4"; return; }
    local stem="${bb%-NVFP4}"; [[ "$stem" != "$bb" ]] && { echo "${stem}-assistant-NVFP4"; return; }
    # Also handle the LM Studio-cased "-it-MXFP4" pattern that converts to
    # "-it-assistant-MXFP4". The substitutions above already cover that
    # because they only strip the trailing quant.
    echo ""  # caller checks for empty
}

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <backbone-name> [<drafter-name>] \"<prompt>\" [extra flags]" >&2
    exit 1
fi

BB_NAME="$1"; shift

# If the next arg looks like another model name (no spaces, no leading dash),
# treat it as an explicit drafter.
DF_NAME=""
case "$1" in
    -*|*\ *|*\?*) ;;
    *)  fuzzy=( "$ROOT"/*"$1"*(N) "$ROOT"/converted/*"$1"*(N) )
        if [[ -d "$ROOT/$1" || -d "$ROOT/converted/$1" || ${#fuzzy[@]} -gt 0 ]]; then
            DF_NAME="$1"; shift
        fi
        ;;
esac

if [[ $# -lt 1 ]]; then
    echo "ERR: missing prompt." >&2
    exit 1
fi

PROMPT="$1"; shift

BB_PATH=$(resolve "$BB_NAME") || exit 1
if [[ -z "$DF_NAME" ]]; then
    DF_NAME=$(derive_drafter "$(basename "$BB_PATH")")
    if [[ -z "$DF_NAME" ]]; then
        echo "ERR: could not auto-derive drafter from '$BB_NAME'." >&2
        echo "     Pass it explicitly as the second argument." >&2
        exit 1
    fi
fi
DF_PATH=$(resolve "$DF_NAME") || exit 1

echo "[run_mtp] backbone:  $BB_PATH" >&2
echo "[run_mtp] drafter:   $DF_PATH" >&2

cd "$(dirname "$0")/.."

exec uv run python -m mlx_fun.mtp_driver \
    --backbone "$BB_PATH" \
    --drafter  "$DF_PATH" \
    --prompt   "$PROMPT" \
    "$@"
