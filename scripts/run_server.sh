#!/bin/zsh
# Resolve a model name to a path under ~/.lmstudio/models/sombra (or its
# converted/ sub-folder) and start `mlx-fun serve` on port 8899. Any extra
# arguments are forwarded to mlx-fun serve verbatim.
#
# Usage:
#   ./scripts/run_server.sh <model-name>                    # serve, idle-timeout 0
#   ./scripts/run_server.sh <model-name> --port 8080        # custom port
#   ./scripts/run_server.sh <model-name> --enable-counting  # extra flags
#
# Examples:
#   ./scripts/run_server.sh Gemma-4-26B-A4B-it-NVFP4
#   ./scripts/run_server.sh 31B-MXFP4              # fuzzy match
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <model-name> [extra mlx-fun serve flags]" >&2
    exit 1
fi

NAME="$1"; shift

ROOT="$HOME/.lmstudio/models/sombra"

# Resolution order:
#   1. exact under ROOT
#   2. exact under ROOT/converted
#   3. unique fuzzy match across both
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

MODEL=$(resolve "$NAME") || exit 1
echo "[run_server] $NAME -> $MODEL" >&2

# Pick a default port unless the user passed one.
PORT_FLAG="--port 8899"
for a in "$@"; do [[ "$a" == "--port" ]] && PORT_FLAG=""; done

LOG="/tmp/mlx_fun_server_$(basename "$MODEL").log"

cd "$(dirname "$0")/.."

exec uv run mlx-fun serve \
    $=PORT_FLAG \
    --model "$MODEL" \
    --idle-timeout 0 \
    --trust-remote-code \
    --log-level INFO \
    "$@" 2>&1 | tee "$LOG"
