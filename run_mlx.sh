#!/usr/bin/env bash
# Vanilla mlx_lm.server — same model and chat template as run.sh,
# but no mlx_fun hooks. Used to isolate whether the GLM-5.1 tool-call
# divergence comes from mlx_fun's gate counters or from mlx-lm itself.
#
# Send requests with  "model": "default_model"  to hit the loaded model.

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

CHAT_TEMPLATE_FILE="src/mlx_fun/templates/glm51.jinja"

.venv/bin/python -m mlx_lm.server \
    --host 0.0.0.0 \
    --port 8898 \
    --model /Users/sombrax/.lmstudio/models/sombra/GLM-5.1-Q3-g32 \
    --chat-template "$(cat "$CHAT_TEMPLATE_FILE")" \
    --log-level INFO 2>&1 | tee /tmp/mlx_lm_server.log
