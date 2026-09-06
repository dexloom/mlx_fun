"""Unified checkpoint loading: mlx-lm for text models, mlx-vlm for vision ones.

Vision-language checkpoints (Qwen3.8-Flash-Next / ``qwen4_exp``, Qwen3-VL,
GLM-4V, …) carry a vision tower next to the language stack and a nested
``text_config``. mlx-lm does not implement them; ``mlx-vlm`` does. Since
mlx_fun's analysis stack (REAP saliency, SAFEx, domain scan, steering) only
ever touches the *language* half — the MoE blocks, their gates, and their
SwitchGLU experts — a VLM is a perfectly ordinary analysis target once it is
loaded. This module is the seam that loads it.

``load_model()`` mirrors ``mlx_lm.load(..., return_config=True)`` so call sites
change by one line, and routes to whichever backend can handle the checkpoint.

mlx-vlm is an optional dependency: ``pip install "mlx-fun[vlm]"``. Text models
never import it.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional, Tuple

# Model types that only mlx-vlm implements, even though their config carries no
# ``vision_config`` (or carries one we would otherwise not recognise).
_VLM_ONLY_MODEL_TYPES = {
    "qwen4_exp",     # Qwen/Qwen3.8-Flash-Next
    "qwen3_5_moe",   # Qwen/Qwen3.6-35B-A3B and the rest of the Qwen3.5/3.6 MoE line
    "glm5_next",     # zai-org/GLM-5.3-Flash
}

_VLM_INSTALL_HINT = (
    'mlx-vlm is required for vision-language checkpoints. Install it with:\n'
    '    uv pip install "mlx-fun[vlm]"\n'
    "    (or: uv pip install mlx-vlm)"
)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def read_config(model_path: str) -> dict:
    """Read a checkpoint's ``config.json`` without loading any weights.

    Accepts a local directory or a HuggingFace repo id.
    """
    expanded = os.path.expanduser(model_path)
    local = Path(expanded) / "config.json"
    if local.is_file():
        with open(local) as f:
            return json.load(f)

    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=model_path, filename="config.json")
    with open(path) as f:
        return json.load(f)


def is_vision_model(config: dict) -> bool:
    """True when a checkpoint pairs a vision tower with its language stack.

    Detected by an explicit ``vision_config`` block, an image-token id (models
    that inline their vision config), or membership in the mlx-vlm-only list.
    """
    if config.get("model_type") in _VLM_ONLY_MODEL_TYPES:
        return True
    if isinstance(config.get("vision_config"), dict) and config["vision_config"]:
        return True
    return "image_token_id" in config and "text_config" in config


def text_config(config: dict) -> dict:
    """Return the language-model half of a possibly-nested config.

    Multimodal checkpoints nest the language hyperparameters under
    ``text_config``; text-only ones keep them at the top level.
    """
    nested = config.get("text_config")
    if isinstance(nested, dict) and nested:
        return nested
    return config


def language_model(model: Any) -> Any:
    """Return the text stack of a possibly-multimodal model.

    mlx-vlm wraps ``vision_tower`` and ``language_model`` in one module; calling
    the wrapper expects pixel values. Analysis passes are text-only, so they run
    against the language stack directly.
    """
    return getattr(model, "language_model", model)


def text_forward(model: Any, config: dict) -> Any:
    """Return the callable to use for a text-only (token ids) forward pass.

    Calibration and routing scans feed token ids only. A multimodal wrapper's
    ``__call__`` expects pixel values alongside them, so vision checkpoints run
    against their language stack instead. Text models are returned untouched.
    """
    if is_vision_model(config):
        return language_model(model)
    return model


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_model(
    model_path: str,
    *,
    tokenizer_config: Optional[dict] = None,
    trust_remote_code: bool = False,
    lazy: bool = False,
) -> Tuple[Any, Any, dict]:
    """Load a checkpoint with whichever backend implements it.

    Returns ``(model, tokenizer, config)``, matching
    ``mlx_lm.load(..., return_config=True)``. For vision checkpoints the second
    element is mlx-vlm's processor, which exposes the same ``encode`` /
    ``decode`` / ``apply_chat_template`` surface the calibration paths use.

    ``tokenizer_config["chat_template"]`` is honoured on both backends: mlx-lm
    consumes it directly, and on the mlx-vlm path it is assigned to the returned
    tokenizer after loading, since ``mlx_vlm.load`` takes no tokenizer config.
    """
    expanded = os.path.expanduser(model_path)
    if os.path.exists(expanded):
        model_path = expanded

    try:
        config = read_config(model_path)
    except Exception as e:
        # Fall through to mlx-lm, whose error messages for a bad path/repo id
        # are the ones callers already handle.
        logging.debug(f"could not pre-read config for '{model_path}': {e}")
        config = {}

    if config and is_vision_model(config):
        return _load_vlm(model_path, config, lazy=lazy,
                         trust_remote_code=trust_remote_code,
                         tokenizer_config=tokenizer_config)

    return _load_text(model_path, tokenizer_config=tokenizer_config,
                      trust_remote_code=trust_remote_code, lazy=lazy)


def _load_text(model_path, *, tokenizer_config, trust_remote_code, lazy):
    from mlx_lm import load as mlx_load

    tok_cfg = dict(tokenizer_config or {})
    if trust_remote_code:
        tok_cfg["trust_remote_code"] = True

    return mlx_load(
        model_path,
        tokenizer_config=tok_cfg or None,
        lazy=lazy,
        return_config=True,
    )


def _load_vlm(model_path, config, *, lazy, trust_remote_code, tokenizer_config=None):
    try:
        from mlx_vlm import load as vlm_load
    except ImportError as e:
        raise ImportError(
            f"'{config.get('model_type')}' is a vision-language checkpoint. "
            f"{_VLM_INSTALL_HINT}"
        ) from e

    logging.info(
        f"Loading vision-language model via mlx-vlm: "
        f"{config.get('model_type')} ({model_path})"
    )
    model, processor = vlm_load(
        model_path, lazy=lazy, trust_remote_code=trust_remote_code,
    )

    # mlx-vlm hands back a processor; the calibration/dataset paths want the
    # tokenizer surface. Processors proxy encode/decode to .tokenizer, but be
    # explicit so callers that introspect get the real thing.
    tokenizer = getattr(processor, "tokenizer", processor)

    # mlx_vlm.load takes no tokenizer_config, so an override has to be applied
    # after the fact. Only chat_template is honoured here — the rest of a
    # tokenizer_config would have to reach the constructor to have any effect.
    chat_template = (tokenizer_config or {}).get("chat_template")
    if chat_template:
        tokenizer.chat_template = chat_template
        logging.info("Applied an explicit chat template to the mlx-vlm tokenizer")

    return model, tokenizer, config


__all__ = [
    "is_vision_model",
    "language_model",
    "load_model",
    "read_config",
    "text_config",
    "text_forward",
]
