"""Out-of-tree model types that mlx_fun contributes to mlx-lm.

mlx-lm resolves a checkpoint's architecture with
``importlib.import_module(f"mlx_lm.models.{model_type}")`` (see
``mlx_lm.utils._get_classes``). Since ``import_module`` consults
``sys.modules`` first, inserting a module under that name makes stock
``mlx_lm.load()`` pick up a model class that lives in mlx_fun — no fork of
mlx-lm, and no changes at any call site.

``register_model_types()`` is called from ``mlx_fun/__init__.py``, so merely
importing anything from mlx_fun (CLI, server, tests) is enough. Upstream
always wins: a model type mlx-lm already ships is left untouched, so these
registrations retire themselves as the types land upstream.
"""

from __future__ import annotations

import importlib
import logging
import sys

# model_type -> module inside this package implementing it.
_OUT_OF_TREE = {
    "gemma4_assistant": "mlx_fun.models.gemma4_assistant",
}

_registered: list[str] = []


def register_model_types() -> list[str]:
    """Publish mlx_fun's model types under ``mlx_lm.models.*``.

    Idempotent. Returns the model types this call registered (empty when
    they were already present, either from a prior call or from upstream).
    """
    newly: list[str] = []

    for model_type, source in _OUT_OF_TREE.items():
        target = f"mlx_lm.models.{model_type}"

        if target in sys.modules:
            continue

        # Prefer upstream if it has since shipped this model type.
        try:
            importlib.import_module(target)
            continue
        except ImportError:
            pass

        try:
            sys.modules[target] = importlib.import_module(source)
        except ImportError as e:
            # mlx-lm missing entirely, or the model module failed to import.
            # Not fatal: only the drafter path needs it.
            logging.debug(f"could not register model type '{model_type}': {e}")
            continue

        newly.append(model_type)

    _registered.extend(newly)
    return newly


def registered_model_types() -> list[str]:
    """Model types this process registered into ``mlx_lm.models``."""
    return list(_registered)


__all__ = ["register_model_types", "registered_model_types"]
