"""MLX-FUN: MoE model compression, analysis, and domain specialization for Apple Silicon."""

__version__ = "0.1.0"

# Publish mlx_fun's out-of-tree model types (e.g. ``gemma4_assistant``) under
# ``mlx_lm.models.*`` so stock upstream mlx-lm can load them. Keeps mlx_fun off
# a forked mlx-lm. Safe to call before mlx-lm is importable — it no-ops.
from .models import register_model_types as _register_model_types

_register_model_types()

del _register_model_types
