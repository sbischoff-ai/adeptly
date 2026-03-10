"""Spawn adaptive agents in real-time environments."""

from __future__ import annotations

import warnings
from contextlib import nullcontext

from adeptly.agents.dqn import DQNAgent


class AdeptlyEngine:
    """Deprecated no-op context manager kept for backward compatibility."""

    def __new__(cls):
        warnings.warn(
            "AdeptlyEngine is deprecated and now acts as a no-op context manager. "
            "PyTorch-based agents no longer require TensorFlow graph scopes.",
            DeprecationWarning,
            stacklevel=2,
        )
        return nullcontext()


__all__ = ["AdeptlyEngine", "DQNAgent"]
