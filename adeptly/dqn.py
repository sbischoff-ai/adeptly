"""Deprecated compatibility shim for DQN agent imports."""

from __future__ import annotations

import warnings

from adeptly.agents.dqn import DQNAgent

warnings.warn(
    "Importing DQNAgent from adeptly.dqn is deprecated and will be removed in a future release. "
    "Use `from adeptly.agents.dqn import DQNAgent` instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["DQNAgent"]
