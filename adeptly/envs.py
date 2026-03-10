"""Environment protocol helpers compatible with Gymnasium-style semantics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


class EnvironmentProtocol(Protocol):
    """Minimal runtime protocol for RL environments.

    Mirrors Gymnasium signatures while exposing a convenience ``done`` concept
    via :class:`StepResult`.
    """

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        """Start a fresh episode and return initial observation and info."""

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance one step and return observation, reward, terminated, truncated, info."""


@dataclass(slots=True)
class StepResult:
    """Convenience container for one environment transition."""

    observation: np.ndarray
    action: int
    reward: float
    done: bool
    info: dict[str, Any]
