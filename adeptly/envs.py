"""Environment protocol helpers compatible with Gymnasium-style semantics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

import numpy as np

ObservationT = TypeVar("ObservationT", bound=np.ndarray)
ActionT = TypeVar("ActionT")
InfoT = TypeVar("InfoT", bound=dict[str, Any])


class EnvironmentProtocol(Protocol[ObservationT, ActionT, InfoT]):
    """Minimal runtime protocol for RL environments.

    Mirrors Gymnasium signatures while exposing a convenience ``done`` concept
    via :class:`StepResult`.
    """

    def reset(self, *, seed: int | None = None) -> tuple[ObservationT, InfoT]:
        """Start a fresh episode and return initial observation and info."""

    def step(self, action: ActionT) -> tuple[ObservationT, float, bool, bool, InfoT]:
        """Advance one step and return observation, reward, terminated, truncated, info."""


@dataclass(slots=True)
class StepResult:
    """Convenience container for one environment transition."""

    observation: np.ndarray
    reward: float
    done: bool
    info: dict[str, Any]
