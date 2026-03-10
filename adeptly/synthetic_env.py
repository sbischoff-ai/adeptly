"""Synthetic reference environments for validation and examples."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class CounterEnv:
    """Small deterministic environment suitable for smoke tests.

    State is a single scalar in ``[0, target]``. Action ``1`` increments and
    ``0`` decrements (clipped to bounds). Reward is +1 on reaching target, else
    a small negative shaping reward.
    """

    target: int = 5
    max_steps: int = 20
    _state: int = field(init=False, default=0)
    _steps: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self._state = 0
        self._steps = 0

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            np.random.seed(seed)
        self._state = 0
        self._steps = 0
        return np.array([float(self._state)], dtype=np.float32), {"target": self.target}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self._steps += 1
        if action == 1:
            self._state = min(self.target, self._state + 1)
        else:
            self._state = max(0, self._state - 1)

        terminated = self._state >= self.target
        truncated = self._steps >= self.max_steps and not terminated
        reward = 1.0 if terminated else -0.01
        info = {"steps": self._steps, "state": self._state}
        return np.array([float(self._state)], dtype=np.float32), reward, terminated, truncated, info
