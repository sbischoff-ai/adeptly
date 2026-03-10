"""Training and real-time inference utilities for DQN agents."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from adeptly.agents.dqn import DQNAgent
from adeptly.envs import EnvironmentProtocol


@dataclass(slots=True)
class TrainerConfig:
    """Configurable controls for DQN training loops."""

    total_steps: int = 10_000
    update_frequency: int = 1
    replay_warmup_steps: int = 1_000
    target_update_cadence: int = 1_000
    checkpoint_interval: int = 2_000
    checkpoint_dir: str = "checkpoints"
    evaluation_episodes: int = 5


class DQNTrainer:
    """Simple trainer that drives a ``DQNAgent`` in an environment."""

    def __init__(
        self,
        agent: DQNAgent,
        env_factory: Callable[[], EnvironmentProtocol],
        config: TrainerConfig,
    ) -> None:
        self.agent = agent
        self.env_factory = env_factory
        self.config = config
        self.agent.config.target_update_interval = config.target_update_cadence

    def train(self) -> dict[str, float]:
        env = self.env_factory()
        observation, _ = env.reset()
        episodes = 0

        for step in range(1, self.config.total_steps + 1):
            action = self.agent.predict_best_action(observation)
            next_observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            self.agent.remember(observation, action, reward, next_observation, done)
            observation = next_observation

            if step >= self.config.replay_warmup_steps and step % self.config.update_frequency == 0:
                self.agent.replay()

            if step % self.config.checkpoint_interval == 0:
                self.save_checkpoint(step)

            if done:
                episodes += 1
                observation, _ = env.reset()

        metrics = {"episodes": float(episodes), "evaluation_reward": self.evaluate()}
        return metrics

    def evaluate(self) -> float:
        total = 0.0
        previous_epsilon = self.agent.epsilon
        self.agent.epsilon = 0.0
        try:
            for _ in range(self.config.evaluation_episodes):
                env = self.env_factory()
                observation, _ = env.reset()
                done = False
                while not done:
                    action = self.agent.predict_best_action(observation)
                    observation, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    total += reward
        finally:
            self.agent.epsilon = previous_epsilon
        return total / max(self.config.evaluation_episodes, 1)

    def save_checkpoint(self, step: int) -> Path:
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = checkpoint_dir / f"dqn_step_{step}.pt"
        torch.save(
            {
                "online_network": self.agent.online_network.state_dict(),
                "target_network": self.agent.target_network.state_dict(),
                "optimizer": self.agent.optimizer.state_dict(),
                "global_step": self.agent.global_step,
                "epsilon": self.agent.epsilon,
            },
            path,
        )
        return path


class RealTimeInferenceLoop:
    """Decouple action-selection latency from training via buffered learner updates."""

    def __init__(self, agent: DQNAgent, train_every: int = 1, max_buffer_size: int = 10_000) -> None:
        self.agent = agent
        self.train_every = train_every
        self._buffer: deque[tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=max_buffer_size)
        self._ticks = 0

    def actor_step(self, observation: np.ndarray) -> int:
        """Low-latency actor path: only selects an action."""
        return self.agent.predict_best_action(observation)

    def submit_transition(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
    ) -> None:
        """Push transition for deferred learner consumption."""
        self._buffer.append((observation, action, reward, next_observation, done))

    def learner_update(self, max_updates: int = 1) -> list[float]:
        """Drain buffered transitions and run replay updates separately from actor."""
        losses: list[float] = []
        while self._buffer:
            transition = self._buffer.popleft()
            self.agent.remember(*transition)

        for _ in range(max_updates):
            self._ticks += 1
            if self._ticks % self.train_every != 0:
                continue
            loss = self.agent.replay()
            if loss is not None:
                losses.append(loss)
        return losses
