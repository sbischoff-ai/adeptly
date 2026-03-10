"""PyTorch-based Deep Q-Network agents and utilities."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Deque

import numpy as np
import torch
from torch import Tensor, nn


class QNetwork(nn.Module):
    """Small MLP used for value approximation in DQN."""

    def __init__(self, observation_size: int, action_size: int, hidden_sizes: tuple[int, ...] = (64, 64)) -> None:
        super().__init__()
        if observation_size <= 0:
            raise ValueError("observation_size must be > 0")
        if action_size <= 0:
            raise ValueError("action_size must be > 0")

        layers: list[nn.Module] = []
        in_features = observation_size
        for hidden_size in hidden_sizes:
            layers.extend((nn.Linear(in_features, hidden_size), nn.ReLU()))
            in_features = hidden_size
        layers.append(nn.Linear(in_features, action_size))
        self.model = nn.Sequential(*layers)

    def forward(self, observation: Tensor) -> Tensor:
        return self.model(observation)


class ReplayBuffer:
    """Replay buffer abstraction for storing and sampling transitions."""

    def __init__(self, capacity: int = 100_000) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self.capacity = capacity
        self._buffer: Deque[tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=capacity)

    def add(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
    ) -> None:
        self._buffer.append((observation, action, reward, next_observation, done))

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        transitions = random.sample(self._buffer, batch_size)
        observations, actions, rewards, next_observations, dones = zip(*transitions)
        return (
            np.asarray(observations, dtype=np.float32),
            np.asarray(actions, dtype=np.int64),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(next_observations, dtype=np.float32),
            np.asarray(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self._buffer)


class EpsilonScheduler:
    """Linear epsilon decay scheduler for epsilon-greedy exploration."""

    def __init__(self, start: float = 1.0, end: float = 0.05, decay_steps: int = 20_000) -> None:
        if decay_steps <= 0:
            raise ValueError("decay_steps must be > 0")
        self.start = start
        self.end = end
        self.decay_steps = decay_steps

    def value(self, step: int) -> float:
        progress = min(max(step, 0), self.decay_steps) / self.decay_steps
        return self.start + progress * (self.end - self.start)


@dataclass(slots=True)
class DQNConfig:
    """Configuration for :class:`DQNAgent`."""

    gamma: float = 0.99
    learning_rate: float = 1e-3
    batch_size: int = 64
    replay_capacity: int = 100_000
    min_replay_size: int = 1_000
    target_update_interval: int = 1_000
    soft_update_tau: float | None = None
    double_dqn: bool = True


class DQNAgent:
    """Modern DQN agent with replay buffer and target network support."""

    def __init__(
        self,
        observation_size: int,
        action_size: int,
        action_weights: list[float] | None = None,
        config: DQNConfig | None = None,
        epsilon_scheduler: EpsilonScheduler | None = None,
        device: str | None = None,
    ) -> None:
        self.observation_size = observation_size
        self.action_size = action_size
        self.action_weights = action_weights or [1.0 / action_size] * action_size

        self.config = config or DQNConfig()
        self.epsilon_scheduler = epsilon_scheduler or EpsilonScheduler()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.online_network = QNetwork(observation_size, action_size).to(self.device)
        self.target_network = QNetwork(observation_size, action_size).to(self.device)
        self.hard_update_target_network()
        self.optimizer = torch.optim.Adam(self.online_network.parameters(), lr=self.config.learning_rate)

        self.memory = ReplayBuffer(capacity=self.config.replay_capacity)
        self.global_step = 0
        self.epsilon = self.epsilon_scheduler.value(0)

    def remember(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        next_observation: np.ndarray,
        done: bool,
    ) -> None:
        """Store one transition in replay memory."""
        self.memory.add(
            np.asarray(observation, dtype=np.float32),
            action,
            reward,
            np.asarray(next_observation, dtype=np.float32),
            done,
        )

    def predict_best_action(self, observation: np.ndarray) -> int:
        """Select action using epsilon-greedy policy."""
        if np.random.random() < self.epsilon:
            return random.choices(range(self.action_size), weights=self.action_weights, k=1)[0]

        observation_tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device).view(1, -1)
        with torch.no_grad():
            q_values = self.online_network(observation_tensor)
        return int(torch.argmax(q_values, dim=1).item())

    def replay(self, batch_size: int | None = None) -> float | None:
        """Run one optimization step from replay memory."""
        train_batch_size = batch_size or self.config.batch_size
        if len(self.memory) < max(train_batch_size, self.config.min_replay_size):
            return None

        observations, actions, rewards, next_observations, dones = self.memory.sample(train_batch_size)

        obs_t = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        next_obs_t = torch.as_tensor(next_observations, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        q_values = self.online_network(obs_t).gather(1, actions_t).squeeze(1)

        with torch.no_grad():
            if self.config.double_dqn:
                next_actions = self.online_network(next_obs_t).argmax(dim=1, keepdim=True)
                next_q_values = self.target_network(next_obs_t).gather(1, next_actions).squeeze(1)
            else:
                next_q_values = self.target_network(next_obs_t).max(dim=1).values
            targets = rewards_t + (1.0 - dones_t) * self.config.gamma * next_q_values

        loss = nn.functional.mse_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.global_step += 1
        self.epsilon = self.epsilon_scheduler.value(self.global_step)

        if self.config.soft_update_tau is not None:
            self.soft_update_target_network(self.config.soft_update_tau)
        elif self.global_step % self.config.target_update_interval == 0:
            self.hard_update_target_network()

        return float(loss.item())

    def hard_update_target_network(self) -> None:
        """Copy online network weights to target network."""
        self.target_network.load_state_dict(self.online_network.state_dict())

    def soft_update_target_network(self, tau: float) -> None:
        """Polyak averaging update for target network."""
        if not 0.0 < tau <= 1.0:
            raise ValueError("tau must be in (0, 1]")
        with torch.no_grad():
            for target_param, online_param in zip(self.target_network.parameters(), self.online_network.parameters()):
                target_param.data.mul_(1.0 - tau).add_(tau * online_param.data)
