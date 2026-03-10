import pytest
import numpy as np
import torch

from adeptly.agents.dqn import DQNAgent, DQNConfig, EpsilonScheduler, ReplayBuffer


def test_replay_buffer_sample_shapes():
    buffer = ReplayBuffer(capacity=10)
    for i in range(5):
        obs = np.array([i], dtype=np.float32)
        next_obs = np.array([i + 1], dtype=np.float32)
        buffer.add(obs, 0, 1.0, next_obs, False)

    observations, actions, rewards, next_observations, dones = buffer.sample(4)
    assert observations.shape == (4, 1)
    assert actions.shape == (4,)
    assert rewards.shape == (4,)
    assert next_observations.shape == (4, 1)
    assert dones.shape == (4,)


def test_epsilon_scheduler_linear_decay():
    scheduler = EpsilonScheduler(start=1.0, end=0.1, decay_steps=10)
    assert scheduler.value(0) == 1.0
    assert scheduler.value(10) == pytest.approx(0.1)
    assert scheduler.value(20) == pytest.approx(0.1)


def test_dqn_agent_replay_updates_network_and_epsilon():
    config = DQNConfig(min_replay_size=4, batch_size=4, target_update_interval=2)
    agent = DQNAgent(observation_size=1, action_size=2, config=config)

    for i in range(6):
        obs = np.array([float(i)], dtype=np.float32)
        next_obs = np.array([float(i + 1)], dtype=np.float32)
        agent.remember(obs, i % 2, 1.0, next_obs, False)

    before = [param.detach().clone() for param in agent.online_network.parameters()]
    loss = agent.replay()
    assert loss is not None
    assert agent.epsilon < 1.0

    after = list(agent.online_network.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after))


def test_double_dqn_disabled_path_runs():
    config = DQNConfig(min_replay_size=4, batch_size=4, double_dqn=False)
    agent = DQNAgent(observation_size=1, action_size=2, config=config)

    for i in range(4):
        obs = np.array([float(i)], dtype=np.float32)
        next_obs = np.array([float(i + 1)], dtype=np.float32)
        agent.remember(obs, i % 2, 0.5, next_obs, i == 3)

    assert agent.replay() is not None
