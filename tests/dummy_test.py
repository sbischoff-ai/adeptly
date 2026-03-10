import random

import numpy as np
import pytest
import torch

from adeptly.agents.dqn import DQNAgent, DQNConfig, EpsilonScheduler, ReplayBuffer
from adeptly.synthetic_env import CounterEnv
from adeptly.trainer import DQNTrainer, TrainerConfig


def _set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def test_replay_buffer_sampling_is_deterministic_with_fixed_seed() -> None:
    buffer = ReplayBuffer(capacity=10)
    for i in range(6):
        obs = np.array([i], dtype=np.float32)
        next_obs = np.array([i + 1], dtype=np.float32)
        buffer.add(obs, i % 2, float(i), next_obs, i == 5)

    random.seed(7)
    sample_one = buffer.sample(4)
    random.seed(7)
    sample_two = buffer.sample(4)

    for left, right in zip(sample_one, sample_two):
        np.testing.assert_allclose(left, right)


def test_epsilon_greedy_action_selection_is_deterministic_for_explore_and_exploit() -> None:
    _set_all_seeds(11)
    agent = DQNAgent(
        observation_size=1,
        action_size=2,
        action_weights=[0.0, 1.0],
        config=DQNConfig(min_replay_size=1, batch_size=1),
    )

    # Exploration path: weighted sampling should always choose action 1.
    agent.epsilon = 1.0
    explore_action = agent.predict_best_action(np.array([0.0], dtype=np.float32))
    assert explore_action == 1

    # Exploitation path: make argmax deterministic via fixed network outputs.
    with torch.no_grad():
        for parameter in agent.online_network.parameters():
            parameter.zero_()
        final_linear = agent.online_network.model[-1]
        assert isinstance(final_linear, torch.nn.Linear)
        final_linear.bias.copy_(torch.tensor([0.25, 1.25]))

    agent.epsilon = 0.0
    exploit_action = agent.predict_best_action(np.array([0.0], dtype=np.float32))
    assert exploit_action == 1


def test_replay_target_calculation_and_learning_step_invariants() -> None:
    _set_all_seeds(13)
    scheduler = EpsilonScheduler(start=0.9, end=0.3, decay_steps=10)
    config = DQNConfig(
        gamma=0.5,
        learning_rate=0.0,
        min_replay_size=1,
        batch_size=1,
        target_update_interval=1,
        double_dqn=False,
    )
    agent = DQNAgent(observation_size=1, action_size=2, config=config, epsilon_scheduler=scheduler)

    with torch.no_grad():
        for parameter in agent.online_network.parameters():
            parameter.zero_()
        for parameter in agent.target_network.parameters():
            parameter.zero_()
        target_output = agent.target_network.model[-1]
        assert isinstance(target_output, torch.nn.Linear)
        target_output.bias.copy_(torch.tensor([2.0, 1.0]))

    agent.remember(
        observation=np.array([0.0], dtype=np.float32),
        action=1,
        reward=1.0,
        next_observation=np.array([0.0], dtype=np.float32),
        done=False,
    )

    loss = agent.replay(batch_size=1)

    assert loss is not None
    # q(action=1)=0, target=max([2,1])*gamma + reward = 1 + 0.5*2 = 2.0
    assert loss == pytest.approx(4.0)
    assert agent.global_step == 1
    assert agent.epsilon == pytest.approx(scheduler.value(1))


def test_smoke_short_training_run_on_counter_env() -> None:
    _set_all_seeds(21)
    agent = DQNAgent(
        observation_size=1,
        action_size=2,
        config=DQNConfig(min_replay_size=4, batch_size=4, target_update_interval=2),
    )
    trainer = DQNTrainer(
        agent=agent,
        env_factory=lambda: CounterEnv(target=2, max_steps=6),
        config=TrainerConfig(
            total_steps=12,
            update_frequency=1,
            replay_warmup_steps=4,
            target_update_cadence=2,
            checkpoint_interval=100,
            evaluation_episodes=1,
        ),
    )

    metrics = trainer.train()

    assert "episodes" in metrics
    assert "evaluation_reward" in metrics
    assert agent.global_step > 0
