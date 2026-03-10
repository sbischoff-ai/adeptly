from pathlib import Path

import numpy as np

from adeptly.agents.dqn import DQNAgent, DQNConfig
from adeptly.synthetic_env import CounterEnv
from adeptly.trainer import DQNTrainer, RealTimeInferenceLoop, TrainerConfig


def test_trainer_runs_e2e_and_writes_checkpoint(tmp_path: Path):
    agent = DQNAgent(
        observation_size=1,
        action_size=2,
        config=DQNConfig(min_replay_size=8, batch_size=8),
    )
    trainer = DQNTrainer(
        agent=agent,
        env_factory=lambda: CounterEnv(target=3, max_steps=8),
        config=TrainerConfig(
            total_steps=40,
            update_frequency=2,
            replay_warmup_steps=8,
            target_update_cadence=4,
            checkpoint_interval=20,
            checkpoint_dir=str(tmp_path),
            evaluation_episodes=2,
        ),
    )

    metrics = trainer.train()
    checkpoints = sorted(tmp_path.glob("dqn_step_*.pt"))

    assert checkpoints
    assert metrics["episodes"] >= 1
    assert "evaluation_reward" in metrics


def test_realtime_loop_decouples_actor_and_learner_updates():
    agent = DQNAgent(
        observation_size=1,
        action_size=2,
        config=DQNConfig(min_replay_size=4, batch_size=4),
    )
    loop = RealTimeInferenceLoop(agent=agent, train_every=1)

    for i in range(8):
        obs = np.array([float(i)], dtype=np.float32)
        action = loop.actor_step(obs)
        next_obs = np.array([float(i + 1)], dtype=np.float32)
        loop.submit_transition(obs, action, 0.2, next_obs, i == 7)

    assert len(agent.memory) == 0
    losses = loop.learner_update(max_updates=4)

    assert len(agent.memory) == 8
    assert losses
