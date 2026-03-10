[![CI](https://github.com/sbischoff-ai/adeptly/actions/workflows/ci.yml/badge.svg)](https://github.com/sbischoff-ai/adeptly/actions/workflows/ci.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
![PyPI](https://img.shields.io/pypi/v/adeptly.svg)

# Adeptly
Python 3.12+ library for adaptive intelligent agents in real-time environments (e.g. games) based on a combination of
rule-based policies and Deep Q Neural Networks.

## Current State
Right now this is little more than a basic DQN implementation. The vision is for this to become a library dedicated to reinforcement learning agents that can learn as the act in an environment (which is my definition of *adaptive* here).
In the next step this will then be integrated with decision and behaviour trees as well as finite state machines in way that q-learning agents become nodes in a behaviour or decision tree.

## Usage
```python
import numpy as np
from adeptly.agents.dqn import DQNAgent

actions = ["Foo", "Bar"]
agent = DQNAgent(observation_size=1, action_size=2)

observation = np.array([0.0], dtype=np.float32)
for step in range(1_000):
    action_index = agent.predict_best_action(observation)
    next_observation = np.array([float(step % 10)], dtype=np.float32)
    reward = 1.0 if actions[action_index] == "Bar" and next_observation[0] > 8 else 0.0
    done = step == 999
    agent.remember(observation, action_index, reward, next_observation, done)
    loss = agent.replay()
    observation = next_observation
```



### Environment protocol and trainer
```python
from adeptly import CounterEnv, DQNAgent, DQNTrainer, TrainerConfig
from adeptly.agents.dqn import DQNConfig

agent = DQNAgent(observation_size=1, action_size=2, config=DQNConfig(min_replay_size=32, batch_size=32))
trainer = DQNTrainer(
    agent=agent,
    env_factory=lambda: CounterEnv(target=5, max_steps=16),
    config=TrainerConfig(
        total_steps=2_000,
        update_frequency=2,
        replay_warmup_steps=64,
        target_update_cadence=100,
        checkpoint_interval=500,
        evaluation_episodes=5,
    ),
)
metrics = trainer.train()
print(metrics)
```

### Real-time inference loop (actor/learner split)
```python
import numpy as np
from adeptly import RealTimeInferenceLoop

loop = RealTimeInferenceLoop(agent)
obs = np.array([0.0], dtype=np.float32)
action = loop.actor_step(obs)  # low-latency action selection path

# Later/on another thread: enqueue transitions and update learner independently.
loop.submit_transition(obs, action, reward=0.2, next_observation=np.array([1.0], dtype=np.float32), done=False)
loop.learner_update(max_updates=1)
```

### Migration notes
- `adeptly.dqn.DQNAgent` is deprecated; use `adeptly.agents.dqn.DQNAgent`.
- `AdeptlyEngine` is deprecated and now a no-op context manager.
- The TensorFlow/Keras implementation has been replaced by a PyTorch 2.x DQN agent with replay buffer, target network updates, epsilon scheduling, and Double DQN support.

## Development

This project now uses [`uv`](https://docs.astral.sh/uv/) for dependency management and reproducible environments.

### Setup
```bash
uv python install 3.12
uv venv --python 3.12
```

### Lock dependencies
```bash
uv lock
```

### Sync environment from lockfile
```bash
uv sync --frozen --extra dev
```

### Run checks
```bash
uv run make lint
uv run make typecheck
uv run make test
```

GitHub Actions CI runs these direct commands on Python 3.12 and 3.13: black, mypy (on `adeptly` and `tests`), and `pytest -q`.

You can also run the underlying commands directly:

```bash
uv run black --check adeptly tests
uv run mypy adeptly tests
uv run pytest -q
```


### Multimodal DQN inference (batched)
```python
import torch
from adeptly.observations import MultimodalQNetwork, ObservationBatch

batch_size = 8
num_actions = 4
model = MultimodalQNetwork(
    image_channels=3,
    telemetry_dim=6,
    action_size=num_actions,
    sequence_vocab_size=256,
)

# Minimal batched environment loop for inference-only usage.
for _ in range(5):
    observations = ObservationBatch(
        image_frames=torch.randint(0, 256, (batch_size, 3, 84, 84), dtype=torch.uint8),
        scalar_telemetry=torch.randn(batch_size, 6),
        events_or_text=torch.randint(0, 256, (batch_size, 12), dtype=torch.int64),
    )
    q_values = model(observations)
    actions = torch.argmax(q_values, dim=1)
    # send `actions` back to your vectorized environment
```
