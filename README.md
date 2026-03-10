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

GitHub Actions CI runs this same command set on Python 3.12 and 3.13.

You can also run the underlying commands directly:

```bash
uv run black --check adeptly tests
uv run mypy adeptly
uv run pytest
```
