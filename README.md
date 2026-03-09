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
from adeptly import AdeptlyEngine, DQNAgent

actions = ['Foo', 'Bar']

bob = DQNAgent(1, 2, [0.5, 0.5])

with AdeptlyEngine():
    observation = None
    action_index = None
    reward = None
    for i in range(1000):
        next_observation = i % 10
        if i > 0:
            bob.remember(observation, action_index, reward, next_observation, True if i == 999 else False)
        observation = next_observation
        action_index = bob.predict_best_action(observation)
        # Train Bob to always Foo, expect if 9 or 10 is observed, then Bar is better.
        reward = 0 if actions[index] == 'Foo' else 1
        if observation > 8 and actions[index] == 'Foo':
            reward = -1
        # Let Bob learn from the past at every 100 steps.
        if i % 100 == 0 and i != 0:
            bob.replay(100)

```

The above example is trivial, of course, as the environment an the agent are actually decoupled, but it suffices to illustrate the basic usage of the DQN interface.

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
