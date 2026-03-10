[![CI](https://github.com/sbischoff-ai/adeptly/actions/workflows/ci.yml/badge.svg)](https://github.com/sbischoff-ai/adeptly/actions/workflows/ci.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
![PyPI](https://img.shields.io/pypi/v/adeptly.svg)

# Adeptly

Adeptly is a Python 3.12+ library for building adaptive agents in real-time environments. The current core is a PyTorch DQN stack, with trainer and low-latency runtime utilities designed for interactive loops (games, simulations, online control).

## Current capabilities

- **PyTorch 2.x DQN agent** with:
  - replay buffer
  - target networks (hard or soft updates)
  - epsilon scheduling
  - optional Double DQN target computation
- **Trainer orchestration** (`DQNTrainer`) for:
  - warmup/update cadence controls
  - checkpointing
  - evaluation episodes
- **Real-time actor/learner split** (`RealTimeInferenceLoop`) for latency-sensitive action selection with deferred learning.
- **Multimodal observation stack** for batched image + telemetry + optional event/text inputs via `ObservationBatch` and `MultimodalQNetwork`.
- **Compatibility shims** for legacy imports (`adeptly.dqn.DQNAgent`) and `AdeptlyEngine` while migration is in progress.

## Architecture documentation

- [Architecture overview](ARCHITECTURE.md): agent components, end-to-end data flow, training loop orchestration, and multimodal pipeline.
- [Migration guide](MIGRATION.md): step-by-step migration from legacy `DQNAgent` usage to the current API.

## Migration away from TensorFlow 1.x

Adeptly has moved to a PyTorch-first implementation and no longer relies on TensorFlow graph/session semantics.

- `AdeptlyEngine` is deprecated and now a no-op context manager.
- `adeptly.dqn.DQNAgent` is deprecated; import from `adeptly.agents.dqn` instead.
- Training/inference loops should use `DQNTrainer` and `RealTimeInferenceLoop` for modern usage patterns.

See [MIGRATION.md](MIGRATION.md) for concrete before/after examples.

## Quick usage

### Agent + trainer

```python
from adeptly import CounterEnv, DQNTrainer, TrainerConfig
from adeptly.agents.dqn import DQNAgent, DQNConfig

agent = DQNAgent(
    observation_size=1,
    action_size=2,
    config=DQNConfig(min_replay_size=32, batch_size=32),
)

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
action = loop.actor_step(obs)

loop.submit_transition(
    obs,
    action,
    reward=0.2,
    next_observation=np.array([1.0], dtype=np.float32),
    done=False,
)
loop.learner_update(max_updates=1)
```

### Multimodal DQN inference (batched)

```python
import torch
from adeptly.observations import MultimodalQNetwork, ObservationBatch

model = MultimodalQNetwork(
    image_channels=3,
    telemetry_dim=6,
    action_size=4,
    sequence_vocab_size=256,
)

observations = ObservationBatch(
    image_frames=torch.randint(0, 256, (8, 3, 84, 84), dtype=torch.uint8),
    scalar_telemetry=torch.randn(8, 6),
    events_or_text=torch.randint(0, 256, (8, 12), dtype=torch.int64),
)

q_values = model(observations)
actions = torch.argmax(q_values, dim=1)
```

## Roadmap

Near-term priorities:

1. Expand trainer ergonomics (resume flows, richer metrics export, and callback hooks).
2. Add end-to-end examples for real-time actor/learner deployment patterns.
3. Extend multimodal training utilities beyond inference-only examples.
4. Continue reducing legacy surface area and remove deprecated shims in a future major release.
5. Improve benchmarking coverage across synthetic and game-like environments.

## Development

This repository uses [`uv`](https://docs.astral.sh/uv/) for deterministic local workflows.

### Setup

```bash
uv python install 3.12
uv venv --python 3.12
uv sync --frozen --extra dev
```

### Canonical checks

```bash
uv run make format
uv run make lint
uv run make typecheck
uv run make test
uv run make docs
uv run pre-commit run --all-files
```

### Rebuild generated documentation site

The `/docs` directory is generated output and is not rebuilt automatically during normal edits.
To rebuild it locally, run:

```bash
uv run make docs
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for contributor workflow and PR checklist details.
