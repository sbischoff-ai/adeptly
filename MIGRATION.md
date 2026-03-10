# Migration Guide: Legacy `DQNAgent` to Current API

This guide helps migrate older Adeptly integrations (including TensorFlow 1.x-era patterns and legacy import paths) to the current PyTorch-based API.

## What changed

- `DQNAgent` now lives at `adeptly.agents.dqn.DQNAgent`.
- `adeptly.dqn.DQNAgent` still works as a temporary compatibility shim, but emits a `DeprecationWarning`.
- `AdeptlyEngine` is deprecated and now a no-op context manager.
- Runtime no longer depends on TensorFlow graph/session management.
- Configuration is consolidated in typed dataclasses (`DQNConfig`, `TrainerConfig`).

## 1) Update imports

### Before

```python
from adeptly.dqn import DQNAgent
from adeptly import AdeptlyEngine
```

### After

```python
from adeptly.agents.dqn import DQNAgent, DQNConfig
```

If you still import from `adeptly.dqn`, your code can run for now, but treat this as transitional.

## 2) Remove TensorFlow/AdeptlyEngine scaffolding

### Before

```python
from adeptly import AdeptlyEngine

with AdeptlyEngine():
    agent = DQNAgent(observation_size=obs_size, action_size=num_actions)
```

### After

```python
agent = DQNAgent(observation_size=obs_size, action_size=num_actions)
```

No explicit graph/session context is required.

## 3) Migrate agent construction to `DQNConfig`

### Before (implicit defaults and ad-hoc patterns)

```python
agent = DQNAgent(observation_size=obs_size, action_size=num_actions)
```

### After (explicit config)

```python
from adeptly.agents.dqn import DQNAgent, DQNConfig

agent = DQNAgent(
    observation_size=obs_size,
    action_size=num_actions,
    config=DQNConfig(
        batch_size=64,
        min_replay_size=1_000,
        target_update_interval=1_000,
        double_dqn=True,
    ),
)
```

## 4) Training loop migration

You can keep manual loops, but the preferred API is `DQNTrainer`.

### Before (manual loop)

```python
for step in range(total_steps):
    action = agent.predict_best_action(observation)
    next_observation, reward, done, info = env.step(action)
    agent.remember(observation, action, reward, next_observation, done)
    agent.replay()
    observation = next_observation
```

### After (trainer)

```python
from adeptly import DQNTrainer, TrainerConfig

trainer = DQNTrainer(
    agent=agent,
    env_factory=my_env_factory,
    config=TrainerConfig(
        total_steps=10_000,
        update_frequency=1,
        replay_warmup_steps=1_000,
        target_update_cadence=1_000,
    ),
)
metrics = trainer.train()
```

## 5) Real-time integration migration

If you previously coupled acting and learning in the same latency-sensitive loop, move to:

- `RealTimeInferenceLoop.actor_step` for action selection
- `RealTimeInferenceLoop.submit_transition` + `learner_update` for deferred training

This preserves responsiveness while still ingesting transitions and training incrementally.

## 6) Multimodal pipeline adoption (optional)

For image + telemetry (+ optional sequence/event) observations, use:

- `ObservationBatch`
- `MultimodalQNetwork`

The pipeline includes shape validation and built-in normalization helpers before modality encoding and fusion.

## Common migration pitfalls

- **Continuing to rely on deprecated imports**: update import paths now to avoid future breakage.
- **Calling deprecated context managers expecting runtime behavior**: `AdeptlyEngine` is only a compatibility no-op.
- **Ignoring warmup requirements**: `replay()` returns `None` until enough samples are in replay memory.
- **Assuming old environment step signatures**: ensure `done` is derived from `terminated or truncated` when using Gymnasium-like APIs.

## Recommended validation after migration

Run the standard checks:

```bash
uv run mypy adeptly tests
uv run pytest -q
```
