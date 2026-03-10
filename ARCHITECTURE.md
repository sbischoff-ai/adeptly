# Architecture

This document describes the current Adeptly runtime architecture for agent components, training/inference data flow, and multimodal observation processing.

## Component map

- `adeptly.agents.dqn.DQNAgent`
  - Owns the online Q-network, target Q-network, optimizer, replay buffer, epsilon scheduler, and training step counters.
  - Provides `predict_best_action`, `remember`, and `replay` as the minimal interaction surface for actor/trainer loops.
- `adeptly.agents.dqn.ReplayBuffer`
  - Stores `(observation, action, reward, next_observation, done)` transitions.
  - Supports random batch sampling for off-policy updates.
- `adeptly.agents.dqn.EpsilonScheduler`
  - Produces linearly decayed epsilon values for epsilon-greedy exploration.
- `adeptly.trainer.DQNTrainer`
  - Coordinates environment interaction, replay warmup, optimization cadence, checkpointing, and evaluation episodes.
- `adeptly.trainer.RealTimeInferenceLoop`
  - Splits low-latency action selection from learner updates by buffering transitions and replaying asynchronously.
- `adeptly.observations.*`
  - Defines multimodal observation schemas, normalization/validation utilities, modality-specific encoders, and a fused `MultimodalQNetwork` for batched Q-value inference/training.

## DQN data flow

### 1) Acting (online interaction)

1. Environment returns an observation.
2. `DQNAgent.predict_best_action` selects an action with epsilon-greedy policy:
   - random weighted action with probability `epsilon`
   - argmax of online network Q-values otherwise.
3. Action is applied to the environment.

### 2) Transition capture

1. Environment returns `(next_observation, reward, terminated, truncated, info)`.
2. Caller computes `done = terminated or truncated`.
3. `DQNAgent.remember` stores transition in replay memory.

### 3) Learning update

1. `DQNAgent.replay` checks warmup requirements (`min_replay_size` and batch size).
2. Sampled transition batches are converted to tensors on the configured device.
3. Online Q-values are computed for selected actions.
4. TD targets are computed from the target network:
   - Double DQN path (default):
     - action selection from online network
     - action evaluation from target network
   - Vanilla target max path when `double_dqn=False`.
5. MSE loss is optimized via Adam.
6. `epsilon` is decayed using `EpsilonScheduler`.
7. Target network updates are applied:
   - soft update when `soft_update_tau` is set
   - periodic hard update otherwise.

## Training orchestration flow (`DQNTrainer`)

`DQNTrainer.train` wraps the loop above with deterministic cadence controls:

- `total_steps`: total interaction steps.
- `replay_warmup_steps`: minimum steps before learning starts.
- `update_frequency`: how often `replay` is called.
- `target_update_cadence`: pushed into agent config as `target_update_interval`.
- `checkpoint_interval`: periodic serialized state snapshots.
- `evaluation_episodes`: greedy-policy rollouts for summary metrics.

At completion, trainer returns a metrics dictionary (`episodes`, `evaluation_reward`).

## Real-time actor/learner split (`RealTimeInferenceLoop`)

The real-time loop is intended for scenarios where action latency is more sensitive than learner throughput:

- Actor path: `actor_step(observation)` only runs action selection.
- Learner path:
  - `submit_transition(...)` appends to a bounded deque.
  - `learner_update(max_updates=...)` drains buffered transitions into replay and performs replay updates according to `train_every`.

This decoupling allows you to run updates on another thread/process tick while keeping action selection predictable.

## Multimodal observation pipeline

### Observation contract

`ObservationBatch` defines batched tensors for:

- `image_frames`: `[batch, channels, height, width]`
- `scalar_telemetry`: `[batch, features]`
- `events_or_text` (optional): `[batch, sequence_length]`

`validate_observation_batch` enforces shape consistency across modalities before encoding.

### Normalization

- `normalize_image_frames` converts image tensors to float and scales to `[0, 1]` when needed.
- `normalize_scalar_telemetry` applies per-batch z-score normalization with epsilon clamping.

### Encoding and fusion

`MultimodalQNetwork` composes:

1. `VisionEncoder`: CNN backbone to visual embedding.
2. `TelemetryEncoder`: MLP to telemetry embedding.
3. Optional `EventSequenceEncoder`: embedding + GRU to sequence embedding.
4. `ModalityFusion`:
   - concatenation + linear projection (default), or
   - single-head self-attention pooling when modality dimensions match.
5. `action_head`: final linear layer mapping fused embedding to per-action Q-values.

## Backward compatibility and deprecations

- `adeptly.dqn.DQNAgent` import path remains available as a deprecation shim.
- `AdeptlyEngine` remains as a deprecated no-op context manager for legacy integrations.
- Core agent implementation is PyTorch-based; TensorFlow graph/session semantics are no longer part of the runtime architecture.
