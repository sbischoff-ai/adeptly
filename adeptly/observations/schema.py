"""Observation schema and validation utilities for multimodal DQN inputs."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(slots=True)
class ObservationBatch:
    """Batched multimodal observations consumed by multimodal DQN networks."""

    image_frames: Tensor
    scalar_telemetry: Tensor
    events_or_text: Tensor | None = None


def validate_observation_batch(observations: ObservationBatch) -> None:
    """Validate batch shapes early to prevent downstream runtime shape errors."""

    image = observations.image_frames
    telemetry = observations.scalar_telemetry
    sequence = observations.events_or_text

    if image.ndim != 4:
        raise ValueError("image_frames must have shape [batch, channels, height, width]")
    if telemetry.ndim != 2:
        raise ValueError("scalar_telemetry must have shape [batch, features]")

    batch_size = image.shape[0]
    if telemetry.shape[0] != batch_size:
        raise ValueError("image_frames and scalar_telemetry must share the same batch dimension")

    if sequence is not None:
        if sequence.ndim != 2:
            raise ValueError("events_or_text must have shape [batch, sequence_length]")
        if sequence.shape[0] != batch_size:
            raise ValueError("events_or_text must share the same batch dimension as image_frames")


def normalize_image_frames(image_frames: Tensor) -> Tensor:
    """Convert image frames to float tensors in [0, 1]."""

    if not torch.is_floating_point(image_frames):
        image_frames = image_frames.to(torch.float32)
    return image_frames / 255.0 if image_frames.max().item() > 1.0 else image_frames


def normalize_scalar_telemetry(scalar_telemetry: Tensor, eps: float = 1e-6) -> Tensor:
    """Per-batch z-score normalization for scalar telemetry channels."""

    telemetry = scalar_telemetry.to(torch.float32)
    mean = telemetry.mean(dim=0, keepdim=True)
    std = telemetry.std(dim=0, keepdim=True).clamp_min(eps)
    return (telemetry - mean) / std
