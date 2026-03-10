"""Composable multimodal encoders for DQN observations."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class VisionEncoder(nn.Module):
    """Small CNN encoder for image frame observations."""

    def __init__(self, in_channels: int, embedding_dim: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, embedding_dim),
            nn.ReLU(),
        )

    def forward(self, image_frames: Tensor) -> Tensor:
        return self.backbone(image_frames)


class TelemetryEncoder(nn.Module):
    """MLP encoder for scalar telemetry channels."""

    def __init__(self, input_dim: int, embedding_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.ReLU(),
        )

    def forward(self, scalar_telemetry: Tensor) -> Tensor:
        return self.model(scalar_telemetry)


class EventSequenceEncoder(nn.Module):
    """Embedding + GRU encoder for event/text token sequences."""

    def __init__(self, vocab_size: int, embedding_dim: int, output_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, output_dim, batch_first=True)

    def forward(self, events_or_text: Tensor) -> Tensor:
        embedded = self.embedding(events_or_text.long())
        _, hidden = self.gru(embedded)
        return hidden.squeeze(0)


class ModalityFusion(nn.Module):
    """Fuse modality embeddings via concatenation projection or attention pooling."""

    def __init__(self, input_dims: list[int], fused_dim: int, use_attention: bool = False) -> None:
        super().__init__()
        self.use_attention = use_attention
        self.fused_dim = fused_dim

        if use_attention:
            if len(set(input_dims)) != 1:
                raise ValueError("All modality dims must match when use_attention=True")
            self.attention = nn.MultiheadAttention(embed_dim=input_dims[0], num_heads=1, batch_first=True)
            self.output_projection = nn.Linear(input_dims[0], fused_dim)
        else:
            self.output_projection = nn.Linear(sum(input_dims), fused_dim)

    def forward(self, modality_embeddings: list[Tensor]) -> Tensor:
        if not modality_embeddings:
            raise ValueError("modality_embeddings cannot be empty")

        if self.use_attention:
            stacked = torch.stack(modality_embeddings, dim=1)
            attended, _ = self.attention(stacked, stacked, stacked)
            pooled = attended.mean(dim=1)
            return self.output_projection(pooled)

        concatenated = torch.cat(modality_embeddings, dim=1)
        return self.output_projection(concatenated)
