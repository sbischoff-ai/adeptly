"""Multimodal DQN network composed from observation encoders and fusion."""

from __future__ import annotations

from torch import Tensor, nn

from adeptly.observations.encoders import EventSequenceEncoder, ModalityFusion, TelemetryEncoder, VisionEncoder
from adeptly.observations.schema import (
    ObservationBatch,
    normalize_image_frames,
    normalize_scalar_telemetry,
    validate_observation_batch,
)


class MultimodalQNetwork(nn.Module):
    """DQN action-value network for multimodal observations."""

    def __init__(
        self,
        image_channels: int,
        telemetry_dim: int,
        action_size: int,
        *,
        vision_embedding_dim: int = 128,
        telemetry_embedding_dim: int = 64,
        sequence_vocab_size: int | None = None,
        sequence_embedding_dim: int = 32,
        sequence_output_dim: int = 64,
        fused_dim: int = 256,
        use_attention_fusion: bool = False,
    ) -> None:
        super().__init__()

        if action_size <= 0:
            raise ValueError("action_size must be > 0")

        self.vision_encoder = VisionEncoder(image_channels, vision_embedding_dim)
        self.telemetry_encoder = TelemetryEncoder(telemetry_dim, telemetry_embedding_dim)

        self.sequence_encoder: EventSequenceEncoder | None = None
        encoder_dims = [vision_embedding_dim, telemetry_embedding_dim]
        if sequence_vocab_size is not None:
            self.sequence_encoder = EventSequenceEncoder(
                vocab_size=sequence_vocab_size,
                embedding_dim=sequence_embedding_dim,
                output_dim=sequence_output_dim,
            )
            encoder_dims.append(sequence_output_dim)

        self.fusion = ModalityFusion(encoder_dims, fused_dim=fused_dim, use_attention=use_attention_fusion)
        self.action_head = nn.Linear(fused_dim, action_size)

    def forward(self, observations: ObservationBatch) -> Tensor:
        validate_observation_batch(observations)

        image_frames = normalize_image_frames(observations.image_frames)
        scalar_telemetry = normalize_scalar_telemetry(observations.scalar_telemetry)

        embeddings = [
            self.vision_encoder(image_frames),
            self.telemetry_encoder(scalar_telemetry),
        ]

        if observations.events_or_text is not None:
            if self.sequence_encoder is None:
                raise ValueError("events_or_text provided but sequence encoder was not configured")
            embeddings.append(self.sequence_encoder(observations.events_or_text))

        fused = self.fusion(embeddings)
        return self.action_head(fused)
