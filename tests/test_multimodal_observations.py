import pytest
import torch

from adeptly.observations import (
    MultimodalQNetwork,
    ObservationBatch,
    normalize_image_frames,
    normalize_scalar_telemetry,
    validate_observation_batch,
)


def test_validate_observation_batch_rejects_mismatched_batch_dims():
    observations = ObservationBatch(
        image_frames=torch.zeros(2, 3, 84, 84),
        scalar_telemetry=torch.zeros(3, 4),
    )

    with pytest.raises(ValueError, match="share the same batch dimension"):
        validate_observation_batch(observations)


def test_normalization_helpers_return_expected_ranges_and_stats():
    image = torch.randint(0, 256, (4, 3, 16, 16), dtype=torch.uint8)
    normalized_image = normalize_image_frames(image)
    assert normalized_image.dtype == torch.float32
    assert float(normalized_image.max()) <= 1.0

    telemetry = torch.tensor([[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]])
    normalized_telemetry = normalize_scalar_telemetry(telemetry)
    assert torch.allclose(normalized_telemetry.mean(dim=0), torch.zeros(2), atol=1e-5)


def test_multimodal_q_network_outputs_action_values_for_batched_input():
    model = MultimodalQNetwork(
        image_channels=3,
        telemetry_dim=5,
        action_size=4,
        sequence_vocab_size=128,
        use_attention_fusion=False,
    )

    observations = ObservationBatch(
        image_frames=torch.randint(0, 256, (6, 3, 84, 84), dtype=torch.uint8),
        scalar_telemetry=torch.randn(6, 5),
        events_or_text=torch.randint(0, 128, (6, 10), dtype=torch.int64),
    )

    output = model(observations)
    assert output.shape == (6, 4)


def test_multimodal_q_network_raises_if_sequence_not_configured():
    model = MultimodalQNetwork(image_channels=3, telemetry_dim=3, action_size=2)
    observations = ObservationBatch(
        image_frames=torch.randint(0, 256, (2, 3, 84, 84), dtype=torch.uint8),
        scalar_telemetry=torch.randn(2, 3),
        events_or_text=torch.randint(0, 100, (2, 8), dtype=torch.int64),
    )

    with pytest.raises(ValueError, match="sequence encoder"):
        model(observations)
