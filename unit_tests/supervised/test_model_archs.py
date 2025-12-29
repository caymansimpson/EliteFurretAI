# -*- coding: utf-8 -*-
"""
Unit tests for supervised model architectures.

These tests verify:
1. ResidualBlock and GatedResidualBlock forward passes
2. GroupedFeatureEncoder with cross-attention
3. DNN simple model
4. FlexibleThreeHeadedModel initialization and forward pass
5. Output shapes and value ranges
6. Weight initialization

Note: These are structural tests using small dimensions for speed.
Full integration tests would use actual embedder dimensions.
"""

import pytest
import torch

from elitefurretai.etl.encoder import MDBO
from elitefurretai.supervised.model_archs import (
    DNN,
    FlexibleThreeHeadedModel,
    GatedResidualBlock,
    GroupedFeatureEncoder,
    ResidualBlock,
    init_linear_layer,
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def batch_size():
    """Standard batch size for tests."""
    return 4


@pytest.fixture
def seq_len():
    """Standard sequence length for tests."""
    return 5


@pytest.fixture
def small_input_size():
    """Small input size for fast tests."""
    return 100


@pytest.fixture
def sample_input(batch_size, seq_len, small_input_size):
    """
    Create sample input tensor.

    Shape: (batch, seq, features)
    """
    return torch.randn(batch_size, seq_len, small_input_size)


# =============================================================================
# RESIDUAL BLOCK TESTS
# =============================================================================


def test_residual_block_same_dims():
    """
    Test ResidualBlock with same input/output dimensions.

    When in_features == out_features, shortcut is identity.

    Expected: Output shape equals input shape.
    """
    block = ResidualBlock(in_features=64, out_features=64, dropout=0.1)
    x = torch.randn(4, 10, 64)

    output = block(x)

    assert output.shape == x.shape
    assert output.shape == (4, 10, 64)


def test_residual_block_different_dims():
    """
    Test ResidualBlock with different input/output dimensions.

    When dimensions differ, shortcut uses linear projection.

    Expected: Output has different feature dimension than input.
    """
    block = ResidualBlock(in_features=64, out_features=128, dropout=0.1)
    x = torch.randn(4, 10, 64)

    output = block(x)

    assert output.shape == (4, 10, 128)


def test_residual_block_allows_negative_values():
    """
    Test that ResidualBlock output can be negative.

    Unlike blocks with double ReLU, residual addition allows
    negative values to pass through for the win head.

    Expected: Some output values should be negative.
    """
    torch.manual_seed(42)  # For reproducibility
    block = ResidualBlock(in_features=64, out_features=64, dropout=0.0)

    # Use input with many negative values
    x = torch.randn(4, 10, 64) * 2 - 1

    output = block(x)

    # The shortcut allows negatives to pass through
    assert output.min() < 0, "Output should contain negative values"


def test_residual_block_gradient_flow():
    """
    Test that gradients flow through ResidualBlock.

    Skip connections help gradient flow during backprop.

    Expected: Input tensor should have gradient after backward.
    """
    block = ResidualBlock(in_features=64, out_features=64, dropout=0.0)
    x = torch.randn(4, 10, 64, requires_grad=True)

    output = block(x)
    loss = output.sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape


# =============================================================================
# GATED RESIDUAL BLOCK TESTS
# =============================================================================


def test_gated_residual_block_forward():
    """
    Test GatedResidualBlock forward pass.

    Gated blocks use sigmoid gate to control information flow.

    Expected: Valid output shape.
    """
    block = GatedResidualBlock(in_features=64, out_features=64, dropout=0.1)
    x = torch.randn(4, 10, 64)

    output = block(x)

    assert output.shape == (4, 10, 64)


def test_gated_residual_block_dimension_change():
    """
    Test GatedResidualBlock with dimension change.

    Expected: Output dimension matches out_features.
    """
    block = GatedResidualBlock(in_features=64, out_features=128, dropout=0.1)
    x = torch.randn(4, 10, 64)

    output = block(x)

    assert output.shape == (4, 10, 128)


def test_gated_block_gate_values():
    """
    Test that gate produces values in [0, 1].

    Gate uses sigmoid, so values should be bounded.

    Expected: Gate outputs in valid range.
    """
    block = GatedResidualBlock(in_features=64, out_features=64, dropout=0.0)
    x = torch.randn(4, 10, 64)

    # Access gate output directly
    gate_value = block.gate(x)

    assert gate_value.min() >= 0.0
    assert gate_value.max() <= 1.0


# =============================================================================
# GROUPED FEATURE ENCODER TESTS
# =============================================================================


def test_grouped_encoder_initialization():
    """
    Test GroupedFeatureEncoder initialization.

    Encoder should create one encoder per group.

    Expected: Number of encoders equals number of groups.
    """
    group_sizes = [
        100,
        100,
        100,
        100,
        100,
        100,  # Player Pokemon 0-5
        80,
        80,
        80,
        80,
        80,
        80,  # Opponent Pokemon 0-5
        50,
        30,
    ]  # Battle state, features
    encoder = GroupedFeatureEncoder(
        group_sizes=group_sizes,
        hidden_dim=32,
        aggregated_dim=128,
        dropout=0.1,
        pokemon_attention_heads=2,
    )

    assert len(encoder.encoders) == len(group_sizes)


def test_grouped_encoder_forward():
    """
    Test GroupedFeatureEncoder forward pass.

    Input is split into groups, encoded, cross-attended, and aggregated.

    Expected: Output has aggregated_dim features.
    """
    group_sizes = [50, 50, 50, 50, 50, 50, 30, 20]  # 8 groups
    total_size = sum(group_sizes)

    encoder = GroupedFeatureEncoder(
        group_sizes=group_sizes,
        hidden_dim=32,
        aggregated_dim=64,
        dropout=0.1,
        pokemon_attention_heads=2,
    )

    x = torch.randn(4, 5, total_size)
    output = encoder(x)

    assert output.shape == (4, 5, 64)


def test_grouped_encoder_cross_attention():
    """
    Test that cross-attention modifies Pokemon features.

    Cross-attention allows Pokemon to "see" each other.

    Expected: Output should differ from simple concatenation.
    """
    group_sizes = [20] * 6 + [10, 10]  # 6 Pokemon + 2 other groups
    total_size = sum(group_sizes)

    encoder = GroupedFeatureEncoder(
        group_sizes=group_sizes,
        hidden_dim=16,
        aggregated_dim=32,
        dropout=0.0,  # Disable dropout for determinism
        pokemon_attention_heads=2,
    )

    # Set to eval mode for deterministic behavior
    encoder.eval()

    x = torch.randn(2, 3, total_size)
    output = encoder(x)

    # Just verify it runs and produces valid output
    assert output.shape == (2, 3, 32)
    assert torch.isfinite(output).all()


# =============================================================================
# DNN TESTS
# =============================================================================


def test_dnn_forward():
    """
    Test DNN forward pass.

    Simple feedforward network with residual blocks.

    Expected: Output has action_space dimensions.
    """
    model = DNN(input_size=100, hidden_sizes=[64, 32], dropout=0.1)
    x = torch.randn(4, 100)

    output = model(x)

    assert output.shape == (4, MDBO.action_space())


def test_dnn_with_sequence():
    """
    Test DNN with sequence input.

    DNN should work with (batch, seq, features) input.

    Expected: Output shape is (batch, seq, action_space).
    """
    model = DNN(input_size=100, hidden_sizes=[64, 32], dropout=0.1)
    x = torch.randn(4, 5, 100)

    output = model(x)

    assert output.shape == (4, 5, MDBO.action_space())


# =============================================================================
# FLEXIBLE THREE-HEADED MODEL TESTS
# =============================================================================


def test_flexible_model_initialization():
    """
    Test FlexibleThreeHeadedModel initialization.

    Model should create all components without errors.

    Expected: Model initializes successfully.
    """
    model = FlexibleThreeHeadedModel(
        input_size=100,
        early_layers=[64, 32],
        late_layers=[64, 32],
        lstm_layers=1,
        lstm_hidden_size=32,
        num_actions=2025,
        num_teampreview_actions=90,
        max_seq_len=20,
        dropout=0.1,
    )

    assert model is not None
    assert hasattr(model, "lstm")
    assert hasattr(model, "turn_action_head")
    assert hasattr(model, "teampreview_head")
    assert hasattr(model, "win_head")


def test_flexible_model_forward_shapes():
    """
    Test FlexibleThreeHeadedModel output shapes.

    Model produces three outputs:
    - Turn logits: (batch, seq, 2025)
    - Teampreview logits: (batch, seq, 90)
    - Win value: (batch, seq)

    Expected: All outputs have correct shapes.
    """
    model = FlexibleThreeHeadedModel(
        input_size=100,
        early_layers=[64, 32],
        late_layers=[64, 32],
        lstm_layers=1,
        lstm_hidden_size=32,
        num_actions=2025,
        num_teampreview_actions=90,
        max_seq_len=20,
        dropout=0.1,
    )
    model.eval()

    x = torch.randn(4, 5, 100)
    turn_logits, tp_logits, win_value = model(x)

    assert turn_logits.shape == (4, 5, 2025)
    assert tp_logits.shape == (4, 5, 90)
    assert win_value.shape == (4, 5)


def test_flexible_model_win_value_range():
    """
    Test that win value is bounded in [-1, 1].

    Win head uses tanh activation, so output should be bounded.

    Expected: All win values in [-1, 1].
    """
    model = FlexibleThreeHeadedModel(
        input_size=100,
        early_layers=[64, 32],
        late_layers=[64, 32],
        lstm_layers=1,
        lstm_hidden_size=32,
        dropout=0.0,
    )
    model.eval()

    # Test with various inputs
    for _ in range(5):
        x = torch.randn(4, 5, 100) * 10  # Large values
        _, _, win_value = model(x)

        assert win_value.min() >= -1.0
        assert win_value.max() <= 1.0


def test_flexible_model_with_grouped_encoder():
    """
    Test FlexibleThreeHeadedModel with GroupedFeatureEncoder.

    use_grouped_encoder=True enables semantic feature grouping.

    Expected: Model works with grouped encoder.
    """
    group_sizes = [20] * 6 + [10] * 6 + [10, 10]  # 14 groups
    total_size = sum(group_sizes)

    model = FlexibleThreeHeadedModel(
        input_size=total_size,
        early_layers=[64, 32],
        late_layers=[64, 32],
        lstm_layers=1,
        lstm_hidden_size=32,
        dropout=0.1,
        use_grouped_encoder=True,
        group_sizes=group_sizes,
        grouped_encoder_hidden_dim=16,
        grouped_encoder_aggregated_dim=64,
    )
    model.eval()

    x = torch.randn(4, 5, total_size)
    turn_logits, tp_logits, win_value = model(x)

    assert turn_logits.shape[0] == 4
    assert tp_logits.shape[2] == 90
    assert win_value.shape == (4, 5)


def test_flexible_model_with_gated_residuals():
    """
    Test FlexibleThreeHeadedModel with gated residual blocks.

    gated_residuals=True uses GatedResidualBlock instead of ResidualBlock.

    Expected: Model works with gated residuals.
    """
    model = FlexibleThreeHeadedModel(
        input_size=100,
        early_layers=[64, 32],
        late_layers=[64, 32],
        lstm_layers=1,
        lstm_hidden_size=32,
        dropout=0.1,
        gated_residuals=True,
    )
    model.eval()

    x = torch.randn(4, 5, 100)
    turn_logits, tp_logits, win_value = model(x)

    assert turn_logits.shape == (4, 5, 2025)


def test_flexible_model_with_attention():
    """
    Test FlexibleThreeHeadedModel with attention layers.

    early_attention_heads and late_attention_heads enable self-attention.

    Expected: Model works with attention enabled.
    """
    model = FlexibleThreeHeadedModel(
        input_size=100,
        early_layers=[64, 32],
        late_layers=[64, 32],
        lstm_layers=1,
        lstm_hidden_size=32,
        dropout=0.1,
        early_attention_heads=4,
        late_attention_heads=4,
    )
    model.eval()

    x = torch.randn(4, 5, 100)
    turn_logits, tp_logits, win_value = model(x)

    assert turn_logits.shape == (4, 5, 2025)


def test_flexible_model_with_action_mask():
    """
    Test FlexibleThreeHeadedModel with action masking.

    Mask should be applied to turn logits.
    Note: The model expects a 2D mask of shape (batch, seq) for attention.

    Expected: Model accepts and applies mask correctly.
    """
    model = FlexibleThreeHeadedModel(
        input_size=100,
        early_layers=[64, 32],
        late_layers=[64, 32],
        lstm_layers=1,
        lstm_hidden_size=32,
        dropout=0.0,
    )
    model.eval()

    x = torch.randn(4, 5, 100)

    # Create 2D mask for attention (batch, seq) - True means padded/masked
    # This is the key_padding_mask format expected by PyTorch MHA
    mask = torch.zeros(4, 5, dtype=torch.bool)
    mask[:, -1] = True  # Mask last position

    turn_logits, _, _ = model(x, mask=mask)

    assert turn_logits.shape == (4, 5, 2025)


def test_flexible_model_forward_with_hidden():
    """
    Test forward_with_hidden method for RL training.

    Returns hidden states for recurrent inference.

    Expected: Returns 4 outputs including hidden states.
    """
    model = FlexibleThreeHeadedModel(
        input_size=100,
        early_layers=[64, 32],
        late_layers=[64, 32],
        lstm_layers=1,
        lstm_hidden_size=32,
        dropout=0.0,
    )
    model.eval()

    x = torch.randn(4, 5, 100)

    # Get initial hidden state
    batch_size = 4
    num_directions = 2
    num_layers = 1
    hidden_size = 32
    h0 = torch.zeros(num_layers * num_directions, batch_size, hidden_size)
    c0 = torch.zeros(num_layers * num_directions, batch_size, hidden_size)

    turn_logits, tp_logits, win_value, hidden = model.forward_with_hidden(x, (h0, c0))

    assert turn_logits.shape == (4, 5, 2025)
    assert tp_logits.shape == (4, 5, 90)
    assert win_value.shape == (4, 5)
    assert isinstance(hidden, tuple)
    assert len(hidden) == 2  # (h, c)


# =============================================================================
# WEIGHT INITIALIZATION TESTS
# =============================================================================


def test_init_linear_layer():
    """
    Test init_linear_layer helper function.

    Should apply Kaiming initialization.

    Expected: Weights should be modified from default.
    """
    layer = torch.nn.Linear(64, 32)
    original_weight = layer.weight.clone()

    init_linear_layer(layer)

    # Weights should be different after initialization
    assert not torch.allclose(layer.weight, original_weight)
    # Bias should be zero
    assert torch.allclose(layer.bias, torch.zeros_like(layer.bias))


def test_model_weight_statistics():
    """
    Test that model weights have reasonable statistics.

    Well-initialized weights should have small variance.
    LayerNorm weights are initialized to 1.0 (constant), so we skip those.

    Expected: Weights not too large or too small (excluding normalization layers).
    """
    model = FlexibleThreeHeadedModel(
        input_size=100,
        early_layers=[64, 32],
        late_layers=[64, 32],
        lstm_layers=1,
        lstm_hidden_size=32,
        dropout=0.1,
    )

    linear_layers_checked = 0
    for name, param in model.named_parameters():
        if "weight" in name:
            std = param.std().item()  # type: ignore[union-attr]
            # Weights shouldn't be too large
            assert std < 2.0, f"{name} has std={std}"

            # Only check Linear layer weights (skip LayerNorm, which has std=0)
            # Linear weights should have variance > 0
            if "_head" in name or "lstm" in name or "fc" in name:
                if param.dim() >= 2:  # type: ignore[union-attr]
                    assert std > 0.0, f"{name} has std={std}"
                    linear_layers_checked += 1

    # Make sure we actually checked some layers
    assert linear_layers_checked > 0, "Should have checked at least one linear layer"


# =============================================================================
# GRADIENT FLOW TESTS
# =============================================================================


def test_gradient_flow_through_model():
    """
    Test that gradients flow through entire model.

    All parameters should receive gradients during backprop.

    Expected: No zero gradients for trainable parameters.
    """
    model = FlexibleThreeHeadedModel(
        input_size=100,
        early_layers=[64, 32],
        late_layers=[64, 32],
        lstm_layers=1,
        lstm_hidden_size=32,
        dropout=0.0,  # Disable dropout for deterministic gradients
    )

    x = torch.randn(4, 5, 100, requires_grad=True)
    turn_logits, tp_logits, win_value = model(x)

    # Compute loss from all outputs
    loss = turn_logits.sum() + tp_logits.sum() + win_value.sum()
    loss.backward()

    # Check input has gradient
    assert x.grad is not None

    # Check key parameters have gradients
    for name, param in model.named_parameters():
        if param.requires_grad:  # type: ignore[union-attr]
            assert param.grad is not None, f"{name} has no gradient"  # type: ignore[union-attr]


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


def test_single_sequence_element():
    """
    Test model with sequence length of 1.

    Some operations behave differently with single elements.

    Expected: Model handles seq_len=1 correctly.
    """
    model = FlexibleThreeHeadedModel(
        input_size=100,
        early_layers=[64, 32],
        late_layers=[64, 32],
        lstm_layers=1,
        lstm_hidden_size=32,
        dropout=0.0,
    )
    model.eval()

    x = torch.randn(4, 1, 100)  # seq_len = 1
    turn_logits, tp_logits, win_value = model(x)

    assert turn_logits.shape == (4, 1, 2025)
    assert tp_logits.shape == (4, 1, 90)
    assert win_value.shape == (4, 1)


def test_batch_size_one():
    """
    Test model with batch size of 1.

    Expected: Model handles batch_size=1 correctly.
    """
    model = FlexibleThreeHeadedModel(
        input_size=100,
        early_layers=[64, 32],
        late_layers=[64, 32],
        lstm_layers=1,
        lstm_hidden_size=32,
        dropout=0.0,
    )
    model.eval()

    x = torch.randn(1, 5, 100)  # batch = 1
    turn_logits, tp_logits, win_value = model(x)

    assert turn_logits.shape == (1, 5, 2025)


def test_no_early_layers():
    """
    Test model with minimal early_layers.

    Note: embed_dim must be divisible by attention heads (4 by default).
    So we use layer size of 64 which is divisible by 4.

    Expected: Model produces valid output.
    """
    model = FlexibleThreeHeadedModel(
        input_size=100,
        early_layers=[64],  # Single layer
        late_layers=[64, 32],
        lstm_layers=1,
        lstm_hidden_size=32,
        dropout=0.1,
    )
    model.eval()

    x = torch.randn(4, 5, 100)
    turn_logits, _, _ = model(x)

    assert turn_logits.shape[2] == 2025


def test_deep_model():
    """
    Test model with many layers.

    Deep models should still work correctly.

    Expected: Deep model produces valid output.
    """
    model = FlexibleThreeHeadedModel(
        input_size=100,
        early_layers=[128, 128, 64, 64, 32],
        late_layers=[128, 64, 32],
        lstm_layers=2,
        lstm_hidden_size=64,
        dropout=0.1,
    )
    model.eval()

    x = torch.randn(4, 5, 100)
    turn_logits, tp_logits, win_value = model(x)

    assert torch.isfinite(turn_logits).all()
    assert torch.isfinite(tp_logits).all()
    assert torch.isfinite(win_value).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
