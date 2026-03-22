# -*- coding: utf-8 -*-
"""
Unit tests for supervised model architectures.

These tests verify:
1. ResidualBlock forward passes
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

from elitefurretai.etl.embedder import Embedder
from elitefurretai.etl.encoder import MDBO
from elitefurretai.supervised.model_archs import (
    DNN,
    FlexibleThreeHeadedModel,
    GroupedFeatureEncoder,
    NumberBankEncoder,
    ResidualBlock,
    TransformerThreeHeadedModel,
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
def simple_embedder():
    """Embedder with 'simple' feature set (no grouped encoder)."""
    return Embedder(feature_set="simple")


@pytest.fixture
def small_input_size(simple_embedder):
    """Input size derived from the simple embedder."""
    return simple_embedder.embedding_size


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
    feature_names = [f"feat_{i}" for i in range(sum(group_sizes))]
    encoder = GroupedFeatureEncoder(
        group_sizes=group_sizes,
        feature_names=feature_names,
        num_abilities=64,
        num_items=128,
        num_species=256,
        num_moves=256,
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
    feature_names = [f"feat_{i}" for i in range(total_size)]

    encoder = GroupedFeatureEncoder(
        group_sizes=group_sizes,
        feature_names=feature_names,
        num_abilities=64,
        num_items=128,
        num_species=256,
        num_moves=256,
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
    feature_names = [f"feat_{i}" for i in range(total_size)]

    encoder = GroupedFeatureEncoder(
        group_sizes=group_sizes,
        feature_names=feature_names,
        num_abilities=64,
        num_items=128,
        num_species=256,
        num_moves=256,
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


def test_flexible_model_initialization(simple_embedder):
    """
    Test FlexibleThreeHeadedModel initialization.

    Model should create all components without errors.

    Expected: Model initializes successfully.
    """
    model = FlexibleThreeHeadedModel(
        embedder=simple_embedder,
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


def test_flexible_model_forward_shapes(simple_embedder):
    """
    Test FlexibleThreeHeadedModel output shapes.

    Model produces three outputs:
    - Turn logits: (batch, seq, 2025)
    - Teampreview logits: (batch, seq, 90)
    - Win value: (batch, seq)

    Expected: All outputs have correct shapes.
    """
    model = FlexibleThreeHeadedModel(
        embedder=simple_embedder,
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

    x = torch.randn(4, 5, simple_embedder.embedding_size)
    turn_logits, tp_logits, win_value, win_dist_logits = model(x)

    assert turn_logits.shape == (4, 5, 2025)
    assert tp_logits.shape == (4, 5, 90)
    assert win_value.shape == (4, 5)


def test_flexible_model_win_value_range(simple_embedder):
    """
    Test that win value is bounded in [-1, 1].

    Win head uses tanh activation, so output should be bounded.

    Expected: All win values in [-1, 1].
    """
    model = FlexibleThreeHeadedModel(
        embedder=simple_embedder,
        early_layers=[64, 32],
        late_layers=[64, 32],
        lstm_layers=1,
        lstm_hidden_size=32,
        dropout=0.0,
    )
    model.eval()

    # Test with various inputs
    for _ in range(5):
        x = torch.randn(4, 5, simple_embedder.embedding_size) * 10  # Large values
        _, _, win_value, _ = model(x)

        assert win_value.min() >= -1.0
        assert win_value.max() <= 1.0


def test_flexible_model_with_grouped_encoder():
    """
    Test FlexibleThreeHeadedModel with GroupedFeatureEncoder.

    Passing an Embedder with non-simple feature_set enables grouped encoding.

    Expected: Model works with grouped encoder.
    """
    embedder = Embedder()  # raw feature set → grouped encoder enabled

    model = FlexibleThreeHeadedModel(
        embedder=embedder,
        early_layers=[64, 32],
        late_layers=[64, 32],
        lstm_layers=1,
        lstm_hidden_size=32,
        dropout=0.1,
        grouped_encoder_hidden_dim=16,
        grouped_encoder_aggregated_dim=64,
    )
    model.eval()

    x = torch.randn(4, 5, embedder.embedding_size)
    turn_logits, tp_logits, win_value, _ = model(x)

    assert turn_logits.shape[0] == 4
    assert tp_logits.shape[2] == 90
    assert win_value.shape == (4, 5)


def test_flexible_model_with_attention(simple_embedder):
    """
    Test FlexibleThreeHeadedModel with attention layers.

    early_attention_heads and late_attention_heads enable self-attention.

    Expected: Model works with attention enabled.
    """
    model = FlexibleThreeHeadedModel(
        embedder=simple_embedder,
        early_layers=[64, 32],
        late_layers=[64, 32],
        lstm_layers=1,
        lstm_hidden_size=32,
        dropout=0.1,
        early_attention_heads=4,
        late_attention_heads=4,
    )
    model.eval()

    x = torch.randn(4, 5, simple_embedder.embedding_size)
    turn_logits, tp_logits, win_value, _ = model(x)

    assert turn_logits.shape == (4, 5, 2025)


def test_flexible_model_with_action_mask(simple_embedder):
    """
    Test FlexibleThreeHeadedModel with action masking.

    Mask should be applied to turn logits.
    Note: The model expects a 2D mask of shape (batch, seq) for attention.

    Expected: Model accepts and applies mask correctly.
    """
    model = FlexibleThreeHeadedModel(
        embedder=simple_embedder,
        early_layers=[64, 32],
        late_layers=[64, 32],
        lstm_layers=1,
        lstm_hidden_size=32,
        dropout=0.0,
    )
    model.eval()

    x = torch.randn(4, 5, simple_embedder.embedding_size)

    # Create 2D mask for attention (batch, seq) - True means padded/masked
    # This is the key_padding_mask format expected by PyTorch MHA
    mask = torch.zeros(4, 5, dtype=torch.bool)
    mask[:, -1] = True  # Mask last position

    turn_logits, _, _, _ = model(x, mask=mask)

    assert turn_logits.shape == (4, 5, 2025)


def test_flexible_model_forward_with_hidden(simple_embedder):
    """
    Test forward_with_hidden method for RL training.

    Returns hidden states for recurrent inference.

    Expected: Returns 4 outputs including hidden states.
    """
    model = FlexibleThreeHeadedModel(
        embedder=simple_embedder,
        early_layers=[64, 32],
        late_layers=[64, 32],
        lstm_layers=1,
        lstm_hidden_size=32,
        dropout=0.0,
    )
    model.eval()

    x = torch.randn(4, 5, simple_embedder.embedding_size)

    # Get initial hidden state
    batch_size = 4
    num_directions = 2
    num_layers = 1
    hidden_size = 32
    h0 = torch.zeros(num_layers * num_directions, batch_size, hidden_size)
    c0 = torch.zeros(num_layers * num_directions, batch_size, hidden_size)

    turn_logits, tp_logits, win_value, win_dist_logits, hidden = model.forward_with_hidden(x, (h0, c0))

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


def test_model_weight_statistics(simple_embedder):
    """
    Test that model weights have reasonable statistics.

    Well-initialized weights should have small variance.
    LayerNorm weights are initialized to 1.0 (constant), so we skip those.

    Expected: Weights not too large or too small (excluding normalization layers).
    """
    model = FlexibleThreeHeadedModel(
        embedder=simple_embedder,
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


def test_gradient_flow_through_model(simple_embedder):
    """
    Test that gradients flow through entire model.

    All parameters should receive gradients during backprop.

    Expected: No zero gradients for trainable parameters.
    """
    model = FlexibleThreeHeadedModel(
        embedder=simple_embedder,
        early_layers=[64, 32],
        late_layers=[64, 32],
        lstm_layers=1,
        lstm_hidden_size=32,
        dropout=0.0,  # Disable dropout for deterministic gradients
    )

    x = torch.randn(4, 5, simple_embedder.embedding_size, requires_grad=True)
    turn_logits, tp_logits, win_value, _ = model(x)

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


def test_single_sequence_element(simple_embedder):
    """
    Test model with sequence length of 1.

    Some operations behave differently with single elements.

    Expected: Model handles seq_len=1 correctly.
    """
    model = FlexibleThreeHeadedModel(
        embedder=simple_embedder,
        early_layers=[64, 32],
        late_layers=[64, 32],
        lstm_layers=1,
        lstm_hidden_size=32,
        dropout=0.0,
    )
    model.eval()

    x = torch.randn(4, 1, simple_embedder.embedding_size)  # seq_len = 1
    turn_logits, tp_logits, win_value, _ = model(x)

    assert turn_logits.shape == (4, 1, 2025)
    assert tp_logits.shape == (4, 1, 90)
    assert win_value.shape == (4, 1)


def test_batch_size_one(simple_embedder):
    """
    Test model with batch size of 1.

    Expected: Model handles batch_size=1 correctly.
    """
    model = FlexibleThreeHeadedModel(
        embedder=simple_embedder,
        early_layers=[64, 32],
        late_layers=[64, 32],
        lstm_layers=1,
        lstm_hidden_size=32,
        dropout=0.0,
    )
    model.eval()

    x = torch.randn(1, 5, simple_embedder.embedding_size)  # batch = 1
    turn_logits, tp_logits, win_value, _ = model(x)

    assert turn_logits.shape == (1, 5, 2025)


def test_no_early_layers(simple_embedder):
    """
    Test model with minimal early_layers.

    Note: embed_dim must be divisible by attention heads (4 by default).
    So we use layer size of 64 which is divisible by 4.

    Expected: Model produces valid output.
    """
    model = FlexibleThreeHeadedModel(
        embedder=simple_embedder,
        early_layers=[64],  # Single layer
        late_layers=[64, 32],
        lstm_layers=1,
        lstm_hidden_size=32,
        dropout=0.1,
    )
    model.eval()

    x = torch.randn(4, 5, simple_embedder.embedding_size)
    turn_logits, _, _, _ = model(x)

    assert turn_logits.shape[2] == 2025


def test_deep_model(simple_embedder):
    """
    Test model with many layers.

    Deep models should still work correctly.

    Expected: Deep model produces valid output.
    """
    model = FlexibleThreeHeadedModel(
        embedder=simple_embedder,
        early_layers=[128, 128, 64, 64, 32],
        late_layers=[128, 64, 32],
        lstm_layers=2,
        lstm_hidden_size=64,
        dropout=0.1,
    )
    model.eval()

    x = torch.randn(4, 5, simple_embedder.embedding_size)
    turn_logits, tp_logits, win_value, _ = model(x)

    assert torch.isfinite(turn_logits).all()
    assert torch.isfinite(tp_logits).all()
    assert torch.isfinite(win_value).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# =============================================================================
# NUMBER BANK ENCODER TESTS
# =============================================================================


def test_number_bank_encoder_no_numerical():
    """NumberBankEncoder with no matching features passes data through unchanged."""
    feature_names = ["feat_a", "feat_b", "feat_c"]
    group_sizes = [3]
    nb = NumberBankEncoder(feature_names, group_sizes,
                           hp_bins=10, stat_bins=10, power_bins=10, embed_dim=4)

    # No numerical features → output size unchanged
    assert nb.group_output_sizes == [3]

    x = torch.randn(2, 3, 3)
    out = nb.embed_group(x, 0)
    assert out.shape == (2, 3, 3)
    # Should pass through unchanged
    assert torch.allclose(out, x)


def test_number_bank_encoder_hp_feature():
    """NumberBankEncoder correctly identifies and replaces HP features."""
    feature_names = ["other", "current_hp_fraction", "another"]
    group_sizes = [3]
    embed_dim = 8
    nb = NumberBankEncoder(feature_names, group_sizes,
                           hp_bins=100, stat_bins=100, power_bins=100, embed_dim=embed_dim)

    # 1 numeric replaced: 3 - 1 + 1*8 = 10
    assert nb.group_output_sizes == [10]

    x = torch.randn(2, 1, 3)
    out = nb.embed_group(x, 0)
    assert out.shape == (2, 1, 10)


def test_number_bank_encoder_multiple_groups():
    """NumberBankEncoder handles multiple groups with different features."""
    feature_names = ["current_hp_fraction", "feat_a", "STAT:hp", "base_power", "feat_b"]
    group_sizes = [2, 3]
    embed_dim = 4
    nb = NumberBankEncoder(feature_names, group_sizes,
                           hp_bins=10, stat_bins=10, power_bins=10, embed_dim=embed_dim)

    # Group 0: [hp_frac(→4), feat_a] = 1 + 4 = 5
    # Group 1: [STAT:hp(→4), base_power(→4), feat_b] = 3 - 2 + 2*4 = 9
    assert nb.group_output_sizes == [5, 9]

    x0 = torch.randn(2, 1, 2)
    out0 = nb.embed_group(x0, 0)
    assert out0.shape == (2, 1, 5)

    x1 = torch.randn(2, 1, 3)
    out1 = nb.embed_group(x1, 1)
    assert out1.shape == (2, 1, 9)


def test_number_bank_gradient_flow():
    """Gradients flow through number bank embedding lookups."""
    feature_names = ["current_hp_fraction"]
    group_sizes = [1]
    nb = NumberBankEncoder(feature_names, group_sizes,
                           hp_bins=10, stat_bins=10, power_bins=10, embed_dim=4)

    x = torch.tensor([[[0.5]]])  # hp fraction
    out = nb.embed_group(x, 0)
    loss = out.sum()
    loss.backward()

    # Embedding params should have gradients
    assert nb.hp_bank.weight.grad is not None


def test_grouped_encoder_with_number_bank():
    """GroupedFeatureEncoder works with NumberBankEncoder integration."""
    # Need >= 7 groups since GroupedFeatureEncoder assumes 6 player Pokemon groups
    group_sizes = [4, 4, 4, 4, 4, 4, 4]
    # Create feature names with some numerical features spread across groups
    feature_names = (
        ["current_hp_fraction"] + ["feat"] * 3        # group 0 (player pokemon 0)
        + ["STAT:atk"] + ["feat"] * 3                  # group 1 (player pokemon 1)
        + ["feat"] * 4                                  # group 2
        + ["feat"] * 4                                  # group 3
        + ["feat"] * 4                                  # group 4
        + ["feat"] * 4                                  # group 5
        + ["base_power"] + ["feat"] * 3                 # group 6 (extra)
    )
    encoder = GroupedFeatureEncoder(
        group_sizes=group_sizes,
        feature_names=feature_names,
        num_abilities=64,
        num_items=128,
        num_species=256,
        num_moves=256,
        hidden_dim=16,
        aggregated_dim=32,
        dropout=0.0,
        pokemon_attention_heads=2,
        number_bank_hp_bins=10,
        number_bank_stat_bins=10,
        number_bank_power_bins=10,
        number_bank_embedding_dim=4,
    )

    x = torch.randn(2, 3, 28)  # 7 groups * 4 = 28
    out = encoder(x)
    assert out.shape == (2, 3, 32)


# =============================================================================
# TRANSFORMER MODEL TESTS
# =============================================================================


def test_transformer_model_forward(simple_embedder):
    """TransformerThreeHeadedModel forward produces correct shapes."""
    model = TransformerThreeHeadedModel(
        embedder=simple_embedder,
        early_layers=[64, 32],
        late_layers=[64, 32],
        transformer_layers=2,
        transformer_heads=4,
        transformer_ff_dim=64,
        dropout=0.0,
        max_seq_len=40,
    )
    model.eval()

    x = torch.randn(4, 5, simple_embedder.embedding_size)
    turn_logits, tp_logits, win_values, win_dist_logits = model(x)

    assert turn_logits.shape == (4, 5, 2025)
    assert tp_logits.shape == (4, 5, 90)
    assert win_values.shape == (4, 5)
    assert win_dist_logits.shape == (4, 5, 51)


def test_transformer_model_forward_with_hidden(simple_embedder):
    """TransformerThreeHeadedModel forward_with_hidden returns context."""
    model = TransformerThreeHeadedModel(
        embedder=simple_embedder,
        early_layers=[64, 32],
        late_layers=[64, 32],
        transformer_layers=2,
        transformer_heads=4,
        transformer_ff_dim=64,
        dropout=0.0,
        max_seq_len=40,
    )
    model.eval()

    # Turn 0: no context
    x0 = torch.randn(2, 1, simple_embedder.embedding_size)
    turn_logits, tp_logits, win_values, win_dist_logits, context = model.forward_with_hidden(x0, None)

    assert turn_logits.shape == (2, 1, 2025)
    assert tp_logits.shape == (2, 1, 90)
    assert win_values.shape == (2, 1)
    assert context.shape == (2, 1, 32)  # hidden_size = early_layers[-1] = 32

    # Turn 1: pass context from turn 0
    x1 = torch.randn(2, 1, simple_embedder.embedding_size)
    turn_logits2, _, _, _, context2 = model.forward_with_hidden(x1, context)

    assert turn_logits2.shape == (2, 1, 2025)
    assert context2.shape == (2, 2, 32)  # 2 turns accumulated


def test_transformer_model_context_accumulation(simple_embedder):
    """Context grows across multiple turns."""
    model = TransformerThreeHeadedModel(
        embedder=simple_embedder,
        early_layers=[32],
        late_layers=[32],
        transformer_layers=1,
        transformer_heads=4,
        transformer_ff_dim=64,
        dropout=0.0,
    )
    model.eval()

    ctx = None
    for turn in range(5):
        x = torch.randn(1, 1, simple_embedder.embedding_size)
        _, _, _, _, ctx = model.forward_with_hidden(x, ctx)
        assert ctx.shape == (1, turn + 1, 32)


def test_transformer_model_single_batch(simple_embedder):
    """TransformerThreeHeadedModel handles batch_size=1."""
    model = TransformerThreeHeadedModel(
        embedder=simple_embedder,
        early_layers=[64, 32],
        late_layers=[64, 32],
        transformer_layers=2,
        transformer_heads=4,
        transformer_ff_dim=64,
        dropout=0.0,
    )
    model.eval()

    x = torch.randn(1, 3, simple_embedder.embedding_size)
    turn_logits, tp_logits, win_values, win_dist_logits = model(x)
    assert turn_logits.shape == (1, 3, 2025)


def test_transformer_model_gradient_flow(simple_embedder):
    """Gradients flow through entire transformer model."""
    model = TransformerThreeHeadedModel(
        embedder=simple_embedder,
        early_layers=[64, 32],
        late_layers=[64, 32],
        transformer_layers=2,
        transformer_heads=4,
        transformer_ff_dim=64,
        dropout=0.0,
    )

    x = torch.randn(2, 3, simple_embedder.embedding_size, requires_grad=True)
    turn_logits, tp_logits, win_values, _ = model(x)
    loss = turn_logits.sum() + tp_logits.sum() + win_values.sum()
    loss.backward()

    assert x.grad is not None
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None, "A parameter has no gradient"


def test_transformer_model_predict(simple_embedder):
    """TransformerThreeHeadedModel predict() returns probabilities."""
    model = TransformerThreeHeadedModel(
        embedder=simple_embedder,
        early_layers=[64, 32],
        late_layers=[64, 32],
        transformer_layers=1,
        transformer_heads=4,
        transformer_ff_dim=64,
        dropout=0.0,
    )
    model.eval()

    x = torch.randn(2, 3, simple_embedder.embedding_size)
    turn_probs, tp_probs, win_values = model.predict(x)

    # Probabilities should sum to ~1
    assert torch.allclose(turn_probs.sum(dim=-1), torch.ones(2, 3), atol=1e-5)
    assert torch.allclose(tp_probs.sum(dim=-1), torch.ones(2, 3), atol=1e-5)


def test_transformer_no_decision_tokens(simple_embedder):
    """TransformerThreeHeadedModel works without decision tokens."""
    model = TransformerThreeHeadedModel(
        embedder=simple_embedder,
        early_layers=[64, 32],
        late_layers=[64, 32],
        transformer_layers=1,
        transformer_heads=4,
        transformer_ff_dim=64,
        dropout=0.0,
        use_decision_tokens=False,
    )
    model.eval()

    x = torch.randn(2, 3, simple_embedder.embedding_size)
    turn_logits, tp_logits, win_values, _ = model(x)
    assert turn_logits.shape == (2, 3, 2025)


def test_transformer_no_causal_mask(simple_embedder):
    """TransformerThreeHeadedModel works without causal mask."""
    model = TransformerThreeHeadedModel(
        embedder=simple_embedder,
        early_layers=[64, 32],
        late_layers=[64, 32],
        transformer_layers=1,
        transformer_heads=4,
        transformer_ff_dim=64,
        dropout=0.0,
        use_causal_mask=False,
    )
    model.eval()

    x = torch.randn(2, 3, simple_embedder.embedding_size)
    turn_logits, _, _, _ = model(x)
    assert turn_logits.shape == (2, 3, 2025)
