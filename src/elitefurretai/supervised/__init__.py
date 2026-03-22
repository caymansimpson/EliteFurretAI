"""
Supervised Learning Module

This module contains supervised learning models, training utilities, and analysis tools.

Models:
    - three_headed_transformer: Main model architecture (teampreview + turn + win)
    - feed_forward_action: Simple feedforward baseline
    - behavior_clone_player: Player agent using trained models

Training:
    - train_utils: Training loop helpers, evaluation metrics, loss functions

Configs (YAML):
    - action.yaml: Action-only prediction config (raw features)
    - all_in_one_full.yaml: Full all-in-one config (dauntless-hill-95)
    - win.yaml: Win prediction focused config
    - finetune.yaml: Fine-tuning configuration
"""

# Models
from elitefurretai.supervised.behavior_clone_player import BCPlayer

# Model architectures
from elitefurretai.supervised.model_archs import (
    DNN,
    FlexibleThreeHeadedModel,
    GroupedFeatureEncoder,
    NumberBankEncoder,
    ResidualBlock,
    TransformerThreeHeadedModel,
    init_linear_layer,
)

# Training utilities
from elitefurretai.supervised.train_utils import (
    analyze,
    evaluate,
    focal_topk_cross_entropy_loss,
    format_time,
    topk_cross_entropy_loss,
)

__all__ = [
    # Models
    "BCPlayer",
    # Model architectures
    "init_linear_layer",
    "ResidualBlock",
    "GroupedFeatureEncoder",
    "NumberBankEncoder",
    "DNN",
    "FlexibleThreeHeadedModel",
    "TransformerThreeHeadedModel",
    # Training utilities
    "evaluate",
    "analyze",
    "topk_cross_entropy_loss",
    "focal_topk_cross_entropy_loss",
    "format_time",
]
