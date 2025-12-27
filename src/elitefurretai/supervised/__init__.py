"""
Supervised Learning Module

This module contains supervised learning models, training utilities, and analysis tools.

Models:
    - three_headed_transformer: Main model architecture (teampreview + turn + win)
    - feed_forward_action: Simple feedforward baseline
    - behavior_clone_player: Player agent using trained models

Training:
    - train_utils: Training loop helpers, evaluation metrics, loss functions

Configs:
    - action_improved.cfg: Optimized action prediction config
    - win.cfg: Win prediction focused config
    - teampreview.cfg: Teampreview-only config
    - fine_tune.cfg: Fine-tuning configuration
"""

# Models
from elitefurretai.supervised.behavior_clone_player import BCPlayer

# Model architectures
from elitefurretai.supervised.model_archs import (
    init_linear_layer,
    ResidualBlock,
    GatedResidualBlock,
    GroupedFeatureEncoder,
    DNN,
    FlexibleThreeHeadedModel,
)

# Training utilities
from elitefurretai.supervised.train_utils import (
    evaluate,
    analyze,
    topk_cross_entropy_loss,
    focal_topk_cross_entropy_loss,
    format_time,
)

__all__ = [
    # Models
    "BCPlayer",
    # Model architectures
    "init_linear_layer",
    "ResidualBlock",
    "GatedResidualBlock",
    "GroupedFeatureEncoder",
    "DNN",
    "FlexibleThreeHeadedModel",
    # Training utilities
    "evaluate",
    "analyze",
    "topk_cross_entropy_loss",
    "focal_topk_cross_entropy_loss",
    "format_time",
]
