"""Model construction utilities for RL training."""

from typing import Any, Dict, Optional, Tuple

import torch

from elitefurretai.etl import MDBO, Embedder
from elitefurretai.supervised import FlexibleThreeHeadedModel


def build_model_from_config(
    model_config: Dict[str, Any],
    embedder: Embedder,
    device: str,
    state_dict: Optional[Dict[str, Any]] = None,
) -> FlexibleThreeHeadedModel:
    model = FlexibleThreeHeadedModel(
        input_size=embedder.embedding_size,
        early_layers=model_config["early_layers"],
        late_layers=model_config["late_layers"],
        lstm_layers=model_config.get("lstm_layers", 2),
        lstm_hidden_size=model_config.get("lstm_hidden_size", 512),
        dropout=model_config.get("dropout", 0.1),
        gated_residuals=model_config.get("gated_residuals", False),
        early_attention_heads=model_config.get("early_attention_heads", 8),
        late_attention_heads=model_config.get("late_attention_heads", 8),
        use_grouped_encoder=model_config.get("use_grouped_encoder", False),
        group_sizes=(
            embedder.group_embedding_sizes
            if model_config.get("use_grouped_encoder", False)
            else None
        ),
        grouped_encoder_hidden_dim=model_config.get("grouped_encoder_hidden_dim", 128),
        grouped_encoder_aggregated_dim=model_config.get("grouped_encoder_aggregated_dim", 1024),
        pokemon_attention_heads=model_config.get("pokemon_attention_heads", 2),
        teampreview_head_layers=model_config.get("teampreview_head_layers", []),
        teampreview_head_dropout=model_config.get("teampreview_head_dropout", 0.1),
        teampreview_attention_heads=model_config.get("teampreview_attention_heads", 4),
        turn_head_layers=model_config.get("turn_head_layers", []),
        num_actions=MDBO.action_space(),
        num_teampreview_actions=MDBO.teampreview_space(),
        max_seq_len=model_config.get("max_seq_len", 17),
    ).to(device)

    if state_dict:
        model.load_state_dict(state_dict)

    return model


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: str,
    embedder: Optional[Embedder] = None,
) -> Tuple[FlexibleThreeHeadedModel, Embedder, Dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    state_dict = checkpoint["model_state_dict"]

    if embedder is None:
        embedder = Embedder(
            format=config.get("battle_format", "gen9vgc2023regc"),
            feature_set=config.get("embedder_feature_set", "full"),
            omniscient=False,
        )

    model = build_model_from_config(config, embedder, device, state_dict)
    return model, embedder, config


__all__ = ["build_model_from_config", "load_model_from_checkpoint"]
