"""Model I/O helpers for RL training.

Includes model construction utilities and checkpoint load/save helpers.
"""

import os
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import torch

from elitefurretai.etl import MDBO, Embedder
from elitefurretai.rl.config import RNaDConfig
from elitefurretai.rl.players import RNaDAgent
from elitefurretai.supervised import FlexibleThreeHeadedModel

MODEL_ARCH_CONFIG_KEYS = (
    "battle_format",
    "embedder_feature_set",
    "early_layers",
    "late_layers",
    "lstm_layers",
    "lstm_hidden_size",
    "dropout",
    "gated_residuals",
    "early_attention_heads",
    "late_attention_heads",
    "use_grouped_encoder",
    "grouped_encoder_hidden_dim",
    "grouped_encoder_aggregated_dim",
    "pokemon_attention_heads",
    "teampreview_head_layers",
    "teampreview_head_dropout",
    "teampreview_attention_heads",
    "turn_head_layers",
    "max_seq_len",
)


def _normalize_config_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return list(value)
    return value


def is_checkpoint_compatible_with_model_config(
    checkpoint_path: str,
    model_config: Dict[str, Any],
) -> bool:
    """Return True when a checkpoint config matches the current model architecture."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_config = checkpoint.get("config")
    if not isinstance(checkpoint_config, dict):
        return False

    for key in MODEL_ARCH_CONFIG_KEYS:
        current_value = _normalize_config_value(model_config.get(key))
        checkpoint_value = _normalize_config_value(checkpoint_config.get(key))
        if current_value != checkpoint_value:
            return False

    return True


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


def load_agent_from_checkpoint(
    checkpoint_path: str,
    device: str,
    embedder: Optional[Embedder] = None,
) -> RNaDAgent:
    model, _, _ = load_model_from_checkpoint(checkpoint_path, device, embedder)
    return RNaDAgent(model)


def save_checkpoint(
    model: RNaDAgent,
    optimizer,
    step: int,
    config: RNaDConfig,
    curriculum: Dict,
    save_dir: str = "data/models",
):
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, f"main_model_step_{step}.pt")

    checkpoint = {
        "model_state_dict": model.model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "curriculum": curriculum,
        "config": config.to_dict(),
        "timestamp": datetime.now().isoformat(),
    }

    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")
    return filepath


def load_checkpoint(
    filepath: str, model: RNaDAgent, optimizer, device: str
) -> Tuple[int, RNaDConfig]:
    print(f"Resuming from checkpoint: {filepath}")
    checkpoint = torch.load(filepath, map_location=device)

    model.model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    step = checkpoint["step"]
    old_config = RNaDConfig.from_dict(checkpoint["config"])

    print(f"Resumed from step {step}")
    return step, old_config


__all__ = [
    "build_model_from_config",
    "load_model_from_checkpoint",
    "load_agent_from_checkpoint",
    "is_checkpoint_compatible_with_model_config",
    "save_checkpoint",
    "load_checkpoint",
]
