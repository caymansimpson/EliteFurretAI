"""Model I/O helpers for RL training.

Includes model construction utilities and checkpoint load/save helpers.
"""

import os
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, Union

import torch

from elitefurretai.etl import MDBO, Embedder
from elitefurretai.rl.config import RNaDConfig
from elitefurretai.rl.players import RNaDAgent
from elitefurretai.supervised import FlexibleThreeHeadedModel
from elitefurretai.supervised.model_archs import TransformerThreeHeadedModel

MODEL_ARCH_CONFIG_KEYS = (
    "battle_format",
    "embedder_feature_set",
    "early_layers",
    "late_layers",
    "lstm_layers",
    "lstm_hidden_size",
    "dropout",
    "early_attention_heads",
    "late_attention_heads",
    "grouped_encoder_hidden_dim",
    "grouped_encoder_aggregated_dim",
    "pokemon_attention_heads",
    "teampreview_head_layers",
    "teampreview_head_dropout",
    "teampreview_attention_heads",
    "turn_head_layers",
    "max_seq_len",
    "num_value_bins",
    "value_min",
    "value_max",
    "number_bank_hp_bins",
    "number_bank_stat_bins",
    "number_bank_power_bins",
    "number_bank_embedding_dim",
    # Transformer
    "use_transformer",
    "transformer_layers",
    "transformer_heads",
    "transformer_ff_dim",
    "transformer_dropout",
    "use_decision_tokens",
    "use_causal_mask",
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
) -> Union[FlexibleThreeHeadedModel, TransformerThreeHeadedModel]:
    use_transformer = model_config.get("use_transformer", False)

    # Shared kwargs for both model classes
    common_kwargs: Dict[str, Any] = dict(
        embedder=embedder,
        early_layers=model_config["early_layers"],
        late_layers=model_config["late_layers"],
        dropout=model_config.get("dropout", 0.1),
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
        num_value_bins=model_config.get("num_value_bins", 51),
        value_min=model_config.get("value_min", -1.0),
        value_max=model_config.get("value_max", 1.0),
        # Grouped feature expansion hyperparameters
        number_bank_hp_bins=model_config.get("number_bank_hp_bins", 100),
        number_bank_stat_bins=model_config.get("number_bank_stat_bins", 600),
        number_bank_power_bins=model_config.get("number_bank_power_bins", 250),
        number_bank_embedding_dim=model_config.get("number_bank_embedding_dim", 16),
        number_bank_damage_bins=model_config.get("number_bank_damage_bins", 600),
        number_bank_damage_embed_dim=model_config.get("number_bank_damage_embed_dim", 4),
        number_bank_turn_bins=model_config.get("number_bank_turn_bins", 40),
        number_bank_turn_embed_dim=model_config.get("number_bank_turn_embed_dim", 16),
        number_bank_rating_bins=model_config.get("number_bank_rating_bins", 100),
        number_bank_rating_embed_dim=model_config.get("number_bank_rating_embed_dim", 16),
        ability_embed_dim=model_config.get("ability_embed_dim", 16),
        item_embed_dim=model_config.get("item_embed_dim", 16),
        species_embed_dim=model_config.get("species_embed_dim", 32),
        move_embed_dim=model_config.get("move_embed_dim", 16),
    )

    if use_transformer:
        model: Union[FlexibleThreeHeadedModel, TransformerThreeHeadedModel] = TransformerThreeHeadedModel(
            **common_kwargs,
            transformer_layers=model_config.get("transformer_layers", 6),
            transformer_heads=model_config.get("transformer_heads", 16),
            transformer_ff_dim=model_config.get("transformer_ff_dim", 2048),
            transformer_dropout=model_config.get("transformer_dropout", 0.1),
            use_decision_tokens=model_config.get("use_decision_tokens", True),
            use_causal_mask=model_config.get("use_causal_mask", True),
            # Pass LSTM params for config compat (unused internally)
            lstm_layers=model_config.get("lstm_layers", 2),
            lstm_hidden_size=model_config.get("lstm_hidden_size", 512),
            early_attention_heads=model_config.get("early_attention_heads", 8),
            late_attention_heads=model_config.get("late_attention_heads", 8),
        ).to(device)
    else:
        model = FlexibleThreeHeadedModel(
            **common_kwargs,
            lstm_layers=model_config.get("lstm_layers", 2),
            lstm_hidden_size=model_config.get("lstm_hidden_size", 512),
            early_attention_heads=model_config.get("early_attention_heads", 8),
            late_attention_heads=model_config.get("late_attention_heads", 8),
        ).to(device)

    if state_dict:
        model.load_state_dict(state_dict)

    return model


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: str,
    embedder: Optional[Embedder] = None,
) -> Tuple[Union[FlexibleThreeHeadedModel, TransformerThreeHeadedModel], Embedder, Dict[str, Any]]:
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
