"""Checkpoint helpers for RL training."""

import os
from datetime import datetime
from typing import Dict, Tuple

import torch

from elitefurretai.rl.config import RNaDConfig
from elitefurretai.rl.players import RNaDAgent


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


__all__ = ["save_checkpoint", "load_checkpoint"]
