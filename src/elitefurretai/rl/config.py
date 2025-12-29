import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import yaml


@dataclass
class RNaDConfig:
    """Configuration for RNaD training."""

    # ===== Format =====
    battle_format: str = "gen9vgc2023regulationc"  # Pokemon Showdown battle format string

    # ===== Model paths =====
    initialize_path: Optional[str] = (
        None  # Initial model checkpoint to initialize weights from (starts fresh training at step 0)
    )
    bc_teampreview_path: str = (
        "data/models/bc_teampreview_model.pt"  # BC model for teampreview in opponent pool
    )
    bc_action_path: str = (
        "data/models/bc_action_model.pt"  # BC model for turn actions in opponent pool
    )
    bc_win_path: str = (
        "data/models/bc_win_model.pt"  # BC model for win prediction in opponent pool
    )
    resume_from: Optional[str] = (
        None  # Specific checkpoint to resume training from (restores model + optimizer + step state)
    )

    # ===== Team configuration =====
    base_team_path: str = (
        "data/teams"  # Root directory containing format folders (e.g., data/teams/)
    )
    team_pool_path: Optional[str] = (
        None  # Optional subdirectory within format (e.g., "easy") to restrict team sampling. If None, samples from all teams in format
    )
    use_random_teams: bool = (
        True  # Whether to sample random teams from pool for each battle (vs fixed team)
    )

    # ===== Training hyperparameters =====
    num_workers: int = 1  # Number of parallel worker threads generating battle data
    players_per_worker: int = 4  # Number of concurrent players per worker (total players = num_workers * players_per_worker)
    batch_size: int = 16  # Batch size for model inference during battle play
    train_batch_size: int = (
        32  # Number of trajectories to collect before performing a training update
    )
    max_updates: int = 100000  # Total number of training updates before stopping
    lr: float = 1e-4  # Learning rate for Adam optimizer
    rnad_alpha: float = (
        0.01  # RNaD regularization strength (KL divergence penalty vs reference policy)
    )
    gamma: float = 0.99  # Discount factor for future rewards in RL
    gae_lambda: float = (
        0.95  # GAE lambda for advantage estimation (trade-off between bias and variance)
    )
    clip_range: float = (
        0.2  # PPO clip range for policy updates (prevents too large policy changes)
    )
    ent_coef: float = 0.01  # Entropy bonus coefficient (encourages exploration)
    vf_coef: float = (
        0.5  # Value function loss coefficient (weight of value loss in total loss)
    )

    # ===== Portfolio regularization =====
    use_portfolio_regularization: bool = (
        False  # Use portfolio of multiple reference models instead of single reference
    )
    max_portfolio_size: int = (
        5  # Maximum number of reference models to maintain in portfolio
    )
    portfolio_update_strategy: str = "diverse"  # Strategy for portfolio management: "diverse" (maximize diversity), "best" (keep strongest), or "recent" (keep latest)
    portfolio_add_interval: int = (
        5000  # Add new snapshot to portfolio every N training updates
    )

    # ===== Mixed precision training =====
    use_mixed_precision: bool = (
        True  # Enable automatic mixed precision (FP16) for ~2x speedup and reduced memory
    )
    gradient_clip: float = (
        0.5  # Maximum gradient norm for gradient clipping (prevents exploding gradients)
    )

    # ===== Checkpointing =====
    checkpoint_interval: int = 1000  # Save model checkpoint every N training updates
    ref_update_interval: int = 1000  # Update reference model(s) every N training updates (for RNaD regularization)
    save_dir: str = (
        "data/models"  # Directory to save training checkpoints and final models
    )
    past_models_dir: str = "data/models/past_versions"  # Directory to store past model versions for opponent pool
    max_past_models: int = (
        10  # Maximum number of past model versions to keep in opponent pool
    )

    # ===== Opponent pool configuration =====
    curriculum: Dict[str, float] = field(
        default_factory=lambda: {
            "self_play": 0.40,  # Proportion of games vs current main model
            "bc_player": 0.20,  # Proportion of games vs BC models
            "exploiters": 0.20,  # Proportion of games vs trained exploiter agents
            "ghosts": 0.20,  # Proportion of games vs past checkpoints of main model ("ghosts")
        }
    )  # Curriculum weights for opponent sampling (must sum to 1.0). Keys must match opponent_pool.py
    exploiter_registry_path: str = "data/models/exploiter_registry.json"  # JSON file tracking trained exploiter models
    curriculum_update_interval: int = (
        500  # Update curriculum weights adaptively every N updates based on win rates
    )

    # ===== Exploiter training =====
    train_exploiters: bool = False  # Enable automatic exploiter training pipeline (trains agents to exploit main model)
    exploiter_interval: int = 10000  # Train new exploiter every N training updates
    exploiter_updates: int = (
        50000  # Number of training updates when training each exploiter agent
    )
    exploiter_min_win_rate: float = 0.55  # Minimum win rate for exploiter to be added to opponent pool (filters weak exploiters)
    exploiter_eval_games: int = (
        100  # Number of evaluation games to measure exploiter win rate
    )
    exploiter_output_dir: str = (
        "data/models/exploiters"  # Directory to save successful exploiter models
    )
    exploiter_team_pool_path: Optional[str] = (
        None  # Team pool for exploiter training (None = use main team_pool_path)
    )
    exploiter_lr: float = (
        1e-4  # Learning rate for exploiter training (separate from main agent)
    )
    exploiter_ent_coef: float = 0.02  # Entropy coefficient for exploiter (higher than main to encourage diverse exploitation strategies)

    # ===== Hardware configuration =====
    device: str = "cuda"  # Device for training ("cuda" for GPU, "cpu" for CPU)
    num_showdown_servers: int = (
        1  # Number of Pokemon Showdown servers to run (for distributed battle generation)
    )
    showdown_start_port: int = (
        8000  # Starting port for Showdown servers (increments for multiple servers)
    )

    # ===== Logging =====
    use_wandb: bool = True  # Enable Weights & Biases logging for metrics and artifacts
    wandb_project: str = "elitefurretai-rnad"  # W&B project name
    wandb_run_name: Optional[str] = (
        None  # Custom run name (None = auto-generated timestamp)
    )
    wandb_tags: Optional[List[str]] = None  # Tags for organizing W&B runs
    log_interval: int = (
        1  # Log metrics to wandb every N training updates (1 = every update)
    )

    def save(self, filepath: str):
        """Save config to YAML file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False)

    @classmethod
    def load(cls, filepath: str) -> "RNaDConfig":
        """Load config from YAML file."""
        with open(filepath, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def __str__(self) -> str:
        """Pretty print config."""
        lines = ["RNaD Training Configuration", "=" * 50]
        for key, value in asdict(self).items():
            lines.append(f"{key}: {value}")
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return asdict(self)


def get_default_config() -> RNaDConfig:
    """Get default configuration."""
    return RNaDConfig()


if __name__ == "__main__":
    # Example usage
    config = get_default_config()
    config.save("src/elitefurretai/rlconfigs/default.yaml")
    print("Default config saved to configs/default.yaml")
