import yaml
import os
from typing import Optional, Dict, List
from dataclasses import dataclass, asdict, field


@dataclass
class RNaDConfig:
    """Configuration for RNaD training."""

    # Model paths
    checkpoint_path: str = "data/models/bc_action_model.pt"
    bc_teampreview_path: str = "data/models/bc_teampreview_model.pt"
    bc_action_path: str = "data/models/bc_action_model.pt"
    bc_win_path: str = "data/models/bc_win_model.pt"
    resume_from: Optional[str] = None  # Resume from checkpoint

    # Team configuration
    team_pool_path: str = "data/teams/gen9vgc2023regulationc/easy"  # Directory with team files (required)
    use_random_teams: bool = True  # Sample random teams from pool

    # Training hyperparameters
    num_workers: int = 1
    players_per_worker: int = 4
    batch_size: int = 16
    train_batch_size: int = 32
    max_updates: int = 100000  # Total training updates
    lr: float = 1e-4
    rnad_alpha: float = 0.01
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5

    # Portfolio regularization
    use_portfolio_regularization: bool = False  # Use multiple reference models
    max_portfolio_size: int = 5  # Maximum reference models to keep
    portfolio_update_strategy: str = "diverse"  # "diverse", "best", or "recent"
    portfolio_add_interval: int = 5000  # Add new reference every N updates

    # Mixed precision training
    use_mixed_precision: bool = True  # Enable FP16 for 2x speedup
    gradient_clip: float = 0.5  # Gradient clipping for stability

    # Checkpointing
    checkpoint_interval: int = 1000
    ref_update_interval: int = 1000
    save_dir: str = "data/models"
    past_models_dir: str = "data/models/past_versions"
    max_past_models: int = 10

    # Opponent pool configuration
    curriculum: Dict[str, float] = field(default_factory=lambda: {
        'self_play': 0.40,
        'bc_player': 0.20,
        'exploiters': 0.20,
        'past_versions': 0.20
    })
    exploiter_registry_path: str = "data/models/exploiter_registry.json"

    # Exploiter training
    train_exploiters: bool = False  # Enable automatic exploiter training
    exploiter_interval: int = 10000  # Train exploiter every N updates (also known as exploiter_check_interval)
    exploiter_updates: int = 50000  # Number of updates for exploiter training (also known as exploiter_train_steps)
    exploiter_min_win_rate: float = 0.55  # Minimum win rate to add exploiter (also known as exploiter_win_threshold)
    exploiter_eval_games: int = 100  # Number of games to evaluate exploiter
    exploiter_output_dir: str = "data/models/exploiters"  # Directory to save exploiters
    exploiter_team_pool_path: Optional[str] = None  # Team pool for exploiter training (None = use main team_pool_path)
    exploiter_lr: float = 1e-4
    exploiter_ent_coef: float = 0.02  # Higher entropy for exploration

    # Hardware configuration
    device: str = "cuda"
    num_showdown_servers: int = 1
    showdown_start_port: int = 8000

    # Logging
    use_wandb: bool = True
    wandb_project: str = "elitefurretai-rnad"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_tags: Optional[List[str]] = None
    log_interval: int = 10
    curriculum_update_interval: int = 500

    # Format
    battle_format: str = "gen9vgc2023regulationc"

    def save(self, filepath: str):
        """Save config to YAML file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            yaml.dump(asdict(self), f, default_flow_style=False)

    @classmethod
    def load(cls, filepath: str) -> 'RNaDConfig':
        """Load config from YAML file."""
        with open(filepath, 'r') as f:
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
