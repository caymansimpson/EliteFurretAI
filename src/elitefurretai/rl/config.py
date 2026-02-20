import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import yaml


@dataclass
class RNaDConfig:
    """Configuration for RNaD training."""

    # ===== RNaD related =====
    clip_range: float = (
        0.2  # PPO clip range for policy updates (prevents too large policy changes)
    )
    ent_coef: float = 0.01  # Entropy bonus coefficient (encourages exploration)
    gae_lambda: float = (
        0.95  # GAE lambda for advantage estimation (trade-off between bias and variance)
    )
    gamma: float = 0.99  # Discount factor for future rewards in RL
    max_grad_norm: float = (
        0.5  # Maximum gradient norm for gradient clipping (prevents exploding gradients)
    )
    max_portfolio_size: int = (
        5  # Learner-side: max number of reference models used in Portfolio RNaD loss
    )
    portfolio_add_interval: int = (
        5000  # Add new snapshot to portfolio every N training updates
    )
    portfolio_update_strategy: str = "diverse"  # Strategy for portfolio management: "diverse" (maximize diversity), "best" (keep strongest), or "recent" (keep latest)
    ref_update_interval: int = 1000  # Update reference model(s) every N training updates (for RNaD regularization)
    rnad_alpha: float = (
        0.01  # RNaD regularization strength (KL divergence penalty vs reference policy)
    )
    use_portfolio_regularization: bool = (
        False  # Use portfolio of multiple reference models instead of single reference
    )
    vf_coef: float = (
        0.5  # Value function loss coefficient (weight of value loss in total loss)
    )

    # ===== Performance / Hardware related =====
    batch_size: int = 16  # Batch size for model inference during battle play
    batch_timeout: float = 0.05  # Max wait time (s) before sending incomplete batch for inference
    device: str = "cuda"  # Device for training ("cuda" for GPU, "cpu" for CPU)
    max_battle_steps: int = (
        40  # Maximum trajectory steps per battle before forfeiting to prevent memory explosion.
        # VGC battles typically end in 10-20 turns (~20-40 steps). Battles exceeding this
        # are likely stuck in error-retry loops and should be terminated.
    )
    num_battles_per_pair: int = (
        20  # Number of battles each player-opponent pair runs per batch.
        # Higher values = more efficient GPU utilization but longer between weight updates.
    )
    num_players: int = 3  # Total number of concurrent battle players across all workers
    num_servers: int = 3  # Number of Pokemon Showdown servers to launch
    num_workers: int = 3  # Number of worker processes (IMPALA architecture)
    showdown_start_port: int = (
        8000  # Starting port for Showdown servers (increments for multiple servers)
    )
    use_mixed_precision: bool = (
        True  # Enable automatic mixed precision (FP16) for ~2x speedup and reduced memory
    )
    use_multiprocessing: bool = (
        False  # Option 1: Use true multiprocessing (separate processes) instead of threading.
        # Each process has its own GIL and model copy, enabling true parallelism.
        # Requires more memory but can achieve higher throughput.
    )

    # ===== Curriculum / Opponent related =====
    base_team_path: str = (
        "data/teams"  # Root directory containing format folders (e.g., data/teams/)
    )
    battle_format: str = "gen9vgc2023regc"  # Pokemon Showdown battle format string
    bc_action_path: str = (
        "data/models/bc_action_model.pt"  # BC model for turn actions in opponent pool
    )
    bc_teampreview_path: str = (
        "data/models/bc_teampreview_model.pt"  # BC model for teampreview in opponent pool
    )
    bc_win_path: str = (
        "data/models/bc_win_model.pt"  # BC model for win prediction in opponent pool
    )
    curriculum: Dict[str, float] = field(
        default_factory=lambda: {
            "self_play": 0.40,  # Proportion of games vs current main model
            "bc_player": 0.20,  # Proportion of games vs BC models
            "exploiters": 0.20,  # Proportion of games vs trained exploiter agents
            "ghosts": 0.20,  # Proportion of games vs past checkpoints of main model ("ghosts")
        }
    )  # Curriculum weights for opponent sampling (must sum to 1.0). Keys must match opponent_pool.py
    adaptive_curriculum: bool = True  # Whether to adapt curriculum weights based on win rates against each opponent type (dynamic difficulty adjustment)
    exploiter_models_dir: str = "data/models/exploiters"  # Directory of exploiter checkpoints used as exploiter opponents
    max_exploiter_models: int = (
        10  # Worker-side: max number of exploiter checkpoints tracked for opponent sampling
    )
    max_loaded_exploiter_models: int = (
        2  # Worker-side: max exploiter models cached in memory
    )
    max_loaded_ghost_models: int = (
        2  # Worker-side: max ghost/past models cached in memory
    )
    max_past_models: int = (
        10  # Opponent-side: max number of past checkpoints retained for ghost opponents
    )
    past_models_dir: str = "data/models/past_versions"  # Directory to store past model versions for opponent pool
    vgc_bench_checkpoint_path: str = "data/models/vgc-bench-sb3-model.zip"  # SB3 zip checkpoint for vgc-bench baseline
    external_vgcbench_usernames: Optional[List[str]] = None  # Optional external Showdown usernames for vgc-bench opponents (isolated env)
    external_vgcbench_startup_wait_s: float = 5.0  # Seconds each worker waits before first external vgc-bench challenge batch
    auto_launch_external_vgcbench: bool = False  # If True, train.py launches/stops external vgc-bench runner processes automatically
    external_vgcbench_python_executable: Optional[str] = None  # Python executable from the dedicated vgc-bench environment
    external_vgcbench_runner_script: str = "src/elitefurretai/rl/analyze/vgcbench_external_runner.py"  # Runner script launched by train.py
    external_vgcbench_team_file: str = "data/teams/gen9vgc2024regg/vgcbench.txt"  # Team file consumed by external runners
    external_vgcbench_log_dir: str = "data/logs/vgcbench_runners"  # Directory for auto-launched runner logs
    external_vgcbench_log_to_files: bool = True  # If False, route auto-launched runner stdout/stderr to /dev/null
    external_vgcbench_password: Optional[str] = None  # Optional showdown password for external runner accounts
    external_vgcbench_n_challenges: int = 1000000  # Large default so runners remain available through long training runs
    external_vgcbench_runner_wait_for_server_timeout: float = 180.0  # Server wait timeout passed to runner
    external_vgcbench_accept_open_team_sheet: bool = False  # Whether auto-launched runners should accept OTS
    team_pool_path: Optional[str] = (
        None  # Optional subdirectory within format (e.g., "easy") to restrict team sampling. If None, samples from all teams in format
    )
    use_random_teams: bool = (
        True  # Whether to sample random teams from pool for each battle (vs fixed team)
    )

    # ===== Logging related =====
    log_interval: int = (
        1  # Log metrics to wandb every N training updates (1 = every update)
    )
    use_wandb: bool = True  # Enable Weights & Biases logging for metrics and artifacts
    wandb_project: str = "elitefurretai-rnad"  # W&B project name
    wandb_run_name: Optional[str] = (
        None  # Custom run name (None = auto-generated timestamp)
    )
    wandb_tags: Optional[List[str]] = None  # Tags for organizing W&B runs

    # ===== Training related =====
    checkpoint_interval: int = 1000  # Save model checkpoint every N training updates
    embedder_feature_set: str = "full"  # Feature set used by Embedder when constructing models
    exploiter_ent_coef: float = 0.02  # Entropy coefficient for exploiter (higher than main to encourage diverse exploitation strategies)
    exploiter_eval_games: int = (
        100  # Number of evaluation games to measure exploiter win rate
    )
    exploiter_interval: int = 10000  # Train new exploiter every N training updates
    exploiter_lr: float = (
        1e-4  # Learning rate for exploiter training (separate from main agent)
    )
    exploiter_min_win_rate: float = 0.55  # Minimum win rate for exploiter to be added to opponent pool (filters weak exploiters)
    exploiter_team_pool_path: Optional[str] = (
        None  # Team pool for exploiter training (None = use main team_pool_path)
    )
    exploiter_updates: int = (
        50000  # Number of training updates when training each exploiter agent
    )
    initialize_path: Optional[str] = (
        None  # Initial model checkpoint to initialize weights from (starts fresh training at step 0)
    )
    lr: float = 1e-4  # Learning rate for Adam optimizer
    max_updates: int = 100000  # Total number of training updates before stopping
    resume_from: Optional[str] = (
        None  # Specific checkpoint to resume training from (restores model + optimizer + step state)
    )
    save_dir: str = (
        "data/models"  # Directory to save training checkpoints and final models
    )
    train_batch_size: int = (
        32  # Number of trajectories to collect before performing a training update
    )
    train_exploiters: bool = False  # Enable automatic exploiter training pipeline (trains agents to exploit main model)

    # ===== Architecture related =====
    dropout: float = 0.15
    early_attention_heads: int = 16
    early_layers: List[int] = field(default_factory=lambda: [4096, 2048, 2048, 1024])
    gated_residuals: bool = False
    grouped_encoder_aggregated_dim: int = 4096
    grouped_encoder_hidden_dim: int = 512
    late_attention_heads: int = 32
    late_layers: List[int] = field(default_factory=lambda: [2048, 2048, 1024, 1024])
    lstm_hidden_size: int = 512
    lstm_layers: int = 4
    max_seq_len: int = 40
    pokemon_attention_heads: int = 16
    teampreview_attention_heads: int = 8
    teampreview_head_dropout: float = 0.3
    teampreview_head_layers: List[int] = field(default_factory=lambda: [512, 256])
    turn_head_layers: List[int] = field(default_factory=lambda: [2048, 1024, 1024, 1024])
    use_grouped_encoder: bool = True

    def __post_init__(self):
        if self.max_portfolio_size <= 0:
            raise ValueError("max_portfolio_size must be > 0")
        if self.max_past_models <= 0:
            raise ValueError("max_past_models must be > 0")
        if self.max_exploiter_models <= 0:
            raise ValueError("max_exploiter_models must be > 0")
        if self.max_loaded_exploiter_models <= 0:
            raise ValueError("max_loaded_exploiter_models must be > 0")
        if self.max_loaded_ghost_models <= 0:
            raise ValueError("max_loaded_ghost_models must be > 0")

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

    @classmethod
    def from_dict(cls, data: Dict) -> "RNaDConfig":
        """
        Create a RNaDConfig from a dictionary (e.g., loaded from YAML or JSON).
        Ignores extra keys and fills missing keys with defaults.
        """
        # Only use keys that are valid fields for the dataclass
        field_names = set(f.name for f in cls.__dataclass_fields__.values())
        filtered = {k: v for k, v in data.items() if k in field_names}

        # Handle curriculum default if missing or None
        if "curriculum" in filtered and filtered["curriculum"] is None:
            filtered["curriculum"] = cls().curriculum

        # Backward compatibility: old configs may still use exploiter_registry_path
        if "exploiter_models_dir" not in filtered and "exploiter_registry_path" in data:
            old_registry_path = data.get("exploiter_registry_path")
            if isinstance(old_registry_path, str) and old_registry_path:
                filtered["exploiter_models_dir"] = os.path.join(
                    os.path.dirname(old_registry_path), "exploiters"
                )
            else:
                filtered["exploiter_models_dir"] = cls().exploiter_models_dir

        return cls(**filtered)

    def __str__(self) -> str:
        """Pretty print config."""
        lines = ["RNaD Training Configuration", "=" * 50]
        for key, value in asdict(self).items():
            lines.append(f"{key}: {value}")
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return asdict(self)

    def verify(self) -> None:
        """Verify that configured paths exist and are consistent."""
        assert os.path.exists(
            self.bc_teampreview_path
        ), f"BC teampreview path not found: {self.bc_teampreview_path}"
        assert os.path.exists(
            self.bc_action_path
        ), f"BC action path not found: {self.bc_action_path}"
        assert os.path.exists(
            self.bc_win_path
        ), f"BC win path not found: {self.bc_win_path}"
        assert os.path.exists(
            self.base_team_path
        ), f"Base team path not found: {self.base_team_path}"

        if self.team_pool_path:
            full_team_pool_path = os.path.join(
                self.base_team_path, self.battle_format, self.team_pool_path
            )
            assert os.path.exists(
                full_team_pool_path
            ), f"Team pool path not found: {full_team_pool_path}"
        if self.resume_from:
            assert os.path.exists(
                self.resume_from
            ), f"Resume checkpoint not found: {self.resume_from}"
        if self.initialize_path:
            assert os.path.exists(
                self.initialize_path
            ), f"Initialize checkpoint not found: {self.initialize_path}"

        if self.auto_launch_external_vgcbench:
            assert (
                self.external_vgcbench_usernames
                and len(self.external_vgcbench_usernames) > 0
            ), "external_vgcbench_usernames must be set when auto_launch_external_vgcbench=True"
            assert (
                self.external_vgcbench_python_executable
            ), "external_vgcbench_python_executable must be set when auto_launch_external_vgcbench=True"
            assert os.path.exists(
                self.external_vgcbench_python_executable
            ), (
                "external_vgcbench_python_executable not found: "
                f"{self.external_vgcbench_python_executable}"
            )
            assert os.path.exists(
                self.external_vgcbench_runner_script
            ), (
                "external_vgcbench_runner_script not found: "
                f"{self.external_vgcbench_runner_script}"
            )
            assert os.path.exists(
                self.external_vgcbench_team_file
            ), (
                "external_vgcbench_team_file not found: "
                f"{self.external_vgcbench_team_file}"
            )

    @property
    def players_per_worker(self) -> int:
        """Calculate how many players each worker should run (rounded up)."""
        return (self.num_players + self.num_workers - 1) // self.num_workers

    @property
    def max_players_per_server(self) -> int:
        """Calculate maximum players per server based on balanced allocation.

        Each player needs 2 connections during battles (player + opponent).
        This distributes total connections across servers with headroom.
        """
        return (self.num_players + self.num_servers - 1) // self.num_servers

    @property
    def num_showdown_servers(self) -> int:
        """Alias for backward compatibility."""
        return self.num_servers


def get_default_config() -> RNaDConfig:
    """Get default configuration."""
    return RNaDConfig()


if __name__ == "__main__":
    # Example usage
    config = get_default_config()
    config.save("src/elitefurretai/rlconfigs/default.yaml")
    print("Default config saved to configs/default.yaml")
