"""config.py — Hierarchical configuration for RNaD RL training.

What this file is
-----------------
A single source of truth for every knob that the RL pipeline cares about.
Configs are stored as YAML and loaded into typed `dataclass` objects so the
rest of the code can reference fields like `config.algorithm.clip_range`
instead of digging into untyped dicts.

Why it exists
-------------
RL has *a lot* of hyperparameters. Without this module they would be scattered
across CLI flags, magic numbers in code, and informal conventions in docs.
Centralizing them in YAML means:
  - Every training run is reproducible from one file (the YAML).
  - We can `--resume` a run and recover the exact hyperparameters used.
  - Sweeps and ablations only have to vary the YAML.

Big-picture map of the sub-configs
----------------------------------
- AlgorithmConfig    — RNaD/PPO loss math (clip range, GAE λ, KL weight α, …)
- PortfolioConfig    — Multiple reference models for KL regularization
- ExplorationConfig  — Action-sampling: temperature anneal, top-p (nucleus)
- OptimizerConfig    — AdamW / Adam, per-group LRs, schedule (cosine/linear)
- ValueHeadConfig    — C51 distributional value head bin layout
- ArchitectureConfig — Model shape: transformer layer sizes, heads
- HardwareConfig     — Worker topology, device, battle backend, batching
- CurriculumConfig   — Opponent mix, team pools, BC model paths, ghosts
- ExploiterConfig    — In-process exploiter co-training (graduation-based)
- TrainingConfig     — Loop limits, checkpoint intervals, wandb

The two annealing helpers (`temperature_at_step`, `ent_coef_at_step`) and the
LR scheduler (`lr_lambda`) live on the top-level RNaDConfig because they need
to read across multiple sub-configs.

Battle backends
---------------
There are two ways to actually play battles during training:
  - "showdown_websocket": a real local Pokemon Showdown server, talked to over
    websockets. The current primary path, as proven faster w/ optimizations.
  - "rust_engine": an in-process Rust simulator. Faster but less production-
    proven. Kept available for fallback and parity checks.
"""

import math
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import yaml

# Backend identifiers — referenced by hardware.battle_backend and by the
# worker/learner branches that pick which battle execution path to use.
SHOWDOWN_WEBSOCKET_BACKEND = "showdown_websocket"
RUST_ENGINE_BACKEND = "rust_engine"
SUPPORTED_BATTLE_BACKENDS = {SHOWDOWN_WEBSOCKET_BACKEND, RUST_ENGINE_BACKEND}


@dataclass
class AlgorithmConfig:
    """PPO + RNaD loss coefficients and GAE parameters.

    Quick glossary for ML researchers new to this code:
    - clip_range: PPO's importance-weight clip ε. Larger ε = bigger policy
      updates allowed per epoch; smaller ε = more conservative.
    - ent_coef / ent_coef_end: entropy bonus γ. Linearly annealed from start
      to end over `temperature_anneal_steps` updates. Encourages exploration.
    - gae_lambda: λ for Generalized Advantage Estimation (Schulman 2016).
      0 = pure 1-step TD, 1 = pure Monte-Carlo. 0.95 is the standard sweet spot.
    - gamma: discount factor for future rewards. We use 1.0 in our YAML
      (episodic / undiscounted) because battles are short and end with a clear
      win/loss reward.
    - max_grad_norm: gradient clipping threshold. Prevents loss spikes from
      blowing up the model.
    - rnad_alpha: weight α on the KL-to-reference term — the "Regularized" in
      Regularized Nash Dynamics. This is the term that keeps the policy from
      drifting too far from a frozen anchor and prevents catastrophic forgetting.
    - vf_coef: weight on the value-head loss in the total loss.
    """

    clip_range: float = 0.2
    ent_coef: float = 0.01
    ent_coef_end: float = 0.001
    gae_lambda: float = 0.95
    gamma: float = 0.99
    max_grad_norm: float = 0.5
    rnad_alpha: float = 0.01
    vf_coef: float = 0.5
    # PPO mini-epochs: how many gradient passes to take over the same batch
    # per RL update. K=1 reproduces the original behavior. Standard PPO
    # uses 3-10; we default to 3 — multiplies learning per battle without
    # changing actor collection rate. Old log probs from collection are
    # reused across epochs; reference model forwards happen once per update
    # (frozen across epochs).
    ppo_epochs: int = 3
    # Optional safety: stop the PPO inner loop early when approximate KL
    # between current and old policies exceeds this threshold. None disables.
    # Recommended range when used: 0.01 - 0.05.
    ppo_kl_early_stop: Optional[float] = None


@dataclass
class PortfolioConfig:
    """Portfolio of reference models used for RNaD regularization. Different
    from curriculum which creates trajectories by playing the agent.

    To replicate standard RNaD (single fixed reference, no portfolio),
    set max_portfolio_size=1 and portfolio_update_strategy="recent".
    PortfolioRNaDLearner reduces to the base algorithm in that configuration.

    TODO: enumerate that we want to be able to only accept: recent, best
    and diverse; implement all of these, and add an explanation on what they
    are
    """

    max_portfolio_size: int = 5
    portfolio_add_interval: int = 500
    portfolio_update_strategy: str = "recent"


@dataclass
class ExploiterConfig:
    """In-process exploiter co-training (league-style graduation).

    The exploiter is a *second learner* that runs in the same process as
    main training. It optimizes a pure exploitation objective (no RNaD
    regularization) against a frozen copy of the main agent ("the
    victim"). Snapshots are NOT taken on a fixed schedule — the live
    exploiter must "graduate" by sustaining a win rate above
    `graduation_threshold` over the last `graduation_window` battles
    against the victim. On graduation, the exploiter is saved to disk
    (entering the EXPLOITERS curriculum slot for main to defend against)
    and a new generation begins, fresh-initialized from the BC checkpoint
    to find a *different* exploit.

    Why graduation instead of fixed-interval snapshots:
        - Quality control: every entry in the curriculum is a validated
          weakness, not an arbitrary checkpoint.
        - Diversity: fresh-init each generation forces convergence to a
          different basin (next gen finds a different exploit because
          main has learned to defend the previous one).
        - Implicit difficulty curriculum: as main improves, hitting the
          threshold gets harder. When main is robust enough that no
          exploiter can graduate, the pipeline naturally winds down
          rather than polluting the curriculum with weak exploits.

    Why in-process (and not a separate script as in the previous
    subprocess pipeline):
        - Reuses the existing worker pool — no extra CPU pressure.
        - Uses spare GPU capacity for the second learner's gradients.
        - One run, one config, one wandb stream — no orchestration.

    Required collaborators:
        - `curriculum.curriculum_weights["train_exploiter"]` is the
          single gate: > 0 activates the entire pipeline. Default 0.0
          keeps it inert. Typical when on: 0.20 (20% to exploiter,
          80% to main).
        - `curriculum.bc_model_path` is the init source for each new
          generation (must be set when train_exploiter > 0).

    Schedule:

        update     0 ────── 200 ──────────────── 1200 ──────────────── ???
                   │         │                     │                    │
                   │         └─ warmup ends        └─ victim refresh #1 └─ generation N
                   └─ training start                                       graduates when
                      (main only)                                          win_rate > threshold
                                                                           over graduation_window
                                                                           OR max_updates_per_gen
    """

    # Per-update batch size for the exploiter learner. Smaller than main's
    # `training.train_batch_size` (256) because the exploiter sees only
    # ~20% of trajectories — at 64, gradient updates fire at a similar
    # wall-clock cadence to main. Tune jointly with the curriculum weight.
    batch_size: int = 64

    # Win-rate threshold for graduating an exploiter (saving it to the
    # curriculum and starting a fresh generation). Computed over the last
    # `graduation_window` train_exploiter battles. At 0.65 with
    # window=1000, SE ≈ 1.5% — so a graduating exploiter has true win rate
    # ≥ 62% with high confidence. Tighter thresholds (0.70+) catch only
    # strong exploits but risk pipeline stall once main is robust; looser
    # thresholds (0.60) churn the pool with borderline cases.
    graduation_threshold: float = 0.70

    # Number of recent train_exploiter battles to compute the rolling win
    # rate over. Larger = more confident graduation gate (lower variance)
    # but slower pool growth. 1000 ≈ ~5–10 min wall-clock — fast enough
    # to keep generations turning over, slow enough to filter noise.
    graduation_window: int = 1000

    # Stall safeguard. If a generation can't graduate within this many
    # exploiter updates, force a reset — main is likely robust against
    # this basin and we should try a different one. Without this, an
    # exploiter that's stuck at 60% would train forever, wasting compute
    # and never adding to the curriculum.
    max_updates_per_generation: int = 5000

    # Updates between victim weight refreshes. The victim is the frozen
    # opponent the exploiter trains *against* — refreshed by copying
    # `main_agent.state_dict()` into `victim_agent` and broadcasting.
    #
    # Why freezing matters: an exploiter needs a stationary target to
    # converge. Pointing it at live main creates a non-stationary RL
    # problem (gradients chase a moving distribution → oscillation, not
    # exploitation). Freezing turns each refresh window into a clean
    # ~1000-update RL problem with a fixed reward landscape.
    #
    # Why not freeze forever: target would represent an outdated main;
    # the exploit becomes irrelevant by the time it lands in the
    # curriculum.
    #
    # Indexed on *main* updates (not exploiter updates) so the schedule
    # is independent of `batch_size`. At a 20% trajectory share, 1000
    # main updates ≈ ~200 exploiter updates of stable target.
    victim_refresh_interval: int = 1000

    # Initial main updates during which the exploiter learner is skipped
    # and `train_exploiter` is excluded from curriculum sampling. Without
    # this, the exploiter would train against a barely-trained (or
    # BC-init) main — learning trivial exploits that vanish once main
    # improves, and poisoning the first snapshot fed into the curriculum.
    # Warmup gives main time to settle into the RNaD basin before
    # adversarial co-training begins.
    warmup_updates: int = 200

    # Single LR override applied to BOTH backbone and heads parameter
    # groups in the exploiter's optimizer. Slower than main's heads_lr
    # (3e-4 typical) to keep the adversarial chase stable — the
    # exploiter has a shorter generation budget than main, so we can't
    # afford instability from too-aggressive updates.
    lr: float = 1e-4

    # Constant entropy bonus for the exploiter (no annealing — main
    # anneals because main trains for 100K+ updates, but the exploiter's
    # 5K-cap per generation doesn't justify a schedule). Matches the
    # warmup-phase main entropy and prevents premature mode collapse
    # onto a single exploit.
    ent_coef: float = 0.005


@dataclass
class ExplorationConfig:
    """Temperature annealing and nucleus sampling for action selection."""

    # TODO: add comments here on what these are, the implications of raising/
    # lowering them are, and where they're used
    temperature_anneal_steps: int = 50000
    temperature_end: float = 0.5
    temperature_start: float = 1.5
    top_p: float = 0.95


@dataclass
class OptimizerConfig:
    """Optimizer type, learning rates, and LR schedule.

    backbone_lr / heads_lr set per-group learning rates. lr is a fallback
    global rate used when a sub-group is not otherwise configured.
    """

    backbone_lr: float = 1e-4
    backbone_weight_decay: float = 1e-4
    heads_lr: float = 3e-4
    heads_weight_decay: float = 0.0
    lr: float = 1e-4
    schedule: str = "cosine"
    type: str = "adamw"
    warmup_steps: int = 1000
    weight_decay: float = 1e-4


@dataclass
class ValueHeadConfig:
    """C51 distributional value head support bounds and bin count."""

    num_value_bins: int = 51
    value_max: float = 1.0
    value_min: float = -1.0


@dataclass
class ArchitectureConfig:
    """Model architecture hyperparameters for TransformerThreeHeadedModel."""

    # Core encoder
    dropout: float = 0.15
    early_attention_heads: int = 16
    early_layers: List[int] = field(default_factory=lambda: [4096, 2048, 2048, 1024])
    grouped_encoder_aggregated_dim: int = 4096
    grouped_encoder_hidden_dim: int = 512
    late_attention_heads: int = 32
    late_layers: List[int] = field(default_factory=lambda: [2048, 2048, 1024, 1024])
    max_seq_len: int = 40
    pokemon_attention_heads: int = 16
    # Decision heads
    teampreview_attention_heads: int = 8
    teampreview_head_dropout: float = 0.3
    teampreview_head_layers: List[int] = field(default_factory=lambda: [512, 256])
    turn_head_layers: List[int] = field(default_factory=lambda: [2048, 1024, 1024, 1024])
    # Optional deep value head: if non-empty, a ResidualBlock stack with these
    # widths is inserted between late_ff_stack and the final value linear,
    # giving the value head its own integrative depth instead of sharing all
    # representation capacity with the policy head. When empty, the legacy
    # 2-layer MLP (output_size -> 128 -> num_value_bins) is used.
    value_head_layers: List[int] = field(default_factory=lambda: [])
    # Number bank embeddings
    number_bank_embedding_dim: int = 16
    number_bank_hp_bins: int = 100
    number_bank_power_bins: int = 250
    number_bank_stat_bins: int = 600
    # Transformer backbone
    transformer_dropout: float = 0.1
    transformer_ff_dim: int = 2048
    transformer_heads: int = 16
    transformer_layers: int = 6

    # TODO: treat these all as constants; they shouldnt be configs
    use_causal_mask: bool = True
    use_decision_tokens: bool = True


@dataclass
class HardwareConfig:
    """Worker topology, device, and battle backend selection.

    Topology mental model:
        num_workers   = how many separate OS processes generate trajectories.
                        Each worker has its own Python interpreter (bypasses GIL).
        num_players   = total number of "player slots" across all workers.
                        These split into pairs (player + opponent).
        num_servers   = how many local Showdown server processes to launch
                        (each is single-threaded Node.js, so we run several).
        num_battles_per_pair = how many battles each pair plays per task batch
                               before the worker checks for new weights.

    Inference batching (showdown_websocket path):
        batch_size    = max number of pending policy queries the inference
                        loop will gather before flushing to the model.
        batch_timeout = max seconds to wait while assembling a batch.
                        Smaller = lower latency, smaller average batch.
                        Larger = bigger batches, more throughput, more lag.

    Other knobs:
        max_battle_steps          = trajectory truncation. Battles longer than
                                    this have early steps dropped (we keep the
                                    last N decisions, since later turns matter more).
        use_mixed_precision       = use FP16 autocast in the learner (~2× speedup
                                    on RTX cards, ~30% VRAM savings).
        rust_max_concurrent_..    = override for the Rust backend's per-worker
                                    concurrency cap. Only used if rust backend.
    """

    batch_size: int = 16
    batch_timeout: float = 0.05
    battle_backend: str = SHOWDOWN_WEBSOCKET_BACKEND
    device: str = "cuda"
    max_battle_steps: int = 40
    num_battles_per_pair: int = 20
    num_players: int = 3
    num_servers: int = 3
    num_workers: int = 3
    # TODO: is this necessary? what does this do?
    # Wondering if we should remove, especially
    # given we are moving away from Rust; don't want this extra
    # complexity. Feels we can add a comment in the rust engine part
    # of this codebase that this is a potential configurable parameter
    # if we ever want to revisit
    rust_max_concurrent_battles_override: Optional[int] = None
    showdown_start_port: int = 8000
    use_mixed_precision: bool = True
    use_multiprocessing: bool = False
    # Per-player concurrent-battle cap (poke-env's `max_concurrent_battles`
    # kwarg on Player). None = poke-env's default of 1, which serialises
    # battle setup behind the previous battle's full duration via
    # `_battle_count_queue.put(None)`
    # Set to `num_battles_per_pair` to let a player run all its pair's
    # battles concurrently without queue-blocking. Only flows to
    # `BatchInferencePlayer` constructions in `WorkerOpponentFactory`
    # (Showdown training path); analysis scripts are unaffected.
    max_concurrent_battles_per_player: Optional[int] = 20

    # TODO: revisit. Shouldnt this be a constant?
    # torch.compile the inference model in workers. The 2026-05-13 profile
    # showed model forward (linear + transformer + layer_norm) was the
    # dominant useful work (~36% OwnTime); compile should fuse small
    # kernels and remove Python dispatch overhead. None/False = eager;
    # "default" / "reduce-overhead" / "max-autotune" select the mode.
    # Note: first call after launch pays compile cost (10–60s typical);
    # subsequent calls reuse the cached graph.
    compile_inference_model: Optional[str] = None

    # TODO: I think this should be a default, and set to true?
    # Centralized inference: when True, the trainer process owns a
    # ModelRegistry of InferenceServices (one per registered model
    # name) and workers submit requests via mp.Queue instead of holding
    # their own model copies. Unlocks bigger batch sizes (combines all
    # workers' main requests into one forward) and lets torch.compile
    # be applied once trainer-side instead of N times worker-side.
    # See src/elitefurretai/rl/RL.md section 8b for the full design.
    enable_centralized_inference: bool = False

    def __post_init__(self) -> None:
        if self.battle_backend not in SUPPORTED_BATTLE_BACKENDS:
            raise ValueError(
                f"battle_backend must be one of {sorted(SUPPORTED_BATTLE_BACKENDS)}, "
                f"got {self.battle_backend!r}"
            )

    @property
    def players_per_worker(self) -> int:
        return (self.num_players + self.num_workers - 1) // self.num_workers

    @property
    def cpu_budget_per_worker(self) -> int:
        logical_cpus = os.cpu_count() or 1
        return max(1, (logical_cpus + self.num_workers - 1) // max(1, self.num_workers))

    @property
    def max_players_per_server(self) -> int:
        return (self.num_players + self.num_servers - 1) // self.num_servers

    @property
    def rust_max_concurrent_battles_per_worker(self) -> int:
        if self.rust_max_concurrent_battles_override is not None:
            return max(
                1,
                min(self.num_battles_per_pair, self.rust_max_concurrent_battles_override),
            )
        if self.num_battles_per_pair <= 0:
            return 1
        worker_player_budget = max(1, self.players_per_worker)
        worker_cpu_budget = self.cpu_budget_per_worker
        return min(
            self.num_battles_per_pair,
            max(2, min(worker_player_budget, worker_cpu_budget)),
        )


@dataclass
class CurriculumConfig:
    """Opponent sampling, team pools, BC models, and ghost/exploiter directories."""

    # Battle format and team sources
    agent_team_path: Optional[str] = None
    base_team_path: str = "data/teams"
    battle_format: str = "gen9vgc2023regc"
    team_pool_path: Optional[str] = None
    # Behavior cloning model used as opponent
    bc_model_path: Optional[str] = "data/models/bc_model.pt"
    # Opponent sampling distribution (must sum to 1.0).
    #
    # `exploiters` = main fights *frozen snapshots* of past exploiters from
    #   `<run_dir>/exploiters/`. Always available (snapshots may be empty).
    # `train_exploiter` = exploiter-vs-victim battles whose trajectories feed
    #   the in-process exploiter learner. Only sampled when
    #   `train_exploiter > 0` AND the warmup window has passed.
    #   Default 0.0 so the slot exists but stays inert until enabled.
    curriculum_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "self_play": 0.30,
            "bc_player": 0.10,
            "exploiters": 0.10,
            "ghosts": 0.15,
            "train_exploiter": 0.10,
            "max_damage": 0.05,
            "vgc_bench_baseline": 0.05,
            "simple_heuristic_baseline": 0.05,
            "max_base_power_baseline": 0.05,
            "random_baseline": 0.05,
        }
    )
    # TODO: add comment on the algorithm
    adaptive_curriculum: bool = True
    # Exploiter and ghost model pool sizes (paths are derived from training.run_dir)
    max_exploiter_models: int = 10
    max_ghosts: int = 10
    # TODO: are there any of these vgcbench that can just be constants?
    # like usernames are not valuable at all, and many of these will never change
    # VGC bench external runner
    auto_launch_external_vgcbench: bool = False
    dedicated_vgcbench_workers: int = 0  # TODO: i think this is old, and we can remove
    external_vgcbench_python_executable: Optional[str] = None
    external_vgcbench_startup_wait_s: float = 5.0
    external_vgcbench_team_file: str = "data/teams/gen9vgc2024regg/vgcbench.txt"
    external_vgcbench_usernames: Optional[List[str]] = None
    vgc_bench_checkpoint_path: str = "data/models/vgc-bench-sb3-model.zip"


@dataclass
class TrainingConfig:
    """Training loop, checkpointing, and logging settings.

    Exploiter co-training settings live in `ExploiterConfig` and are
    accessed at `config.exploiter.*` rather than `config.training.*`.
    """

    # TODO: probably helpful to talka bout what this does
    checkpoint_interval: int = 1000
    # TODO: valuable to comment on all the possibilities here
    embedder_feature_set: str = "raw"
    initialize_path: Optional[str] = None
    log_interval: int = 1
    max_updates: int = 100000
    resume_from: Optional[str] = None
    save_dir: str = "data/models"
    # Set at runtime in train.py main() after the run name is resolved.
    # Layout: <save_dir>/<run_name>/{ghosts,exploiters,*.pt}
    run_dir: Optional[str] = None
    train_batch_size: int = 32
    use_wandb: bool = True
    wandb_project: str = "elitefurretai-rnad"
    wandb_run_name: Optional[str] = None
    wandb_tags: Optional[List[str]] = None
    # Combined RSS ceiling (trainer + all child processes) at which the
    # memory watchdog requests a graceful shutdown. On WSL2 with 23 GiB of
    # memory, 20 GB leaves ~3 GiB headroom for the in-flight learner step,
    # checkpoint save, and wandb flush before Hyper-V would otherwise kill
    # the VM. Set to None or 0 to disable.
    memory_watchdog_threshold_gb: Optional[float] = 20.0


# TODO: is this necessary or used at all?
def _make_sub(klass: Any, d: Dict[str, Any]) -> Any:
    """Construct a dataclass from a dict, ignoring unknown keys."""
    known = {f for f in klass.__dataclass_fields__}
    return klass(**{k: v for k, v in d.items() if k in known})


@dataclass
class RNaDConfig:
    """Hierarchical configuration for RNaD training.

    Fields are grouped into nine typed sub-configs. Access them as:
        config.algorithm.clip_range
        config.hardware.num_workers
        config.curriculum.curriculum_weights
    Methods (temperature_at_step, lr_lambda, save/load) remain on this class.
    """

    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    architecture: ArchitectureConfig = field(default_factory=ArchitectureConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    exploiter: ExploiterConfig = field(default_factory=ExploiterConfig)
    exploration: ExplorationConfig = field(default_factory=ExplorationConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    value_head: ValueHeadConfig = field(default_factory=ValueHeadConfig)

    # ── Computed properties (delegate to sub-configs) ──────────────────────────

    @property
    def players_per_worker(self) -> int:
        return self.hardware.players_per_worker

    @property
    def cpu_budget_per_worker(self) -> int:
        return self.hardware.cpu_budget_per_worker

    @property
    def max_players_per_server(self) -> int:
        return self.hardware.max_players_per_server

    @property
    def rust_max_concurrent_battles_per_worker(self) -> int:
        return self.hardware.rust_max_concurrent_battles_per_worker

    @property
    def num_showdown_servers(self) -> int:
        return self.hardware.num_servers

    # ── Annealing helpers ──────────────────────────────────────────────────────

    def temperature_at_step(self, step: int) -> float:
        """Compute linearly annealed temperature at a given training step."""
        cfg = self.exploration
        progress = min(step / max(cfg.temperature_anneal_steps, 1), 1.0)
        return cfg.temperature_start + progress * (
            cfg.temperature_end - cfg.temperature_start
        )

    def ent_coef_at_step(self, step: int) -> float:
        """Compute linearly annealed entropy coefficient at a given training step."""
        cfg = self.exploration
        progress = min(step / max(cfg.temperature_anneal_steps, 1), 1.0)
        return self.algorithm.ent_coef + progress * (
            self.algorithm.ent_coef_end - self.algorithm.ent_coef
        )

    def lr_lambda(self, step: int) -> float:
        """Compute LR multiplier for scheduler at a given step."""
        warmup = self.optimizer.warmup_steps
        schedule = self.optimizer.schedule
        total = self.training.max_updates

        if step < warmup:
            return step / max(warmup, 1)
        if schedule == "cosine":
            progress = (step - warmup) / max(total - warmup, 1)
            return 0.5 * (1 + math.cos(math.pi * progress))
        elif schedule == "linear":
            progress = (step - warmup) / max(total - warmup, 1)
            return max(1.0 - progress, 0.0)
        return 1.0

    # ── Serialization ──────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    @classmethod
    def load(cls, filepath: str) -> "RNaDConfig":
        with open(filepath, "r") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RNaDConfig":
        """Create a RNaDConfig from a dict (flat or nested). Extra keys are ignored."""

        return cls(
            algorithm=_make_sub(AlgorithmConfig, data.get("algorithm", {})),
            architecture=_make_sub(ArchitectureConfig, data.get("architecture", {})),
            curriculum=_make_sub(CurriculumConfig, data.get("curriculum", {})),
            exploiter=_make_sub(ExploiterConfig, data.get("exploiter", {})),
            exploration=_make_sub(ExplorationConfig, data.get("exploration", {})),
            hardware=_make_sub(HardwareConfig, data.get("hardware", {})),
            optimizer=_make_sub(OptimizerConfig, data.get("optimizer", {})),
            portfolio=_make_sub(PortfolioConfig, data.get("portfolio", {})),
            training=_make_sub(TrainingConfig, data.get("training", {})),
            value_head=_make_sub(ValueHeadConfig, data.get("value_head", {})),
        )

    def __str__(self) -> str:
        lines = ["RNaD Training Configuration", "=" * 50]
        for key, value in self.to_dict().items():
            lines.append(f"{key}: {value}")
        return "\n".join(lines)

    def verify(self) -> None:
        """Verify that configured paths exist and are consistent."""
        cur = self.curriculum
        trn = self.training

        if cur.bc_model_path is not None:
            assert os.path.exists(cur.bc_model_path), (
                f"BC model path not found: {cur.bc_model_path}"
            )
        assert os.path.exists(cur.base_team_path), (
            f"Base team path not found: {cur.base_team_path}"
        )

        if cur.team_pool_path:
            full_pool = os.path.join(
                cur.base_team_path, cur.battle_format, cur.team_pool_path
            )
            assert os.path.exists(full_pool), f"Team pool path not found: {full_pool}"
        if cur.agent_team_path:
            assert os.path.exists(cur.agent_team_path), (
                f"Agent team path not found: {cur.agent_team_path}"
            )
            if os.path.isdir(cur.agent_team_path):
                team_files = [
                    f for f in os.listdir(cur.agent_team_path) if f.endswith(".txt")
                ]
                assert len(team_files) > 0, (
                    f"No .txt team files in agent_team_path: {cur.agent_team_path}"
                )
        if trn.resume_from:
            assert os.path.exists(trn.resume_from), (
                f"Resume checkpoint not found: {trn.resume_from}"
            )
        if trn.initialize_path:
            assert os.path.exists(trn.initialize_path), (
                f"Initialize checkpoint not found: {trn.initialize_path}"
            )

        if cur.auto_launch_external_vgcbench:
            assert (
                cur.external_vgcbench_usernames
                and len(cur.external_vgcbench_usernames) > 0
            ), (
                "external_vgcbench_usernames must be set when auto_launch_external_vgcbench=True"
            )
            assert cur.external_vgcbench_python_executable, (
                "external_vgcbench_python_executable must be set when auto_launch_external_vgcbench=True"
            )
            assert os.path.exists(cur.external_vgcbench_python_executable), (
                f"external_vgcbench_python_executable not found: {cur.external_vgcbench_python_executable}"
            )
            assert os.path.exists(cur.external_vgcbench_team_file), (
                f"external_vgcbench_team_file not found: {cur.external_vgcbench_team_file}"
            )


def get_default_config() -> RNaDConfig:
    return RNaDConfig()
