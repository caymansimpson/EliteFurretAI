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
- HardwareConfig     — Worker topology, device, batching
- CurriculumConfig   — Opponent mix, team pools, BC model paths, ghosts
- ExploiterConfig    — In-process exploiter co-training (graduation-based)
- TrainingConfig     — Loop limits, checkpoint intervals, wandb

The two annealing helpers (`temperature_at_step`, `ent_coef_at_step`) and the
LR scheduler (`lr_lambda`) live on the top-level RNaDConfig because they need
to read across multiple sub-configs.
"""

import math
import os
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, List, Optional, Union

import yaml


@dataclass
class AlgorithmConfig:
    """PPO + RNaD loss coefficients and GAE parameters.

    Quick glossary for ML researchers new to this code:
    - clip_range: PPO's importance-weight clip ε. Larger ε = bigger policy
      updates allowed per epoch; smaller ε = more conservative.
    - ent_coef / ent_coef_end: entropy bonus γ on the PPO loss term. Linearly
      annealed from start to end over `ExplorationConfig.exploration_anneal_steps`
      updates (shared with the temperature schedule so rollout-side and
      learner-side exploration collapse in sync — see `ent_coef_at_step` below).
      Encourages the *learned* policy to keep entropy and resist mode collapse,
      complementing the sampling-side `temperature` knob.
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

    portfolio_update_strategy (see `PortfolioRNaDLearner._prune_portfolio`):
      - "recent"  : when the portfolio is full, evict the oldest reference.
                    Cheapest and most stable; recommended default.
      - "best"    : evict the reference that has been selected (i.e. been the
                    closest KL anchor) least often. Keeps the most-load-bearing
                    anchors around longer.
      - "random"  : evict a uniformly-random reference. Cheapest unbiased
                    alternative to "recent" — keeps a wider mix of anchor
                    ages on average without preferring any selection signal.
      - "diverse" : reserved name for a diversity-maximising eviction policy
                    (e.g. drop the reference whose representations are closest
                    to another's). Calculates N^2 KL between all models and
                    drops the one that is most similar to the others on a small
                    set of holdout trajectories.
    """

    max_portfolio_size: int = 5
    # Updates between adding the current main model into the portfolio as a
    # new reference. Larger = more diverse anchors but slower portfolio churn.
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
    """Annealing schedule + nucleus sampling for exploration.

    Two exploration mechanisms run on the same schedule, at different points
    in the training loop:
      - Sampling-side: `temperature` reshapes the action distribution at
        rollout time (workers). Pushed via `update_sampling` broadcasts.
        Higher temperature → more diverse trajectories in the buffer.
      - Learner-side: `AlgorithmConfig.ent_coef` adds an entropy bonus to
        the PPO loss. Regularizes the *learned* policy against mode collapse
        during gradient updates.
    Both knobs anneal linearly over `exploration_anneal_steps`. Keeping them
    on one schedule prevents drift (e.g. ent_coef still pushing for
    exploration after temperature has already collapsed to greedy).

    - temperature_start / temperature_end: softmax temperature on action
      logits. Higher = flatter distribution = more exploration. Linearly
      annealed from start → end over `exploration_anneal_steps` updates.
      Raising the start widens early exploration but slows convergence;
      lowering the end sharpens late-stage exploitation.
    - exploration_anneal_steps: horizon of the linear schedule. Drives both
      `temperature_at_step` (sampling) and `ent_coef_at_step` (loss). Should
      be set relative to `training.max_updates` (typically ~half of it).
      Smaller = faster collapse to greedy play; larger = sustained exploration.
    - top_p: nucleus-sampling cutoff applied AFTER temperature. Restricts
      sampling to the smallest set of actions whose cumulative probability
      exceeds top_p. 1.0 disables nucleus filtering; 0.95 is the standard
      "drop the long tail" setting. Used in `inference_trainer.softmax_with_top_p`.
    """

    exploration_anneal_steps: int = 50000
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
    # Mixes a uniform component into the twohot C51 target before the
    # cross-entropy loss: smoothed = (1 - ε) · twohot + ε / num_value_bins.
    # 0.0 = stock behavior (one/two-hot target). Small positive values
    # (0.03–0.10) address value-head over-confidence — see
    # src/elitefurretai/rl/analyze/MODEL_EVALUATION.md Priority 2.
    value_label_smoothing: float = 0.0


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
    value_head_layers: List[int] = field(default_factory=lambda: [512, 256])
    # Multiplier applied to the value-head gradient as it flows back into the
    # shared `late_ff_stack` and the transformer trunk. Forward is identity;
    # backward through the inserted node multiplies grad by this value.
    # 1.0 reproduces the legacy behavior; <1.0 dampens value-side
    # contribution to shared representations, addressing trunk hijacking when
    # `vf_coef * value_loss / |policy_loss|` runs >> 1. Differing values are
    # checkpoint-compatible because weight shapes are unchanged. Lives in
    # ArchitectureConfig because it is a model-construction attribute set on
    # `self` (parallels `use_decision_tokens`), not a loss coefficient.
    value_to_trunk_grad_scale: float = 1.0
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
        rust_max_concurrent_..    = override for the Rust backend's per-worker
                                    concurrency cap. Only used if rust backend.

    Note: mixed-precision is no longer a knob. The learner unconditionally
    uses autocast and creates a GradScaler when running on CUDA.
    """

    batch_size: int = 16
    batch_timeout: float = 0.05
    device: str = "cuda"
    max_battle_steps: int = 40
    num_battles_per_pair: int = 20
    num_players: int = 3
    num_servers: int = 3
    num_workers: int = 3
    showdown_start_port: int = 8000
    use_multiprocessing: bool = False
    # Per-player concurrent-battle cap (poke-env's `max_concurrent_battles`
    # kwarg on Player). None = poke-env's default of 1, which serialises
    # battle setup behind the previous battle's full duration via
    # `_battle_count_queue.put(None)`
    # Set to `num_battles_per_pair` to let a player run all its pair's
    # battles concurrently without queue-blocking. Only flows to
    # `RLTrajectoryPlayer` constructions in `WorkerOpponentFactory`
    # (Showdown training path); analysis scripts are unaffected.
    max_concurrent_battles_per_player: Optional[int] = 20

    # torch.compile the inference model in workers. The 2026-05-13 profile
    # showed model forward (linear + transformer + layer_norm) was the
    # dominant useful work (~36% OwnTime); compile should fuse small
    # kernels and remove Python dispatch overhead. None/False = eager;
    # "default" / "reduce-overhead" / "max-autotune" select the mode.
    # Note: first call after launch pays compile cost (10–60s typical);
    # subsequent calls reuse the cached graph.
    compile_inference_model: Optional[str] = None

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


@dataclass
class AdaptiveAxisConfig:
    """Adaptive-curriculum parameters shared by the team-axis and
    agent-axis updates in `OpponentPool`.

    Both axes consume the same primitive
    (`elitefurretai.rl.rl_utils.adaptive_distribution`); they differ
    only in the parameter values chosen below. See `team_axis_defaults`
    and `agent_axis_defaults` for the values that reproduce Change 7
    (team) and the pre-unification `update_curriculum` (agent).

    Fields:
        enabled: Master switch for this axis. False = bypass entirely.
        min_samples: Per-key minimum sample count before the score is
            trusted (under-warm keys fall back to the base preference).
        half_life: EWMA half-life in *recorded battles*; decay factor
            per battle is `0.5 ** (1 / half_life)`.
        prior_alpha, prior_beta: Beta pseudo-counts for win-rate
            smoothing.
        pfsp_mix: Weight on the PFSP component (peaks at wr=0.5).
        weakness_mix: Weight on the asymmetric weakness component.
        weakness_exponent: Shape of the weakness component (1.0 linear).
        target_win_rate: Win rate above which weakness is zero.
        base_blend: Mix factor with the base curriculum (0.0 = pure
            adaptive, 1.0 = pure base).
        per_key_floor: Uniform per-key minimum mass after the floor
            pass. The agent-axis caller may override on a per-key basis
            by constructing its own `floors` dict before calling
            `adaptive_distribution` directly.
    """

    enabled: bool = True
    min_samples: int = 40
    half_life: float = 100.0
    prior_alpha: float = 8.0
    prior_beta: float = 8.0
    pfsp_mix: float = 0.70
    weakness_mix: float = 0.30
    weakness_exponent: float = 1.0
    target_win_rate: float = 0.55
    base_blend: float = 0.50
    per_key_floor: float = 0.0

    @classmethod
    def team_axis_defaults(cls) -> "AdaptiveAxisConfig":
        """Defaults that reproduce Change 7 team-axis behavior."""
        return cls(
            enabled=True,
            min_samples=20,
            half_life=50.0,
            prior_alpha=8.0,
            prior_beta=8.0,
            pfsp_mix=0.0,
            weakness_mix=1.0,
            weakness_exponent=1.0,
            target_win_rate=1.0,  # (1 - wr) shape: weakness = 1 - wr
            base_blend=0.0,
            per_key_floor=0.005,
        )

    @classmethod
    def agent_axis_defaults(cls) -> "AdaptiveAxisConfig":
        """Defaults that reproduce the pre-unification update_curriculum.

        Note: the per-key anchor floors (SELF_PLAY=0.20, BC_PLAYER=0.10,
        GHOSTS=0.10) are NOT in this config — they are constructed
        per-call by the caller using opponent availability. `per_key_floor`
        stays 0 for agent-axis because the floors are heterogeneous."""
        return cls(
            enabled=True,
            min_samples=40,
            half_life=100.0,
            prior_alpha=8.0,
            prior_beta=8.0,
            pfsp_mix=0.70,
            weakness_mix=0.30,
            weakness_exponent=1.0,
            target_win_rate=0.55,
            base_blend=0.50,
            per_key_floor=0.0,
        )


@dataclass
class CurriculumConfig:
    """Opponent sampling, team pools, BC models, and ghost/exploiter directories.

    Path conventions
    ----------------
    Both ``agent_team_path`` and ``opponent_team_pool_path`` are *relative* paths
    under ``<base_team_path>/<format>/``. Each accepts three forms:

    - ``None``: no path configured. ``resolved_agent_team_paths()`` returns ``{}``;
      ``resolved_opponent_team_pool_paths()`` maps every format to ``None``.
    - ``str``: a single subdirectory name broadcast to every format in
      ``battle_formats``. Resolved to ``<base_team_path>/<fmt>/<path>`` for each
      format.
    - ``Dict[str, str]`` (or ``Dict[str, Optional[str]]`` for the opponent pool):
      per-format paths. Every active format in ``battle_formats`` MUST appear as a
      key (validated in ``__post_init__``).

    Use ``resolved_agent_team_paths()`` (plural) to obtain a ``{fmt: abs_path}``
    dict.

    ``opponent_team_pool_path`` values are subdirectory strings passed as the
    ``subdirectory=`` argument to ``TeamRepo.sample_team``; they are NOT full
    filesystem paths.
    """

    # Relative path (file or directory) under <base_team_path>/<fmt>/ for the
    # agent's team. Accepts None, a single string (broadcast), or a per-format
    # dict; see class docstring.
    agent_team_path: Optional[Union[str, Dict[str, str]]] = None
    base_team_path: str = "data/teams"
    # Probability distribution over battle formats. Must sum to 1.0.
    # Each rollout pair is pinned to one sampled format at create_agents time
    # (see WorkerOpponentFactory). The embedder is built once against
    # primary_format — vocab is gen-keyed (format_str[3]) so all entries must
    # share the same gen.
    battle_formats: Dict[str, float] = field(
        default_factory=lambda: {"gen9vgc2023regc": 1.0}
    )
    # Subdirectory under <base_team_path>/<fmt>/ sampled from for opponent
    # teams. Accepts None, a single string (broadcast), or a per-format dict
    # (values may be None to disable the subdirectory filter for that format).
    opponent_team_pool_path: Optional[Union[str, Dict[str, Optional[str]]]] = None
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
    # Exploiter and ghost model pool sizes (paths are derived from training.run_dir)
    max_exploiter_models: int = 10
    max_ghosts: int = 10
    # VGC bench external runner — launches a separate venv'd VGCBench
    # instance and registers it under OpponentPool.VGC_BENCH_BASELINE.
    # Usernames and startup wait are hardcoded module constants in
    # `engine.showdown_server_manager` (they don't vary across runs); the
    # paths here are environment-level and can change between machines.
    # Launch is triggered automatically when curriculum_weights gives
    # vgc_bench_baseline a positive weight.
    external_vgcbench_python_executable: Optional[str] = None
    external_vgcbench_team_file: str = "data/teams/gen9vgc2024regg/vgcbench.txt"
    vgc_bench_checkpoint_path: str = "data/models/vgc-bench-sb3-model.zip"
    # Adaptive curriculum: two axes, same algorithm, different defaults.
    # See `AdaptiveAxisConfig.{team,agent}_axis_defaults` and the shared
    # `rl_utils.adaptive_distribution` primitive for the algorithm itself.
    adaptive_team_axis: AdaptiveAxisConfig = field(
        default_factory=AdaptiveAxisConfig.team_axis_defaults
    )
    adaptive_agent_axis: AdaptiveAxisConfig = field(
        default_factory=AdaptiveAxisConfig.agent_axis_defaults
    )

    def __post_init__(self) -> None:
        if not self.battle_formats:
            raise ValueError("battle_formats must not be empty")
        for fmt, weight in self.battle_formats.items():
            if not isinstance(fmt, str) or not fmt:
                raise ValueError(
                    f"battle_formats key must be non-empty string, got {fmt!r}"
                )
            if weight <= 0:
                raise ValueError(
                    f"battle_formats weight for {fmt!r} must be positive, got {weight}"
                )
        total = sum(self.battle_formats.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"battle_formats weights must sum to 1.0, got {total} "
                f"({self.battle_formats})"
            )
        for fmt in self.battle_formats:
            if len(fmt) < 4:
                raise ValueError(
                    f"battle_formats key {fmt!r} is too short to identify a gen "
                    f"(need at least 4 chars, e.g. 'gen9...')"
                )
        gens = {fmt[3] for fmt in self.battle_formats}
        if len(gens) > 1:
            raise ValueError(
                f"All battle_formats must share the same gen (format[3]); got {gens}"
            )
        for attr in ("agent_team_path", "opponent_team_pool_path"):
            value = getattr(self, attr)
            if isinstance(value, dict):
                expected = set(self.battle_formats)
                actual = set(value)
                missing = expected - actual
                extra = actual - expected
                if missing or extra:
                    raise ValueError(
                        f"{attr} (dict form) keys must match battle_formats exactly; "
                        f"missing={sorted(missing)} extra={sorted(extra)}"
                    )

    @property
    def primary_format(self) -> str:
        """Highest-weight format. Used by VGCBench (single-format binding) and
        team-pool routing where a representative format string is needed.

        For embedder construction, prefer ``self.gen`` — vocab is gen-keyed,
        not format-keyed.
        """
        return max(self.battle_formats.items(), key=lambda kv: kv[1])[0]

    @property
    def gen(self) -> int:
        """Pokemon gen number derived from any battle_formats entry.

        Every entry in ``battle_formats`` shares the same gen (enforced by
        ``__post_init__``), so reading the first key is unambiguous.
        """
        return int(next(iter(self.battle_formats))[3])

    def resolved_agent_team_paths(self) -> Dict[str, str]:
        """Return absolute agent-team paths per format, or {} if unset.

        - ``agent_team_path is None`` → ``{}``
        - ``agent_team_path: str`` → ``{fmt: <base>/<fmt>/<path> for fmt in battle_formats}``
        - ``agent_team_path: Dict[str, str]`` → ``{fmt: <base>/<fmt>/<path[fmt]>}``
        """
        if self.agent_team_path is None:
            return {}
        if isinstance(self.agent_team_path, dict):
            return {
                fmt: os.path.join(self.base_team_path, fmt, self.agent_team_path[fmt])
                for fmt in self.battle_formats
            }
        return {
            fmt: os.path.join(self.base_team_path, fmt, self.agent_team_path)
            for fmt in self.battle_formats
        }

    def resolved_opponent_team_pool_paths(self) -> Dict[str, Optional[str]]:
        """Return per-format opponent-team subdirectory values for TeamRepo.

        These are subdirectory strings (NOT full paths) since
        ``TeamRepo.sample_team`` takes ``(format, subdirectory=...)``. When
        ``opponent_team_pool_path`` is ``None``, every format maps to ``None``.
        When it's a string, it broadcasts to all formats. When it's a dict,
        it is used verbatim.
        """
        if self.opponent_team_pool_path is None:
            return {fmt: None for fmt in self.battle_formats}
        if isinstance(self.opponent_team_pool_path, dict):
            return {fmt: self.opponent_team_pool_path[fmt] for fmt in self.battle_formats}
        return {fmt: self.opponent_team_pool_path for fmt in self.battle_formats}


@dataclass
class FoulplayEvalConfig:
    """Inline-during-training eval against the external foul-play-doubles bot.

    Disabled by default: the subprocess requires a separately installed
    ``../venv-foulplay`` (with ``poke-engine-doubles``, a Rust extension)
    and runs at ~750 ms / move × 8 cores, which saturates the machine
    during eval. See planning/stage2/2026-05-25-23-37-foulplay-eval-scope-confirmed.md.

    Multi-format from v1: the eval driver iterates over the active
    ``CurriculumConfig.battle_formats`` and runs one self-contained
    FoulPlay cycle per format. ``n_battles_per_format`` is per-format,
    so total wall-clock per eval pass scales linearly with the number
    of active formats.

    ``foulplay_team_pool_paths`` is the directory FoulPlay samples its
    teams from, per format. When ``None``, the eval driver falls back
    to the curriculum's ``opponent_team_pool_paths[fmt]`` so most users
    don't need to configure a separate FoulPlay-side pool.
    """

    enabled: bool = False
    eval_every_n_updates: int = 50
    n_battles_per_format: int = 100
    search_time_ms: int = 750
    parallelism: int = 4
    python_executable: Optional[str] = None
    foulplay_team_pool_paths: Optional[Dict[str, str]] = None
    model_probabilistic: bool = False


@dataclass
class OpponentEvalSpec:
    """Per-opponent eval config. Lives in EvalConfig.opponents keyed by
    canonical opponent name. Fields apply across all curriculum formats:
    n_battles is split per-format using curriculum.battle_formats weights.
    """

    target: float
    weight: float
    n_battles: int


@dataclass
class EvalConfig:
    """Inline-during-training multi-bucket eval (replaces FoulplayEvalConfig).

    Drives a generic eval pass against every opponent in `opponents` whose
    weight > 0. The pass runs at checkpoint cadence (every
    `eval_every_n_updates`), pauses training, and emits a scalar
    `eval/score` metric for W&B sweeps. See
    planning/stage2/2026-05-29-21-00-rl-wandb-sweep-eval-design.md.

    FoulPlay is one opponent in this dict. Its default weight is 0.0
    until the subprocess is stable; flipping the weight in YAML is the
    only thing needed to enable it.
    """

    enabled: bool = False
    eval_every_n_updates: int = 500
    pause_training: bool = True
    surplus_alpha: float = 1.0

    opponents: Dict[str, OpponentEvalSpec] = field(
        default_factory=lambda: {
            "simple_heuristic_baseline": OpponentEvalSpec(
                target=0.80, weight=1.0, n_battles=150
            ),
            "max_damage": OpponentEvalSpec(target=0.80, weight=1.0, n_battles=150),
            "vgc_bench": OpponentEvalSpec(target=0.60, weight=1.0, n_battles=100),
            "bc_player": OpponentEvalSpec(target=0.80, weight=1.0, n_battles=150),
            "foul_play": OpponentEvalSpec(target=0.50, weight=0.0, n_battles=40),
        }
    )

    # Opponent-specific runtime knobs (read only when the corresponding
    # opponent's weight > 0). Naming follows the `<opp>_*` convention
    # used by player_factory for cross-venv subprocess kwargs.
    vgcbench_checkpoint_path: str = "data/models/vgc-bench-sb3-model.zip"
    vgcbench_team_file: str = "data/teams/gen9vgc2024regg/vgcbench.txt"
    vgcbench_python_executable: str = "/home/cayman/Repositories/venv-vgcbench/bin/python"

    foulplay_search_time_ms: int = 750
    foulplay_python_executable: str = "/home/cayman/Repositories/venv-foulplay/bin/python"
    foulplay_team_pool_paths: Optional[Dict[str, str]] = None
    foulplay_parallelism: int = 8
    foulplay_model_probabilistic: bool = False


@dataclass
class TrainingConfig:
    """Training loop, checkpointing, and logging settings.

    Exploiter co-training settings live in `ExploiterConfig` and are
    accessed at `config.exploiter.*` rather than `config.training.*`.
    """

    # Cadence (in main-process updates) for the per-update "rendezvous":
    # save a .pt checkpoint, broadcast fresh weights + curriculum to all
    # workers, refresh ghost slots, and (when enabled) run curriculum
    # adaptation. Smaller = tighter weight sync at the cost of throughput
    # (broadcast is the dominant blocking op); larger = workers drift
    # further from the live learner between updates.
    checkpoint_interval: int = 1000
    # Embedder feature set fed into the model. Values must match constants
    # defined on `etl.embedder.Embedder`:
    #   - "simple"             : minimal hand-picked features (debug / smoke).
    #   - "raw"                : default. Token-style raw fields without
    #                            engineered transition features. Used by the
    #                            current best checkpoint (cool-bee-85).
    #   - "full"               : raw + engineered features incl. transition
    #                            features. Larger embedding, slower.
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


def _make_sub(klass: Any, d: Dict[str, Any]) -> Any:
    """Construct a dataclass from a dict, ignoring unknown keys."""
    known = {f for f in klass.__dataclass_fields__}
    return klass(**{k: v for k, v in d.items() if k in known})


def _merge_sub(factory: Any, d: Dict[str, Any]) -> Any:
    """Merge a partial-YAML dict onto an instance built by ``factory``.

    Use this for nested dataclasses whose containing field uses
    ``default_factory=<factory>`` to set non-trivial defaults that differ
    from the dataclass-level field defaults. ``_make_sub`` would silently
    drop those factory defaults when YAML supplies only a subset of fields.
    """
    base = factory()
    known = {f for f in type(base).__dataclass_fields__}
    return replace(base, **{k: v for k, v in d.items() if k in known})


def _make_eval_sub(data: dict) -> "EvalConfig":
    """Construct EvalConfig from YAML dict, merging the opponents map
    with EvalConfig defaults so partial YAML overrides work.
    """
    defaults = EvalConfig()
    data = dict(data)
    opponents_override = data.pop("opponents", {})
    merged_opponents = dict(defaults.opponents)
    for name, opp_data in opponents_override.items():
        existing = merged_opponents.get(
            name, OpponentEvalSpec(target=0.5, weight=0.0, n_battles=0)
        )
        merged_opponents[name] = OpponentEvalSpec(
            target=opp_data.get("target", existing.target),
            weight=opp_data.get("weight", existing.weight),
            n_battles=opp_data.get("n_battles", existing.n_battles),
        )
    known = {f for f in EvalConfig.__dataclass_fields__}
    return EvalConfig(
        opponents=merged_opponents, **{k: v for k, v in data.items() if k in known}
    )


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
    eval: EvalConfig = field(default_factory=EvalConfig)
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
    def num_showdown_servers(self) -> int:
        return self.hardware.num_servers

    # ── Annealing helpers ──────────────────────────────────────────────────────

    def temperature_at_step(self, step: int) -> float:
        """Compute linearly annealed temperature at a given training step."""
        cfg = self.exploration
        progress = min(step / max(cfg.exploration_anneal_steps, 1), 1.0)
        return cfg.temperature_start + progress * (
            cfg.temperature_end - cfg.temperature_start
        )

    def ent_coef_at_step(self, step: int) -> float:
        """Compute linearly annealed entropy coefficient at a given training step."""
        cfg = self.exploration
        progress = min(step / max(cfg.exploration_anneal_steps, 1), 1.0)
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
    def from_yaml(cls, filepath: str) -> "RNaDConfig":
        return cls.load(filepath)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RNaDConfig":
        """Create a RNaDConfig from a dict (flat or nested). Extra keys are ignored."""

        # CurriculumConfig has nested AdaptiveAxisConfig sub-dataclasses
        # whose defaults come from the team_axis_defaults() /
        # agent_axis_defaults() factories — these differ from the
        # dataclass-level field defaults. Merge partial YAML onto the
        # right factory so unspecified fields keep the factory's values.
        curriculum_data = dict(data.get("curriculum", {}))
        if "adaptive_team_axis" in curriculum_data:
            curriculum_data["adaptive_team_axis"] = _merge_sub(
                AdaptiveAxisConfig.team_axis_defaults,
                curriculum_data["adaptive_team_axis"],
            )
        if "adaptive_agent_axis" in curriculum_data:
            curriculum_data["adaptive_agent_axis"] = _merge_sub(
                AdaptiveAxisConfig.agent_axis_defaults,
                curriculum_data["adaptive_agent_axis"],
            )

        return cls(
            algorithm=_make_sub(AlgorithmConfig, data.get("algorithm", {})),
            architecture=_make_sub(ArchitectureConfig, data.get("architecture", {})),
            curriculum=_make_sub(CurriculumConfig, curriculum_data),
            exploiter=_make_sub(ExploiterConfig, data.get("exploiter", {})),
            exploration=_make_sub(ExplorationConfig, data.get("exploration", {})),
            eval=_make_eval_sub(data.get("eval", {})),
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

        for fmt, pool_subdir in cur.resolved_opponent_team_pool_paths().items():
            if pool_subdir:
                full_pool = os.path.join(cur.base_team_path, fmt, pool_subdir)
                assert os.path.exists(full_pool), (
                    f"Opponent team pool path not found: {full_pool}"
                )
        for fmt, ap in cur.resolved_agent_team_paths().items():
            assert os.path.exists(ap), f"Agent team path not found ({fmt}): {ap}"
            if os.path.isdir(ap):
                team_files = [f for f in os.listdir(ap) if f.endswith(".txt")]
                assert len(team_files) > 0, (
                    f"No .txt team files in agent_team_path ({fmt}): {ap}"
                )
        if trn.resume_from:
            assert os.path.exists(trn.resume_from), (
                f"Resume checkpoint not found: {trn.resume_from}"
            )
        if trn.initialize_path:
            assert os.path.exists(trn.initialize_path), (
                f"Initialize checkpoint not found: {trn.initialize_path}"
            )

        # External vgc-bench runners launch automatically when the
        # curriculum gives vgc_bench_baseline positive weight.
        if cur.curriculum_weights.get("vgc_bench_baseline", 0.0) > 0:
            assert cur.external_vgcbench_python_executable, (
                "external_vgcbench_python_executable must be set when "
                "curriculum_weights['vgc_bench_baseline'] > 0"
            )
            assert os.path.exists(cur.external_vgcbench_python_executable), (
                f"external_vgcbench_python_executable not found: {cur.external_vgcbench_python_executable}"
            )
            assert os.path.exists(cur.external_vgcbench_team_file), (
                f"external_vgcbench_team_file not found: {cur.external_vgcbench_team_file}"
            )

        ev = self.eval
        if ev.enabled:
            fp_spec = ev.opponents.get("foul_play")
            if fp_spec is not None and fp_spec.weight > 0:
                if not ev.foulplay_python_executable:
                    raise ValueError(
                        "eval.foulplay_python_executable must be set when "
                        "eval.opponents['foul_play'].weight > 0"
                    )
                if not os.path.exists(ev.foulplay_python_executable):
                    raise ValueError(
                        f"eval.foulplay_python_executable not found: "
                        f"{ev.foulplay_python_executable}"
                    )
            vgc_spec = ev.opponents.get("vgc_bench")
            if vgc_spec is not None and vgc_spec.weight > 0:
                if not os.path.exists(ev.vgcbench_python_executable):
                    raise ValueError(
                        f"eval.vgcbench_python_executable not found: "
                        f"{ev.vgcbench_python_executable}"
                    )


def get_default_config() -> RNaDConfig:
    return RNaDConfig()
