"""
Enhanced RNaD training with:
- Config-driven training
- Resume from checkpoint
- Integrated exploiter training
- Team pool randomization
- Comprehensive wandb tracking
"""

import argparse
import asyncio
import copy
import os
import queue
import subprocess
import sys
import threading
from datetime import datetime
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
from poke_env.ps_client import AccountConfiguration, ServerConfiguration

import wandb
from elitefurretai.etl.embedder import Embedder
from elitefurretai.etl.encoder import MDBO
from elitefurretai.etl.team_repo import TeamRepo
from elitefurretai.rl.agent import RNaDAgent
from elitefurretai.rl.config import RNaDConfig
from elitefurretai.rl.learner import RNaDLearner
from elitefurretai.rl.opponent_pool import ExploiterRegistry, OpponentPool
from elitefurretai.rl.portfolio_learner import PortfolioRNaDLearner
from elitefurretai.rl.worker import BatchInferencePlayer
from elitefurretai.supervised.model_archs import FlexibleThreeHeadedModel


def load_model(filepath, device):
    """Load model from checkpoint or create new one."""
    embedder = Embedder(
        format="gen9vgc2023regulationc", feature_set=Embedder.FULL, omniscient=False
    )
    input_size = embedder.embedding_size

    if not os.path.exists(filepath):
        print(f"Checkpoint not found at {filepath}. Initializing random model.")
        config: Dict[str, Any] = {
            "dropout": 0.1,
            "teampreview_head_dropout": 0.3,
            "gated_residuals": False,
            "use_grouped_encoder": True,
            "grouped_encoder_hidden_dim": 512,
            "grouped_encoder_aggregated_dim": 4096,
            "pokemon_attention_heads": 16,
            "early_layers": [4096, 2048, 2048, 1024],
            "early_attention_heads": 16,
            "lstm_layers": 4,
            "lstm_hidden_size": 512,
            "late_layers": [2048, 2048, 1024, 1024],
            "late_attention_heads": 32,
            "teampreview_head_layers": [512, 256],
            "teampreview_attention_heads": 8,
            "turn_head_layers": [2048, 1024, 1024, 1024],
        }
        state_dict = None
    else:
        checkpoint = torch.load(filepath, map_location=device)
        config = checkpoint["config"]
        state_dict = checkpoint["model_state_dict"]

    model = FlexibleThreeHeadedModel(
        input_size=input_size,
        early_layers=config["early_layers"],
        late_layers=config["late_layers"],
        lstm_layers=config.get("lstm_layers", 2),
        lstm_hidden_size=config.get("lstm_hidden_size", 512),
        dropout=config.get("dropout", 0.1),
        gated_residuals=config.get("gated_residuals", False),
        early_attention_heads=config.get("early_attention_heads", 8),
        late_attention_heads=config.get("late_attention_heads", 8),
        use_grouped_encoder=config.get("use_grouped_encoder", False),
        group_sizes=(
            embedder.group_embedding_sizes
            if config.get("use_grouped_encoder", False)
            else None
        ),
        grouped_encoder_hidden_dim=config.get("grouped_encoder_hidden_dim", 128),
        grouped_encoder_aggregated_dim=config.get("grouped_encoder_aggregated_dim", 1024),
        pokemon_attention_heads=config.get("pokemon_attention_heads", 2),
        teampreview_head_layers=config.get("teampreview_head_layers", []),
        teampreview_head_dropout=config.get("teampreview_head_dropout", 0.1),
        teampreview_attention_heads=config.get("teampreview_attention_heads", 4),
        turn_head_layers=config.get("turn_head_layers", []),
        num_actions=MDBO.action_space(),
        num_teampreview_actions=MDBO.teampreview_space(),
        max_seq_len=17,
    ).to(device)

    if state_dict:
        model.load_state_dict(state_dict)

    return model


def worker_loop(
    opponent_pool: OpponentPool,
    traj_queue: queue.Queue,
    num_players: int,
    worker_id: int,
    port_offset: int,
    team_repo: TeamRepo,
    battle_format: str,
    team_subdirectory: Optional[str] = None,
    batch_size: int = 16,
):
    """
    Runs an asyncio loop for a set of players with dynamic opponent selection.

    Args:
        opponent_pool: OpponentPool for sampling opponents
        traj_queue: Queue for pushing completed trajectories
        num_players: Number of concurrent players in this worker
        worker_id: Unique worker ID
        port_offset: Port offset for showdown server (0, 1, 2, 3 for ports 8000-8003)
        team_repo: TeamRepo for random team sampling
        battle_format: Game format string for battle initialization and team sampling (e.g., gen9vgc2023regulationc)
        team_subdirectory: Optional subdirectory within format to sample teams from (e.g., "easy", "rental_teams")
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    server_port = 8000 + port_offset
    server_config = ServerConfiguration(
        f"ws://localhost:{server_port}/showdown/websocket",
        None,  # type: ignore[arg-type]
    )

    players = []
    for i in range(num_players):
        # Sample random team if repo provided
        team = team_repo.sample_team(battle_format, subdirectory=team_subdirectory)

        player = BatchInferencePlayer(
            model=opponent_pool.main_model,
            device=opponent_pool.device,
            batch_size=batch_size,
            account_configuration=AccountConfiguration(f"Worker_{worker_id}_{i}", None),
            server_configuration=server_config,
            trajectory_queue=traj_queue,
            battle_format=battle_format,
            team=team,
        )
        players.append(player)

    # Start inference loops
    for p in players:
        loop.create_task(p.start_inference_loop())

    async def run_battles():
        while True:
            # Sample opponents for each player (with random teams if available)
            opponents = []
            for i in range(num_players):
                opp_config = AccountConfiguration(f"Opponent_{worker_id}_{i}", None)
                opponent_team = team_repo.sample_team(
                    battle_format, subdirectory=team_subdirectory
                )
                opponent = opponent_pool.sample_opponent(
                    opp_config, server_config, team=opponent_team
                )
                opponents.append(opponent)

            # CRITICAL: Start inference loops for BatchInferencePlayer opponents
            # Without this, opponents using our neural network will hang forever
            # waiting for inference results that never come (queue never processed)
            for opponent in opponents:
                if isinstance(opponent, BatchInferencePlayer):
                    await opponent.start_inference_loop()

            # Run a batch of battles
            tasks = []
            for player, opponent in zip(players, opponents):
                tasks.append(player.battle_against(opponent, n_battles=20))
            await asyncio.gather(*tasks)

            # Clean up: Cancel inference loops for opponents to prevent memory leak
            # Opponents are recreated each battle batch, so their loops must be stopped
            for opponent in opponents:
                if isinstance(opponent, BatchInferencePlayer) and opponent._inference_task:
                    opponent._inference_task.cancel()
                    try:
                        await opponent._inference_task
                    except asyncio.CancelledError:
                        pass  # Expected when cancelling

    try:
        loop.run_until_complete(run_battles())
    except Exception as e:
        print(f"Worker {worker_id} crashed: {e}")


def collate_trajectories(trajectories, device):
    """Collate list of trajectories into batched tensors with padding."""
    batch_size = len(trajectories)
    max_len = max(len(t) for t in trajectories)

    # Get dim from first state
    dim = len(trajectories[0][0]["state"])

    states = torch.zeros(batch_size, max_len, dim)
    actions = torch.zeros(batch_size, max_len, dtype=torch.long)
    rewards = torch.zeros(batch_size, max_len)
    log_probs = torch.zeros(batch_size, max_len)
    values = torch.zeros(batch_size, max_len)
    is_tp = torch.zeros(batch_size, max_len, dtype=torch.bool)
    padding_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

    processed_advantages = []
    processed_returns = []

    for i, traj in enumerate(trajectories):
        seq_len = len(traj)

        # Extract fields
        traj_states = torch.tensor(np.array([step["state"] for step in traj]))
        traj_actions = torch.tensor([step["action"] for step in traj])
        traj_rewards = torch.tensor([step["reward"] for step in traj])
        traj_log_probs = torch.tensor([step["log_prob"] for step in traj])
        traj_values = torch.tensor([step["value"] for step in traj])
        traj_is_tp = torch.tensor([step["is_teampreview"] for step in traj])

        states[i, :seq_len] = traj_states
        actions[i, :seq_len] = traj_actions
        rewards[i, :seq_len] = traj_rewards
        log_probs[i, :seq_len] = traj_log_probs
        values[i, :seq_len] = traj_values
        is_tp[i, :seq_len] = traj_is_tp
        padding_mask[i, :seq_len] = 1

        # Compute GAE
        advs = torch.zeros(seq_len)
        rets = torch.zeros(seq_len)
        last_gae_lam = torch.tensor(0.0)
        gamma = 0.99
        lam = 0.95

        next_val = torch.tensor(0.0)

        for t in reversed(range(seq_len)):
            delta = traj_rewards[t] + gamma * next_val - traj_values[t]
            last_gae_lam = delta + gamma * lam * last_gae_lam
            advs[t] = last_gae_lam
            rets[t] = traj_values[t] + last_gae_lam
            next_val = traj_values[t]

        processed_advantages.append(advs)
        processed_returns.append(rets)

    # Pad advantages and returns
    advantages = torch.zeros(batch_size, max_len)
    returns = torch.zeros(batch_size, max_len)

    for i in range(batch_size):
        seq_len = len(processed_advantages[i])
        advantages[i, :seq_len] = processed_advantages[i]
        returns[i, :seq_len] = processed_returns[i]

    return {
        "states": states.to(device),
        "actions": actions.to(device),
        "rewards": rewards.to(device),
        "log_probs": log_probs.to(device),
        "values": values.to(device),
        "is_teampreview": is_tp.to(device),
        "advantages": advantages.to(device),
        "returns": returns.to(device),
        "padding_mask": padding_mask.to(device),
    }


def save_checkpoint(
    model: RNaDAgent,
    optimizer,
    step: int,
    config: RNaDConfig,
    curriculum: Dict,
    save_dir: str = "data/models",
):
    """Save complete training state for resuming."""
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
    filepath: str, model: RNaDAgent, optimizer, config: RNaDConfig, device: str
):
    """Load training state from checkpoint."""
    if not os.path.exists(filepath):
        print(f"Checkpoint not found: {filepath}")
        return 0, {}

    print(f"Resuming from checkpoint: {filepath}")
    checkpoint = torch.load(filepath, map_location=device)

    model.model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    step = checkpoint["step"]
    curriculum = checkpoint.get("curriculum", {})

    print(f"Resumed from step {step}")
    return step, curriculum


def train_exploiter_subprocess(victim_checkpoint: str, config: RNaDConfig):
    """Launch exploiter training as subprocess."""
    print(f"\n{'=' * 60}")
    print("EXPLOITER TRAINING TRIGGERED")
    print(f"Victim: {victim_checkpoint}")
    print(f"{'=' * 60}\n")

    # Launch train_exploiter.py as subprocess
    cmd = [
        sys.executable,
        "src/elitefurretai/rl/train_exploiter.py",
        "--victim",
        victim_checkpoint,
        "--steps",
        str(config.exploiter_updates),
        "--eval-games",
        str(config.exploiter_eval_games),
        "--threshold",
        str(config.exploiter_min_win_rate),
        "--output-dir",
        (
            str(config.exploiter_output_dir)
            if config.exploiter_output_dir
            else "data/models/exploiters"
        ),
        "--team-pool",
        config.exploiter_team_pool_path if config.exploiter_team_pool_path else "",
        "--learning-rate",
        str(config.exploiter_lr),
        "--ent-coef",
        str(config.exploiter_ent_coef),
    ]

    subprocess.run(cmd, check=True)
    print("\nExploiter training complete!")


def main():
    parser = argparse.ArgumentParser(description="RNaD RL Training for Pokemon VGC")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument(
        "--resume", action="store_true", help="Resume from last checkpoint"
    )
    parser.add_argument(
        "--initialize",
        type=str,
        default=None,
        help="Initialize model weights from checkpoint (starts fresh training)",
    )
    args = parser.parse_args()

    # Load or create config
    if args.config:
        config = RNaDConfig.load(args.config)
        print(f"Loaded config from {args.config}")
    else:
        config = RNaDConfig()
        print("Using default config")

    # Save config for this run
    os.makedirs(config.save_dir, exist_ok=True)
    config_path = os.path.join(
        config.save_dir, f"config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
    )
    config.save(config_path)
    print(f"Config saved to {config_path}")

    device = config.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize wandb
    if config.use_wandb:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name
            or f"rnad_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=config.to_dict(),
            tags=config.wandb_tags,
        )
        # Save config as artifact
        artifact = wandb.Artifact(name="config", type="config")
        artifact.add_file(config_path)
        wandb.log_artifact(artifact)

    # Load team repo (required)
    # TeamRepo expects base_team_path to contain format folders (e.g., data/teams/gen9vgc2023regc/)
    team_repo = TeamRepo(config.base_team_path)

    # Determine team sampling parameters
    if config.team_pool_path:
        # Sample only from specific subdirectory within the format
        team_subdirectory = config.team_pool_path
        print(
            f"Loaded team repository from {config.base_team_path} (sampling from {config.battle_format}/{team_subdirectory})"
        )
    else:
        # Sample from all teams in the format
        team_subdirectory = None
        print(
            f"Loaded team repository from {config.base_team_path} (sampling from all {config.battle_format} teams)"
        )

    # Determine model path for initialization
    # Priority: config.initialize_path > args.initialize
    model_init_path = config.initialize_path or args.initialize
    print(f"Loading main model from {model_init_path}...")
    base_model = load_model(model_init_path, device)
    agent = RNaDAgent(base_model)

    # Create learner (portfolio or standard)
    learner: Union[RNaDLearner, PortfolioRNaDLearner]
    if config.use_portfolio_regularization:
        print(f"Using Portfolio RNaD with {config.max_portfolio_size} references")
        # Create initial portfolio with current model
        ref_base_model = copy.deepcopy(base_model)
        ref_agent = RNaDAgent(ref_base_model)
        ref_models = [ref_agent]  # Start with one reference

        learner = PortfolioRNaDLearner(
            agent,
            ref_models,
            lr=config.lr,
            rnad_alpha=config.rnad_alpha,
            device=device,
            gamma=config.gamma,
            clip_range=config.clip_range,
            ent_coef=config.ent_coef,
            vf_coef=config.vf_coef,
            use_mixed_precision=config.use_mixed_precision,
            max_portfolio_size=config.max_portfolio_size,
            portfolio_update_strategy=config.portfolio_update_strategy,
        )
    else:
        print("Using standard RNaD with single reference")
        # Create ref model (deep copy)
        ref_base_model = copy.deepcopy(base_model)
        ref_agent = RNaDAgent(ref_base_model)

        learner = RNaDLearner(
            agent,
            ref_agent,
            lr=config.lr,
            rnad_alpha=config.rnad_alpha,
            device=device,
            gamma=config.gamma,
            clip_range=config.clip_range,
            ent_coef=config.ent_coef,
            vf_coef=config.vf_coef,
            use_mixed_precision=config.use_mixed_precision,
            gradient_clip=config.gradient_clip,
        )

    # Resume from checkpoint if requested (restore optimizer + step state)
    # Note: --checkpoint is for model initialization only, not resuming training state
    start_step = 0
    curriculum = {}
    if config.resume_from or args.resume:
        # Priority: CLI flag --resume > config.resume_from
        if args.resume:
            checkpoint_path = os.path.join(
                config.save_dir,
                sorted(
                    [
                        f
                        for f in os.listdir(config.save_dir)
                        if f.startswith("main_model_step_")
                    ]
                )[-1],
            )
        else:
            assert isinstance(config.resume_from, str)
            checkpoint_path = config.resume_from
        assert isinstance(checkpoint_path, str)
        start_step, curriculum = load_checkpoint(
            checkpoint_path, agent, learner.optimizer, config, device
        )

    # Create opponent pool
    print("Initializing opponent pool...")
    opponent_pool = OpponentPool(
        main_model=agent,
        device=device,
        battle_format=config.battle_format,
        bc_teampreview_path=(
            config.bc_teampreview_path
            if os.path.exists(config.bc_teampreview_path)
            else None
        ),
        bc_action_path=(
            config.bc_action_path if os.path.exists(config.bc_action_path) else None
        ),
        bc_win_path=config.bc_win_path if os.path.exists(config.bc_win_path) else None,
        exploiter_registry_path=config.exploiter_registry_path,
        past_models_dir=config.past_models_dir,
        max_past_models=config.max_past_models,
    )

    # Restore curriculum if resuming
    if curriculum:
        opponent_pool.curriculum = curriculum
        print(f"Restored curriculum: {curriculum}")

    # ==================== START WORKER THREADS ====================
    # Create shared queue for workers to push completed battle trajectories
    # Workers run battles asynchronously and add trajectory data here for training
    traj_queue: queue.Queue = queue.Queue()

    # Workers generate training data by playing battles against opponent pool
    # Each worker runs multiple players concurrently in an asyncio loop
    threads = []
    for i in range(config.num_workers):
        # Distribute workers across multiple Showdown servers (ports 8000-8003)
        # This prevents server bottlenecks when running many concurrent battles
        port_offset = i % 4  # Cycle through 4 servers on ports 8000-8003

        # Create daemon thread for this worker (exits when main thread exits)
        t = threading.Thread(
            target=worker_loop,
            args=(
                opponent_pool,
                traj_queue,
                config.players_per_worker,
                i,
                port_offset,
                team_repo,
                config.battle_format,
                team_subdirectory,
                config.batch_size,
            ),
            daemon=True,
        )
        t.start()
        threads.append(t)

    print(
        f"Started {config.num_workers} workers with {config.players_per_worker} players each."
    )
    if config.num_showdown_servers > 1:
        print(
            f"Workers distributed across ports 8000-{8000 + config.num_showdown_servers - 1}"
        )
    else:
        print("All workers using single showdown server on port 8000")

    # ==================== MAIN TRAINING LOOP ====================
    # Collect trajectories from workers, batch them, and perform policy updates
    trajectories = []  # Buffer to accumulate trajectories before training
    updates = start_step  # Current training step (may be >0 if resumed from checkpoint)
    last_exploiter_check = start_step  # Track when we last checked for exploiter training

    try:
        while updates < config.max_updates:
            # ===== COLLECT TRAJECTORIES FROM WORKERS =====
            # Workers push completed battle trajectories to the queue asynchronously
            # We collect them here until we have enough for a training batch
            try:
                traj = traj_queue.get(
                    timeout=1.0
                )  # Wait up to 1 second for new trajectory
                trajectories.append(traj)
            except queue.Empty:
                # No new trajectories yet, continue waiting
                continue

            # ===== PERFORM TRAINING UPDATE =====
            # Once we've collected enough trajectories, train the model
            if len(trajectories) >= config.train_batch_size:
                # Collate trajectories into padded batches (handles variable length sequences)
                batch = collate_trajectories(trajectories, device)

                # Execute one RNaD policy update (PPO + KL regularization vs reference)
                metrics = learner.update(batch)
                updates += 1

                # ===== LOG TRAINING METRICS =====
                if config.use_wandb and updates % config.log_interval == 0:
                    # Core training metrics from the RNaD update
                    log_dict = {
                        "loss": metrics["loss"],
                        "policy_loss": metrics["policy_loss"],
                        "value_loss": metrics["value_loss"],
                        "entropy": metrics["entropy"],
                        "rnad_loss": metrics["rnad_loss"],
                        "update_step": updates,
                    }

                    # Portfolio-specific metrics
                    if config.use_portfolio_regularization and isinstance(
                        learner, PortfolioRNaDLearner
                    ):
                        log_dict["portfolio_size"] = metrics.get("portfolio_size", 1)
                        portfolio_stats = learner.get_portfolio_stats()
                        # Track how often each reference model is selected for regularization
                        for i, count in enumerate(portfolio_stats["selection_counts"]):
                            log_dict[f"portfolio/ref_{i}_selections"] = count
                        # Track average KL divergence from each reference model
                        for i, avg_kl in enumerate(portfolio_stats["avg_kl_per_ref"]):
                            log_dict[f"portfolio/ref_{i}_avg_kl"] = avg_kl

                    # Log win rates against different opponent types
                    # Helps track progress against BC models, past versions, exploiters, etc.
                    win_rate_stats = opponent_pool.get_win_rate_stats()
                    for opp_type, win_rate in win_rate_stats.items():
                        log_dict[f"win_rate/{opp_type}"] = win_rate

                    # Log curriculum weights (how often we sample each opponent type)
                    for opp_type, weight in opponent_pool.curriculum.items():
                        log_dict[f"curriculum/{opp_type}"] = weight

                    wandb.log(log_dict)

                # Print progress to console periodically
                if updates % 10 == 0:
                    print(
                        f"Update {updates}: Loss={metrics['loss']:.4f}, "
                        f"Policy={metrics['policy_loss']:.4f}, "
                        f"Value={metrics['value_loss']:.4f}"
                    )

                # ===== UPDATE REFERENCE MODEL =====
                # Reference model is used for KL regularization in RNaD
                # Periodically sync it with current policy to prevent policy collapse
                if updates % config.ref_update_interval == 0:
                    print(f"[Update {updates}] Updating reference model...")
                    if config.use_portfolio_regularization and isinstance(
                        learner, PortfolioRNaDLearner
                    ):
                        learner.update_main_reference()  # Update primary reference in portfolio
                    else:
                        assert isinstance(learner, RNaDLearner)
                        learner.update_ref_model()  # Update single reference model

                # ===== ADD NEW REFERENCE TO PORTFOLIO =====
                # Portfolio regularization: maintain multiple reference snapshots
                # Helps prevent cyclic behavior by regularizing against diverse past policies
                if config.use_portfolio_regularization and (
                    updates % config.portfolio_add_interval == 0
                ):
                    print(f"[Update {updates}] Adding new reference to portfolio...")
                    new_ref = RNaDAgent(
                        copy.deepcopy(agent.model)
                    )  # Snapshot current policy
                    if isinstance(learner, PortfolioRNaDLearner):
                        learner.add_reference_model(new_ref)
                        portfolio_stats = learner.get_portfolio_stats()
                        print(f"  Portfolio size: {portfolio_stats['portfolio_size']}")
                        print(f"  Selection counts: {portfolio_stats['selection_counts']}")

                # ===== SAVE CHECKPOINT =====
                # Periodically save model, optimizer state, and training progress
                # Allows resuming training if interrupted
                if updates % config.checkpoint_interval == 0:
                    print(f"[Update {updates}] Saving checkpoint...")
                    checkpoint_path = save_checkpoint(
                        agent,
                        learner.optimizer,
                        updates,
                        config,
                        opponent_pool.curriculum,
                        config.save_dir,
                    )

                    # Add checkpoint to past models pool for opponent diversity
                    # Workers can sample these past versions as opponents
                    opponent_pool.add_past_model(updates, checkpoint_path)

                    if config.use_wandb:
                        # Upload model to W&B for versioning and easy access
                        artifact = wandb.Artifact(f"model_step_{updates}", type="model")
                        artifact.add_file(checkpoint_path)
                        wandb.log_artifact(artifact)

                # ===== UPDATE CURRICULUM (ADAPTIVE OPPONENT SAMPLING) =====
                # Adjust weights for sampling different opponent types based on win rates
                # E.g., if we're beating BC players too easily, reduce their sampling weight
                if updates % config.curriculum_update_interval == 0:
                    opponent_pool.update_curriculum(adaptive=True)
                    print(
                        f"[Update {updates}] Curriculum updated: {opponent_pool.curriculum}"
                    )

                # ===== TRAIN EXPLOITER (FIND WEAKNESSES) =====
                # Periodically train a new exploiter agent to beat current policy
                # Exploiters are added to opponent pool to patch discovered weaknesses
                if config.train_exploiters and (
                    updates - last_exploiter_check >= config.exploiter_interval
                ):
                    last_exploiter_check = updates

                    # Save current policy as "victim" for exploiter to train against
                    victim_path = os.path.join(
                        config.save_dir, f"victim_step_{updates}.pt"
                    )
                    save_checkpoint(
                        agent,
                        learner.optimizer,
                        updates,
                        config,
                        opponent_pool.curriculum,
                        config.save_dir,
                    )

                    # Launch exploiter training in subprocess (runs independently)
                    try:
                        train_exploiter_subprocess(victim_path, config)

                        # Reload exploiter registry to include newly trained exploiter
                        opponent_pool.exploiter_registry = ExploiterRegistry(
                            config.exploiter_registry_path
                        )
                        print(
                            f"Reloaded exploiter registry: {len(opponent_pool.exploiter_registry.exploiters)} exploiters"
                        )
                    except Exception as e:
                        print(f"Exploiter training failed: {e}")

                # Clear trajectory buffer after successful update
                trajectories = []

    except KeyboardInterrupt:
        # ===== GRACEFUL SHUTDOWN =====
        # User interrupted training (Ctrl+C) - save progress before exiting
        print("\nStopping training...")

        # Save final checkpoint so training can be resumed later
        final_path = save_checkpoint(
            agent,
            learner.optimizer,
            updates,
            config,
            opponent_pool.curriculum,
            config.save_dir,
        )
        print(f"Final model saved to {final_path}")

        # Close W&B run properly to ensure all logs are synced
        if config.use_wandb:
            wandb.finish()


if __name__ == "__main__":
    # Set sharing strategy for WSL compatibility
    torch.multiprocessing.set_sharing_strategy("file_system")
    main()
