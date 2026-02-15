#!/usr/bin/env python3
"""
Evaluate two RL checkpoints by playing battles between them.

Usage:
    python -m elitefurretai.rl.analyze.evaluate_agents \\
        data/models/rl/main_model_step_100.pt \\
        data/models/rl/main_model_step_200.pt \\
        --num-battles 100

This script loads two model checkpoints from rl/train.py and plays them
against each other to measure relative strength.

By default, the script will:
1. Start a Pokemon Showdown server on the specified port
2. Run battles between the two agents
3. Shut down the server when done (or on interrupt)

Use --no-server if you have a server already running.
"""

import argparse
import asyncio
import os
import signal
import subprocess
import sys
import time
from typing import Any, Dict, Optional

import torch
from poke_env import AccountConfiguration, ServerConfiguration

from elitefurretai.etl.embedder import Embedder
from elitefurretai.etl.encoder import MDBO
from elitefurretai.etl.team_repo import TeamRepo
from elitefurretai.rl.players import BatchInferencePlayer, RNaDAgent
from elitefurretai.supervised.model_archs import FlexibleThreeHeadedModel

# Global for cleanup on interrupt
_server_process: Optional[subprocess.Popen] = None


def launch_showdown_server(port: int, showdown_dir: str = "../pokemon-showdown") -> subprocess.Popen:
    """
    Launch a Pokemon Showdown server.

    Args:
        port: Port for the server to listen on
        showdown_dir: Path to pokemon-showdown directory (relative to cwd or absolute)

    Returns:
        The subprocess.Popen object for the server
    """
    global _server_process

    # Resolve showdown directory
    if not os.path.isabs(showdown_dir):
        showdown_dir = os.path.abspath(showdown_dir)

    if not os.path.exists(showdown_dir):
        raise FileNotFoundError(
            f"Pokemon Showdown directory not found: {showdown_dir}\n"
            f"Please clone it from https://github.com/smogon/pokemon-showdown"
        )

    print(f"Starting Pokemon Showdown server on port {port}...")
    process = subprocess.Popen(
        ["node", "pokemon-showdown", "start", "--no-security", "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=showdown_dir,
        preexec_fn=os.setsid,  # Create new process group for cleanup
    )

    # Give server time to start
    time.sleep(3)

    # Check if server started successfully
    if process.poll() is not None:
        raise RuntimeError(
            f"Pokemon Showdown server failed to start (exit code: {process.returncode})"
        )

    _server_process = process
    print(f"Server started (PID: {process.pid})")
    return process


def shutdown_showdown_server(process: Optional[subprocess.Popen] = None) -> None:
    """
    Shutdown a Pokemon Showdown server.

    Args:
        process: The server process to kill. If None, uses the global _server_process.
    """
    global _server_process

    proc = process or _server_process
    if proc is None:
        return

    if proc.poll() is not None:
        # Already terminated
        _server_process = None
        return

    print("Shutting down Pokemon Showdown server...")
    try:
        # Kill entire process group
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait(timeout=2)
    except ProcessLookupError:
        pass  # Already dead
    except Exception as e:
        print(f"Warning: Error during server shutdown: {e}")

    _server_process = None
    print("Server stopped.")


def load_checkpoint(filepath: str, device: str = "cuda") -> Dict[str, Any]:
    """Load a checkpoint and extract config and state dict."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")

    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    return checkpoint


def build_model_from_checkpoint(
    checkpoint: Dict[str, Any], device: str = "cuda"
) -> RNaDAgent:
    """Build a model from checkpoint data."""
    config = checkpoint.get("config", checkpoint.get("model_config", {}))
    state_dict = checkpoint["model_state_dict"]

    # Get embedder feature set from config
    feature_set = config.get("embedder_feature_set", "full")
    embedder = Embedder(
        format="gen9vgc2023regc",
        feature_set=feature_set,
        omniscient=False,
    )

    model = FlexibleThreeHeadedModel(
        input_size=embedder.embedding_size,
        early_layers=config.get("early_layers", [4096, 2048, 2048, 1024]),
        late_layers=config.get("late_layers", [2048, 2048, 1024, 1024]),
        lstm_layers=config.get("lstm_layers", 4),
        lstm_hidden_size=config.get("lstm_hidden_size", 512),
        dropout=config.get("dropout", 0.1),
        gated_residuals=config.get("gated_residuals", False),
        early_attention_heads=config.get("early_attention_heads", 16),
        late_attention_heads=config.get("late_attention_heads", 32),
        use_grouped_encoder=config.get("use_grouped_encoder", True),
        group_sizes=(
            embedder.group_embedding_sizes
            if config.get("use_grouped_encoder", True)
            else None
        ),
        grouped_encoder_hidden_dim=config.get("grouped_encoder_hidden_dim", 512),
        grouped_encoder_aggregated_dim=config.get("grouped_encoder_aggregated_dim", 4096),
        pokemon_attention_heads=config.get("pokemon_attention_heads", 16),
        teampreview_head_layers=config.get("teampreview_head_layers", [512, 256]),
        teampreview_head_dropout=config.get("teampreview_head_dropout", 0.3),
        teampreview_attention_heads=config.get("teampreview_attention_heads", 8),
        turn_head_layers=config.get("turn_head_layers", [2048, 1024, 1024, 1024]),
        num_actions=MDBO.action_space(),
        num_teampreview_actions=MDBO.teampreview_space(),
        max_seq_len=config.get("max_seq_len", 40),
    ).to(device)

    model.load_state_dict(state_dict)
    model.eval()

    return RNaDAgent(model)


async def run_evaluation(
    checkpoint1_path: str,
    checkpoint2_path: str,
    num_battles: int = 100,
    device: str = "cuda",
    server_port: int = 8000,
    battle_format: str = "gen9vgc2023regc",
    team_path: Optional[str] = None,
    team_subdirectory: Optional[str] = None,
    probabilistic: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run evaluation battles between two checkpoints.

    Args:
        checkpoint1_path: Path to first model checkpoint
        checkpoint2_path: Path to second model checkpoint
        num_battles: Number of battles to run
        device: Device for model inference
        server_port: Pokemon Showdown server port
        battle_format: Battle format string
        team_path: Path to team repository (e.g., "data/teams")
        team_subdirectory: Subdirectory within format for teams
        probabilistic: Whether to sample actions (True) or use argmax (False)
        verbose: Print progress updates

    Returns:
        Dictionary with evaluation results
    """
    if verbose:
        print(f"\n{'=' * 60}")
        print("AGENT EVALUATION")
        print(f"{'=' * 60}")
        print(f"Agent 1: {checkpoint1_path}")
        print(f"Agent 2: {checkpoint2_path}")
        print(f"Battles: {num_battles}")
        print(f"Device: {device}")
        print(f"Server: localhost:{server_port}")
        print(f"{'=' * 60}\n")

    # Load checkpoints
    if verbose:
        print("Loading checkpoints...")
    checkpoint1 = load_checkpoint(checkpoint1_path, device)
    checkpoint2 = load_checkpoint(checkpoint2_path, device)

    # Extract step numbers for labeling
    step1 = checkpoint1.get("step", "unknown")
    step2 = checkpoint2.get("step", "unknown")

    if verbose:
        print(f"  Agent 1: step {step1}")
        print(f"  Agent 2: step {step2}")

    # Build models
    if verbose:
        print("Building models...")
    agent1 = build_model_from_checkpoint(checkpoint1, device)
    agent2 = build_model_from_checkpoint(checkpoint2, device)

    # Create embedder for players
    feature_set = checkpoint1.get("config", {}).get("embedder_feature_set", "full")
    embedder = Embedder(
        format=battle_format,
        feature_set=feature_set,
        omniscient=False,
    )

    # Setup team repo if provided
    team_repo = None
    if team_path and os.path.exists(team_path):
        team_repo = TeamRepo(team_path)
        if verbose:
            print(f"Using teams from: {team_path}")
            if team_subdirectory:
                print(f"  Subdirectory: {team_subdirectory}")

    # Server configuration
    server_config = ServerConfiguration(
        f"ws://localhost:{server_port}/showdown/websocket",
        None,  # type: ignore[arg-type]
    )

    # Sample teams
    team1 = None
    team2 = None
    if team_repo:
        team1 = team_repo.sample_team(battle_format, subdirectory=team_subdirectory)
        team2 = team_repo.sample_team(battle_format, subdirectory=team_subdirectory)

    # Create players
    if verbose:
        print("Creating players...")

    player1 = BatchInferencePlayer(
        model=agent1,
        device=device,
        batch_size=16,
        account_configuration=AccountConfiguration("Agent1", None),
        server_configuration=server_config,
        battle_format=battle_format,
        team=team1,
        probabilistic=probabilistic,
        embedder=embedder,
        trajectory_queue=None,  # No trajectory collection for evaluation
    )

    player2 = BatchInferencePlayer(
        model=agent2,
        device=device,
        batch_size=16,
        account_configuration=AccountConfiguration("Agent2", None),
        server_configuration=server_config,
        battle_format=battle_format,
        team=team2,
        probabilistic=probabilistic,
        embedder=embedder,
        trajectory_queue=None,
    )

    # Start inference loops
    player1.start_inference_loop()
    player2.start_inference_loop()

    # Give time for inference loops to start
    await asyncio.sleep(0.5)

    if verbose:
        print(f"\nRunning {num_battles} battles...")

    start_time = time.time()

    # Run battles
    await player1.battle_against(player2, n_battles=num_battles)

    elapsed = time.time() - start_time

    # Collect results
    agent1_wins = sum(1 for battle in player1._battles.values() if battle.won)
    agent2_wins = sum(1 for battle in player2._battles.values() if battle.won)
    draws = num_battles - agent1_wins - agent2_wins
    battles_played = len(player1._battles)

    win_rate_1 = agent1_wins / battles_played if battles_played > 0 else 0.0
    win_rate_2 = agent2_wins / battles_played if battles_played > 0 else 0.0
    battles_per_second = battles_played / elapsed if elapsed > 0 else 0.0

    results = {
        "agent1_path": checkpoint1_path,
        "agent2_path": checkpoint2_path,
        "agent1_step": step1,
        "agent2_step": step2,
        "agent1_wins": agent1_wins,
        "agent2_wins": agent2_wins,
        "draws": draws,
        "agent1_win_rate": win_rate_1,
        "agent2_win_rate": win_rate_2,
        "battles_played": battles_played,
        "elapsed_seconds": elapsed,
        "battles_per_second": battles_per_second,
    }

    # Print results
    if verbose:
        print(f"\n{'=' * 60}")
        print("RESULTS")
        print(f"{'=' * 60}")
        print(f"Agent 1 (step {step1}): {agent1_wins} wins ({win_rate_1 * 100:.1f}%)")
        print(f"Agent 2 (step {step2}): {agent2_wins} wins ({win_rate_2 * 100:.1f}%)")
        if draws > 0:
            print(f"Draws: {draws}")
        print(f"\nBattles played: {battles_played}")
        print(f"Time elapsed: {elapsed:.1f}s")
        print(f"Speed: {battles_per_second:.2f} battles/sec")
        print(f"{'=' * 60}\n")

    # Cleanup
    try:
        await player1.stop_listening()
        await player2.stop_listening()
    except Exception:
        pass

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate two RL checkpoints by playing battles between them.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Compare two checkpoints
    python -m elitefurretai.rl.analyze.evaluate_agents \\
        data/models/rl/main_model_step_100.pt \\
        data/models/rl/main_model_step_500.pt \\
        --num-battles 100

    # Use specific teams
    python -m elitefurretai.rl.analyze.evaluate_agents \\
        model1.pt model2.pt \\
        --team-path data/teams \\
        --team-subdirectory easy

    # Run with probabilistic action selection
    python -m elitefurretai.rl.analyze.evaluate_agents \\
        model1.pt model2.pt \\
        --probabilistic
        """,
    )
    parser.add_argument(
        "checkpoint1", type=str, help="Path to first model checkpoint"
    )
    parser.add_argument(
        "checkpoint2", type=str, help="Path to second model checkpoint"
    )
    parser.add_argument(
        "-n",
        "--num-battles",
        type=int,
        default=100,
        help="Number of battles to run (default: 100)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for inference (default: cuda)",
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=8000,
        help="Pokemon Showdown server port (default: 8000)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="gen9vgc2023regc",
        help="Battle format (default: gen9vgc2023regc)",
    )
    parser.add_argument(
        "--team-path",
        type=str,
        default="data/teams",
        help="Path to team repository (default: data/teams)",
    )
    parser.add_argument(
        "--team-subdirectory",
        type=str,
        default=None,
        help="Subdirectory within format for teams (e.g., 'easy')",
    )
    parser.add_argument(
        "--probabilistic",
        action="store_true",
        help="Use probabilistic action selection instead of argmax",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    parser.add_argument(
        "--no-server",
        action="store_true",
        help="Don't start/stop Pokemon Showdown server (assume already running)",
    )
    parser.add_argument(
        "--showdown-dir",
        type=str,
        default="../pokemon-showdown",
        help="Path to pokemon-showdown directory (default: ../pokemon-showdown)",
    )

    args = parser.parse_args()

    # Validate checkpoint paths
    if not os.path.exists(args.checkpoint1):
        print(f"Error: Checkpoint not found: {args.checkpoint1}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(args.checkpoint2):
        print(f"Error: Checkpoint not found: {args.checkpoint2}", file=sys.stderr)
        sys.exit(1)

    # Start server if needed
    server_process = None
    if not args.no_server:
        try:
            server_process = launch_showdown_server(args.server_port, args.showdown_dir)
        except Exception as e:
            print(f"Error starting server: {e}", file=sys.stderr)
            sys.exit(1)

    # Run evaluation
    try:
        results = asyncio.run(
            run_evaluation(
                checkpoint1_path=args.checkpoint1,
                checkpoint2_path=args.checkpoint2,
                num_battles=args.num_battles,
                device=args.device,
                server_port=args.server_port,
                battle_format=args.format,
                team_path=args.team_path,
                team_subdirectory=args.team_subdirectory,
                probabilistic=args.probabilistic,
                verbose=not args.quiet,
            )
        )
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.")
        shutdown_showdown_server(server_process)
        sys.exit(1)
    except Exception as e:
        print(f"Error during evaluation: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        shutdown_showdown_server(server_process)
        sys.exit(1)

    # Shutdown server
    if server_process:
        shutdown_showdown_server(server_process)

    # Save results if requested
    if args.output:
        import json

        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")

    # Return exit code based on relative performance
    # (useful for CI/automated testing)
    sys.exit(0)


if __name__ == "__main__":
    main()
