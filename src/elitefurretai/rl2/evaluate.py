import asyncio
import argparse
import torch
from typing import Dict, Any, Optional
from poke_env.ps_client import AccountConfiguration, ServerConfiguration, LocalhostServerConfiguration, DefaultAuthenticationURL

from elitefurretai.rl2.agent import RNaDAgent
from elitefurretai.rl2.worker import BatchInferencePlayer
from elitefurretai.agents.behavior_clone_player import FlexibleThreeHeadedModel
from elitefurretai.model_utils.embedder import Embedder
from elitefurretai.model_utils.encoder import MDBO


def load_model(filepath: str, device: str) -> RNaDAgent:
    """Load a model checkpoint and wrap it in RNaDAgent."""
    checkpoint = torch.load(filepath, map_location=device)
    config = checkpoint['config']
    state_dict = checkpoint['model_state_dict']

    embedder = Embedder(format="gen9vgc2023regulationc", feature_set=Embedder.FULL, omniscient=False)
    input_size = embedder.embedding_size

    base_model = FlexibleThreeHeadedModel(
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
        group_sizes=embedder.group_embedding_sizes if config.get("use_grouped_encoder", False) else None,
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

    base_model.load_state_dict(state_dict)
    return RNaDAgent(base_model)


async def run_evaluation(
    model1_path: str,
    model2_path: str,
    num_battles: int,
    device: str = "cuda",
    server_config: Optional[ServerConfiguration] = None
) -> Dict[str, Any]:
    """
    Run battles between two models and compute win rates.

    Args:
        model1_path: Path to first model checkpoint
        model2_path: Path to second model checkpoint
        num_battles: Number of battles to run
        device: Device to run models on
        server_config: Server configuration (defaults to localhost:8000)

    Returns:
        Dictionary with evaluation results:
            - model1_wins: Number of wins for model 1
            - model2_wins: Number of wins for model 2
            - win_rate: Win rate of model 1 (0-1)
            - battles_played: Total battles completed
    """
    print(f"Loading model 1 from {model1_path}...")
    model1 = load_model(model1_path, device)

    print(f"Loading model 2 from {model2_path}...")
    model2 = load_model(model2_path, device)

    if server_config is None:
        server_config = LocalhostServerConfiguration

    # Create players
    player1 = BatchInferencePlayer(
        model=model1,
        device=device,
        batch_size=16,
        account_configuration=AccountConfiguration("Model1", None),
        server_configuration=server_config,
        battle_format="gen9vgc2023regulationc",
        probabilistic=False  # Use deterministic for evaluation
    )

    player2 = BatchInferencePlayer(
        model=model2,
        device=device,
        batch_size=16,
        account_configuration=AccountConfiguration("Model2", None),
        server_configuration=server_config,
        battle_format="gen9vgc2023regulationc",
        probabilistic=False
    )

    # Start inference loops
    await player1.start_inference_loop()
    await player2.start_inference_loop()

    print(f"Running {num_battles} battles...")

    # Run battles
    await player1.battle_against(player2, n_battles=num_battles)

    # Collect results
    model1_wins = sum(1 for battle in player1._battles.values() if battle.won)
    model2_wins = sum(1 for battle in player2._battles.values() if battle.won)
    battles_played = len(player1._battles)

    win_rate = model1_wins / battles_played if battles_played > 0 else 0.0

    results = {
        "model1_path": model1_path,
        "model2_path": model2_path,
        "model1_wins": model1_wins,
        "model2_wins": model2_wins,
        "win_rate": win_rate,
        "battles_played": battles_played
    }

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Model 1: {model1_path}")
    print(f"Model 2: {model2_path}")
    print(f"Battles: {battles_played}")
    print(f"Model 1 Wins: {model1_wins} ({win_rate * 100:.1f}%)")
    print(f"Model 2 Wins: {model2_wins} ({(1 - win_rate) * 100:.1f}%)")
    print("=" * 50 + "\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate two Pokemon VGC models by running battles")
    parser.add_argument("model1", type=str, help="Path to first model checkpoint")
    parser.add_argument("model2", type=str, help="Path to second model checkpoint")
    parser.add_argument("--num-battles", type=int, default=100, help="Number of battles to run (default: 100)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on (default: cuda)")
    parser.add_argument("--server", type=str, default="localhost:8000", help="Showdown server address")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file for results")

    args = parser.parse_args()

    server_config = ServerConfiguration(args.server, DefaultAuthenticationURL)

    # Run evaluation
    results = asyncio.run(run_evaluation(
        args.model1,
        args.model2,
        args.num_battles,
        args.device,
        server_config
    ))

    # Save results if requested
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
