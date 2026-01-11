"""
Comprehensive scaling benchmark for RL training throughput.
Tests various combinations of servers and players_per_worker.
"""

import asyncio
import gc
import time
import uuid
from dataclasses import dataclass
from typing import List

import psutil
import torch
from poke_env import AccountConfiguration, ServerConfiguration

from elitefurretai.etl.embedder import Embedder
from elitefurretai.etl.encoder import MDBO
from elitefurretai.rl.agent import RNaDAgent
from elitefurretai.rl.worker import BatchInferencePlayer
from elitefurretai.supervised.model_archs import FlexibleThreeHeadedModel


@dataclass
class BenchmarkResult:
    num_servers: int
    pairs_per_server: int
    total_pairs: int
    battles_completed: int
    elapsed_time: float
    battles_per_hour: float
    memory_mb: float


def get_memory_mb():
    """Get current memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


def create_server_config(port: int) -> ServerConfiguration:
    """Create server configuration for a port."""
    return ServerConfiguration(
        f"ws://localhost:{port}/showdown/websocket",
        None,  # type: ignore[arg-type]
    )


async def run_benchmark(
    num_servers: int,
    pairs_per_server: int,
    battles_per_pair: int,
    available_ports: List[int],
    model,
    format_: str,
    team: str,
    run_id: str,
) -> BenchmarkResult:
    """Run benchmark for a specific configuration."""

    total_pairs = num_servers * pairs_per_server

    # Memory before
    gc.collect()
    mem_before = get_memory_mb()

    # Create player pairs, distributed across servers
    players = []
    opponents = []

    for pair_idx in range(total_pairs):
        server_idx = pair_idx % num_servers
        port = available_ports[server_idx]
        server_config = create_server_config(port)

        # Create agent wrapper
        agent = RNaDAgent(model=model)

        # Create player
        player = BatchInferencePlayer(
            model=agent,
            device="cuda",
            account_configuration=AccountConfiguration(f"P{run_id}_{pair_idx}", None),
            server_configuration=server_config,
            battle_format=format_,
            team=team,
            accept_open_team_sheet=True,
        )

        # Create opponent
        opponent = BatchInferencePlayer(
            model=agent,
            device="cuda",
            account_configuration=AccountConfiguration(f"O{run_id}_{pair_idx}", None),
            server_configuration=server_config,
            battle_format=format_,
            team=team,
            accept_open_team_sheet=True,
        )

        players.append(player)
        opponents.append(opponent)

    # Start inference loops
    for p in players:
        p.start_inference_loop()
    for o in opponents:
        o.start_inference_loop()

    # Give time to start
    await asyncio.sleep(0.2)

    # Run battles
    start_time = time.time()

    tasks = []
    for player, opponent in zip(players, opponents):
        tasks.append(player.battle_against(opponent, n_battles=battles_per_pair))

    try:
        await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=180,  # 3 min timeout
        )
    except asyncio.TimeoutError:
        print("  [TIMEOUT]", end="", flush=True)

    elapsed = time.time() - start_time

    # Count completed battles
    completed = sum(p.n_finished_battles for p in players)

    mem_after = get_memory_mb()

    # Cleanup
    for p in players:
        if hasattr(p, "_inference_future"):
            p._inference_future.cancel()
    for o in opponents:
        if hasattr(o, "_inference_future"):
            o._inference_future.cancel()

    del players, opponents
    gc.collect()

    battles_per_hour = (completed / elapsed) * 3600 if elapsed > 0 else 0

    return BenchmarkResult(
        num_servers=num_servers,
        pairs_per_server=pairs_per_server,
        total_pairs=total_pairs,
        battles_completed=completed,
        elapsed_time=elapsed,
        battles_per_hour=battles_per_hour,
        memory_mb=mem_after - mem_before,
    )


async def main():
    print("=" * 70)
    print("SCALING BENCHMARK: Servers × Pairs per Server")
    print("=" * 70)

    # Configuration
    available_ports = [8000, 8001, 8002, 8003]
    battles_per_pair = 10  # Reduced for faster iteration, still enough for signal
    battle_format = "gen9vgc2023regc"  # Showdown format string
    embedder_format = "gen9vgc2023regulationc"  # Embedder format string
    team_path = (
        "/home/cayman/Repositories/EliteFurretAI/data/teams/gen9vgc2023regc/easy/basic.txt"
    )
    checkpoint_path = "/home/cayman/Repositories/EliteFurretAI/data/models/supervised/ethereal-flower-59-extended.pt"

    with open(team_path) as f:
        team = f.read()

    # Load model
    print("\nLoading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embedder = Embedder(
        format=embedder_format,
        omniscient=False,
        feature_set="full",
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    state_dict = checkpoint["model_state_dict"]

    model = FlexibleThreeHeadedModel(
        input_size=embedder.embedding_size,
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
    model.load_state_dict(state_dict)
    model.eval()

    # Test configurations: (num_servers, pairs_per_server)
    configs = [
        # Push hardware limits - test high concurrency
        (4, 1),  # Baseline best: 4 servers × 1 pair = 4 concurrent
        (4, 2),  # 4 servers × 2 pairs = 8 concurrent
        (4, 3),  # 4 servers × 3 pairs = 12 concurrent
        (4, 4),  # 4 servers × 4 pairs = 16 concurrent
    ]

    results = []
    baseline_mem = get_memory_mb()
    run_id = uuid.uuid4().hex[:6]

    print(f"\nBaseline memory: {baseline_mem:.1f} MB")
    print(f"Battles per pair: {battles_per_pair}")
    print(f"Run ID: {run_id}")
    print("\n" + "-" * 70)
    print(
        f"{'Servers':>8} {'Pairs/Srv':>10} {'Total':>6} {'Battles':>8} {'Time(s)':>8} {'Rate/hr':>10} {'MemΔ(MB)':>10}"
    )
    print("-" * 70)

    for num_servers, pairs_per_server in configs:
        gc.collect()
        await asyncio.sleep(1)  # Let servers settle

        try:
            result = await run_benchmark(
                num_servers=num_servers,
                pairs_per_server=pairs_per_server,
                battles_per_pair=battles_per_pair,
                available_ports=available_ports,
                model=model,
                format_=battle_format,
                team=team,
                run_id=f"{run_id}_{num_servers}_{pairs_per_server}",
            )
            results.append(result)

            print(
                f"{result.num_servers:>8} {result.pairs_per_server:>10} {result.total_pairs:>6} "
                f"{result.battles_completed:>8} {result.elapsed_time:>8.1f} "
                f"{result.battles_per_hour:>10.0f} {result.memory_mb:>10.1f}"
            )

        except Exception as e:
            print(f"{num_servers:>8} {pairs_per_server:>10} - FAILED: {e}")
            import traceback

            traceback.print_exc()

    print("-" * 70)

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    if results:
        # Find best config
        best = max(results, key=lambda r: r.battles_per_hour)
        print(
            f"\nBest configuration: {best.num_servers} servers × {best.pairs_per_server} pairs/server"
        )
        print(f"  → {best.battles_per_hour:.0f} battles/hour")

        # Calculate marginal rates
        print("\n--- Marginal Impact of Adding Pairs (same server count) ---")
        for num_srv in [1, 2, 4]:
            srv_results = [r for r in results if r.num_servers == num_srv]
            srv_results.sort(key=lambda r: r.pairs_per_server)
            for i in range(1, len(srv_results)):
                prev = srv_results[i - 1]
                curr = srv_results[i]
                rate_delta = curr.battles_per_hour - prev.battles_per_hour
                pairs_delta = curr.total_pairs - prev.total_pairs
                per_pair = rate_delta / pairs_delta if pairs_delta > 0 else 0
                print(
                    f"  {num_srv} srv: {prev.total_pairs}→{curr.total_pairs} pairs: "
                    f"+{rate_delta:.0f}/hr (+{per_pair:.0f}/hr per pair)"
                )

        print("\n--- Marginal Impact of Adding Servers (same total pairs) ---")
        for total_pairs in [2, 4]:
            pair_results = [r for r in results if r.total_pairs == total_pairs]
            pair_results.sort(key=lambda r: r.num_servers)
            for i in range(1, len(pair_results)):
                prev = pair_results[i - 1]
                curr = pair_results[i]
                rate_delta = curr.battles_per_hour - prev.battles_per_hour
                srv_delta = curr.num_servers - prev.num_servers
                per_srv = rate_delta / srv_delta if srv_delta > 0 else 0
                print(
                    f"  {total_pairs} pairs: {prev.num_servers}→{curr.num_servers} srv: "
                    f"+{rate_delta:.0f}/hr (+{per_srv:.0f}/hr per server)"
                )


if __name__ == "__main__":
    asyncio.run(main())
