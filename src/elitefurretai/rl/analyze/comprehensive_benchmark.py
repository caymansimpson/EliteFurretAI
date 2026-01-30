"""
Comprehensive benchmark to measure battle throughput vs RAM usage.

Tests:
- RandomPlayer vs RandomPlayer (simple baseline)
- BCPlayer vs BCPlayer (realistic opponent with model inference)

Configurations:
- Players per server: 2, 4, 8
- Number of servers: 2, 4, 8

Metrics:
- Battles per hour
- RAM usage (MB)
- CPU utilization
"""

import asyncio
import gc
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from typing import List

import psutil
import torch
from poke_env import AccountConfiguration, ServerConfiguration
from poke_env.player import RandomPlayer

from elitefurretai.etl.embedder import Embedder
from elitefurretai.etl.encoder import MDBO
from elitefurretai.etl.team_repo import TeamRepo
from elitefurretai.supervised.behavior_clone_player import BCPlayer
from elitefurretai.supervised.model_archs import FlexibleThreeHeadedModel

# ==================== SERVER MANAGEMENT ====================

def launch_showdown_servers(num_servers: int, start_port: int = 8000) -> List[subprocess.Popen]:
    """Launch multiple Showdown servers on consecutive ports."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, "..", "..", "..", ".."))
    showdown_dir = os.path.join(repo_root, "..", "pokemon-showdown")

    if not os.path.exists(showdown_dir):
        raise FileNotFoundError(f"Pokemon Showdown not found at {showdown_dir}")

    print(f"Launching {num_servers} Showdown servers (ports {start_port}-{start_port + num_servers - 1})...")

    processes = []
    for i in range(num_servers):
        port = start_port + i
        process = subprocess.Popen(
            ["node", "pokemon-showdown", "start", "--no-security", "--port", str(port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=showdown_dir,
            preexec_fn=os.setsid if os.name != "nt" else None,
        )
        processes.append(process)
        time.sleep(0.3)

    # Wait for servers to start (longer wait for multiple servers)
    wait_time = 2 + (num_servers * 0.5)  # Base 2s + 0.5s per server
    print(f"Waiting {wait_time:.1f}s for servers to initialize...")
    time.sleep(wait_time)
    print(f"✓ All {num_servers} servers ready\n")
    return processes


def shutdown_showdown_servers(processes: List[subprocess.Popen]):
    """Gracefully shut down all server processes."""
    if not processes:
        return

    print(f"\nShutting down {len(processes)} servers...")
    for process in processes:
        if process.poll() is None:
            try:
                if os.name != "nt":
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                else:
                    process.terminate()
                process.wait(timeout=3)
            except Exception:
                process.kill()
    print("✓ All servers shut down\n")


def create_server_config(port: int) -> ServerConfiguration:
    """Create server configuration for a given port."""
    return ServerConfiguration(
        f"ws://localhost:{port}/showdown/websocket",
        None,  # type: ignore
    )


# ==================== MEMORY TRACKING ====================

def get_memory_usage_mb() -> float:
    """Get current process RAM usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


def get_gpu_memory_usage_mb() -> float:
    """Get GPU memory usage in MB."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated() / (1024 * 1024)


# ==================== BENCHMARK RUNNERS ====================

def _create_bc_player_with_cached_model(
    cached_model,
    cached_embedder,
    cached_config,
    device,
    account_config,
    server_config,
    team,
):
    """
    Creates a BCPlayer instance and manually initializes it to use a shared,
    cached model, avoiding repeated disk loads and memory bloat.
    """
    # Manually create the player instance without calling __init__
    player = BCPlayer.__new__(BCPlayer)

    # Manually call the parent Player.__init__ to set up necessary attributes.
    # The correct signature is (self, account_configuration, battle_format, ...).
    # We pass keyword arguments to be safe.
    super(BCPlayer, player).__init__(
        account_configuration=account_config,
        battle_format="gen9vgc2023regc",
        server_configuration=server_config,
        team=team,
        log_level=20,
        max_concurrent_battles=1,
    )

    # Set attributes for the BCPlayer
    player.model = cached_model
    player.embedder = cached_embedder
    player.config = cached_config
    player._device = device
    player._verbose = False

    # Share cached model (no copy - saves ~500MB per instance)
    player.teampreview_model = cached_model
    player.action_model = cached_model
    player.win_model = cached_model
    player.teampreview_embedder = cached_embedder
    player.action_embedder = cached_embedder
    player.win_embedder = cached_embedder
    player.teampreview_config = cached_config
    player.action_config = cached_config
    player.win_config = cached_config

    # Initialize tracking attributes (required by choose_move)
    player._trajectories = {}
    player._hidden_states = {}
    player._last_message_error = {}
    player._last_message = {}
    player._last_win_advantage = {}
    player._win_advantage_threshold = 0.5
    player._probabilistic = False

    return player


async def run_random_player_benchmark(
    players_per_server: int,
    num_servers: int,
    battles_per_pair: int,
    start_port: int = 8000,
    run_id: int = 0,
) -> dict:
    """Benchmark RandomPlayer vs RandomPlayer battles."""

    team_repo = TeamRepo("data/teams")
    team = team_repo.sample_team("gen9vgc2023regc", subdirectory="straightforward")

    # Distribute players across servers
    players = []
    opponents = []

    player_idx = 0
    for server_idx in range(num_servers):
        port = start_port + server_idx
        server_config = create_server_config(port)

        for pair_idx in range(players_per_server):
            player = RandomPlayer(
                account_configuration=AccountConfiguration(f"RP{run_id}_{player_idx}", None),
                server_configuration=server_config,
                battle_format="gen9vgc2023regc",
                team=team,
            )
            opponent = RandomPlayer(
                account_configuration=AccountConfiguration(f"RO{run_id}_{player_idx}", None),
                server_configuration=server_config,
                battle_format="gen9vgc2023regc",
                team=team,
            )
            players.append(player)
            opponents.append(opponent)
            player_idx += 1

    # Wait for connections to establish
    await asyncio.sleep(1.5)

    # Record memory before battles
    mem_before = get_memory_usage_mb()
    gpu_mem_before = get_gpu_memory_usage_mb()

    # Run battles
    start_time = time.time()

    tasks = []
    for player, opponent in zip(players, opponents):
        tasks.append(player.battle_against(opponent, n_battles=battles_per_pair))

    await asyncio.gather(*tasks)

    elapsed = time.time() - start_time

    # Record memory after battles
    mem_after = get_memory_usage_mb()
    gpu_mem_after = get_gpu_memory_usage_mb()

    # Calculate metrics
    total_battles = sum(p.n_finished_battles for p in players)
    rate_hr = (total_battles / elapsed) * 3600

    # Cleanup
    for p in players:
        try:
            await p.stop_listening()  # type: ignore
        except Exception:
            pass
    for o in opponents:
        try:
            await o.stop_listening()  # type: ignore
        except Exception:
            pass

    await asyncio.sleep(0.5)

    return {
        "player_type": "RandomPlayer",
        "players_per_server": players_per_server,
        "num_servers": num_servers,
        "total_players": len(players),
        "battles": total_battles,
        "elapsed": elapsed,
        "rate_hr": rate_hr,
        "ram_before_mb": mem_before,
        "ram_after_mb": mem_after,
        "ram_delta_mb": mem_after - mem_before,
        "gpu_mem_before_mb": gpu_mem_before,
        "gpu_mem_after_mb": gpu_mem_after,
        "gpu_mem_delta_mb": gpu_mem_after - gpu_mem_before,
    }


async def run_bc_player_benchmark(
    players_per_server: int,
    num_servers: int,
    battles_per_pair: int,
    model_path: str,
    start_port: int = 8000,
    run_id: int = 0,
) -> dict:
    """Benchmark BCPlayer vs BCPlayer battles with cached model."""

    team_repo = TeamRepo("data/teams")
    team = team_repo.sample_team("gen9vgc2023regc", subdirectory="straightforward")

    # Force CPU for this benchmark to avoid CUDA multiprocessing overhead issues
    device = "cpu"

    # Record memory before loading model
    mem_before_model = get_memory_usage_mb()
    gpu_mem_before_model = get_gpu_memory_usage_mb()

    # Load model ONCE and cache it (saves ~500MB per BCPlayer instance)
    print(f"    Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint["config"]

    # Create embedder
    embedder = Embedder(
        format="gen9vgc2023regc",
        feature_set=config.get("embedder_feature_set", "raw"),
        omniscient=False,
    )

    # Create model
    cached_model = FlexibleThreeHeadedModel(
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
        max_seq_len=config.get("max_seq_len", 17),
    ).to(device)

    cached_model.load_state_dict(checkpoint["model_state_dict"])
    cached_model.eval()

    # Distribute players across servers
    players = []
    opponents = []

    player_idx = 0
    for server_idx in range(num_servers):
        port = start_port + server_idx
        server_config = create_server_config(port)

        for pair_idx in range(players_per_server):
            # Create BCPlayer instances with shared cached model (no disk loading)
            player = _create_bc_player_with_cached_model(
                cached_model=cached_model,
                cached_embedder=embedder,
                cached_config=config,
                device=device,
                account_config=AccountConfiguration(f"BP{run_id}_{player_idx}", None),
                server_config=server_config,
                team=team,
            )
            opponent = _create_bc_player_with_cached_model(
                cached_model=cached_model,
                cached_embedder=embedder,
                cached_config=config,
                device=device,
                account_config=AccountConfiguration(f"BO{run_id}_{player_idx}", None),
                server_config=server_config,
                team=team,
            )
            players.append(player)
            opponents.append(opponent)
            player_idx += 1

    # Record memory after loading models
    mem_after_model = get_memory_usage_mb()
    gpu_mem_after_model = get_gpu_memory_usage_mb()

    # Wait for connections to establish
    await asyncio.sleep(1.5)

    # Record memory before battles
    mem_before = get_memory_usage_mb()
    gpu_mem_before = get_gpu_memory_usage_mb()

    # Run battles
    start_time = time.time()

    tasks = []
    for player, opponent in zip(players, opponents):
        tasks.append(player.battle_against(opponent, n_battles=battles_per_pair))

    await asyncio.gather(*tasks)

    elapsed = time.time() - start_time

    # Record memory after battles
    mem_after = get_memory_usage_mb()
    gpu_mem_after = get_gpu_memory_usage_mb()

    # Calculate metrics
    total_battles = sum(p.n_finished_battles for p in players)
    rate_hr = (total_battles / elapsed) * 3600

    # Cleanup
    for p in players:
        try:
            await p.stop_listening()
        except Exception:
            pass
    for o in opponents:
        try:
            await o.stop_listening()
        except Exception:
            pass

    await asyncio.sleep(0.5)

    return {
        "player_type": "BCPlayer",
        "players_per_server": players_per_server,
        "num_servers": num_servers,
        "total_players": len(players),
        "battles": total_battles,
        "elapsed": elapsed,
        "rate_hr": rate_hr,
        "ram_before_mb": mem_before,
        "ram_after_mb": mem_after,
        "ram_delta_mb": mem_after - mem_before,
        "ram_model_mb": mem_after_model - mem_before_model,
        "gpu_mem_before_mb": gpu_mem_before,
        "gpu_mem_after_mb": gpu_mem_after,
        "gpu_mem_delta_mb": gpu_mem_after - gpu_mem_before,
        "gpu_mem_model_mb": gpu_mem_after_model - gpu_mem_before_model,
    }


# ==================== MAIN BENCHMARK ====================

async def run_all_benchmarks(
    players_per_server_list: List[int],
    num_servers_list: List[int],
    battles_per_pair: int,
    model_path: str,
):
    """Run comprehensive benchmark suite."""

    results = []
    test_num = 1
    total_tests = len(players_per_server_list) * len(num_servers_list) * 2  # × 2 for Random + BC

    print("=" * 80)
    print("COMPREHENSIVE BENCHMARK: RandomPlayer vs BCPlayer")
    print("=" * 80)
    print(f"\nTotal configurations to test: {total_tests}")
    print(f"Battles per configuration: {battles_per_pair * max(players_per_server_list) * max(num_servers_list)}")
    print("\n")

    for num_servers in num_servers_list:
        # Launch servers for this configuration
        servers = launch_showdown_servers(num_servers, start_port=8000)

        try:
            for players_per_server in players_per_server_list:
                total_players = players_per_server * num_servers

                # Test 1: RandomPlayer
                print(f"[{test_num}/{total_tests}] RandomPlayer: {players_per_server} players/server × {num_servers} servers = {total_players} total")
                try:
                    result = await run_random_player_benchmark(
                        players_per_server=players_per_server,
                        num_servers=num_servers,
                        battles_per_pair=battles_per_pair,
                        run_id=test_num,
                    )
                    results.append(result)
                    print(f"    → {result['rate_hr']:.0f} battles/hr, RAM: {result['ram_after_mb']:.0f}MB (+{result['ram_delta_mb']:.0f}MB)")
                except Exception as e:
                    print(f"    ✗ FAILED: {e}")

                test_num += 1

                # Force garbage collection between tests
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                await asyncio.sleep(2)

                # Test 2: BCPlayer
                print(f"[{test_num}/{total_tests}] BCPlayer: {players_per_server} players/server × {num_servers} servers = {total_players} total")
                try:
                    result = await run_bc_player_benchmark(
                        players_per_server=players_per_server,
                        num_servers=num_servers,
                        battles_per_pair=battles_per_pair,
                        model_path=model_path,
                        run_id=test_num,
                    )
                    results.append(result)
                    print(f"    → {result['rate_hr']:.0f} battles/hr, RAM: {result['ram_after_mb']:.0f}MB (+{result['ram_delta_mb']:.0f}MB), Model: {result['ram_model_mb']:.0f}MB")
                except Exception as e:
                    print(f"    ✗ FAILED: {e}")

                test_num += 1

                # Force garbage collection between tests
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                await asyncio.sleep(2)

        finally:
            # Shutdown servers for this configuration
            shutdown_showdown_servers(servers)
            await asyncio.sleep(1)

    return results


def print_results(results: List[dict]):
    """Print formatted results table."""

    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    print()

    # Group by player type
    random_results = [r for r in results if r["player_type"] == "RandomPlayer"]
    bc_results = [r for r in results if r["player_type"] == "BCPlayer"]

    # RandomPlayer results
    if random_results:
        print("RandomPlayer Benchmarks:")
        print("-" * 80)
        print(f"{'Servers':>8} {'Players/Srv':>12} {'Total':>8} {'Battles':>10} {'Rate (b/hr)':>12} {'RAM (MB)':>12}")
        print("-" * 80)
        for r in random_results:
            print(f"{r['num_servers']:>8} {r['players_per_server']:>12} {r['total_players']:>8} "
                  f"{r['battles']:>10} {r['rate_hr']:>12.0f} {r['ram_after_mb']:>12.0f}")
        print()

    # BCPlayer results
    if bc_results:
        print("BCPlayer Benchmarks:")
        print("-" * 80)
        print(f"{'Servers':>8} {'Players/Srv':>12} {'Total':>8} {'Battles':>10} {'Rate (b/hr)':>12} {'RAM (MB)':>12} {'Model (MB)':>12}")
        print("-" * 80)
        for r in bc_results:
            print(f"{r['num_servers']:>8} {r['players_per_server']:>12} {r['total_players']:>8} "
                  f"{r['battles']:>10} {r['rate_hr']:>12.0f} {r['ram_after_mb']:>12.0f} {r['ram_model_mb']:>12.0f}")
        print()

    # Key findings
    print("Key Findings:")
    print("-" * 80)

    if random_results:
        best_random = max(random_results, key=lambda x: x["rate_hr"])
        print(f"Best RandomPlayer config: {best_random['players_per_server']} players/srv × {best_random['num_servers']} servers")
        print(f"  → {best_random['rate_hr']:.0f} battles/hr at {best_random['ram_after_mb']:.0f}MB RAM")

    if bc_results:
        best_bc = max(bc_results, key=lambda x: x["rate_hr"])
        print(f"Best BCPlayer config: {best_bc['players_per_server']} players/srv × {best_bc['num_servers']} servers")
        print(f"  → {best_bc['rate_hr']:.0f} battles/hr at {best_bc['ram_after_mb']:.0f}MB RAM")
        print(f"  → Model loading overhead: {best_bc['ram_model_mb']:.0f}MB per {best_bc['total_players']} players")

    print()


async def main():
    # Configuration
    players_per_server_list = [2, 4, 8]
    num_servers_list = [2, 4, 8]
    battles_per_pair = 50  # 50 battles per pair for decent sample size
    model_path = "data/models/supervised/magic-resonance-88.pt"

    # Check model exists
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        sys.exit(1)

    # Run benchmarks
    results = await run_all_benchmarks(
        players_per_server_list=players_per_server_list,
        num_servers_list=num_servers_list,
        battles_per_pair=battles_per_pair,
        model_path=model_path,
    )

    # Print results
    print_results(results)

    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"benchmark_results_{timestamp}.txt"

    print("COMPREHENSIVE BENCHMARK RESULTS\n")
    print("=" * 80 + "\n\n")

    for r in results:
        print(f"Player: {r['player_type']}\n")
        print(f"  Servers: {r['num_servers']}, Players/Server: {r['players_per_server']}, Total: {r['total_players']}\n")
        print(f"  Battles: {r['battles']}, Time: {r['elapsed']:.1f}s, Rate: {r['rate_hr']:.0f} b/hr\n")
        print(f"  RAM: {r['ram_after_mb']:.0f}MB (Δ{r['ram_delta_mb']:.0f}MB)\n")
        if "ram_model_mb" in r:
            print(f"  Model RAM: {r['ram_model_mb']:.0f}MB\n")
        print("\n")

    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    # Set sharing strategy for WSL compatibility and CUDA
    torch.multiprocessing.set_start_method("forkserver", force=True)
    torch.multiprocessing.set_sharing_strategy("file_system")
    asyncio.run(main())
