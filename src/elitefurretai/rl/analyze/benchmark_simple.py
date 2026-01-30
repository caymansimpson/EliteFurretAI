#!/usr/bin/env python3
"""
Simple, focused benchmark script for RL training optimization.

This script runs battles with different configurations to measure throughput.
Results are appended to OPTIMIZATIONS_V2.md.

Usage:
    python src/elitefurretai/rl/benchmark_simple.py --test random
    python src/elitefurretai/rl/benchmark_simple.py --test bc_cpu
    python src/elitefurretai/rl/benchmark_simple.py --test bc_gpu
    python src/elitefurretai/rl/benchmark_simple.py --test batch
    python src/elitefurretai/rl/benchmark_simple.py --test all
"""

import argparse
import asyncio
import gc

# Reduce logging verbosity
import logging
import os
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import List

logging.getLogger("poke_env").setLevel(logging.WARNING)

import psutil
import torch
from poke_env.player import RandomPlayer
from poke_env.ps_client import AccountConfiguration, ServerConfiguration


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    name: str
    battles_completed: int
    duration_seconds: float
    battles_per_second: float
    avg_gpu_utilization: float
    avg_gpu_memory_mb: float
    avg_ram_usage_mb: float
    peak_ram_usage_mb: float
    errors: int


def wait_for_port(port: int, timeout: float = 30) -> bool:
    """Wait for a port to become available."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1)
                result = sock.connect_ex(('localhost', port))
                if result == 0:
                    return True
        except socket.error:
            pass
        time.sleep(0.5)
    return False


def start_showdown_servers(num_servers: int, start_port: int = 8000) -> List[subprocess.Popen]:
    """Start Showdown servers and return process handles."""
    processes = []
    showdown_path = os.path.expanduser("~/Repositories/pokemon-showdown")

    print(f"Starting {num_servers} Showdown servers...")
    for i in range(num_servers):
        port = start_port + i
        proc = subprocess.Popen(
            ["node", "pokemon-showdown", "start", "--no-security", "--port", str(port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=showdown_path,
        )
        processes.append(proc)

    # Wait for servers to be ready
    for i in range(num_servers):
        port = start_port + i
        if wait_for_port(port):
            print(f"  Server on port {port} ready")
        else:
            print(f"  WARNING: Server on port {port} may not be ready")

    return processes


def stop_showdown_servers(processes: List[subprocess.Popen]):
    """Stop Showdown servers."""
    print("Stopping servers...")
    for proc in processes:
        proc.terminate()
    time.sleep(1)
    for proc in processes:
        if proc.poll() is None:
            proc.kill()


def get_server_config(port: int) -> ServerConfiguration:
    """Create server configuration for a port."""
    return ServerConfiguration(
        f"ws://localhost:{port}/showdown/websocket",
        None,  # type: ignore[arg-type]
    )


def load_team() -> str:
    """Load a team from TeamRepo."""
    from elitefurretai.etl.team_repo import TeamRepo
    repo = TeamRepo("data/teams")
    team = repo.sample_team("gen9vgc2023regc", subdirectory="straightforward")
    return team


async def run_random_vs_random(
    num_battles: int,
    num_servers: int = 2,
    players_per_server: int = 4,
    start_port: int = 8000,
) -> BenchmarkResult:
    """Benchmark RandomPlayer vs RandomPlayer."""
    print(f"\n{'='*60}")
    print("Test: RandomPlayer vs RandomPlayer")
    print(f"Servers: {num_servers}, Players/server: {players_per_server}")
    print(f"{'='*60}")

    # Load team
    team = load_team()

    # Create players
    players = []
    opponents = []
    player_id = 0

    for server_idx in range(num_servers):
        server_config = get_server_config(start_port + server_idx)

        for _ in range(players_per_server):
            p = RandomPlayer(
                battle_format="gen9vgc2023regc",
                server_configuration=server_config,
                account_configuration=AccountConfiguration(f"BenchP{player_id}", None),
                max_concurrent_battles=1,
                team=team,
            )
            o = RandomPlayer(
                battle_format="gen9vgc2023regc",
                server_configuration=server_config,
                account_configuration=AccountConfiguration(f"BenchO{player_id}", None),
                max_concurrent_battles=1,
                team=team,
            )
            players.append(p)
            opponents.append(o)
            player_id += 1

    # Run battles
    start_time = time.time()
    battles_completed = 0
    errors = 0

    total_pairs = len(players)
    battles_per_pair = (num_battles + total_pairs - 1) // total_pairs

    print(f"Running {battles_per_pair} battles per pair ({total_pairs} pairs)...")

    async def run_pair_battles(player, opponent, n):
        nonlocal battles_completed, errors
        try:
            await player.battle_against(opponent, n_battles=n)
            battles_completed += n
        except Exception as e:
            errors += 1
            print(f"Error: {e}")

    # Sample RAM periodically
    tasks = []
    for p, o in zip(players, opponents):
        tasks.append(run_pair_battles(p, o, battles_per_pair))

    # Run all battles concurrently
    await asyncio.gather(*tasks, return_exceptions=True)

    duration = time.time() - start_time

    # Get RAM usage
    process = psutil.Process()
    ram_mb = process.memory_info().rss / 1024 / 1024

    result = BenchmarkResult(
        name="RandomPlayer vs RandomPlayer",
        battles_completed=battles_completed,
        duration_seconds=duration,
        battles_per_second=battles_completed / duration if duration > 0 else 0,
        avg_gpu_utilization=0,
        avg_gpu_memory_mb=0,
        avg_ram_usage_mb=ram_mb,
        peak_ram_usage_mb=ram_mb,
        errors=errors,
    )

    print("\nResults:")
    print(f"  Battles: {result.battles_completed}")
    print(f"  Duration: {result.duration_seconds:.1f}s")
    print(f"  Throughput: {result.battles_per_second:.2f} battles/s")
    print(f"  RAM: {result.avg_ram_usage_mb:.0f} MB")
    print(f"  Errors: {result.errors}")

    return result


async def run_bc_vs_bc(
    num_battles: int,
    model_path: str,
    device: str = "cpu",
    num_servers: int = 2,
    players_per_server: int = 2,
    start_port: int = 8000,
) -> BenchmarkResult:
    """Benchmark BCPlayer vs BCPlayer."""
    from elitefurretai.supervised.behavior_clone_player import BCPlayer

    print(f"\n{'='*60}")
    print(f"Test: BCPlayer vs BCPlayer ({device.upper()})")
    print(f"Servers: {num_servers}, Players/server: {players_per_server}")
    print(f"Model: {model_path}")
    print(f"{'='*60}")

    # Load team
    team = load_team()

    # Create players
    players = []
    opponents = []
    player_id = 0

    for server_idx in range(num_servers):
        server_config = get_server_config(start_port + server_idx)

        for _ in range(players_per_server):
            p = BCPlayer(
                unified_model_filepath=model_path,
                battle_format="gen9vgc2023regc",
                server_configuration=server_config,
                account_configuration=AccountConfiguration(f"BCP{player_id}", None),
                device=device,
                probabilistic=True,
                max_concurrent_battles=1,
                team=team,
            )
            o = BCPlayer(
                unified_model_filepath=model_path,
                battle_format="gen9vgc2023regc",
                server_configuration=server_config,
                account_configuration=AccountConfiguration(f"BCO{player_id}", None),
                device=device,
                probabilistic=True,
                max_concurrent_battles=1,
                team=team,
            )
            players.append(p)
            opponents.append(o)
            player_id += 1

    # Run battles
    start_time = time.time()
    battles_completed = 0
    errors = 0

    total_pairs = len(players)
    battles_per_pair = (num_battles + total_pairs - 1) // total_pairs

    print(f"Running {battles_per_pair} battles per pair ({total_pairs} pairs)...")

    async def run_pair_battles(player, opponent, n):
        nonlocal battles_completed, errors
        try:
            await player.battle_against(opponent, n_battles=n)
            battles_completed += n
        except Exception as e:
            errors += 1
            print(f"Error: {e}")

    tasks = []
    for p, o in zip(players, opponents):
        tasks.append(run_pair_battles(p, o, battles_per_pair))

    await asyncio.gather(*tasks, return_exceptions=True)

    duration = time.time() - start_time

    # Get resource usage
    process = psutil.Process()
    ram_mb = process.memory_info().rss / 1024 / 1024

    gpu_util = 0
    gpu_mem = 0
    if device == "cuda" and torch.cuda.is_available():
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                util, mem = result.stdout.strip().split(",")
                gpu_util = float(util)
                gpu_mem = float(mem)
        except Exception:
            pass

    result = BenchmarkResult(
        name=f"BCPlayer vs BCPlayer ({device})",
        battles_completed=battles_completed,
        duration_seconds=duration,
        battles_per_second=battles_completed / duration if duration > 0 else 0,
        avg_gpu_utilization=gpu_util,
        avg_gpu_memory_mb=gpu_mem,
        avg_ram_usage_mb=ram_mb,
        peak_ram_usage_mb=ram_mb,
        errors=errors,
    )

    print("\nResults:")
    print(f"  Battles: {result.battles_completed}")
    print(f"  Duration: {result.duration_seconds:.1f}s")
    print(f"  Throughput: {result.battles_per_second:.2f} battles/s")
    print(f"  GPU Util: {result.avg_gpu_utilization:.0f}%")
    print(f"  GPU Mem: {result.avg_gpu_memory_mb:.0f} MB")
    print(f"  RAM: {result.avg_ram_usage_mb:.0f} MB")
    print(f"  Errors: {result.errors}")

    # Cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


async def run_batch_inference(
    num_battles: int,
    model_path: str,
    batch_size: int = 8,
    num_servers: int = 2,
    players_per_server: int = 2,
    start_port: int = 8000,
) -> BenchmarkResult:
    """Benchmark BatchInferencePlayer vs BatchInferencePlayer."""
    from elitefurretai.rl.agent import RNaDAgent
    from elitefurretai.rl.multiprocess_actor import BatchInferencePlayer
    from elitefurretai.rl.train import load_model

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*60}")
    print("Test: BatchInferencePlayer vs BatchInferencePlayer")
    print(f"Servers: {num_servers}, Players/server: {players_per_server}")
    print(f"Batch size: {batch_size}, Device: {device}")
    print(f"Model: {model_path}")
    print(f"{'='*60}")

    # Load team
    team = load_team()

    # Load model once
    print("Loading model...")
    base_model = load_model(model_path, device)
    agent = RNaDAgent(base_model)
    agent.to(device)
    agent.eval()

    # Create players
    players = []
    opponents = []
    player_id = 0

    for server_idx in range(num_servers):
        server_config = get_server_config(start_port + server_idx)

        for _ in range(players_per_server):
            p = BatchInferencePlayer(
                model=agent,
                device=device,
                batch_size=batch_size,
                probabilistic=True,
                battle_format="gen9vgc2023regc",
                server_configuration=server_config,
                account_configuration=AccountConfiguration(f"BIP{player_id}", None),
                max_concurrent_battles=1,
                team=team,
            )
            o = BatchInferencePlayer(
                model=agent,
                device=device,
                batch_size=batch_size,
                probabilistic=True,
                battle_format="gen9vgc2023regc",
                server_configuration=server_config,
                account_configuration=AccountConfiguration(f"BIO{player_id}", None),
                max_concurrent_battles=1,
                team=team,
            )
            players.append(p)
            opponents.append(o)
            player_id += 1

    # Start inference loops
    for p in players:
        p.start_inference_loop()
    for o in opponents:
        o.start_inference_loop()

    # Run battles
    start_time = time.time()
    battles_completed = 0
    errors = 0

    total_pairs = len(players)
    battles_per_pair = (num_battles + total_pairs - 1) // total_pairs

    print(f"Running {battles_per_pair} battles per pair ({total_pairs} pairs)...")

    async def run_pair_battles(player, opponent, n):
        nonlocal battles_completed, errors
        try:
            await player.battle_against(opponent, n_battles=n)
            battles_completed += n
        except Exception as e:
            errors += 1
            print(f"Error: {e}")

    tasks = []
    for p, o in zip(players, opponents):
        tasks.append(run_pair_battles(p, o, battles_per_pair))

    await asyncio.gather(*tasks, return_exceptions=True)

    duration = time.time() - start_time

    # Get resource usage
    process = psutil.Process()
    ram_mb = process.memory_info().rss / 1024 / 1024

    gpu_util = 0
    gpu_mem = 0
    if torch.cuda.is_available():
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                util, mem = result.stdout.strip().split(",")
                gpu_util = float(util)
                gpu_mem = float(mem)
        except Exception:
            pass

    result = BenchmarkResult(
        name=f"BatchInferencePlayer (bs={batch_size})",
        battles_completed=battles_completed,
        duration_seconds=duration,
        battles_per_second=battles_completed / duration if duration > 0 else 0,
        avg_gpu_utilization=gpu_util,
        avg_gpu_memory_mb=gpu_mem,
        avg_ram_usage_mb=ram_mb,
        peak_ram_usage_mb=ram_mb,
        errors=errors,
    )

    print("\nResults:")
    print(f"  Battles: {result.battles_completed}")
    print(f"  Duration: {result.duration_seconds:.1f}s")
    print(f"  Throughput: {result.battles_per_second:.2f} battles/s")
    print(f"  GPU Util: {result.avg_gpu_utilization:.0f}%")
    print(f"  GPU Mem: {result.avg_gpu_memory_mb:.0f} MB")
    print(f"  RAM: {result.avg_ram_usage_mb:.0f} MB")
    print(f"  Errors: {result.errors}")

    # Cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def save_results(results: List[BenchmarkResult], output_file: str):
    """Append results to markdown file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Check if file exists and has content
    file_exists = os.path.exists(output_file) and os.path.getsize(output_file) > 0

    with open(output_file, "a") as f:
        if not file_exists:
            f.write("# RL Training Optimizations V2\n\n")
            f.write("This document contains benchmark results for optimizing RL training throughput.\n\n")

        f.write(f"## Benchmark Run - {timestamp}\n\n")

        # Results table
        f.write("| Test | Battles | Duration | Battles/s | GPU% | GPU MB | RAM MB | Errors |\n")
        f.write("|------|---------|----------|-----------|------|--------|--------|--------|\n")

        for r in results:
            f.write(
                f"| {r.name} | {r.battles_completed} | {r.duration_seconds:.1f}s | "
                f"{r.battles_per_second:.2f} | {r.avg_gpu_utilization:.0f} | "
                f"{r.avg_gpu_memory_mb:.0f} | {r.avg_ram_usage_mb:.0f} | {r.errors} |\n"
            )

        f.write("\n")

        # Summary
        if results:
            best = max(results, key=lambda r: r.battles_per_second)
            f.write(f"**Best:** {best.name} at {best.battles_per_second:.2f} battles/s\n\n")


async def main():
    parser = argparse.ArgumentParser(description="Benchmark RL training throughput")
    parser.add_argument("--test", type=str, default="all",
                       choices=["random", "bc_cpu", "bc_gpu", "batch", "all", "scaling", "batch_sizes"],
                       help="Which test to run")
    parser.add_argument("--battles", type=int, default=100,
                       help="Number of battles to run per test")
    parser.add_argument("--servers", type=int, default=2,
                       help="Number of Showdown servers")
    parser.add_argument("--players", type=int, default=2,
                       help="Players per server")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Batch size for BatchInferencePlayer")
    parser.add_argument("--model", type=str,
                       default="data/models/supervised/dauntless-hill-95.pt",
                       help="Path to model checkpoint")
    parser.add_argument("--output", type=str,
                       default="src/elitefurretai/rl/OPTIMIZATIONS_V2.md",
                       help="Output markdown file")
    args = parser.parse_args()

    # Check model exists
    if args.test in ["bc_cpu", "bc_gpu", "batch", "all"]:
        if not os.path.exists(args.model):
            print(f"ERROR: Model not found at {args.model}")
            sys.exit(1)

    # Start servers
    servers = start_showdown_servers(args.servers, 8000)

    try:
        results = []

        if args.test in ["random", "all"]:
            r = await run_random_vs_random(
                args.battles, args.servers, args.players, 8000
            )
            results.append(r)
            # Brief pause between tests
            await asyncio.sleep(2)

        if args.test in ["bc_cpu", "all"]:
            r = await run_bc_vs_bc(
                args.battles, args.model, "cpu", args.servers, args.players, 8000
            )
            results.append(r)
            await asyncio.sleep(2)

        if args.test in ["bc_gpu", "all"]:
            if torch.cuda.is_available():
                r = await run_bc_vs_bc(
                    args.battles, args.model, "cuda", args.servers, args.players, 8000
                )
                results.append(r)
                await asyncio.sleep(2)
            else:
                print("CUDA not available, skipping bc_gpu test")

        if args.test in ["batch", "all"]:
            r = await run_batch_inference(
                args.battles, args.model, args.batch_size, args.servers, args.players, 8000
            )
            results.append(r)

        # Test scaling: vary number of concurrent players
        if args.test == "scaling":
            print("\n" + "="*60)
            print("SCALING TEST: Varying concurrent players")
            print("="*60)
            for players_per_server in [1, 2, 4, 8]:
                r = await run_batch_inference(
                    args.battles, args.model, 8, args.servers, players_per_server, 8000
                )
                results.append(r)
                await asyncio.sleep(2)

        # Test batch sizes
        if args.test == "batch_sizes":
            print("\n" + "="*60)
            print("BATCH SIZE TEST: Varying batch sizes")
            print("="*60)
            for batch_size in [1, 2, 4, 8, 16, 32]:
                r = await run_batch_inference(
                    args.battles, args.model, batch_size, args.servers, args.players, 8000
                )
                results.append(r)
                await asyncio.sleep(2)

        # Save results
        save_results(results, args.output)
        print(f"\nResults saved to {args.output}")

    finally:
        stop_showdown_servers(servers)


if __name__ == "__main__":
    asyncio.run(main())
