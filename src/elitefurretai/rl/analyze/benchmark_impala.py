#!/usr/bin/env python3
"""
Benchmark script for IMPALA-style multiprocessing architecture.

This script empirically tests different configurations to find optimal
throughput for the new IMPALA architecture.

Usage:
    python src/elitefurretai/rl/analyze/benchmark_impala.py --quick
    python src/elitefurretai/rl/analyze/benchmark_impala.py --full
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import psutil
import torch
import torch.multiprocessing as mp

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

from elitefurretai.etl.embedder import Embedder
from elitefurretai.rl.multiprocess_actor import (
    ActorConfig,
    actor_process,
)


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    name: str
    num_actors: int
    num_servers: int
    actors_per_server: int
    duration_seconds: float
    battles_completed: int
    battles_per_hour: float
    trajectories_received: int
    avg_ram_gb: float
    peak_ram_gb: float
    errors: int
    notes: str = ""


def get_model_config(model_path: str) -> Dict:
    """Load model config from checkpoint."""
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    full_config = checkpoint.get("config", {})

    embedder = Embedder(
        format="gen9vgc2023regc",
        feature_set=full_config.get("embedder_feature_set", "full"),
        omniscient=False
    )

    return {
        "input_size": embedder.embedding_size,
        "group_sizes": embedder.group_embedding_sizes if full_config.get("use_grouped_encoder", True) else None,
        "dropout": full_config.get("dropout", 0.1),
        "gated_residuals": full_config.get("gated_residuals", False),
        "use_grouped_encoder": full_config.get("use_grouped_encoder", True),
        "grouped_encoder_hidden_dim": full_config.get("grouped_encoder_hidden_dim", 512),
        "grouped_encoder_aggregated_dim": full_config.get("grouped_encoder_aggregated_dim", 4096),
        "pokemon_attention_heads": full_config.get("pokemon_attention_heads", 16),
        "early_layers": full_config.get("early_layers", [4096, 2048, 2048, 1024]),
        "early_attention_heads": full_config.get("early_attention_heads", 16),
        "lstm_layers": full_config.get("lstm_layers", 4),
        "lstm_hidden_size": full_config.get("lstm_hidden_size", 512),
        "late_layers": full_config.get("late_layers", [2048, 2048, 1024, 1024]),
        "late_attention_heads": full_config.get("late_attention_heads", 32),
        "teampreview_head_layers": full_config.get("teampreview_head_layers", [512, 256]),
        "teampreview_head_dropout": full_config.get("teampreview_head_dropout", 0.3),
        "teampreview_attention_heads": full_config.get("teampreview_attention_heads", 8),
        "turn_head_layers": full_config.get("turn_head_layers", [2048, 1024, 1024, 1024]),
        "max_seq_len": full_config.get("max_seq_len", 40),
        "num_actions": 2025,  # MDBO.action_space()
        "num_teampreview_actions": 90,  # MDBO.teampreview_space()
    }


def start_showdown_servers(num_servers: int, start_port: int = 8000) -> List[subprocess.Popen]:
    """Start Pokemon Showdown servers."""
    processes = []
    showdown_path = os.path.expanduser("~/Repositories/pokemon-showdown")

    print(f"\nStarting {num_servers} Showdown servers...")
    for i in range(num_servers):
        port = start_port + i
        proc = subprocess.Popen(
            ["node", "pokemon-showdown", "start", "--no-security", "--port", str(port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=showdown_path,
            preexec_fn=os.setsid if os.name != "nt" else None,
        )
        processes.append(proc)
        print(f"  ✓ Started server on port {port} (PID {proc.pid})")
        time.sleep(0.3)

    print("Waiting for servers to initialize...")
    time.sleep(2)
    return processes


def stop_showdown_servers(processes: List[subprocess.Popen]):
    """Stop Showdown servers."""
    for proc in processes:
        if proc.poll() is None:
            try:
                if os.name != "nt":
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                else:
                    proc.terminate()
                proc.wait(timeout=3)
            except Exception:
                try:
                    if os.name != "nt":
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    else:
                        proc.kill()
                except Exception:
                    pass


def run_impala_benchmark(
    model_path: str,
    model_config: Dict,
    num_actors: int,
    num_servers: int,
    duration_seconds: float = 60.0,
    battles_per_actor: int = 100,
    start_port: int = 8000,
    team_path: Optional[str] = None,
) -> BenchmarkResult:
    """
    Run benchmark with IMPALA-style multiprocessing.

    Args:
        model_path: Path to model checkpoint
        model_config: Model configuration dict
        num_actors: Number of actor processes
        num_servers: Number of Showdown servers
        duration_seconds: How long to run benchmark
        battles_per_actor: Max battles per actor
        start_port: Starting port for servers
        team_path: Path to team file or directory
    """
    actors_per_server = num_actors / num_servers if num_servers > 0 else num_actors
    config_name = f"{num_actors}actors_{num_servers}servers"

    print(f"\n{'='*60}")
    print(f"BENCHMARK: {config_name}")
    print(f"  Actors: {num_actors}, Servers: {num_servers}")
    print(f"  Actors per server: {actors_per_server:.1f}")
    print(f"{'='*60}")

    # Start servers
    server_processes = start_showdown_servers(num_servers, start_port)

    # Set up multiprocessing
    mp.set_start_method("spawn", force=True)
    trajectory_queue = mp.Queue(maxsize=1000)
    weight_queues = [mp.Queue(maxsize=10) for _ in range(num_actors)]
    stop_event = mp.Event()

    # Distribute actors across servers
    server_ports = []
    for i in range(num_actors):
        server_idx = i % num_servers
        server_ports.append(start_port + server_idx)

    # Start actors
    actors = []
    for i in range(num_actors):
        config = ActorConfig(
            actor_id=i,
            server_port=server_ports[i],
            model_path=model_path,
            model_config=model_config,
            battle_format="gen9vgc2023regc",
            num_battles=battles_per_actor,
            device="cpu",  # IMPALA actors use CPU
            team_path=team_path,
        )
        p = mp.Process(
            target=actor_process,
            args=(config, trajectory_queue, weight_queues[i], stop_event),
        )
        p.start()
        actors.append(p)
        print(f"  Started actor {i} (PID {p.pid}) on port {server_ports[i]}")

    # Collect metrics
    start_time = time.time()
    trajectories_received = 0
    ram_samples = []
    errors = 0

    process = psutil.Process()

    print(f"\nRunning benchmark for {duration_seconds}s...")

    try:
        while time.time() - start_time < duration_seconds:
            # Collect trajectories
            try:
                while not trajectory_queue.empty():
                    trajectory_queue.get_nowait()
                    trajectories_received += 1
            except Exception:
                pass

            # Sample RAM
            try:
                ram_gb = process.memory_info().rss / (1024**3)
                # Add child processes
                for child in process.children(recursive=True):
                    try:
                        ram_gb += child.memory_info().rss / (1024**3)
                    except Exception:
                        pass
                ram_samples.append(ram_gb)
            except Exception:
                pass

            # Check actor health
            dead_actors = [p for p in actors if not p.is_alive()]
            if len(dead_actors) == num_actors:
                print("All actors finished or died")
                break

            time.sleep(0.5)

            # Progress update
            elapsed = time.time() - start_time
            rate = trajectories_received / (elapsed / 3600) if elapsed > 0 else 0
            print(f"  {elapsed:.0f}s: {trajectories_received} trajectories ({rate:.0f}/hr)", end="\r")

    except KeyboardInterrupt:
        print("\nInterrupted!")

    finally:
        # Stop actors
        stop_event.set()
        for p in actors:
            p.join(timeout=3)
            if p.is_alive():
                p.terminate()

        # Drain remaining trajectories
        try:
            while not trajectory_queue.empty():
                trajectory_queue.get_nowait()
                trajectories_received += 1
        except Exception:
            pass

        # Stop servers
        stop_showdown_servers(server_processes)

    # Calculate results
    duration = time.time() - start_time
    battles_per_hour = (trajectories_received / duration) * 3600 if duration > 0 else 0
    avg_ram = sum(ram_samples) / len(ram_samples) if ram_samples else 0
    peak_ram = max(ram_samples) if ram_samples else 0

    result = BenchmarkResult(
        name=config_name,
        num_actors=num_actors,
        num_servers=num_servers,
        actors_per_server=actors_per_server,
        duration_seconds=duration,
        battles_completed=trajectories_received,  # Each trajectory = 1 battle
        battles_per_hour=battles_per_hour,
        trajectories_received=trajectories_received,
        avg_ram_gb=avg_ram,
        peak_ram_gb=peak_ram,
        errors=errors,
    )

    print(f"\n\nResults for {config_name}:")
    print(f"  Duration: {duration:.1f}s")
    print(f"  Battles: {trajectories_received}")
    print(f"  Rate: {battles_per_hour:.0f} battles/hr")
    print(f"  RAM: {avg_ram:.2f} GB avg, {peak_ram:.2f} GB peak")

    return result


def run_full_benchmark(model_path: str, duration: float = 60.0, team_path: str = None):
    """Run comprehensive benchmark across configurations."""
    model_config = get_model_config(model_path)

    # Configurations to test: (num_actors, num_servers)
    # Based on documented findings, test around the optimal range
    configs = [
        # Baseline
        (1, 1),
        (2, 2),
        # Mid-range (documented best: 4 servers × 4 actors)
        (4, 4),
        (4, 2),
        # Higher scale (documented: 8×2 = ~2750/hr)
        (8, 4),
        (8, 8),
        # Stress test
        (16, 8),
        (12, 6),
    ]

    results = []

    for num_actors, num_servers in configs:
        try:
            result = run_impala_benchmark(
                model_path=model_path,
                model_config=model_config,
                num_actors=num_actors,
                num_servers=num_servers,
                duration_seconds=duration,
                start_port=8000,
                team_path=team_path,
            )
            results.append(result)

            # Cool down between tests
            print("\nCooling down for 5s...")
            time.sleep(5)

        except Exception as e:
            print(f"ERROR with {num_actors} actors, {num_servers} servers: {e}")
            import traceback
            traceback.print_exc()

    return results


def run_quick_benchmark(model_path: str, team_path: str = None):
    """Run quick benchmark with fewer configurations."""
    model_config = get_model_config(model_path)

    # Quick test: just a few key configs
    configs = [
        (2, 2),
        (4, 4),
        (8, 8),
    ]

    results = []

    for num_actors, num_servers in configs:
        try:
            result = run_impala_benchmark(
                model_path=model_path,
                model_config=model_config,
                num_actors=num_actors,
                num_servers=num_servers,
                duration_seconds=30.0,  # Shorter duration for quick test
                start_port=8000,
                team_path=team_path,
            )
            results.append(result)
            time.sleep(3)
        except Exception as e:
            print(f"ERROR: {e}")

    return results


def print_results_table(results: List[BenchmarkResult]):
    """Print results as a formatted table."""
    print("\n" + "="*80)
    print("IMPALA BENCHMARK RESULTS")
    print("="*80)
    print(f"{'Config':<20} {'Actors':>7} {'Servers':>8} {'Rate/hr':>10} {'RAM (GB)':>10} {'Status':<10}")
    print("-"*80)

    # Sort by throughput
    sorted_results = sorted(results, key=lambda r: r.battles_per_hour, reverse=True)

    best_rate = sorted_results[0].battles_per_hour if sorted_results else 0

    for r in sorted_results:
        status = "✓ BEST" if r.battles_per_hour == best_rate else ""
        print(f"{r.name:<20} {r.num_actors:>7} {r.num_servers:>8} "
              f"{r.battles_per_hour:>10.0f} {r.avg_ram_gb:>10.2f} {status:<10}")

    print("-"*80)

    if sorted_results:
        best = sorted_results[0]
        print(f"\n🏆 OPTIMAL CONFIG: {best.num_actors} actors, {best.num_servers} servers")
        print(f"   Throughput: {best.battles_per_hour:.0f} battles/hr")
        print(f"   RAM Usage: {best.avg_ram_gb:.2f} GB avg, {best.peak_ram_gb:.2f} GB peak")


def save_results_to_md(results: List[BenchmarkResult], output_path: str):
    """Save results to markdown file."""
    sorted_results = sorted(results, key=lambda r: r.battles_per_hour, reverse=True)

    with open(output_path, "a") as f:
        f.write(f"\n## IMPALA Benchmark Results - {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write("| Config | Actors | Servers | Act/Srv | Rate/hr | RAM (GB) | Notes |\n")
        f.write("|--------|--------|---------|---------|---------|----------|-------|\n")

        for r in sorted_results:
            f.write(f"| {r.name} | {r.num_actors} | {r.num_servers} | "
                    f"{r.actors_per_server:.1f} | {r.battles_per_hour:.0f} | "
                    f"{r.avg_ram_gb:.2f} | {r.notes} |\n")

        if sorted_results:
            best = sorted_results[0]
            f.write(f"\n**Optimal**: {best.num_actors} actors × {best.num_servers} servers "
                    f"= {best.battles_per_hour:.0f} battles/hr\n")


def main():
    parser = argparse.ArgumentParser(description="Benchmark IMPALA multiprocessing")
    parser.add_argument("--model", default="data/models/supervised/dauntless-hill-95.pt",
                       help="Model checkpoint path")
    parser.add_argument("--team-path", default="data/teams/gen9vgc2023regc/easy",
                       help="Path to team file or directory")
    parser.add_argument("--quick", action="store_true", help="Quick benchmark (30s each)")
    parser.add_argument("--full", action="store_true", help="Full benchmark (60s each)")
    parser.add_argument("--duration", type=float, default=60.0,
                       help="Duration per config (seconds)")
    parser.add_argument("--output", default="src/elitefurretai/rl/RL.md",
                       help="Output markdown file")
    args = parser.parse_args()

    # Set multiprocessing start method for WSL compatibility
    torch.multiprocessing.set_sharing_strategy("file_system")

    print("="*60)
    print("IMPALA MULTIPROCESSING BENCHMARK")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Teams: {args.team_path}")
    print(f"Mode: {'quick' if args.quick else 'full'}")

    if not os.path.exists(args.model):
        print(f"ERROR: Model not found at {args.model}")
        return

    if not os.path.exists(args.team_path):
        print(f"ERROR: Team path not found at {args.team_path}")
        return

    if args.quick:
        results = run_quick_benchmark(args.model, team_path=args.team_path)
    else:
        results = run_full_benchmark(args.model, duration=args.duration,
                                     team_path=args.team_path)

    if results:
        print_results_table(results)
        save_results_to_md(results, args.output)
        print(f"\nResults appended to {args.output}")


if __name__ == "__main__":
    main()
