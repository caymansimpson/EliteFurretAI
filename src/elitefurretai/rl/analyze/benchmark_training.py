"""
Comprehensive benchmarking script for RL training optimization.

This script tests different configurations to find the optimal setup for:
- Battle throughput (battles/second)
- GPU utilization
- Memory usage
- Time per training step

Usage:
    python src/elitefurretai/rl/benchmark_training.py
"""

import asyncio
import gc
import json
import os
import subprocess
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import psutil
import torch
from poke_env.player import RandomPlayer
from poke_env.ps_client import AccountConfiguration, ServerConfiguration

from elitefurretai.rl.players import BatchInferencePlayer, RNaDAgent
from elitefurretai.rl.train import load_model
from elitefurretai.supervised.behavior_clone_player import BCPlayer


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark run."""

    num_showdown_servers: int
    max_players_per_server: int
    num_workers: int
    players_per_worker: int
    batch_size: int
    device: str  # 'cuda' or 'cpu'
    player_type: str  # 'random', 'bc', 'batch_inference'
    opponent_type: str  # 'random', 'bc', 'batch_inference'

    def __str__(self) -> str:
        return (
            f"servers={self.num_showdown_servers}, "
            f"pps={self.max_players_per_server}, "
            f"workers={self.num_workers}, "
            f"ppw={self.players_per_worker}, "
            f"bs={self.batch_size}, "
            f"device={self.device}, "
            f"p1={self.player_type}, "
            f"p2={self.opponent_type}"
        )


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    config: BenchmarkConfig
    battles_completed: int
    duration_seconds: float
    battles_per_second: float
    avg_gpu_utilization: float  # 0-100
    avg_gpu_memory_mb: float
    avg_ram_usage_mb: float
    peak_ram_usage_mb: float
    avg_cpu_percent: float
    errors: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": asdict(self.config),
            "battles_completed": self.battles_completed,
            "duration_seconds": self.duration_seconds,
            "battles_per_second": self.battles_per_second,
            "avg_gpu_utilization": self.avg_gpu_utilization,
            "avg_gpu_memory_mb": self.avg_gpu_memory_mb,
            "avg_ram_usage_mb": self.avg_ram_usage_mb,
            "peak_ram_usage_mb": self.peak_ram_usage_mb,
            "avg_cpu_percent": self.avg_cpu_percent,
            "errors": self.errors,
        }


import socket


class ShowdownServerManager:
    """Manages spawning and cleanup of local Showdown servers."""

    def __init__(self, num_servers: int, start_port: int = 8000):
        self.num_servers = num_servers
        self.start_port = start_port
        self.processes: List[subprocess.Popen] = []

    def _wait_for_port(self, port: int, timeout: float = 30) -> bool:
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

    def start_servers(self):
        """Spawn Showdown servers."""
        print(f"Starting {self.num_servers} Showdown servers...")
        for i in range(self.num_servers):
            port = self.start_port + i
            # Spawn server in background
            proc = subprocess.Popen(
                ["node", "pokemon-showdown", "start", "--no-security", "--port", str(port)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=os.path.expanduser("~/Repositories/pokemon-showdown"),
            )
            self.processes.append(proc)

        # Wait for each server to actually be listening
        for i in range(self.num_servers):
            port = self.start_port + i
            if self._wait_for_port(port):
                print(f"  Server on port {port} is ready")
            else:
                print(f"  WARNING: Server on port {port} may not be ready")

        print(f"Servers started on ports {self.start_port}-{self.start_port + self.num_servers - 1}")

    def stop_servers(self):
        """Stop all Showdown servers."""
        print("Stopping Showdown servers...")
        for proc in self.processes:
            proc.terminate()
        # Wait for clean shutdown
        time.sleep(2)
        # Force kill if still running
        for proc in self.processes:
            if proc.poll() is None:
                proc.kill()
        self.processes.clear()
        print("Servers stopped.")


class ResourceMonitor:
    """Monitors GPU, RAM, and CPU usage during benchmarking."""

    def __init__(self):
        self.gpu_utils: List[float] = []
        self.gpu_memories: List[float] = []
        self.ram_usages: List[float] = []
        self.cpu_percents: List[float] = []
        self.monitoring = False
        self.monitor_task: Optional[asyncio.Task] = None

    async def start_monitoring(self):
        """Start monitoring resources."""
        self.monitoring = True
        self.monitor_task = asyncio.create_task(self._monitor_loop())

    async def stop_monitoring(self):
        """Stop monitoring resources."""
        self.monitoring = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass

    async def _monitor_loop(self):
        """Periodically sample resource usage."""
        while self.monitoring:
            try:
                # GPU stats (if CUDA available)
                if torch.cuda.is_available():
                    # nvidia-smi based monitoring
                    result = subprocess.run(
                        [
                            "nvidia-smi",
                            "--query-gpu=utilization.gpu,memory.used",
                            "--format=csv,noheader,nounits",
                        ],
                        capture_output=True,
                        text=True,
                    )
                    if result.returncode == 0:
                        lines = result.stdout.strip().split("\n")
                        for line in lines:
                            util, mem = line.split(",")
                            self.gpu_utils.append(float(util.strip()))
                            self.gpu_memories.append(float(mem.strip()))

                # RAM usage
                process = psutil.Process()
                mem_info = process.memory_info()
                self.ram_usages.append(mem_info.rss / 1024 / 1024)  # MB

                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=None)
                self.cpu_percents.append(cpu_percent)

                await asyncio.sleep(1)  # Sample every second
            except Exception as e:
                print(f"Monitor error: {e}")
                await asyncio.sleep(1)

    def get_stats(self) -> Dict[str, float]:
        """Get average statistics."""
        return {
            "avg_gpu_util": np.mean(self.gpu_utils) if self.gpu_utils else 0.0,
            "avg_gpu_mem_mb": np.mean(self.gpu_memories) if self.gpu_memories else 0.0,
            "avg_ram_mb": np.mean(self.ram_usages) if self.ram_usages else 0.0,
            "peak_ram_mb": max(self.ram_usages) if self.ram_usages else 0.0,
            "avg_cpu_percent": np.mean(self.cpu_percents) if self.cpu_percents else 0.0,
        }


class BattleBenchmark:
    """Runs battle simulations to benchmark throughput."""

    def __init__(
        self,
        config: BenchmarkConfig,
        model_path: str,
        battle_format: str = "gen9vgc2023regc",
        team_path: str = "data/teams",
    ):
        self.config = config
        self.model_path = model_path
        self.battle_format = battle_format
        self.team_path = team_path
        self.battles_completed = 0
        self.errors: List[str] = []

        # Load model once for reuse
        self.model = None
        self.rnad_agent = None
        if config.player_type in ["bc", "batch_inference"] or config.opponent_type in [
            "bc",
            "batch_inference",
        ]:
            self._load_model()

    def _load_model(self):
        """Load the BC model for reuse."""
        print(f"Loading model from {self.model_path}...")
        if self.config.device == "cuda" and torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        # For BatchInferencePlayer, we need RNaDAgent
        if self.config.player_type == "batch_inference" or self.config.opponent_type == "batch_inference":
            # Use the existing load_model function from train.py
            base_model = load_model(self.model_path, device)
            self.rnad_agent = RNaDAgent(base_model)
            self.rnad_agent.to(device)
            self.rnad_agent.eval()
            print(f"RNaDAgent loaded on {device}")

    def _create_player(self, player_type: str, player_id: str, server_idx: int):
        """Create a player of the specified type."""
        server_config = ServerConfiguration(
            f"ws://localhost:{8000 + server_idx}/showdown/websocket",
            None,  # type: ignore[arg-type]
        )
        account_config = AccountConfiguration(player_id, None)

        if player_type == "random":
            return RandomPlayer(
                battle_format=self.battle_format,
                team=self._get_random_team(),
                server_configuration=server_config,
                account_configuration=account_config,
                max_concurrent_battles=self.config.max_players_per_server,
            )
        elif player_type == "bc":
            return BCPlayer(
                unified_model_filepath=self.model_path,
                battle_format=self.battle_format,
                team=self._get_random_team(),
                server_configuration=server_config,
                account_configuration=account_config,
                device=self.config.device,
                probabilistic=True,
                max_concurrent_battles=self.config.max_players_per_server,
            )
        elif player_type == "batch_inference":
            return BatchInferencePlayer(
                model=self.rnad_agent,
                device=self.config.device,
                batch_size=self.config.batch_size,
                probabilistic=True,
                battle_format=self.battle_format,
                team=self._get_random_team(),
                server_configuration=server_config,
                account_configuration=account_config,
                max_concurrent_battles=self.config.max_players_per_server,
            )
        else:
            raise ValueError(f"Unknown player type: {player_type}")

    def _get_random_team(self) -> Optional[str]:
        """Get a random team from the team pool."""
        # For now, return None to use random teams
        # TODO: Load from team_path if needed
        return None

    async def run_battles(self, duration_seconds: float = 300) -> BenchmarkResult:
        """Run battles for the specified duration."""
        print(f"\n{'='*80}")
        print(f"Running benchmark: {self.config}")
        print(f"Duration: {duration_seconds}s")
        print(f"{'='*80}")

        # Start resource monitoring
        monitor = ResourceMonitor()
        await monitor.start_monitoring()

        start_time = time.time()
        self.battles_completed = 0
        self.errors.clear()

        try:
            # Create players distributed across servers
            players = []
            opponents = []

            # Distribute players across servers
            for worker_idx in range(self.config.num_workers):
                server_idx = worker_idx % self.config.num_showdown_servers

                for pair_idx in range(self.config.players_per_worker):
                    player_id = f"p{worker_idx}_{pair_idx}"
                    opponent_id = f"o{worker_idx}_{pair_idx}"

                    try:
                        player = self._create_player(self.config.player_type, player_id, server_idx)
                        opponent = self._create_player(
                            self.config.opponent_type, opponent_id, server_idx
                        )
                        players.append(player)
                        opponents.append(opponent)
                    except Exception as e:
                        self.errors.append(f"Player creation error: {e}")
                        continue

            # Start inference loops for BatchInferencePlayers
            for player in players:
                if isinstance(player, BatchInferencePlayer):
                    player.start_inference_loop()
            for opponent in opponents:
                if isinstance(opponent, BatchInferencePlayer):
                    opponent.start_inference_loop()

            # Run battles until time limit
            battle_tasks = []
            while time.time() - start_time < duration_seconds:
                # Start new battles
                for player, opponent in zip(players, opponents):
                    task = asyncio.create_task(
                        player.battle_against(opponent, n_battles=1)
                    )
                    battle_tasks.append(task)

                # Wait for at least one battle to complete
                if battle_tasks:
                    done, pending = await asyncio.wait(
                        battle_tasks, return_when=asyncio.FIRST_COMPLETED, timeout=1.0
                    )
                    for task in done:
                        try:
                            await task
                            self.battles_completed += 1
                        except Exception as e:
                            self.errors.append(f"Battle error: {e}")
                    battle_tasks = list(pending)

            # Wait for remaining battles
            if battle_tasks:
                results = await asyncio.gather(*battle_tasks, return_exceptions=True)
                for result in results:
                    if isinstance(result, Exception):
                        self.errors.append(f"Battle error: {result}")
                    else:
                        self.battles_completed += 1

            # Cleanup players
            for player in players:
                if hasattr(player, "stop_listening"):
                    await player.stop_listening()
            for opponent in opponents:
                if hasattr(opponent, "stop_listening"):
                    await opponent.stop_listening()

        except Exception as e:
            self.errors.append(f"Critical error: {e}")
        finally:
            await monitor.stop_monitoring()

        # Calculate metrics
        end_time = time.time()
        duration = end_time - start_time
        battles_per_second = self.battles_completed / duration if duration > 0 else 0

        stats = monitor.get_stats()

        result = BenchmarkResult(
            config=self.config,
            battles_completed=self.battles_completed,
            duration_seconds=duration,
            battles_per_second=battles_per_second,
            avg_gpu_utilization=stats["avg_gpu_util"],
            avg_gpu_memory_mb=stats["avg_gpu_mem_mb"],
            avg_ram_usage_mb=stats["avg_ram_mb"],
            peak_ram_usage_mb=stats["peak_ram_mb"],
            avg_cpu_percent=stats["avg_cpu_percent"],
            errors=self.errors[:10],  # Keep only first 10 errors
        )

        print("\nResults:")
        print(f"  Battles completed: {result.battles_completed}")
        print(f"  Duration: {result.duration_seconds:.1f}s")
        print(f"  Battles/sec: {result.battles_per_second:.2f}")
        print(f"  Avg GPU util: {result.avg_gpu_utilization:.1f}%")
        print(f"  Avg GPU mem: {result.avg_gpu_memory_mb:.1f} MB")
        print(f"  Avg RAM: {result.avg_ram_usage_mb:.1f} MB")
        print(f"  Peak RAM: {result.peak_ram_usage_mb:.1f} MB")
        print(f"  Avg CPU: {result.avg_cpu_percent:.1f}%")
        if self.errors:
            print(f"  Errors: {len(self.errors)}")

        # Cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result


def generate_benchmark_configs() -> List[BenchmarkConfig]:
    """Generate all benchmark configurations to test."""
    configs = []

    # Test 1: RandomPlayer baseline (cheapest, fastest)
    configs.append(
        BenchmarkConfig(
            num_showdown_servers=2,
            max_players_per_server=4,
            num_workers=4,
            players_per_worker=2,
            batch_size=8,
            device="cpu",
            player_type="random",
            opponent_type="random",
        )
    )

    # Test 2: BCPlayer vs BCPlayer on CPU
    configs.append(
        BenchmarkConfig(
            num_showdown_servers=2,
            max_players_per_server=2,
            num_workers=2,
            players_per_worker=2,
            batch_size=8,
            device="cpu",
            player_type="bc",
            opponent_type="bc",
        )
    )

    # Test 3: BCPlayer vs BCPlayer on GPU
    configs.append(
        BenchmarkConfig(
            num_showdown_servers=2,
            max_players_per_server=2,
            num_workers=2,
            players_per_worker=2,
            batch_size=8,
            device="cuda",
            player_type="bc",
            opponent_type="bc",
        )
    )

    # Test 4: BatchInferencePlayer vs BatchInferencePlayer on GPU (small batch)
    configs.append(
        BenchmarkConfig(
            num_showdown_servers=2,
            max_players_per_server=2,
            num_workers=2,
            players_per_worker=2,
            batch_size=4,
            device="cuda",
            player_type="batch_inference",
            opponent_type="batch_inference",
        )
    )

    # Test 5: BatchInferencePlayer vs BatchInferencePlayer on GPU (medium batch)
    configs.append(
        BenchmarkConfig(
            num_showdown_servers=2,
            max_players_per_server=2,
            num_workers=2,
            players_per_worker=2,
            batch_size=8,
            device="cuda",
            player_type="batch_inference",
            opponent_type="batch_inference",
        )
    )

    # Test 6: BatchInferencePlayer vs BatchInferencePlayer on GPU (large batch)
    configs.append(
        BenchmarkConfig(
            num_showdown_servers=2,
            max_players_per_server=2,
            num_workers=2,
            players_per_worker=2,
            batch_size=16,
            device="cuda",
            player_type="batch_inference",
            opponent_type="batch_inference",
        )
    )

    # Test 7: More servers, more players
    configs.append(
        BenchmarkConfig(
            num_showdown_servers=4,
            max_players_per_server=2,
            num_workers=4,
            players_per_worker=2,
            batch_size=8,
            device="cuda",
            player_type="batch_inference",
            opponent_type="batch_inference",
        )
    )

    # Test 8: More workers per server
    configs.append(
        BenchmarkConfig(
            num_showdown_servers=2,
            max_players_per_server=4,
            num_workers=4,
            players_per_worker=2,
            batch_size=8,
            device="cuda",
            player_type="batch_inference",
            opponent_type="batch_inference",
        )
    )

    # Test 9: Max parallelism push
    configs.append(
        BenchmarkConfig(
            num_showdown_servers=4,
            max_players_per_server=4,
            num_workers=8,
            players_per_worker=2,
            batch_size=16,
            device="cuda",
            player_type="batch_inference",
            opponent_type="batch_inference",
        )
    )

    return configs


async def main():
    """Run all benchmarks."""
    # Configuration
    model_path = "data/models/supervised/dauntless-hill-95.pt"
    duration_per_test = 300  # 5 minutes
    output_file = "src/elitefurretai/rl/OPTIMIZATIONS_V2.md"
    results_json = "benchmark_results.json"

    # Check if model exists
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        return

    # Generate benchmark configs
    configs = generate_benchmark_configs()
    print(f"Generated {len(configs)} benchmark configurations")

    # Start Showdown servers (use max needed)
    max_servers = max(c.num_showdown_servers for c in configs)
    server_manager = ShowdownServerManager(max_servers, start_port=8000)
    server_manager.start_servers()

    try:
        # Run benchmarks
        results = []
        for i, config in enumerate(configs):
            print(f"\n\n[{i+1}/{len(configs)}] Starting benchmark...")
            benchmark = BattleBenchmark(config, model_path)
            result = await benchmark.run_battles(duration_per_test)
            results.append(result)

            # Brief pause between tests
            print("Cooling down...")
            await asyncio.sleep(5)

        # Save results to JSON
        with open(results_json, "w") as f:
            json.dump([r.to_dict() for r in results], f, indent=2)
        print(f"\nResults saved to {results_json}")

        # Generate markdown report
        generate_report(results, output_file)
        print(f"Report saved to {output_file}")

    finally:
        # Cleanup
        server_manager.stop_servers()


def generate_report(results: List[BenchmarkResult], output_file: str):
    """Generate markdown report from benchmark results."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(output_file, "w") as f:
        f.write("# RL Training Optimizations V2\n\n")
        f.write(f"**Generated:** {timestamp}\n\n")
        f.write("## Summary\n\n")
        f.write(
            "This document contains comprehensive benchmark results for optimizing RL training throughput.\n\n"
        )

        # Find best config
        best_result = max(results, key=lambda r: r.battles_per_second)
        f.write("### Best Configuration\n\n")
        f.write(f"- **Battles/sec:** {best_result.battles_per_second:.2f}\n")
        f.write(f"- **Config:** {best_result.config}\n")
        f.write(f"- **GPU Utilization:** {best_result.avg_gpu_utilization:.1f}%\n")
        f.write(f"- **GPU Memory:** {best_result.avg_gpu_memory_mb:.1f} MB\n")
        f.write(f"- **RAM Usage:** {best_result.avg_ram_usage_mb:.1f} MB\n\n")

        # Detailed results table
        f.write("## Detailed Results\n\n")
        f.write("| Player | Opponent | Servers | PPS | Workers | PPW | Batch | Device | Battles/s | GPU% | GPU MB | RAM MB | CPU% |\n")
        f.write("|--------|----------|---------|-----|---------|-----|-------|--------|-----------|------|--------|--------|------|\n")

        for r in results:
            c = r.config
            f.write(
                f"| {c.player_type} | {c.opponent_type} | {c.num_showdown_servers} | "
                f"{c.max_players_per_server} | {c.num_workers} | {c.players_per_worker} | "
                f"{c.batch_size} | {c.device} | {r.battles_per_second:.2f} | "
                f"{r.avg_gpu_utilization:.1f} | {r.avg_gpu_memory_mb:.0f} | "
                f"{r.avg_ram_usage_mb:.0f} | {r.avg_cpu_percent:.1f} |\n"
            )

        # Key findings
        f.write("\n## Key Findings\n\n")

        # Compare player types
        random_results = [r for r in results if r.config.player_type == "random"]
        bc_results = [r for r in results if r.config.player_type == "bc"]
        batch_results = [r for r in results if r.config.player_type == "batch_inference"]

        if random_results:
            avg_random = np.mean([r.battles_per_second for r in random_results])
            f.write("### RandomPlayer Baseline\n")
            f.write(f"- Average throughput: {avg_random:.2f} battles/s\n\n")

        if bc_results:
            avg_bc = np.mean([r.battles_per_second for r in bc_results])
            f.write("### BCPlayer\n")
            f.write(f"- Average throughput: {avg_bc:.2f} battles/s\n")
            if random_results:
                f.write(f"- Overhead vs Random: {(1 - avg_bc/avg_random)*100:.1f}%\n")

            # CPU vs GPU comparison
            bc_cpu = [r for r in bc_results if r.config.device == "cpu"]
            bc_gpu = [r for r in bc_results if r.config.device == "cuda"]
            if bc_cpu and bc_gpu:
                cpu_avg = np.mean([r.battles_per_second for r in bc_cpu])
                gpu_avg = np.mean([r.battles_per_second for r in bc_gpu])
                f.write(f"- CPU throughput: {cpu_avg:.2f} battles/s\n")
                f.write(f"- GPU throughput: {gpu_avg:.2f} battles/s\n")
                f.write(f"- GPU speedup: {gpu_avg/cpu_avg:.2f}x\n")
            f.write("\n")

        if batch_results:
            avg_batch = np.mean([r.battles_per_second for r in batch_results])
            f.write("### BatchInferencePlayer\n")
            f.write(f"- Average throughput: {avg_batch:.2f} battles/s\n")
            if bc_results:
                f.write(f"- Speedup vs BCPlayer: {avg_batch/avg_bc:.2f}x\n")

            # Batch size comparison
            batch_sizes = {}
            for r in batch_results:
                bs = r.config.batch_size
                if bs not in batch_sizes:
                    batch_sizes[bs] = []
                batch_sizes[bs].append(r.battles_per_second)

            f.write("\n**Batch Size Impact:**\n")
            for bs in sorted(batch_sizes.keys()):
                avg = np.mean(batch_sizes[bs])
                f.write(f"- Batch size {bs}: {avg:.2f} battles/s\n")
            f.write("\n")

        # Server scaling
        f.write("### Server Scaling\n")
        server_counts = defaultdict(list)
        for r in results:
            server_counts[r.config.num_showdown_servers].append(r.battles_per_second)
        for count in sorted(server_counts.keys()):
            avg = np.mean(server_counts[count])
            f.write(f"- {count} servers: {avg:.2f} battles/s\n")
        f.write("\n")

        # Resource utilization
        f.write("### Resource Utilization\n")
        f.write(f"- Peak RAM usage: {max(r.peak_ram_usage_mb for r in results):.0f} MB\n")
        if torch.cuda.is_available():
            f.write(f"- Peak GPU memory: {max(r.avg_gpu_memory_mb for r in results):.0f} MB\n")
            f.write(f"- Max GPU utilization: {max(r.avg_gpu_utilization for r in results):.1f}%\n")
        f.write("\n")

        # Recommendations
        f.write("## Recommendations\n\n")
        f.write("Based on the benchmark results:\n\n")
        f.write(f"1. **Best overall config:** {best_result.config}\n")
        f.write(f"2. **Expected throughput:** {best_result.battles_per_second:.2f} battles/s\n")

        if batch_results and bc_results:
            if avg_batch > avg_bc:
                f.write(f"3. **Use BatchInferencePlayer** - {avg_batch/avg_bc:.2f}x faster than BCPlayer\n")
            else:
                f.write("3. **Use BCPlayer** - BatchInferencePlayer shows no improvement\n")

        f.write("\n")


if __name__ == "__main__":
    asyncio.run(main())
