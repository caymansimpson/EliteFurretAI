"""
Training Profiler and Performance Analyzer

Profiles training pipeline and recommends optimal parameters for faster training.
Analyzes CPU/GPU utilization, data loading bottlenecks, and suggests improvements.
"""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import GPUtil
import numpy as np
import psutil
import torch

from elitefurretai.etl.embedder import Embedder
from elitefurretai.etl.encoder import MDBO
from elitefurretai.rl.agent import RNaDAgent
from elitefurretai.rl.learner import RNaDLearner
from elitefurretai.supervised.behavior_clone_player import FlexibleThreeHeadedModel


@dataclass
class ProfilingResult:
    """Results from profiling a specific configuration."""

    # Configuration
    num_workers: int
    players_per_worker: int
    batch_size: int
    train_batch_size: int
    use_mixed_precision: bool

    # Performance metrics
    battles_per_hour: float
    updates_per_hour: float
    timesteps_per_hour: float

    # Resource utilization
    avg_cpu_percent: float
    peak_cpu_percent: float
    avg_gpu_utilization: float
    peak_gpu_utilization: float
    avg_gpu_memory_mb: float
    peak_gpu_memory_mb: float
    avg_ram_gb: float
    peak_ram_gb: float

    # Timing breakdown
    avg_inference_time_ms: float
    avg_training_time_ms: float
    avg_data_collection_time_ms: float

    # Bottleneck analysis
    primary_bottleneck: str  # "cpu", "gpu", "network", "balanced"
    bottleneck_score: float  # 0-1, higher = more severe bottleneck

    # Recommendations
    recommended_num_workers: Optional[int] = None
    recommended_players_per_worker: Optional[int] = None
    recommended_batch_size: Optional[int] = None
    recommended_train_batch_size: Optional[int] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)

    def save(self, filepath: str):
        """Save results to JSON."""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class TrainingProfiler:
    """
    Profiles training performance and suggests optimal hyperparameters.
    """

    def __init__(self, model_path: str, device: str = "cuda"):
        """
        Args:
            model_path: Path to model checkpoint
            device: Device to profile on
        """
        self.model_path = model_path
        self.device = device
        self.results: List[ProfilingResult] = []

    def _load_model(self) -> RNaDAgent:
        """Load model for profiling."""
        embedder = Embedder(
            format="gen9vgc2023regulationc", feature_set=Embedder.FULL, omniscient=False
        )

        if Path(self.model_path).exists():
            checkpoint = torch.load(self.model_path, map_location=self.device)
            config = checkpoint["config"]
            state_dict = checkpoint["model_state_dict"]
        else:
            # Create dummy model
            config = {
                "early_layers": [512, 512],
                "late_layers": [512],
                "lstm_layers": 2,
                "lstm_hidden_size": 512,
                "dropout": 0.1,
            }
            state_dict = None

        model = FlexibleThreeHeadedModel(
            input_size=embedder.embedding_size,
            early_layers=config["early_layers"],
            late_layers=config["late_layers"],
            lstm_layers=config.get("lstm_layers", 2),
            lstm_hidden_size=config.get("lstm_hidden_size", 512),
            dropout=config.get("dropout", 0.1),
            num_actions=MDBO.action_space(),
            num_teampreview_actions=MDBO.teampreview_space(),
            max_seq_len=17,
        ).to(self.device)

        if state_dict:
            model.load_state_dict(state_dict)

        return RNaDAgent(model)

    def profile_inference(
        self, agent: RNaDAgent, batch_size: int, num_iterations: int = 100
    ) -> Dict[str, float]:
        """
        Profile inference speed with different batch sizes.

        Returns:
            Dict with timing statistics
        """
        embedder = Embedder(
            format="gen9vgc2023regulationc", feature_set=Embedder.FULL, omniscient=False
        )

        # Create dummy batch
        dummy_state = torch.randn(
            batch_size, 1, embedder.embedding_size, device=self.device
        )
        hidden = agent.get_initial_state(batch_size, self.device)

        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = agent.forward(dummy_state, hidden)

        # Profile
        torch.cuda.synchronize() if self.device == "cuda" else None
        times = []

        for _ in range(num_iterations):
            start = time.time()
            with torch.no_grad():
                _ = agent.forward(dummy_state, hidden)
            torch.cuda.synchronize() if self.device == "cuda" else None
            end = time.time()
            times.append((end - start) * 1000)  # Convert to ms

        return {
            "mean_ms": float(np.mean(times)),
            "std_ms": float(np.std(times)),
            "min_ms": float(np.min(times)),
            "max_ms": float(np.max(times)),
            "p50_ms": float(np.percentile(times, 50)),
            "p95_ms": float(np.percentile(times, 95)),
            "throughput_samples_per_sec": batch_size / float(np.mean(times) / 1000),
        }

    def profile_training(
        self,
        agent: RNaDAgent,
        train_batch_size: int,
        num_iterations: int = 50,
        use_mixed_precision: bool = False,
    ) -> Dict[str, float]:
        """
        Profile training update speed.

        Returns:
            Dict with timing statistics
        """
        import copy

        ref_agent = RNaDAgent(copy.deepcopy(agent.model))
        learner = RNaDLearner(agent, ref_agent, lr=1e-4, device=self.device)

        # If mixed precision, enable it
        if use_mixed_precision and hasattr(learner, "use_mixed_precision"):
            learner.use_mixed_precision = use_mixed_precision
            from torch.cuda.amp import GradScaler

            learner.scaler = GradScaler()

        embedder = Embedder(
            format="gen9vgc2023regulationc", feature_set=Embedder.FULL, omniscient=False
        )

        # Create dummy batch
        seq_len = 10
        batch = {
            "states": torch.randn(
                train_batch_size, seq_len, embedder.embedding_size, device=self.device
            ),
            "actions": torch.randint(
                0, 2025, (train_batch_size, seq_len), device=self.device
            ),
            "log_probs": torch.randn(train_batch_size, seq_len, device=self.device),
            "advantages": torch.randn(train_batch_size, seq_len, device=self.device),
            "returns": torch.randn(train_batch_size, seq_len, device=self.device),
            "is_teampreview": torch.zeros(
                train_batch_size, seq_len, dtype=torch.bool, device=self.device
            ),
            "padding_mask": torch.ones(
                train_batch_size, seq_len, dtype=torch.bool, device=self.device
            ),
        }

        # Warmup
        for _ in range(5):
            _ = learner.update(batch)

        # Profile
        torch.cuda.synchronize() if self.device == "cuda" else None
        times = []

        for _ in range(num_iterations):
            start = time.time()
            _ = learner.update(batch)
            torch.cuda.synchronize() if self.device == "cuda" else None
            end = time.time()
            times.append((end - start) * 1000)

        return {
            "mean_ms": float(np.mean(times)),
            "std_ms": float(np.std(times)),
            "min_ms": float(np.min(times)),
            "max_ms": float(np.max(times)),
            "updates_per_hour": 3600 / (float(np.mean(times)) / 1000),
        }

    def profile_resource_utilization(self, duration_seconds: int = 30) -> Dict[str, float]:
        """
        Monitor CPU/GPU/RAM utilization over time.

        Returns:
            Dict with resource statistics
        """
        cpu_samples = []
        gpu_util_samples = []
        gpu_mem_samples = []
        ram_samples = []

        start_time = time.time()
        while time.time() - start_time < duration_seconds:
            # CPU
            cpu_samples.append(psutil.cpu_percent(interval=0.1))

            # GPU (if available)
            if self.device == "cuda":
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu = gpus[0]  # Assume first GPU
                        gpu_util_samples.append(gpu.load * 100)
                        gpu_mem_samples.append(gpu.memoryUsed)
                except Exception:
                    pass

            # RAM
            ram_samples.append(psutil.virtual_memory().used / (1024**3))  # GB

            time.sleep(0.5)

        result = {
            "avg_cpu_percent": np.mean(cpu_samples),
            "peak_cpu_percent": np.max(cpu_samples),
            "avg_ram_gb": np.mean(ram_samples),
            "peak_ram_gb": np.max(ram_samples),
        }

        if gpu_util_samples:
            result.update(
                {
                    "avg_gpu_utilization": np.mean(gpu_util_samples),
                    "peak_gpu_utilization": np.max(gpu_util_samples),
                    "avg_gpu_memory_mb": np.mean(gpu_mem_samples),
                    "peak_gpu_memory_mb": np.max(gpu_mem_samples),
                }
            )
        else:
            result.update(
                {
                    "avg_gpu_utilization": 0.0,
                    "peak_gpu_utilization": 0.0,
                    "avg_gpu_memory_mb": 0.0,
                    "peak_gpu_memory_mb": 0.0,
                }
            )

        return result

    def profile_configuration(
        self,
        num_workers: int,
        players_per_worker: int,
        batch_size: int,
        train_batch_size: int,
        use_mixed_precision: bool = False,
    ) -> ProfilingResult:
        """
        Profile a specific training configuration.

        Args:
            num_workers: Number of worker threads
            players_per_worker: Players per worker
            batch_size: Inference batch size
            train_batch_size: Training batch size
            use_mixed_precision: Use FP16 mixed precision

        Returns:
            ProfilingResult with detailed metrics
        """
        print(f"\n{'=' * 60}")
        print("Profiling Configuration:")
        print(f"  Workers: {num_workers}")
        print(f"  Players/Worker: {players_per_worker}")
        print(f"  Batch Size: {batch_size}")
        print(f"  Train Batch Size: {train_batch_size}")
        print(f"  Mixed Precision: {use_mixed_precision}")
        print(f"{'=' * 60}\n")

        # Load model
        agent = self._load_model()

        # Profile inference
        print("Profiling inference...")
        inference_stats = self.profile_inference(agent, batch_size)

        # Profile training
        print("Profiling training updates...")
        training_stats = self.profile_training(
            agent, train_batch_size, use_mixed_precision=use_mixed_precision
        )

        # Profile resources (simulated - would need actual worker threads for real profile)
        print("Monitoring resource utilization...")
        resource_stats = self.profile_resource_utilization(duration_seconds=10)

        # Estimate throughput
        # Simplified estimates (real profiling would run actual workers)
        total_players = num_workers * players_per_worker
        avg_battle_duration_sec = 60  # Estimate
        battles_per_hour = total_players * (3600 / avg_battle_duration_sec)
        timesteps_per_battle = 4  # Average
        timesteps_per_hour = battles_per_hour * timesteps_per_battle
        updates_per_hour = training_stats["updates_per_hour"]

        # Identify bottleneck
        gpu_util = resource_stats["avg_gpu_utilization"]
        cpu_util = resource_stats["avg_cpu_percent"]

        if gpu_util > 80:
            primary_bottleneck = "gpu"
            bottleneck_score = gpu_util / 100
        elif cpu_util > 80:
            primary_bottleneck = "cpu"
            bottleneck_score = cpu_util / 100
        elif gpu_util < 30 and cpu_util < 30:
            primary_bottleneck = "network"  # Showdown servers
            bottleneck_score = 0.7
        else:
            primary_bottleneck = "balanced"
            bottleneck_score = max(gpu_util, cpu_util) / 100

        # Generate recommendations
        recommendations = self._generate_recommendations(
            num_workers,
            players_per_worker,
            batch_size,
            train_batch_size,
            primary_bottleneck,
            resource_stats,
            inference_stats,
            training_stats,
        )

        result = ProfilingResult(
            num_workers=num_workers,
            players_per_worker=players_per_worker,
            batch_size=batch_size,
            train_batch_size=train_batch_size,
            use_mixed_precision=use_mixed_precision,
            battles_per_hour=battles_per_hour,
            updates_per_hour=updates_per_hour,
            timesteps_per_hour=timesteps_per_hour,
            avg_cpu_percent=resource_stats["avg_cpu_percent"],
            peak_cpu_percent=resource_stats["peak_cpu_percent"],
            avg_gpu_utilization=resource_stats["avg_gpu_utilization"],
            peak_gpu_utilization=resource_stats["peak_gpu_utilization"],
            avg_gpu_memory_mb=resource_stats["avg_gpu_memory_mb"],
            peak_gpu_memory_mb=resource_stats["peak_gpu_memory_mb"],
            avg_ram_gb=resource_stats["avg_ram_gb"],
            peak_ram_gb=resource_stats["peak_ram_gb"],
            avg_inference_time_ms=inference_stats["mean_ms"],
            avg_training_time_ms=training_stats["mean_ms"],
            avg_data_collection_time_ms=avg_battle_duration_sec * 1000 / total_players,
            primary_bottleneck=primary_bottleneck,
            bottleneck_score=bottleneck_score,
            **recommendations,
        )

        self.results.append(result)
        return result

    def _generate_recommendations(
        self,
        num_workers: int,
        players_per_worker: int,
        batch_size: int,
        train_batch_size: int,
        bottleneck: str,
        resource_stats: Dict,
        inference_stats: Dict,
        training_stats: Dict,
    ) -> Dict[str, int]:
        """Generate optimal hyperparameter recommendations."""
        recommendations = {}

        gpu_util = resource_stats["avg_gpu_utilization"]
        gpu_mem = resource_stats["peak_gpu_memory_mb"]

        if bottleneck == "gpu":
            # GPU is bottleneck - reduce load or increase efficiency
            if gpu_mem > 20000:  # >20GB used
                recommendations["recommended_batch_size"] = max(8, batch_size // 2)
                recommendations["recommended_train_batch_size"] = max(
                    16, train_batch_size // 2
                )
            else:
                # GPU compute-bound, not memory
                recommendations["recommended_batch_size"] = batch_size
                recommendations["recommended_train_batch_size"] = train_batch_size
            recommendations["recommended_num_workers"] = num_workers
            recommendations["recommended_players_per_worker"] = players_per_worker

        elif bottleneck == "cpu":
            # CPU is bottleneck - reduce workers or increase batch size
            recommendations["recommended_num_workers"] = max(2, num_workers - 1)
            recommendations["recommended_players_per_worker"] = max(
                2, players_per_worker - 1
            )
            recommendations["recommended_batch_size"] = min(32, batch_size + 4)
            recommendations["recommended_train_batch_size"] = train_batch_size

        elif bottleneck == "network":
            # Network (Showdown) is bottleneck - increase parallelism
            recommendations["recommended_num_workers"] = min(8, num_workers + 2)
            recommendations["recommended_players_per_worker"] = min(
                8, players_per_worker + 2
            )
            recommendations["recommended_batch_size"] = min(32, batch_size + 8)
            recommendations["recommended_train_batch_size"] = train_batch_size

        else:  # balanced
            # System is well-balanced, minor tweaks
            if gpu_util < 50:
                recommendations["recommended_batch_size"] = min(32, batch_size + 4)
                recommendations["recommended_train_batch_size"] = min(
                    64, train_batch_size + 8
                )
            else:
                recommendations["recommended_batch_size"] = batch_size
                recommendations["recommended_train_batch_size"] = train_batch_size
            recommendations["recommended_num_workers"] = num_workers
            recommendations["recommended_players_per_worker"] = players_per_worker

        return recommendations

    def profile_sweep(
        self,
        worker_configs: Optional[List[Tuple[int, int]]] = None,
        batch_configs: Optional[List[Tuple[int, int]]] = None,
        test_mixed_precision: bool = True,
    ):
        """
        Sweep through multiple configurations to find optimal settings.

        Args:
            worker_configs: List of (num_workers, players_per_worker) tuples
            batch_configs: List of (batch_size, train_batch_size) tuples
            test_mixed_precision: Test both FP32 and FP16
        """
        if worker_configs is None:
            worker_configs = [(2, 2), (4, 4), (4, 6), (8, 4)]

        if batch_configs is None:
            batch_configs = [(8, 16), (16, 32), (24, 48), (32, 64)]

        mp_configs = [False, True] if test_mixed_precision else [False]

        print("\nStarting profiling sweep:")
        print(f"  Worker configs: {worker_configs}")
        print(f"  Batch configs: {batch_configs}")
        print(f"  Mixed precision: {mp_configs}")
        print(
            f"  Total configurations: {len(worker_configs) * len(batch_configs) * len(mp_configs)}\n"
        )

        for workers, players in worker_configs:
            for batch_size, train_batch_size in batch_configs:
                for use_mp in mp_configs:
                    try:
                        result = self.profile_configuration(
                            workers, players, batch_size, train_batch_size, use_mp
                        )
                        print(
                            f"\nResult: {result.battles_per_hour:.0f} battles/hr, "
                            f"{result.updates_per_hour:.0f} updates/hr, "
                            f"GPU: {result.avg_gpu_utilization:.1f}%"
                        )
                    except Exception as e:
                        print(f"Failed to profile config: {e}")
                        continue

        # Find best configuration
        best_result = max(self.results, key=lambda r: r.updates_per_hour)

        print(f"\n{'=' * 60}")
        print("BEST CONFIGURATION FOUND:")
        print(f"{'=' * 60}")
        print(f"Workers: {best_result.num_workers}")
        print(f"Players/Worker: {best_result.players_per_worker}")
        print(f"Batch Size: {best_result.batch_size}")
        print(f"Train Batch Size: {best_result.train_batch_size}")
        print(f"Mixed Precision: {best_result.use_mixed_precision}")
        print("\nPerformance:")
        print(f"  Battles/hour: {best_result.battles_per_hour:.0f}")
        print(f"  Updates/hour: {best_result.updates_per_hour:.0f}")
        print(f"  GPU Util: {best_result.avg_gpu_utilization:.1f}%")
        print(f"  CPU Util: {best_result.avg_cpu_percent:.1f}%")
        print(f"{'=' * 60}\n")

        return best_result

    def save_results(self, filepath: str):
        """Save all profiling results to JSON."""
        data = {
            "results": [r.to_dict() for r in self.results],
            "best_config": (
                max(self.results, key=lambda r: r.updates_per_hour).to_dict()
                if self.results
                else None
            ),
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Profiling results saved to {filepath}")


def main():
    """Example usage of the profiler."""
    import argparse

    parser = argparse.ArgumentParser(description="Profile RNaD training configuration")
    parser.add_argument(
        "--model",
        type=str,
        default="data/models/bc_action_model.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument("--sweep", action="store_true", help="Run full parameter sweep")
    parser.add_argument("--workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--players", type=int, default=4, help="Players per worker")
    parser.add_argument("--batch-size", type=int, default=16, help="Inference batch size")
    parser.add_argument(
        "--train-batch-size", type=int, default=32, help="Training batch size"
    )
    parser.add_argument(
        "--mixed-precision", action="store_true", help="Use mixed precision"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="profiling_results.json",
        help="Output file for results",
    )

    args = parser.parse_args()

    profiler = TrainingProfiler(args.model)

    if args.sweep:
        best = profiler.profile_sweep()
        print("\nRecommended config.yaml settings:")
        print(f"num_workers: {best.recommended_num_workers or best.num_workers}")
        print(
            f"players_per_worker: {best.recommended_players_per_worker or best.players_per_worker}"
        )
        print(f"batch_size: {best.recommended_batch_size or best.batch_size}")
        print(
            f"train_batch_size: {best.recommended_train_batch_size or best.train_batch_size}"
        )
        print(f"use_mixed_precision: {best.use_mixed_precision}")
    else:
        result = profiler.profile_configuration(
            args.workers,
            args.players,
            args.batch_size,
            args.train_batch_size,
            args.mixed_precision,
        )

        print("\nRecommendations:")
        if result.recommended_num_workers:
            print(
                f"  num_workers: {result.num_workers} → {result.recommended_num_workers}"
            )
        if result.recommended_players_per_worker:
            print(
                f"  players_per_worker: {result.players_per_worker} → {result.recommended_players_per_worker}"
            )
        if result.recommended_batch_size:
            print(f"  batch_size: {result.batch_size} → {result.recommended_batch_size}")
        if result.recommended_train_batch_size:
            print(
                f"  train_batch_size: {result.train_batch_size} → {result.recommended_train_batch_size}"
            )

    profiler.save_results(args.output)


if __name__ == "__main__":
    main()
