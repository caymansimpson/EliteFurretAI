import argparse
import asyncio
import json
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict

from poke_env.battle import Pokemon

import elitefurretai.etl.embedder as embedder_module
from elitefurretai.engine.analyze import showdown_benchmark
from elitefurretai.etl.embedder import Embedder


@dataclass
class TimerStat:
    calls: int = 0
    total_seconds: float = 0.0


class EmbedderProfiler:
    def __init__(self) -> None:
        self._stats: Dict[str, TimerStat] = defaultdict(TimerStat)
        self._lock = threading.Lock()

    @contextmanager
    def measure(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            with self._lock:
                stat = self._stats[name]
                stat.calls += 1
                stat.total_seconds += elapsed

    def snapshot(self) -> Dict[str, Dict[str, float]]:
        result: Dict[str, Dict[str, float]] = {}
        for name, stat in sorted(
            self._stats.items(), key=lambda item: item[1].total_seconds, reverse=True
        ):
            result[name] = {
                "calls": float(stat.calls),
                "total_seconds": stat.total_seconds,
                "avg_ms": (stat.total_seconds * 1000.0 / stat.calls) if stat.calls else 0.0,
            }
        return result


def _wrap_method(
    profiler: EmbedderProfiler,
    owner: Any,
    attr_name: str,
    metric_name: str,
) -> Callable[[], None]:
    original = getattr(owner, attr_name)

    def wrapped(*args, **kwargs):
        with profiler.measure(metric_name):
            return original(*args, **kwargs)

    setattr(owner, attr_name, wrapped)

    def restore() -> None:
        setattr(owner, attr_name, original)

    return restore


def _build_parser() -> argparse.ArgumentParser:
    parser = showdown_benchmark.build_parser()
    parser.description = (
        "Profile embedder subphases over a real Showdown websocket benchmark run."
    )
    parser.add_argument(
        "--report-output",
        required=True,
        help="Path to write JSON timing report.",
    )
    return parser


async def _run_profile(args: argparse.Namespace) -> None:
    profiler = EmbedderProfiler()
    created_embedders: list[Embedder] = []
    original_init = Embedder.__init__

    def instrumented_init(self, *init_args, **init_kwargs):
        original_init(self, *init_args, **init_kwargs)
        created_embedders.append(self)

    Embedder.__init__ = instrumented_init
    restore_callbacks = [
        _wrap_method(profiler, Embedder, "embed_to_array", "Embedder.embed_to_array"),
        _wrap_method(profiler, Embedder, "embed", "Embedder.embed"),
        _wrap_method(
            profiler,
            Embedder,
            "generate_feature_engineered_features",
            "Embedder.generate_feature_engineered_features",
        ),
        _wrap_method(
            profiler,
            Embedder,
            "generate_transition_features",
            "Embedder.generate_transition_features",
        ),
        _wrap_method(
            profiler,
            Embedder,
            "generate_battle_features",
            "Embedder.generate_battle_features",
        ),
        _wrap_method(
            profiler,
            Embedder,
            "generate_pokemon_features",
            "Embedder.generate_pokemon_features",
        ),
        _wrap_method(
            profiler,
            Embedder,
            "generate_opponent_pokemon_features",
            "Embedder.generate_opponent_pokemon_features",
        ),
        _wrap_method(profiler, embedder_module, "compute_stats", "embedder.compute_stats"),
        _wrap_method(
            profiler,
            embedder_module,
            "calculate_damage",
            "embedder.calculate_damage",
        ),
        _wrap_method(
            profiler,
            Pokemon,
            "available_moves_from_request",
            "Pokemon.available_moves_from_request",
        ),
    ]

    benchmark_start = time.perf_counter()
    try:
        await showdown_benchmark._run_benchmark(args)
    finally:
        benchmark_seconds = time.perf_counter() - benchmark_start
        for restore in reversed(restore_callbacks):
            restore()
        Embedder.__init__ = original_init

    aggregate_cache_stats = {
        "hits": 0.0,
        "misses": 0.0,
        "evictions": 0.0,
        "size": 0.0,
    }
    for embedder in created_embedders:
        get_stats = getattr(embedder, "get_damage_cache_stats", None)
        if get_stats is None:
            continue
        cache_stats = get_stats()
        aggregate_cache_stats["hits"] += float(cache_stats.get("hits", 0))
        aggregate_cache_stats["misses"] += float(cache_stats.get("misses", 0))
        aggregate_cache_stats["evictions"] += float(cache_stats.get("evictions", 0))
        aggregate_cache_stats["size"] = max(
            aggregate_cache_stats["size"],
            float(cache_stats.get("size", 0)),
        )

    report = {
        "benchmark_seconds": benchmark_seconds,
        "battles": float(args.battles),
        "format": args.format,
        "policy": args.policy,
        "batch_size": float(args.batch_size),
        "batch_timeout": args.batch_timeout,
        "max_concurrent_battles": float(args.max_concurrent_battles),
        "device": args.device,
        "feature_set": args.feature_set,
        "damage_cache_stats": aggregate_cache_stats,
        "timings": profiler.snapshot(),
    }
    Path(args.report_output).write_text(json.dumps(report, indent=2, sort_keys=True))


def main() -> None:
    args = _build_parser().parse_args()
    asyncio.run(_run_profile(args))


if __name__ == "__main__":
    main()
