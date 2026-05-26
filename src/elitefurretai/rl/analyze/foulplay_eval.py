# -*- coding: utf-8 -*-
"""Driver for FoulPlay eval (per-checkpoint cadence + standalone CLI).

Two entry points:

* :func:`run` — called from ``train.py`` at checkpoint cadence. Iterates
  over :class:`CurriculumConfig.battle_formats`, runs one self-contained
  FoulPlay cycle per format (launch → battles → teardown via
  ``_run_worker`` → ``launch_external_player``), aggregates per-format
  and overall results into :class:`FoulplayEvalResult`.
* :func:`main` — argparse CLI for manual checkpoint eval.

The driver does NOT use :class:`FoulPlayManager` directly. The subprocess
launch happens inside ``_run_worker`` via the existing
``launch_external_player`` path (now dispatching foul_play to
``_launch_foulplay_subprocess``). That keeps the trajectory parquet +
gzipped replay capture working "for free" — FoulPlay-vs-model battles
land in the same place as every other eval.

See planning/stage2/2026-05-25-23-37-foulplay-eval-scope-confirmed.md.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from elitefurretai.engine.showdown_server_manager import (
    launch_showdown_servers,
    shutdown_showdown_servers,
)
from elitefurretai.rl.analyze.evaluate import EvalResult, run_eval_parallel
from elitefurretai.rl.analyze.player_factory import parse_player_specification
from elitefurretai.rl.config import CurriculumConfig, FoulplayEvalConfig


@dataclass
class FoulplayEvalResult:
    """Aggregated outcome of one FoulPlay eval pass across all active formats.

    The per-format breakdown is preserved so wandb logs can record
    both per-format win rates and a weight-averaged overall. Aggregate
    properties (``model_wins``, ``foulplay_wins``, ``battles_played``,
    ``overall_win_rate``) summarize across formats.
    """

    per_format: Dict[str, EvalResult] = field(default_factory=dict)
    battle_formats_weights: Dict[str, float] = field(default_factory=dict)
    wall_time_s: float = 0.0

    @property
    def battles_played(self) -> int:
        return sum(r.battles_played for r in self.per_format.values())

    @property
    def model_wins(self) -> int:
        return sum(r.player1_wins for r in self.per_format.values())

    @property
    def foulplay_wins(self) -> int:
        return sum(r.player2_wins for r in self.per_format.values())

    @property
    def ties(self) -> int:
        return sum(r.ties for r in self.per_format.values())

    @property
    def overall_win_rate(self) -> float:
        """Weight-averaged win rate across active formats.

        Weights come from ``CurriculumConfig.battle_formats``. Formats
        present in ``per_format`` but missing from
        ``battle_formats_weights`` are excluded from the average.
        """
        weights = {
            fmt: self.battle_formats_weights.get(fmt, 0.0)
            for fmt in self.per_format
        }
        total_weight = sum(weights.values())
        if total_weight == 0:
            return 0.0
        weighted = sum(
            self.per_format[fmt].player1_win_rate * w for fmt, w in weights.items()
        )
        return weighted / total_weight


def _resolve_foulplay_team_pool(
    config: FoulplayEvalConfig, curriculum: CurriculumConfig, fmt: str
) -> str:
    """Pick the team pool FoulPlay samples its teams from for ``fmt``.

    If ``config.foulplay_team_pool_paths[fmt]`` is set, use it
    verbatim. Otherwise fall back to the curriculum's opponent team
    pool for that format (``base_team_path/<fmt>/<opp_pool_subdir>``).
    """
    if config.foulplay_team_pool_paths is not None:
        return config.foulplay_team_pool_paths[fmt]

    opp_pool = curriculum.resolved_opponent_team_pool_paths().get(fmt)
    if opp_pool is None:
        raise ValueError(
            f"No FoulPlay team pool resolved for format {fmt!r}: "
            f"foulplay_team_pool_paths is None and "
            f"opponent_team_pool_paths[{fmt!r}] is None."
        )
    return f"{curriculum.base_team_path}/{fmt}/{opp_pool}"


def _resolve_agent_team_text(
    curriculum: CurriculumConfig,
    fmt: str,
    override: Optional[str] = None,
) -> str:
    """Return the agent's team text for ``fmt`` — deterministic across passes.

    The eval uses a fixed agent team (decoupled from the Change 7 adaptive
    sampler) so win-rate is comparable across checkpoints.

    Resolution order:

    1. ``override`` (file or directory path, taken verbatim — used by
       the CLI which doesn't necessarily have a full CurriculumConfig).
    2. ``curriculum.resolved_agent_team_paths()[fmt]`` — used by the
       training-loop hook, where the curriculum is fully populated.

    Whichever path is selected: file → return contents; directory →
    return contents of the first ``.txt`` file in sorted order.
    """
    if override is not None:
        raw: Optional[str] = override
    else:
        paths = curriculum.resolved_agent_team_paths()
        raw = paths.get(fmt)
    if raw is None:
        raise FileNotFoundError(
            f"No agent_team_path resolved for format {fmt!r}; FoulPlay eval "
            f"requires a fixed agent team. Set CurriculumConfig.agent_team_path "
            f"or pass agent_team_paths to run()."
        )
    path = Path(raw)
    if path.is_file():
        return path.read_text()
    if path.is_dir():
        team_files = sorted(path.glob("*.txt"))
        if not team_files:
            raise FileNotFoundError(
                f"No .txt files in agent team directory for {fmt!r}: {path}"
            )
        return team_files[0].read_text()
    raise FileNotFoundError(f"agent_team_path for {fmt!r} not found: {path}")


def run(
    *,
    checkpoint_path: str,
    config: FoulplayEvalConfig,
    curriculum: CurriculumConfig,
    device: str,
    server_urls: List[str],
    run_tag: str,
    collect_run_dir: Optional[str] = None,
    agent_team_paths: Optional[Dict[str, str]] = None,
) -> FoulplayEvalResult:
    """Run one FoulPlay eval pass across every active battle format.

    For each format in ``curriculum.battle_formats`` (order: dict
    insertion order, which is YAML order for loaded configs):

    1. Resolve agent team source + FoulPlay team pool.
    2. Build the model + foul_play :class:`PlayerSpecification`.
    3. Call ``run_eval_parallel`` with ``workers=1`` and a single
       cell — FoulPlay's 8-core search saturates the machine, so
       there's nothing to parallelize at the eval-worker level.
       ``_run_worker`` spawns the FoulPlay subprocess via
       ``launch_external_player`` and tears it down on exit.
    4. Record the per-format result.

    Returns a :class:`FoulplayEvalResult` with per-format and
    weight-averaged win rates.

    Battle-finish trajectory parquet + gzipped replays land under
    ``collect_run_dir`` (when set) the same way every other eval's
    artifacts do — no FoulPlay-specific capture pipeline.
    """
    started = time.time()
    per_format: Dict[str, EvalResult] = {}
    server_url = server_urls[0]

    for fmt in curriculum.battle_formats:
        override = (
            agent_team_paths.get(fmt) if agent_team_paths is not None else None
        )
        agent_team_text = _resolve_agent_team_text(curriculum, fmt, override=override)
        foulplay_team_pool = _resolve_foulplay_team_pool(config, curriculum, fmt)

        model_spec = parse_player_specification(
            checkpoint_path,
            device=device,
            battle_format=fmt,
        )
        assert config.python_executable is not None, (
            "FoulplayEvalConfig.python_executable must be set "
            "before calling run() — verify() should have caught this."
        )
        foul_play_spec = parse_player_specification(
            "foul_play",
            device=device,
            battle_format=fmt,
            foul_play_python_executable=config.python_executable,
            foul_play_team_pool_path=foulplay_team_pool,
            foul_play_search_time_ms=config.search_time_ms,
            foul_play_parallelism=config.parallelism,
        )

        # Single cell — fixed agent team for clean cross-checkpoint
        # comparison. FoulPlay picks its own team from --team-list-dir,
        # so the opp-side team string is unused (the worker's external
        # branch never builds a P2 Player object).
        cells = [(agent_team_text, "")]

        eval_result = run_eval_parallel(
            p1=model_spec,
            p2=foul_play_spec,
            cells=cells,
            battles_per_cell=config.n_battles_per_format,
            server_urls=[server_url],
            workers=1,
            run_tag=run_tag,
            collect_run_dir=collect_run_dir,
        )
        per_format[fmt] = eval_result

    return FoulplayEvalResult(
        per_format=per_format,
        battle_formats_weights=dict(curriculum.battle_formats),
        wall_time_s=round(time.time() - started, 2),
    )


def _format_result_line(fmt: str, r: EvalResult) -> str:
    return (
        f"  {fmt}: battles={r.battles_played} "
        f"model_wins={r.player1_wins} foulplay_wins={r.player2_wins} "
        f"ties={r.ties} win_rate={r.player1_win_rate * 100:.2f}%"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run FoulPlay eval against a model checkpoint."
    )
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument(
        "--battle-formats",
        type=str,
        default="gen9vgc2024regg:1.0",
        help="Comma-separated format:weight pairs, e.g. "
        "'gen9vgc2024regg:0.7,gen9vgc2024regh:0.3'. Weights must sum to 1.0.",
    )
    parser.add_argument("--n-battles-per-format", type=int, default=100)
    parser.add_argument("--search-time-ms", type=int, default=750)
    parser.add_argument("--parallelism", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--python-executable",
        required=True,
        type=str,
        help="Path to ../venv-foulplay/bin/python",
    )
    parser.add_argument(
        "--foulplay-team-pool",
        required=True,
        type=str,
        help="Directory FoulPlay samples teams from (single-format CLI)",
    )
    parser.add_argument(
        "--agent-team-pool",
        type=str,
        default=None,
        help="Directory or file the model samples teams from. If omitted, "
        "uses the format's default team directory under data/teams/<format>/.",
    )
    parser.add_argument("--base-team-path", type=str, default="data/teams")
    parser.add_argument("--num-servers", type=int, default=1)
    parser.add_argument("--start-port", type=int, default=8000)
    parser.add_argument(
        "--launch-servers",
        action="store_true",
        help="Launch local Showdown servers automatically",
    )
    args = parser.parse_args()

    # Parse battle_formats CLI arg into the dict form CurriculumConfig wants.
    battle_formats: Dict[str, float] = {}
    for pair in args.battle_formats.split(","):
        fmt, _, weight = pair.partition(":")
        battle_formats[fmt.strip()] = float(weight or "1.0")
    total = sum(battle_formats.values())
    battle_formats = {k: v / total for k, v in battle_formats.items()}

    # Build a minimal CurriculumConfig — the driver only reads
    # battle_formats, base_team_path, and resolved_*_paths.
    curriculum = CurriculumConfig(
        battle_formats=battle_formats,
        base_team_path=args.base_team_path,
    )
    config = FoulplayEvalConfig(
        enabled=True,
        n_battles_per_format=args.n_battles_per_format,
        search_time_ms=args.search_time_ms,
        parallelism=args.parallelism,
        python_executable=args.python_executable,
        # Single-format CLI broadcasts one team pool to every active format.
        foulplay_team_pool_paths={fmt: args.foulplay_team_pool for fmt in battle_formats},
    )

    server_processes = []
    if args.launch_servers:
        server_processes = launch_showdown_servers(args.num_servers, args.start_port)

    try:
        server_urls = [
            f"localhost:{args.start_port + i}" for i in range(args.num_servers)
        ]
        run_tag = format(int(time.time() * 1000) % 65536, "04x")

        # CLI broadcasts a single agent-team-pool to every active format.
        # Defaults to --foulplay-team-pool if not given (sensible
        # default — model and FoulPlay sample from the same pool, so
        # win rate measures policy strength, not team-mismatch).
        agent_pool = args.agent_team_pool or args.foulplay_team_pool
        agent_team_paths = {fmt: agent_pool for fmt in battle_formats}

        result = run(
            checkpoint_path=args.checkpoint,
            config=config,
            curriculum=curriculum,
            device=args.device,
            server_urls=server_urls,
            run_tag=run_tag,
            agent_team_paths=agent_team_paths,
        )

        print(
            f"FoulPlay eval | overall battles={result.battles_played} "
            f"model_wins={result.model_wins} foulplay_wins={result.foulplay_wins} "
            f"ties={result.ties} win_rate={result.overall_win_rate * 100:.2f}% "
            f"wall_time={result.wall_time_s:.1f}s"
        )
        for fmt, r in result.per_format.items():
            print(_format_result_line(fmt, r))
    finally:
        if server_processes:
            shutdown_showdown_servers(server_processes)


if __name__ == "__main__":
    main()
