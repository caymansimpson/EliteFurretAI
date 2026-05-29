# -*- coding: utf-8 -*-
"""Multi-bucket eval driver, scoring, log payload, and standalone CLI.

Replaces the FoulPlay-specific driver in foulplay_eval.py with a generic
driver that handles every opponent in EvalConfig.opponents identically.
FoulPlay becomes one bucket; its weight defaults to 0.0 until the
FoulPlay subprocess is stable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

from elitefurretai.rl.analyze.evaluate import EvalResult, run_eval_parallel
from elitefurretai.rl.analyze.player_factory import parse_player_specification
from elitefurretai.rl.config import CurriculumConfig, EvalConfig, OpponentEvalSpec

# Opponent canonical names whose construction goes through the external
# subprocess path inside run_eval_parallel. Mirrors
# player_factory._EXTERNAL_BASELINES.
_EXTERNAL_OPPONENTS = ("vgc_bench", "foul_play")


def compute_score(
    win_rates: Dict[str, float],
    targets: Dict[str, float],
    weights: Dict[str, float],
    surplus_alpha: float,
) -> Tuple[float, Dict[str, float]]:
    """Compute the W&B sweep metric from per-opponent win rates.

    Hinge L2 on deficit + linear surplus, both scaled to percentage
    points (x100) so floors dominate at any non-trivial deficit. See
    planning/stage2/2026-05-29-21-00-rl-wandb-sweep-eval-design.md
    Section 1 for the rationale.

    Maximize. score=0 means every floor met exactly.
    """
    deficit_pp = {k: max(0.0, 100.0 * (targets[k] - win_rates[k])) for k in weights}
    surplus_pp = {k: max(0.0, 100.0 * (win_rates[k] - targets[k])) for k in weights}
    deficit_term = sum(weights[k] * deficit_pp[k] ** 2 for k in weights)
    surplus_term = sum(weights[k] * surplus_pp[k] for k in weights)
    score = surplus_alpha * surplus_term - deficit_term
    return score, {"deficit_l2_pp": deficit_term, "surplus_sum_pp": surplus_term}


@dataclass
class BucketRunResult:
    """Aggregated outcome for ONE opponent (one entry in
    EvalConfig.opponents) after running its full eval cycle across all
    curriculum.battle_formats.

    Granularity: one BucketRunResult per active opponent per eval pass.

    win_rate is the format-weighted mean of per-format win rates and is
    the single number that feeds compute_score for this opponent.
    per_format preserves the per-format breakdown so the wandb logger
    can compute eval/<format>/win_rate cross-opponent aggregates.
    """

    win_rate: float
    n_battles: int
    per_format: Dict[str, EvalResult] = field(default_factory=dict)
    wall_time_s: float = 0.0


@dataclass
class MultiBucketEvalResult:
    """Outcome of ONE full multi-bucket eval pass.

    Granularity: one MultiBucketEvalResult per call to baseline_eval.run,
    i.e. one per checkpoint boundary during training or one per
    standalone CLI invocation.

    per_bucket has one entry per active (weight > 0) opponent.
    Disabled opponents (weight == 0) do not appear here.
    score is the scalar W&B sweep metric.
    """

    per_bucket: Dict[str, BucketRunResult] = field(default_factory=dict)
    score: float = 0.0
    breakdown: Dict[str, float] = field(default_factory=dict)
    wall_time_s: float = 0.0


def _resolve_agent_team_text(curriculum: CurriculumConfig, fmt: str) -> str:
    """Return the agent's team text for ``fmt`` — deterministic across passes.

    Ported from foulplay_eval._resolve_agent_team_text. Eval uses a fixed
    agent team so win-rate is comparable across checkpoints.
    """
    paths = curriculum.resolved_agent_team_paths()
    raw = paths.get(fmt)
    if raw is None:
        raise FileNotFoundError(
            f"No agent_team_path resolved for format {fmt!r}; eval requires "
            f"a fixed agent team. Set CurriculumConfig.agent_team_path."
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
    raise FileNotFoundError(f"Path is neither file nor directory: {path}")


def _resolve_opponent_team_text(
    opp_name: str, curriculum: CurriculumConfig, fmt: str
) -> str:
    """Resolve opponent team text for one (opp, fmt) cell.

    External opponents (vgc_bench, foul_play) pick their own teams via
    subprocess kwargs, so we pass an empty string. In-process baselines
    sample the first team from the curriculum's opponent team pool.
    """
    if opp_name in _EXTERNAL_OPPONENTS:
        return ""
    opp_pool = curriculum.resolved_opponent_team_pool_paths().get(fmt)
    if opp_pool is None:
        raise ValueError(
            f"No opponent team pool for {fmt!r}; in-process baseline "
            f"{opp_name!r} requires a sampled opponent team."
        )
    pool_dir = Path(f"{curriculum.base_team_path}/{fmt}/{opp_pool}")
    teams = sorted(pool_dir.glob("*.txt"))
    if not teams:
        raise FileNotFoundError(f"No .txt teams in {pool_dir}")
    return teams[0].read_text()


def _foulplay_team_pool_for_fmt(
    eval_cfg: EvalConfig, curriculum: CurriculumConfig, fmt: str
) -> str:
    """Resolve FoulPlay's team pool path for ``fmt``. Ported from
    foulplay_eval._resolve_foulplay_team_pool.
    """
    if eval_cfg.foulplay_team_pool_paths is not None:
        return eval_cfg.foulplay_team_pool_paths[fmt]
    opp_pool = curriculum.resolved_opponent_team_pool_paths().get(fmt)
    if opp_pool is None:
        raise ValueError(
            f"No FoulPlay team pool resolved for {fmt!r}: "
            f"foulplay_team_pool_paths is None and "
            f"opponent_team_pool_paths[{fmt!r}] is None."
        )
    return f"{curriculum.base_team_path}/{fmt}/{opp_pool}"


def _opponent_kwargs(opp_name: str, eval_cfg: EvalConfig) -> Dict[str, Any]:
    """Map EvalConfig's vgcbench_*/foulplay_* fields to the kwarg names
    parse_player_specification expects (vgc_bench_*, foul_play_*). The
    foul_play team-pool kwarg is fmt-specific and added by the caller.
    """
    if opp_name == "vgc_bench":
        return {
            "vgc_bench_checkpoint_path": eval_cfg.vgcbench_checkpoint_path,
            "vgc_bench_team_file": eval_cfg.vgcbench_team_file,
            "vgc_bench_python_executable": eval_cfg.vgcbench_python_executable,
        }
    if opp_name == "foul_play":
        return {
            "foul_play_python_executable": eval_cfg.foulplay_python_executable,
            "foul_play_search_time_ms": eval_cfg.foulplay_search_time_ms,
            "foul_play_parallelism": eval_cfg.foulplay_parallelism,
        }
    return {}


def _split_battles_by_format(
    n_total: int, format_weights: Dict[str, float]
) -> Dict[str, int]:
    """Allocate per-format battle counts proportional to format weights.
    Largest-remainder so the sum matches n_total exactly.
    """
    if not format_weights or n_total <= 0:
        return {fmt: 0 for fmt in format_weights}
    total_weight = sum(format_weights.values())
    if total_weight <= 0:
        return {fmt: 0 for fmt in format_weights}
    raw = {fmt: n_total * w / total_weight for fmt, w in format_weights.items()}
    floors = {fmt: int(v) for fmt, v in raw.items()}
    remainder = n_total - sum(floors.values())
    fracs = sorted(format_weights.keys(), key=lambda f: raw[f] - floors[f], reverse=True)
    for fmt in fracs[:remainder]:
        floors[fmt] += 1
    return floors


def _run_opponent_bucket(
    opp_name: str,
    spec: OpponentEvalSpec,
    eval_cfg: EvalConfig,
    curriculum: CurriculumConfig,
    checkpoint_path: str,
    server_urls: List[str],
    device: str,
    run_tag: str,
) -> BucketRunResult:
    """Run one opponent across all curriculum formats.

    Mirrors foulplay_eval._run_worker but parametrized by opponent name.
    Subprocess lifecycle for external opponents is handled internally by
    run_eval_parallel, so no explicit RunningExternal tracking is needed
    at this level (layer-1 cleanup is therefore implicit).
    """
    t0 = time.time()
    per_format: Dict[str, EvalResult] = {}
    format_weights: Dict[str, float] = dict(curriculum.battle_formats)
    battles_per_format = _split_battles_by_format(spec.n_battles, format_weights)

    for fmt in curriculum.battle_formats:
        n_for_fmt = battles_per_format.get(fmt, 0)
        if n_for_fmt <= 0:
            continue

        agent_team_text = _resolve_agent_team_text(curriculum, fmt)
        opp_team_text = _resolve_opponent_team_text(opp_name, curriculum, fmt)

        kwargs = _opponent_kwargs(opp_name, eval_cfg)
        if opp_name == "foul_play":
            kwargs["foul_play_team_pool_path"] = _foulplay_team_pool_for_fmt(
                eval_cfg, curriculum, fmt
            )

        model_spec = parse_player_specification(
            checkpoint_path, device=device, battle_format=fmt
        )
        opp_spec = parse_player_specification(
            opp_name, device=device, battle_format=fmt, **kwargs
        )

        cells = [(agent_team_text, opp_team_text)]
        fmt_result = run_eval_parallel(
            p1=model_spec,
            p2=opp_spec,
            cells=cells,
            battles_per_cell=n_for_fmt,
            server_urls=server_urls,
            workers=1,
            run_tag=run_tag,
        )
        per_format[fmt] = fmt_result

    total_w = sum(format_weights[f] for f in per_format)
    if total_w > 0:
        win_rate = (
            sum(
                format_weights[f]
                * (per_format[f].player1_wins / max(per_format[f].battles_played, 1))
                for f in per_format
            )
            / total_w
        )
    else:
        win_rate = 0.0
    n_total = sum(r.battles_played for r in per_format.values())

    return BucketRunResult(
        win_rate=win_rate,
        n_battles=n_total,
        per_format=per_format,
        wall_time_s=time.time() - t0,
    )
