# -*- coding: utf-8 -*-
"""Multi-bucket eval driver, scoring, log payload, and standalone CLI.

Replaces the FoulPlay-specific driver in foulplay_eval.py with a generic
driver that handles every opponent in EvalConfig.opponents identically.
FoulPlay becomes one bucket; its weight defaults to 0.0 until the
FoulPlay subprocess is stable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

from elitefurretai.rl.analyze.evaluate import EvalResult


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
