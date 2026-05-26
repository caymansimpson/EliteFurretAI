"""Tests for the shared adaptive-curriculum primitives in rl_utils.

These primitives are called from two places:
- WorkerOpponentFactory.update_curriculum (agent-axis: opponent type mix)
- WorkerOpponentFactory.update_team_distribution (team-axis: per-format
  team mix)

The tests below pin down behavior shared between both call sites.
"""

import math

import pytest

from elitefurretai.rl.rl_utils import adaptive_distribution, adaptive_score

# ── adaptive_score ────────────────────────────────────────────────────


def test_adaptive_score_pfsp_peaks_at_50_percent_winrate():
    """PFSP component is 1.0 when win rate = 0.5, 0.0 at extremes."""
    s_50 = adaptive_score(
        wins=50.0,
        n=100.0,
        prior_alpha=8.0,
        prior_beta=8.0,
        pfsp_mix=1.0,
        weakness_mix=0.0,
        weakness_exponent=1.0,
        target_win_rate=0.55,
    )
    s_100 = adaptive_score(
        wins=100.0,
        n=100.0,
        prior_alpha=8.0,
        prior_beta=8.0,
        pfsp_mix=1.0,
        weakness_mix=0.0,
        weakness_exponent=1.0,
        target_win_rate=0.55,
    )
    assert s_50 > s_100
    assert s_50 > 0.85
    assert s_100 < 0.20


def test_adaptive_score_weakness_grows_as_winrate_drops():
    """Weakness component is 0 above target, grows linearly below."""
    s_low = adaptive_score(
        wins=10.0,
        n=100.0,
        prior_alpha=8.0,
        prior_beta=8.0,
        pfsp_mix=0.0,
        weakness_mix=1.0,
        weakness_exponent=1.0,
        target_win_rate=0.55,
    )
    s_high = adaptive_score(
        wins=80.0,
        n=100.0,
        prior_alpha=8.0,
        prior_beta=8.0,
        pfsp_mix=0.0,
        weakness_mix=1.0,
        weakness_exponent=1.0,
        target_win_rate=0.55,
    )
    assert s_low > 0.5
    assert s_high == 0.0


def test_adaptive_score_weakness_exponent_asymmetric():
    """weakness_exponent > 1 reduces magnitude on values in (0,1) — team-axis
    uses this to soften the asymmetric weakness shape."""
    s_lin = adaptive_score(
        wins=20.0,
        n=100.0,
        prior_alpha=8.0,
        prior_beta=8.0,
        pfsp_mix=0.0,
        weakness_mix=1.0,
        weakness_exponent=1.0,
        target_win_rate=0.55,
    )
    s_squared = adaptive_score(
        wins=20.0,
        n=100.0,
        prior_alpha=8.0,
        prior_beta=8.0,
        pfsp_mix=0.0,
        weakness_mix=1.0,
        weakness_exponent=2.0,
        target_win_rate=0.55,
    )
    assert s_lin > s_squared > 0.0


# ── adaptive_distribution ────────────────────────────────────────────


def test_adaptive_distribution_normalizes_to_one():
    scores = {"a": 2.0, "b": 1.0, "c": 1.0}
    d = adaptive_distribution(scores)
    assert math.isclose(sum(d.values()), 1.0, abs_tol=1e-9)
    assert d["a"] > d["b"] == d["c"]


def test_adaptive_distribution_water_fills_below_floor():
    """When a key's natural share is below floor, pin it at floor and
    renormalize remaining mass over unpinned keys by score share."""
    scores = {"a": 100.0, "b": 100.0, "c": 0.01}
    floors = {"a": 0.0, "b": 0.0, "c": 0.10}
    d = adaptive_distribution(scores, floors=floors)
    assert math.isclose(sum(d.values()), 1.0, abs_tol=1e-9)
    assert math.isclose(d["c"], 0.10, abs_tol=1e-9)
    assert math.isclose(d["a"], 0.45, abs_tol=1e-9)
    assert math.isclose(d["b"], 0.45, abs_tol=1e-9)


def test_adaptive_distribution_uniform_floor_dict_matches_team_axis():
    """Team-axis builds floors={t: per_team_floor for t in teams}. With
    per_team_floor=0.005 and one of three teams essentially zero score,
    that team gets pinned at 0.005 and the rest split the residual."""
    scores = {"t1": 1.0, "t2": 1.0, "t3": 1e-9}
    floors = {k: 0.005 for k in scores}
    d = adaptive_distribution(scores, floors=floors)
    assert math.isclose(sum(d.values()), 1.0, abs_tol=1e-9)
    assert math.isclose(d["t3"], 0.005, abs_tol=1e-9)


def test_adaptive_distribution_falls_back_to_uniform_when_floors_too_aggressive():
    """If total floor mass >= 1.0, fall back to uniform across all keys."""
    scores = {"a": 1.0, "b": 1.0, "c": 1.0}
    floors = {"a": 0.5, "b": 0.5, "c": 0.5}
    d = adaptive_distribution(scores, floors=floors)
    assert math.isclose(sum(d.values()), 1.0, abs_tol=1e-9)
    assert d == pytest.approx({"a": 1 / 3, "b": 1 / 3, "c": 1 / 3})


def test_adaptive_distribution_blends_base_curriculum():
    """When base_blend > 0 and a base curriculum is provided, the final
    per-key score is base_blend*base[k] + (1-base_blend)*scores[k]
    before water-filling. Agent-axis uses base_blend=0.5."""
    scores = {"a": 1.0, "b": 0.0}
    base = {"a": 0.0, "b": 1.0}
    d_pure = adaptive_distribution(scores)
    d_mixed = adaptive_distribution(scores, base=base, base_blend=0.5)
    assert d_pure["a"] > d_pure["b"]
    assert math.isclose(d_mixed["a"], 0.5, abs_tol=1e-9)
    assert math.isclose(d_mixed["b"], 0.5, abs_tol=1e-9)


def test_adaptive_distribution_handles_all_zero_scores():
    """All-zero scores → uniform fallback over keys."""
    scores = {"a": 0.0, "b": 0.0, "c": 0.0}
    d = adaptive_distribution(scores)
    assert d == pytest.approx({"a": 1 / 3, "b": 1 / 3, "c": 1 / 3})
