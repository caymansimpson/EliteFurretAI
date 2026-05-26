"""Unit tests for Change 7 — agent-team-axis adaptive curriculum.

Each test exercises one slice of the per-(battle_format, agent_team)
sampling design from
planning/stage2/2026-05-24-12-30-change7-team-axis-curriculum-design.md.
"""

from __future__ import annotations

from elitefurretai.rl.config import CurriculumConfig


def test_curriculum_config_has_team_axis_defaults():
    """CurriculumConfig exposes three new team-axis fields with documented defaults."""
    cfg = CurriculumConfig()
    assert cfg.team_axis_enabled is True
    assert cfg.team_warmup_threshold == 20
    assert cfg.team_per_team_floor == 0.005
