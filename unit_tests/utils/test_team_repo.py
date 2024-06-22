# -*- coding: utf-8 -*-
from elitefurretai.utils.team_repo import TeamRepo


def test_team_repo():
    tr = TeamRepo("data/fixture")
    assert "example_team" in tr.teams
    assert len(tr.teams) > 5
    for team in tr.teams:
        assert isinstance(tr.teams[team], str)
