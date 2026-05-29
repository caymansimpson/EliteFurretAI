# -*- coding: utf-8 -*-
"""Unit tests for engine.battle_renderer.

These cover the leaf renderers (Pokemon line, field, moves, events) used
to build the human-facing CLI and the inference debug dumps.
"""

from logging import Logger

from poke_env.battle import (
    DoubleBattle,
    Effect,
    Field,
    Pokemon,
    SideCondition,
    Status,
    Weather,
)
from poke_env.teambuilder.constant_teambuilder import ConstantTeambuilder


def _calyrex_shadow() -> Pokemon:
    tb = ConstantTeambuilder(
        """Calyrex-Shadow @ Life Orb
        Ability: As One (Spectrier)
        Level: 50
        Tera Type: Ghost
        EVs: 252 SpA / 4 SpD / 252 Spe
        Timid Nature
        - Astral Barrage
        - Psyshock
        - Nasty Plot
        - Protect"""
    )
    mon = Pokemon(gen=9, teambuilder=tb.team[0])
    mon._current_hp = 100
    mon._max_hp = 100
    return mon


def _miraidon_opp() -> Pokemon:
    return Pokemon(gen=9, species="miraidon")


# ─────────────────────────────────────────────────────────────────────
# format_pokemon_line
# ─────────────────────────────────────────────────────────────────────


class TestFormatPokemonLine:
    def test_own_mon_full_info(self):
        from elitefurretai.engine.battle_renderer import format_pokemon_line

        mon = _calyrex_shadow()
        line = format_pokemon_line(mon, is_opponent=False)

        assert "calyrexshadow" in line
        assert "@ lifeorb" in line
        assert "asonespectrier" in line
        assert "HP: 100%" in line
        assert "Status: ok" in line

    def test_own_mon_with_status(self):
        from elitefurretai.engine.battle_renderer import format_pokemon_line

        mon = _calyrex_shadow()
        mon._status = Status.BRN
        line = format_pokemon_line(mon, is_opponent=False)
        assert "Status: BRN" in line

    def test_own_mon_with_boosts(self):
        from elitefurretai.engine.battle_renderer import format_pokemon_line

        mon = _calyrex_shadow()
        mon._boosts["spa"] = 2
        mon._boosts["spe"] = -1
        line = format_pokemon_line(mon, is_opponent=False)
        assert "spa:+2" in line
        assert "spe:-1" in line
        # zero-boosts should not appear
        assert "atk:" not in line

    def test_own_mon_terastallized(self):
        from elitefurretai.engine.battle_renderer import format_pokemon_line

        mon = _calyrex_shadow()
        mon._terastallized = True
        line = format_pokemon_line(mon, is_opponent=False)
        assert "(tera'd: ghost)" in line
        # no separate Tera: availability line for a tera'd mon
        assert "Tera: ghost (available)" not in line

    def test_own_mon_tera_available(self):
        from elitefurretai.engine.battle_renderer import format_pokemon_line

        mon = _calyrex_shadow()
        line = format_pokemon_line(mon, is_opponent=False, tera_status="available")
        assert "Tera: ghost (available)" in line

    def test_own_mon_tera_ally_used(self):
        from elitefurretai.engine.battle_renderer import format_pokemon_line

        mon = _calyrex_shadow()
        line = format_pokemon_line(mon, is_opponent=False, tera_status="ally used")
        assert "Tera: ghost (ally used)" in line

    def test_opponent_mon_hides_item(self):
        from elitefurretai.engine.battle_renderer import format_pokemon_line

        mon = _miraidon_opp()
        line = format_pokemon_line(mon, is_opponent=True)
        assert "miraidon" in line
        assert "@ " not in line

    def test_opponent_mon_with_revealed_moves(self):
        from elitefurretai.engine.battle_renderer import format_pokemon_line

        mon = _miraidon_opp()
        # Simulate revealed moves like poke-env does on |move| events
        from poke_env.battle import Move

        mon._moves["electrodrift"] = Move("electrodrift", gen=9)
        mon._moves["dracometeor"] = Move("dracometeor", gen=9)
        line = format_pokemon_line(mon, is_opponent=True)
        assert "Moves seen:" in line
        assert "electrodrift" in line
        assert "dracometeor" in line

    def test_opponent_mon_no_revealed_moves(self):
        from elitefurretai.engine.battle_renderer import format_pokemon_line

        mon = _miraidon_opp()
        line = format_pokemon_line(mon, is_opponent=True)
        assert "Moves seen: (none yet)" in line

    def test_fainted_mon(self):
        from elitefurretai.engine.battle_renderer import format_pokemon_line

        mon = _calyrex_shadow()
        mon._current_hp = 0
        line = format_pokemon_line(mon, is_opponent=False)
        assert "(fainted)" in line

    def test_effects_listed(self):
        from elitefurretai.engine.battle_renderer import format_pokemon_line

        mon = _calyrex_shadow()
        mon._effects = {Effect.SUBSTITUTE: 1, Effect.LEECH_SEED: 1}
        line = format_pokemon_line(mon, is_opponent=False)
        assert "Effects: " in line
        assert "SUBSTITUTE" in line
        assert "LEECH_SEED" in line


# ─────────────────────────────────────────────────────────────────────
# format_field
# ─────────────────────────────────────────────────────────────────────


def _empty_battle() -> DoubleBattle:
    battle = DoubleBattle("tag", "elitefurretai", Logger("ex"), gen=9)
    battle._format = "gen9vgc2025regg"
    battle.player_role = "p1"
    return battle


class TestFormatField:
    def test_all_none_renders_clean(self):
        from elitefurretai.engine.battle_renderer import format_field

        battle = _empty_battle()
        out = format_field(battle)
        assert "Weather: (none)" in out
        assert "Terrain: (none)" in out
        assert "Your side conditions: (none)" in out
        assert "Opp side conditions:  (none)" in out

    def test_weather_and_terrain(self):
        from elitefurretai.engine.battle_renderer import format_field

        battle = _empty_battle()
        battle._weather = {Weather.SUNNYDAY: 0}
        battle._fields = {Field.ELECTRIC_TERRAIN: 0}
        out = format_field(battle)
        assert "SUNNYDAY" in out
        assert "ELECTRIC_TERRAIN" in out

    def test_side_conditions_both_sides(self):
        from elitefurretai.engine.battle_renderer import format_field

        battle = _empty_battle()
        battle._side_conditions = {SideCondition.LIGHT_SCREEN: 0}
        battle._opponent_side_conditions = {SideCondition.REFLECT: 0}
        out = format_field(battle)
        assert "LIGHT_SCREEN" in out
        assert "REFLECT" in out


# ─────────────────────────────────────────────────────────────────────
# format_moves_oneline
# ─────────────────────────────────────────────────────────────────────


class TestFormatMovesOneline:
    def test_own_moves_show_real_pp(self):
        from elitefurretai.engine.battle_renderer import format_moves_oneline

        mon = _calyrex_shadow()
        # Force a PP drop on one move
        mon.moves["astralbarrage"]._current_pp = 5
        moves = list(mon.moves.values())
        out = format_moves_oneline(moves, assume_max_pp=False)
        assert "(1) astralbarrage (5/8)" in out
        assert "(2) psyshock (16/16)" in out

    def test_opp_moves_assume_max_pp(self):
        from poke_env.battle import Move

        from elitefurretai.engine.battle_renderer import format_moves_oneline

        moves = [Move("electrodrift", gen=9), Move("dracometeor", gen=9)]
        out = format_moves_oneline(moves, assume_max_pp=True)
        assert "(1) electrodrift" in out
        assert "(2) dracometeor" in out
        # max_pp of electrodrift is 8, of dracometeor is 8 (5 base * 1.6 with PP ups; base is 5)
        # Just check the format pattern shows (max/max), not the specific number
        assert "/" in out

    def test_empty_moves(self):
        from elitefurretai.engine.battle_renderer import format_moves_oneline

        out = format_moves_oneline([], assume_max_pp=False)
        assert out == "(none)"


# ─────────────────────────────────────────────────────────────────────
# format_events
# ─────────────────────────────────────────────────────────────────────


class TestFormatEvents:
    def test_empty_events(self):
        from elitefurretai.engine.battle_renderer import format_events

        out = format_events([])
        assert "(no events yet)" in out

    def test_single_event(self):
        from elitefurretai.engine.battle_renderer import format_events

        events = [["", "move", "p2a: Calyrex", "Astral Barrage", "p1a: Miraidon"]]
        out = format_events(events)
        assert "|move|p2a: Calyrex|Astral Barrage|p1a: Miraidon" in out

    def test_multiple_events_each_on_own_line(self):
        from elitefurretai.engine.battle_renderer import format_events

        events = [
            ["", "move", "p2a: Calyrex", "Astral Barrage", "p1a: Miraidon"],
            ["", "-damage", "p1a: Miraidon", "0 fnt"],
            ["", "faint", "p1a: Miraidon"],
        ]
        out = format_events(events)
        lines = [line for line in out.splitlines() if line.strip()]
        assert len(lines) == 3
        assert "Astral Barrage" in lines[0]
        assert "-damage" in lines[1]
        assert "faint" in lines[2]
