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


# ─────────────────────────────────────────────────────────────────────
# format_action_reference
# ─────────────────────────────────────────────────────────────────────


class TestFormatActionReference:
    def test_contains_banner_and_target_codes(self):
        from elitefurretai.engine.battle_renderer import format_action_reference

        out = format_action_reference()
        assert "ACTION REFERENCE" in out
        assert "1,  2" in out or "1, 2" in out
        assert "-1, -2" in out
        assert "comma-separated" in out
        assert "tera" in out


# ─────────────────────────────────────────────────────────────────────
# format_teampreview
# ─────────────────────────────────────────────────────────────────────


def _miraidon_preview() -> Pokemon:
    return Pokemon(gen=9, species="miraidon")


def _flutter_mane_preview() -> Pokemon:
    return Pokemon(gen=9, species="fluttermane")


class TestFormatTeampreview:
    def test_contains_banner_and_format(self):
        from elitefurretai.engine.battle_renderer import format_teampreview

        battle = _empty_battle()
        battle._teampreview_team = [_calyrex_shadow()]
        battle._teampreview_opponent_team = [_miraidon_preview()]
        out = format_teampreview(battle)
        assert "TEAM PREVIEW" in out
        assert "gen9vgc2025regg" in out

    def test_lists_own_team_with_ability_item_tera(self):
        from elitefurretai.engine.battle_renderer import format_teampreview

        battle = _empty_battle()
        mon = _calyrex_shadow()
        battle._teampreview_team = [mon]
        out = format_teampreview(battle)
        assert "calyrexshadow" in out
        assert "lifeorb" in out
        assert "asonespectrier" in out
        assert "Tera: ghost" in out

    def test_lists_opp_team_by_species_only(self):
        from elitefurretai.engine.battle_renderer import format_teampreview

        battle = _empty_battle()
        battle._teampreview_opponent_team = [
            _miraidon_preview(),
            _flutter_mane_preview(),
        ]
        out = format_teampreview(battle)
        assert "miraidon" in out
        assert "fluttermane" in out

    def test_no_hp_bars_in_teampreview(self):
        from elitefurretai.engine.battle_renderer import format_teampreview

        battle = _empty_battle()
        battle._teampreview_team = [_calyrex_shadow()]
        out = format_teampreview(battle)
        # The U+2588 full-block char is the HP-bar char; teampreview shouldn't use it
        assert "█" not in out


# ─────────────────────────────────────────────────────────────────────
# format_battle_state
# ─────────────────────────────────────────────────────────────────────


def _populated_battle() -> DoubleBattle:
    """A battle with 2 active mons per side and a 2-mon bench."""
    battle = _empty_battle()
    battle._turn = 1

    own_a = _calyrex_shadow()
    own_a._active = True
    own_b = _calyrex_shadow()
    own_b._species = "urshifurapidstrike"
    own_b._active = True
    bench_a = _calyrex_shadow()
    bench_a._species = "incineroar"
    bench_b = _calyrex_shadow()
    bench_b._species = "rillaboom"
    battle._team = {
        "p1: Calyrex": own_a,
        "p1: Urshifu": own_b,
        "p1: Incineroar": bench_a,
        "p1: Rillaboom": bench_b,
    }
    battle._active_pokemon = {"p1a": own_a, "p1b": own_b}

    opp_a = _miraidon_preview()
    opp_a._active = True
    opp_a._current_hp = 100
    opp_a._max_hp = 100
    opp_b = _flutter_mane_preview()
    opp_b._active = True
    opp_b._current_hp = 100
    opp_b._max_hp = 100
    battle._opponent_team = {"p2: Miraidon": opp_a, "p2: Flutter Mane": opp_b}
    battle._opponent_active_pokemon = {"p2a": opp_a, "p2b": opp_b}

    battle._available_moves = [list(own_a.moves.values()), list(own_b.moves.values())]
    battle._available_switches = [[bench_a, bench_b], [bench_a, bench_b]]
    battle._can_tera = [True, True]
    return battle


class TestFormatBattleState:
    def test_contains_top_level_sections(self):
        from elitefurretai.engine.battle_renderer import format_battle_state

        battle = _populated_battle()
        out = format_battle_state(battle)
        assert "Turn 1" in out
        assert "FIELD" in out
        assert "ACTIVE POKEMON" in out
        assert "YOUR OPTIONS" in out
        assert "Bench" in out

    def test_own_slots_carry_negative_target_codes(self):
        from elitefurretai.engine.battle_renderer import format_battle_state

        battle = _populated_battle()
        out = format_battle_state(battle)
        assert "Slot 1 (-1)" in out
        assert "Slot 2 (-2)" in out

    def test_opp_slots_carry_positive_target_codes(self):
        from elitefurretai.engine.battle_renderer import format_battle_state

        battle = _populated_battle()
        out = format_battle_state(battle)
        assert "Slot 1 (1)" in out
        assert "Slot 2 (2)" in out

    def test_options_lists_moves_per_slot(self):
        from elitefurretai.engine.battle_renderer import format_battle_state

        battle = _populated_battle()
        out = format_battle_state(battle)
        assert "(1) astralbarrage" in out

    def test_bench_lists_switchable_mons(self):
        from elitefurretai.engine.battle_renderer import format_battle_state

        battle = _populated_battle()
        out = format_battle_state(battle)
        assert "incineroar" in out
        assert "rillaboom" in out

    def test_last_turn_section_when_no_prior_observation(self):
        from elitefurretai.engine.battle_renderer import format_battle_state

        battle = _populated_battle()
        out = format_battle_state(battle)
        assert "LAST TURN" in out


# ─────────────────────────────────────────────────────────────────────
# format_observation
# ─────────────────────────────────────────────────────────────────────


class TestFormatObservation:
    def test_empty_observation_renders_no_events_marker(self):
        from poke_env.battle import Observation

        from elitefurretai.engine.battle_renderer import format_observation

        out = format_observation(Observation())
        assert "(no events yet)" in out

    def test_observation_with_events(self):
        from poke_env.battle import Observation

        from elitefurretai.engine.battle_renderer import format_observation

        obs = Observation(
            events=[
                ["", "move", "p2a: Calyrex", "Astral Barrage", "p1a: Miraidon"],
                ["", "faint", "p1a: Miraidon"],
            ]
        )
        out = format_observation(obs)
        assert "Astral Barrage" in out
        assert "faint" in out


# ─────────────────────────────────────────────────────────────────────
# format_battle_log
# ─────────────────────────────────────────────────────────────────────


class TestFormatBattleLog:
    def test_header_contains_battle_tag_and_usernames(self):
        from elitefurretai.engine.battle_renderer import format_battle_log

        battle = _empty_battle()
        battle._player_username = "me"
        battle._opponent_username = "you"
        out = format_battle_log(battle)
        assert "tag" in out
        assert "me" in out
        assert "you" in out

    def test_teampreview_teams_listed(self):
        from elitefurretai.engine.battle_renderer import format_battle_log

        battle = _empty_battle()
        battle._player_username = "me"
        battle._opponent_username = "you"
        battle._teampreview_team = [_calyrex_shadow()]
        battle._teampreview_opponent_team = [_miraidon_preview()]
        out = format_battle_log(battle)
        assert "Calyrex" in out
        assert "Miraidon" in out

    def test_observations_rendered_per_turn(self):
        from poke_env.battle import Observation

        from elitefurretai.engine.battle_renderer import format_battle_log

        battle = _empty_battle()
        battle._player_username = "me"
        battle._opponent_username = "you"
        battle._observations = {
            1: Observation(events=[["", "move", "p2a: Calyrex", "Astral Barrage"]]),
            2: Observation(events=[["", "faint", "p1a: Miraidon"]]),
        }
        out = format_battle_log(battle)
        assert "Turn #1" in out
        assert "Turn #2" in out
        assert "Astral Barrage" in out
        assert "faint" in out

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
