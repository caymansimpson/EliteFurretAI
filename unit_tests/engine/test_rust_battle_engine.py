# -*- coding: utf-8 -*-

from types import SimpleNamespace
from typing import Any, Dict, List, Optional, cast

from elitefurretai.engine.rust_battle_engine import (
    CachedRustBattleBinding,
    RustBattleEngine,
    RustBattleSideSnapshot,
    adapt_rust_battle,
    pokepaste_to_rust_team,
    pokepaste_to_rust_team_json,
)


class FakeRustBattle:
    def __init__(self):
        self._ended = False
        self._winner: Optional[str] = None
        self._turn = 0
        self.request_json_calls = 0
        self._requests: Dict[str, Optional[str]] = {
            "p1": '{"teamPreview": true}',
            "p2": '{"teamPreview": true}',
        }
        self._messages: Dict[str, List[str]] = {"p1": [], "p2": []}
        self._pending: Dict[str, str] = {}

    @property
    def ended(self) -> bool:
        return self._ended

    @property
    def winner(self) -> Optional[str]:
        return self._winner

    @property
    def turn(self) -> int:
        return self._turn

    def get_messages(self, side: str) -> List[str]:
        messages = list(self._messages[side])
        self._messages[side].clear()
        return messages

    def get_request_json(self, side: str) -> Optional[str]:
        self.request_json_calls += 1
        return self._requests.get(side)

    def get_request_dict(self, side: str) -> Optional[Dict[str, Any]]:
        raw_request = self.get_request_json(side)
        if raw_request is None:
            return None
        if raw_request == '{"teamPreview": true}':
            return {"teamPreview": True}
        if raw_request == '{"active": [{}, {}]}':
            return {"active": [{}, {}]}
        raise AssertionError(f"Unexpected request payload: {raw_request}")

    def get_side_snapshot(self, side: str):
        return SimpleNamespace(
            side=side,
            request=self.get_request_dict(side),
            raw_request=self.get_request_json(side),
            pending_messages=tuple(self._messages[side]),
            turn=self._turn,
            ended=self._ended,
            winner=self._winner,
        )

    def choose(self, side: str, choice: str) -> bool:
        self._pending[side] = choice
        if set(self._pending) != {"p1", "p2"}:
            return True

        if self._requests["p1"] == '{"teamPreview": true}':
            self._turn = 1
            self._requests = {
                "p1": '{"active": [{}, {}]}',
                "p2": '{"active": [{}, {}]}',
            }
            self._messages["p1"].append("|turn|1")
            self._messages["p2"].append("|turn|1")
        else:
            self._turn = 2
            self._ended = True
            self._winner = "Alice"
            self._requests = cast(Dict[str, Optional[str]], {"p1": None, "p2": None})
            self._messages["p1"].extend(["|turn|2", "|win|Alice"])
            self._messages["p2"].extend(["|turn|2", "|win|Alice"])

        self._pending.clear()
        return True


def _sample_team() -> List[dict]:
    return [
        {
            "name": f"Mon{i}",
            "species": "Furret",
            "item": "",
            "ability": "Frisk",
            "gender": "",
            "level": 50,
            "nature": "Jolly",
            "teraType": "Normal",
            "moves": ["Protect"],
            "evs": {"hp": 0, "atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0},
            "ivs": {"hp": 31, "atk": 31, "def": 31, "spa": 31, "spd": 31, "spe": 31},
        }
        for i in range(6)
    ]


def test_rust_battle_engine_uses_standalone_double_battle_state():
    engine = RustBattleEngine(
        rust_battle=cast(Any, FakeRustBattle()),
        battle_tag="rust-battle-1",
        p1_username="Alice",
        p2_username="Bob",
        p1_team=_sample_team(),
        p2_team=_sample_team(),
    )

    assert engine.battle_for("p1").player_username == "Alice"
    assert engine.battle_for("p1").opponent_username == "Bob"
    assert engine.battle_for("p2").player_username == "Bob"
    assert len(engine.battle_for("p1").teampreview_team) == 6
    assert engine.request_type("p1") == "teamPreview"
    assert engine.needs_action("p1") is True


def test_rust_battle_engine_step_updates_turn_and_winner():
    fake = FakeRustBattle()
    engine = RustBattleEngine(
        rust_battle=cast(Any, fake),
        battle_tag="rust-battle-2",
        p1_username="Alice",
        p2_username="Bob",
        p1_team=_sample_team(),
        p2_team=_sample_team(),
    )

    done, p1_accepted, p2_accepted = engine.step("team 1234", "team 1234")
    assert done is False
    assert p1_accepted is True
    assert p2_accepted is True
    assert engine.turn == 1
    assert engine.battle_for("p1").turn == 1
    assert engine.request_type("p1") == "move"

    done, p1_accepted, p2_accepted = engine.step("move 1, move 1", "move 1, move 1")
    assert done is True
    assert p1_accepted is True
    assert p2_accepted is True
    assert engine.ended is True
    assert engine.winner == "Alice"
    assert engine.battle_for("p1").finished is True


def test_rust_battle_engine_keeps_recent_protocol_history_per_side():
    fake = FakeRustBattle()
    engine = RustBattleEngine(
        rust_battle=cast(Any, fake),
        battle_tag="rust-battle-history",
        p1_username="Alice",
        p2_username="Bob",
        p1_team=_sample_team(),
        p2_team=_sample_team(),
    )

    assert engine.protocol_history("p1") == []

    done, _, _ = engine.step("team 1234", "team 1234")

    assert done is False
    assert engine.protocol_history("p1") == [{"raw": "|turn|1", "normalized": "|turn|1"}]


def test_rust_battle_engine_caches_requests_between_drain_cycles():
    fake = FakeRustBattle()
    engine = RustBattleEngine(
        rust_battle=cast(Any, fake),
        battle_tag="rust-battle-cache",
        p1_username="Alice",
        p2_username="Bob",
        p1_team=_sample_team(),
        p2_team=_sample_team(),
    )

    assert fake.request_json_calls == 2
    assert engine.request_json("p1") == {"teamPreview": True}
    assert engine.request_json("p1") == {"teamPreview": True}
    assert engine.request_type("p1") == "teamPreview"
    assert fake.request_json_calls == 2


def test_cached_rust_battle_binding_exposes_dict_and_side_snapshot():
    fake = FakeRustBattle()
    binding = CachedRustBattleBinding(fake)

    assert binding.get_request_dict("p1") == {"teamPreview": True}
    assert binding.get_request_dict("p1") == {"teamPreview": True}
    snapshot = binding.get_side_snapshot("p1")

    assert snapshot.side == "p1"
    assert snapshot.request == {"teamPreview": True}
    assert snapshot.raw_request == '{"teamPreview": true}'
    assert snapshot.turn == 0
    assert fake.request_json_calls == 1


def test_adapt_rust_battle_returns_cached_adapter():
    binding = adapt_rust_battle(FakeRustBattle())

    assert isinstance(binding, CachedRustBattleBinding)


class FakeCorruptedRustBattle:
    def __init__(self):
        self._ended = False
        self._winner: Optional[str] = None
        self._turn = 10

    @property
    def ended(self) -> bool:
        return self._ended

    @property
    def winner(self) -> Optional[str]:
        return self._winner

    @property
    def turn(self) -> int:
        return self._turn

    def choose(self, side: str, choice: str) -> bool:
        del side, choice
        return True

    def get_messages(self, side: str) -> List[str]:
        del side
        return []

    def get_request_json(self, side: str) -> Optional[str]:
        del side
        return None

    def get_request_dict(self, side: str) -> Optional[Dict[str, Any]]:
        return {
            "active": [
                {
                    "moves": [
                        {"id": "woodhammer", "target": "", "disabled": False, "pp": 16},
                        {"id": "highhorsepower", "target": "", "disabled": False, "pp": 16},
                    ],
                    "canTerastallize": "Water",
                },
                {
                    "moves": [
                        {"id": "spore", "target": "", "disabled": False, "pp": 16},
                        {"id": "pollenpuff", "target": "", "disabled": False, "pp": 16},
                    ],
                    "canTerastallize": "Water",
                },
            ],
            "noCancel": True,
            "side": {
                "id": side,
                "pokemon": [
                    {
                        "ident": f"{side}: Calyrex",
                        "active": False,
                        "condition": "0 fnt",
                        "moves": ["glaciallance", "highhorsepower", "trickroom", "encore"],
                    },
                    {
                        "ident": f"{side}: Iron Valiant",
                        "active": False,
                        "condition": "0 fnt",
                        "moves": ["moonblast", "knockoff", "encore", "coaching"],
                    },
                    {
                        "ident": f"{side}: Amoonguss",
                        "active": True,
                        "condition": "219/219 100",
                        "moves": ["spore", "pollenpuff", "ragepowder", "clearsmog"],
                    },
                    {
                        "ident": f"{side}: Pelipper",
                        "active": False,
                        "condition": "130/167 100",
                        "moves": ["weatherball", "hurricane", "terablast", "wideguard"],
                    },
                    {
                        "ident": f"{side}: Thundurus",
                        "active": False,
                        "condition": "32/181 100",
                        "moves": ["thunderbolt", "taunt", "eerieimpulse", "sunnyday"],
                    },
                    {
                        "ident": f"{side}: Rillaboom",
                        "active": True,
                        "condition": "158/207 100",
                        "moves": ["woodhammer", "highhorsepower", "uturn", "fakeout"],
                    },
                ],
            },
        }

    def get_side_snapshot(self, side: str) -> RustBattleSideSnapshot:
        return RustBattleSideSnapshot(
            side=side,
            request=self.get_request_dict(side),
            raw_request='{"raw": true}',
            pending_messages=(),
            turn=self._turn,
            ended=self._ended,
            winner=self._winner,
        )


def test_rust_battle_engine_sanitizes_move_request_for_snapshot_and_cache():
    engine = RustBattleEngine(
        rust_battle=cast(Any, FakeCorruptedRustBattle()),
        battle_tag="rust-battle-sanitized",
        p1_username="Alice",
        p2_username="Bob",
        p1_team=_sample_team(),
        p2_team=_sample_team(),
    )
    engine._active_slots["p1"] = ["p1a: Amoonguss", "p1b: Rillaboom"]
    engine._refresh_request_cache()

    request = cast(Dict[str, Any], engine.request_json("p1"))
    snapshot = engine.side_snapshot("p1")
    pokemon_list = cast(List[Dict[str, Any]], request["side"]["pokemon"])
    active_moves = cast(List[Dict[str, Any]], request["active"])

    assert [pokemon["ident"] for pokemon in pokemon_list[:2]] == ["p1: Amoonguss", "p1: Rillaboom"]
    assert [pokemon["active"] for pokemon in pokemon_list[:4]] == [True, True, False, False]
    assert [pokemon["condition"] for pokemon in pokemon_list[:4]] == ["219/219", "158/207", "0 fnt", "0 fnt"]
    assert [active_moves[0]["moves"][0]["id"], active_moves[1]["moves"][0]["id"]] == ["spore", "woodhammer"]
    assert [pokemon["_request_index"] for pokemon in pokemon_list[:2]] == [2, 5]
    assert snapshot.request == request
    assert snapshot.raw_request == '{"raw": true}'


def test_rust_battle_engine_reconciles_active_slots_from_request_front_pair():
    battle = SimpleNamespace(
        player_role="p1",
        _active_pokemon={
            "p1a": "stale-left",
            "p1b": "stale-right",
        },
    )
    requested_pokemon = {
        "p1: Amoonguss": object(),
        "p1: Rillaboom": object(),
    }

    def fake_get_pokemon(ident: str, force_self_team: bool, details: str):
        assert force_self_team is True
        assert details.endswith(", L50")
        return requested_pokemon[ident]

    battle.get_pokemon = fake_get_pokemon

    RustBattleEngine._reconcile_active_slots_from_request(
        cast(Any, battle),
        {
            "active": [{}, {}],
            "side": {
                "pokemon": [
                    {"ident": "p1: Amoonguss", "details": "amoonguss, L50"},
                    {"ident": "p1: Rillaboom", "details": "rillaboom, L50"},
                    {"ident": "p1: Pelipper", "details": "pelipper, L50"},
                ]
            },
        },
    )

    assert battle._active_pokemon["p1a"] is requested_pokemon["p1: Amoonguss"]
    assert battle._active_pokemon["p1b"] is requested_pokemon["p1: Rillaboom"]


def test_pokepaste_to_rust_team_parses_basic_vgc_team():
    team = """Furret @ Choice Band
Ability: Frisk
Level: 50
Tera Type: Normal
EVs: 4 HP / 252 Atk / 252 Spe
Adamant Nature
- Double-Edge
- U-turn
- Trick
- Knock Off

Smeargle (Maushold) @ Focus Sash
Ability: Technician
Gender: F
EVs: 252 HP / 4 Def / 252 Spe
Jolly Nature
IVs: 0 Atk / 30 SpA
- Follow Me
- Spore
- Fake Out
- Protect
"""

    converted = pokepaste_to_rust_team(team)

    assert len(converted) == 2
    assert converted[0]["name"] == "Furret"
    assert converted[0]["species"] == "Furret"
    assert converted[0]["item"] == "Choice Band"
    assert converted[0]["ability"] == "Frisk"
    assert converted[0]["nature"] == "Adamant"
    assert converted[0]["teraType"] == "Normal"
    assert converted[0]["evs"] == {
        "hp": 4,
        "atk": 252,
        "def": 0,
        "spa": 0,
        "spd": 0,
        "spe": 252,
    }
    assert converted[1]["name"] == "Smeargle"
    assert converted[1]["species"] == "Maushold"
    assert converted[1]["gender"] == "F"
    ivs = cast(Dict[str, Any], converted[1]["ivs"])
    assert ivs["atk"] == 0
    assert ivs["spa"] == 30


def test_pokepaste_to_rust_team_json_serializes_output():
    team = """Furret
Ability: Frisk
- Protect
"""

    as_json = pokepaste_to_rust_team_json(team)

    assert '"species": "Furret"' in as_json
    assert '"moves": ["Protect"]' in as_json


def test_pokepaste_to_rust_team_parses_header_gender_without_corrupting_species():
    team = """Gardevoir (F) @ Sitrus Berry
Ability: Trace
- Protect

Smeargle (Maushold) (F) @ Focus Sash
Ability: Own Tempo
- Follow Me
"""

    converted = pokepaste_to_rust_team(team)

    assert converted[0]["name"] == "Gardevoir"
    assert converted[0]["species"] == "Gardevoir"
    assert converted[0]["gender"] == "F"
    assert converted[1]["name"] == "Smeargle"
    assert converted[1]["species"] == "Maushold"
    assert converted[1]["gender"] == "F"
