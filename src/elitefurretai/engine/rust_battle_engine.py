"""Rust battle synchronization helpers for the simplified fallback backend.

This module keeps the Rust backend runnable without preserving the old benchmark
ablation surface. The maintained path is now intentionally small:

- one cached binding adapter around the compiled Rust battle object
- one battle-engine synchronizer that replays sanitized protocol into poke-env
- one request cache that avoids repeated JSON parsing in the Python fallback path

Stage 2 showed that request sanitization and request caching were worth keeping,
while multiple adapter variants only added maintenance overhead once the project
chose to move forward with Showdown as the primary RL backend.

This module also owns the Rust binding's PokePaste team-conversion helpers so
the Rust engine boundary stays consolidated in one place.
"""

import json
import logging
import re
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Protocol, Sequence, Set, Tuple, cast

from poke_env.battle import DoubleBattle

from elitefurretai.etl.battle_data import team_from_json

_UNSLOTTED_IDENT_RE = re.compile(r"^(p[1-4]):\s+(.*)$")
_BARE_SLOT_RE = re.compile(r"^(p[1-4][ab])$")
_PERSPECTIVE_SLOT_RE = re.compile(r"^([12][ab])(?::\s+(.*))?$")
_STAT_NAME_MAP = {
    "hp": "hp",
    "atk": "atk",
    "def": "def",
    "spa": "spa",
    "spd": "spd",
    "spe": "spe",
    "spatk": "spa",
    "spdef": "spd",
    "spdefense": "spd",
    "spattack": "spa",
    "speed": "spe",
}
_HEADER_RE = re.compile(r"^(?P<lead>.*?)(?: @ (?P<item>.*))?$")
_NICK_SPECIES_RE = re.compile(r"^(?P<name>.+?) \((?P<species>.+)\)$")
_TRAILING_GENDER_RE = re.compile(r"^(?P<lead>.+?) \((?P<gender>[MF])\)$")


def _default_stats(default: int) -> Dict[str, int]:
    return {
        "hp": default,
        "atk": default,
        "def": default,
        "spa": default,
        "spd": default,
        "spe": default,
    }


def _normalize_stat_name(raw_name: str) -> str:
    normalized = raw_name.strip().lower().replace(".", "")
    if normalized not in _STAT_NAME_MAP:
        raise ValueError(f"Unsupported stat name in team export: {raw_name!r}")
    return _STAT_NAME_MAP[normalized]


def _parse_stat_spread(raw_value: str, default: int) -> Dict[str, int]:
    stats = _default_stats(default)
    for chunk in raw_value.split("/"):
        part = chunk.strip()
        if not part:
            continue
        value_str, stat_name = part.split(" ", 1)
        stats[_normalize_stat_name(stat_name)] = int(value_str)
    return stats


def _parse_team_header(header: str) -> Dict[str, str]:
    match = _HEADER_RE.match(header.strip())
    if match is None:
        raise ValueError(f"Could not parse team header line: {header!r}")

    lead = (match.group("lead") or "").strip()
    item = match.group("item") or ""
    gender = ""

    gender_match = _TRAILING_GENDER_RE.match(lead)
    if gender_match is not None:
        lead = gender_match.group("lead").strip()
        gender = gender_match.group("gender").strip()

    name = lead
    species = lead
    nick_match = _NICK_SPECIES_RE.match(lead)
    if nick_match is not None:
        name = nick_match.group("name").strip()
        species = nick_match.group("species").strip()

    return {
        "name": name,
        "species": species,
        "gender": gender,
        "item": item.strip(),
    }


def pokepaste_to_rust_team(team: str, default_level: int = 50) -> List[Dict[str, object]]:
    """Convert a PokePaste export into dictionaries matching the Rust PokemonSet shape."""
    blocks = [block.strip() for block in team.replace("\r\n", "\n").split("\n\n") if block.strip()]
    if not blocks:
        raise ValueError("Team export is empty")

    converted: List[Dict[str, object]] = []
    for block in blocks:
        lines = [line.strip() for line in block.split("\n") if line.strip()]
        header = _parse_team_header(lines[0])
        pokemon: Dict[str, object] = {
            "name": header["name"],
            "species": header["species"],
            "item": header["item"],
            "ability": "",
            "gender": header["gender"],
            "level": default_level,
            "nature": "Serious",
            "teraType": None,
            "moves": [],
            "evs": _default_stats(0),
            "ivs": _default_stats(31),
        }

        for line in lines[1:]:
            if line.startswith("Ability:"):
                pokemon["ability"] = line.split(":", 1)[1].strip()
            elif line.startswith("Level:"):
                pokemon["level"] = int(line.split(":", 1)[1].strip())
            elif line.startswith("Gender:"):
                pokemon["gender"] = line.split(":", 1)[1].strip()
            elif line.startswith("Tera Type:"):
                pokemon["teraType"] = line.split(":", 1)[1].strip()
            elif line.startswith("EVs:"):
                pokemon["evs"] = _parse_stat_spread(line.split(":", 1)[1].strip(), 0)
            elif line.startswith("IVs:"):
                pokemon["ivs"] = _parse_stat_spread(line.split(":", 1)[1].strip(), 31)
            elif line.endswith(" Nature"):
                pokemon["nature"] = line[: -len(" Nature")].strip()
            elif line.startswith("- "):
                moves = pokemon["moves"]
                assert isinstance(moves, list)
                moves.append(line[2:].strip())

        converted.append(pokemon)

    return converted


def pokepaste_to_rust_team_json(team: str, default_level: int = 50) -> str:
    """Serialize a PokePaste export into a JSON string consumable by Rust bindings."""
    return json.dumps(pokepaste_to_rust_team(team, default_level=default_level))


class RustBattleBinding(Protocol):
    def choose(self, side: str, choice: str) -> bool: ...

    def get_messages(self, side: str) -> Sequence[str]: ...

    def get_request_json(self, side: str) -> Optional[str]: ...

    def get_request_dict(self, side: str) -> Optional[Dict[str, Any]]: ...

    def get_side_snapshot(self, side: str) -> "RustBattleSideSnapshot": ...

    @property
    def ended(self) -> bool: ...

    @property
    def winner(self) -> Optional[str]: ...

    @property
    def turn(self) -> int: ...


@dataclass
class RustBattleSideSnapshot:
    side: str
    request: Optional[Dict[str, Any]]
    raw_request: Optional[str]
    pending_messages: Tuple[str, ...]
    turn: int
    ended: bool
    winner: Optional[str]


class CachedRustBattleBinding:
    """Python wrapper that exposes dict-based request access around the builtin binding.

    The installed binding in this environment does not yet expose native PyO3 dict
    methods, so this wrapper centralizes JSON parsing and presents the interface the
    Python RL path wants to consume today.
    """

    def __init__(self, rust_battle: Any):
        self._rust_battle = rust_battle
        self._request_json_cache: Dict[str, Optional[str]] = {"p1": None, "p2": None}
        self._request_dict_cache: Dict[str, Optional[Dict[str, Any]]] = {"p1": None, "p2": None}
        self._message_cache: Dict[str, Tuple[str, ...]] = {"p1": (), "p2": ()}

    def choose(self, side: str, choice: str) -> bool:
        accepted = bool(self._rust_battle.choose(side, choice))
        self._request_json_cache["p1"] = None
        self._request_json_cache["p2"] = None
        self._request_dict_cache["p1"] = None
        self._request_dict_cache["p2"] = None
        return accepted

    def get_messages(self, side: str) -> Sequence[str]:
        messages = tuple(self._rust_battle.get_messages(side))
        self._message_cache[side] = messages
        return messages

    def get_request_json(self, side: str) -> Optional[str]:
        raw_request = self._request_json_cache.get(side)
        if raw_request is None:
            raw_request = self._rust_battle.get_request_json(side)
            self._request_json_cache[side] = raw_request
        return raw_request

    def get_request_dict(self, side: str) -> Optional[Dict[str, Any]]:
        request = self._request_dict_cache.get(side)
        if request is None:
            raw_request = self.get_request_json(side)
            if not raw_request:
                return None
            request = cast(Dict[str, Any], json.loads(raw_request))
            self._request_dict_cache[side] = request
        return request

    def get_side_snapshot(self, side: str) -> RustBattleSideSnapshot:
        return RustBattleSideSnapshot(
            side=side,
            request=self.get_request_dict(side),
            raw_request=self.get_request_json(side),
            pending_messages=self._message_cache.get(side, ()),
            turn=int(self.turn),
            ended=bool(self.ended),
            winner=self.winner,
        )

    @property
    def ended(self) -> bool:
        return bool(self._rust_battle.ended)

    @property
    def winner(self) -> Optional[str]:
        return self._rust_battle.winner

    @property
    def turn(self) -> int:
        return int(self._rust_battle.turn)


def adapt_rust_battle(
    rust_battle: Any,
) -> RustBattleBinding:
    return CachedRustBattleBinding(rust_battle)


def create_standalone_double_battle(
    battle_tag: str,
    player_username: str,
    opponent_username: str,
    perspective: str,
    team: Optional[List[Dict[str, Any]]] = None,
    gen: int = 9,
) -> DoubleBattle:
    battle_logger = logging.getLogger(player_username)
    battle_logger.setLevel(logging.ERROR)
    battle_logger.propagate = False
    battle = DoubleBattle(
        battle_tag,
        player_username,
        battle_logger,
        gen=gen,
    )
    battle.player_role = perspective
    battle.player_username = player_username
    battle.opponent_username = opponent_username
    if team is not None:
        sanitized_team = []
        for member in team:
            sanitized = dict(member)
            if sanitized.get("teraType") is None:
                sanitized.pop("teraType", None)
            sanitized_team.append(sanitized)
        battle.teampreview_team = [mon.to_pokemon() for mon in team_from_json(sanitized_team)]
    return battle


def _apply_protocol_line(battle: DoubleBattle, line: str) -> None:
    parts = line.split("|")
    if len(parts) <= 1 or parts[1] in {"", "t:"}:
        return
    if parts[1] == "win":
        battle.won_by(parts[2])
        return
    if parts[1] == "tie":
        battle.tied()
        return
    try:
        battle.parse_message(parts)
    except NotImplementedError:
        return
    except Exception:
        return


class RustBattleEngine:
    """Standalone battle adapter for the future Rust-backed RL path.

    This class intentionally does not depend on poke-env's Player class. It owns
    a Rust binding object plus two standalone DoubleBattle instances and keeps them
    synchronized from emitted protocol lines.
    """

    _PROTOCOL_HISTORY_LIMIT = 20

    def __init__(
        self,
        rust_battle: RustBattleBinding,
        battle_tag: str,
        p1_username: str,
        p2_username: str,
        p1_team: Optional[List[Dict[str, Any]]] = None,
        p2_team: Optional[List[Dict[str, Any]]] = None,
        gen: int = 9,
    ):
        self.rust_battle = rust_battle
        self.battle_tag = battle_tag
        self._active_slots: Dict[str, List[Optional[str]]] = {
            "p1": [None, None],
            "p2": [None, None],
        }
        self._request_cache: Dict[str, Optional[Dict[str, Any]]] = {
            "p1": None,
            "p2": None,
        }
        self._protocol_history: Dict[str, Deque[Dict[str, str]]] = {
            "p1": deque(maxlen=self._PROTOCOL_HISTORY_LIMIT),
            "p2": deque(maxlen=self._PROTOCOL_HISTORY_LIMIT),
        }
        self._protocol_log: Dict[str, List[Dict[str, str]]] = {
            "p1": [],
            "p2": [],
        }
        self.p1_battle = create_standalone_double_battle(
            battle_tag=battle_tag,
            player_username=p1_username,
            opponent_username=p2_username,
            perspective="p1",
            team=p1_team,
            gen=gen,
        )
        self.p2_battle = create_standalone_double_battle(
            battle_tag=battle_tag,
            player_username=p2_username,
            opponent_username=p1_username,
            perspective="p2",
            team=p2_team,
            gen=gen,
        )
        self._drain_messages()

    @property
    def ended(self) -> bool:
        return bool(self.rust_battle.ended)

    @property
    def winner(self) -> Optional[str]:
        return self.rust_battle.winner

    @property
    def turn(self) -> int:
        return int(self.rust_battle.turn)

    def battle_for(self, side: str) -> DoubleBattle:
        if side == "p1":
            return self.p1_battle
        if side == "p2":
            return self.p2_battle
        raise ValueError(f"Unknown side: {side!r}")

    def request_json(self, side: str) -> Optional[Dict[str, Any]]:
        if side not in self._request_cache:
            raise ValueError(f"Unknown side: {side!r}")
        return self._request_cache[side]

    def side_snapshot(self, side: str) -> RustBattleSideSnapshot:
        snapshot = self.rust_battle.get_side_snapshot(side)
        request = self.request_json(side)
        if request is None:
            return snapshot
        return RustBattleSideSnapshot(
            side=snapshot.side,
            request=request,
            raw_request=snapshot.raw_request,
            pending_messages=snapshot.pending_messages,
            turn=snapshot.turn,
            ended=snapshot.ended,
            winner=snapshot.winner,
        )

    def protocol_history(self, side: str, limit: int = 5) -> List[Dict[str, str]]:
        history = self._protocol_history.get(side)
        if history is None:
            raise ValueError(f"Unknown side: {side!r}")
        if limit <= 0:
            return []
        return list(history)[-limit:]

    def protocol_log(self, side: str) -> List[Dict[str, str]]:
        protocol_log = self._protocol_log.get(side)
        if protocol_log is None:
            raise ValueError(f"Unknown side: {side!r}")
        return list(protocol_log)

    def request_type(self, side: str) -> Optional[str]:
        if self.ended:
            return None
        request = self.request_json(side)
        if request is None:
            return None
        if request.get("wait"):
            return "wait"
        if request.get("teamPreview"):
            return "teamPreview"
        force_switch = request.get("forceSwitch")
        if isinstance(force_switch, list) and any(force_switch):
            return "switch"
        if "active" in request:
            return "move"
        return None

    def needs_action(self, side: str) -> bool:
        return self.request_type(side) in {"move", "switch", "teamPreview"}

    def choose(self, side: str, choice: str) -> bool:
        accepted = self.rust_battle.choose(side, choice)
        self._drain_messages()
        return accepted

    def step(
        self,
        p1_choice: Optional[str] = None,
        p2_choice: Optional[str] = None,
    ) -> Tuple[bool, bool, bool]:
        p1_accepted = True
        p2_accepted = True
        if p1_choice is not None:
            p1_accepted = bool(self.rust_battle.choose("p1", p1_choice))
        if p2_choice is not None:
            p2_accepted = bool(self.rust_battle.choose("p2", p2_choice))
        self._drain_messages()
        return self.ended, p1_accepted, p2_accepted

    def _drain_messages(self) -> None:
        for line in self.rust_battle.get_messages("p1"):
            normalized = self._normalize_protocol_line(line, perspective_side="p1")
            self._record_protocol_line("p1", line, normalized)
            _apply_protocol_line(self.p1_battle, normalized)
        for line in self.rust_battle.get_messages("p2"):
            normalized = self._normalize_protocol_line(line, perspective_side="p2")
            self._record_protocol_line("p2", line, normalized)
            _apply_protocol_line(self.p2_battle, normalized)
        self._refresh_request_cache()
        self._apply_requests()

    def _record_protocol_line(self, side: str, raw_line: str, normalized_line: str) -> None:
        history = self._protocol_history.get(side)
        if history is None:
            return
        record = {"raw": raw_line, "normalized": normalized_line}
        history.append(record)
        self._protocol_log[side].append(record)

    def _refresh_request_cache(self) -> None:
        self._request_cache["p1"] = self._read_request_from_binding("p1")
        self._request_cache["p2"] = self._read_request_from_binding("p2")

    def _read_request_from_binding(self, side: str) -> Optional[Dict[str, Any]]:
        request = self.rust_battle.get_request_dict(side)
        if request is None:
            return None
        return self._sanitize_request_for_poke_env(cast(Dict[str, Any], request), side)

    def _apply_requests(self) -> None:
        for side, battle in (("p1", self.p1_battle), ("p2", self.p2_battle)):
            request = self.request_json(side)
            if request is None:
                continue
            try:
                battle.parse_request(request)
                self._reconcile_active_slots_from_request(battle, request)
            except Exception:
                # Fall back to the minimal state update path if poke-env rejects an
                # edge-case request shape. This keeps the adapter functional while
                # still preferring full request parsing whenever possible.
                battle._last_request = request
                battle._wait = bool(request.get("wait"))
                battle._teampreview = bool(request.get("teamPreview"))
                force_switch = request.get("forceSwitch", [False, False])
                if isinstance(force_switch, list) and len(force_switch) == 2:
                    battle._force_switch = [bool(force_switch[0]), bool(force_switch[1])]
                self._reconcile_active_slots_from_request(battle, request)

    @staticmethod
    def _reconcile_active_slots_from_request(battle: DoubleBattle, request: Dict[str, Any]) -> None:
        active_requests = request.get("active")
        side = request.get("side")
        if not isinstance(active_requests, list) or not isinstance(side, dict):
            return
        pokemon_list = side.get("pokemon")
        if not isinstance(pokemon_list, list) or battle.player_role is None:
            return

        for active_pokemon_number, pokemon_dict in enumerate(pokemon_list[: len(active_requests)]):
            if not isinstance(pokemon_dict, dict):
                continue
            ident = pokemon_dict.get("ident")
            details = pokemon_dict.get("details")
            if not isinstance(ident, str) or not isinstance(details, str):
                continue
            pokemon = battle.get_pokemon(
                ident,
                force_self_team=True,
                details=details,
            )
            slot_key = f"{battle.player_role}{'a' if active_pokemon_number == 0 else 'b'}"
            battle._active_pokemon[slot_key] = pokemon

    def _sanitize_request_for_poke_env(self, request: Dict[str, Any], side: str) -> Dict[str, Any]:
        sanitized = json.loads(json.dumps(request))
        side_data = sanitized.get("side")
        if isinstance(side_data, dict):
            pokemon_list = side_data.get("pokemon")
            if isinstance(pokemon_list, list):
                self._annotate_request_indices(pokemon_list)
                active_requests = sanitized.get("active")
                if isinstance(active_requests, list):
                    self._rebuild_front_pair_from_active_slots(
                        pokemon_list,
                        active_requests=active_requests,
                        side_id=side_data.get("id") or side,
                    )
                else:
                    self._reorder_side_pokemon_for_parse_request(
                        pokemon_list,
                        active_requests=active_requests,
                        side_id=side_data.get("id") or side,
                    )
                for pokemon in pokemon_list:
                    if not isinstance(pokemon, dict):
                        continue
                    condition = pokemon.get("condition")
                    if isinstance(condition, str):
                        condition_parts = condition.split()
                        if len(condition_parts) >= 2 and condition_parts[-1].isdigit():
                            pokemon["condition"] = condition_parts[0]
        return sanitized

    @staticmethod
    def _annotate_request_indices(pokemon_list: List[Dict[str, Any]]) -> None:
        for request_index, pokemon in enumerate(pokemon_list):
            if isinstance(pokemon, dict):
                pokemon.setdefault("_request_index", request_index)

    def _rebuild_front_pair_from_active_slots(
        self,
        pokemon_list: List[Dict[str, Any]],
        active_requests: List[Any],
        side_id: Optional[str],
    ) -> None:
        if side_id not in {"p1", "p2"}:
            return
        if not active_requests:
            return

        matched_request_indices = self._match_active_requests_to_pokemon(active_requests, pokemon_list)
        if not any(index is not None for index in matched_request_indices):
            return

        ordered_front_indices: List[int] = []
        ordered_active_requests: List[Any] = []
        used_request_indices: Set[int] = set()

        for occupant in self._active_slots.get(side_id, [None, None]):
            matched_pair = self._find_request_index_for_occupant(
                pokemon_list,
                matched_request_indices,
                occupant,
                used_request_indices,
            )
            if matched_pair is None:
                continue
            request_index, pokemon_index = matched_pair
            ordered_front_indices.append(pokemon_index)
            ordered_active_requests.append(active_requests[request_index])
            used_request_indices.add(request_index)

        for request_index, matched_pokemon_index in enumerate(matched_request_indices):
            if request_index in used_request_indices or matched_pokemon_index is None:
                continue
            ordered_front_indices.append(matched_pokemon_index)
            ordered_active_requests.append(active_requests[request_index])
            used_request_indices.add(request_index)

        if not ordered_front_indices:
            return

        ordered: List[Dict[str, Any]] = []
        used_pokemon_indices = set(ordered_front_indices)
        for pokemon_index in ordered_front_indices:
            ordered.append(pokemon_list[pokemon_index])
        for pokemon_index, pokemon in enumerate(pokemon_list):
            if pokemon_index in used_pokemon_indices:
                continue
            ordered.append(pokemon)

        for pokemon_index, pokemon in enumerate(ordered):
            pokemon["active"] = pokemon_index < len(ordered_front_indices)

        pokemon_list[:] = ordered
        active_requests[:] = ordered_active_requests

    def _match_active_requests_to_pokemon(
        self,
        active_requests: List[Any],
        pokemon_list: List[Dict[str, Any]],
    ) -> List[Optional[int]]:
        matched_indices: List[Optional[int]] = []
        used_pokemon_indices: set[int] = set()
        move_id_to_indices: Dict[Tuple[str, ...], List[int]] = {}

        for pokemon_index, pokemon in enumerate(pokemon_list):
            move_ids = tuple(pokemon.get("moves", [])) if isinstance(pokemon, dict) else ()
            if move_ids:
                move_id_to_indices.setdefault(move_ids, []).append(pokemon_index)

        for active_request in active_requests:
            move_ids = self._request_move_ids(active_request)
            match_index: Optional[int] = None
            for pokemon_index in move_id_to_indices.get(move_ids, []):
                if pokemon_index in used_pokemon_indices:
                    continue
                match_index = pokemon_index
                break
            if match_index is None:
                for pokemon_index, pokemon in enumerate(pokemon_list):
                    if pokemon_index in used_pokemon_indices:
                        continue
                    if not isinstance(pokemon, dict):
                        continue
                    if self._pokemon_matches_request_move_ids(pokemon, move_ids):
                        match_index = pokemon_index
                        break
            matched_indices.append(match_index)
            if match_index is not None:
                used_pokemon_indices.add(match_index)

        return matched_indices

    def _find_request_index_for_occupant(
        self,
        pokemon_list: List[Dict[str, Any]],
        matched_request_indices: List[Optional[int]],
        occupant: Optional[str],
        used_request_indices: Set[int],
    ) -> Optional[Tuple[int, int]]:
        occupant_name = self._ident_name(occupant or "")
        if occupant_name is None:
            return None
        for request_index, pokemon_index in enumerate(matched_request_indices):
            if request_index in used_request_indices or pokemon_index is None:
                continue
            pokemon = pokemon_list[pokemon_index]
            pokemon_name = self._ident_name(str(pokemon.get("ident", "")))
            if pokemon_name == occupant_name:
                return request_index, pokemon_index
        return None

    def _reorder_side_pokemon_for_parse_request(
        self,
        pokemon_list: List[Dict[str, Any]],
        active_requests: Optional[Any],
        side_id: Optional[str],
    ) -> None:
        if side_id not in {"p1", "p2"}:
            return

        ordered: List[Dict[str, Any]] = []
        used_indices: set[int] = set()
        active_pokemon = [
            (index, pokemon)
            for index, pokemon in enumerate(pokemon_list)
            if pokemon.get("active")
        ]

        if isinstance(active_requests, list):
            for active_request in active_requests:
                move_ids = self._request_move_ids(active_request)
                if not move_ids:
                    continue
                for index, pokemon in active_pokemon:
                    if index in used_indices:
                        continue
                    if self._pokemon_matches_request_move_ids(pokemon, move_ids):
                        ordered.append(pokemon)
                        used_indices.add(index)
                        break

        slots = self._active_slots.get(side_id, [None, None])
        for occupant in slots:
            if occupant is None:
                continue
            occupant_name = self._ident_name(occupant)
            if occupant_name is None:
                continue
            for index, pokemon in active_pokemon:
                if index in used_indices:
                    continue
                pokemon_name = self._ident_name(str(pokemon.get("ident", "")))
                if pokemon_name == occupant_name:
                    ordered.append(pokemon)
                    used_indices.add(index)
                    break

        for index, pokemon in enumerate(pokemon_list):
            if index in used_indices:
                continue
            ordered.append(pokemon)

        pokemon_list[:] = ordered

    @staticmethod
    def _ident_name(ident: str) -> Optional[str]:
        if ": " not in ident:
            return None
        return ident.split(": ", 1)[1]

    @staticmethod
    def _request_move_ids(active_request: Any) -> Tuple[str, ...]:
        if not isinstance(active_request, dict):
            return ()
        moves = active_request.get("moves")
        if not isinstance(moves, list):
            return ()
        move_ids: List[str] = []
        for move in moves:
            if not isinstance(move, dict):
                continue
            move_id = move.get("id")
            if isinstance(move_id, str):
                move_ids.append(move_id)
        return tuple(move_ids)

    @staticmethod
    def _pokemon_matches_request_move_ids(pokemon: Dict[str, Any], move_ids: Tuple[str, ...]) -> bool:
        if not move_ids:
            return False
        pokemon_move_ids = tuple(str(move_id) for move_id in pokemon.get("moves", []))
        if pokemon_move_ids == move_ids:
            return True
        return all(move_id in pokemon_move_ids for move_id in move_ids)

    def _normalize_protocol_line(self, line: str, perspective_side: str) -> str:
        parts = line.split("|")
        if len(parts) <= 2:
            return line

        for index in range(2, len(parts)):
            parts[index] = self._expand_bare_slot_ident(parts[index], perspective_side)

        message_type = parts[1]
        if message_type in {"switch", "drag", "replace"}:
            parts[2] = self._normalize_ident(parts[2], consume_empty_slot=True)
            self._mark_slot_occupied(parts[2])
            return "|".join(parts)

        if message_type == "faint":
            parts[2] = self._normalize_ident(parts[2], consume_empty_slot=False)
            self._clear_slot(parts[2])
            return "|".join(parts)

        return "|".join(parts)

    def _normalize_ident(self, ident: str, consume_empty_slot: bool) -> str:
        if len(ident) >= 3 and ident[2] in {"a", "b"}:
            return ident

        match = _UNSLOTTED_IDENT_RE.match(ident)
        if match is None:
            return ident

        side = match.group(1)
        name = match.group(2)
        slot_index = self._infer_slot_index(
            side,
            name,
            consume_empty_slot=consume_empty_slot,
        )
        slot_suffix = "a" if slot_index == 0 else "b"
        return f"{side}{slot_suffix}: {name}"

    def _infer_slot_index(self, side: str, name: str, consume_empty_slot: bool) -> int:
        slots = self._active_slots.setdefault(side, [None, None])
        for index, occupant in enumerate(slots):
            if occupant is not None and self._ident_name(occupant) == name:
                return index
        for index, occupant in enumerate(slots):
            if occupant is None:
                return index
        return 0 if consume_empty_slot else 0

    def _mark_slot_occupied(self, ident: str) -> None:
        if len(ident) < 4 or ident[2] not in {"a", "b"}:
            return
        side = ident[:2]
        slot_index = 0 if ident[2] == "a" else 1
        self._active_slots.setdefault(side, [None, None])[slot_index] = ident

    def _clear_slot(self, ident: str) -> None:
        if len(ident) < 4 or ident[2] not in {"a", "b"}:
            return
        side = ident[:2]
        slot_index = 0 if ident[2] == "a" else 1
        self._active_slots.setdefault(side, [None, None])[slot_index] = None

    def _expand_bare_slot_ident(self, token: str, perspective_side: str) -> str:
        perspective_match = _PERSPECTIVE_SLOT_RE.match(token)
        if perspective_match is not None:
            relative_slot = perspective_match.group(1)
            absolute_slot = self._absolute_slot_from_perspective(
                relative_slot,
                perspective_side,
            )
            side = absolute_slot[:2]
            slot_index = 0 if absolute_slot[2] == "a" else 1
            occupant = self._active_slots.get(side, [None, None])[slot_index]
            if occupant is not None:
                return occupant
            suffix = perspective_match.group(2)
            return f"{absolute_slot}: {suffix}" if suffix else absolute_slot

        match = _BARE_SLOT_RE.match(token)
        if match is None:
            return token
        slot = match.group(1)
        side = slot[:2]
        slot_index = 0 if slot[2] == "a" else 1
        occupant = self._active_slots.get(side, [None, None])[slot_index]
        return occupant or token

    @staticmethod
    def _absolute_slot_from_perspective(relative_slot: str, perspective_side: str) -> str:
        if len(relative_slot) != 2:
            return relative_slot
        team_token = relative_slot[0]
        slot_token = relative_slot[1]
        if team_token == "1":
            absolute_side = perspective_side
        else:
            absolute_side = "p2" if perspective_side == "p1" else "p1"
        return f"{absolute_side}{slot_token}"
