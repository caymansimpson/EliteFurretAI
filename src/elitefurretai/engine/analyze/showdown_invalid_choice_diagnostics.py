# -*- coding: utf-8 -*-
"""Showdown invalid-choice diagnostics + masked-random fuzz harness.

Two modes share this file:

  --player model         (default) — model-driven self-play used to capture
                                     invalid-choice errors against a trained
                                     BatchInferencePlayer for ad-hoc analysis.

  --player random-masked            — fuzz-test rl/masking.py end-to-end with
                                     two masked-random players. Loops team
                                     resamplings until SIGINT or the first
                                     invalid-choice rejection (or empty mask),
                                     then writes a self-contained failure
                                     report to data/fuzz_results/ and exits.

The masked-random fuzz harness is the entry point of a debug-fix cycle that
hunts down masking bugs across all of gen9vgc2024regg. The full cycle —
fuzz → diagnose → synthetic regression test → fix → re-fuzz — is documented
in:

    planning/stage2/2026-05-05-22-42-fuzz-masking-harness-design.md

Future Claude sessions: when a failure report lands in data/fuzz_results/,
read both that report AND the design doc above before starting the per-bug
workflow (Section 4 of the design doc has the exact step-by-step).
"""

import argparse
import asyncio
import json
import random
import resource
import signal
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Dict, List, Optional, Sequence, Tuple, Union, cast

import numpy as np
from poke_env import AccountConfiguration, ServerConfiguration
from poke_env.battle import AbstractBattle, DoubleBattle
from poke_env.battle.observation import Observation
from poke_env.player import Player
from poke_env.player.battle_order import BattleOrder, DefaultBattleOrder
from poke_env.teambuilder.teambuilder import Teambuilder

from elitefurretai.engine.analyze.showdown_benchmark import _load_team_text
from elitefurretai.engine.showdown_server_manager import (
    launch_showdown_servers,
    shutdown_showdown_servers,
)
from elitefurretai.etl import Embedder
from elitefurretai.etl.encoder import MDBO
from elitefurretai.etl.team_repo import TeamRepo
from elitefurretai.rl.config import RNaDConfig
from elitefurretai.rl.masking import fast_get_action_mask
from elitefurretai.rl.players import SimpleModelPlayer


async def _capture_invalid_choice_errors_before_super(
    player: Player,
    split_messages: List[List[str]],
    on_invalid_choice: "callable[[AbstractBattle, str], None]",  # type: ignore[valid-type]
) -> None:
    """Walk a poke-env split_messages list for [Invalid choice] errors.

    Calls ``on_invalid_choice(battle, error_message_str)`` for each one,
    *before* poke-env's own error handler reacts (which clobbers the
    last-sent-message state used to attribute the rejection).

    Skips the init-battle message because no battle object exists yet.
    Subclasses use this from their own ``_handle_battle_message`` override.
    """
    if not split_messages or not split_messages[0]:
        return
    is_init = (
        len(split_messages) > 1
        and len(split_messages[1]) > 1
        and split_messages[1][1] == "init"
    )
    if is_init:
        return
    battle = await player._get_battle(split_messages[0][0])
    for split_msg in split_messages[1:]:
        if (
            len(split_msg) >= 3
            and split_msg[1] == "error"
            and split_msg[2].startswith("[Invalid choice]")
        ):
            on_invalid_choice(battle, split_msg[2])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture Showdown websocket invalid choices with source battle/request traces."
    )
    parser.add_argument("--format", default="gen9vgc2024regg")
    parser.add_argument(
        "--player",
        choices=["model", "random-masked"],
        default="model",
        help=(
            "model: existing diagnostic mode using a trained BatchInferencePlayer. "
            "random-masked: fuzz-test masking.py with two players that sample uniformly "
            "from legal-mask actions, looping team-resamplings until SIGINT or first failure."
        ),
    )
    parser.add_argument(
        "--config",
        required=False,
        help="Required for --player model; ignored for --player random-masked.",
    )
    parser.add_argument("--checkpoint")
    parser.add_argument("--opponent-checkpoint")
    parser.add_argument("--battles", type=int, default=60)
    parser.add_argument("--max-concurrent-battles", type=int, default=1)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-level", type=int, default=50)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--batch-timeout", type=float, default=0.01)
    parser.add_argument("--feature-set")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--max-battle-steps", type=int, default=40)
    parser.add_argument("--team-path")
    parser.add_argument("--opponent-team-path")
    parser.add_argument("--random-teams", action="store_true")
    parser.add_argument("--random-opponent-teams", action="store_true")
    parser.add_argument("--team-subdirectory")
    parser.add_argument("--opponent-team-subdirectory")
    parser.add_argument("--no-mirror", action="store_true")
    parser.add_argument(
        "--output-dir",
        required=False,
        help="Required for --player model. For --player random-masked, defaults to data/fuzz_results/.",
    )
    parser.add_argument("--render-limit", type=int, default=10)
    parser.add_argument("--max-error-turn", type=int, default=3)
    parser.add_argument(
        "--fuzz-concurrent-battles",
        type=int,
        default=1,
        help="(--player random-masked only) Up to N battles in flight per resampling.",
    )
    parser.add_argument(
        "--fuzz-battles-per-pair",
        type=int,
        default=100,
        help="(--player random-masked only) Battles between a sampled team pair before resampling.",
    )
    return parser


class RandomTeamRepoTeambuilder(Teambuilder):
    def __init__(
        self,
        *,
        repo: TeamRepo,
        format_id: str,
        subdirectory: Optional[str] = None,
    ) -> None:
        self._repo = repo
        self._format_id = format_id
        self._subdirectory = subdirectory

    def yield_team(self) -> str:
        return self.join_team(
            self.parse_showdown_team(
                self._repo.sample_team(self._format_id, subdirectory=self._subdirectory)
            )
        )


def _build_team_source(
    *,
    repo: TeamRepo,
    format_id: str,
    team_path: Optional[str],
    random_teams: bool,
    subdirectory: Optional[str],
) -> Union[str, Teambuilder]:
    if random_teams:
        return RandomTeamRepoTeambuilder(
            repo=repo,
            format_id=format_id,
            subdirectory=subdirectory,
        )
    return _load_team_text(format_id, team_path, repo)


def _request_type_from_request(request: Dict[str, Any]) -> str:
    if not request:
        return "unknown"
    if request.get("teamPreview"):
        return "teampreview"
    force_switch = request.get("forceSwitch")
    if isinstance(force_switch, list) and any(bool(value) for value in force_switch):
        return "force_switch"
    if request.get("wait"):
        return "wait"
    return "turn"


def _stringify_event(split_event: Sequence[str]) -> str:
    return "|".join(str(part) for part in split_event)


def _observation_to_lines(observation: Optional[Observation]) -> List[str]:
    if observation is None:
        return ["<no observation>"]

    def _render_team(team: Dict[str, Any]) -> str:
        if not team:
            return "<empty>"
        rendered = []
        for ident, mon in team.items():
            if mon is None:
                rendered.append(f"{ident}: <unknown>")
                continue
            rendered.append(
                f"{ident}: {getattr(mon, 'species', '<unknown>')} [active={getattr(mon, 'active', None)}, fainted={getattr(mon, 'fainted', None)}]"
            )
        return "; ".join(rendered)

    lines = [
        f"Observed My Team: {_render_team(getattr(observation, 'team', {}))}",
        f"Observed Opp Team: {_render_team(getattr(observation, 'opponent_team', {}))}",
        "Observed Events:",
    ]
    events = getattr(observation, "events", [])
    if events:
        lines.extend(f"  - {_stringify_event(event)}" for event in events)
    else:
        lines.append("  - <no events yet>")
    return lines


def _battle_state_to_string(battle: AbstractBattle) -> str:
    def _mon_label(mon: Any) -> str:
        if mon is None:
            return "None"
        species = getattr(mon, "species", getattr(mon, "name", "<unknown>"))
        current_hp_fraction = getattr(mon, "current_hp_fraction", None)
        status = getattr(mon, "status", None)
        return (
            f"{species} [active={getattr(mon, 'active', None)}, fainted={getattr(mon, 'fainted', None)}, "
            f"hp={current_hp_fraction}, status={status}]"
        )

    def _team_lines(header: str, team: Dict[str, Any]) -> List[str]:
        lines = [header]
        if not team:
            lines.append("  - <empty>")
            return lines
        for ident, mon in team.items():
            lines.append(f"  - {ident}: {_mon_label(mon)}")
        return lines

    def _preview_lines(header: str, team: Sequence[Any]) -> List[str]:
        lines = [header]
        if not team:
            lines.append("  - <empty>")
            return lines
        for index, mon in enumerate(team, start=1):
            lines.append(f"  - slot {index}: {_mon_label(mon)}")
        return lines

    lines = [
        f"Battle {battle.battle_tag} turn={battle.turn} player_role={battle.player_role}",
        f"Perspective: {battle.player_username} vs {battle.opponent_username}",
        f"My Active: [{', '.join(_mon_label(mon) for mon in getattr(battle, 'active_pokemon', []) or [])}]",
        f"Opp Active: [{', '.join(_mon_label(mon) for mon in getattr(battle, 'opponent_active_pokemon', []) or [])}]",
        f"Force Switch: {getattr(battle, 'force_switch', None)}",
        f"Can Tera: {getattr(battle, 'can_tera', None)}",
    ]
    if bool(getattr(battle, "teampreview", False)):
        lines.extend(
            _preview_lines(
                "My Teampreview Team:", getattr(battle, "teampreview_team", []) or []
            )
        )
        lines.extend(
            _preview_lines(
                "Opp Teampreview Team:",
                getattr(battle, "teampreview_opponent_team", []) or [],
            )
        )
    else:
        lines.extend(_team_lines("My Team:", getattr(battle, "team", {}) or {}))
        lines.extend(_team_lines("Opp Team:", getattr(battle, "opponent_team", {}) or {}))

    current_observation = getattr(battle, "current_observation", None)
    if current_observation is not None:
        lines.append("Current Observation:")
        lines.extend(_observation_to_lines(current_observation))

    return "\n".join(lines)


def _request_state_to_string(request: Dict[str, Any]) -> str:
    if not request:
        return "<no request>"

    lines = [
        f"Request type: {_request_type_from_request(request)}",
        f"rqid={request.get('rqid')} wait={request.get('wait', False)} teamPreview={request.get('teamPreview', False)} forceSwitch={request.get('forceSwitch')}",
    ]

    active_entries = request.get("active")
    if isinstance(active_entries, list):
        lines.append("Active Request Payloads:")
        for slot, active in enumerate(active_entries):
            if not isinstance(active, dict):
                lines.append(f"  - slot {slot}: <invalid payload>")
                continue
            moves = active.get("moves") or []
            move_parts = []
            for move in moves:
                if not isinstance(move, dict):
                    continue
                move_parts.append(
                    f"{move.get('id')}[disabled={move.get('disabled', False)}, target={move.get('target')}, pp={move.get('pp')}]"
                )
            lines.append(
                f"  - slot {slot}: canTera={active.get('canTerastallize')} trapped={active.get('trapped')} maybeTrapped={active.get('maybeTrapped')} moves={move_parts}"
            )

    side = request.get("side")
    if isinstance(side, dict):
        lines.append("Side Pokemon:")
        for slot, mon in enumerate(side.get("pokemon") or []):
            if not isinstance(mon, dict):
                continue
            lines.append(
                "  - slot {slot}: {ident} active={active} condition={condition} item={item} teraUsed={terastallized}".format(
                    slot=slot,
                    ident=mon.get("ident"),
                    active=mon.get("active"),
                    condition=mon.get("condition"),
                    item=mon.get("item"),
                    terastallized=mon.get("terastallized"),
                )
            )

    return "\n".join(lines)


def _observations_to_string(battle: AbstractBattle, turn_limit: int) -> str:
    lines: List[str] = []
    for turn in sorted(getattr(battle, "observations", {}).keys()):
        if turn > turn_limit:
            break
        lines.append(f"Turn #{turn}:")
        lines.extend(_observation_to_lines(battle.observations[turn]))

    current_observation = getattr(battle, "current_observation", None)
    if current_observation is not None:
        current_turn = getattr(battle, "turn", turn_limit)
        if current_turn <= turn_limit:
            last_observation = getattr(battle, "observations", {}).get(current_turn)
            last_events = getattr(last_observation, "events", None)
            current_events = getattr(current_observation, "events", None)
            if current_events != last_events:
                lines.append(f"Current Observation; Turn #{current_turn}:")
                lines.extend(_observation_to_lines(current_observation))

    return "\n".join(lines) if lines else "<no observations>"


def _render_record(record: Dict[str, Any]) -> str:
    parts = [
        f"## {record['battle_tag']}",
        f"Turn: {record['turn']}",
        f"Player Role: {record['player_role']}",
        f"Error: {record['error_message']}",
        f"Attempted Choice: {record.get('attempted_message')}",
        "",
        "### Battle State",
        record["battle_state"],
        "",
        "### Request",
        record["request_state"],
        "",
        "### Observations",
        record["observations"],
        "",
    ]
    return "\n".join(parts)


class DiagnosticSimpleModelPlayer(SimpleModelPlayer):
    """SimpleModelPlayer that captures invalid-choice errors with full context.

    Records the last sent message per battle (via _handle_battle_request
    override) and snapshots battle state on every [Invalid choice] error
    (via _handle_battle_message override) so the message that was rejected
    can be attributed to the request that produced it.
    """

    def __init__(self, *, diagnostic_records: List[Dict[str, Any]], **kwargs: Any):
        self._diagnostic_records = diagnostic_records
        self._last_messages: Dict[str, Optional[str]] = {}
        super().__init__(**kwargs)

    async def _handle_battle_request(
        self, battle: AbstractBattle, maybe_default_order: bool = False
    ) -> None:
        if getattr(battle, "finished", False):
            return

        if maybe_default_order and random.random() < self.DEFAULT_CHOICE_CHANCE:
            message = self.choose_default_move().message
        elif battle.teampreview:
            tp_result = self.teampreview(battle)
            if isinstance(tp_result, Awaitable):
                tp_result = await tp_result
            message = cast(str, tp_result)
        else:
            choice = self.choose_move(battle)
            if isinstance(choice, Awaitable):
                choice = await choice
            message = choice.message if hasattr(choice, "message") else str(choice)

        # Patch around two showdown quirks where masking can produce a
        # technically-legal-but-rejected message:
        # - force_switch + a move choice
        # - "terastallize" when no slot has can_tera
        if (
            isinstance(battle, DoubleBattle)
            and any(battle.force_switch)
            and isinstance(message, str)
            and message.startswith("/choose move")
        ):
            message = self.choose_default_move().message

        if (
            isinstance(battle, DoubleBattle)
            and isinstance(message, str)
            and "terastallize" in message
            and hasattr(battle, "can_tera")
            and not any(getattr(battle, "can_tera", []))
        ):
            message = message.replace(" terastallize", "")

        self._last_messages[battle.battle_tag] = message

        if message:
            try:
                await self.ps_client.send_message(message, battle.battle_tag)
            except Exception:
                self.hidden_states.pop(battle.battle_tag, None)

    def _record_invalid_choice(self, battle: AbstractBattle, error_message: str) -> None:
        record = {
            "battle_tag": battle.battle_tag,
            "turn": getattr(battle, "turn", -1),
            "player_role": getattr(battle, "player_role", None),
            "error_message": error_message,
            "attempted_message": self._last_messages.get(battle.battle_tag),
            "request_type": _request_type_from_request(
                getattr(battle, "last_request", {}) or {}
            ),
            "request": deepcopy(getattr(battle, "last_request", {}) or {}),
            "battle_state": _battle_state_to_string(battle),
            "request_state": _request_state_to_string(
                getattr(battle, "last_request", {}) or {}
            ),
            "observations": _observations_to_string(battle, getattr(battle, "turn", 0)),
        }
        self._diagnostic_records.append(record)

    async def _handle_battle_message(self, split_messages: List[List[str]]) -> None:
        await _capture_invalid_choice_errors_before_super(
            self, split_messages, self._record_invalid_choice
        )
        await super()._handle_battle_message(split_messages)


class MaskedRandomPlayer(Player):
    """Plays uniformly at random from indices the action mask permits.

    Pure exercise of `masking.py` end-to-end: for each turn (regular or
    force-switch) it snapshots `battle.last_request`, runs `fast_get_action_mask`,
    samples one index where mask==1, and decodes via `MDBO.from_int`. Teampreview
    uses a random `/team {permutation}` since teampreview isn't masked.

    On a Showdown invalid-choice rejection it appends a record to a shared list
    that the outer fuzz loop drains to write the failure-report artifacts.
    Empty masks are themselves recorded as failures (a mask that says "no legal
    moves exist" is by definition a bug here).
    """

    def __init__(
        self,
        *,
        fuzz_records: List[Dict[str, Any]],
        rng: Optional[random.Random] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._fuzz_records = fuzz_records
        self._rng = rng if rng is not None else random.Random()
        # Per-battle scratch state for the most recent decision, consumed by
        # _handle_battle_error to attribute the rejected command back to the
        # mask + sampled index that produced it.
        self._last_messages: Dict[str, Optional[str]] = {}
        self._last_masks: Dict[str, np.ndarray] = {}
        self._last_actions: Dict[str, int] = {}
        self._last_request_snapshots: Dict[str, Optional[Dict[str, Any]]] = {}

    def teampreview(self, battle: AbstractBattle) -> str:  # type: ignore[override]
        return self.random_teampreview(battle)

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:  # type: ignore[override]
        if not isinstance(battle, DoubleBattle):
            return DefaultBattleOrder()

        request_snapshot = (
            deepcopy(battle.last_request) if battle.last_request is not None else None
        )
        mask = fast_get_action_mask(battle, request_override=request_snapshot)
        legal = np.where(mask > 0.5)[0]

        if len(legal) == 0:
            self._record_failure(
                battle=battle,
                request_snapshot=request_snapshot,
                mask=mask,
                sampled_index=-1,
                attempted_message=None,
                error_message="[FuzzHarness] EMPTY_MASK: fast_get_action_mask returned no legal actions",
            )
            return DefaultBattleOrder()

        action_idx = int(self._rng.choice(legal))
        action_type = MDBO.FORCE_SWITCH if any(battle.force_switch) else MDBO.TURN
        try:
            mdbo = MDBO.from_int(action_idx, type=action_type)
            order = mdbo.to_double_battle_order(battle, request=request_snapshot)
            message = order.message
        except Exception:
            order = DefaultBattleOrder()
            message = order.message

        self._last_request_snapshots[battle.battle_tag] = request_snapshot
        self._last_masks[battle.battle_tag] = mask
        self._last_actions[battle.battle_tag] = action_idx
        self._last_messages[battle.battle_tag] = message
        return order

    def _record_failure(
        self,
        *,
        battle: AbstractBattle,
        request_snapshot: Optional[Dict[str, Any]],
        mask: Optional[np.ndarray],
        sampled_index: int,
        attempted_message: Optional[str],
        error_message: str,
    ) -> None:
        record: Dict[str, Any] = {
            "battle_tag": battle.battle_tag,
            "turn": getattr(battle, "turn", -1),
            "player_role": getattr(battle, "player_role", None),
            "player_username": self.username,
            "error_message": error_message,
            "attempted_message": attempted_message,
            "sampled_index": sampled_index,
            "mask": mask.tolist() if mask is not None else None,
            "mask_legal_count": int(mask.sum()) if mask is not None else None,
            "request_type": _request_type_from_request(request_snapshot or {}),
            "request": deepcopy(request_snapshot or {}),
            "battle_state": _battle_state_to_string(battle),
            "request_state": _request_state_to_string(request_snapshot or {}),
            "observations": _observations_to_string(battle, getattr(battle, "turn", 0)),
        }
        self._fuzz_records.append(record)

    def _record_invalid_choice(self, battle: AbstractBattle, error_message: str) -> None:
        self._record_failure(
            battle=battle,
            request_snapshot=self._last_request_snapshots.get(battle.battle_tag),
            mask=self._last_masks.get(battle.battle_tag),
            sampled_index=self._last_actions.get(battle.battle_tag, -1),
            attempted_message=self._last_messages.get(battle.battle_tag),
            error_message=error_message,
        )

    async def _handle_battle_message(self, split_messages: List[List[str]]) -> None:
        await _capture_invalid_choice_errors_before_super(
            self, split_messages, self._record_invalid_choice
        )
        await super()._handle_battle_message(split_messages)


def _render_fuzz_failure_report(record: Dict[str, Any]) -> str:
    """Human-readable failure report for an offline diagnosis pass."""
    mask_legal_count = record.get("mask_legal_count")
    sampled = record.get("sampled_index")
    mask = record.get("mask")
    mask_at_sampled = (
        mask[sampled]
        if mask is not None and sampled is not None and 0 <= sampled < len(mask)
        else "n/a"
    )

    lines = [
        "================ FUZZ FAILURE REPORT ================",
        f"format:                {record.get('format', '<unknown>')}",
        f"timestamp:             {record.get('timestamp', '<unknown>')}",
        f"battle_tag:            {record.get('battle_tag')}",
        f"failing player:        {record.get('player_role')} (username={record.get('player_username')})",
        f"resampling_count:      {record.get('resampling_count')}",
        f"battle_in_round:       {record.get('battle_in_round')}/{record.get('battles_per_pair')}",
        f"total_battles_so_far:  {record.get('total_battles')}",
        "",
        "team_a (p1 source team):",
        record.get("team_a", "<unknown>"),
        "",
        "team_b (p2 source team):",
        record.get("team_b", "<unknown>"),
        "",
        f"attempted command:     {record.get('attempted_message')}",
        f"sampled action index:  {sampled}",
        f"mask[sampled]:         {mask_at_sampled}    (1.0 means: mask said legal but Showdown rejected → likely over-permissive)",
        f"mask legal count:      {mask_legal_count}    (0 means EMPTY_MASK bug)",
        f"showdown error msg:    {record.get('error_message')}",
        f"request type:          {record.get('request_type')}",
        "",
        "────────── LAST REQUEST PAYLOAD (seen by masking) ──────────",
        json.dumps(record.get("request", {}), indent=2, default=str),
        "",
        "────────── REQUEST STATE SUMMARY ──────────",
        record.get("request_state", "<no request_state>"),
        "",
        "────────── BATTLE STATE AT FAILURE ──────────",
        record.get("battle_state", "<no battle_state>"),
        "",
        "────────── OBSERVATIONS (full event log up to failure) ──────────",
        record.get("observations", "<no observations>"),
    ]
    return "\n".join(lines)


def _jsonify_fuzz_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Strip non-JSON-serializable bits and produce a clean sidecar dict."""
    out = dict(record)
    # mask is already a list (from .tolist()); request is plain JSON
    return out


async def _run(args: argparse.Namespace) -> None:
    repo = TeamRepo(shuffle=False)
    rng = random.Random(args.seed)
    config = RNaDConfig.load(args.config)
    feature_set = args.feature_set or config.training.embedder_feature_set

    if args.checkpoint is None:
        raise SystemExit("--checkpoint is required for --player model")

    team_subdirectory = args.team_subdirectory
    if team_subdirectory is None and args.random_teams:
        team_subdirectory = config.curriculum.opponent_team_pool_path

    opponent_team_subdirectory = args.opponent_team_subdirectory
    if opponent_team_subdirectory is None and args.random_opponent_teams:
        opponent_team_subdirectory = config.curriculum.opponent_team_pool_path

    p1_team = _build_team_source(
        repo=repo,
        format_id=args.format,
        team_path=args.team_path,
        random_teams=args.random_teams,
        subdirectory=team_subdirectory,
    )
    if args.opponent_team_path is not None:
        p2_team: Union[str, Teambuilder] = Path(args.opponent_team_path).read_text()
    elif args.random_opponent_teams:
        p2_team = _build_team_source(
            repo=repo,
            format_id=args.format,
            team_path=None,
            random_teams=True,
            subdirectory=opponent_team_subdirectory,
        )
    elif args.no_mirror:
        p2_team = repo.sample_team(
            args.format,
            subdirectory=config.curriculum.opponent_team_pool_path,
        )
    else:
        p2_team = p1_team

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "invalid_choices.jsonl"
    markdown_path = output_dir / "first_invalid_p1_turn_le_3.md"
    summary_path = output_dir / "summary.json"

    diagnostic_records: List[Dict[str, Any]] = []

    server_launch_start = time.perf_counter()
    server_processes = launch_showdown_servers(1, args.port)
    server_launch_seconds = time.perf_counter() - server_launch_start
    battle_loop_seconds = 0.0
    player_setup_seconds = 0.0
    teardown_seconds = 0.0
    player1: Optional[Player] = None
    player2: Optional[Player] = None

    try:
        setup_start = time.perf_counter()
        suffix = rng.randint(100000, 999999)
        server_config = ServerConfiguration(
            f"ws://localhost:{args.port}/showdown/websocket", ""
        )

        embedder = Embedder(
            format=config.curriculum.battle_format,
            feature_set=feature_set,
            omniscient=False,
        )
        opponent_checkpoint = args.opponent_checkpoint or args.checkpoint

        player1 = DiagnosticSimpleModelPlayer(
            model_path=args.checkpoint,
            device=args.device,
            battle_format=config.curriculum.battle_format,
            probabilistic=not args.greedy,
            embedder=embedder,
            team=p1_team,
            max_concurrent_battles=args.max_concurrent_battles,
            server_configuration=server_config,
            account_configuration=AccountConfiguration(f"showdiagp1{suffix}", None),
            log_level=args.log_level,
            diagnostic_records=diagnostic_records,
        )
        player2 = SimpleModelPlayer(
            model_path=opponent_checkpoint,
            device=args.device,
            battle_format=config.curriculum.battle_format,
            probabilistic=not args.greedy,
            embedder=embedder,
            team=p2_team,
            max_concurrent_battles=args.max_concurrent_battles,
            server_configuration=server_config,
            account_configuration=AccountConfiguration(f"showdiagp2{suffix}", None),
            log_level=args.log_level,
        )

        await player1.ps_client.wait_for_login()
        await player2.ps_client.wait_for_login()
        player_setup_seconds = time.perf_counter() - setup_start

        battle_loop_start = time.perf_counter()
        await player1.battle_against(player2, n_battles=args.battles)
        battle_loop_seconds = time.perf_counter() - battle_loop_start
    finally:
        teardown_start = time.perf_counter()
        shutdown_showdown_servers(server_processes)
        teardown_seconds = time.perf_counter() - teardown_start

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in diagnostic_records:
            handle.write(json.dumps(record) + "\n")

    first_p1_early_records: List[Dict[str, Any]] = []
    seen_tags = set()
    for record in diagnostic_records:
        if record.get("player_role") != "p1":
            continue
        if int(record.get("turn", 999)) > args.max_error_turn:
            continue
        battle_tag = record.get("battle_tag")
        if battle_tag in seen_tags:
            continue
        seen_tags.add(battle_tag)
        first_p1_early_records.append(record)
        if len(first_p1_early_records) >= args.render_limit:
            break

    markdown_path.write_text(
        "\n\n".join(_render_record(record) for record in first_p1_early_records),
        encoding="utf-8",
    )

    error_family_counts: Dict[str, int] = {}
    request_type_counts: Dict[str, int] = {}
    for record in diagnostic_records:
        message = str(record.get("error_message", ""))
        family = message.split(":", 1)[0]
        error_family_counts[family] = error_family_counts.get(family, 0) + 1
        request_type = str(record.get("request_type", "unknown"))
        request_type_counts[request_type] = request_type_counts.get(request_type, 0) + 1

    usage = resource.getrusage(resource.RUSAGE_SELF)
    summary = {
        "completed_battles": getattr(player1, "n_finished_battles", 0),
        "p1_wins": getattr(player1, "n_won_battles", 0),
        "duration_seconds": server_launch_seconds
        + player_setup_seconds
        + battle_loop_seconds,
        "server_launch_seconds": server_launch_seconds,
        "player_setup_seconds": player_setup_seconds,
        "battle_loop_seconds": battle_loop_seconds,
        "teardown_seconds": teardown_seconds,
        "cpu_user_seconds": usage.ru_utime,
        "cpu_system_seconds": usage.ru_stime,
        "invalid_choice_count": len(diagnostic_records),
        "p1_invalid_choice_count": sum(
            1 for record in diagnostic_records if record.get("player_role") == "p1"
        ),
        "p1_turn_le_3_count": len(first_p1_early_records),
        "error_family_counts": error_family_counts,
        "request_type_counts": request_type_counts,
        "jsonl_path": str(jsonl_path),
        "markdown_path": str(markdown_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"completed_battles={summary['completed_battles']}")
    print(f"p1_wins={summary['p1_wins']}")
    print(f"invalid_choice_count={summary['invalid_choice_count']}")
    print(f"p1_invalid_choice_count={summary['p1_invalid_choice_count']}")
    print(f"p1_turn_le_{args.max_error_turn}_count={summary['p1_turn_le_3_count']}")
    print(f"duration_seconds={summary['duration_seconds']:.3f}")
    print(f"jsonl_path={jsonl_path}")
    print(f"markdown_path={markdown_path}")
    print(f"summary_path={summary_path}")


def _sample_two_distinct_teams(
    repo: TeamRepo,
    format_id: str,
    rng: random.Random,
) -> Tuple[Tuple[str, str], Tuple[str, str]]:
    """Returns ((team_a_name, team_a_str), (team_b_name, team_b_str))."""
    teams_dict = repo.teams.get(format_id, {})
    if len(teams_dict) < 2:
        raise RuntimeError(
            f"Need ≥2 teams for format {format_id}, found {len(teams_dict)}"
        )
    items = list(teams_dict.items())
    pair = rng.sample(items, 2)
    return pair[0], pair[1]


async def _run_fuzz_loop(args: argparse.Namespace) -> None:
    """Drives the masked-random fuzz harness.

    Loop: sample 2 distinct teams from the format, build two `MaskedRandomPlayer`s,
    play `--fuzz-battles-per-pair` battles between them. On the first invalid-choice
    rejection (from either player, including EMPTY_MASK detection in the player
    itself), write the failure-report artifacts and exit. Otherwise log a clean
    progress line and resample. Exits cleanly on SIGINT.
    """
    output_dir = Path(args.output_dir or "data/fuzz_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    repo = TeamRepo(shuffle=False)
    rng = random.Random(args.seed)

    sigint_state = {"received": False}

    def _sigint_handler(signum: int, frame: Any) -> None:  # noqa: ARG001 - signal API
        sigint_state["received"] = True
        print("\n[fuzz] SIGINT received; will exit after current battle finishes.")

    previous_handler = signal.signal(signal.SIGINT, _sigint_handler)

    server_processes = launch_showdown_servers(1, args.port)
    server_config = ServerConfiguration(
        f"ws://localhost:{args.port}/showdown/websocket", ""
    )

    fuzz_records: List[Dict[str, Any]] = []
    resampling_count = 0
    total_battles = 0
    failure_record: Optional[Dict[str, Any]] = None
    start_time = time.perf_counter()

    print(
        f"[fuzz] starting masked-random fuzz on {args.format} "
        f"(battles_per_pair={args.fuzz_battles_per_pair}, "
        f"concurrent_battles={args.fuzz_concurrent_battles}, "
        f"max_battle_steps={args.max_battle_steps})"
    )
    print(f"[fuzz] output_dir={output_dir}")
    print("[fuzz] Ctrl-C to stop")

    try:
        while not sigint_state["received"] and failure_record is None:
            resampling_count += 1
            (team_a_name, team_a_str), (team_b_name, team_b_str) = (
                _sample_two_distinct_teams(repo, args.format, rng)
            )

            suffix = rng.randint(100000, 999999)
            player1: Optional[MaskedRandomPlayer] = None
            player2: Optional[MaskedRandomPlayer] = None
            try:
                player1 = MaskedRandomPlayer(
                    fuzz_records=fuzz_records,
                    rng=random.Random(rng.randint(0, 2**31 - 1)),
                    battle_format=args.format,
                    team=team_a_str,
                    max_concurrent_battles=args.fuzz_concurrent_battles,
                    server_configuration=server_config,
                    account_configuration=AccountConfiguration(f"fuzzp1{suffix}", None),
                    log_level=args.log_level,
                )
                player2 = MaskedRandomPlayer(
                    fuzz_records=fuzz_records,
                    rng=random.Random(rng.randint(0, 2**31 - 1)),
                    battle_format=args.format,
                    team=team_b_str,
                    max_concurrent_battles=args.fuzz_concurrent_battles,
                    server_configuration=server_config,
                    account_configuration=AccountConfiguration(f"fuzzp2{suffix}", None),
                    log_level=args.log_level,
                )

                # battle_against handles login internally via to_wait on logged_in.
                await player1.battle_against(player2, n_battles=args.fuzz_battles_per_pair)
                total_battles += args.fuzz_battles_per_pair
            finally:
                # Disconnect this round's clients before next resampling.
                if player1 is not None:
                    try:
                        await player1.ps_client.stop_listening()
                    except Exception:
                        pass
                if player2 is not None:
                    try:
                        await player2.ps_client.stop_listening()
                    except Exception:
                        pass

            if fuzz_records:
                # Write the FIRST captured failure (deterministic by record order).
                # Drop any subsequent failures from the same batch — we'll re-fuzz
                # after the fix to surface them again if they're independent bugs.
                first = fuzz_records[0]
                failure_record = first
                first["format"] = args.format
                first["timestamp"] = datetime.now().isoformat(timespec="seconds")
                first["resampling_count"] = resampling_count
                first["battles_per_pair"] = args.fuzz_battles_per_pair
                # battle_in_round is approximate — we only know the round-total
                # ran; finer attribution would require per-battle hooks.
                first["battle_in_round"] = "?/?"
                first["total_battles"] = total_battles
                first["team_a"] = team_a_str
                first["team_b"] = team_b_str
                first["team_a_name"] = team_a_name
                first["team_b_name"] = team_b_name
                first["additional_failures_in_same_batch"] = max(0, len(fuzz_records) - 1)
                break

            print(
                f"[resample {resampling_count}] still clean after {total_battles} battles "
                f"(pair: {team_a_name} vs {team_b_name})"
            )
    finally:
        shutdown_showdown_servers(server_processes)
        signal.signal(signal.SIGINT, previous_handler)

    elapsed = time.perf_counter() - start_time

    if failure_record is not None:
        battle_tag = failure_record["battle_tag"].replace("/", "_").replace(" ", "_")
        ts = failure_record["timestamp"].replace(":", "-")
        stem = f"{ts}-{battle_tag}"
        txt_path = output_dir / f"{stem}.txt"
        json_path = output_dir / f"{stem}.artifacts.json"
        txt_path.write_text(_render_fuzz_failure_report(failure_record), encoding="utf-8")
        json_path.write_text(
            json.dumps(_jsonify_fuzz_record(failure_record), indent=2, default=str),
            encoding="utf-8",
        )

        print()
        print("================ FUZZ FAILURE CAPTURED ================")
        print(f"  resampling:     #{failure_record['resampling_count']}")
        print(f"  total battles:  {total_battles}")
        print(f"  elapsed:        {elapsed:.1f}s")
        print(f"  error:          {failure_record['error_message']}")
        print(f"  failing player: {failure_record['player_role']}")
        print(f"  attempted:      {failure_record['attempted_message']}")
        print(f"  report:         {txt_path}")
        print(f"  artifacts:      {json_path}")
        if failure_record.get("additional_failures_in_same_batch", 0) > 0:
            print(
                f"  (note: {failure_record['additional_failures_in_same_batch']} "
                f"additional failures in same batch were dropped; re-fuzz after fix)"
            )
    else:
        print()
        print(
            f"[fuzz] interrupted: {resampling_count} resamplings, "
            f"{total_battles} clean battles in {elapsed:.1f}s"
        )


def main() -> None:
    args = build_parser().parse_args()
    random.seed(args.seed)
    if args.player == "random-masked":
        asyncio.run(_run_fuzz_loop(args))
    else:
        if args.config is None:
            raise SystemExit("--config is required for --player model")
        if args.output_dir is None:
            raise SystemExit("--output-dir is required for --player model")
        asyncio.run(_run(args))


if __name__ == "__main__":
    main()
