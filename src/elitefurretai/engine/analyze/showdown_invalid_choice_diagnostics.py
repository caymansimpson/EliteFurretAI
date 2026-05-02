import argparse
import asyncio
import json
import random
import resource
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union, cast

from poke_env import AccountConfiguration, ServerConfiguration
from poke_env.battle import AbstractBattle, DoubleBattle
from poke_env.battle.observation import Observation
from poke_env.player import Player
from poke_env.teambuilder.teambuilder import Teambuilder

from elitefurretai.engine.analyze.showdown_benchmark import _build_agent, _load_team_text
from elitefurretai.engine.showdown_server_manager import (
    launch_showdown_servers,
    shutdown_showdown_servers,
)
from elitefurretai.etl import Embedder
from elitefurretai.etl.team_repo import TeamRepo
from elitefurretai.rl.config import RNaDConfig
from elitefurretai.rl.players import BatchInferencePlayer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture Showdown websocket invalid choices with source battle/request traces."
    )
    parser.add_argument("--format", default="gen9vgc2024regg")
    parser.add_argument("--config", required=True)
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
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--render-limit", type=int, default=10)
    parser.add_argument("--max-error-turn", type=int, default=3)
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


class DiagnosticBatchInferencePlayer(BatchInferencePlayer):
    def __init__(self, *, diagnostic_records: List[Dict[str, Any]], **kwargs: Any):
        self._diagnostic_records = diagnostic_records
        self._last_messages: Dict[str, Optional[str]] = {}
        super().__init__(**kwargs)

    async def _handle_battle_request(
        self, battle: AbstractBattle, maybe_default_order: bool = False
    ):
        if getattr(battle, "finished", False):
            return

        request_generation = self._request_generation.get(battle.battle_tag, 0) + 1
        self._request_generation[battle.battle_tag] = request_generation

        if maybe_default_order and random.random() < self.DEFAULT_CHOICE_CHANCE:
            message = self.choose_default_move().message
            self._last_messages[battle.battle_tag] = message
            try:
                await self.ps_client.send_message(message, battle.battle_tag)
            except Exception:
                self.current_trajectories.pop(battle.battle_tag, None)
                self.hidden_states.pop(battle.battle_tag, None)
            return

        choice = await self._choose_move_async(
            battle,
            request_generation=request_generation,
        )

        if self._request_generation.get(battle.battle_tag, -1) != request_generation:
            return

        if getattr(battle, "finished", False):
            return

        if isinstance(choice, str):
            message = choice
        elif hasattr(choice, "message"):
            message = choice.message
        else:
            message = str(choice)

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
                self.current_trajectories.pop(battle.battle_tag, None)
                self.hidden_states.pop(battle.battle_tag, None)

    async def _handle_battle_error(
        self, battle: AbstractBattle, split_message: List[str]
    ) -> None:
        message = split_message[2] if len(split_message) > 2 else ""
        if message.startswith("[Invalid choice]"):
            record = {
                "battle_tag": battle.battle_tag,
                "turn": getattr(battle, "turn", -1),
                "player_role": getattr(battle, "player_role", None),
                "error_message": message,
                "attempted_message": self._last_messages.get(battle.battle_tag),
                "request_type": _request_type_from_request(
                    getattr(battle, "last_request", {}) or {}
                ),
                "request": deepcopy(getattr(battle, "last_request", {}) or {}),
                "battle_state": _battle_state_to_string(battle),
                "request_state": _request_state_to_string(
                    getattr(battle, "last_request", {}) or {}
                ),
                "observations": _observations_to_string(
                    battle, getattr(battle, "turn", 0)
                ),
            }
            self._diagnostic_records.append(record)

        await super()._handle_battle_error(battle, split_message)


async def _run(args: argparse.Namespace) -> None:
    repo = TeamRepo(shuffle=False)
    rng = random.Random(args.seed)
    config = RNaDConfig.load(args.config)
    feature_set = args.feature_set or config.training.embedder_feature_set
    temperature = (
        args.temperature if args.temperature is not None else config.temperature_at_step(0)
    )
    top_p = args.top_p if args.top_p is not None else config.exploration.top_p

    team_subdirectory = args.team_subdirectory
    if team_subdirectory is None and args.random_teams:
        team_subdirectory = config.curriculum.team_pool_path

    opponent_team_subdirectory = args.opponent_team_subdirectory
    if opponent_team_subdirectory is None and args.random_opponent_teams:
        opponent_team_subdirectory = config.curriculum.team_pool_path

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
            subdirectory=config.curriculum.team_pool_path,
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
        p1_agent = _build_agent(config, args.device, args.checkpoint)
        p2_agent = _build_agent(
            config, args.device, args.opponent_checkpoint or args.checkpoint
        )

        player1 = DiagnosticBatchInferencePlayer(
            model=p1_agent,
            device=args.device,
            batch_size=args.batch_size,
            batch_timeout=args.batch_timeout,
            probabilistic=not args.greedy,
            embedder=embedder,
            max_battle_steps=args.max_battle_steps,
            battle_format=config.curriculum.battle_format,
            team=p1_team,
            max_concurrent_battles=args.max_concurrent_battles,
            server_configuration=server_config,
            account_configuration=AccountConfiguration(f"showdiagp1{suffix}", None),
            log_level=args.log_level,
            diagnostic_records=diagnostic_records,
        )
        player2 = BatchInferencePlayer(
            p2_agent,
            device=args.device,
            batch_size=args.batch_size,
            batch_timeout=args.batch_timeout,
            probabilistic=not args.greedy,
            embedder=embedder,
            max_battle_steps=args.max_battle_steps,
            battle_format=config.curriculum.battle_format,
            team=p2_team,
            max_concurrent_battles=args.max_concurrent_battles,
            server_configuration=server_config,
            account_configuration=AccountConfiguration(f"showdiagp2{suffix}", None),
            log_level=args.log_level,
        )

        p1_model_player = cast(BatchInferencePlayer, player1)
        p2_model_player = cast(BatchInferencePlayer, player2)
        p1_model_player.temperature = temperature
        p1_model_player.top_p = top_p
        p2_model_player.temperature = temperature
        p2_model_player.top_p = top_p
        p1_model_player.start_inference_loop()
        p2_model_player.start_inference_loop()

        await player1.ps_client.wait_for_login()
        await player2.ps_client.wait_for_login()
        player_setup_seconds = time.perf_counter() - setup_start

        battle_loop_start = time.perf_counter()
        await player1.battle_against(player2, n_battles=args.battles)
        battle_loop_seconds = time.perf_counter() - battle_loop_start
    finally:
        teardown_start = time.perf_counter()
        if isinstance(player1, BatchInferencePlayer):
            player1.teardown_runtime()
        if isinstance(player2, BatchInferencePlayer):
            player2.teardown_runtime()
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


def main() -> None:
    args = build_parser().parse_args()
    random.seed(args.seed)
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
