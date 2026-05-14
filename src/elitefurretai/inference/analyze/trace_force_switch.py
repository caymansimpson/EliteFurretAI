# -*- coding: utf-8 -*-
"""Trace force_switch handling across a small number of live battles.

Goal: empirically verify whether Showdown's reversed request/event ordering has
landed, so we can drop the `force_switch` workaround branch in
`speed_inference.update()` and `item_inference.update()`.

What this records
-----------------
At every `choose_move` call on the instrumented player, we dump a JSON line
containing the inference-relevant slice of battle state:

    {
        "battle_tag": "...",
        "call_idx":   <monotonic counter per battle>,
        "turn":       battle.turn,
        "force_switch": battle.force_switch,
        "observation_keys": sorted(battle.observations.keys()),
        "max_obs_event_count": len(battle.observations[max(...)].events),
        "max_obs_tail":        last 3 events of the latest observation,
        "current_obs_event_count": len(battle.current_observation.events),
        "current_obs_tail":        last 3 events of the in-flight observation,
        "rqid":   battle.last_request.get("rqid"),
        "force_switch_request": battle.last_request.get("forceSwitch"),
    }

How to read the output
----------------------
For every line where `force_switch` is truthy, compare
`max_obs_event_count` vs `current_obs_event_count`:

- If `current_obs_event_count == 0` (or the tails match), Showdown has
  committed the triggering events to `battle.observations[max(...)]` before
  the request arrived. The workaround in both inference modules can be
  dropped.

- If `current_obs_event_count > 0` AND its tail contains events not present
  in the max observation's tail, the in-flight buffer still carries
  pre-commit data and the workaround is still load-bearing.

Run
---
1. Make sure a local Showdown server is running. Easiest path:
       cd <showdown-checkout> && node pokemon-showdown start --no-security
2. From the repo root, with the venv active:
       source ../venv/bin/activate
       python -m elitefurretai.inference.analyze.trace_force_switch \
           --battles 5 --format gen9vgc2024regh --out /tmp/force_switch_trace.jsonl
3. Inspect:
       grep '"force_switch": .true' /tmp/force_switch_trace.jsonl | head
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Union

from poke_env.battle.battle import Battle
from poke_env.battle.double_battle import DoubleBattle
from poke_env.player.baselines import RandomPlayer
from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration

from elitefurretai.etl.team_repo import TeamRepo


def _event_tail(events: List[Any], n: int = 3) -> List[Any]:
    """Last `n` events. Each event is itself a list[str]; we keep it as-is so the
    JSON consumer can grep for `force_switch`-adjacent triggers (faint, |request|)."""
    return events[-n:] if events else []


def _snapshot(battle: Union[Battle, DoubleBattle], *, call_idx: int, hook: str) -> Dict[str, Any]:
    obs_keys = sorted(battle.observations.keys())
    if obs_keys:
        max_obs = battle.observations[obs_keys[-1]]
        max_obs_count = len(max_obs.events)
        max_obs_tail = _event_tail(max_obs.events)
    else:
        max_obs_count = 0
        max_obs_tail = []

    current_obs = battle.current_observation
    last_req = battle.last_request or {}

    return {
        "hook": hook,
        "call_idx": call_idx,
        "battle_tag": battle.battle_tag,
        "player_role": battle.player_role,
        "turn": battle.turn,
        "force_switch": battle.force_switch,
        "teampreview": battle.teampreview,
        "observation_keys": obs_keys,
        "max_obs_event_count": max_obs_count,
        "max_obs_tail": max_obs_tail,
        "current_obs_event_count": len(current_obs.events),
        "current_obs_tail": _event_tail(current_obs.events),
        "rqid": last_req.get("rqid"),
        "force_switch_request": last_req.get("forceSwitch"),
        "wait": last_req.get("wait"),
    }


class TracePlayer(RandomPlayer):
    """Random player that logs inference-relevant state to a JSONL sink.

    Only p1 logs, to keep the trace one-sided (matches how `SpeedInference` /
    `ItemInference` are usually wired in `fuzz_inference.py`).
    """

    def __init__(self, *args: Any, sink: Optional[Any] = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._sink = sink
        self._call_counter: Dict[str, int] = {}

    def _log(self, battle: Union[Battle, DoubleBattle], hook: str) -> None:
        if self._sink is None or battle.player_role != "p1":
            return
        idx = self._call_counter.get(battle.battle_tag, 0)
        self._call_counter[battle.battle_tag] = idx + 1
        snap = _snapshot(battle, call_idx=idx, hook=hook)
        self._sink.write(json.dumps(snap, default=str) + "\n")
        self._sink.flush()

    def teampreview(self, battle):  # type: ignore[override]
        self._log(battle, hook="teampreview")
        return "/team 1234"

    def choose_move(self, battle):  # type: ignore[override]
        self._log(battle, hook="choose_move")
        return self.choose_random_doubles_move(battle)  # pyright: ignore[reportAttributeAccessIssue]


def _build_pair(
    team_repo: TeamRepo, battle_format: str, sink: Any, suffix: str
) -> List[TracePlayer]:
    """Pick two clean teams from the repo and build a player pair sharing one sink."""
    players: List[TracePlayer] = []
    seen_names: set[str] = set()
    for team_name, team in team_repo.teams[battle_format].items():
        if len(players) == 2:
            break
        # Skip teams that historically break inference fuzzing.
        if (
            "Ditto" in team
            or "Zoroark" in team
            or ("Dondozo" in team and "Tatsugiri" in team)
            or "Lagging Tail" in team
            or "Iron Ball" in team
        ):
            continue
        name = team_name[:14] + suffix
        if bool(re.search(r"[^a-zA-Z0-9\- _]", name)) or name in seen_names:
            continue
        seen_names.add(name)
        players.append(
            TracePlayer(
                AccountConfiguration(name, None),
                battle_format=battle_format,
                team=team,
                server_configuration=LocalhostServerConfiguration,
                sink=sink,
            )
        )
    if len(players) < 2:
        raise RuntimeError(
            f"Could not find two clean teams for {battle_format!r} in TeamRepo"
        )
    return players


async def run(num_battles: int, battle_format: str, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    team_repo = TeamRepo(validate=False, verbose=False)

    with open(out_path, "w") as sink:
        # Write a header line so consumers know which run produced this file.
        sink.write(
            json.dumps(
                {
                    "_meta": True,
                    "started_at": time.time(),
                    "battle_format": battle_format,
                    "num_battles_requested": num_battles,
                }
            )
            + "\n"
        )

        # Single pair, sequential battles — enough to surface force_switch turns
        # without needing the threaded harness in fuzz_inference.py.
        suffix = f"-{int(time.time()) % 10000:04d}"
        p1, p2 = _build_pair(team_repo, battle_format, sink, suffix=suffix)

        for i in range(num_battles):
            print(f"[trace] battle {i + 1}/{num_battles} ...", flush=True)
            await p1.battle_against(p2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--battles", type=int, default=5)
    parser.add_argument("--format", default="gen9vgc2024regh")
    parser.add_argument("--out", default="/tmp/force_switch_trace.jsonl")
    args = parser.parse_args()
    asyncio.run(run(args.battles, args.format, args.out))


if __name__ == "__main__":
    main()
