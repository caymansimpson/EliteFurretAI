# -*- coding: utf-8 -*-
"""This module plays a bunch of players against each other to find errors in Inference classes.
Supports both sequential and multithreaded execution via CLI arguments.
"""

import asyncio
import itertools
import os.path
import re
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from poke_env.battle.move_category import MoveCategory
from poke_env.data.gen_data import GenData
from poke_env.player.baselines import RandomPlayer
from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.ps_client.server_configuration import (
    LocalhostServerConfiguration,
    ServerConfiguration,
)
from poke_env.teambuilder.teambuilder import Teambuilder

from elitefurretai.etl.team_repo import TeamRepo
from elitefurretai.inference.battle_inference import BattleInference
from elitefurretai.inference.inference_utils import battle_to_str
from elitefurretai.inference.item_inference import ItemInference
from elitefurretai.inference.speed_inference import SpeedInference


class FuzzTestPlayer(RandomPlayer):
    def __init__(
        self,
        account_configuration: Optional[AccountConfiguration] = None,
        *,
        avatar: Optional[str] = None,
        battle_format: str = "gen9randombattle",
        log_level: Optional[int] = None,
        max_concurrent_battles: int = 1,
        accept_open_team_sheet: bool = False,
        save_replays: Union[bool, str] = False,
        server_configuration: ServerConfiguration = LocalhostServerConfiguration,
        start_timer_on_battle_start: bool = False,
        start_listening: bool = True,
        ping_interval: Optional[float] = 20.0,
        ping_timeout: Optional[float] = 20.0,
        team: Optional[Union[str, Teambuilder]] = None,
    ):
        super().__init__(
            account_configuration=account_configuration,
            avatar=avatar,
            battle_format=battle_format,
            log_level=log_level,
            max_concurrent_battles=max_concurrent_battles,
            accept_open_team_sheet=accept_open_team_sheet,
            save_replays=save_replays,
            server_configuration=server_configuration,
            start_timer_on_battle_start=start_timer_on_battle_start,
            start_listening=start_listening,
            ping_interval=ping_interval,
            ping_timeout=ping_timeout,
            team=team,
        )
        self._inferences: Dict[str, Any] = {}
        self._speed_inferences: Dict[str, Any] = {}
        self._item_inferences: Dict[str, Any] = {}

    def teampreview(self, battle):
        # Initialize Inferences
        inferences = BattleInference(battle)
        self._inferences[battle.battle_tag] = inferences

        # Speed and Item Inferences will fill inferences; only p1 will do inferences
        if battle.player_role == "p1":
            self._speed_inferences[battle.battle_tag] = SpeedInference(battle, inferences)
            self._item_inferences[battle.battle_tag] = ItemInference(battle, inferences)

            self._speed_inferences[battle.battle_tag].update(battle)
            self._item_inferences[battle.battle_tag].update(battle)
        return "/team 1234"

    def choose_move(self, battle):
        if battle.battle_tag not in self._inferences:
            raise ValueError("SHOWDOWN IS BROKEN WITH TEAMPREVIEW BUG! RESTART IT!")

        # Don't deal with battle.force_switch until we fix it in poke_env
        if not any(battle.force_switch) and battle.player_role == "p1":
            self._speed_inferences[battle.battle_tag].update(battle)
            self._item_inferences[battle.battle_tag].update(battle)

        return self.choose_random_doubles_move(battle)  # pyright: ignore


def check_ground_truth(p1, p2, battle_tag):
    msg = ""
    counts: Dict[str, int] = {}

    # Check P2's BattleInference
    for mon in p1.battles[battle_tag].teampreview_team:
        ident = mon.identifier(p2.battles[battle_tag].opponent_role)
        if not p2._inferences[battle_tag].is_tracking(ident):
            continue

        flags = p2._inferences[battle_tag].get_flags(ident)
        mon_in_battle = p1.battles[battle_tag].team[ident]  # Checking in case of knockoff

        if (
            flags["item"] not in [mon.item, None, GenData.UNKNOWN_ITEM, mon_in_battle.item]
            and flags["item"] == "choicescarf"
        ):
            msg += "\n\n======================================================================\n=========================== ERROR FOUND :( ===========================\n======================================================================\n"
            msg += "Error Type: (error_speed_item) Erroneously found item that the mon didn't have, due to incorrect Speed Calculations\n"
            msg += f"{mon_in_battle.name} was found to have {flags['item']} when it actually had {mon.item}\n\n"
            msg += battle_to_str(p2.battles[battle_tag], p1)
            counts["error_speed_item"] = counts.get("speed_item", 0) + 1

        elif flags["item"] not in [
            mon.item,
            None,
            GenData.UNKNOWN_ITEM,
            mon_in_battle.item,
        ]:
            msg += "\n\n======================================================================\n=========================== ERROR FOUND :( ===========================\n======================================================================\n"
            msg += "Error Type (error_item): Erroneously found item that the mon didn't have\n"
            msg += f"{mon_in_battle.name} was found to have {flags['item']} when it actually had {mon.item}\n\n"
            msg += battle_to_str(p2.battles[battle_tag], p1)
            counts["error_item"] = counts.get("error_item", 0) + 1

        elif (
            not flags["can_be_choice"]
            and mon.item
            and mon_in_battle.item
            and mon_in_battle.item.startswith("choice")
            and mon.item.startswith("choice")
        ):
            msg += "\n\n======================================================================\n=========================== ERROR FOUND :( ===========================\n======================================================================\n"
            msg += "Error Type (error_can_be_choice): Erroneously found a mon can't have a choice item when they do\n"
            msg += f"{mon_in_battle.name} was found not to have choice, but has the item '{mon.item}'\n\n"
            msg += battle_to_str(p2.battles[battle_tag], p1)
            counts["error_can_be_choice"] = counts.get("error_can_be_choice", 0) + 1

        elif flags["has_status_move"] and not any(
            map(lambda x: x.category == MoveCategory.STATUS, mon.moves.values())
        ):
            msg += "\n\n======================================================================\n=========================== ERROR FOUND :( ===========================\n======================================================================\n"
            msg += "Error Type (error_has_status_move): Erroneously found a mon can't be assault vest (e.g. they used a status move) when they could be\n"
            msg += f"{mon_in_battle.name} has the moves: [{', '.join(map(lambda x: x.id, mon.moves.values()))}]\n\n"
            msg += battle_to_str(p2.battles[battle_tag], p1)
            counts["error_has_status_move"] = counts.get("error_has_status_move", 0) + 1

        elif (
            mon_in_battle.stats["spe"] < flags["spe"][0]
            or mon_in_battle.stats["spe"] > flags["spe"][1]
        ) and not (
            mon_in_battle.item == "choicescarf"
            and flags["spe"][1]
            == BattleInference.load_opponent_set(mon)["spe"][1]  # We guess theyre maxspeed
        ):
            msg += "\n\n======================================================================\n=========================== ERROR FOUND :( ===========================\n======================================================================\n"
            msg += "Error Type (error_speed): Found an erroneous speed.\n"
            msg += f"{mon_in_battle.name} has a {mon_in_battle.stats['spe']} Speed stat, but we've bounded it between {flags['spe'][0]} and {flags['spe'][1]}\n\n"
            msg += battle_to_str(p2.battles[battle_tag], p1)
            counts["error_speed"] = counts.get("error_speed", 0) + 1

    return msg, counts


def get_players(team_repo: TeamRepo, format: str) -> List[FuzzTestPlayer]:
    players: List[FuzzTestPlayer] = []
    for team_name, team in team_repo.teams[format].items():
        name = team_name[:17]

        # ef this noise
        if (
            "Ditto" in team
            or "Zoroark" in team
            or ("Dondozo" in team and "Tatsugiri" in team)
            or "Lagging Tail" in team
            or "Iron Ball" in team
        ):
            continue

        # Don't want to deal with non-english characters
        if bool(re.search(r"[^a-zA-Z0-9\- _]", name)):
            continue

        # No duplicate usernames
        if name in map(lambda x: x.username, players):
            continue

        players.append(
            FuzzTestPlayer(
                AccountConfiguration(name, None),
                battle_format=format,
                team=team,
            )
        )
    return players


# ========================= MULTITHREADED COMPONENTS =========================


@dataclass
class BattleResult:
    """Represents the outcome of a battle"""

    p1_username: str
    p2_username: str
    battle_tag: str
    errors: str
    error_counts: Dict[str, int]
    success: bool


class ResultAggregator:
    """Thread-safe aggregator for battle results"""

    def __init__(self):
        import threading

        self.lock = threading.Lock()
        self.total_battles = 0
        self.successful_battles = 0
        self.error_counts: Dict[str, int] = {}
        self.error_log = ""

    def add_result(self, result: BattleResult):
        with self.lock:
            self.total_battles += 1
            if result.success:
                self.successful_battles += 1

            for error_type, count in result.error_counts.items():
                self.error_counts[error_type] = (
                    self.error_counts.get(error_type, 0) + count
                )

            if result.errors:
                self.error_log += result.errors


class PlayerPool:
    """Manages a pool of players, ensuring each player is only used in one battle at a time"""

    def __init__(self, players: List[FuzzTestPlayer]):
        import threading

        self.players = set(players)
        self.in_use: Set[FuzzTestPlayer] = set()
        self.lock = threading.Lock()

    async def acquire_pair(self) -> Optional[Tuple[FuzzTestPlayer, FuzzTestPlayer]]:
        """Attempt to acquire a pair of available players"""
        with self.lock:
            available = self.players - self.in_use
            if len(available) < 2:
                return None

            # Get first two available players
            p1, p2 = list(available)[:2]
            self.in_use.add(p1)
            self.in_use.add(p2)
            return p1, p2

    def release_pair(self, p1: FuzzTestPlayer, p2: FuzzTestPlayer):
        """Release a pair of players back to the available pool"""
        with self.lock:
            self.in_use.remove(p1)
            self.in_use.remove(p2)


async def run_battle(p1: FuzzTestPlayer, p2: FuzzTestPlayer) -> BattleResult:
    """Run a single battle between two players and return the results"""
    await p1.battle_against(p2)

    # Find the battle tag for this matchup
    battle_tag = None
    for tag, battle in p1.battles.items():
        if battle.opponent_username == p2.username:
            battle_tag = tag
            break

    if not battle_tag:
        return BattleResult(p1.username, p2.username, "", "", {}, False)

    # Check ground truth from both perspectives
    msg1, counts1 = check_ground_truth(p1, p2, battle_tag)
    msg2, counts2 = check_ground_truth(p2, p1, battle_tag)

    # Combine error counts
    combined_counts = {}
    for key in set(counts1.keys()) | set(counts2.keys()):
        combined_counts[key] = counts1.get(key, 0) + counts2.get(key, 0)

    return BattleResult(
        p1_username=p1.username,
        p2_username=p2.username,
        battle_tag=battle_tag,
        errors=msg1 + msg2,
        error_counts=combined_counts,
        success=not bool(msg1 or msg2),
    )


async def battle_worker(
    player_pool: PlayerPool,
    result_aggregator: ResultAggregator,
    remaining_battles: asyncio.Event,
):
    """Worker function that processes battles using available players from the pool"""
    while remaining_battles.is_set():
        try:
            # Try to get an available pair of players
            player_pair = await player_pool.acquire_pair()
            if not player_pair:
                # No players available, wait a bit and try again
                await asyncio.sleep(0.1)
                continue

            p1, p2 = player_pair
            try:
                result = await run_battle(p1, p2)
                result_aggregator.add_result(result)
            finally:
                # Always release the players back to the pool
                player_pool.release_pair(p1, p2)

        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"Error in battle worker: {e}")


# ========================= MAIN EXECUTION =========================


async def run_sequential(
    total_battles: int, format: str, should_print: bool, filepath: str
):
    """Run battles sequentially (original implementation)"""
    print("\n\n\n\n\n\n\n\n\033[92mStarting Sequential Fuzz Test!\033[0m\n")
    print("Loading and validating teams, then creating players...")

    # Load Teams into one Player each
    players = get_players(TeamRepo(validate=False, verbose=False), format)
    print(f"Finished loading {len(players)} teams to battle against each other!")

    num_matchups = max(
        1, int(total_battles / len(list(itertools.combinations(players, 2))))
    )
    print(f"Now starting Player battles with {num_matchups} battles per matchup")

    total_errors = ""
    total = 0
    original_time = time.time()

    for p1, p2 in itertools.combinations(players, 2):
        t0 = time.time()

        for i in range(num_matchups):
            total += 1
            print(
                f"\rStarting battle #{total} between {p1.username} and {p2.username}...\r",
                end="",
                flush=True,
            )
            await p1.battle_against(p2)
            if total >= total_battles:
                break

        print(
            f"Finished {num_matchups} battles between {p1.username} and {p2.username} in {round((time.time() - t0), 2)}"
            + f"s ({round((time.time() - t0) * 1.0 / num_matchups, 2)}s each)"
        )

        # Stop when we've done enough
        if total >= total_battles:
            break

    print("\n\n======== Finished all battles!! ========")
    print(
        f"\tin {round(time.time() - original_time, 2)} sec for "
        + f"{round((time.time() - original_time) * 1.0 / total, 2)} sec per battle"
    )
    print(
        "Now going through Inference to check the right ground truth! This method won't catch cases when"
        + " I should have inferred items, but didn't. It will only catch when I infer incorrectly..."
    )

    # Check for implicit inference errors
    total_success = 0
    total = 0
    counts: Dict[str, int] = {}
    for p1, p2 in itertools.combinations(players, 2):
        for battle_tag, battle in p1.battles.items():
            if battle.opponent_username == p2.username:
                total += 2

                msg, counts1 = check_ground_truth(p1, p2, battle_tag)
                if msg == "":
                    total_success += 1
                else:
                    total_errors += msg

                msg, counts2 = check_ground_truth(p2, p1, battle_tag)
                if msg == "":
                    total_success += 1
                else:
                    total_errors += msg

                for key in counts1:
                    counts[key] = counts.get(key, 0) + counts1[key]
                for key in counts2:
                    counts[key] = counts.get(key, 0) + counts2[key]

    print("\n======== Finished checking all battles for inferences ========")
    print(
        f"Finished {total} battles with {round(total_success * 1.0 / total * 100, 2)}% success rate"
    )
    for error_type in counts:
        print(f"\tError Type [{error_type}] was found {counts[error_type]} times")

    print(
        f"Writing these to {filepath}). {'Also, printing them here:' if should_print else ''}"
    )
    with open(filepath, "w") as f:
        f.write(total_errors)

    if should_print:
        print(total_errors)


async def run_multithreaded(
    total_battles: int, format: str, should_print: bool, filepath: str, num_threads: int
):
    """Run battles with multiple concurrent workers"""
    print("\n\n\n\n\n\n\n\n\033[92mStarting Multithreaded Fuzz Test!\033[0m\n")
    print(f"Running with up to {num_threads} concurrent battles")
    print("Loading and validating teams, then creating players...")

    # Create players
    players = get_players(TeamRepo(validate=False, verbose=False), format)
    print(
        f"Created {len(players)} legal players (filtered out Dondozo, Ditto and Zoroark teams)"
    )

    # Initialize player pool and result aggregator
    player_pool = PlayerPool(players)
    result_aggregator = ResultAggregator()

    # Create an event to signal when we should stop creating new battles
    remaining_battles = asyncio.Event()
    remaining_battles.set()

    print(f"Starting {total_battles} total battles! Will update progress every second:")

    # Create and start worker tasks
    original_time = time.time()
    workers = []
    for _ in range(num_threads):
        worker = asyncio.create_task(
            battle_worker(player_pool, result_aggregator, remaining_battles)
        )
        workers.append(worker)

    # Monitor progress and stop when we reach total_battles
    while result_aggregator.total_battles < total_battles:
        pct = result_aggregator.total_battles * 100.0 / total_battles
        num = result_aggregator.total_battles  # type: ignore
        print(f"\r\tCompleted {pct:.1f}% battles with {num} done...", end="", flush=True)
        await asyncio.sleep(1)

    # Signal workers to stop
    remaining_battles.clear()

    # Cancel workers
    for worker in workers:
        worker.cancel()
    await asyncio.gather(*workers, return_exceptions=True)

    # Print results
    print("\r\n\n======== Finished all battles!! ========")
    print(
        f"Finished {result_aggregator.total_battles} battles in {round(time.time() - original_time, 2)} sec"
    )
    print(
        f"\tfor {round((time.time() - original_time) * 1.0 / result_aggregator.total_battles, 2)} sec per battle"
    )

    success_rate = round(
        result_aggregator.successful_battles * 100.0 / result_aggregator.total_battles,
        2,
    )
    print(f"\nFinished with {success_rate}% success rate")

    for error_type, count in result_aggregator.error_counts.items():
        print(f"\tError Type [{error_type}] was found {count} times")

    # Write errors to file
    print(f"\nWriting errors to {filepath}")
    with open(filepath, "w") as f:
        f.write(result_aggregator.error_log)

    if should_print:
        print(result_aggregator.error_log)


async def main():
    """
    Main entry point. Supports CLI arguments:
    - <number>: Total battles to run (default: 1000)
    - print: Print errors to console
    - threads=<N> or --threads <N>: Number of concurrent workers (default: 1 for sequential)

    Examples:
        python fuzz_inference.py 500 print threads=4
        python fuzz_inference.py 1000 --threads 8
    """
    # Parse CLI arguments
    total_battles = 1000
    num_threads = 1
    should_print = False
    format = "gen9vgc2024regh"
    filepath = os.path.expanduser("~/Desktop/fuzz_errors.txt")

    for i, arg in enumerate(sys.argv[1:]):
        if arg.isdigit():
            total_battles = int(arg)
        elif arg == "print":
            should_print = True
        elif arg.startswith("threads="):
            num_threads = int(arg.split("=")[1])
        elif arg == "--threads" and i + 1 < len(sys.argv) - 1:
            num_threads = int(sys.argv[i + 2])

    # Limit threads to reasonable bounds
    max_threads = min((os.cpu_count() or 8) // 2, 8)
    if num_threads > max_threads:
        print(f"Warning: Limiting threads from {num_threads} to {max_threads}")
        num_threads = max_threads

    # Run appropriate version
    if num_threads > 1:
        await run_multithreaded(total_battles, format, should_print, filepath, num_threads)
    else:
        await run_sequential(total_battles, format, should_print, filepath)


if __name__ == "__main__":
    asyncio.run(main())
