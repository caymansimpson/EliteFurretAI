# -*- coding: utf-8 -*-
"""Threaded version of fuzz_test, where we play a bunch of players against each
other to find errors in our Inference classes. This threaded version is
specifically for when you get to the last 0.01% of errors, and need to find those errors
more quickly.
"""

import asyncio
import os.path
import sys
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from elitefurretai.scripts.fuzz_test import (
    FuzzTestPlayer,
    check_ground_truth,
    get_players,
)
from elitefurretai.utils.team_repo import TeamRepo


# A Dataclass that represents the outcome of a battle
@dataclass
class BattleResult:
    p1_username: str
    p2_username: str
    battle_tag: str
    errors: str
    error_counts: Dict[str, int]
    success: bool


# The ResultAggregator class is used to keep track of the results of a set of battles.
# It is thread-safe and can be used by multiple threads to report the results of their battles.
class ResultAggregator:
    def __init__(self):
        self.lock = threading.Lock()
        self.total_battles = 0
        self.successful_battles = 0
        self.error_counts = {}
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
        self.players = set(players)
        self.in_use = set()
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


async def main():

    # Get Parameters of fuzz testing; for the number of threads, you have to initiate showdown
    # with the appropriate number of workers (can do this through CLI or config) to take advantage
    # of parallelism
    num = None
    for arg in sys.argv:
        if arg.isdigit():
            num = arg
    total_battles = int(num) if num is not None else 1000
    format = "gen9vgc2024regh"
    should_print = "print" in sys.argv
    filepath = os.path.expanduser("~/Desktop/fuzz_errors.txt")
    num_threads = int(min((os.cpu_count() or 8) / 2, 8))  # Limit to 8 threads maximum

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
        num = result_aggregator.total_battles
        print(f"\r\tCompleted {pct}% battles with {num} done...", end="", flush=True)
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
        f"\tfor {round((time.time() - original_time) * 1. / result_aggregator.total_battles, 2)} sec per battle"
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


if __name__ == "__main__":
    asyncio.run(main())
