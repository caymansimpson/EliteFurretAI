"""Round-robin evaluation across 5 agents using vgcbench.txt for both sides.

200 battles per pairing; vgc_bench, max_damage, max_base_power,
simple_heuristic, and balmy-cloud-70 step 10700.
"""

from __future__ import annotations

import json
import time
from itertools import combinations
from pathlib import Path

from elitefurretai.engine.showdown_server_manager import (
    launch_showdown_servers,
    shutdown_showdown_servers,
)
from elitefurretai.rl.analyze.evaluate import build_cells, run_eval_parallel
from elitefurretai.rl.analyze.player_factory import parse_player_specification
from elitefurretai.rl.analyze.team_provider import parse_team_specification

AGENTS = [
    ("vgc_bench", "vgc_bench"),
    ("max_damage", "max_damage"),
    ("max_base_power", "max_base_power"),
    ("simple_heuristic", "simple_heuristic"),
    (
        "data/models/rl/balmy-cloud-70/ghosts/main_model_step_10700.pt",
        "balmy70_step10700",
    ),
]
TEAM_FILE = "data/teams/gen9vgc2024regg/vgcbench.txt"
BATTLE_FORMAT = "gen9vgc2024regg"
BATTLES_PER_PAIR = 200
NUM_SERVERS = 4
WORKERS = 4
START_PORT = 8200
DEVICE = "cuda"
LAUNCH_SERVERS = False  # Reuse already-running servers on START_PORT..+NUM_SERVERS-1


def main() -> None:
    t1 = parse_team_specification(TEAM_FILE, battle_format=BATTLE_FORMAT)
    t2 = parse_team_specification(TEAM_FILE, battle_format=BATTLE_FORMAT)
    cells = build_cells(
        t1, t2, cell_iteration=False, team1_path=TEAM_FILE, team2_path=TEAM_FILE
    )

    server_procs = (
        launch_showdown_servers(NUM_SERVERS, START_PORT) if LAUNCH_SERVERS else []
    )
    server_urls = [f"localhost:{START_PORT + i}" for i in range(NUM_SERVERS)]

    results = []
    try:
        for i, j in combinations(range(len(AGENTS)), 2):
            raw1, label1 = AGENTS[i]
            raw2, label2 = AGENTS[j]
            p1 = parse_player_specification(
                raw1, device=DEVICE, battle_format=BATTLE_FORMAT
            )
            p2 = parse_player_specification(
                raw2, device=DEVICE, battle_format=BATTLE_FORMAT
            )
            run_tag = format(int(time.time() * 1000) % 65536, "04x")

            print(f"\n=== {label1} vs {label2} ===", flush=True)
            t_start = time.time()
            res = run_eval_parallel(
                p1=p1,
                p2=p2,
                cells=cells,
                battles_per_cell=BATTLES_PER_PAIR,
                server_urls=server_urls,
                workers=WORKERS,
                run_tag=run_tag,
                executor="process",
            )
            duration = time.time() - t_start
            wr1 = res.player1_win_rate * 100
            wr2 = (
                res.player2_wins / res.battles_played * 100 if res.battles_played else 0.0
            )
            print(
                f"{label1} ({wr1:.2f}%) vs {label2} ({wr2:.2f}%) | "
                f"P1={res.player1_wins} P2={res.player2_wins} "
                f"Ties={res.ties} N={res.battles_played} "
                f"[{duration:.1f}s]",
                flush=True,
            )
            results.append(
                {
                    "p1_label": label1,
                    "p2_label": label2,
                    "p1_wins": res.player1_wins,
                    "p2_wins": res.player2_wins,
                    "ties": res.ties,
                    "battles_played": res.battles_played,
                    "p1_win_rate": res.player1_win_rate,
                    "p2_win_rate": (
                        res.player2_wins / res.battles_played
                        if res.battles_played
                        else 0.0
                    ),
                    "duration_sec": round(duration, 2),
                }
            )
    finally:
        shutdown_showdown_servers(server_procs)

    out_path = Path("data/eval_results/round_robin_5agents.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved results to {out_path}", flush=True)

    print("\n=== ROUND-ROBIN SUMMARY ===")
    print(f"{'P1':<22} {'P2':<22} {'P1WR':>7} {'N':>5}")
    for r in results:
        print(
            f"{r['p1_label']:<22} {r['p2_label']:<22} "
            f"{r['p1_win_rate'] * 100:>6.2f}% {r['battles_played']:>5}"
        )

    labels = [lbl for _, lbl in AGENTS]
    wins = {lbl: 0 for lbl in labels}
    games = {lbl: 0 for lbl in labels}
    for r in results:
        wins[r["p1_label"]] += r["p1_wins"]
        wins[r["p2_label"]] += r["p2_wins"]
        games[r["p1_label"]] += r["battles_played"]
        games[r["p2_label"]] += r["battles_played"]
    print("\n=== OVERALL WIN RATES (across all pairings) ===")
    for lbl in labels:
        n = games[lbl]
        wr = wins[lbl] / n * 100 if n else 0.0
        print(f"  {lbl:<22} {wr:6.2f}% ({wins[lbl]}/{n})")


if __name__ == "__main__":
    main()
