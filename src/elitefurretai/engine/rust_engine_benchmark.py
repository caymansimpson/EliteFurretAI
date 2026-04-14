"""Low-overhead Rust engine benchmark for the simplified fallback runtime.

This benchmark is intentionally narrower than the model-backed benchmark. It is
useful for checking whether pure Rust battle stepping still works after engine
changes and for separating simulator issues from policy-inference issues.
"""

import argparse
from pathlib import Path

from elitefurretai.etl.team_repo import TeamRepo
from elitefurretai.engine.sync_battle_driver import SyncRustBattleDriver


def _load_team_text(
    format_id: str,
    team_path: str | None,
    repo: TeamRepo,
) -> str:
    if team_path is None:
        return repo.sample_team(format_id)
    return Path(team_path).read_text()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark the in-process Rust battle backend with a constrained RL-style rollout loop."
    )
    parser.add_argument("--format", default="gen9vgc2023regc")
    parser.add_argument("--battles", type=int, default=1000)
    parser.add_argument("--max-concurrent", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--feature-set", default="simple")
    parser.add_argument("--collect-rollouts", action="store_true")
    parser.add_argument("--max-turns-per-battle", type=int, default=100)
    parser.add_argument("--team-path")
    parser.add_argument("--opponent-team-path")
    parser.add_argument("--no-mirror", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo = TeamRepo(shuffle=False)

    p1_team = _load_team_text(args.format, args.team_path, repo)
    if args.opponent_team_path is not None:
        p2_team = Path(args.opponent_team_path).read_text()
    elif args.no_mirror:
        p2_team = repo.sample_team(args.format)
    else:
        p2_team = p1_team

    driver = SyncRustBattleDriver(
        format_id=args.format,
        p1_team=p1_team,
        p2_team=p2_team,
        feature_set=args.feature_set,
        collect_rollouts=args.collect_rollouts,
        seed=args.seed,
        max_turns_per_battle=args.max_turns_per_battle,
    )
    stats = driver.run(
        total_battles=args.battles,
        max_concurrent=args.max_concurrent,
    )

    print(f"completed_battles={stats.completed_battles}")
    print(f"p1_wins={stats.p1_wins}")
    print(f"truncated_battles={stats.truncated_battles}")
    print(f"duration_seconds={stats.duration_seconds:.3f}")
    print(f"battles_per_second={stats.battles_per_second:.3f}")
    print(f"p1_decisions={stats.p1_decisions}")
    print(f"decisions_per_second={stats.decisions_per_second:.3f}")
    print(f"rollout_steps={stats.rollout_steps}")
    print(f"battle_setup_seconds={stats.profile.battle_setup_seconds:.3f}")
    print(f"snapshot_build_seconds={stats.profile.snapshot_build_seconds:.3f}")
    print(f"baseline_choice_seconds={stats.profile.baseline_choice_seconds:.3f}")
    print(f"batched_policy_seconds={stats.profile.batched_policy_seconds:.3f}")
    print(f"engine_step_seconds={stats.profile.engine_step_seconds:.3f}")
    print(f"trajectory_finalize_seconds={stats.profile.trajectory_finalize_seconds:.3f}")
    print(f"policy_embed_seconds={stats.profile.policy_embed_seconds:.3f}")
    print(f"policy_inference_seconds={stats.profile.policy_inference_seconds:.3f}")
    print(f"policy_action_decode_seconds={stats.profile.policy_action_decode_seconds:.3f}")
    print(f"policy_rollout_record_seconds={stats.profile.policy_rollout_record_seconds:.3f}")
    print(f"total_profiled_seconds={stats.profile.total_profiled_seconds:.3f}")


if __name__ == "__main__":
    main()