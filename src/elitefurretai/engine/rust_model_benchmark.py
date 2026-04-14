"""Model-backed Rust benchmark for the simplified fallback runtime.

This benchmark still exists because it is useful for debugging correctness and
for checking whether the fallback Rust path remains runnable, but it no longer
exposes the old optimization-ablation surface. The maintained benchmark matches
the simplified runtime used by `train.py`: local inference, automatic fast
embedding when available, cached request access, and optional diagnostics.
"""

import argparse
from pathlib import Path

from elitefurretai.etl import Embedder
from elitefurretai.etl.team_repo import TeamRepo
from elitefurretai.engine.sync_battle_driver import SyncPolicyPlayer, SyncRustBattleDriver
from elitefurretai.rl.config import RNaDConfig
from elitefurretai.rl.model_io import build_model_from_config, load_agent_from_checkpoint
from elitefurretai.rl.players import RNaDAgent


def _load_team_text(format_id: str, team_path: str | None, repo: TeamRepo) -> str:
    if team_path is None:
        return repo.sample_team(format_id)
    return Path(team_path).read_text()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark the Rust backend with model-backed synchronous self-play policies."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--opponent-checkpoint")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--battles", type=int, default=100)
    parser.add_argument("--max-concurrent", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--feature-set")
    parser.add_argument("--max-turns-per-battle", type=int)
    parser.add_argument("--max-stalled-steps-per-battle", type=int)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--collect-rollouts", action="store_true")
    parser.add_argument("--team-path")
    parser.add_argument("--opponent-team-path")
    parser.add_argument("--no-mirror", action="store_true")
    parser.add_argument("--disable-binding-snapshots", action="store_true")
    parser.add_argument("--diagnostic-log-path")
    parser.add_argument("--error-battle-record-path")
    return parser


def _build_agent(config: RNaDConfig, device: str, checkpoint: str | None) -> RNaDAgent:
    if checkpoint is not None:
        return load_agent_from_checkpoint(checkpoint, device)

    embedder = Embedder(
        format=config.battle_format,
        feature_set=config.embedder_feature_set,
        omniscient=False,
    )
    model = build_model_from_config(config.to_dict(), embedder, device, None)
    return RNaDAgent(model)


def main() -> None:
    args = build_parser().parse_args()
    config = RNaDConfig.load(args.config)
    repo = TeamRepo(shuffle=False)
    feature_set = args.feature_set or config.embedder_feature_set
    max_concurrent = args.max_concurrent or max(1, config.players_per_worker // 2)
    max_turns_per_battle = args.max_turns_per_battle or max(100, config.max_battle_steps * 2)
    max_stalled_steps_per_battle = args.max_stalled_steps_per_battle or max(25, config.max_battle_steps // 2)

    p1_team = _load_team_text(config.battle_format, args.team_path, repo)
    if args.opponent_team_path is not None:
        p2_team = Path(args.opponent_team_path).read_text()
    elif args.no_mirror:
        p2_team = repo.sample_team(config.battle_format)
    else:
        p2_team = p1_team

    p1_agent = _build_agent(config, args.device, args.checkpoint)
    p2_agent = _build_agent(config, args.device, args.opponent_checkpoint or args.checkpoint)

    temperature = args.temperature if args.temperature is not None else config.temperature_at_step(0)
    top_p = args.top_p if args.top_p is not None else config.top_p
    p1_policy = SyncPolicyPlayer(
        p1_agent,
        config.battle_format,
        device=args.device,
        feature_set=feature_set,
        collect_trajectories=args.collect_rollouts,
        probabilistic=not args.greedy,
        temperature=temperature,
        top_p=top_p,
        max_battle_steps=config.max_battle_steps,
        opponent_type="self_play",
    )
    p2_policy = SyncPolicyPlayer(
        p2_agent,
        config.battle_format,
        device=args.device,
        feature_set=feature_set,
        collect_trajectories=False,
        probabilistic=not args.greedy,
        temperature=temperature,
        top_p=top_p,
        max_battle_steps=config.max_battle_steps,
        opponent_type="self_play",
    )

    driver = SyncRustBattleDriver(
        format_id=config.battle_format,
        p1_team=p1_team,
        p2_team=p2_team,
        feature_set=feature_set,
        collect_rollouts=args.collect_rollouts,
        seed=args.seed,
        max_turns_per_battle=max_turns_per_battle,
        max_stalled_steps_per_battle=max_stalled_steps_per_battle,
        p1_policy=p1_policy,
        p2_policy=p2_policy,
        include_binding_snapshots=not args.disable_binding_snapshots,
        diagnostic_log_path=args.diagnostic_log_path,
        error_battle_record_path=args.error_battle_record_path,
    )
    stats = driver.run(
        total_battles=args.battles,
        max_concurrent=max_concurrent,
    )

    print(f"completed_battles={stats.completed_battles}")
    print(f"p1_wins={stats.p1_wins}")
    print(f"truncated_battles={stats.truncated_battles}")
    print(f"non_truncated_battles={stats.non_truncated_battles}")
    print(f"duration_seconds={stats.duration_seconds:.3f}")
    print(f"battles_per_second={stats.battles_per_second:.3f}")
    print(f"non_truncated_battles_per_second={stats.non_truncated_battles_per_second:.3f}")
    print(f"p1_decisions={stats.p1_decisions}")
    print(f"decisions_per_second={stats.decisions_per_second:.3f}")
    print(f"rollout_steps={stats.rollout_steps}")
    print(f"turn_limit_truncations={stats.turn_limit_truncations}")
    print(f"stalled_limit_truncations={stats.stalled_limit_truncations}")
    print(f"dual_limit_truncations={stats.dual_limit_truncations}")
    print(f"p1_rejected_choices={stats.p1_rejected_choices}")
    print(f"p2_rejected_choices={stats.p2_rejected_choices}")
    print(f"p1_fallback_recoveries={stats.p1_fallback_recoveries}")
    print(f"p2_fallback_recoveries={stats.p2_fallback_recoveries}")
    print(f"p1_unrecovered_rejections={stats.p1_unrecovered_rejections}")
    print(f"p2_unrecovered_rejections={stats.p2_unrecovered_rejections}")
    print(f"max_concurrent={max_concurrent}")
    print(f"device={args.device}")
    print(f"model_checkpoint={args.checkpoint or 'fresh_config_model'}")
    print(f"opponent_checkpoint={args.opponent_checkpoint or args.checkpoint or 'fresh_config_model'}")
    print(f"feature_set={feature_set}")
    print(f"max_turns_per_battle={max_turns_per_battle}")
    print(f"max_stalled_steps_per_battle={max_stalled_steps_per_battle}")
    print(f"binding_snapshots={not args.disable_binding_snapshots}")
    print(f"diagnostic_log_path={args.diagnostic_log_path or 'disabled'}")
    print(f"error_battle_record_path={args.error_battle_record_path or 'disabled'}")
    print(f"probabilistic={not args.greedy}")
    print(f"temperature={temperature}")
    print(f"top_p={top_p}")
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
