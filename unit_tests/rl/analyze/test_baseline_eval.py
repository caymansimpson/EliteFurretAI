# -*- coding: utf-8 -*-
"""Unit tests for baseline_eval (compute_score, dispatch, payload, cleanup)."""

import math
from typing import Dict
from unittest.mock import MagicMock, patch

import pytest
import yaml

from elitefurretai.rl.analyze.baseline_eval import (
    BucketRunResult,
    MultiBucketEvalResult,
    _opponent_kwargs,
    _run_opponent_bucket,
    _split_battles_by_format,
    build_eval_log_payload,
    compute_score,
    run,
)
from elitefurretai.rl.analyze.evaluate import EvalResult
from elitefurretai.rl.config import (
    CurriculumConfig,
    EvalConfig,
    OpponentEvalSpec,
    RNaDConfig,
)


def _targets_all(t: float):
    return {"a": t, "b": t, "c": t, "d": t}


def _weights_unit():
    return {"a": 1.0, "b": 1.0, "c": 1.0, "d": 1.0}


class TestComputeScore:
    def test_all_floors_met_exactly_returns_zero(self):
        rates = {"a": 0.8, "b": 0.8, "c": 0.8, "d": 0.8}
        score, br = compute_score(
            rates, _targets_all(0.8), _weights_unit(), surplus_alpha=1.0
        )
        assert score == 0.0
        assert br["deficit_l2_pp"] == 0.0
        assert br["surplus_sum_pp"] == 0.0

    def test_uniform_overshoot_pure_surplus(self):
        # 10pp surplus on each of 4 buckets at alpha=1 -> 10*4 = 40
        rates = {"a": 0.9, "b": 0.9, "c": 0.9, "d": 0.9}
        score, br = compute_score(
            rates, _targets_all(0.8), _weights_unit(), surplus_alpha=1.0
        )
        assert score == pytest.approx(40.0)
        assert br["deficit_l2_pp"] == 0.0
        assert br["surplus_sum_pp"] == pytest.approx(40.0)

    def test_uniform_deficit_pure_l2_penalty(self):
        # 10pp deficit on each of 4 buckets -> 10^2 * 4 = 400, score is negative
        rates = {"a": 0.7, "b": 0.7, "c": 0.7, "d": 0.7}
        score, br = compute_score(
            rates, _targets_all(0.8), _weights_unit(), surplus_alpha=1.0
        )
        assert score == pytest.approx(-400.0)
        assert br["deficit_l2_pp"] == pytest.approx(400.0)
        assert br["surplus_sum_pp"] == 0.0

    def test_mixed_deficit_and_surplus(self):
        # a: 20pp deficit -> 400 ; b: 10pp surplus -> 10 ; c,d on target -> 0
        rates = {"a": 0.6, "b": 0.9, "c": 0.8, "d": 0.8}
        score, br = compute_score(
            rates, _targets_all(0.8), _weights_unit(), surplus_alpha=1.0
        )
        assert br["deficit_l2_pp"] == pytest.approx(400.0)
        assert br["surplus_sum_pp"] == pytest.approx(10.0)
        assert score == pytest.approx(10.0 - 400.0)

    def test_weights_scale_both_terms(self):
        rates = {"a": 0.7, "b": 0.9}  # 10pp deficit / 10pp surplus
        targets = {"a": 0.8, "b": 0.8}
        weights = {"a": 2.0, "b": 3.0}
        score, br = compute_score(rates, targets, weights, surplus_alpha=1.0)
        # deficit: 2.0 * 10^2 = 200 ; surplus: 3.0 * 10 = 30
        assert br["deficit_l2_pp"] == pytest.approx(200.0)
        assert br["surplus_sum_pp"] == pytest.approx(30.0)
        assert score == pytest.approx(30.0 - 200.0)

    def test_surplus_alpha_scales_surplus_term_only(self):
        rates = {"a": 0.9}  # 10pp surplus
        targets = {"a": 0.8}
        weights = {"a": 1.0}
        s0, _ = compute_score(rates, targets, weights, surplus_alpha=0.0)
        s5, _ = compute_score(rates, targets, weights, surplus_alpha=0.5)
        s1, _ = compute_score(rates, targets, weights, surplus_alpha=1.0)
        assert s0 == 0.0
        assert s5 == pytest.approx(5.0)
        assert s1 == pytest.approx(10.0)

    def test_edge_rate_zero_max_deficit(self):
        rates = {"a": 0.0}
        targets = {"a": 1.0}
        weights = {"a": 1.0}
        score, br = compute_score(rates, targets, weights, surplus_alpha=1.0)
        assert br["deficit_l2_pp"] == pytest.approx(10000.0)
        assert br["surplus_sum_pp"] == 0.0
        assert score == pytest.approx(-10000.0)

    def test_edge_rate_one_max_surplus(self):
        rates = {"a": 1.0}
        targets = {"a": 0.0}
        weights = {"a": 1.0}
        score, br = compute_score(rates, targets, weights, surplus_alpha=1.0)
        assert br["deficit_l2_pp"] == 0.0
        assert br["surplus_sum_pp"] == pytest.approx(100.0)
        assert score == pytest.approx(100.0)

    def test_empty_inputs_return_zero(self):
        score, br = compute_score({}, {}, {}, surplus_alpha=1.0)
        assert score == 0.0
        assert br["deficit_l2_pp"] == 0.0
        assert br["surplus_sum_pp"] == 0.0

    def test_finite_score(self):
        # Sanity: no NaNs/Infs for plausible inputs
        rates = {"a": 0.55, "b": 0.62, "c": 0.81, "d": 0.99}
        targets = {"a": 0.8, "b": 0.6, "c": 0.8, "d": 0.8}
        weights = {"a": 1.0, "b": 1.0, "c": 1.0, "d": 1.0}
        score, br = compute_score(rates, targets, weights, surplus_alpha=1.0)
        assert math.isfinite(score)
        assert math.isfinite(br["deficit_l2_pp"])
        assert math.isfinite(br["surplus_sum_pp"])


class TestEvalConfig:
    def test_defaults_include_four_baselines_plus_foulplay_off(self):
        cfg = EvalConfig()
        assert cfg.enabled is False
        assert cfg.eval_every_n_updates == 500
        assert cfg.pause_training is True
        assert cfg.surplus_alpha == 1.0
        # All five expected canonical names present
        assert set(cfg.opponents.keys()) == {
            "simple_heuristic_baseline",
            "max_damage",
            "vgc_bench",
            "bc_player",
            "foul_play",
        }
        # FoulPlay shipped off
        assert cfg.opponents["foul_play"].weight == 0.0
        # Other four shipped on with weight 1.0
        for k in ["simple_heuristic_baseline", "max_damage", "vgc_bench", "bc_player"]:
            assert cfg.opponents[k].weight == 1.0
        # Targets match the design spec
        assert cfg.opponents["simple_heuristic_baseline"].target == 0.80
        assert cfg.opponents["max_damage"].target == 0.80
        assert cfg.opponents["vgc_bench"].target == 0.60
        assert cfg.opponents["bc_player"].target == 0.80

    def test_opponent_spec_defaults(self):
        spec = OpponentEvalSpec(target=0.5, weight=1.0, n_battles=100)
        assert spec.target == 0.5
        assert spec.weight == 1.0
        assert spec.n_battles == 100

    def test_eval_section_loads_from_yaml(self, tmp_path):
        # Minimal RNaDConfig YAML override of the eval section
        data = {
            "eval": {
                "enabled": True,
                "eval_every_n_updates": 1234,
                "surplus_alpha": 0.5,
                "opponents": {
                    "vgc_bench": {"target": 0.65, "weight": 2.0, "n_battles": 80},
                },
            },
        }
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.safe_dump(data))
        cfg = RNaDConfig.from_yaml(str(p))
        assert cfg.eval.enabled is True
        assert cfg.eval.eval_every_n_updates == 1234
        assert cfg.eval.surplus_alpha == 0.5
        # YAML-supplied vgc_bench overrides defaults
        assert cfg.eval.opponents["vgc_bench"].target == 0.65
        assert cfg.eval.opponents["vgc_bench"].weight == 2.0
        assert cfg.eval.opponents["vgc_bench"].n_battles == 80
        # Other opponents fall back to defaults
        assert cfg.eval.opponents["foul_play"].weight == 0.0
        assert cfg.eval.opponents["simple_heuristic_baseline"].weight == 1.0


class TestResultDataclasses:
    def test_bucket_run_result_instantiation(self):
        ev = EvalResult(
            label="test",
            player1_wins=80,
            player2_wins=20,
            ties=0,
            battles_played=100,
        )
        b = BucketRunResult(
            win_rate=0.80,
            n_battles=100,
            per_format={"gen9vgc2023regc": ev},
            wall_time_s=12.5,
        )
        assert b.win_rate == 0.80
        assert b.n_battles == 100
        assert "gen9vgc2023regc" in b.per_format
        assert b.wall_time_s == 12.5

    def test_multi_bucket_eval_result_instantiation(self):
        ev = EvalResult(
            label="test",
            player1_wins=80,
            player2_wins=20,
            ties=0,
            battles_played=100,
        )
        bucket = BucketRunResult(
            win_rate=0.80,
            n_battles=100,
            per_format={"gen9vgc2023regc": ev},
            wall_time_s=1.0,
        )
        r = MultiBucketEvalResult(
            per_bucket={"max_damage": bucket},
            score=-5.0,
            breakdown={"deficit_l2_pp": 5.0, "surplus_sum_pp": 0.0},
            wall_time_s=2.0,
        )
        assert r.score == -5.0
        assert r.per_bucket["max_damage"].win_rate == 0.80


# ============================================================================
# Task 4 — _run_opponent_bucket dispatch
# ============================================================================


class TestOpponentKwargs:
    def test_vgc_bench_kwargs(self):
        cfg = EvalConfig()
        k = _opponent_kwargs("vgc_bench", cfg)
        assert k["vgc_bench_checkpoint_path"] == cfg.vgcbench_checkpoint_path
        assert k["vgc_bench_team_file"] == cfg.vgcbench_team_file
        assert k["vgc_bench_python_executable"] == cfg.vgcbench_python_executable
        assert "foul_play_python_executable" not in k

    def test_foul_play_kwargs(self):
        cfg = EvalConfig()
        k = _opponent_kwargs("foul_play", cfg)
        assert k["foul_play_python_executable"] == cfg.foulplay_python_executable
        assert k["foul_play_search_time_ms"] == cfg.foulplay_search_time_ms
        assert k["foul_play_parallelism"] == cfg.foulplay_parallelism
        assert "vgc_bench_checkpoint_path" not in k

    def test_inprocess_baseline_empty_kwargs(self):
        cfg = EvalConfig()
        assert _opponent_kwargs("max_damage", cfg) == {}
        assert _opponent_kwargs("simple_heuristic_baseline", cfg) == {}
        assert _opponent_kwargs("bc_player", cfg) == {}


class TestSplitBattlesByFormat:
    def test_single_format(self):
        assert _split_battles_by_format(100, {"a": 1.0}) == {"a": 100}

    def test_two_formats_even_weight(self):
        assert _split_battles_by_format(100, {"a": 1.0, "b": 1.0}) == {"a": 50, "b": 50}

    def test_two_formats_unequal_weight_sums_to_total(self):
        result = _split_battles_by_format(100, {"a": 0.6, "b": 0.4})
        assert sum(result.values()) == 100
        assert result["a"] == 60
        assert result["b"] == 40

    def test_largest_remainder_preserves_total(self):
        # 7 battles across 3 formats with equal weight: floors=[2,2,2], remainder=1
        result = _split_battles_by_format(7, {"a": 1.0, "b": 1.0, "c": 1.0})
        assert sum(result.values()) == 7
        assert sorted(result.values()) == [2, 2, 3]

    def test_zero_total(self):
        assert _split_battles_by_format(0, {"a": 1.0}) == {"a": 0}

    def test_empty_formats(self):
        assert _split_battles_by_format(100, {}) == {}


def _make_curriculum(formats: Dict[str, float]):
    cur = MagicMock(spec=CurriculumConfig)
    cur.battle_formats = formats
    return cur


class TestRunOpponentBucket:
    @patch("elitefurretai.rl.analyze.baseline_eval._resolve_opponent_team_text")
    @patch("elitefurretai.rl.analyze.baseline_eval._resolve_agent_team_text")
    @patch("elitefurretai.rl.analyze.baseline_eval.run_eval_parallel")
    @patch("elitefurretai.rl.analyze.baseline_eval.parse_player_specification")
    def test_vgc_bench_receives_vgcbench_kwargs(
        self, mock_parse, mock_run, mock_agent_team, mock_opp_team
    ):
        mock_parse.return_value = MagicMock()
        mock_agent_team.return_value = "AGENT_TEAM"
        mock_opp_team.return_value = ""
        mock_run.return_value = EvalResult(
            label="vgc_bench", player1_wins=60, player2_wins=40, ties=0, battles_played=100
        )
        eval_cfg = EvalConfig()
        spec = eval_cfg.opponents["vgc_bench"]
        result = _run_opponent_bucket(
            opp_name="vgc_bench",
            spec=spec,
            eval_cfg=eval_cfg,
            curriculum=_make_curriculum({"gen9vgc2023regc": 1.0}),
            checkpoint_path="/tmp/model.pt",
            server_urls=["localhost:8000"],
            device="cpu",
            run_tag="abcd",
        )
        # parse_player_specification called twice: once for model, once for vgc_bench opp
        assert mock_parse.call_count == 2
        vgc_call_kwargs = mock_parse.call_args_list[1].kwargs
        assert (
            vgc_call_kwargs["vgc_bench_checkpoint_path"]
            == eval_cfg.vgcbench_checkpoint_path
        )
        assert "foul_play_python_executable" not in vgc_call_kwargs
        assert result.win_rate == pytest.approx(0.60)

    @patch("elitefurretai.rl.analyze.baseline_eval._foulplay_team_pool_for_fmt")
    @patch("elitefurretai.rl.analyze.baseline_eval._resolve_opponent_team_text")
    @patch("elitefurretai.rl.analyze.baseline_eval._resolve_agent_team_text")
    @patch("elitefurretai.rl.analyze.baseline_eval.run_eval_parallel")
    @patch("elitefurretai.rl.analyze.baseline_eval.parse_player_specification")
    def test_foul_play_receives_foulplay_kwargs(
        self, mock_parse, mock_run, mock_agent_team, mock_opp_team, mock_fp_pool
    ):
        mock_parse.return_value = MagicMock()
        mock_agent_team.return_value = "AGENT_TEAM"
        mock_opp_team.return_value = ""
        mock_fp_pool.return_value = "/data/teams/x"
        mock_run.return_value = EvalResult(
            label="foul_play", player1_wins=20, player2_wins=20, ties=0, battles_played=40
        )
        eval_cfg = EvalConfig()
        spec = eval_cfg.opponents["foul_play"]
        _run_opponent_bucket(
            opp_name="foul_play",
            spec=spec,
            eval_cfg=eval_cfg,
            curriculum=_make_curriculum({"gen9vgc2023regc": 1.0}),
            checkpoint_path="/tmp/model.pt",
            server_urls=["localhost:8000"],
            device="cpu",
            run_tag="abcd",
        )
        fp_call_kwargs = mock_parse.call_args_list[1].kwargs
        assert (
            fp_call_kwargs["foul_play_python_executable"]
            == eval_cfg.foulplay_python_executable
        )
        assert fp_call_kwargs["foul_play_team_pool_path"] == "/data/teams/x"
        assert "vgc_bench_checkpoint_path" not in fp_call_kwargs

    @patch("elitefurretai.rl.analyze.baseline_eval._resolve_opponent_team_text")
    @patch("elitefurretai.rl.analyze.baseline_eval._resolve_agent_team_text")
    @patch("elitefurretai.rl.analyze.baseline_eval.run_eval_parallel")
    @patch("elitefurretai.rl.analyze.baseline_eval.parse_player_specification")
    def test_inprocess_baseline_no_external_kwargs(
        self, mock_parse, mock_run, mock_agent_team, mock_opp_team
    ):
        mock_parse.return_value = MagicMock()
        mock_agent_team.return_value = "AGENT_TEAM"
        mock_opp_team.return_value = "OPP_TEAM"
        mock_run.return_value = EvalResult(
            label="max_damage",
            player1_wins=120,
            player2_wins=30,
            ties=0,
            battles_played=150,
        )
        eval_cfg = EvalConfig()
        spec = eval_cfg.opponents["max_damage"]
        _run_opponent_bucket(
            opp_name="max_damage",
            spec=spec,
            eval_cfg=eval_cfg,
            curriculum=_make_curriculum({"gen9vgc2023regc": 1.0}),
            checkpoint_path="/tmp/model.pt",
            server_urls=["localhost:8000"],
            device="cpu",
            run_tag="abcd",
        )
        md_call_kwargs = mock_parse.call_args_list[1].kwargs
        assert "vgc_bench_checkpoint_path" not in md_call_kwargs
        assert "foul_play_python_executable" not in md_call_kwargs

    @patch("elitefurretai.rl.analyze.baseline_eval._resolve_opponent_team_text")
    @patch("elitefurretai.rl.analyze.baseline_eval._resolve_agent_team_text")
    @patch("elitefurretai.rl.analyze.baseline_eval.run_eval_parallel")
    @patch("elitefurretai.rl.analyze.baseline_eval.parse_player_specification")
    def test_two_formats_aggregates_with_format_weights(
        self, mock_parse, mock_run, mock_agent_team, mock_opp_team
    ):
        mock_parse.return_value = MagicMock()
        mock_agent_team.return_value = "AGENT_TEAM"
        mock_opp_team.return_value = "OPP_TEAM"
        # Format A: 90% win rate (54/60) ; Format B: 50% win rate (20/40)
        # Weighted by 0.6/0.4 → aggregate = (0.6*0.9 + 0.4*0.5)/1.0 = 0.74
        mock_run.side_effect = [
            EvalResult(
                label="md", player1_wins=54, player2_wins=6, ties=0, battles_played=60
            ),
            EvalResult(
                label="md", player1_wins=20, player2_wins=20, ties=0, battles_played=40
            ),
        ]
        eval_cfg = EvalConfig()
        spec = OpponentEvalSpec(target=0.5, weight=1.0, n_battles=100)
        result = _run_opponent_bucket(
            opp_name="max_damage",
            spec=spec,
            eval_cfg=eval_cfg,
            curriculum=_make_curriculum({"gen9vgc2023regc": 0.6, "gen9vgc2024regg": 0.4}),
            checkpoint_path="/tmp/model.pt",
            server_urls=["localhost:8000"],
            device="cpu",
            run_tag="abcd",
        )
        assert result.win_rate == pytest.approx(0.74)
        assert result.n_battles == 100
        assert set(result.per_format.keys()) == {"gen9vgc2023regc", "gen9vgc2024regg"}


# ============================================================================
# Task 5 — baseline_eval.run driver
# ============================================================================


class TestRunDriver:
    @patch("elitefurretai.rl.analyze.baseline_eval._run_opponent_bucket")
    def test_weight_zero_bucket_is_skipped(self, mock_bucket):
        eval_cfg = EvalConfig(enabled=True)  # foul_play weight=0.0 by default
        mock_bucket.return_value = BucketRunResult(
            win_rate=0.8, n_battles=100, per_format={}, wall_time_s=1.0
        )
        result = run(
            eval_cfg=eval_cfg,
            curriculum=_make_curriculum({"gen9vgc2023regc": 1.0}),
            checkpoint_path="/tmp/m.pt",
            server_urls=["localhost:8000"],
            device="cpu",
            run_tag="abcd",
        )
        # _run_opponent_bucket called exactly 4 times — foul_play skipped
        assert mock_bucket.call_count == 4
        called_opponents = {c.kwargs["opp_name"] for c in mock_bucket.call_args_list}
        assert "foul_play" not in called_opponents
        assert "foul_play" not in result.per_bucket
        assert isinstance(result.score, float)

    @patch("elitefurretai.rl.analyze.baseline_eval._run_opponent_bucket")
    def test_score_uses_active_opponents_only(self, mock_bucket):
        eval_cfg = EvalConfig(enabled=True)
        # All four active opponents return 0.80 win_rate; only vgc_bench has target 0.60
        # All four exceed their target (others 0.80 vs 0.80 = on target)
        mock_bucket.return_value = BucketRunResult(
            win_rate=0.80, n_battles=150, per_format={}, wall_time_s=1.0
        )
        result = run(
            eval_cfg=eval_cfg,
            curriculum=_make_curriculum({"gen9vgc2023regc": 1.0}),
            checkpoint_path="/tmp/m.pt",
            server_urls=["localhost:8000"],
            device="cpu",
            run_tag="abcd",
        )
        # SHP/MD/BC: 0.80 == 0.80 target → 0pp deficit, 0pp surplus
        # vgc_bench: 0.80 vs 0.60 target → 20pp surplus, 0pp deficit
        # score = surplus_alpha(=1.0) * 20pp = 20.0
        assert result.score == pytest.approx(20.0)
        assert result.breakdown["surplus_sum_pp"] == pytest.approx(20.0)
        assert result.breakdown["deficit_l2_pp"] == 0.0


# ============================================================================
# Task 6 — build_eval_log_payload
# ============================================================================


def _mk_result(win_rates_per_fmt: Dict[str, Dict[str, float]]) -> MultiBucketEvalResult:
    """Helper: per_bucket from a {opp: {fmt: win_rate}} map."""
    per_bucket: Dict[str, BucketRunResult] = {}
    for opp, fmts in win_rates_per_fmt.items():
        per_format = {}
        n_total = 0
        for fmt, wr in fmts.items():
            n = 100
            per_format[fmt] = EvalResult(
                label=opp,
                player1_wins=int(wr * n),
                player2_wins=n - int(wr * n),
                ties=0,
                battles_played=n,
            )
            n_total += n
        per_bucket[opp] = BucketRunResult(
            win_rate=sum(fmts.values()) / len(fmts),
            n_battles=n_total,
            per_format=per_format,
            wall_time_s=1.0,
        )
    return MultiBucketEvalResult(
        per_bucket=per_bucket,
        score=-3.0,
        breakdown={"deficit_l2_pp": 5.0, "surplus_sum_pp": 2.0},
        wall_time_s=10.0,
    )


class TestBuildEvalLogPayload:
    def test_top_level_keys(self):
        cfg = EvalConfig(enabled=True)
        r = _mk_result({"max_damage": {"gen9vgc2023regc": 0.8}})
        p = build_eval_log_payload(r, update_step=200, eval_cfg=cfg)
        assert p["eval/score"] == -3.0
        assert p["eval/deficit_l2_pp"] == 5.0
        assert p["eval/surplus_sum_pp"] == 2.0
        assert p["eval/wall_time_s"] == 10.0
        assert p["eval/update_step"] == 200

    def test_per_opponent_win_rate(self):
        cfg = EvalConfig(enabled=True)
        r = _mk_result({"max_damage": {"gen9vgc2023regc": 0.8}})
        p = build_eval_log_payload(r, update_step=0, eval_cfg=cfg)
        assert p["eval/max_damage/win_rate"] == pytest.approx(0.8)

    def test_per_format_opponent_weighted_mean(self):
        cfg = EvalConfig(enabled=True)
        cfg.opponents["max_damage"].weight = 2.0
        cfg.opponents["bc_player"].weight = 1.0
        r = _mk_result(
            {
                "max_damage": {"gen9vgc2023regc": 0.9},
                "bc_player": {"gen9vgc2023regc": 0.6},
            }
        )
        p = build_eval_log_payload(r, update_step=0, eval_cfg=cfg)
        # (2 * 0.9 + 1 * 0.6) / 3 = 0.8
        assert p["eval/gen9vgc2023regc/win_rate"] == pytest.approx(0.8)

    def test_no_per_opponent_per_format_keys(self):
        cfg = EvalConfig(enabled=True)
        r = _mk_result({"max_damage": {"gen9vgc2023regc": 0.8, "gen9vgc2024regg": 0.7}})
        p = build_eval_log_payload(r, update_step=0, eval_cfg=cfg)
        assert "eval/max_damage/gen9vgc2023regc/win_rate" not in p
        assert "eval/max_damage/gen9vgc2024regg/win_rate" not in p


# ============================================================================
# Task 9 — standalone CLI output serialization
# ============================================================================


class TestStandaloneCLIOutput:
    def test_serialize_result_to_dict(self):
        import json

        from elitefurretai.rl.analyze.baseline_eval import _serialize_result_to_dict

        ev = EvalResult(
            label="md",
            player1_wins=80,
            player2_wins=20,
            ties=0,
            battles_played=100,
        )
        bucket = BucketRunResult(
            win_rate=0.80,
            n_battles=100,
            per_format={"gen9vgc2023regc": ev},
            wall_time_s=1.0,
        )
        result = MultiBucketEvalResult(
            per_bucket={"max_damage": bucket},
            score=-5.0,
            breakdown={"deficit_l2_pp": 5.0, "surplus_sum_pp": 0.0},
            wall_time_s=2.0,
        )
        d = _serialize_result_to_dict(
            result=result,
            checkpoint_path="/tmp/m.pt",
            config_path="/tmp/cfg.yaml",
            run_tag="abcd",
        )
        assert d["checkpoint_path"] == "/tmp/m.pt"
        assert d["config_path"] == "/tmp/cfg.yaml"
        assert d["run_tag"] == "abcd"
        assert "timestamp_utc" in d
        assert d["score"] == -5.0
        assert d["breakdown"]["deficit_l2_pp"] == 5.0
        assert d["per_opponent"]["max_damage"]["win_rate"] == 0.80
        assert d["per_opponent"]["max_damage"]["n_battles"] == 100
        assert (
            d["per_opponent"]["max_damage"]["per_format"]["gen9vgc2023regc"]["win_rate"]
            == 0.80
        )
        # Must be JSON-serializable
        json.dumps(d)
