# -*- coding: utf-8 -*-
"""Unit tests for baseline_eval (compute_score, dispatch, payload, cleanup)."""

import math

import pytest
import yaml

from elitefurretai.rl.analyze.baseline_eval import compute_score
from elitefurretai.rl.config import EvalConfig, OpponentEvalSpec, RNaDConfig


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
