# -*- coding: utf-8 -*-
"""Unit tests for baseline_eval (compute_score, dispatch, payload, cleanup)."""

import math

import pytest

from elitefurretai.rl.analyze.baseline_eval import compute_score


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
