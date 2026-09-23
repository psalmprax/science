# Databricks notebook source
"""Tests for the validation-architecture layers (items 1-6).

Dimensional layer, ablation harness + fit-cost ledger, grouped-conformal
intervals, hierarchical partial pooling, and grammar expressiveness audit.
All cases are synthetic and deterministic; no network, no fixtures.
"""
from __future__ import annotations

import numpy as np
import pytest

from apedp.ablation import RungEvaluation, run_ablation_ladder, selection_report
from apedp.conformal import conformalize, coverage_report, grouped_conformal
from apedp.dimensional_layer import admit_with_units, check_units, filter_candidates
from apedp.expressiveness import audit_grammar_fit
from apedp.hierarchical import partial_pool_linear, prior_predictive_interval


# ---------------------------------------------------------------- dimensional
def test_dim_target_match_passes():
    verdict = check_units("sqrt(dx**2 + dy**2 + dz**2)", {"dx": "meter", "dy": "meter", "dz": "meter"}, "meter")
    assert verdict.status == "PASS"


def test_dim_internal_mismatch_fails():
    verdict = check_units("dx + T", {"dx": "meter", "T": "second"}, "meter")
    assert verdict.status == "FAIL"


def test_dim_target_mismatch_fails_despite_internal_consistency():
    # L**2 is self-consistent but wrong for a length target: the key gap closed.
    verdict = check_units("dx**2 + dy**2", {"dx": "meter", "dy": "meter"}, "meter")
    assert verdict.status == "FAIL"
    assert "target" in (verdict.reason or "")


def test_dim_transcendental_of_dimensioned_fails():
    verdict = check_units("exp(T)", {"T": "kelvin"}, "dimensionless")
    assert verdict.status == "FAIL"


def test_dim_transcendental_hidden_among_coefficients_fails():
    # The hole found in the engine rematch: sin(T) slipped through because
    # neighboring fit symbols were undeclared. No coefficient repairs this.
    verdict = check_units("sin(T) - p0 + p1*(T - p2)/sqrt(T)", {"T": "kelvin"}, None)
    assert verdict.status == "FAIL"
    assert "no coefficient" in (verdict.reason or "")


def test_dim_transcendental_with_free_scale_is_unevaluated():
    # p may carry inverse units: benefit of the doubt preserved.
    verdict = check_units("sin(p*T)", {"T": "kelvin"}, None)
    assert verdict.status == "UNEVALUATED"


def test_dim_missing_unit_is_unevaluated():
    verdict = check_units("a*x + b", {"x": "meter"}, "meter")
    assert verdict.status == "UNEVALUATED"
    assert verdict.all_symbols_declared is False


def test_dim_filter_enforces_before_fitting():
    result = filter_candidates(
        ["sqrt(dx**2)", "dx**2", "sqrt(T)", "dx", "cbrt(dx**2+dy**2+dz**2)"],
        {"dx": "meter", "dy": "meter", "dz": "meter", "T": "kelvin"},
        "meter",
    )
    assert "sqrt(dx**2)" in result.admissible and "dx" in result.admissible
    rejected = {expr for expr, _ in result.rejected}
    assert rejected == {"dx**2", "sqrt(T)", "cbrt(dx**2+dy**2+dz**2)"}


def test_dim_admit_with_units_refuses_invalid():
    decision = admit_with_units(
        "sqrt(T)",
        variable_units={"T": "kelvin"},
        target_unit="kelvin",
        repair_status="ACCEPTED",
    )
    assert decision.status.value == "REFUSED"
    assert any("imensional" in reason for reason in decision.reasons)
    assert decision.evidence["unit_verdict"]["status"] == "FAIL"


def test_dim_admit_with_units_persists_valid():
    decision = admit_with_units(
        "a*T + b",
        variable_units={"T": "kelvin", "a": "dimensionless", "b": "kelvin"},
        target_unit="kelvin",
        repair_status="ACCEPTED",
    )
    assert decision.status.value == "PERSISTED"


# ------------------------------------------------------------------ ablation
def test_ablation_ladder_measures_layer_value():
    rng = np.random.default_rng(0)
    x = np.linspace(0.5, 3.0, 40)
    y = 2.0 * x**2 + rng.normal(0, 0.05, len(x))
    xtr, xte, ytr, yte = x[:28], x[28:], y[:28], y[28:]

    def poly_errors(degree):
        coef = np.polyfit(xtr, ytr, degree)
        train = float(np.sqrt(np.mean((np.polyval(coef, xtr) - ytr) ** 2)))
        test = float(np.sqrt(np.mean((np.polyval(coef, xte) - yte) ** 2)))
        return train, test

    measured = {"poly8": poly_errors(8), "quadratic": poly_errors(2)}
    assert measured["poly8"][0] < measured["quadratic"][0]  # overfit wins train
    assert measured["poly8"][1] > measured["quadratic"][1]  # loses holdout

    def evaluate(expression):
        train_rmse, test_rmse = measured[expression]
        return RungEvaluation(expression, train_rmse, test_rmse)

    out = run_ablation_ladder(
        [
            ("raw", ("search",), ["poly8", "quadratic"], "train"),
            ("+dimensions", ("search", "dimensions"), ["poly8", "quadratic"], "train"),
            ("+holdout", ("search", "dimensions", "holdout"), ["poly8", "quadratic"], "test"),
        ],
        evaluate,
    )
    assert out["selections"] == ["poly8", "poly8", "quadratic"]
    assert out["first_selection_change"] == "+holdout"
    assert out["layers_measured"] == ["dimensions", "holdout", "search"]


def test_selection_report_prices_constraints():
    ledger = selection_report(
        [
            RungEvaluation("soup", 0.01, 0.05, ("fit",), ("dimensions", "holdout"), True),
            RungEvaluation("linear", 0.5, 0.6, ("fit", "holdout"), ("dimensions",)),
            RungEvaluation("true", 0.1, 0.2, ("fit", "dimensions", "holdout"), ()),
        ]
    )
    assert ledger["best_unconstrained"] == "soup"
    assert ledger["selected"] == "true"
    assert ledger["admissibility_price"] == pytest.approx(0.2 - 0.05)


# ----------------------------------------------------------------- conformal
def test_conformal_coverage_sane_and_grouped():
    rng = np.random.default_rng(1)
    x = np.linspace(0, 10, 120)
    y = 3.0 * x + rng.normal(0, 1.0, len(x))
    groups = np.repeat(["a", "b", "c"], 40)
    A = np.column_stack([x, np.ones(len(x))])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    calib = np.arange(120)[::2]
    interval = grouped_conformal(
        pred[calib], y[calib], alpha=0.1,
        calib_group_ids=["a", "b", "c"], group_of_calib=groups[calib],
    )
    test = np.arange(120)[1::2]
    lower, upper = interval.band((np.column_stack([x[test], np.ones(len(test))]) @ coef))
    report = coverage_report(y[test], lower, upper, groups[test])
    assert report["coverage"] >= 0.75
    assert set(report["per_group"]) == {"a", "b", "c"}
    assert interval.halfwidth > 0


def test_conformal_rejects_bad_inputs():
    with pytest.raises(ValueError):
        conformalize([1.0], [1.0], alpha=1.5)
    with pytest.raises(ValueError):
        grouped_conformal(np.array([1.0]), np.array([1.0, 2.0]), alpha=0.1)
    with pytest.raises(ValueError):
        grouped_conformal(
            np.array([1.0, 2.0]), np.array([1.0, 2.0]), alpha=0.1,
            calib_group_ids=["missing"], group_of_calib=["a", "a"],
        )


# -------------------------------------------------------------- hierarchical
def _group_data():
    rng = np.random.default_rng(2)
    designs, targets = [], []
    for slope in (2.0, 2.4, 1.6):
        x = np.linspace(0, 5, 25)
        designs.append(np.column_stack([x, np.ones_like(x)]))
        targets.append(slope * x + 1.0 + rng.normal(0, 0.1, len(x)))
    return designs, targets


def test_hierarchical_partial_pooling():
    designs, targets = _group_data()
    fit = partial_pool_linear(designs, targets, group_ids=["g1", "g2", "g3"])
    assert fit.pooled_params[0] == pytest.approx(2.0, abs=0.15)
    for group in ("g1", "g2", "g3"):
        assert all(0.0 <= w <= 1.0 for w in fit.shrinkage_weights[group])
        # Shrunk slopes move toward the pooled mean relative to raw OLS.
        assert abs(fit.group_params[group][0] - fit.pooled_params[0]) <= abs(
            {"g1": 2.0, "g2": 2.4, "g3": 1.6}[group] - fit.pooled_params[0]
        ) + 0.05
    assert all(v >= 0.0 for v in fit.between_variance)


def test_hierarchical_prior_predictive_covers_new_group():
    designs, targets = _group_data()
    fit = partial_pool_linear(designs[:2], targets[:2], group_ids=["g1", "g2"])
    x_new = np.array([[2.5, 1.0]])
    lower, upper = prior_predictive_interval(fit, x_new)
    truth = 2.0 * 2.5 + 1.0
    assert lower[0] <= truth <= upper[0]


def test_hierarchical_rejects_degenerate():
    with pytest.raises(ValueError):
        partial_pool_linear([np.ones((5, 2))], [np.ones(5)])


# ------------------------------------------------------------ expressiveness
def test_expressiveness_flags_insufficient_grammar():
    rng = np.random.default_rng(3)
    x = np.linspace(0.2, 12.0, 80)
    y = np.sin(x) + rng.normal(0, 0.02, len(x))
    preds = {}
    for name, deg in (("linear", 1), ("quadratic", 2)):
        coef = np.polyfit(x, y, deg)
        preds[name] = (y, np.polyval(coef, x))
    verdict = audit_grammar_fit(preds)
    assert verdict.status == "STRUCTURE_OUTSIDE_GRAMMAR"
    assert verdict.residual_structure is not None and verdict.residual_structure > 0.3


def test_expressiveness_sufficient_when_fit():
    rng = np.random.default_rng(4)
    x = np.linspace(0.5, 3.0, 60)
    y = 4.0 * x + rng.normal(0, 0.05, len(x))
    preds = {}
    for name, deg in (("linear", 1), ("quadratic", 2)):
        coef = np.polyfit(x, y, deg)
        preds[name] = (y, np.polyval(coef, x))
    verdict = audit_grammar_fit(preds)
    assert verdict.status == "SUFFICIENT"
    assert verdict.best_form in ("linear", "quadratic")


def test_expressiveness_inconclusive_on_empty():
    verdict = audit_grammar_fit({})
    assert verdict.status == "INCONCLUSIVE"
