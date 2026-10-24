"""Regression pins for the scale-free Occam penalty in ``MultiVariableSRE.fitness``.

Background (2026-10-22 trading constant-collapse forensics): the legacy
fitness added an absolute ``1e-4 * complexity`` penalty calibrated for
O(1)-variance targets. On the tutorial's microstructure price-impact data
(``var(dp) ~ 2.5e-3``, true linear signal ``R^2 ~ 1.6%``), the penalty for
two extra nodes (``2e-4``) exceeded the entire MSE gain of the true
``lambda*ofi`` model (``4.0e-5``; measured ratio 4.96x), so the search
returned the sample-mean constant ``-0.0005104...`` regardless of
population size or generation count (verified: identical constant at
generations=60, pop_size=200). The fix multiplies the penalty by
``min(var(y), 1.0)``: bit-identical for var(y) >= 1 targets, signal-dominated
for small-variance ones.

These pins require BOTH directions of that contract:
1. the trading case must discover the ``ofi``-dependent family (no collapse);
2. var(y) >= 1 targets keep the exact legacy fitness (legacy exact-value pins
   and benchmark records stay honest).
"""

import numpy as np
import pytest
import sympy as sp

from apedp import ClosedLoopAssuranceSystem, DimensionalAssuranceEngine, MultiVariableSRE


def _simulate_tutorial_microstructure():
    """Bit-exact copy of the tutorial notebook Step-2 simulation (seed 42)."""
    np.random.seed(42)
    n_steps, dt, base_intensity = 2000, 0.1, 0.5
    arrivals = np.zeros(n_steps)
    curr_intensity = base_intensity
    for t in range(n_steps):
        arrivals[t] = np.random.poisson(curr_intensity * dt)
        curr_intensity = (
            base_intensity + 0.8 * (curr_intensity - base_intensity) + 0.5 * arrivals[t]
        )
    noise = np.random.normal(0, 0.05, n_steps)
    price_changes = 0.02 * (arrivals - np.mean(arrivals)) + noise
    return arrivals - np.mean(arrivals), price_changes


def _build_regressor():
    engine = DimensionalAssuranceEngine()
    engine.register_variable('ofi', 'dimensionless')
    assurance_system = ClosedLoopAssuranceSystem(engine)
    return MultiVariableSRE(
        assurance_system=assurance_system,
        independent_vars=['ofi'],
        target_unit_str='dimensionless',
        variables={'ofi': 'dimensionless'},
        allow_exp=True,
        allow_log=False,
        allow_sqrt=True,
    )


def test_small_variance_target_does_not_collapse_to_constant():
    ofi, dp = _simulate_tutorial_microstructure()
    y_var = float(np.var(dp))
    assert y_var < 1e-2  # the regime that collapsed pre-fix

    regressor = _build_regressor()
    best_expr, best_mse, _ = regressor.run_discovery(
        {'ofi': ofi}, dp, generations=12, pop_size=30
    )

    assert best_expr is not None
    assert sp.Symbol('ofi') in best_expr.free_symbols, (
        f"constant-collapse regression: engine returned the sample-mean "
        f"constant {best_expr!r} instead of an ofi-dependent impact model"
    )
    # The engine must strictly dominate the pre-fix winner (the constant
    # sample mean, MSE == var(y)) and stay within 1% of the free OLS
    # baseline. The 1% slack is NOT a weakened assertion: the engine's
    # documented outer-fit calibration snaps coefficients to exact
    # fractions when close (see optimize_parameters), so a tiny deviation
    # from free-OLS optimality is legacy contract, while the pre-fix
    # failure mode (constant winning by ~5%) is what this pin forbids.
    ols_mse = float(
        np.mean(
            (dp - np.polyval(np.polyfit(ofi, dp, 1), ofi)) ** 2
        )
    )
    assert best_mse <= ols_mse * (1.0 + 0.01)
    # and strictly better than the constant that pre-fix won
    assert best_mse < y_var


def test_penalty_scale_is_one_for_var_ge_one_targets():
    # Legacy preservation: min(var, 1.0) means var(y) >= 1 targets keep
    # the exact legacy `1e-4*complexity` penalty (scale == 1.0), so every
    # pre-existing exact fitness value / benchmark record stays honest.
    regressor = _build_regressor()
    ofi = np.linspace(-3.0, 3.0, 401)
    y_big = 3.0 * ofi                       # var ~ 27
    y_var_one = ofi                          # var ~ 3
    y_mid = 0.5 * ofi                        # var ~ 0.75 (below the bound)
    assert float(np.var(y_big)) >= 1.0 and float(np.var(y_var_one)) >= 1.0
    assert float(np.var(y_mid)) < 1.0

    assert regressor._fitness_penalty_scale(y_big) == 1.0
    assert regressor._fitness_penalty_scale(y_var_one) == 1.0
    mid_scale = regressor._fitness_penalty_scale(y_mid)
    assert mid_scale == pytest.approx(float(np.var(y_mid)))
    assert 0.0 < mid_scale < 1.0


def test_penalty_scale_shrinks_with_small_variance():
    regressor = _build_regressor()
    ofi, dp = _simulate_tutorial_microstructure()
    scale = regressor._fitness_penalty_scale(dp)
    assert 0.0 < scale == float(np.var(dp))  # scale == var(y) for var(y) < 1
    assert scale < 1.0

    unit_var = np.array([-0.1, 0.1, -0.3, 0.3, -0.5, 0.5])  # mean 0, var 0.7/6
    expected = float(np.var(unit_var))
    assert 0.0 < expected < 1.0
    assert regressor._fitness_penalty_scale(unit_var) == pytest.approx(expected)


def test_degenerate_constant_target_keeps_legacy_absolute_penalty():
    regressor = _build_regressor()
    scale = regressor._fitness_penalty_scale(np.zeros(10))
    assert scale == 1.0


def test_big_budget_no_collapse_invariant():
    """The pre-fix failure was budget-independent; the pin must be too."""
    ofi, dp = _simulate_tutorial_microstructure()
    regressor = _build_regressor()
    best_expr, best_mse, _ = regressor.run_discovery(
        {'ofi': ofi}, dp, generations=60, pop_size=200
    )
    assert sp.Symbol('ofi') in best_expr.free_symbols
    assert best_mse < float(np.var(dp))
