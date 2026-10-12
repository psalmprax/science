"""Tests for Fluid Dynamics and Dynamical Systems Discovery Sweep across Universes and Horizons.

Validates:
  - Feature construction and matrix conditioning across 15, 30, 60, 90 assets
  - Evaluation across 1m, 5m, 15m, 30m, 45m, 60m horizons
  - The 5 equations: Navier-Stokes, Euler, Autonomous ODE, Forced ODE, 2nd-Order ODE
  - Dual Adjudication: APEDP Gated Arm vs Best-Fit ML Arm
  - Gate 0.5 (CAN-G) risk parity verification
"""
import pytest
from apedp_dyn.sandbox.fluid_ode_dynamical_discovery import (
    CANDIDATE_EQUATION_SPECS,
    TIMEFRAMES_MINUTES,
    UNIVERSE_SIZES,
    run_single_universe_timeframe_experiment,
)


def test_candidate_equation_specs_integrity():
    """Verify that all 5 candidate equations are well-defined and structurally distinct."""
    assert len(CANDIDATE_EQUATION_SPECS) == 5
    expected_keys = {
        "Eq1_NavierStokes",
        "Eq2_Euler",
        "Eq3_ODE_Autonomous",
        "Eq4_ODE_Forced",
        "Eq5_ODE_2ndOrder",
    }
    assert set(CANDIDATE_EQUATION_SPECS.keys()) == expected_keys
    for eq_id, spec in CANDIDATE_EQUATION_SPECS.items():
        assert "formula" in spec
        assert "feature_keys" in spec
        assert len(spec["feature_keys"]) >= 2
        assert "physical_interpretation" in spec


def test_single_cell_evaluation_30_assets_15m():
    """Verify evaluation on 30 assets at 15m duration."""
    res = run_single_universe_timeframe_experiment(
        universe_size=30,
        timeframe_minutes=15,
        month_start_idx=80,
    )
    assert res.universe_size == 30
    assert res.timeframe_minutes == 15
    assert len(res.evaluations) == 5
    assert res.gate_05_cang_passed is True

    # Check each equation evaluation
    for eq_id, ev in res.evaluations.items():
        assert ev.condition_number <= 1000.0
        assert ev.best_fit_verdict == "PROMOTES (Ungated Argmin)"
        assert ev.gated_verdict in (
            "TERMINAL_ABSTAIN",
            "REFUSED_HOLDOUT_GATE_4",
            "PROMOTED_FORWARD_ONLY",
            "PROMOTED_BIDIRECTIONAL",
        )
        assert len(ev.coefficients) == len(ev.feature_keys)


def test_coverage_across_universe_sizes():
    """Verify discovery executes across all 4 universe sizes: 15, 30, 60, 90."""
    for u_size in (15, 30, 60, 90):
        res = run_single_universe_timeframe_experiment(
            universe_size=u_size,
            timeframe_minutes=30,
            month_start_idx=80,
        )
        assert res.universe_size == u_size
        assert len(res.evaluations) == 5
        assert res.winning_eq_id in CANDIDATE_EQUATION_SPECS


def test_coverage_across_horizons():
    """Verify discovery executes across all 6 timeframes: 1m, 5m, 15m, 30m, 45m, 60m."""
    for tf in TIMEFRAMES_MINUTES:
        res = run_single_universe_timeframe_experiment(
            universe_size=15,
            timeframe_minutes=tf,
            month_start_idx=80,
        )
        assert res.timeframe_minutes == tf
        assert len(res.evaluations) == 5
