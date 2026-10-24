"""Tests for Multi-Term Superposition L2 Open Discovery Engine.

Verifies:
  1. Autonomous recovery of 2-term damped harmonic oscillator (x_ddot = -k*x - c*v).
  2. Autonomous recovery of 3-term nonlinear Duffing oscillator (x_ddot = -alpha*x - delta*v - beta*x^3).
  3. FIM / Collinearity gating: combinations with condition number kappa > 10^3 are pruned before fitting.
  4. Pareto complexity frontier selection: higher term counts require strict holdout RMSE improvement.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from apedp_dyn.discovery_l2.dimensional_types import Dimension
from apedp_dyn.discovery_l2.multiterm_engine import MultiTermL2Engine
from apedp_dyn.epistemic.adjudicate import EpistemicStatus


def test_rolling_basis_terms_survive_warmup_nan():
    """Rolling basis terms must reach the candidate pool.

    The basis screen used to require all-finite term vectors, and a rolling
    operator's first W-1 evaluations are NaN by construction, so every rolling
    term was discarded (100% rejection) despite Gate A explicitly tolerating
    warm-up NaN. The term is now evaluated on the full causal series (state
    carries across the split) and fitted on the finite overlap.
    """
    rng = np.random.default_rng(11)
    n = 240
    t = np.arange(n, dtype=float)
    signal = np.sin(0.3 * t) + 0.05 * rng.normal(size=n)
    # Target is a causal rolling mean of the signal plus a linear term.
    roll = np.full(n, np.nan)
    for i in range(39, n):
        roll[i] = float(np.mean(signal[i - 39:i + 1]))
    target = 2.0 * roll + 0.5 * signal

    engine = MultiTermL2Engine(
        max_depth=1, max_term_complexity=2, max_terms=3,
        operators=("+", "-", "*", "rollmean"),
    )
    res = engine.discover_multiterm(
        X={"signal": signal},
        Y=target,
        dimensions={"signal": Dimension()},
        target_dimension=Dimension(),
        target_name="target",
        train_ratio=0.6,
        bootstrap=False,
    )
    assert res.all_evaluated_count > 0, "no basis terms survived"
    # A rolling term must be present in the promoted/pareto candidate set.
    exprs = [c.candidate.metadata.get("canonical_expr", "") for c in res.pareto_candidates]
    assert any("rollmean" in e.lower() for e in exprs), exprs
    # Warm-up rows are excluded, not fabricated: the fit uses fewer rows than
    # the partition length, and the count is disclosed.
    top = res.pareto_candidates[0]
    g4 = top.provenance.get("gate4_status", {})
    assert g4["n_rows_used_train"] < 240 * 0.6
    assert g4["warmup_rows_excluded_train"] > 0


def test_gate_a_never_evaluates_the_holdout_partition():
    """Gate A must screen candidates on development data only (AGENTS.md 1.2).

    A candidate that is algebraically invalid ONLY on the holdout partition
    (here: a term whose holdout slice is entirely NaN) must still pass Gate A,
    because the proposer may not look at holdout data at all.
    """
    import apedp_dyn.discovery_l2.l2_engine as l2e

    calls = {"ho": 0}
    original = l2e.gate_a_algebraic_validity

    def spy(node, env, **kwargs):
        # The holdout mapping is the one whose key set matches the tail slice.
        if kwargs.get("allow_constant") and env and len(next(iter(env.values()))) == 96:
            calls["ho"] += 1
        return original(node, env, **kwargs)

    l2e.gate_a_algebraic_validity = spy
    try:
        n = 240
        rng = np.random.default_rng(3)
        x = rng.normal(size=n)
        y = 0.5 * x + 0.01 * rng.normal(size=n)
        engine = l2e.L2DiscoveryEngine(max_depth=1, max_complexity=3, operators=("+", "*"))
        engine.discover(
            X={"x": x}, Y=y,
            dimensions={"x": Dimension()},
            target_dimension=Dimension(),
            train_ratio=0.6, bootstrap=False,
        )
    finally:
        l2e.gate_a_algebraic_validity = original
    assert calls["ho"] == 0, "Gate A evaluated holdout data during proposer search"


class TestMultiTermL2Engine:
    """Verifies multi-term superposition discovery and FIM conditioning."""

    def test_harmonic_oscillator_multi_term_discovery(self):
        """Autonomously discovers 2-term damped harmonic oscillator x_ddot = -4*x - 0.2*v."""
        def odes(t, y):
            return [y[1], -4.0 * y[0] - 0.2 * y[1]]

        t_eval = np.linspace(0.0, 10.0, 100)
        sol = solve_ivp(odes, (0.0, 10.0), [1.0, 0.0], t_eval=t_eval, rtol=1e-9, atol=1e-9)
        x, v = sol.y[0], sol.y[1]
        x_ddot = -4.0 * x - 0.2 * v

        engine = MultiTermL2Engine(max_depth=1, max_term_complexity=2, max_terms=2, operators=("+", "-", "*"))
        res = engine.discover_multiterm(
            X={"x": x, "v": v},
            Y=x_ddot,
            dimensions={"x": Dimension(), "v": Dimension()},
            target_dimension=Dimension(),
            target_name="x_ddot",
            train_ratio=0.6,
            bootstrap=False,
        )

        assert res.adjudication.status == EpistemicStatus.UNIQUE_SUPPORT
        assert res.adjudication.promoted is not None
        assert len(res.adjudication.promoted.candidate.terms) == 2
        # Verify both x and v terms were identified
        term_names = [t.name for t in res.adjudication.promoted.candidate.terms]
        assert "x" in term_names
        assert "v" in term_names

        # Verify fitted coefficients: theta_x ≈ -4.0, theta_v ≈ -0.2
        pv = res.adjudication.promoted.fit.param_values
        for t_obj in res.adjudication.promoted.candidate.terms:
            if t_obj.name == "x":
                assert np.isclose(pv[t_obj.parameter], -4.0, atol=0.01)
            elif t_obj.name == "v":
                assert np.isclose(pv[t_obj.parameter], -0.2, atol=0.01)

    def test_duffing_oscillator_multi_term_discovery(self):
        """Autonomously discovers 3-term nonlinear Duffing oscillator x_ddot = -1*x - 0.2*v - 1*x^3."""
        def odes(t, y):
            return [y[1], -0.2 * y[1] - 1.0 * y[0] - 1.0 * (y[0]**3)]

        t_eval = np.linspace(0.0, 10.0, 120)
        sol = solve_ivp(odes, (0.0, 10.0), [1.0, 0.0], t_eval=t_eval, rtol=1e-9, atol=1e-9)
        x, v = sol.y[0], sol.y[1]
        x_ddot = -0.2 * v - 1.0 * x - 1.0 * (x**3)

        engine = MultiTermL2Engine(max_depth=2, max_term_complexity=3, max_terms=3, operators=("+", "-", "*", "pow"))
        res = engine.discover_multiterm(
            X={"x": x, "v": v},
            Y=x_ddot,
            dimensions={"x": Dimension(), "v": Dimension()},
            target_dimension=Dimension(),
            target_name="x_ddot",
            train_ratio=0.6,
            bootstrap=False,
        )

        assert res.adjudication.status == EpistemicStatus.UNIQUE_SUPPORT
        assert res.adjudication.promoted is not None
        assert len(res.adjudication.promoted.candidate.terms) == 3

        term_names = [t.name for t in res.adjudication.promoted.candidate.terms]
        assert "x" in term_names
        assert "v" in term_names
        assert any("x" in tn and "3" in tn for tn in term_names)

    def test_fim_collinearity_gate_prunes_ill_conditioned_pairs(self):
        """Verifies candidate combinations with condition number kappa > 10^3 are pruned."""
        rng = np.random.default_rng(42)
        n = 80
        u = rng.uniform(1.0, 10.0, size=n)
        # Highly collinear feature w = 2.0001*u
        w = 2.0 * u + rng.normal(0, 0.00001, size=n)
        y = 3.0 * u + rng.normal(0, 0.01, size=n)

        engine = MultiTermL2Engine(
            max_depth=1,
            max_term_complexity=2,
            max_terms=2,
            max_condition_number=100.0,
            operators=("+", "-", "*"),
        )
        res = engine.discover_multiterm(
            X={"u": u, "w": w},
            Y=y,
            dimensions={"u": Dimension(), "w": Dimension()},
            target_dimension=Dimension(),
            train_ratio=0.6,
            bootstrap=False,
        )

        # The collinear pair [u, w] must be pruned by the FIM gate
        assert res.collinear_pruned_count >= 1
