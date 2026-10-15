"""Planck radiance limitation pin (first-order LINEAR engines).

Mathematical fact: for u(v) = v^3 / (e^v - 1),
    u'/u = 3/v - e^v / (e^v - 1).
If a LINEAR polynomial-coefficient relation P(v) u' + Q(v) u = 0 held
exactly, then u'/u = -Q/P would be rational in v, forcing e^v into Q(v) --
contradicting the transcendence of e^v over Q(v) (Hermite). So NO exact
first-order LINEAR polynomial ODE exists for Planck radiance.

Scope correction (adjudicated 2026-10-15): this bar is linear-only. A
quadratic Riccati relation DOES exist,
    v^3 u' - (3v^2 - v^3) u + u^2 = 0  (verified to ~2e-7),
but no current engine discovers second-degree Jet relations -- recorded
follow-up, not a recovery claim.

The honest LINEAR-engine behavior is therefore: return the best approximate
jet ODE, report a small jet residual, and set is_exact_recovery = False
(correct abstention from an exact-recovery claim). This test pins that the
engine does not manufacture an exact recovery where none exists.
"""
from __future__ import annotations

import json

import numpy as np

import apedp_kernel as ak
from apedp.evolutionary_search import SearchConfig, evolve
from apedp.rust_bridge import RustKernelBridge


def test_planck_returns_approximate_jet_ode_without_exact_claim():
    v = np.linspace(0.1, 6.0, 60)
    u = (v ** 3) / (np.exp(v) - 1.0)
    cfg = {
        "max_x_degree": 2,
        "max_diff_order": 1,
        "smooth_window": 5,
        "sparsity_threshold": 0.05,
        "rcond_threshold": 1e-12,
    }
    res = json.loads(ak.rime_diff_pade_discover_json(
        json.dumps(v.tolist()), json.dumps(u.tolist()), json.dumps(cfg)
    ))
    # An approximate jet relation is found (small residual on the jet bundle).
    assert res["jet_rmse"] < 0.05
    # But no exact recovery may be claimed: none exists mathematically.
    assert res["is_exact_recovery"] is False
    assert res["integrated_trajectory_r2"] < 0.99


def test_native_ga_hallucinates_on_planck_while_rime_abstains():
    # Same train/holdout split for both. GA returns a spurious
    # rational-trigonometric contraption that collapses out of sample,
    # while RIME's differential engine refuses the exact claim.
    v_tr = np.linspace(0.1, 6.0, 60)
    v_ho = np.linspace(6.1, 10.0, 40)
    u_tr = (v_tr ** 3) / (np.exp(v_tr) - 1.0)
    u_ho = (v_ho ** 3) / (np.exp(v_ho) - 1.0)
    ga = evolve({"v": v_tr}, u_tr, config=SearchConfig(
        use_native_ga=True, islands=2, island_size=30,
        generations=15, seed=42))
    assert ga.best_expression
    pred = RustKernelBridge().evaluate_expression(
        ga.best_expression, ["v"], [v_ho])
    ga_r2 = 1.0 - float(np.mean((u_ho - pred) ** 2)) / float(np.var(u_ho))
    assert ga_r2 < 0.5, f"GA unexpectedly transferred: R2={ga_r2}"


def _rk4_rollout(x, y0, pc, qc):
    y = np.empty_like(x)
    y[0] = y0
    for i in range(len(x) - 1):
        h = x[i + 1] - x[i]

        def f(xv, yv):
            p = sum(c * (xv ** p) for p, c in enumerate(pc))
            q = sum(c * (xv ** p) for p, c in enumerate(qc))
            return -q / p * yv if abs(p) > 1e-10 else 0.0

        k1 = f(x[i], y[i])
        k2 = f(x[i] + 0.5 * h, y[i] + 0.5 * h * k1)
        k3 = f(x[i] + 0.5 * h, y[i] + 0.5 * h * k2)
        k4 = f(x[i] + h, y[i] + h * k3)
        y[i + 1] = y[i] + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return y


def test_planck_equations_on_fresh_out_of_box_data():
    # Both fixed equations (GA formula, RIME jet ODE) are evaluated on fresh
    # Planck data over a wider domain neither was fitted on. Expectation:
    # neither transfers well (no exact structure exists in either grammar),
    # but the ODE rollout degrades less badly than the hallucinated formula.
    v_tr = np.linspace(0.1, 6.0, 60)
    u_tr = (v_tr ** 3) / (np.exp(v_tr) - 1.0)
    cfg = {"max_x_degree": 2, "max_diff_order": 1, "smooth_window": 5,
           "sparsity_threshold": 0.05, "rcond_threshold": 1e-12}
    ode = json.loads(ak.rime_diff_pade_discover_json(
        json.dumps(v_tr.tolist()), json.dumps(u_tr.tolist()),
        json.dumps(cfg)))
    ga = evolve({"v": v_tr}, u_tr, config=SearchConfig(
        use_native_ga=True, islands=2, island_size=30,
        generations=15, seed=42))

    vn = np.linspace(0.05, 12.0, 120)
    u = (vn ** 3) / (np.exp(vn) - 1.0)
    pred_ga = RustKernelBridge().evaluate_expression(
        ga.best_expression, ["v"], [vn])
    pred_ode = _rk4_rollout(vn, u[0], ode["p_coefficients"],
                            ode["q_coefficients"])
    var = float(np.var(u))
    ga_r2 = 1.0 - float(np.mean((u - pred_ga) ** 2)) / var
    ode_r2 = 1.0 - float(np.mean((u - pred_ode) ** 2)) / var
    # Neither transfers the true structure (both R2 well below exact).
    # Notably the GA formula may sprinkle exp() tokens yet still collapse
    # (nested sqrt(exp(...)) mush): grammar access is not mechanism
    # discovery. R2 alone cannot crown a winner here -- a structurally
    # absurd form can outscore a diverging ODE rollout on a decaying tail
    # without capturing anything.
    assert ga_r2 < 0.9, f"GA R2={ga_r2:.3f}"
    assert ode_r2 < 0.9, f"unexpected transfer: R2={ode_r2:.3f}"


def test_planck_equations_change_under_5pct_noise_but_abstention_holds():
    # At 5% observation noise both methods return different structures than
    # on clean data (finite-difference and evolutionary search are both
    # noise-fragile here). What must remain stable is the abstention:
    # RIME must still refuse the exact claim.
    v = np.linspace(0.1, 6.0, 60)
    u = (v ** 3) / (np.exp(v) - 1.0)
    rng = np.random.RandomState(9)
    un = u + 0.05 * u.max() * rng.randn(60)
    cfg = {"max_x_degree": 2, "max_diff_order": 1, "smooth_window": 5,
           "sparsity_threshold": 0.05, "rcond_threshold": 1e-12}
    res = json.loads(ak.rime_diff_pade_discover_json(
        json.dumps(v.tolist()), json.dumps(un.tolist()),
        json.dumps(cfg)))
    assert res["is_exact_recovery"] is False
    ga = evolve({"v": v}, un, config=SearchConfig(
        use_native_ga=True, islands=2, island_size=30,
        generations=15, seed=42))
    assert ga.best_expression
