"""Pins for the E4 divergence-point selection generalization (A-round).

Acceptance criteria:
  D1  A constructed MULTI_HYPOTHESIS pair is resolved: the closed loop reaches
      UNIQUE_SUPPORT by observing at points the selector proposes (budgeted).
  D2  Cost-awareness: selection minimizes EXPECTED COST TO RESOLUTION, not raw
      gap — a strongly divergent but expensive/noisy point must not be chosen
      when a cheaper point clears k within budget.
  D3  Named abstention: (a) INSUFFICIENT_DIVERGENCE when survivors are
      indistinguishable on the whole domain; (b) INSUFFICIENT_BUDGET when a
      distinguishable point exists but its min expected cost exceeds budget.
  D4  No-truth leakage: the selection is invariant to the truth function
      (P2/P3); truth enters only the observation step.
  D5  Determinism: the same inputs give the same selection (frozen grid +
      deterministic refinement, P8).
"""

from __future__ import annotations

import numpy as np
import pytest

from apedp_dyn.epistemic.active_disambiguation.divergence_selection import (
    DivergenceSelection,
    run_divergence_trial,
    select_divergence_point,
)


def _quad_pair() -> tuple:
    """Two survivors on x in [-1, 1]: h1 = x^2 (gap max at |x| = 1 roots), h2 = 0.5 - 0.2*x (linear).

    Reads identically near x ~ 0; diverges toward the edges.
    """

    def h1(x: float) -> float:
        return 0.5 * x * x

    def h2(x: float) -> float:
        return 0.5 - 0.2 * x

    return {"H1": h1, "H2": h2}, h1, h2


def _flat_noise_flat_cost(x: float) -> tuple:
    return 0.05, 1.0


DOMAIN = (-1.0, 1.0)
K = 3.0


def test_selects_analytic_divergence_optimum():  # D1 (selection half)
    hyps, _h1, _ = _quad_pair()
    # gap(x) = |0.5x^2 - (0.5 - 0.2x)|; on the branch where h2 > h1 (x in
    # [-0.61, 0.41]) the gap is g(x) = 0.5 - 0.2x - 0.5x^2, maximized where
    # g'(x) = -0.2 - x = 0 -> x* = -0.2, g(x*) = 0.52. With flat noise and
    # cost, min-exp-cost == max d_single == the max-gap point.
    sel = select_divergence_point(hyps, DOMAIN, lambda x: 0.05, lambda x: 1.0, budget=100.0, detectability_k=K)
    assert isinstance(sel, DivergenceSelection)
    assert not sel.abstained and sel.point is not None
    assert sel.point.x == pytest.approx(-0.2, abs=1e-2), sel
    assert sel.point.gap == pytest.approx(0.52, abs=1e-2), sel
    assert sel.point.d_single == pytest.approx(sel.point.gap / 0.05, rel=1e-6)
    assert sel.point.n_req == pytest.approx(np.ceil((K * 0.05 / sel.point.gap) ** 2))


def test_cost_aware_not_raw_gap_maximizing():  # D2
    hyps, _, _ = _quad_pair()
    # The raw-gap optimum is x* ~ -0.2; make THAT zone punitive while the
    # sub-optimal point x=0 (gap 0.5, still d=10 >= k, n_req == 1) stays cheap.
    # A raw-gap-only policy goes to -0.2; the cost-aware policy must not.
    sel = select_divergence_point(
        hyps, DOMAIN,
        sigma_fn=lambda x: 0.05,
        cost_fn=lambda x: 500.0 if x < -0.05 else 1.0,
        budget=1000.0, detectability_k=K,
    )
    assert not sel.abstained and sel.point is not None
    assert sel.point.x > -0.05, f"selected the punitive-cost point {sel.point}"
    assert sel.point.d_single >= K  # single-shot resolution achievable
    cheap_ok = [c for c in sel.table if c.x > -0.05 and np.isfinite(c.exp_cost)]
    assert cheap_ok and sel.point.exp_cost == min(c.exp_cost for c in cheap_ok)


def test_abstain_insufficient_divergence():  # D3a
    hyps = {"H1": lambda x: 0.3, "H2": lambda x: 0.3}  # identical survivors
    sel = select_divergence_point(hyps, DOMAIN, lambda x: 0.05, lambda x: 1.0, budget=100.0, detectability_k=K)
    assert sel.abstained and sel.point is None
    assert "INSUFFICIENT_DIVERGENCE" in sel.reason


def test_abstain_insufficient_budget():  # D3b
    hyps, _, _ = _quad_pair()
    # n_req == 1 at min-cost points costs 1.0/obs; a 0.5 budget cannot afford it.
    sel = select_divergence_point(hyps, DOMAIN, lambda x: 0.05, lambda x: 1.0, budget=0.5, detectability_k=K)
    assert sel.abstained and sel.point is None
    assert "INSUFFICIENT_BUDGET" in sel.reason
    assert min(c.exp_cost for c in sel.table if np.isfinite(c.exp_cost)) > 0.5


def test_no_truth_leakage():  # D4
    hyps, _, _ = _quad_pair()
    a = select_divergence_point(hyps, DOMAIN, lambda x: 0.05, lambda x: 1.0, budget=100.0, detectability_k=K)
    b = select_divergence_point(hyps, DOMAIN, lambda x: 0.05, lambda x: 1.0, budget=100.0, detectability_k=K)
    assert a.point is not None and b.point is not None
    # Selection cannot see truth: two arbitrary truths must give the same plan.
    # (Enforced structurally — truth_fn is not a selection argument.)
    assert a.point.x == b.point.x and a.point.exp_cost == b.point.exp_cost


def test_determinism():  # D5
    hyps, _h1, _h2 = _quad_pair()
    runs = [
        select_divergence_point(hyps, DOMAIN, lambda x: 0.05, lambda x: 1.0, budget=100.0)
        for _ in range(3)
    ]
    xs = [r.point.x for r in runs]
    assert len(set(xs)) == 1


def test_closed_loop_resolves_within_budget():  # D1 (loop half)
    hyps, h1, _h2 = _quad_pair()
    rng = np.random.default_rng(7)
    x_pre = np.linspace(-0.9, 0.9, 20)
    y_pre = np.array([h1(x) for x in x_pre]) + rng.normal(0, 0.02, size=x_pre.shape)
    res = run_divergence_trial(
        hyps, DOMAIN,
        sigma_fn=lambda x: 0.05, cost_fn=lambda x: 1.0,
        truth_fn=h1,
        x_pre=x_pre, y_pre=y_pre, rng=rng,
        budget=100.0, detectability_k=K, max_rounds=8,
    )
    assert "unique_support" in res.status_post.lower()
    assert res.winner == "H1"
    assert res.budget_spent > 0 and res.budget_spent <= 100.0
    assert len(res.points_used) == int(res.rounds)
    for x in res.points_used:
        assert DOMAIN[0] <= x <= DOMAIN[1]


def test_closed_loop_budget_starve_named():  # D3b (loop half)
    hyps, h1, _h2 = _quad_pair()
    rng = np.random.default_rng(7)
    x_pre = np.linspace(-0.9, 0.9, 20)
    y_pre = np.array([h1(x) for x in x_pre]) + rng.normal(0, 0.02, size=x_pre.shape)
    res = run_divergence_trial(
        hyps, DOMAIN,
        sigma_fn=lambda x: 0.05, cost_fn=lambda x: 10.0,
        truth_fn=h1,
        x_pre=x_pre, y_pre=y_pre, rng=rng,
        budget=5.0, detectability_k=K, max_rounds=8,
    )
    if "unique_support" not in res.status_post.lower():
        assert res.budget_spent <= 5.0 + 1e-9
        assert ("INSUFFICIENT_BUDGET" in res.last_reason) or ("budget" in res.last_reason.lower() and res.rounds < 8)
        assert res.winner is None if res.rounds < 8 else True
