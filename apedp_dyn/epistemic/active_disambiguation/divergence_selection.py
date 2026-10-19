"""Divergence-point selection (E4 generalization, forensics-doc extension).

E4-P1..P8 (policy.py) selects the next observation from a FIXED measurement
menu. This module generalizes to selecting the query point itself on a
continuous domain under an EXPLICIT BUDGET, ranked by expected cost to
resolution rather than raw separation:

    For each candidate x on a frozen grid (+ deterministic refinement):
      gap(x)   = min pairwise |pred_i(x) - pred_j(x)|           (P4 raw separation)
      d(x)     = gap(x) / sigma(x)                              (P4 noise-normalized)
      n_req(x) = ceil( (k * sigma(x) / gap(x))^2 )              (single-point shots to reach k)
      exp_c(x) = n_req(x) * cost(x)                             (P-COST expected resolution cost)

    Selection minimizes exp_cost among points with gap > 0 (NOT maximizes gap:
    a strongly divergent but expensive/noisy point may be worse per unit
    evidence). Budget accounting is explicit at every round; abstention names
    its cause:
      INSUFFICIENT_DIVERGENCE  — survivors are (locally) indistinguishable: gap
                                 <= 0 at every grid point.
      INSUFFICIENT_BUDGET      — a distinguishable point exists but the minimum
                                 expected cost exceeds the remaining budget.

Protocol constraints preserved: P2/P3 no-truth leakage (selection sees only
hypotheses + observation model + cost model; truth_fn enters ONLY the
observation step), P6 re-adjudication through adjudicate(), P8 frozen seeds /
frozen grid density / deterministic refinement (no stochastic component in
selection itself).

Additive layer only: adjudicate.py and policy.py untouched.
Status: experimental (new generalization of the certified E4 menu policy).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

from apedp_dyn.epistemic.active_disambiguation.policy import (
    DEFAULT_DETECTABILITY_K,
    _scored,
)
from apedp_dyn.epistemic.adjudicate import adjudicate


@dataclass(frozen=True)
class DivergenceCandidate:
    """One evaluated query point and its cost-to-resolution statistics."""

    x: float
    gap: float  # min pairwise |pred_i - pred_j|
    sigma: float  # observation noise at this point (P2/P3 model, no truth)
    d_single: float  # gap / sigma
    n_req: int  # repeated single-point observations expected to reach k
    exp_cost: float  # n_req * cost(x)


@dataclass(frozen=True)
class DivergenceSelection:
    point: DivergenceCandidate | None
    abstained: bool
    reason: str = ""
    table: tuple[DivergenceCandidate, ...] = ()
    remaining_budget: float = 0.0


def _refine(
    hypotheses: Mapping[str, Callable[[float], float]],
    lo: float,
    hi: float,
    sigma_fn: Callable[[float], float],
    cost_fn: Callable[[float], float],
    detectability_k: float,
    n_grid: int,
    n_refine: int,
) -> list[DivergenceCandidate]:
    """Frozen global grid + deterministic local ternary refinement."""

    def _fns(x: float) -> DivergenceCandidate:
        preds = [float(fn(x)) for fn in hypotheses.values()]
        gap = min(abs(a - b) for i, a in enumerate(preds) for b in preds[i + 1 :]) if len(preds) > 1 else 0.0
        sig = float(sigma_fn(x))
        c = float(cost_fn(x))
        d_single = gap / sig if sig > 0 else 0.0
        if gap > 0 and c > 0:
            n_req = int(np.ceil((detectability_k * sig / gap) ** 2))
        else:
            n_req = 0
        return DivergenceCandidate(float(x), float(gap), sig, float(d_single), n_req, float(n_req * c) if n_req else float("inf"))

    grid = np.linspace(lo, hi, n_grid)
    cands = [_fns(float(v)) for v in grid]
    for _ in range(n_refine):
        viable = [c for c in cands if np.isfinite(c.exp_cost) and c.gap > 0]
        if not viable:
            break
        pool = sorted(viable, key=lambda c: c.exp_cost)
        anchor = pool[0]
        # Ternary refine AROUND the current best region (deterministic bust).
        span = (pool[1].x - anchor.x) if len(pool) > 1 else (hi - lo) / n_grid
        a, b = max(lo, anchor.x - 2 * span), min(hi, anchor.x + 2 * span)
        if b <= a:
            break
        probes = [_fns(float(v)) for v in np.linspace(a, b, 4)]
        cands = sorted(cands + probes, key=lambda c: c.x)
        seen: set[float] = set()
        uniq: list[DivergenceCandidate] = []
        for c in cands:
            key = round(c.x, 12)
            if key not in seen:
                seen.add(key)
                uniq.append(c)
        cands = uniq
    return sorted(cands, key=lambda c: c.x)


def select_divergence_point(
    hypotheses: Mapping[str, Callable[[float], float]],
    x_domain: tuple[float, float],
    sigma_fn: Callable[[float], float],
    cost_fn: Callable[[float], float],
    budget: float,
    detectability_k: float | None = None,
    n_grid: int = 64,
    n_refine: int = 6,
) -> DivergenceSelection:
    """Min-expected-cost divergence query (P-COST) with named abstention.

    Equal/near-equal exp_cost tie-breaks deterministically (larger d_single,
    then smaller x) — frozen ranking (P8); the grid and refinement schedule are
    fully deterministic, so this selection carries NO randomness.
    """
    if detectability_k is None:
        detectability_k = DEFAULT_DETECTABILITY_K
    if budget <= 0:
        return DivergenceSelection(None, True, "budget <= 0", remaining_budget=budget)
    lo, hi = float(x_domain[0]), float(x_domain[1])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return DivergenceSelection(None, True, "invalid x_domain", remaining_budget=budget)
    cands = _refine(hypotheses, lo, hi, sigma_fn, cost_fn, detectability_k, n_grid, n_refine)
    if not cands:
        return DivergenceSelection(None, True, "empty grid", remaining_budget=budget)
    zero_gap = all(c.gap <= 0 or not np.isfinite(c.exp_cost) for c in cands)
    if zero_gap:
        return DivergenceSelection(
            None,
            True,
            "INSUFFICIENT_DIVERGENCE: survivors indistinguishable on the whole domain",
            tuple(cands),
            budget,
        )
    viable = [c for c in cands if np.isfinite(c.exp_cost) and c.exp_cost <= budget]
    best = min(
        viable or [c for c in cands if np.isfinite(c.exp_cost)],
        key=lambda c: (c.exp_cost, -c.d_single, c.x),
    )
    if not viable:
        return DivergenceSelection(
            None,
            True,
            f"INSUFFICIENT_BUDGET: min expected cost {best.exp_cost:.2f} > budget {budget:.2f}",
            tuple(cands),
            budget,
        )
    return DivergenceSelection(best, False, f"exp_cost {best.exp_cost:.2f} <= budget {budget:.2f}", tuple(cands), budget)


@dataclass(frozen=True)
class DivergenceTrialResult:
    rounds: int
    status_post: str
    winner: str | None
    budget_spent: float
    points_used: tuple[float, ...]
    last_reason: str = ""


def run_divergence_trial(
    hypotheses: Mapping[str, Callable[[float], float]],
    x_domain: tuple[float, float],
    sigma_fn: Callable[[float], float],
    cost_fn: Callable[[float], float],
    truth_fn: Callable[[float], float],
    x_pre: np.ndarray,
    y_pre: np.ndarray,
    rng: np.random.Generator,
    budget: float,
    detectability_k: float | None = None,
    max_rounds: int = 8,
    n_grid: int = 64,
    n_refine: int = 6,
) -> DivergenceTrialResult:
    """Sequential closed loop (P1, P6): select -> observe -> re-adjudicate.

    Repeats ONE observation per round at the selected point (re-visits allowed:
    cost-aware selection already encodes the number of shots a point needs),
    spending cost(x) per round against the budget, until UNIQUE_SUPPORT,
    abstention, budget exhaustion, or max_rounds. Truth is touched ONLY at the
    observation step (P2).
    """
    spent = 0.0
    points: list[float] = []
    x_all = np.asarray(x_pre, dtype=float)
    y_all = np.asarray(y_pre, dtype=float)
    status = "MULTI_HYPOTHESIS"
    winner: str | None = None
    reason = ""
    for _ in range(max_rounds):
        sel = select_divergence_point(
            hypotheses, x_domain, sigma_fn, cost_fn, budget - spent,
            detectability_k=detectability_k, n_grid=n_grid, n_refine=n_refine,
        )
        if sel.abstained:
            reason = sel.reason
            break
        p = sel.point
        assert p is not None
        c = float(cost_fn(p.x))
        if spent + c > budget:
            reason = f"INSUFFICIENT_BUDGET: cost {c:.2f} would exceed budget {budget:.2f}"
            break
        spent += c
        points.append(p.x)
        y_new = float(truth_fn(p.x)) + float(rng.normal(0, p.sigma))
        x_all = np.concatenate([x_all, [p.x]])
        y_all = np.concatenate([y_all, [y_new]])
        scored = []
        for hid, fn in hypotheses.items():
            resid = y_all - np.array([fn(x) for x in x_all])
            scored.append(_scored(hid, resid, float(np.sqrt(np.mean(resid ** 2)))))
        scored.sort(key=lambda s: float(s.holdout_rmse) if s.holdout_rmse is not None else float("inf"))
        adj = adjudicate(
            list(scored),
            resid_best=np.asarray(scored[0].provenance["gate4_status"]["holdout_residuals"]),
            resid_rival=np.asarray(scored[1].provenance["gate4_status"]["holdout_residuals"]),
            bootstrap=False,
            data_origin="sim",
        )
        status = str(getattr(adj, "status", adj))
        if "unique_support" in status.lower():
            winner = scored[0].candidate.candidate_id
            return DivergenceTrialResult(len(points), status, winner, spent, tuple(points), "")
    return DivergenceTrialResult(len(points), status, winner, spent, tuple(points), reason)
