"""D1.2/D1.3 — parameter fitter: bounded trajectory least squares on frozen IR.

The symbolic structure is already chosen by the grammar; the remaining problem
is numerical parameter estimation, solved with trust-region reflective
least squares (scipy), not a GA. Residuals are per-state normalized trajectory
errors on the TRAIN regime only.

D1.3 invariant: the simulator stays downstream of the structural gates.
fit_candidate() re-verifies them on the materialized model and refuses to
integrate inadmissible candidates — a bad structure can never win merely
because the optimizer makes it fit.

Every gate that run_all_gates() evaluates is evaluated here too. A gate wired
into the audit path but not the promotion path is decorative, so the two sets
are kept equal by _GATE_PROVENANCE_KEY and asserted by a contract test.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from apedp_dyn.gates import (
    dao_validity,
    dimensional_validity,
    equation_admissibility,
    forward_simulation_validity,
    intervention_holdout,
)
from apedp_dyn.grammar import ScoredCandidate, materialize
from apedp_dyn.simulation import simulate

_GATE_PROVENANCE_KEY = {
    "dimensional_validity": "gate1_status",
    "equation_admissibility": "gate2_status",
    "forward_simulation_validity": "gate3_status",
    "intervention_holdout": "gate4_status",
    "dao_validity": "dao_status",
}
"""Gate name -> provenance key, for every gate run_all_gates() evaluates.

Single source of truth: the promotion path and the audit path must not be
allowed to drift apart, because a gate that only the audit path runs cannot
prevent anything being promoted.
"""


@dataclass(frozen=True)
class FitResult:
    param_values: dict[str, float]
    train_loss: float
    nfev: int
    success: bool
    message: str


def _bind(names: Sequence[str],
          values: np.ndarray | Sequence[float]) -> dict[str, float]:
    """Bind parameter names to values, refusing any length mismatch.

    ``dict(zip(names, values))`` silently truncates to the shorter side: fitted
    numbers would be attached to the wrong parameters, or dropped. The mapping
    from estimates to parameter identity is load-bearing, so the invariant is
    enforced here rather than assumed across the lines that build the arrays.
    """
    if len(names) != len(values):
        raise ValueError(
            f"parameter binding mismatch: {len(names)} names {list(names)} "
            f"vs {len(values)} values")
    return {n: float(v) for n, v in zip(names, values, strict=True)}


def _with_values(model, values: dict[str, float]):
    params = tuple(dataclasses.replace(p, value=float(values[p.name]))
                   for p in model.parameters)
    return dataclasses.replace(model, parameters=params)


def _train_arrays(problem):
    reg = problem.train
    t = np.asarray(reg["truth_t"], dtype=float)
    Y = np.asarray(reg["truth_y"], dtype=float)
    U = np.array([reg["u_of_t"](ti) for ti in t], dtype=float).reshape(len(t), -1)
    return reg, t, Y, U


def _warm_start(problem, model0):
    """Derivative least-squares initialization for a FIXED structure.

    Numeric initialization only: the term set comes solely from the IR.
    Per parameterized equation (kinematic pass-throughs skipped), spline-
    estimated derivatives regressed onto the equation's term columns give a
    starting point inside the stable basin; trajectory refinement (below)
    does the actual fitting. General over 1-DOF and multi-DOF models.
    """
    import sympy as sp
    from scipy.interpolate import CubicSpline

    _, t, Y, U = _train_arrays(problem)
    order = [s.name for s in model0.states]
    inames = [u.name for u in model0.inputs]
    col_of_state = {n: i for i, n in enumerate(order)}
    values: dict[str, float] = {}
    for eq in model0.equations:
        eq_params = [p for p in model0.parameters
                     if sp.Symbol(p.name) in eq.sympy_expr.atoms(sp.Symbol)]
        if not eq_params:
            continue  # kinematic pass-through: no parameters to initialize
        target = CubicSpline(t, Y[:, col_of_state[eq.state]]).derivative()(t)
        syms = [sp.Symbol(n) for n in order] + [sp.Symbol(n) for n in inames]
        args = [Y[:, i] for i in range(len(order))] + [U[:, i] for i in range(len(inames))]
        cols = []
        for p in eq_params:
            col = np.asarray(sp.lambdify(
                syms, sp.diff(eq.sympy_expr, sp.Symbol(p.name)),
                modules=["numpy"])(*args), dtype=float).ravel()
            if col.size == 1 and len(t) > 1:
                col = np.full(len(t), col[0])
            cols.append(col)
        coef, *_ = np.linalg.lstsq(np.column_stack(cols), target, rcond=None)
        for p, c in zip(eq_params, coef, strict=True):
            values[p.name] = float(c)
    missing = [p.name for p in model0.parameters if p.name not in values]
    if missing:
        # A parameter in no equation has no derivative estimate; guessing a
        # value here would silently plant a number the data never constrained.
        raise ValueError(f"warm start left parameters uninitialized: {missing}")
    return np.array([values[p.name] for p in model0.parameters])


def fit_candidate(problem, candidate,
                  x0: np.ndarray | None = None,
                  max_nfev: int = 300) -> ScoredCandidate:
    """Fit IR parameters on train trajectories; run gates 3-4 on the result."""
    from scipy.optimize import least_squares

    from apedp_dyn.grammar.candidate_ir import SystemCandidate
    from apedp_dyn.grammar.coupled import materialize_system

    prov: dict[str, Any] = ScoredCandidate(candidate=candidate,
                                           provenance={}).base_provenance()
    try:
        if isinstance(candidate, SystemCandidate):
            model0 = materialize_system(problem, candidate)
        else:
            model0 = materialize(problem, candidate)
    except (TypeError, ValueError) as e:
        # Dimensionally meaningless proposals die here: no integration, no fit.
        prov["rejection_reason"] = f"materialize: {e}"
        return ScoredCandidate(candidate, prov)
    g1 = dimensional_validity(model0)
    prov["gate1_status"] = {"passed": g1.passed, "detail": g1.detail}
    if not g1.passed:
        prov["rejection_reason"] = f"gate1: {g1.detail}"
        return ScoredCandidate(candidate, prov)
    # Gate 1.5 (dao_validity) is structural, so it belongs with gates 1-2 and
    # not after the fit: a DAE-invalid structure should never be integrated.
    # Running it here (not only in run_all_gates) keeps it on the path that
    # actually decides promotion.
    g15 = dao_validity(model0)
    prov["dao_status"] = {"passed": g15.passed, "detail": g15.detail}
    if not g15.passed:
        prov["rejection_reason"] = f"dao_validity: {g15.detail}"
        return ScoredCandidate(candidate, prov)
    names = [p.name for p in model0.parameters]
    bounds = [p.bounds if p.bounds is not None else (-np.inf, np.inf)
              for p in model0.parameters]
    # Gate 2 needs values to check finiteness at IC; warm start provides them
    # (numeric initialization for the frozen structure, not structure search).
    if x0 is not None:
        probe = np.asarray(x0, dtype=float)
    else:
        try:
            probe = _warm_start(problem, model0)
        except (ValueError, np.linalg.LinAlgError) as e:
            # Un-initializable structure: reject it and keep the search alive.
            prov["rejection_reason"] = f"warm start: {e}"
            return ScoredCandidate(candidate, prov)
    g2 = equation_admissibility(_with_values(model0, _bind(names, probe)))
    prov["gate2_status"] = {"passed": g2.passed, "detail": g2.detail}
    if not g2.passed:
        prov["rejection_reason"] = f"gate2: {g2.detail}"
        return ScoredCandidate(candidate, prov)

    reg = problem.train
    t_eval = np.asarray(reg["truth_t"], dtype=float)
    Y = np.asarray(reg["truth_y"], dtype=float)
    scale = np.std(Y, axis=0)
    scale[scale == 0] = 1.0
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)

    def residuals(p):
        m = _with_values(model0, _bind(names, p))
        res = simulate(m, reg["t_span"], reg["u_of_t"],
                       np.asarray(reg["x0"], dtype=float), n_eval=len(t_eval))
        if not res.success:
            return np.full(Y.size, 1e3)
        return ((res.y.T - Y) / scale).ravel()

    sol = least_squares(residuals, probe, bounds=(lo, hi), max_nfev=max_nfev,
                        xtol=1e-10, ftol=1e-10, gtol=1e-10)
    values = _bind(names, sol.x)
    model = _with_values(model0, values)
    train_loss = float(np.mean(residuals(sol.x) ** 2))
    fit = FitResult(values, train_loss, int(sol.nfev), bool(sol.success),
                    f"cost={sol.cost:.3g} optimality={sol.optimality:.3g} "
                    f"status={sol.status}")
    prov["training_loss"] = train_loss
    g3 = forward_simulation_validity(model, reg["t_span"], reg["u_of_t"],
                                     np.asarray(reg["x0"], dtype=float))
    prov["gate3_status"] = {"passed": g3.passed, "detail": g3.detail,
                            **{f"sim_{k}": v for k, v in g3.metrics.items()}}
    g4 = intervention_holdout(model, problem)
    prov["gate4_status"] = {"passed": g4.passed, "detail": g4.detail,
                            **{f"holdout_{k}": v for k, v in g4.metrics.items()}}
    return ScoredCandidate(candidate, prov,
                           gates=(g1, g2, g3, g4), fit=fit,
                           train_loss=train_loss,
                           holdout_rmse=g4.metrics.get("rmse"))
