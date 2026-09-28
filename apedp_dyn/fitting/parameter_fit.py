"""D1.2/D1.3 — parameter fitter: bounded trajectory least squares on frozen IR.

The symbolic structure is already chosen by the grammar; the remaining problem
is numerical parameter estimation, solved with trust-region reflective
least squares (scipy), not a GA. Residuals are per-state normalized trajectory
errors on the TRAIN regime only.

D1.3 invariant: the simulator stays downstream of gates 1-2. fit_candidate()
re-verifies gates 1-2 on the materialized model and refuses to integrate
inadmissible candidates — a bad structure can never win merely because the
optimizer makes it fit.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from apedp_dyn.gates import (
    dimensional_validity,
    equation_admissibility,
    forward_simulation_validity,
    intervention_holdout,
)
from apedp_dyn.grammar import (
    CandidateDynamics,
    ScoredCandidate,
    materialize,
)
from apedp_dyn.simulation import simulate


@dataclass(frozen=True)
class FitResult:
    param_values: dict[str, float]
    train_loss: float
    nfev: int
    success: bool
    message: str


def _with_values(model, values: dict[str, float]):
    params = tuple(dataclasses.replace(p, value=float(values[p.name]))
                   for p in model.parameters)
    return dataclasses.replace(model, parameters=params)


def _warm_start(problem, model0):
    """Derivative least-squares initialization for a FIXED structure.

    Numeric initialization only: the term set comes solely from the IR.
    Spline-estimated derivatives regressed onto the candidate's term columns
    give a starting point inside the stable basin; trajectory refinement
    (below) does the actual fitting.
    """
    from scipy.interpolate import CubicSpline

    import sympy as sp

    reg = problem.train
    t = np.asarray(reg["truth_t"], dtype=float)
    Y = np.asarray(reg["truth_y"], dtype=float)
    U = np.array([reg["u_of_t"](ti) for ti in t], dtype=float).reshape(len(t), -1)
    order = [s.name for s in model0.states]
    inames = [u.name for u in model0.inputs]
    vel = order[1]
    dvel = CubicSpline(t, Y[:, 1]).derivative()(t)
    accel_eq = next(e.sympy_expr for e in model0.equations if e.state == vel)
    # Split the accel sum into per-parameter columns: d(sum)/dparam = term.
    cols = []
    for p in model0.parameters:
        col = np.asarray(sp.lambdify(
            [sp.Symbol(n) for n in order] + [sp.Symbol(n) for n in inames],
            sp.diff(accel_eq, sp.Symbol(p.name)),
            modules=["numpy"])(*[Y[:, i] for i in range(len(order))],
                                *[U[:, i] for i in range(len(inames))]),
            dtype=float).ravel()
        cols.append(col)
    cols = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(cols, dvel, rcond=None)
    return np.asarray(coef, dtype=float)


def fit_candidate(problem, candidate: CandidateDynamics,
                  x0: np.ndarray | None = None,
                  max_nfev: int = 300) -> ScoredCandidate:
    """Fit IR parameters on train trajectories; run gates 3-4 on the result."""
    from scipy.optimize import least_squares

    prov: dict[str, Any] = ScoredCandidate(candidate=candidate,
                                           provenance={}).base_provenance()
    try:
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
    names = [p.name for p in model0.parameters]
    bounds = [p.bounds if p.bounds is not None else (-np.inf, np.inf)
              for p in model0.parameters]
    # Gate 2 needs values to check finiteness at IC; warm start provides them
    # (numeric initialization for the frozen structure, not structure search).
    probe = np.asarray(x0, dtype=float) if x0 is not None else _warm_start(problem, model0)
    g2 = equation_admissibility(_with_values(model0, dict(zip(names, probe))))
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
        m = _with_values(model0, dict(zip(names, p)))
        res = simulate(m, reg["t_span"], reg["u_of_t"],
                       np.asarray(reg["x0"], dtype=float), n_eval=len(t_eval))
        if not res.success:
            return np.full(Y.size, 1e3)
        return ((res.y.T - Y) / scale).ravel()

    sol = least_squares(residuals, probe, bounds=(lo, hi), max_nfev=max_nfev,
                        xtol=1e-10, ftol=1e-10, gtol=1e-10)
    values = dict(zip(names, [float(v) for v in sol.x]))
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
