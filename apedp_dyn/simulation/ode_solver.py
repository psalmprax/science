"""D1 ODE simulation (downstream of gates 1-2; feeds gate 3 and the fitter)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class SimulateResult:
    success: bool
    detail: str
    t: np.ndarray = field(compare=False)
    y: np.ndarray = field(compare=False)
    metrics: dict[str, Any] = field(default_factory=dict)


def compile_rhs(model):
    """Compile a model to a fast vector RHS (ported from apedp/dyn simulator).

    Single lambdified f(t, x, params, inputs); identical trajectories to
    PhysicalModel.rhs at a fraction of the per-step cost. Falls back to
    model.rhs only if lambdification fails (never silently).
    """
    import sympy as sp

    order = [s.name for s in model.states]
    inames = [u.name for u in model.inputs]
    pnames = [p.name for p in model.parameters]
    eq_by_state = {e.state: e.sympy_expr for e in model.equations}
    f = sp.lambdify(
        [sp.Symbol("t")] + [sp.Symbol(n) for n in order]
        + [sp.Symbol(n) for n in pnames] + [sp.Symbol(n) for n in inames],
        [eq_by_state[n] for n in order], modules=["numpy"])

    def rhs(t: float, x: np.ndarray, u_vec: np.ndarray, pdict: dict) -> np.ndarray:
        out = np.asarray(f(t, *[float(v) for v in x],
                           *[float(pdict[k]) for k in pnames],
                           *[float(v) for v in np.atleast_1d(u_vec)]),
                         dtype=float).ravel()
        if out.shape != (len(order),) or not np.all(np.isfinite(out)):
            raise ValueError("non-finite compiled RHS")
        return out

    # Eager validation: compile errors surface here, not mid-integration.
    rhs(0.0, model.initial_vector(),
        np.zeros(max(len(inames), 1)), model.param_values())
    return rhs


def simulate(model, t_span: tuple[float, float],
             u_of_t: Callable[[float], np.ndarray], x0: np.ndarray,
             n_eval: int = 400, rtol: float = 1e-3, atol: float = 1e-6) -> SimulateResult:
    """Integrate an already-admissible model over the required span.

    Default tolerances favor fitter throughput; physics-precision checks
    (e.g. conservation) pass explicit tight tolerances.
    """
    from scipy.integrate import solve_ivp

    t_eval = np.linspace(t_span[0], t_span[1], n_eval)
    empty = (np.empty(0), np.empty((len(x0), 0)))
    metrics: dict[str, Any] = {"nfev": 0, "rhs_path": "compiled"}
    try:
        crhs = compile_rhs(model)
        pdict = model.param_values()
        drift = lambda t, x: crhs(t, x, np.atleast_1d(np.asarray(u_of_t(t), dtype=float)), pdict)  # noqa: E731
    except (ArithmeticError, ValueError, TypeError) as e:
        # Degradation is recorded, never hidden: conformance asserts the
        # compiled path, so which path actually ran must be visible in metrics.
        metrics["rhs_path"] = f"interpreted ({type(e).__name__})"
        drift = lambda t, x: model.rhs(t, x, u_of_t)  # noqa: E731
    try:
        sol = solve_ivp(drift, t_span, np.asarray(x0, dtype=float),
                        method="LSODA", t_eval=t_eval, rtol=rtol, atol=atol)
    except (ArithmeticError, ValueError) as e:
        return SimulateResult(False, f"integrator exception: {e}",
                              *empty, metrics)
    metrics["nfev"] = int(sol.nfev)
    if not sol.success:
        return SimulateResult(False, f"integrator failed: {sol.message}",
                              *empty, metrics)
    if not np.all(np.isfinite(sol.y)):
        return SimulateResult(False, "non-finite trajectory", *empty, metrics)
    if sol.t[-1] < t_span[1]:
        return SimulateResult(False, "span not covered", *empty, metrics)
    metrics["max_abs_state"] = float(np.max(np.abs(sol.y)))
    return SimulateResult(True, "ok", sol.t, sol.y, metrics)
