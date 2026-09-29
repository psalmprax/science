"""D1 ODE simulation (downstream of gates 1-2; feeds gate 3 and the fitter)."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

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
    # The probe arity must match the declared inputs exactly: padding a
    # zero-input model to one dummy made every such model fail here and
    # silently drop to the interpreted path.
    rhs(0.0, model.initial_vector(),
        np.zeros(len(inames)), model.param_values())
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
    # Input arity is a precondition, checked before an RHS path is chosen. A
    # u_of_t of the wrong length is a problem-spec error, not an integrator
    # failure, and it must be reported rather than escaping as a TypeError
    # from inside solve_ivp.
    n_inputs = len(model.inputs)
    if n_inputs:
        try:
            u_probe = np.atleast_1d(np.asarray(u_of_t(t_span[0]), dtype=float)).ravel()
        except (TypeError, ValueError) as e:
            metrics["rhs_path"] = "not-attempted"
            return SimulateResult(
                False, f"u_of_t unusable at t={t_span[0]}: {type(e).__name__}: {e}",
                *empty, metrics)
        if u_probe.size != n_inputs:
            metrics["rhs_path"] = "not-attempted"
            return SimulateResult(
                False,
                f"input arity mismatch: model declares {n_inputs} input(s) "
                f"{[i.name for i in model.inputs]} but u_of_t returned "
                f"{u_probe.size}", *empty, metrics)
    try:
        crhs = compile_rhs(model)
        pdict = model.param_values()

        def _u(t: float) -> np.ndarray:
            # Same arity rule as the precheck above. An autonomous model still
            # carries a problem-supplied u_of_t, and make_signal's "zero"
            # defaults to arity 1, so splatting it handed a length-1 vector to
            # a compiled RHS built for zero inputs -- an arity TypeError raised
            # at call time inside solve_ivp, long after the guard had passed.
            if not n_inputs:
                return np.empty(0)
            return np.atleast_1d(np.asarray(u_of_t(t), dtype=float))

        drift = lambda t, x: crhs(t, x, _u(t), pdict)  # noqa: E731
    except (ArithmeticError, ValueError, TypeError) as e:
        # Degradation is recorded, never hidden: conformance asserts the
        # compiled path, so which path actually ran must be visible in metrics.
        metrics["rhs_path"] = f"interpreted ({type(e).__name__})"
        drift = lambda t, x: model.rhs(t, x, u_of_t)  # noqa: E731
    try:
        sol = solve_ivp(drift, t_span, np.asarray(x0, dtype=float),
                        method="LSODA", t_eval=t_eval, rtol=rtol, atol=atol)
    except (ArithmeticError, ValueError, TypeError) as e:
        # TypeError included: the interpreted fallback reaches model.rhs, whose
        # attribute/arity errors would otherwise escape simulate entirely.
        return SimulateResult(False, f"integrator exception ({type(e).__name__}): {e}",
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
