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


def simulate(model, t_span: tuple[float, float],
             u_of_t: Callable[[float], np.ndarray], x0: np.ndarray,
             n_eval: int = 400) -> SimulateResult:
    """Integrate an already-admissible model over the required span."""
    from scipy.integrate import solve_ivp

    t_eval = np.linspace(t_span[0], t_span[1], n_eval)
    empty = (np.empty(0), np.empty((len(x0), 0)))
    try:
        sol = solve_ivp(lambda t, x: model.rhs(t, x, u_of_t), t_span,
                        np.asarray(x0, dtype=float),
                        method="LSODA", t_eval=t_eval)
    except (ArithmeticError, ValueError) as e:
        return SimulateResult(False, f"integrator exception: {e}",
                              *empty, {"nfev": 0})
    metrics = {"nfev": int(sol.nfev)}
    if not sol.success:
        return SimulateResult(False, f"integrator failed: {sol.message}",
                              *empty, metrics)
    if not np.all(np.isfinite(sol.y)):
        return SimulateResult(False, "non-finite trajectory", *empty, metrics)
    if sol.t[-1] < t_span[1]:
        return SimulateResult(False, "span not covered", *empty, metrics)
    metrics["max_abs_state"] = float(np.max(np.abs(sol.y)))
    return SimulateResult(True, "ok", sol.t, sol.y, metrics)
