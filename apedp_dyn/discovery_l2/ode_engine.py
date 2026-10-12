"""Autonomous Dynamic ODE Discovery (L2-ODE).

Applies unconstrained physically typed AST generation to time-derivative targets:
    \\dot{x}_i(t) = f_i(x_1, ..., x_m, t)

The time-derivative \\dot{x}_i is evaluated numerically via cubic splines or
finite differences, typed with SI dimensions [x_i] * T^{-1}, and submitted to the
L2 open discovery engine without candidate libraries or template hints.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.interpolate import CubicSpline

from apedp_dyn.discovery_l2.dimensional_types import BaseDimension
from apedp_dyn.discovery_l2.l2_engine import L2DiscoveryEngine, L2DiscoveryResult
from apedp_dyn.epistemic.adjudicate import DataOrigin


@dataclass(frozen=True)
class ODEDiscoveryResult:
    """Result of autonomous dynamical ODE discovery."""

    target_state: str
    target_derivative: np.ndarray
    l2_result: L2DiscoveryResult
    inferred_equation: str


def estimate_time_derivative(
    t: np.ndarray,
    x: np.ndarray,
    method: Literal["spline", "central_diff"] = "spline",
) -> np.ndarray:
    """Estimate time derivative dx/dt from discrete trajectory observations."""
    t_arr = np.asarray(t, dtype=float).ravel()
    x_arr = np.asarray(x, dtype=float).ravel()
    if len(t_arr) != len(x_arr):
        raise ValueError(f"Length mismatch: t has {len(t_arr)} pts, x has {len(x_arr)} pts.")
    if len(t_arr) < 4:
        raise ValueError("At least 4 timepoints required for stable derivative estimation.")

    if method == "spline":
        cs = CubicSpline(t_arr, x_arr)
        return np.asarray(cs.derivative()(t_arr))
    elif method == "central_diff":
        return np.asarray(np.gradient(x_arr, t_arr))
    else:
        raise ValueError(f"Unknown derivative estimation method: {method}")


class L2ODEDiscoveryEngine:
    """Autonomous ODE discovery engine discovering f(x, y, ...) in dx/dt = f(x, y, ...)."""

    def __init__(
        self,
        max_depth: int = 2,
        max_complexity: int = 5,
        operators: tuple[str, ...] = ("+", "-", "*", "/", "pow"),
    ) -> None:
        self.engine = L2DiscoveryEngine(
            max_depth=max_depth,
            max_complexity=max_complexity,
            operators=operators,
        )

    def discover_ode(
        self,
        t: np.ndarray,
        trajectories: dict[str, np.ndarray],
        state_dimensions: dict[str, BaseDimension],
        target_state: str,
        target_dimension: BaseDimension | None = None,
        derivative_method: Literal["spline", "central_diff"] = "spline",
        data_origin: str = DataOrigin.SIMULATION,
        train_ratio: float = 0.6,
        bootstrap: bool = True,
        seed: int = 42,
    ) -> ODEDiscoveryResult:
        """Discover the autonomous ODE governing target_state."""
        if target_state not in trajectories:
            raise KeyError(f"Target state {target_state!r} not in trajectories.")
        if target_state not in state_dimensions:
            raise KeyError(f"Target state {target_state!r} dimension not supplied.")

        # Compute numerical derivative for target state
        x_target = trajectories[target_state]
        x_dot = estimate_time_derivative(t, x_target, method=derivative_method)

        # Target dimension: if caller supplied target_dimension, use it.
        # Otherwise: if state is dimensionless, autonomous RHS is dimensionless;
        # if state has dimensions, target is [target_state] / [T].
        if target_dimension is not None:
            target_dim = target_dimension
        elif state_dimensions[target_state].is_dimensionless():
            target_dim = BaseDimension()
        else:
            target_dim = state_dimensions[target_state] / BaseDimension(t=1)

        # Run unconstrained L2 discovery
        l2_res = self.engine.discover(
            X=trajectories,
            t=t,
            Y=x_dot,
            dimensions=state_dimensions,
            target_dimension=target_dim,
            train_ratio=train_ratio,
            data_origin=data_origin,
            bootstrap=bootstrap,
            seed=seed,
        )

        inferred_eq = f"d({target_state})/dt = [NO PROMOTED EQUATION]"
        if l2_res.adjudication.promoted is not None:
            best_cand = l2_res.adjudication.promoted
            expr_str = (
                best_cand.provenance.get("canonical_expr")
                or best_cand.candidate.metadata.get("canonical_expr")
                or best_cand.candidate.candidate_id
            )
            coef = 1.0
            if best_cand.fit and best_cand.fit.param_values:
                coef = best_cand.fit.param_values.get(
                    "theta_1", next(iter(best_cand.fit.param_values.values()))
                )
            inferred_eq = f"d({target_state})/dt = {coef:.4g} * ({expr_str})"

        return ODEDiscoveryResult(
            target_state=target_state,
            target_derivative=x_dot,
            l2_result=l2_res,
            inferred_equation=inferred_eq,
        )
