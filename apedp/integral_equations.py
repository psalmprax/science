"""Bounded Nyström solvers for linear second-kind integral equations."""
from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import sympy as sp

from apedp.eval_extended import _run_bounded
from apedp.safety import safe_parse_expr

_DEFAULT_TIMEOUT = 10.0
_MAX_NODES = 1024


def _validate_common(
    a: float,
    b: float,
    nodes: int,
    coupling: float,
    timeout: float,
    x_var: str,
    integration_var: str,
) -> tuple[float, float, int, float]:
    if not math.isfinite(float(a)) or not math.isfinite(float(b)) or float(b) <= float(a):
        raise ValueError("Need finite a < b")
    if int(nodes) < 8 or int(nodes) > _MAX_NODES:
        raise ValueError(f"nodes must be between 8 and {_MAX_NODES}")
    if not math.isfinite(float(coupling)):
        raise ValueError("coupling must be finite")
    if not math.isfinite(float(timeout)) or float(timeout) <= 0:
        raise ValueError("timeout must be finite and > 0")
    if not x_var or not integration_var or x_var == integration_var:
        raise ValueError("x_var and integration_var must be distinct names")
    return float(a), float(b), int(nodes), float(coupling)


def _prepare_expressions(
    kernel_str: str,
    forcing_str: str,
    x_var: str,
    integration_var: str,
    params: dict[str, float] | None,
) -> tuple[sp.Symbol, sp.Symbol, sp.Expr, sp.Expr]:
    x_sym, t_sym = sp.Symbol(x_var), sp.Symbol(integration_var)
    substitutions: dict[sp.Symbol, float] = {}
    for name, value in (params or {}).items():
        if name in {x_var, integration_var}:
            raise ValueError(f"Parameter '{name}' is reserved by the integral equation")
        if not math.isfinite(float(value)):
            raise ValueError(f"Parameter '{name}' must be finite")
        substitutions[sp.Symbol(name)] = float(value)
    kernel = safe_parse_expr(kernel_str).subs(substitutions)
    forcing = safe_parse_expr(forcing_str).subs(substitutions)
    for label, expr in (("kernel", kernel), ("forcing", forcing)):
        unknown = expr.free_symbols - {x_sym, t_sym}
        if unknown:
            names = ", ".join(sorted(str(s) for s in unknown))
            raise ValueError(f"Unknown {label} symbols: {names}")
    return x_sym, t_sym, kernel, forcing


def _vector_eval(fn: Callable, x: np.ndarray, label: str) -> np.ndarray:
    try:
        values = np.asarray(fn(x), dtype=float)
    except (TypeError, ValueError, ArithmeticError) as exc:
        raise ArithmeticError(f"{label} evaluation failed: {exc}") from exc
    if values.shape == ():
        values = np.full(x.shape, float(values), dtype=float)
    try:
        values = np.broadcast_to(values, x.shape).astype(float, copy=False)
    except ValueError as exc:
        raise ArithmeticError(f"{label} did not broadcast to the quadrature grid") from exc
    if not bool(np.all(np.isfinite(values))):
        raise ArithmeticError(f"{label} returned non-finite values")
    return values


def _matrix_eval(fn: Callable, x: np.ndarray, t: np.ndarray) -> np.ndarray:
    try:
        values = np.asarray(fn(x[:, None], t[None, :]), dtype=float)
    except (TypeError, ValueError, ArithmeticError) as exc:
        raise ArithmeticError(f"kernel evaluation failed: {exc}") from exc
    if values.shape == ():
        values = np.full((x.size, t.size), float(values), dtype=float)
    try:
        values = np.broadcast_to(values, (x.size, t.size)).astype(float, copy=False)
    except ValueError as exc:
        raise ArithmeticError("kernel did not broadcast to the quadrature matrix") from exc
    if not bool(np.all(np.isfinite(values))):
        raise ArithmeticError("kernel returned non-finite values")
    return values


def _resolved_coupling(coupling: float, parameter: float | None) -> float:
    if parameter is not None:
        if coupling != 1.0:
            raise ValueError("Pass only one of coupling and parameter")
        coupling = float(parameter)
    if not math.isfinite(float(coupling)):
        raise ValueError("coupling must be finite")
    return float(coupling)


def solve_nystrom(
    kernel_str: str,
    forcing_str: str,
    a: float,
    b: float,
    nodes: int = 64,
    coupling: float = 1.0,
    kind: str = "fredholm",
    x_var: str = "x",
    integration_var: str = "t",
    params: dict[str, float] | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
    parameter: float | None = None,
) -> dict[str, object]:
    """Solve ``y = f + coupling * K[y]`` by a bounded Nyström discretization.

    ``kind="fredholm"`` solves the second-kind equation with an integral over
    ``[a,b]`` using Gauss-Legendre quadrature. ``kind="volterra"`` solves the
    second-kind equation with an integral over ``[a,x]`` using a lower-triangular
    composite-trapezoid Nyström system. The returned dictionary includes the
    grid, values, residual norm, and a conditioning diagnostic.
    """
    if kind not in {"fredholm", "volterra"}:
        raise ValueError("kind must be 'fredholm' or 'volterra'")
    coupling = _resolved_coupling(coupling, parameter)
    aa, bb, n, lam = _validate_common(
        a, b, nodes, coupling, timeout, x_var, integration_var
    )
    x_sym, t_sym, kernel_expr, forcing_expr = _prepare_expressions(
        kernel_str, forcing_str, x_var, integration_var, params
    )
    kernel_fn = sp.lambdify((x_sym, t_sym), kernel_expr, modules=["numpy"])
    forcing_fn = sp.lambdify(x_sym, forcing_expr.subs(t_sym, x_sym), modules=["numpy"])

    def _solve_fredholm() -> dict[str, object]:
        legendre_nodes, legendre_weights = np.polynomial.legendre.leggauss(n)
        grid = (aa + bb) / 2 + (bb - aa) * legendre_nodes / 2
        weights = (bb - aa) * legendre_weights / 2
        matrix = _matrix_eval(kernel_fn, grid, grid)
        forcing = _vector_eval(forcing_fn, grid, "forcing")
        system = np.eye(n) - lam * matrix * weights[None, :]
        condition = float(np.linalg.cond(system))
        if not math.isfinite(condition) or condition > 1e12:
            raise ArithmeticError(f"Nyström system is ill-conditioned: cond={condition:.3e}")
        try:
            values = np.linalg.solve(system, forcing)
        except np.linalg.LinAlgError as exc:
            raise ArithmeticError(f"Nyström system solve failed: {exc}") from exc
        residual = values - forcing - lam * matrix @ (weights * values)
        if not bool(np.all(np.isfinite(values))):
            raise ArithmeticError("Nyström solution is non-finite")
        return {
            "kind": "fredholm",
            "nodes": grid,
            "solution": values,
            "values": values,
            "weights": weights,
            "residual": residual,
            "residual_norm": float(np.max(np.abs(residual))),
            "condition_number": condition,
            "coupling": lam,
            "num_nodes": n,
        }

    def _solve_volterra() -> dict[str, object]:
        grid = np.linspace(aa, bb, n)
        step = (bb - aa) / (n - 1)
        matrix = _matrix_eval(kernel_fn, grid, grid)
        forcing = _vector_eval(forcing_fn, grid, "forcing")
        values = np.empty(n, dtype=float)
        values[0] = forcing[0]
        diagonal = np.ones(n, dtype=float)
        for i in range(1, n):
            diagonal[i] -= lam * step * 0.5 * matrix[i, i]
            if abs(diagonal[i]) <= 1e-14:
                raise ArithmeticError(f"Volterra Nyström pivot vanished at node {i}")
            previous = 0.5 * matrix[i, 0] * values[0]
            if i > 1:
                previous += float(matrix[i, 1:i] @ values[1:i])
            values[i] = (forcing[i] + lam * step * previous) / diagonal[i]
        residual = np.zeros(n, dtype=float)
        for i in range(1, n):
            integral = 0.5 * matrix[i, 0] * values[0]
            if i > 1:
                integral += float(matrix[i, 1:i] @ values[1:i])
            integral += 0.5 * matrix[i, i] * values[i]
            residual[i] = values[i] - forcing[i] - lam * step * integral
        condition = float(np.max(np.abs(diagonal)) / np.min(np.abs(diagonal)))
        if not bool(np.all(np.isfinite(values))):
            raise ArithmeticError("Nyström solution is non-finite")
        return {
            "kind": "volterra",
            "nodes": grid,
            "solution": values,
            "values": values,
            "weights": np.full(n, step, dtype=float),
            "residual": residual,
            "residual_norm": float(np.max(np.abs(residual))),
            "condition_number": condition,
            "coupling": lam,
            "num_nodes": n,
        }

    result = _run_bounded(_solve_fredholm if kind == "fredholm" else _solve_volterra, timeout)
    if not isinstance(result, dict):
        raise TypeError(f"Bounded solver returned {type(result).__name__}, expected dict")
    return result


def solve_nystrom_fredholm(
    kernel_str: str,
    forcing_str: str,
    a: float,
    b: float,
    nodes: int = 64,
    coupling: float = 1.0,
    x_var: str = "x",
    integration_var: str = "t",
    params: dict[str, float] | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
    parameter: float | None = None,
) -> dict[str, object]:
    """Solve a second-kind Fredholm equation by Gauss-Legendre Nyström."""
    return solve_nystrom(
        kernel_str, forcing_str, a, b, nodes, coupling, "fredholm",
        x_var, integration_var, params, timeout, parameter,
    )


def solve_nystrom_volterra(
    kernel_str: str,
    forcing_str: str,
    a: float,
    b: float,
    nodes: int = 128,
    coupling: float = 1.0,
    x_var: str = "x",
    integration_var: str = "t",
    params: dict[str, float] | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
    parameter: float | None = None,
) -> dict[str, object]:
    """Solve a second-kind Volterra equation by lower-triangular Nyström."""
    return solve_nystrom(
        kernel_str, forcing_str, a, b, nodes, coupling, "volterra",
        x_var, integration_var, params, timeout, parameter,
    )
