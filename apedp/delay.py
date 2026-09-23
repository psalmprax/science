"""Bounded delay-differential equation solvers via the method of steps.

The public solvers accept the same safe expression language as the existing ODE
suite. A delayed state is evaluated only from the prescribed history or from
already-computed solution values; the internal step is capped below the delay so
RK4 never reads an unknown future state.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np
import sympy as sp

from apedp.eval_extended import _run_bounded
from apedp.safety import safe_parse_expr

_DEFAULT_TIMEOUT = 10.0
HistorySpec = float | str | Callable[[float], float]


def _validate_grid(x_grid: Sequence[float] | np.ndarray) -> np.ndarray:
    xs = np.asarray(list(x_grid), dtype=float)
    if xs.size < 2 or not bool(np.all(np.isfinite(xs))) or not bool(np.all(np.diff(xs) > 0)):
        raise ValueError("x_grid must be strictly increasing with >= 2 finite points")
    return xs


def _validate_params(params: dict[str, float] | None, reserved: set[str]) -> dict[sp.Symbol, float]:
    out: dict[sp.Symbol, float] = {}
    for name, value in (params or {}).items():
        if name in reserved:
            raise ValueError(f"Parameter '{name}' is reserved by the DDE state")
        if not math.isfinite(float(value)):
            raise ValueError(f"Parameter '{name}' must be finite")
        out[sp.Symbol(name)] = float(value)
    return out


def _prepare_expr(
    formula: str,
    params: dict[sp.Symbol, float],
    allowed: set[sp.Symbol],
) -> sp.Expr:
    expr = safe_parse_expr(formula).subs(params)
    unknown = expr.free_symbols - allowed
    if unknown:
        names = ", ".join(sorted(str(s) for s in unknown))
        raise ValueError(f"Unknown DDE symbols: {names}")
    return expr


def _history_function(
    spec: HistorySpec,
    x_sym: sp.Symbol,
    params: dict[sp.Symbol, float],
) -> Callable[[float], float]:
    if callable(spec):
        def _callable(value: float) -> float:
            result = float(spec(float(value)))
            if not math.isfinite(result):
                raise ArithmeticError(f"History is non-finite at x={value}")
            return result

        return _callable
    if isinstance(spec, (int, float, np.number)):
        result = float(spec)
        if not math.isfinite(result):
            raise ValueError("History value must be finite")
        return lambda _value: result
    expr = safe_parse_expr(str(spec)).subs(params)
    unknown = expr.free_symbols - {x_sym}
    if unknown:
        names = ", ".join(sorted(str(s) for s in unknown))
        raise ValueError(f"Unknown history symbols: {names}")
    fn = sp.lambdify(x_sym, expr, modules=["numpy"])

    def _evaluate(value: float) -> float:
        try:
            result = float(fn(float(value)))
        except (TypeError, ValueError, ArithmeticError) as exc:
            raise ArithmeticError(f"History evaluation failed at x={value}: {exc}") from exc
        if not math.isfinite(result):
            raise ArithmeticError(f"History is non-finite at x={value}")
        return result

    return _evaluate


def _history_specs(history: HistorySpec | Sequence[HistorySpec], n: int) -> list[HistorySpec]:
    if isinstance(history, (str, int, float, np.number)) or callable(history):
        return [history] * n
    specs = list(history)
    if len(specs) != n:
        raise ValueError("Need one history specification per DDE component")
    return specs


def evaluate_delay_ivp(
    rhs_str: str,
    y0: float,
    x_grid: Sequence[float] | np.ndarray,
    delay: float,
    history: HistorySpec = 0.0,
    params: dict[str, float] | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
    max_step: float | None = None,
    max_steps: int = 200_000,
) -> np.ndarray:
    """Solve ``y' = rhs(x, y, yd)`` with ``yd = y(x-delay)`` by method of steps.

    ``history`` supplies ``y(t)`` for ``t < x_grid[0]`` and may be a finite
    constant, a safe expression in ``x``, or a numeric callable. The internal
    RK4 step is at most half the delay, so every delayed stage is known from the
    history or completed solution. The returned array is sampled on ``x_grid``.
    """
    if not math.isfinite(float(delay)) or delay <= 0:
        raise ValueError("delay must be finite and > 0")
    if not math.isfinite(float(timeout)) or timeout <= 0:
        raise ValueError("timeout must be finite and > 0")
    if int(max_steps) < 1:
        raise ValueError("max_steps must be positive")
    if not math.isfinite(float(y0)):
        raise ValueError("y0 must be finite")
    xs = _validate_grid(x_grid)
    x_sym, y_sym, yd_sym = sp.Symbol("x"), sp.Symbol("y"), sp.Symbol("yd")
    subs = _validate_params(params, {"x", "y", "yd"})
    expr = _prepare_expr(rhs_str, subs, {x_sym, y_sym, yd_sym})
    history_fn = _history_function(history, x_sym, subs)
    if max_step is not None and (not math.isfinite(float(max_step)) or float(max_step) <= 0):
        raise ValueError("max_step must be finite and > 0")
    step_limit = min(float(max_step) if max_step is not None else float(delay) / 4.0, float(delay) / 2.0)
    rhs = sp.lambdify((x_sym, y_sym, yd_sym), expr, modules=["numpy"])

    def _integrate() -> np.ndarray:
        out = np.empty(xs.size, dtype=float)
        out[0] = float(y0)
        times = [float(xs[0])]
        values = [float(y0)]
        current = float(xs[0])
        steps = 0
        tol = 1e-12 * max(1.0, abs(current), abs(float(xs[-1])))

        def delayed_value(value: float) -> float:
            target = float(value) - float(delay)
            if target < float(xs[0]) - tol:
                return history_fn(target)
            if abs(target - float(xs[0])) <= tol:
                return float(y0)
            if target > times[-1] + tol:
                raise ArithmeticError("method-of-steps requested an unknown delayed value")
            return float(np.interp(target, np.asarray(times), np.asarray(values)))

        output_index = 1
        while output_index < xs.size:
            endpoint = float(xs[output_index])
            while current < endpoint - tol:
                h = min(step_limit, endpoint - current)
                yi = float(values[-1])
                try:
                    k1 = float(rhs(current, yi, delayed_value(current)))
                    k2 = float(rhs(current + h / 2, yi + h * k1 / 2, delayed_value(current + h / 2)))
                    k3 = float(rhs(current + h / 2, yi + h * k2 / 2, delayed_value(current + h / 2)))
                    k4 = float(rhs(current + h, yi + h * k3, delayed_value(current + h)))
                except (TypeError, ValueError, ArithmeticError) as exc:
                    raise ArithmeticError(f"DDE RHS evaluation failed near x={current}: {exc}") from exc
                nxt = yi + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
                if not math.isfinite(nxt):
                    raise ArithmeticError(f"DDE went non-finite near x={current}")
                current += h
                if abs(current - endpoint) <= tol:
                    current = endpoint
                times.append(current)
                values.append(float(nxt))
                steps += 1
                if steps > int(max_steps):
                    raise ArithmeticError("DDE exceeded max_steps; reduce the interval or increase max_steps")
            out[output_index] = values[-1]
            output_index += 1
        if not bool(np.all(np.isfinite(out))):
            raise ArithmeticError("DDE returned non-finite values")
        return out

    result = _run_bounded(_integrate, timeout)
    if not isinstance(result, np.ndarray):
        raise TypeError(f"Bounded DDE integration returned {type(result).__name__}, expected ndarray")
    return result


def evaluate_delay_system(
    rhs_list: list[str],
    y0_list: list[float],
    x_grid: Sequence[float] | np.ndarray,
    delay: float,
    history: HistorySpec | Sequence[HistorySpec] = 0.0,
    var_names: list[str] | None = None,
    params: dict[str, float] | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
    max_step: float | None = None,
    max_steps: int = 200_000,
) -> np.ndarray:
    """Solve a vector DDE ``Y' = F(x, Y, Y(x-delay))`` by method of steps.

    State variables default to ``y1..yn`` and delayed variables to ``d1..dn``.
    Each RHS may reference the current and delayed components, so coupled delay
    systems use the same bounded RK4/history contract as the scalar solver.
    """
    if not math.isfinite(float(delay)) or delay <= 0:
        raise ValueError("delay must be finite and > 0")
    if not math.isfinite(float(timeout)) or timeout <= 0:
        raise ValueError("timeout must be finite and > 0")
    if int(max_steps) < 1:
        raise ValueError("max_steps must be positive")
    n = len(rhs_list)
    if n < 1 or len(y0_list) != n:
        raise ValueError("Need at least one equation and matching y0_list")
    names = list(var_names) if var_names is not None else [f"y{k + 1}" for k in range(n)]
    if len(names) != n or len(set(names)) != n or "x" in names:
        raise ValueError("var_names must be n distinct names and must not contain 'x'")
    delayed_names = [f"d{k + 1}" for k in range(n)]
    if set(names) & set(delayed_names):
        raise ValueError("var_names collide with delayed state names")
    y0 = np.asarray(list(y0_list), dtype=float)
    if not bool(np.all(np.isfinite(y0))):
        raise ValueError("y0_list must be finite")
    xs = _validate_grid(x_grid)
    x_sym = sp.Symbol("x")
    state_syms = [sp.Symbol(name) for name in names]
    delayed_syms = [sp.Symbol(name) for name in delayed_names]
    reserved = {"x", *names, *delayed_names}
    subs = _validate_params(params, reserved)
    exprs = [_prepare_expr(text, subs, {x_sym, *state_syms, *delayed_syms}) for text in rhs_list]
    history_fns = [_history_function(spec, x_sym, subs) for spec in _history_specs(history, n)]
    if max_step is not None and (not math.isfinite(float(max_step)) or float(max_step) <= 0):
        raise ValueError("max_step must be finite and > 0")
    step_limit = min(float(max_step) if max_step is not None else float(delay) / 4.0, float(delay) / 2.0)
    rhs = sp.lambdify([x_sym, *state_syms, *delayed_syms], exprs, modules=["numpy"])

    def _integrate() -> np.ndarray:
        out = np.empty((xs.size, n), dtype=float)
        out[0] = y0
        times = [float(xs[0])]
        values = [y0.copy()]
        current = float(xs[0])
        steps = 0
        tol = 1e-12 * max(1.0, abs(current), abs(float(xs[-1])))

        def delayed_value(value: float) -> np.ndarray:
            target = float(value) - float(delay)
            if target < float(xs[0]) - tol:
                return np.asarray([fn(target) for fn in history_fns], dtype=float)
            if abs(target - float(xs[0])) <= tol:
                return y0.copy()
            if target > times[-1] + tol:
                raise ArithmeticError("method-of-steps requested an unknown delayed state")
            ts = np.asarray(times)
            vs = np.asarray(values)
            return np.asarray([np.interp(target, ts, vs[:, j]) for j in range(n)], dtype=float)

        def eval_rhs(at: float, state: np.ndarray) -> np.ndarray:
            try:
                result = np.asarray(rhs(at, *state, *delayed_value(at)), dtype=float).reshape(-1)
            except (TypeError, ValueError, ArithmeticError) as exc:
                raise ArithmeticError(f"DDE system RHS evaluation failed near x={at}: {exc}") from exc
            if result.shape != (n,) or not bool(np.all(np.isfinite(result))):
                raise ArithmeticError(f"DDE system RHS must return {n} finite components")
            return result

        output_index = 1
        while output_index < xs.size:
            endpoint = float(xs[output_index])
            while current < endpoint - tol:
                h = min(step_limit, endpoint - current)
                yi = values[-1]
                k1 = eval_rhs(current, yi)
                k2 = eval_rhs(current + h / 2, yi + h * k1 / 2)
                k3 = eval_rhs(current + h / 2, yi + h * k2 / 2)
                k4 = eval_rhs(current + h, yi + h * k3)
                nxt = yi + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
                if not bool(np.all(np.isfinite(nxt))):
                    raise ArithmeticError(f"DDE system went non-finite near x={current}")
                current += h
                if abs(current - endpoint) <= tol:
                    current = endpoint
                times.append(current)
                values.append(nxt.copy())
                steps += 1
                if steps > int(max_steps):
                    raise ArithmeticError("DDE exceeded max_steps; reduce the interval or increase max_steps")
            out[output_index] = values[-1]
            output_index += 1
        return out

    result = _run_bounded(_integrate, timeout)
    if not isinstance(result, np.ndarray):
        raise TypeError(f"Bounded DDE integration returned {type(result).__name__}, expected ndarray")
    return result


solve_delay_ivp = evaluate_delay_ivp
solve_delay_system = evaluate_delay_system
