"""APEDP Extended evaluation: implicit roots, quadrature, ODE IVPs, PDE residuals.

The core evaluator handles explicit scalar ``y = f(x)``. This module covers
the next four classes without new dependencies (stdlib + sympy + numpy):

- implicit ``F(x, y) = 0`` solved for ``y`` via bounded ``nsolve``,
- unevaluated ``Integral`` via numeric quadrature (trapezoid on a lambdified
  integrand, with an ``evalf`` fallback),
- explicit ODE IVPs ``dy/dx = rhs(x, y)`` via fixed-step RK4,
- PDE residuals evaluated at points for a proposed closed-form solution
  (manufactured-solution check), plus an explicit 1D heat finite-difference
  solver for Dirichlet problems.

Every entry is wall-clock bounded and raises (never returns a silent
wrong number): ``ValueError`` for bad input, ``ArithmeticError`` for
non-finite/no-convergence results.
"""
from __future__ import annotations

import logging
import math
import threading

import numpy as np
import sympy as sp

from apedp.safety import safe_parse_expr, safe_parse_pde

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 10.0
_SEMAPHORE = threading.Semaphore(16)


def _run_bounded(fn, timeout: float = _DEFAULT_TIMEOUT):
    """Run ``fn()`` on a daemon thread; raise ``TimeoutError`` past budget."""
    box: dict[str, object] = {}
    failure: list[BaseException] = []

    def _worker() -> None:
        try:
            box["result"] = fn()
        except BaseException as exc:  # noqa: BLE001 - re-raised on caller thread
            failure.append(exc)
        finally:
            _SEMAPHORE.release()

    if not _SEMAPHORE.acquire(blocking=False):
        raise TimeoutError("too many extended-evaluation workers in flight")
    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        raise TimeoutError(f"evaluation exceeded {timeout}s budget")
    if failure:
        raise failure[0]
    return box.get("result")


def evaluate_implicit(
    formula_str: str,
    var: str,
    guess: float,
    params: dict[str, float] | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
) -> float:
    """Solve ``F(var, params) = 0`` for ``var`` from ``guess`` via nsolve."""
    expr = safe_parse_expr(formula_str)
    sym = sp.Symbol(var)
    if sym not in expr.free_symbols:
        raise ValueError(f"Variable '{var}' not in expression")
    subs = {sp.Symbol(k): float(v) for k, v in (params or {}).items()}
    f = expr.subs(subs)

    def _solve() -> float:
        try:
            root = sp.nsolve(f, sym, float(guess), tol=1e-14, maxsteps=200)
        except (ValueError, ArithmeticError, NotImplementedError) as e:
            raise ArithmeticError(f"nsolve failed for '{formula_str}': {e}") from e
        val = complex(root)
        if abs(val.imag) > 1e-9 or not math.isfinite(val.real):
            raise ArithmeticError(f"Non-finite/non-real root for '{formula_str}': {root}")
        return float(val.real)

    result = _run_bounded(_solve, timeout)
    assert isinstance(result, float)
    return result


def evaluate_quadrature(
    integrand_str: str,
    var: str,
    a: float,
    b: float,
    samples: int = 2001,
    timeout: float = _DEFAULT_TIMEOUT,
) -> float:
    """Numerically integrate ``integrand`` from ``a`` to ``b`` (trapezoid)."""
    if not math.isfinite(a) or not math.isfinite(b) or b <= a:
        raise ValueError(f"Need finite a < b, got {a}, {b}")
    if samples < 11:
        raise ValueError("Need samples >= 11")
    expr = safe_parse_expr(integrand_str)
    sym = sp.Symbol(var)

    def _quad() -> float:
        f = sp.lambdify(sym, expr, modules=["numpy"])
        xs = np.linspace(float(a), float(b), int(samples))
        try:
            vals = np.asarray(f(xs), dtype=float)
        except (ValueError, TypeError, ArithmeticError) as e:
            raise ArithmeticError(f"Integrand evaluation failed: {e}") from e
        if vals.shape != xs.shape:
            vals = np.full(xs.shape, float(vals), dtype=float)
        finite = np.isfinite(vals)
        if not bool(np.all(finite)):
            bad = int(np.sum(~finite))
            raise ArithmeticError(f"Integrand non-finite at {bad}/{len(vals)} points")
        trapz = getattr(np, "trapezoid", None) or np.trapz
        return float(trapz(vals, xs))

    result = _run_bounded(_quad, timeout)
    assert isinstance(result, float)
    return result


def evaluate_ode_ivp(
    rhs_str: str,
    y0: float,
    x_grid: list[float] | np.ndarray,
    params: dict[str, float] | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
) -> np.ndarray:
    """Solve ``dy/dx = rhs(x, y, params)``, ``y(x0) = y0`` via RK4 on ``x_grid``.

    ``x_grid`` must be strictly increasing and finite. Fixed step = grid
    spacing (no adaptivity by design: bounded work, no hidden solver state).
    """
    xs = np.asarray(list(x_grid), dtype=float)
    if xs.size < 2 or not bool(np.all(np.isfinite(xs))) or not bool(np.all(np.diff(xs) > 0)):
        raise ValueError("x_grid must be strictly increasing with >= 2 finite points")
    if not math.isfinite(y0):
        raise ValueError("y0 must be finite")
    expr = safe_parse_expr(rhs_str)
    x_sym, y_sym = sp.Symbol("x"), sp.Symbol("y")
    subs = {sp.Symbol(k): float(v) for k, v in (params or {}).items()}

    def _integrate() -> np.ndarray:
        f = sp.lambdify((x_sym, y_sym), expr.subs(subs), modules=["numpy"])
        ys = np.empty_like(xs)
        ys[0] = float(y0)
        for i in range(1, len(xs)):
            h = float(xs[i] - xs[i - 1])
            xi, yi = float(xs[i - 1]), float(ys[i - 1])
            try:
                k1 = float(f(xi, yi))
                k2 = float(f(xi + h / 2, yi + h * k1 / 2))
                k3 = float(f(xi + h / 2, yi + h * k2 / 2))
                k4 = float(f(xi + h, yi + h * k3))
            except (ValueError, TypeError, ArithmeticError) as e:
                raise ArithmeticError(f"RHS evaluation failed at step {i}: {e}") from e
            nxt = yi + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            if not math.isfinite(nxt):
                raise ArithmeticError(f"ODE went non-finite at step {i} (x={xs[i]})")
            ys[i] = nxt
        return ys

    result = _run_bounded(_integrate, timeout)
    assert isinstance(result, np.ndarray)
    return result


def evaluate_pde_residual(
    pde_str: str,
    candidate_str: str,
    points: dict[str, list[float] | np.ndarray],
    params: dict[str, float] | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
) -> dict[str, object]:
    """Evaluate a PDE residual (``= 0`` form) at ``points`` for a candidate solution.

    Example: pde ``Derivative(T(x,t),t) - D*Derivative(T(x,t),x,2)`` with
    candidate ``exp(-D*t)*sin(x)``, ``points={'x': [...], 't': [...]}``,
    ``params={'D': 0.1}`` returns max/mean absolute residual. Small residual
    means the candidate satisfies the PDE at those points; it proves nothing
    about BCs or uniqueness.
    """
    pde = safe_parse_pde(pde_str)
    derivs = pde.atoms(sp.Derivative)
    if not derivs:
        raise ValueError("No Derivative nodes: not a PDE residual")
    funcs = set()
    for d in derivs:
        try:
            funcs.add(d.args[0].func.__name__)
        except (AttributeError, TypeError):
            funcs.add("?")
    if len(funcs) != 1:
        raise ValueError(f"Need exactly one undefined function, found {sorted(funcs)}")
    cand = safe_parse_expr(candidate_str)
    consts = {sp.Symbol(k): float(v) for k, v in (params or {}).items()}
    pde_c = pde.subs(consts)
    cand_c = cand.subs(consts)
    calls = {d.args[0] for d in pde_c.atoms(sp.Derivative)}
    coord_syms = sorted({s for call in calls for s in call.free_symbols}, key=str)
    if not coord_syms:
        raise ValueError("PDE calls carry no coordinates to evaluate on")
    names = [str(s) for s in coord_syms]
    missing = [n for n in names if n not in points]
    if missing:
        raise ValueError(f"Missing points for coordinates: {missing}")
    grids = np.meshgrid(*[np.asarray(points[n], dtype=float) for n in names], indexing="ij")
    flat = [g.ravel() for g in grids]

    def _residual() -> dict[str, object]:
        mapping = {call: cand_c for call in calls}
        residual = pde_c.xreplace(mapping)
        if residual.has(sp.Derivative):
            residual = residual.doit()
        rfunc = sp.lambdify(coord_syms, residual, modules=["numpy"])
        try:
            vals = np.asarray(rfunc(*flat), dtype=float).ravel()
        except (ValueError, TypeError, ArithmeticError) as e:
            raise ArithmeticError(f"Residual evaluation failed: {e}") from e
        total = int(flat[0].size) if flat else int(vals.size)
        if vals.size == 1 and total > 1:
            # Residual simplified to a constant (e.g. exact 0): broadcast.
            vals = np.full(total, float(vals.ravel()[0]), dtype=float)
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            raise ArithmeticError("Residual non-finite at all points")
        return {
            "max_abs_residual": float(np.max(np.abs(finite))),
            "mean_abs_residual": float(np.mean(np.abs(finite))),
            "num_points": int(finite.size),
            "num_nonfinite": int(vals.size - finite.size),
        }

    result = _run_bounded(_residual, timeout)
    assert isinstance(result, dict)
    return result


def solve_heat_1d_explicit(
    length: float,
    total_time: float,
    nx: int = 51,
    nt: int = 500,
    diffusivity: float = 1.0,
    left_bc: float = 0.0,
    right_bc: float = 0.0,
    initial: str = "sin(pi*x/L)",
    timeout: float = _DEFAULT_TIMEOUT,
) -> dict[str, object]:
    """Explicit finite-difference solve of ``u_t = D*u_xx`` on ``[0, L]``.

    Dirichlet BCs, sinusoidal-or-constant initial profile. Stability is
    enforced (``r = D*dt/dx^2 <= 0.5``, raises otherwise) so the scheme
    cannot silently blow up: violating inputs are refused, not clamped.
    """
    if not (math.isfinite(length) and length > 0):
        raise ValueError("Need finite length > 0")
    if not (math.isfinite(total_time) and total_time > 0):
        raise ValueError("Need finite total_time > 0")
    if nx < 5 or nt < 2:
        raise ValueError("Need nx >= 5, nt >= 2")
    if not (math.isfinite(diffusivity) and diffusivity > 0):
        raise ValueError("Need finite diffusivity > 0")
    dx = float(length) / (int(nx) - 1)
    dt = float(total_time) / int(nt)
    r = float(diffusivity) * dt / dx**2
    if r > 0.5:
        raise ValueError(f"Unstable grid r={r:.3f} > 0.5: raise nx or lower nt/total_time")

    def _solve() -> dict[str, object]:
        x = np.linspace(0.0, float(length), int(nx))
        L = sp.Symbol("L")
        u = np.zeros(int(nx))
        if initial == "sin(pi*x/L)":
            u = np.sin(np.pi * x / float(length))
        elif initial == "constant":
            u = np.ones(int(nx))
        else:
            prof = safe_parse_expr(initial)
            xsym = sp.Symbol("x")
            f = sp.lambdify(xsym, prof.subs({L: float(length)}), modules=["numpy"])
            try:
                u = np.asarray(f(x), dtype=float)
            except (ValueError, TypeError, ArithmeticError) as e:
                raise ValueError(f"Bad initial profile: {e}") from e
            if u.shape != x.shape:
                raise ValueError("Initial profile shape mismatch")
        u[0], u[-1] = float(left_bc), float(right_bc)
        for _ in range(int(nt)):
            nxt = u.copy()
            nxt[1:-1] = u[1:-1] + r * (u[2:] - 2 * u[1:-1] + u[:-2])
            nxt[0], nxt[-1] = float(left_bc), float(right_bc)
            if not bool(np.all(np.isfinite(nxt))):
                raise ArithmeticError("Heat solve went non-finite")
            u = nxt
        return {"x": x, "u": u, "dx": dx, "dt": dt, "r": r}

    result = _run_bounded(_solve, timeout)
    assert isinstance(result, dict)
    return result
