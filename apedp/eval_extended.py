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


def evaluate_implicit_system(
    formulas: list[str],
    variables: list[str],
    guesses: list[float],
    params: dict[str, float] | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
) -> dict[str, float]:
    """Solve a square system ``F_i(vars, params) = 0`` via bounded nsolve.

    Returns ``{var: value}`` with all-real finite values, else raises
    ``ArithmeticError`` (no convergence / non-real branch). The max
    equation residual is stored under ``"residual"`` for the caller to judge.
    """
    if len(formulas) != len(variables) or len(variables) != len(guesses):
        raise ValueError("Need len(formulas) == len(variables) == len(guesses)")
    if not variables:
        raise ValueError("Need at least one variable")
    exprs = [safe_parse_expr(f) for f in formulas]
    syms = [sp.Symbol(v) for v in variables]
    subs = {sp.Symbol(k): float(v) for k, v in (params or {}).items()}
    fs = [e.subs(subs) for e in exprs]
    for s, e in zip(syms, fs):
        if s not in e.free_symbols:
            raise ValueError(f"Variable '{s}' not in its equation")

    def _solve() -> dict[str, float]:
        try:
            roots = sp.nsolve(fs, syms, [float(g) for g in guesses], tol=1e-14, maxsteps=200)
        except (ValueError, ArithmeticError, NotImplementedError) as e:
            raise ArithmeticError(f"nsolve failed for system: {e}") from e
        out: dict[str, float] = {}
        for s, r in zip(syms, roots):
            val = complex(r)
            if abs(val.imag) > 1e-9 or not math.isfinite(val.real):
                raise ArithmeticError(f"Non-finite/non-real root for '{s}': {r}")
            out[str(s)] = float(val.real)
        resids = [abs(complex(e.subs({s: out[str(s)] for s in syms}))) for e in fs]
        out["residual"] = float(max(float(r) for r in resids)) if resids else 0.0
        return out

    result = _run_bounded(_solve, timeout)
    assert isinstance(result, dict)
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
    method: str = "rk4",
    rtol: float = 1e-6,
    atol: float = 1e-9,
) -> np.ndarray:
    """Solve ``dy/dx = rhs(x, y, params)``, ``y(x0) = y0``.

    ``method="rk4"`` steps the grid with fixed-step RK4 (legacy default).
    ``method="rk45"`` uses adaptive Dormand-Prince with ``rtol``/``atol``
    error control, dense output interpolated back onto ``x_grid``. Adaptive
    stepping refuses (raises) when the error cannot be met within 100k
    accepted-plus-rejected steps — stiff systems that need implicit stepping
    fail loudly here instead of returning a wrong smooth curve.
    ``method="backward_euler"`` solves each step implicitly
    (``y_{n+1} = y_n + h·f(x_{n+1}, y_{n+1})`` via nsolve): first-order but
    L-stable, so stiff decays like ``y' = -50·y`` stay bounded where explicit
    schemes explode. Slow per step (one nsolve each) — use it for stiffness,
    not speed.

    ``x_grid`` must be strictly increasing and finite.
    """
    if method not in ("rk4", "rk45", "backward_euler"):
        raise ValueError("method must be 'rk4', 'rk45' or 'backward_euler'")
    xs = np.asarray(list(x_grid), dtype=float)
    if xs.size < 2 or not bool(np.all(np.isfinite(xs))) or not bool(np.all(np.diff(xs) > 0)):
        raise ValueError("x_grid must be strictly increasing with >= 2 finite points")
    if not math.isfinite(y0):
        raise ValueError("y0 must be finite")
    expr = safe_parse_expr(rhs_str)
    x_sym, y_sym = sp.Symbol("x"), sp.Symbol("y")
    subs = {sp.Symbol(k): float(v) for k, v in (params or {}).items()}

    def _rk4_step(f, xi: float, yi: float, h: float) -> float:
        k1 = float(f(xi, yi))
        k2 = float(f(xi + h / 2, yi + h * k1 / 2))
        k3 = float(f(xi + h / 2, yi + h * k2 / 2))
        k4 = float(f(xi + h, yi + h * k3))
        return yi + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    # Dormand-Prince 5(4) coefficients.
    _DP_A = (
        (1 / 5,),
        (3 / 40, 9 / 40),
        (44 / 45, -56 / 15, 32 / 9),
        (19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729),
        (9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656),
        (35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84),
    )
    _DP_B5 = (35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0.0)
    _DP_B4 = (5179 / 57600, 0.0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40)

    def _rk45_dense(f, xs: np.ndarray, y0: float) -> np.ndarray:
        ys = np.empty_like(xs)
        ys[0] = float(y0)
        xi, yi = float(xs[0]), float(y0)
        h = float(min(xs[1] - xs[0], (xs[-1] - xs[0]) / 10 or 1e-6))
        target = 1
        steps = 0
        while target < len(xs):
            if steps > 100000:
                raise ArithmeticError("RK45 exceeded 100k steps: system likely stiff; refusing")
            h = min(h, float(xs[target] - xi))
            if h <= 0:
                ys[target] = yi
                xi = float(xs[target])
                target += 1
                continue
            try:
                k1 = float(f(xi, yi))
                k2 = float(f(xi + h * 1 / 5, yi + h * (_DP_A[0][0] * k1)))
                k3 = float(f(xi + h * 3 / 10, yi + h * (_DP_A[1][0] * k1 + _DP_A[1][1] * k2)))
                k4 = float(f(xi + h * 4 / 5, yi + h * (_DP_A[2][0] * k1 + _DP_A[2][1] * k2 + _DP_A[2][2] * k3)))
                k5 = float(f(xi + h * 8 / 9, yi + h * (_DP_A[3][0] * k1 + _DP_A[3][1] * k2 + _DP_A[3][2] * k3 + _DP_A[3][3] * k4)))
                k6 = float(f(xi + h, yi + h * (_DP_A[4][0] * k1 + _DP_A[4][1] * k2 + _DP_A[4][2] * k3 + _DP_A[4][3] * k4 + _DP_A[4][4] * k5)))
                y5 = yi + h * (_DP_B5[0] * k1 + _DP_B5[2] * k3 + _DP_B5[3] * k4 + _DP_B5[4] * k5 + _DP_B5[5] * k6)
                k7 = float(f(xi + h, y5))
                y4 = yi + h * (_DP_B4[0] * k1 + _DP_B4[2] * k3 + _DP_B4[3] * k4 + _DP_B4[4] * k5 + _DP_B4[5] * k6 + _DP_B4[6] * k7)
            except (ValueError, TypeError, ArithmeticError) as e:
                raise ArithmeticError(f"RHS evaluation failed near x={xi}: {e}") from e
            if not (math.isfinite(y5) and math.isfinite(y4)):
                raise ArithmeticError(f"ODE went non-finite near x={xi}")
            scale = float(atol) + float(rtol) * max(abs(yi), abs(y5))
            err = abs(y5 - y4) / scale if scale > 0 else 0.0
            if err <= 1.0:
                xi += h
                # Linear dense output: accept then fill reached grid points.
                while target < len(xs) and xs[target] <= xi + 1e-15:
                    frac = (float(xs[target]) - (xi - h)) / h if h > 0 else 0.0
                    ys[target] = (yi + frac * (y5 - yi)) if 0.0 <= frac <= 1.0 else y5
                    target += 1
                yi = y5
                steps += 1
            h *= min(5.0, max(0.2, 0.9 * (1.0 / err) ** 0.2 if err > 0 else 5.0))
            steps += 1
        return ys

    def _backward_euler(f, xs: np.ndarray, y0: float) -> np.ndarray:
        """L-stable implicit stepping: each step solves y-yi-h·f(xi+h,y) = 0."""
        ys = np.empty_like(xs)
        ys[0] = float(y0)
        yy = sp.Symbol("_be_y")
        for i in range(1, len(xs)):
            h = float(xs[i] - xs[i - 1])
            xi, yi = float(xs[i - 1]), float(ys[i - 1])
            try:
                fi = float(f(xi, yi))
            except (ValueError, TypeError, ArithmeticError) as e:
                raise ArithmeticError(f"RHS evaluation failed at step {i}: {e}") from e
            if not math.isfinite(fi):
                raise ArithmeticError(f"RHS non-finite at step {i} (x={xi})")
            residual = yy - yi - h * expr.subs(subs).subs({x_sym: xs[i], y_sym: yy})
            try:
                sol = sp.nsolve(residual, yy, yi + h * fi, tol=1e-14, maxsteps=200)
            except (ValueError, ArithmeticError, NotImplementedError) as e:
                raise ArithmeticError(f"Backward-Euler nsolve failed at step {i}: {e}") from e
            nxt = complex(sol)
            if abs(nxt.imag) > 1e-9 or not math.isfinite(nxt.real):
                raise ArithmeticError(f"Backward-Euler non-real root at step {i}: {sol}")
            ys[i] = float(nxt.real)
        return ys

    def _integrate() -> np.ndarray:
        f = sp.lambdify((x_sym, y_sym), expr.subs(subs), modules=["numpy"])
        if method == "rk45":
            return _rk45_dense(f, xs, float(y0))
        if method == "backward_euler":
            return _backward_euler(f, xs, float(y0))
        ys = np.empty_like(xs)
        ys[0] = float(y0)
        for i in range(1, len(xs)):
            h = float(xs[i] - xs[i - 1])
            xi, yi = float(xs[i - 1]), float(ys[i - 1])
            try:
                nxt = _rk4_step(f, xi, yi, h)
            except (ValueError, TypeError, ArithmeticError) as e:
                raise ArithmeticError(f"RHS evaluation failed at step {i}: {e}") from e
            if not math.isfinite(nxt):
                raise ArithmeticError(f"ODE went non-finite at step {i} (x={xs[i]})")
            ys[i] = nxt
        return ys

    result = _run_bounded(_integrate, timeout)
    assert isinstance(result, np.ndarray)
    return result


def evaluate_ode_system(
    rhs_list: list[str],
    y0_list: list[float],
    x_grid: list[float] | np.ndarray,
    var_names: list[str] | None = None,
    params: dict[str, float] | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
    method: str = "rk4",
) -> np.ndarray:
    """Solve a square ODE system ``dY/dx = F(x, Y, params)``, ``Y(x0) = y0``.

    ``rhs_list[k]`` is the RHS for ``var_names[k]`` (default ``y1..yn``) as a
    ``safe_parse_expr`` string in ``x`` + state vars + params. ``method="rk4"``
    is fixed-step vector RK4; ``method="backward_euler"`` solves each step
    implicitly via nsolve (L-stable for stiff systems; refuses on
    non-convergence instead of exploding). Returns shape ``(len(grid), n)``.
    """
    if method not in ("rk4", "backward_euler"):
        raise ValueError("method must be 'rk4' or 'backward_euler'")
    n = len(rhs_list)
    if n < 1:
        raise ValueError("Need at least one equation")
    if len(y0_list) != n:
        raise ValueError("Need len(y0_list) == len(rhs_list)")
    names = list(var_names) if var_names else [f"y{k + 1}" for k in range(n)]
    if len(names) != n or len(set(names)) != n:
        raise ValueError("var_names must be n distinct names")
    if "x" in names:
        raise ValueError("var_names must not contain 'x'")
    xs = np.asarray(list(x_grid), dtype=float)
    if xs.size < 2 or not bool(np.all(np.isfinite(xs))) or not bool(np.all(np.diff(xs) > 0)):
        raise ValueError("x_grid must be strictly increasing with >= 2 finite points")
    y0 = np.asarray(list(y0_list), dtype=float)
    if not bool(np.all(np.isfinite(y0))):
        raise ValueError("y0_list must be finite")
    exprs = [safe_parse_expr(r) for r in rhs_list]
    x_sym = sp.Symbol("x")
    syms = [sp.Symbol(v) for v in names]
    subs = {sp.Symbol(k): float(v) for k, v in (params or {}).items()}

    def _integrate() -> np.ndarray:
        f = sp.lambdify([x_sym, *syms], [e.subs(subs) for e in exprs], modules=["numpy"])
        ys = np.empty((len(xs), n))
        ys[0] = y0
        if method == "rk4":
            for i in range(1, len(xs)):
                h = float(xs[i] - xs[i - 1])
                xi, yi = float(xs[i - 1]), ys[i - 1]
                try:
                    k1 = np.asarray(f(xi, *yi), dtype=float).ravel()
                    k2 = np.asarray(f(xi + h / 2, *(yi + h * k1 / 2)), dtype=float).ravel()
                    k3 = np.asarray(f(xi + h / 2, *(yi + h * k2 / 2)), dtype=float).ravel()
                    k4 = np.asarray(f(xi + h, *(yi + h * k3)), dtype=float).ravel()
                except (ValueError, TypeError, ArithmeticError) as e:
                    raise ArithmeticError(f"RHS evaluation failed at step {i}: {e}") from e
                if any(v.shape != (n,) for v in (k1, k2, k3, k4)):
                    raise ArithmeticError(f"RHS must return {n} components at step {i}")
                nxt = yi + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
                if not bool(np.all(np.isfinite(nxt))):
                    raise ArithmeticError(f"ODE system went non-finite at step {i}")
                ys[i] = nxt
            return ys
        yy = [sp.Symbol(f"_be{i}") for i in range(n)]
        for i in range(1, len(xs)):
            h = float(xs[i] - xs[i - 1])
            xi, yi = float(xs[i - 1]), ys[i - 1]
            try:
                fwd = np.asarray(f(xi, *yi), dtype=float).ravel()
            except (ValueError, TypeError, ArithmeticError) as e:
                raise ArithmeticError(f"RHS evaluation failed at step {i}: {e}") from e
            if fwd.shape != (n,):
                raise ArithmeticError(f"RHS must return {n} components at step {i}")
            guess = yi + h * fwd
            eqs = [
                yy[k] - yi[k] - h * exprs[k].subs(subs).subs({x_sym: float(xs[i])})
                for k in range(n)
            ]
            eqs = [e.subs(dict(zip(syms, yy))) for e in eqs]
            try:
                sol = sp.nsolve(eqs, yy, list(guess), tol=1e-14, maxsteps=200)
            except (ValueError, ArithmeticError, NotImplementedError) as e:
                raise ArithmeticError(f"Backward-Euler nsolve failed at step {i}: {e}") from e
            nxt = np.asarray([complex(v) for v in sol])
            if bool(np.any(np.abs(nxt.imag) > 1e-9)) or not bool(np.all(np.isfinite(nxt.real))):
                raise ArithmeticError(f"Backward-Euler non-real root at step {i}")
            ys[i] = nxt.real.astype(float)
        return ys

    result = _run_bounded(_integrate, timeout)
    assert isinstance(result, np.ndarray)
    return result


def evaluate_einsum(
    subscripts: str,
    *operands: list | np.ndarray,
    timeout: float = _DEFAULT_TIMEOUT,
) -> np.ndarray:
    """Validated ``numpy.einsum``: shape-checked tensor contraction.

    Rejects subscript/operand-count mismatches and non-finite results instead
    of broadcasting silently. Use :func:`evaluate_indexed` for Einstein
    index-notation strings; unstructured meshes remain out of scope only for
    mesh generation (see ``solve_poisson_fem`` for user-supplied meshes).
    """
    arrays = []
    for op in operands:
        try:
            arrays.append(np.asarray(op, dtype=float))
        except (ValueError, TypeError) as e:
            raise ValueError(f"Operand not array-like: {e}") from e
    if not arrays:
        raise ValueError("Need at least one operand")

    def _contract() -> np.ndarray:
        try:
            out = np.einsum(subscripts, *arrays)
        except (ValueError, TypeError) as e:
            raise ValueError(f"einsum failed for '{subscripts}': {e}") from e
        out = np.asarray(out, dtype=float)
        if not bool(np.all(np.isfinite(out))):
            raise ArithmeticError("Contraction produced non-finite entries")
        return out

    result = _run_bounded(_contract, timeout)
    assert isinstance(result, np.ndarray)
    return result


def evaluate_indexed(
    formula: str,
    arrays: dict[str, list | np.ndarray],
    timeout: float = _DEFAULT_TIMEOUT,
) -> np.ndarray:
    """Evaluate an explicit Einstein index-notation equation.

    Form: ``"C[i,j] = A[i,k] * B[k,j]"`` — one output tensor, RHS a sum of
    products of indexed tensors with optional scalar coefficients
    (``"y[i] = 2*A[i,j]*x[j] - b[i]"``); a bare ``Name`` LHS takes a scalar
    (``"t = A[i,i]"``). Single-letter indices, validated per term: a free
    index appears exactly once in each term, any other index exactly twice
    (trace ``A[i,i]`` counts twice). Anything else — differentials, function
    calls, bare unindexed symbols on the RHS — raises ``ValueError``.
    """
    import re

    if formula.count("=") != 1:
        raise ValueError("Need exactly one '=' in an indexed equation")
    lhs_raw, rhs_raw = (part.strip() for part in formula.split("="))
    lhs = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)(?:\[([A-Za-z](?:,[A-Za-z])*)\])?", lhs_raw)
    if not lhs:
        raise ValueError(f"Bad LHS '{lhs_raw}': want Name or Name[i,j,...]")
    out_name = lhs.group(1)
    out_idx = lhs.group(2).split(",") if lhs.group(2) else []
    if len(set(out_idx)) != len(out_idx):
        raise ValueError(f"Duplicate output index in '{lhs_raw}'")
    rhs = rhs_raw.replace("-", "+-")
    terms = [t.strip() for t in rhs.split("+") if t.strip()]
    if not terms:
        raise ValueError("Empty right-hand side")
    factor_re = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\[([A-Za-z](?:,[A-Za-z])*)\]")
    num_re = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\Z")
    try:
        operands = {k: np.asarray(v, dtype=float) for k, v in arrays.items()}
    except (ValueError, TypeError) as e:
        raise ValueError(f"Array not array-like: {e}") from e
    parsed: list[tuple[float, list[str], list[np.ndarray]]] = []
    for term in terms:
        term_sign = 1.0
        term = term.strip()
        if term.startswith("-"):
            term_sign = -1.0
            term = term[1:].strip()
        parts = [p.strip() for p in term.split("*") if p.strip()]
        coeff = term_sign
        subs: list[str] = []
        ops: list[np.ndarray] = []
        for part in parts:
            m = factor_re.fullmatch(part)
            if m:
                name, idx = m.group(1), m.group(2).split(",")
                if name not in operands:
                    raise ValueError(f"Unknown tensor '{name}'")
                arr = operands[name]
                if arr.ndim != len(idx):
                    raise ValueError(f"Tensor '{name}' rank {arr.ndim} != {len(idx)} indices")
                subs.append("".join(idx))
                ops.append(arr)
            elif num_re.match(part):
                coeff *= float(part)
            else:
                raise ValueError(f"Bad factor '{part}': want Name[i,j] or a scalar")
        if not ops:
            raise ValueError(f"Term '{term}' has no tensors")
        counts: dict[str, int] = {}
        for sub in subs:
            for idx in sub:
                counts[idx] = counts.get(idx, 0) + 1
        for idx in out_idx:
            if counts.get(idx, 0) != 1:
                raise ValueError(f"Free index '{idx}' must appear exactly once in each term")
        for idx, n in counts.items():
            if idx not in out_idx and n != 2:
                raise ValueError(f"Index '{idx}' appears {n}x in one term: must appear exactly twice")
        parsed.append((coeff, subs, ops))
    total: np.ndarray | None = None
    for coeff, subs, ops in parsed:
        part_result = evaluate_einsum(",".join(subs) + "->" + "".join(out_idx), *ops, timeout=timeout)
        total = coeff * part_result if total is None else total + coeff * part_result
    assert total is not None
    _ = out_name
    return total


def parse_indexed_to_sympy(
    formula: str,
    dims: dict[str, int],
) -> tuple[sp.Basic, sp.Basic]:
    """Parse ``"C[i,j] = A[i,k]*B[k,j]"`` into (lhs, rhs) sympy indexed forms.

    Free indices stay symbolic; contracted indices become explicit
    :class:`~sympy.Sum` over ``0..dims[idx]-1``. Tensor names (plus
    ``delta`` → :class:`~sympy.KroneckerDelta`) become
    :class:`~sympy.IndexedBase`. ``dims`` must cover every index. Raises
    ``ValueError`` on the same malformed input :func:`evaluate_indexed`
    refuses. Levi-Civita and derivatives stay out of scope.
    """
    import re

    if formula.count("=") != 1:
        raise ValueError("Need exactly one '=' in an indexed equation")
    lhs_raw, rhs_raw = (part.strip() for part in formula.split("="))
    lhs = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)(?:\[([A-Za-z](?:,[A-Za-z])*)\])?", lhs_raw)
    if not lhs:
        raise ValueError(f"Bad LHS '{lhs_raw}': want Name or Name[i,j,...]")
    out_name = lhs.group(1)
    out_idx = lhs.group(2).split(",") if lhs.group(2) else []
    rhs_indices = [
        idx
        for name_idx in re.findall(r"[A-Za-z_][A-Za-z0-9_]*\[([A-Za-z](?:,[A-Za-z])*)\]", rhs_raw)
        for idx in name_idx.split(",")
    ]
    for idx in out_idx + rhs_indices:
        if idx not in dims or int(dims[idx]) < 1:
            raise ValueError(f"Index '{idx}' needs a positive size in dims")

    def _indexed(name: str, idx: list[str]) -> sp.Basic:
        if name == "delta":
            raise ValueError("delta must be eliminated before construction")
        if not idx:
            return sp.Symbol(name)
        base = sp.IndexedBase(sp.Symbol(name))
        return base[tuple(sp.Symbol(i) for i in idx)]

    out_expr: sp.Basic = _indexed(out_name, out_idx)
    rhs = rhs_raw.replace("-", "+-")
    total: sp.Basic | None = None
    factor_re = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\[([A-Za-z](?:,[A-Za-z])*)\]")
    num_re = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\Z")
    for term in [t.strip() for t in rhs.split("+") if t.strip()]:
        sign = 1
        if term.startswith("-"):
            sign, term = -1, term[1:].strip()
        factors: list[tuple[str, list[str]]] = []
        scalars: list[sp.Basic] = []
        for part in [p.strip() for p in term.split("*") if p.strip()]:
            m = factor_re.fullmatch(part)
            if m:
                factors.append((m.group(1), m.group(2).split(",")))
            elif num_re.match(part):
                scalars.append(sp.sympify(part))
            else:
                raise ValueError(f"Bad factor '{part}': want Name[i,j] or a scalar")
        if not factors:
            raise ValueError(f"Term '{term}' has no tensors")
        # Eliminate KroneckerDelta by index renaming (δ(a,b) kills one index).
        rename: dict[str, str] = {}
        kept: list[tuple[str, list[str]]] = []
        for name, idx in factors:
            if name == "delta":
                if len(idx) != 2:
                    raise ValueError("delta needs exactly two indices")
                a, b = (i if not (s := rename.get(i)) else s for i in idx)
                if a == b:
                    continue  # δ(a,a) sums to dims(a): folded into the sum below
                victim, survivor = (b, a) if b not in out_idx else ((a, b) if a not in out_idx else (None, None))
                if victim is None:
                    kept.append((name, idx))  # both free: stays symbolic
                else:
                    rename[victim] = survivor  # type: ignore[assignment]
            else:
                kept.append((name, idx))

        def _apply(idx: list[str]) -> list[str]:
            seen: set[str] = set()
            mapped = []
            for i in idx:
                while i in rename:
                    if i in seen:
                        break
                    seen.add(i)
                    i = rename[i]
                mapped.append(i)
            return mapped

        pieces: list[sp.Basic] = list(scalars)
        term_idx: list[str] = []
        for name, idx in kept:
            mapped = _apply(idx)
            pieces.append(_indexed(name, mapped))
            term_idx.extend(mapped)
        prod: sp.Basic = pieces[0]
        for piece in pieces[1:]:
            prod = prod * piece
        for idx in set(term_idx) - set(out_idx):
            prod = sp.Sum(prod, (sp.Symbol(idx), 0, int(dims[idx]) - 1))
        total = sign * prod if total is None else total + sign * prod
    assert total is not None
    return out_expr, total


def symbolic_contract(formula: str, dims: dict[str, int]) -> sp.Basic:
    """Evaluate the explicit sums of :func:`parse_indexed_to_sympy`.

    Returns the simplified ``lhs - rhs``: exactly ``0`` proves the identity
    (e.g. ``"y[i] = delta[i,j]*x[j]"`` → ``y[i] - x[i]``). KroneckerDelta
    sums evaluate; anything left symbolic is a genuine residual, not a
    matcher failure.
    """
    lhs, rhs = parse_indexed_to_sympy(formula, dims)
    try:
        return sp.simplify(lhs - rhs.doit())
    except (ValueError, TypeError, ArithmeticError) as e:
        raise ArithmeticError(f"Symbolic contraction failed: {e}") from e


def simulate_sde_euler_maruyama(
    drift_str: str,
    diffusion_str: str,
    y0: float,
    total_time: float,
    dt: float,
    n_paths: int = 1000,
    seed: int = 42,
    params: dict[str, float] | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
) -> dict[str, object]:
    """Euler-Maruyama sampling of ``dY = a(x, Y) dt + b(x, Y) dW``.

    Returns paths plus terminal mean/std. This *samples* trajectories with a
    fixed seed — reproducibility is exact, but it is Monte Carlo evidence,
    not a closed-form SDE solution. Refuses non-finite grids and explosions.
    """
    if not (math.isfinite(total_time) and total_time > 0):
        raise ValueError("Need finite total_time > 0")
    if not (math.isfinite(dt) and dt > 0) or dt > total_time:
        raise ValueError("Need finite 0 < dt <= total_time")
    if int(n_paths) < 2:
        raise ValueError("Need n_paths >= 2")
    if not math.isfinite(y0):
        raise ValueError("y0 must be finite")
    a_expr = safe_parse_expr(drift_str)
    b_expr = safe_parse_expr(diffusion_str)
    x_sym, y_sym = sp.Symbol("x"), sp.Symbol("y")
    subs = {sp.Symbol(k): float(v) for k, v in (params or {}).items()}

    def _simulate() -> dict[str, object]:
        fa = sp.lambdify((x_sym, y_sym), a_expr.subs(subs), modules=["numpy"])
        fb = sp.lambdify((x_sym, y_sym), b_expr.subs(subs), modules=["numpy"])
        rng = np.random.RandomState(int(seed))
        n_steps = int(round(float(total_time) / float(dt)))
        xs = np.linspace(0.0, float(total_time), n_steps + 1)
        paths = np.empty((int(n_paths), n_steps + 1))
        paths[:, 0] = float(y0)
        sq = math.sqrt(float(dt))
        for i in range(n_steps):
            try:
                drift = np.asarray(fa(xs[i], paths[:, i]), dtype=float)
                diff = np.asarray(fb(xs[i], paths[:, i]), dtype=float)
            except (ValueError, TypeError, ArithmeticError) as e:
                raise ArithmeticError(f"SDE coefficient evaluation failed at step {i}: {e}") from e
            paths[:, i + 1] = paths[:, i] + drift * float(dt) + diff * sq * rng.normal(size=int(n_paths))
            if not bool(np.all(np.isfinite(paths[:, i + 1]))):
                raise ArithmeticError(f"SDE went non-finite at step {i}")
        terminal = paths[:, -1]
        return {
            "paths": paths,
            "x": xs,
            "terminal_mean": float(np.mean(terminal)),
            "terminal_std": float(np.std(terminal, ddof=1)),
            "n_paths": int(n_paths),
        }

    result = _run_bounded(_simulate, timeout)
    assert isinstance(result, dict)
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


def solve_eigen(
    matrix: list[list[float]] | np.ndarray,
    symmetric: bool = False,
    timeout: float = _DEFAULT_TIMEOUT,
) -> dict[str, object]:
    """Validated eigendecomposition of a real square matrix.

    Uses ``eigh`` for ``symmetric=True`` (orthonormal vectors guaranteed) and
    ``eig`` otherwise. Every pair is residual-checked
    (``||Av - λv|| ≤ 1e-8·(1 + |λ|·||v||)``); complex eigenpairs of a real
    matrix are returned as magnitude/phase-agnostic real+imag parts with the
    residual still enforced. Non-finite input, non-square input, or a failed
    residual raises instead of returning unchecked numbers. Eigenvalues are
    sorted by descending magnitude.
    """
    try:
        mat = np.asarray(matrix, dtype=float)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Matrix not array-like: {e}") from e
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1] or mat.shape[0] < 1:
        raise ValueError(f"Need a nonempty square matrix, got shape {mat.shape}")
    if not bool(np.all(np.isfinite(mat))):
        raise ValueError("Matrix must be finite")

    def _decompose() -> dict[str, object]:
        try:
            if symmetric:
                vals, vecs = np.linalg.eigh((mat + mat.T) / 2)
            else:
                vals, vecs = np.linalg.eig(mat)
        except np.linalg.LinAlgError as e:
            raise ArithmeticError(f"Eigendecomposition failed: {e}") from e
        order = np.argsort(-np.abs(vals), kind="stable")
        vals, vecs = vals[order], vecs[:, order]
        residuals = []
        for k in range(len(vals)):
            lam, vec = vals[k], vecs[:, k]
            norm = float(np.linalg.norm(vec))
            if not math.isfinite(norm) or norm == 0.0:
                raise ArithmeticError(f"Eigenvector {k} degenerate")
            res = float(np.linalg.norm(mat @ vec - lam * vec))
            if res > 1e-8 * (1.0 + abs(complex(lam)) * norm):
                raise ArithmeticError(f"Eigenpair {k} failed residual check ({res:.2e})")
            residuals.append(res)
        return {
            "eigenvalues": np.asarray(vals),
            "eigenvectors": np.asarray(vecs),
            "residuals": residuals,
            "symmetric": bool(symmetric),
        }

    result = _run_bounded(_decompose, timeout)
    assert isinstance(result, dict)
    return result


def _scan_events(
    g_vals: list[np.ndarray],
    xs: np.ndarray,
    ys: np.ndarray,
    event_names: list[str],
) -> dict[str, object] | None:
    """First strict sign change across the sampled trajectory, if any."""
    for k, name in enumerate(event_names):
        col = np.asarray([float(v[k]) if np.ndim(v) else float(v) for v in g_vals])
        for i in range(1, len(xs)):
            if col[i - 1] == 0.0:
                return {"event": name, "index": i - 1, "x": float(xs[i - 1]), "bracket": None}
            if col[i] == 0.0 or col[i - 1] * col[i] < 0:
                return {"event": name, "index": i, "x": float(xs[i]), "bracket": (float(xs[i - 1]), float(xs[i]))}
    return None


def _hit_int(hit: dict[str, object], key: str) -> int:
    """Typed access into scan-hit dicts (mypi-safe)."""
    v = hit[key]
    assert isinstance(v, int)
    return v

def evaluate_ode_events(
    rhs_str: str,
    y0: float,
    x_grid: list[float] | np.ndarray,
    events: list[str],
    params: dict[str, float] | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
    method: str = "rk45",
    refine_substeps: int = 100,
) -> dict[str, object]:
    """Integrate ``dy/dx = rhs`` and stop at the first zero of ``events``.

    Each event is a ``safe_parse_expr`` string in ``x``/``y`` meaning
    ``g(x, y) = 0``. The trajectory is integrated with ``method`` (default
    adaptive ``rk45``), scanned for the first strict sign change, then the
    bracketing interval is re-integrated with 100 fine RK4 substeps and the
    crossing linearly interpolated (precision ~bracket/100 — reported, not
    hidden). Grazing touches without sign change are NOT detected (stated,
    not silently missed). Returns ``xs``, ``ys`` (truncated past the event),
    and ``event`` (``None`` when nothing fires).
    """
    if not events:
        raise ValueError("Need at least one event expression")
    for g in events:
        safe_parse_expr(g)
    xs = np.asarray(list(x_grid), dtype=float)
    ys = evaluate_ode_ivp(rhs_str, y0, xs, params=params, timeout=timeout, method=method)
    x_sym, y_sym = sp.Symbol("x"), sp.Symbol("y")
    subs = {sp.Symbol(k): float(v) for k, v in (params or {}).items()}
    g_funcs = [sp.lambdify((x_sym, y_sym), safe_parse_expr(g).subs(subs), modules=["numpy"]) for g in events]
    g_vals = []
    for xi, yi in zip(xs, ys):
        try:
            g_vals.append(np.asarray([float(g(float(xi), float(yi))) for g in g_funcs], dtype=float))
        except (ValueError, TypeError, ArithmeticError) as e:
            raise ArithmeticError(f"Event evaluation failed: {e}") from e
    hit = _scan_events(g_vals, xs, ys, list(events))
    if hit is None:
        return {"xs": xs, "ys": ys, "event": None}
    hit_index = _hit_int(hit, "index")
    if hit["bracket"] is None:
        i = hit_index
        return {"xs": xs[: i + 1], "ys": ys[: i + 1], "event": hit}
    bracket = hit["bracket"]
    assert isinstance(bracket, tuple) and len(bracket) == 2
    xa, xb = float(bracket[0]), float(bracket[1])
    ev_name = str(hit["event"])
    i = _hit_int(hit, "index")
    ya = float(ys[i - 1])
    fine_x = np.linspace(xa, xb, max(3, int(refine_substeps) + 1))
    fine_y = evaluate_ode_ivp(rhs_str, ya, fine_x, params=params, timeout=timeout, method="rk4")
    g = g_funcs[list(events).index(ev_name)]
    try:
        fine_g = np.asarray([float(g(float(x), float(y))) for x, y in zip(fine_x, fine_y)])
    except (ValueError, TypeError, ArithmeticError) as e:
        raise ArithmeticError(f"Event refinement failed: {e}") from e
    cross = None
    for j in range(1, len(fine_x)):
        if fine_g[j - 1] * fine_g[j] <= 0 and fine_g[j - 1] != fine_g[j]:
            t = abs(fine_g[j - 1]) / (abs(fine_g[j - 1]) + abs(fine_g[j]))
            cross = (float(fine_x[j - 1] + t * (fine_x[j] - fine_x[j - 1])), j, t)
            break
    if cross is None:
        ii = _hit_int(hit, "index")
        return {"xs": xs[: ii + 1], "ys": ys[: ii + 1], "event": hit}
    x_star, j, t = cross
    keep = np.concatenate([xs[:i], [x_star]])
    return {
        "xs": keep,
        "ys": np.concatenate([ys[:i], [fine_y[j - 1] + t * (fine_y[j] - fine_y[j - 1])]]),
        "event": {**hit, "x": x_star, "precision": float(xb - xa) / max(1, int(refine_substeps))},
    }


def evaluate_ode_system_events(
    rhs_list: list[str],
    y0_list: list[float],
    x_grid: list[float] | np.ndarray,
    events: list[str],
    var_names: list[str] | None = None,
    params: dict[str, float] | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
    method: str = "rk4",
    refine_substeps: int = 100,
) -> dict[str, object]:
    """Vector version of :func:`evaluate_ode_events` for ODE systems.

    Events are strings in ``x`` + state vars (default ``y1..yn``). Same
    scan-then-refine contract; truncated ``xs``/``ys`` (shape ``(m, n)``)
    plus the firing ``event`` (``None`` when quiet).
    """
    if not events:
        raise ValueError("Need at least one event expression")
    n = len(rhs_list)
    names = list(var_names) if var_names else [f"y{k + 1}" for k in range(n)]
    for g in events:
        safe_parse_expr(g)
    xs = np.asarray(list(x_grid), dtype=float)
    ys = evaluate_ode_system(rhs_list, y0_list, xs, var_names=names, params=params, timeout=timeout, method=method)
    x_sym = sp.Symbol("x")
    syms = [sp.Symbol(v) for v in names]
    subs = {sp.Symbol(k): float(v) for k, v in (params or {}).items()}
    g_funcs = [sp.lambdify([x_sym, *syms], safe_parse_expr(g).subs(subs), modules=["numpy"]) for g in events]
    g_vals = []
    for xi, row in zip(xs, ys):
        try:
            g_vals.append(np.asarray([float(g(float(xi), *[float(v) for v in row])) for g in g_funcs], dtype=float))
        except (ValueError, TypeError, ArithmeticError) as e:
            raise ArithmeticError(f"Event evaluation failed: {e}") from e
    hit = _scan_events(g_vals, xs, ys, list(events))
    if hit is None:
        return {"xs": xs, "ys": ys, "event": None}
    hit_index = _hit_int(hit, "index")
    if hit["bracket"] is None:
        i = hit_index
        return {"xs": xs[: i + 1], "ys": ys[: i + 1, :], "event": hit}
    bracket = hit["bracket"]
    assert isinstance(bracket, tuple) and len(bracket) == 2
    xa, xb = float(bracket[0]), float(bracket[1])
    ev_name = str(hit["event"])
    i = _hit_int(hit, "index")
    fine_x = np.linspace(xa, xb, max(3, int(refine_substeps) + 1))
    fine_y = evaluate_ode_system(rhs_list, list(ys[i - 1]), fine_x, var_names=names, params=params, timeout=timeout, method="rk4")
    g = g_funcs[list(events).index(ev_name)]
    try:
        fine_g = np.asarray([float(g(float(x), *[float(v) for v in row])) for x, row in zip(fine_x, fine_y)])
    except (ValueError, TypeError, ArithmeticError) as e:
        raise ArithmeticError(f"Event refinement failed: {e}") from e
    cross = None
    for j in range(1, len(fine_x)):
        if fine_g[j - 1] * fine_g[j] <= 0 and fine_g[j - 1] != fine_g[j]:
            t = abs(fine_g[j - 1]) / (abs(fine_g[j - 1]) + abs(fine_g[j]))
            cross = (float(fine_x[j - 1] + t * (fine_x[j] - fine_x[j - 1])), j, t)
            break
    if cross is None:
        return {"xs": xs[: i + 1], "ys": ys[: i + 1, :], "event": hit}
    x_star, j, t = cross
    return {
        "xs": np.concatenate([xs[:i], [x_star]]),
        "ys": np.vstack([ys[:i, :], fine_y[j - 1] + t * (fine_y[j] - fine_y[j - 1])]),
        "event": {**hit, "x": x_star, "precision": float(xb - xa) / max(1, int(refine_substeps))},
    }
