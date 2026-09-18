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

    ``x_grid`` must be strictly increasing and finite.
    """
    if method not in ("rk4", "rk45"):
        raise ValueError("method must be 'rk4' or 'rk45'")
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

    def _integrate() -> np.ndarray:
        f = sp.lambdify((x_sym, y_sym), expr.subs(subs), modules=["numpy"])
        if method == "rk45":
            return _rk45_dense(f, xs, float(y0))
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


def evaluate_einsum(
    subscripts: str,
    *operands: list | np.ndarray,
    timeout: float = _DEFAULT_TIMEOUT,
) -> np.ndarray:
    """Validated ``numpy.einsum``: shape-checked tensor contraction.

    Rejects subscript/operand-count mismatches and non-finite results instead
    of broadcasting silently. This covers explicit tensor *contractions*;
    index-notation parsing and unstructured meshes remain out of scope.
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
