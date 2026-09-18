"""APEDP PDE solvers: explicit finite-difference schemes, numpy only.

Structured grids (1D wave/advection, 2D/3D heat, 2D/3D Laplace/Poisson) plus
dimension-independent Monte Carlo for high dimensions: Feynman-Kac for free-
space heat in any d and walk-on-spheres for Laplace-Dirichlet on the n-ball.
A 31-point structured grid in 20D needs 31**20 ≈ 10**29 points and is refused;
the Monte Carlo solvers cost O(n_paths · d) instead.

Stability and convergence are *refused, not clamped*: violating grids raise
``ValueError`` and unconverged iterations raise ``ArithmeticError``, so a
caller can never mistake a blown-up array for a solution. All long loops are
wall-clock bounded like :mod:`apedp.eval_extended`.
"""
from __future__ import annotations

import logging
import math
import threading

import numpy as np
import sympy as sp

from apedp.safety import safe_parse_expr

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 30.0
_SEMAPHORE = threading.Semaphore(8)


def _run_bounded(fn, timeout: float = _DEFAULT_TIMEOUT):
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
        raise TimeoutError("too many PDE-solver workers in flight")
    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        raise TimeoutError(f"PDE solve exceeded {timeout}s budget")
    if failure:
        raise failure[0]
    return box.get("result")


def _eval_profile_1d(profile: str, x: np.ndarray, length: float) -> np.ndarray:
    L = sp.Symbol("L")
    if profile == "sin":
        return np.asarray(np.sin(np.pi * x / float(length)), dtype=float)
    if profile == "constant":
        return np.ones_like(x, dtype=float)
    expr = safe_parse_expr(profile)
    f = sp.lambdify(sp.Symbol("x"), expr.subs({L: float(length)}), modules=["numpy"])
    try:
        vals = np.asarray(f(x), dtype=float)
    except (ValueError, TypeError, ArithmeticError) as e:
        raise ValueError(f"Bad initial profile: {e}") from e
    if vals.shape != x.shape:
        raise ValueError("Initial profile shape mismatch")
    if not bool(np.all(np.isfinite(vals))):
        raise ValueError("Initial profile non-finite on grid")
    return vals


def solve_wave_1d(
    length: float,
    total_time: float,
    nx: int = 101,
    nt: int = 200,
    speed: float = 1.0,
    left_bc: float = 0.0,
    right_bc: float = 0.0,
    initial: str = "sin",
    initial_velocity: float = 0.0,
    timeout: float = _DEFAULT_TIMEOUT,
) -> dict[str, object]:
    """Leapfrog solve of ``u_tt = c^2 u_xx`` with Dirichlet BCs.

    Zero initial velocity plus ``initial`` displacement by default. Refuses
    ``C = c*dt/dx > 1`` (CFL) instead of running an unstable scheme.
    """
    for name, val, lo in (("length", length, 0), ("total_time", total_time, 0), ("speed", speed, 0)):
        if not (math.isfinite(val) and val > lo):
            raise ValueError(f"Need finite {name} > {lo}")
    if nx < 5 or nt < 2:
        raise ValueError("Need nx >= 5, nt >= 2")
    dx = float(length) / (nx - 1)
    dt = float(total_time) / nt
    courant = float(speed) * dt / dx
    if courant > 1.0:
        raise ValueError(f"Unstable wave grid C={courant:.3f} > 1: raise nx or lower nt/total_time")

    def _solve() -> dict[str, object]:
        x = np.linspace(0.0, float(length), nx)
        u_prev = _eval_profile_1d(initial, x, float(length))
        u_prev[0], u_prev[-1] = float(left_bc), float(right_bc)
        # First step from Taylor expansion with velocity v0 (constant here).
        lap = np.zeros_like(u_prev)
        lap[1:-1] = u_prev[2:] - 2 * u_prev[1:-1] + u_prev[:-2]
        u = u_prev + float(initial_velocity) * dt + 0.5 * courant**2 * lap
        u[0], u[-1] = float(left_bc), float(right_bc)
        for _ in range(1, nt):
            nxt = np.empty_like(u)
            nxt[1:-1] = 2 * u[1:-1] - u_prev[1:-1] + courant**2 * (u[2:] - 2 * u[1:-1] + u[:-2])
            nxt[0], nxt[-1] = float(left_bc), float(right_bc)
            if not bool(np.all(np.isfinite(nxt))):
                raise ArithmeticError("Wave solve went non-finite")
            u_prev, u = u, nxt
        return {"x": x, "u": u, "dx": dx, "dt": dt, "courant": courant}

    result = _run_bounded(_solve, timeout)
    assert isinstance(result, dict)
    return result


def solve_advection_1d(
    length: float,
    total_time: float,
    nx: int = 101,
    nt: int = 200,
    velocity: float = 1.0,
    left_bc: float | None = None,
    initial: str = "sin",
    timeout: float = _DEFAULT_TIMEOUT,
) -> dict[str, object]:
    """Upwind solve of ``u_t + a u_x = 0`` (periodic unless ``left_bc`` given).

    With ``left_bc=None`` the domain wraps (periodic); otherwise the inflow
    boundary is pinned and outflow extrapolates. Refuses ``|C| > 1``.
    """
    for name, val, lo in (("length", length, 0), ("total_time", total_time, 0)):
        if not (math.isfinite(val) and val > lo):
            raise ValueError(f"Need finite {name} > {lo}")
    if not math.isfinite(velocity) or velocity == 0.0:
        raise ValueError("Need finite nonzero velocity")
    if nx < 5 or nt < 2:
        raise ValueError("Need nx >= 5, nt >= 2")
    dx = float(length) / (nx - 1)
    dt = float(total_time) / nt
    courant = float(velocity) * dt / dx
    if abs(courant) > 1.0:
        raise ValueError(f"Unstable advection grid |C|={abs(courant):.3f} > 1")

    def _solve() -> dict[str, object]:
        x = np.linspace(0.0, float(length), nx)
        u = _eval_profile_1d(initial, x, float(length))
        for _ in range(nt):
            nxt = np.empty_like(u)
            if courant > 0:
                nxt[0] = float(left_bc) if left_bc is not None else u[0] - courant * (u[0] - u[-1])
                nxt[1:] = u[1:] - courant * (u[1:] - u[:-1])
            else:
                nxt[-1] = float(left_bc) if left_bc is not None else u[-1] - courant * (u[0] - u[-1])
                nxt[:-1] = u[:-1] - courant * (u[1:] - u[:-1])
            if not bool(np.all(np.isfinite(nxt))):
                raise ArithmeticError("Advection solve went non-finite")
            u = nxt
        return {"x": x, "u": u, "dx": dx, "dt": dt, "courant": courant}

    result = _run_bounded(_solve, timeout)
    assert isinstance(result, dict)
    return result


def solve_heat_2d(
    lx: float,
    ly: float,
    total_time: float,
    nx: int = 31,
    ny: int = 31,
    nt: int = 200,
    diffusivity: float = 1.0,
    bc: float = 0.0,
    initial: str = "sin_sin",
    timeout: float = _DEFAULT_TIMEOUT,
) -> dict[str, object]:
    """Explicit solve of ``u_t = D (u_xx + u_yy)`` with zero Dirichlet BCs.

    ``initial="sin_sin"`` is ``sin(πx/Lx)·sin(πy/Ly)`` (decays analytically as
    ``exp(-π²D(1/Lx²+1/Ly²)t)``); otherwise a ``safe_parse_expr`` profile in
    ``x``/``y``/``Lx``/``Ly``. Refuses ``r = D·dt/dx² > 0.25`` (square grid).
    """
    for name, val in (("lx", lx), ("ly", ly), ("total_time", total_time), ("diffusivity", diffusivity)):
        if not (math.isfinite(val) and val > 0):
            raise ValueError(f"Need finite {name} > 0")
    if nx < 5 or ny < 5 or nt < 2:
        raise ValueError("Need nx, ny >= 5 and nt >= 2")
    dx = float(lx) / (nx - 1)
    dy = float(ly) / (ny - 1)
    if abs(dx - dy) / max(dx, dy) > 0.05:
        raise ValueError("Need near-square grid (|dx-dy|/max <= 5%) for the explicit stencil")
    dt = float(total_time) / nt
    r = float(diffusivity) * dt / dx**2
    if r > 0.25:
        raise ValueError(f"Unstable heat-2D grid r={r:.3f} > 0.25")

    def _solve() -> dict[str, object]:
        x = np.linspace(0.0, float(lx), nx)
        y = np.linspace(0.0, float(ly), ny)
        xx, yy = np.meshgrid(x, y, indexing="ij")
        if initial == "sin_sin":
            u = np.sin(np.pi * xx / float(lx)) * np.sin(np.pi * yy / float(ly))
        else:
            expr = safe_parse_expr(initial)
            f = sp.lambdify(
                (sp.Symbol("x"), sp.Symbol("y"), sp.Symbol("Lx"), sp.Symbol("Ly")),
                expr.subs({sp.Symbol("Lx"): float(lx), sp.Symbol("Ly"): float(ly)}),
                modules=["numpy"],
            )
            try:
                u = np.asarray(f(xx, yy), dtype=float)
            except (ValueError, TypeError, ArithmeticError) as e:
                raise ValueError(f"Bad initial profile: {e}") from e
            if u.shape != xx.shape:
                raise ValueError("Initial profile shape mismatch")
        u[0, :] = u[-1, :] = u[:, 0] = u[:, -1] = float(bc)
        for _ in range(nt):
            nxt = u.copy()
            nxt[1:-1, 1:-1] = u[1:-1, 1:-1] + r * (
                u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2] - 4 * u[1:-1, 1:-1]
            )
            nxt[0, :] = nxt[-1, :] = nxt[:, 0] = nxt[:, -1] = float(bc)
            if not bool(np.all(np.isfinite(nxt))):
                raise ArithmeticError("Heat-2D solve went non-finite")
            u = nxt
        return {"x": x, "y": y, "u": u, "dx": dx, "dt": dt, "r": r}

    result = _run_bounded(_solve, timeout)
    assert isinstance(result, dict)
    return result


def _jacobi_elliptic(
    nx: int,
    ny: int,
    lx: float,
    ly: float,
    bc_fn,
    source_fn,
    tol: float,
    maxiter: int,
) -> dict[str, object]:
    x = np.linspace(0.0, lx, nx)
    y = np.linspace(0.0, ly, ny)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    dx = lx / (nx - 1)
    u = np.asarray(bc_fn(xx, yy), dtype=float)
    if u.shape != xx.shape:
        raise ValueError("BC function shape mismatch")
    # Interior starts from zero: the solver must earn the harmonic field
    # through iteration (initializing interior to the BC profile would make
    # linear cases converge trivially without testing anything).
    u[1:-1, 1:-1] = 0.0
    # Re-pin the boundary (a custom profile is only a boundary condition).
    edge = np.asarray(bc_fn(xx, yy), dtype=float)
    u[0, :] = edge[0, :]
    u[-1, :] = edge[-1, :]
    u[:, 0] = edge[:, 0]
    u[:, -1] = edge[:, -1]
    f = np.asarray(source_fn(xx, yy), dtype=float)
    if f.shape != xx.shape:
        raise ValueError("Source function shape mismatch")
    if not bool(np.all(np.isfinite(u))) or not bool(np.all(np.isfinite(f))):
        raise ValueError("BC/source non-finite on grid")
    it = 0
    while it < maxiter:
        nxt = u.copy()
        nxt[1:-1, 1:-1] = 0.25 * (
            u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2] - dx**2 * f[1:-1, 1:-1]
        )
        delta = float(np.max(np.abs(nxt - u)))
        u = nxt
        it += 1
        if not math.isfinite(delta):
            raise ArithmeticError("Jacobi iteration went non-finite")
        if delta < tol:
            return {"x": x, "y": y, "u": u, "iterations": it, "final_delta": delta, "dx": dx}
    raise ArithmeticError(f"Jacobi did not converge in {maxiter} iterations")


def solve_laplace_2d(
    lx: float = 1.0,
    ly: float = 1.0,
    nx: int = 31,
    ny: int = 31,
    bc: str = "linear",
    tol: float = 1e-6,
    maxiter: int = 20000,
    timeout: float = _DEFAULT_TIMEOUT,
) -> dict[str, object]:
    """Jacobi solve of ``Δu = 0`` with Dirichlet BCs.

    ``bc="linear"`` pins the boundary to ``x + y`` (harmonic, so the exact
    interior is ``x + y``); ``bc="zero"`` pins zero; otherwise a
    ``safe_parse_expr`` profile in ``x``/``y``. Raises on non-convergence.
    """
    for name, val in (("lx", lx), ("ly", ly)):
        if not (math.isfinite(val) and val > 0):
            raise ValueError(f"Need finite {name} > 0")
    if nx < 5 or ny < 5:
        raise ValueError("Need nx, ny >= 5")
    if not (math.isfinite(tol) and tol > 0):
        raise ValueError("Need finite tol > 0")

    def _solve() -> dict[str, object]:
        if bc == "linear":
            bc_fn = lambda xx, yy: xx + yy  # noqa: E731
        elif bc == "zero":
            bc_fn = lambda xx, yy: np.zeros_like(xx)  # noqa: E731
        else:
            expr = safe_parse_expr(bc)
            f = sp.lambdify((sp.Symbol("x"), sp.Symbol("y")), expr, modules=["numpy"])
            bc_fn = lambda xx, yy: np.asarray(f(xx, yy), dtype=float)  # noqa: E731
        out = _jacobi_elliptic(nx, ny, float(lx), float(ly), bc_fn, lambda xx, yy: np.zeros_like(xx), tol, maxiter)
        out["bc"] = bc
        return out

    result = _run_bounded(_solve, timeout)
    assert isinstance(result, dict)
    return result


def solve_poisson_2d(
    source: str,
    lx: float = 1.0,
    ly: float = 1.0,
    nx: int = 31,
    ny: int = 31,
    tol: float = 1e-6,
    maxiter: int = 20000,
    timeout: float = _DEFAULT_TIMEOUT,
) -> dict[str, object]:
    """Jacobi solve of ``Δu = source(x, y)`` with zero Dirichlet BCs.

    ``source`` is a ``safe_parse_expr`` string in ``x``/``y``. Raises on
    non-convergence instead of returning a half-relaxed field.
    """
    for name, val in (("lx", lx), ("ly", ly)):
        if not (math.isfinite(val) and val > 0):
            raise ValueError(f"Need finite {name} > 0")
    if nx < 5 or ny < 5:
        raise ValueError("Need nx, ny >= 5")
    expr = safe_parse_expr(source)

    def _solve() -> dict[str, object]:
        f = sp.lambdify((sp.Symbol("x"), sp.Symbol("y")), expr, modules=["numpy"])
        return _jacobi_elliptic(
            nx, ny, float(lx), float(ly),
            lambda xx, yy: np.zeros_like(xx),
            lambda xx, yy: np.asarray(f(xx, yy), dtype=float),
            tol, maxiter,
        )

    result = _run_bounded(_solve, timeout)
    assert isinstance(result, dict)
    return result


_MAX_3D_POINTS = 31**3


def _check_3d_budget(nx: int, ny: int, nz: int) -> None:
    if nx < 5 or ny < 5 or nz < 5:
        raise ValueError("Need nx, ny, nz >= 5")
    if nx * ny * nz > _MAX_3D_POINTS:
        raise ValueError(
            f"Grid {nx}x{ny}x{nz} exceeds the structured-3D budget "
            f"({_MAX_3D_POINTS} points): coarsen or solve out-of-core. "
            "Unstructured meshes remain out of scope."
        )


def solve_heat_3d(
    lx: float,
    ly: float,
    lz: float,
    total_time: float,
    nx: int = 15,
    ny: int = 15,
    nz: int = 15,
    nt: int = 100,
    diffusivity: float = 1.0,
    bc: float = 0.0,
    timeout: float = _DEFAULT_TIMEOUT,
) -> dict[str, object]:
    """Explicit solve of ``u_t = D Δu`` on a box with zero Dirichlet BCs.

    Sine-product initial profile ``sin(πx/Lx)·sin(πy/Ly)·sin(πz/Lz)`` (exact
    decay rate ``π²D(1/Lx²+1/Ly²+1/Lz²)``). Refuses ``r > 1/6`` and grids
    over the structured-3D budget instead of swapping or blowing up.
    """
    for name, val in (("lx", lx), ("ly", ly), ("lz", lz), ("total_time", total_time), ("diffusivity", diffusivity)):
        if not (math.isfinite(val) and val > 0):
            raise ValueError(f"Need finite {name} > 0")
    _check_3d_budget(nx, ny, nz)
    if nt < 2:
        raise ValueError("Need nt >= 2")
    dx = float(lx) / (nx - 1)
    for L, n in (("ly", ny), ("lz", nz)):
        d = {"ly": float(ly), "lz": float(lz)}[L] / (n - 1)
        if abs(d - dx) / max(d, dx) > 0.05:
            raise ValueError("Need near-cubic grid for the explicit 3D stencil")
    dt = float(total_time) / nt
    r = float(diffusivity) * dt / dx**2
    if r > 1 / 6:
        raise ValueError(f"Unstable heat-3D grid r={r:.3f} > 1/6")

    def _solve() -> dict[str, object]:
        x = np.linspace(0.0, float(lx), nx)
        y = np.linspace(0.0, float(ly), ny)
        z = np.linspace(0.0, float(lz), nz)
        xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
        u = np.sin(np.pi * xx / float(lx)) * np.sin(np.pi * yy / float(ly)) * np.sin(np.pi * zz / float(lz))
        u[0, :, :] = u[-1, :, :] = u[:, 0, :] = u[:, -1, :] = u[:, :, 0] = u[:, :, -1] = float(bc)
        for _ in range(nt):
            nxt = u.copy()
            nxt[1:-1, 1:-1, 1:-1] = u[1:-1, 1:-1, 1:-1] + r * (
                u[2:, 1:-1, 1:-1] + u[:-2, 1:-1, 1:-1]
                + u[1:-1, 2:, 1:-1] + u[1:-1, :-2, 1:-1]
                + u[1:-1, 1:-1, 2:] + u[1:-1, 1:-1, :-2]
                - 6 * u[1:-1, 1:-1, 1:-1]
            )
            nxt[0, :, :] = nxt[-1, :, :] = nxt[:, 0, :] = nxt[:, -1, :] = nxt[:, :, 0] = nxt[:, :, -1] = float(bc)
            if not bool(np.all(np.isfinite(nxt))):
                raise ArithmeticError("Heat-3D solve went non-finite")
            u = nxt
        return {"x": x, "y": y, "z": z, "u": u, "dx": dx, "dt": dt, "r": r}

    result = _run_bounded(_solve, timeout)
    assert isinstance(result, dict)
    return result


def solve_laplace_3d(
    lx: float = 1.0,
    ly: float = 1.0,
    lz: float = 1.0,
    nx: int = 15,
    ny: int = 15,
    nz: int = 15,
    bc: str = "linear",
    tol: float = 1e-5,
    maxiter: int = 20000,
    timeout: float = _DEFAULT_TIMEOUT,
) -> dict[str, object]:
    """Jacobi solve of ``Δu = 0`` on a box with Dirichlet BCs.

    ``bc="linear"`` pins ``x + y + z`` (harmonic: the exact interior);
    ``bc="zero"`` pins zero; otherwise a ``safe_parse_expr`` profile in
    ``x``/``y``/``z`` used on the boundary only (interior starts at zero).
    Raises on non-convergence; refuses grids over the 3D budget.
    """
    for name, val in (("lx", lx), ("ly", ly), ("lz", lz)):
        if not (math.isfinite(val) and val > 0):
            raise ValueError(f"Need finite {name} > 0")
    _check_3d_budget(nx, ny, nz)
    if not (math.isfinite(tol) and tol > 0):
        raise ValueError("Need finite tol > 0")

    def _solve() -> dict[str, object]:
        x = np.linspace(0.0, float(lx), nx)
        y = np.linspace(0.0, float(ly), ny)
        z = np.linspace(0.0, float(lz), nz)
        xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
        if bc == "linear":
            edge = xx + yy + zz
        elif bc == "zero":
            edge = np.zeros_like(xx)
        else:
            expr = safe_parse_expr(bc)
            f = sp.lambdify((sp.Symbol("x"), sp.Symbol("y"), sp.Symbol("z")), expr, modules=["numpy"])
            try:
                edge = np.asarray(f(xx, yy, zz), dtype=float)
            except (ValueError, TypeError, ArithmeticError) as e:
                raise ValueError(f"Bad BC profile: {e}") from e
            if edge.shape != xx.shape:
                raise ValueError("BC profile shape mismatch")
        u = np.zeros_like(edge)
        u[0, :, :] = edge[0, :, :]
        u[-1, :, :] = edge[-1, :, :]
        u[:, 0, :] = edge[:, 0, :]
        u[:, -1, :] = edge[:, -1, :]
        u[:, :, 0] = edge[:, :, 0]
        u[:, :, -1] = edge[:, :, -1]
        it = 0
        while it < maxiter:
            nxt = u.copy()
            nxt[1:-1, 1:-1, 1:-1] = (
                u[2:, 1:-1, 1:-1] + u[:-2, 1:-1, 1:-1]
                + u[1:-1, 2:, 1:-1] + u[1:-1, :-2, 1:-1]
                + u[1:-1, 1:-1, 2:] + u[1:-1, 1:-1, :-2]
            ) / 6.0
            delta = float(np.max(np.abs(nxt - u)))
            u = nxt
            it += 1
            if not math.isfinite(delta):
                raise ArithmeticError("Jacobi-3D went non-finite")
            if delta < tol:
                return {"x": x, "y": y, "z": z, "u": u, "iterations": it, "final_delta": delta}
        raise ArithmeticError(f"Jacobi-3D did not converge in {maxiter} iterations")

    result = _run_bounded(_solve, timeout)
    assert isinstance(result, dict)
    return result


_MAX_MC_DIM = 64


def _mc_coords(dims: int) -> list[sp.Symbol]:
    if not (1 <= int(dims) <= _MAX_MC_DIM):
        raise ValueError(f"Need 1 <= dims <= {_MAX_MC_DIM} for Monte Carlo solvers")
    return [sp.Symbol(f"x{i}") for i in range(int(dims))]


def solve_heat_nd_feynman_kac(
    x0: list[float],
    total_time: float,
    initial: str,
    diffusivity: float = 1.0,
    n_paths: int = 20000,
    seed: int = 42,
    timeout: float = _DEFAULT_TIMEOUT,
) -> dict[str, object]:
    """Free-space heat ``u_t = D Δu`` in d dimensions via Feynman-Kac.

    ``u(x, t) = E[u0(x + √(2Dt)·Z)]`` with ``Z ~ N(0, I)``: cost is
    ``O(n_paths · d)``, so 20D costs ~20x a 1D run instead of 31**20 grid
    points. ``initial`` is a ``safe_parse_expr`` string in ``x0..x{d-1}``.
    Returns mean, standard error, and path count — Monte Carlo evidence with
    an error bar, not a mesh solution.
    """
    dims = len(list(x0))
    syms = _mc_coords(dims)
    if not (math.isfinite(total_time) and total_time > 0):
        raise ValueError("Need finite total_time > 0")
    if not (math.isfinite(diffusivity) and diffusivity > 0):
        raise ValueError("Need finite diffusivity > 0")
    if int(n_paths) < 100:
        raise ValueError("Need n_paths >= 100")
    start = np.asarray(list(x0), dtype=float)
    if start.shape != (dims,) or not bool(np.all(np.isfinite(start))):
        raise ValueError("x0 must be finite with one entry per dimension")
    expr = safe_parse_expr(initial)
    unknown = {str(s) for s in expr.free_symbols} - {str(s) for s in syms}
    if unknown:
        raise ValueError(f"Initial profile uses symbols outside x0..x{dims - 1}: {sorted(unknown)}")

    def _solve() -> dict[str, object]:
        f = sp.lambdify(syms, expr, modules=["numpy"])
        rng = np.random.RandomState(int(seed))
        spread = math.sqrt(2.0 * float(diffusivity) * float(total_time))
        pts = start[None, :] + spread * rng.normal(size=(int(n_paths), dims))
        try:
            vals = np.asarray(f(*[pts[:, i] for i in range(dims)]), dtype=float).ravel()
        except (ValueError, TypeError, ArithmeticError) as e:
            raise ArithmeticError(f"Initial-profile evaluation failed: {e}") from e
        if vals.shape != (int(n_paths),):
            vals = np.full(int(n_paths), float(np.ravel(vals)[0]), dtype=float)
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            raise ArithmeticError("All Feynman-Kac draws non-finite")
        se = float(np.std(finite, ddof=1) / math.sqrt(finite.size)) if finite.size > 1 else 0.0
        return {
            "value": float(np.mean(finite)),
            "stderr": se,
            "n_paths": int(n_paths),
            "dims": dims,
            "nonfinite_fraction": 1.0 - finite.size / int(n_paths),
        }

    result = _run_bounded(_solve, timeout)
    assert isinstance(result, dict)
    return result


def solve_laplace_nd_walk_on_spheres(
    x0: list[float],
    boundary: str,
    radius: float = 1.0,
    eps: float = 1e-3,
    n_paths: int = 20000,
    max_steps: int = 100000,
    seed: int = 42,
    timeout: float = _DEFAULT_TIMEOUT,
) -> dict[str, object]:
    """Laplace-Dirichlet on the d-ball ``|x| < radius`` via walk-on-spheres.

    From ``x`` jump uniformly to the largest inscribed sphere until within
    ``eps`` of the boundary, then read ``boundary`` there; the average is the
    solution (mean-value property). Cost is ``O(n_paths · d · steps)`` with
    typically tens of steps — 20D is routine. ``boundary`` is a
    ``safe_parse_expr`` string in ``x0..x{d-1}``.
    """
    dims = len(list(x0))
    syms = _mc_coords(dims)
    if not (math.isfinite(radius) and radius > 0):
        raise ValueError("Need finite radius > 0")
    if not (math.isfinite(eps) and 0 < eps < radius):
        raise ValueError("Need 0 < eps < radius")
    if int(n_paths) < 100:
        raise ValueError("Need n_paths >= 100")
    start = np.asarray(list(x0), dtype=float)
    if start.shape != (dims,) or not bool(np.all(np.isfinite(start))):
        raise ValueError("x0 must be finite with one entry per dimension")
    if float(np.linalg.norm(start)) >= float(radius):
        raise ValueError("x0 must lie strictly inside the ball")
    expr = safe_parse_expr(boundary)
    unknown = {str(s) for s in expr.free_symbols} - {str(s) for s in syms}
    if unknown:
        raise ValueError(f"Boundary uses symbols outside x0..x{dims - 1}: {sorted(unknown)}")

    def _solve() -> dict[str, object]:
        g = sp.lambdify(syms, expr, modules=["numpy"])
        rng = np.random.RandomState(int(seed))
        pos = np.tile(start, (int(n_paths), 1))
        active = np.ones(int(n_paths), dtype=bool)
        total_steps = 0
        while bool(np.any(active)):
            total_steps += 1
            if total_steps > int(max_steps):
                raise ArithmeticError("Walk-on-spheres exceeded max_steps: refusing")
            norms = np.linalg.norm(pos[active], axis=1)
            radii = float(radius) - norms
            if bool(np.any(radii < 0)):
                raise ArithmeticError("Walk escaped the ball")
            done = radii < float(eps)
            idx = np.where(active)[0]
            active[idx[done]] = False
            moving = idx[~done]
            if moving.size:
                dirs = rng.normal(size=(moving.size, dims))
                dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-300
                pos[moving] += dirs * radii[~done, None]
        try:
            vals = np.asarray(g(*[pos[:, i] for i in range(dims)]), dtype=float).ravel()
        except (ValueError, TypeError, ArithmeticError) as e:
            raise ArithmeticError(f"Boundary evaluation failed: {e}") from e
        if vals.shape != (int(n_paths),):
            vals = np.full(int(n_paths), float(np.ravel(vals)[0]), dtype=float)
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            raise ArithmeticError("All walk-on-spheres values non-finite")
        se = float(np.std(finite, ddof=1) / math.sqrt(finite.size)) if finite.size > 1 else 0.0
        return {
            "value": float(np.mean(finite)),
            "stderr": se,
            "n_paths": int(n_paths),
            "dims": dims,
            "total_steps": total_steps,
            "nonfinite_fraction": 1.0 - finite.size / int(n_paths),
        }

    result = _run_bounded(_solve, timeout)
    assert isinstance(result, dict)
    return result


def solve_poisson_fem(
    nodes: list[list[float]] | np.ndarray,
    triangles: list[list[int]] | np.ndarray,
    source: str = "0",
    dirichlet: dict[int, float] | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
) -> dict[str, object]:
    """P1 finite-element solve of ``-Δu = f`` on a user-supplied triangle mesh.

    Bring your own mesh: ``nodes`` is (N, 2), ``triangles`` is (M, 3) indices
    into ``nodes``. ``source`` is a ``safe_parse_expr`` string in ``x``/``y``;
    ``dirichlet`` maps node index -> pinned value (default: every node on the
    mesh hull loop pinned to 0 — the true mesh boundary, not the bounding
    box, so scattered point sets get a closed problem). Linear P1 elements
    with exact nodal load quadrature; the system solves with
    ``numpy.linalg.solve`` and refuses singular or non-finite assemblies
    instead of returning garbage. No mesher is bundled: mesh generation lives
    in :mod:`apedp.meshing`; only exotic (anisotropic, curved-boundary)
    meshing stays out.
    """
    pts = np.asarray(nodes, dtype=float)
    tris = np.asarray(triangles, dtype=int)
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.shape[0] < 3:
        raise ValueError("nodes must be (N, 2) with N >= 3")
    if tris.ndim != 2 or tris.shape[1] != 3 or tris.shape[0] < 1:
        raise ValueError("triangles must be (M, 3) with M >= 1")
    if not bool(np.all(np.isfinite(pts))):
        raise ValueError("nodes must be finite")
    if bool(np.any(tris < 0)) or bool(np.any(tris >= len(pts))):
        raise ValueError("triangle indices out of range")
    f_expr = safe_parse_expr(source)

    def _solve() -> dict[str, object]:
        fx = sp.lambdify((sp.Symbol("x"), sp.Symbol("y")), f_expr, modules=["numpy"])
        n = len(pts)
        try:
            fnodes = np.asarray(fx(pts[:, 0], pts[:, 1]), dtype=float).ravel()
        except (ValueError, TypeError, ArithmeticError) as e:
            raise ValueError(f"Source evaluation failed: {e}") from e
        if fnodes.shape != (n,):
            fnodes = np.full(n, float(np.ravel(fnodes)[0]), dtype=float)
        if not bool(np.all(np.isfinite(fnodes))):
            raise ValueError("Source non-finite at mesh nodes")
        mat = np.zeros((n, n))
        rhs = np.zeros(n)
        for tri in tris:
            p = pts[np.asarray(tri)]
            # Twice the signed area; degenerate (collinear) elements refused.
            area2 = abs(float(np.cross(p[1] - p[0], p[2] - p[0])))
            if area2 <= 1e-300:
                raise ValueError(f"Degenerate triangle {list(tri)}: zero area")
            # Gradients of the three P1 hats: perpendicular edge rotations / 2A.
            grads = np.empty((3, 2))
            for i in range(3):
                b, c = p[(i + 1) % 3], p[(i + 2) % 3]
                grads[i] = np.array([c[1] - b[1], b[0] - c[0]]) / area2
            ke = (area2 / 2) * (grads @ grads.T)
            # Exact load for the linear interpolant: ∫λ_a f = (A/12)(f_a + Σf).
            s_tri = float(fnodes[tri[0]] + fnodes[tri[1]] + fnodes[tri[2]])
            area = area2 / 2
            for a in range(3):
                rhs[tri[a]] += (area / 12) * (fnodes[tri[a]] + s_tri)
                for b in range(3):
                    mat[tri[a], tri[b]] += ke[a, b]
        if dirichlet is None:
            from apedp.meshing import _convex_hull_ordered

            hull = _convex_hull_ordered(pts)
            # Exact integer on-segment test on a per-call lattice: hull-loop
            # nodes (collinear-inclusive), not the bounding box, so scattered
            # sets get a closed boundary and box meshes behave as before.
            qx = [int(round(float(p[0]) * (1 << 30))) for p in pts]
            qy = [int(round(float(p[1]) * (1 << 30))) for p in pts]
            on_loop = set()
            for k in range(len(hull)):
                ax, ay = qx[hull[k]], qy[hull[k]]
                bx, by = qx[hull[(k + 1) % len(hull)]], qy[hull[(k + 1) % len(hull)]]
                abx, aby = bx - ax, by - ay
                seg2 = abx * abx + aby * aby
                for i in range(n):
                    px, py = qx[i] - ax, qy[i] - ay
                    if px * aby != py * abx:
                        continue
                    dot = px * abx + py * aby
                    if 0 <= dot <= seg2:
                        on_loop.add(i)
            pinned: dict[int, float] = {int(i): 0.0 for i in on_loop}
        else:
            pinned = {}
            for k, v in dirichlet.items():
                if not (0 <= int(k) < n) or not math.isfinite(float(v)):
                    raise ValueError(f"Bad Dirichlet pin {k!r}={v!r}")
                pinned[int(k)] = float(v)
        if not pinned:
            raise ValueError("No Dirichlet nodes: pure-Neumann P1 system is singular by construction")
        for idx, val in pinned.items():
            mat[idx, :] = 0.0
            mat[idx, idx] = 1.0
            rhs[idx] = val
        try:
            u = np.linalg.solve(mat, rhs)
        except np.linalg.LinAlgError as e:
            raise ArithmeticError(f"FEM system singular: {e}") from e
        if not bool(np.all(np.isfinite(u))):
            raise ArithmeticError("FEM solution non-finite")
        return {"u": u, "nodes": pts, "triangles": tris, "num_nodes": n, "num_elements": len(tris)}

    result = _run_bounded(_solve, timeout)
    assert isinstance(result, dict)
    return result
