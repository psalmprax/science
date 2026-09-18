"""APEDP PDE solvers: explicit finite-difference schemes, numpy only.

Covers the canonical second-order linear PDEs behind the L1 templates:

- 1D wave ``u_tt = c^2 u_xx`` (leapfrog, ``C = c*dt/dx <= 1`` enforced),
- 1D advection ``u_t + a u_x = 0`` (upwind, ``|C| <= 1`` enforced),
- 2D heat ``u_t = D (u_xx + u_yy)`` (explicit, ``r <= 0.25`` enforced),
- 2D Laplace / Poisson ``Δu = f`` (Jacobi to tolerance, Dirichlet BCs).

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
