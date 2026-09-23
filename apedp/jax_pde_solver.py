"""APEDP JAX-differentiable PDE solvers.

Enables gradient-based optimization and inverse problems through
JAX automatic differentiation. Supports 1D heat and wave equations
with Neumann/Dirichlet boundary conditions.

Key features:
- JAX JIT-compiled solvers for speed
- `grad()`-compatible for inverse problems
- Boundary condition support (Dirichlet, Neumann) via static args
- Manufactured solution test helpers
"""


import jax
import jax.numpy as jnp
import numpy as np
from jax import grad, jit, vmap

# ---------------------------------------------------------------------------
# Heat equation solver
# ---------------------------------------------------------------------------


def validate_explicit_grid(D: float, dx: float, dt: float, nt: int,
                           *, wave_speed: float | None = None,
                           nx: int | None = None) -> None:
    """Reject invalid or unstable explicit-FD configurations before JIT.

    JAX tracing cannot reliably raise Python validation errors inside a jitted
    function, so public convenience callers use this check before dispatch.
    """
    import math

    values = (D, dx, dt)
    if not all(math.isfinite(float(value)) and float(value) > 0.0 for value in values):
        raise ValueError("D, dx, and dt must be finite and positive")
    if int(nt) < 1:
        raise ValueError("nt must be a positive integer")
    if nx is not None and int(nx) < 3:
        raise ValueError("nx must be at least 3")
    if wave_speed is not None:
        speed = float(wave_speed)
        if not math.isfinite(speed) or speed <= 0.0:
            raise ValueError("wave_speed must be finite and positive")
        courant = speed * float(dt) / float(dx)
        if courant > 1.0:
            raise ValueError(f"unstable wave CFL={courant:g}; require c*dt/dx <= 1")
    else:
        ratio = float(D) * float(dt) / float(dx) ** 2
        if ratio > 0.5:
            raise ValueError(f"unstable heat ratio={ratio:g}; require D*dt/dx**2 <= 0.5")


def validate_boundary_type(bctype: str) -> None:
    if bctype not in {"dirichlet", "neumann"}:
        raise ValueError("bctype must be 'dirichlet' or 'neumann'")


@jit(static_argnames=("bctype", "beta"))
def _heat_1d_step(u: jnp.ndarray, D: float, dx: float, dt: float,
                  beta: float, bctype: str) -> jnp.ndarray:
    """One explicit-FD step for the 1D heat equation.

    The ``jit`` decorator marks ``bctype`` and ``beta`` as static so that
    JAX can differentiate through the branch.
    """
    r = D * dt / dx**2
    if bctype == "dirichlet":
        u = u.at[0].set(beta)
        u = u.at[-1].set(beta)
        u = u.at[1:-1].set(u[1:-1] + r * (u[2:] - 2 * u[1:-1] + u[:-2]))
    elif bctype == "neumann":
        # Zero-flux ghost cells mirror the adjacent interior value. The old
        # negative reflection imposed an unintended sign-changing boundary.
        u = u.at[1:-1].set(u[1:-1] + r * (u[2:] - 2 * u[1:-1] + u[:-2]))
        u = u.at[0].set(u[1])
        u = u.at[-1].set(u[-2])
    return u


@jit(static_argnames=("bctype", "beta", "nt"))
def heat_1d_explicit_jit(
    u0: jnp.ndarray,
    D: float,
    dx: float,
    dt: float,
    nt: int,
    *,
    bctype: str = "dirichlet",
    beta: float = 0.0,
) -> jnp.ndarray:
    """1D heat equation u_t = D * u_xx with JAX differentiation.

    Solves via explicit finite differences over ``nt`` time steps.
    The ``bctype`` and ``beta`` arguments are *static* for JAX JIT.

    Args:
        u0: Initial condition u(x,0), shape (nx,)
        D: Diffusion coefficient (>0)
        dx: Spatial grid spacing (>0)
        dt: Time step (>0)
        nt: Number of time steps
        bctype: "dirichlet" or "neumann" (static)
        beta: Boundary value (Dirichlet) or flux (Neumann)

    Returns:
        u_final: Final solution u(x, nt*dt), shape (nx,)
    """
    # Apply initial boundary conditions. Neumann boundaries copy the adjacent
    # interior value so a constant field remains exactly constant.
    u = u0
    if bctype == "dirichlet":
        u = u.at[0].set(beta)
        u = u.at[-1].set(beta)
    elif bctype == "neumann":
        u = u.at[0].set(u[1])
        u = u.at[-1].set(u[-2])

    for _ in range(nt):
        u = _heat_1d_step(u, D, dx, dt, beta, bctype)

    return u


def heat_1d_manufactured_solution(
    key: jax.random.PRNGKey,
    D: float = 0.1,
    L: float = 1.0,
    T: float = 1.0,
    nx: int = 50,
    nt: int = 100,
    noise_level: float = 0.0,
) -> dict[str, jnp.ndarray]:
    """Generate manufactured solution for the 1D heat equation.

    Exact solution: u(x,t) = sin(pi*x/L) * exp(-D*pi^2*t/L^2)

    Args:
        key: JAX PRNG key for noise
        D: Diffusion coefficient
        L: Domain length [0, L]
        T: Final time
        nx: Number of spatial points
        nt: Number of time steps
        noise_level: Standard deviation of Gaussian noise

    Returns:
        Dict with keys:
        - 'x': Spatial grid, shape (nx,)
        - 't': Time grid, shape (nt+1,)
        - 'u_noisy': u(x,t) + noise, shape (nt+1, nx)
        - 'u_exact': Exact solution, shape (nt+1, nx)
        - 'u0': Initial condition u(x,0), shape (nx,)
        - 'params': Dict with D, L, T, nx, nt
    """
    x = jnp.linspace(0.0, L, nx)
    t = jnp.linspace(0.0, T, nt + 1)

    def u_exact_fn(x, t):
        return jnp.sin(jnp.pi * x / L) * jnp.exp(-D * jnp.pi**2 * t / L**2)

    u_exact = vmap(lambda ti: vmap(lambda xi: u_exact_fn(xi, ti))(x))(t)
    u0 = u_exact_fn(x, 0.0)

    key, subkey = jax.random.split(key)
    noise = noise_level * jax.random.normal(subkey, (nt + 1, nx))
    u_noisy = u_exact + noise

    params = {"D": D, "L": L, "T": T, "nx": nx, "nt": nt}

    return {
        "x": x,
        "t": t,
        "u_noisy": u_noisy,
        "u_exact": u_exact,
        "u0": u0,
        "params": params,
    }


# ---------------------------------------------------------------------------
# Wave equation solver
# ---------------------------------------------------------------------------

@jit(static_argnames=("bctype",))
def _wave_1d_step(u_curr: jnp.ndarray, u_prev: jnp.ndarray,
                  c: float, dx: float, rho2: float,
                  bctype: str) -> jnp.ndarray:
    """One explicit-FD step for the 1D wave equation.

    Returns only u_next; the caller shifts (u_prev, u_curr) = (u_curr, u_next).
    Uses interior stencil (no periodic roll) so Dirichlet/Neumann BCs are exact.
    """
    u_next_interior = (
        2.0 * u_curr[1:-1] - u_prev[1:-1]
        + rho2 * (u_curr[2:] - 2.0 * u_curr[1:-1] + u_curr[:-2])
    )
    u_next = u_curr.at[1:-1].set(u_next_interior)

    if bctype == "dirichlet":
        u_next = u_next.at[0].set(0.0)
        u_next = u_next.at[-1].set(0.0)
    elif bctype == "neumann":
        u_next = u_next.at[0].set(u_next[1])
        u_next = u_next.at[-1].set(u_next[-2])

    return u_next


@jit(static_argnames=("bctype", "nt"))
def wave_1d_explicit_jit(
    u0: jnp.ndarray,
    v0: jnp.ndarray,
    c: float,
    dx: float,
    dt: float,
    nt: int,
    *,
    bctype: str = "dirichlet",
) -> jnp.ndarray:
    """1D wave equation u_tt = c^2 * u_xx with explicit finite differences.

    Solves via central differences in time, central in space:
        u_{n+1}[i] = 2*u_n[i] - u_{n-1}[i] + (c*dt/dx)^2 * (u_n[i+1] - 2*u_n[i] + u_n[i-1])

    Boundary conditions (static):
    - Dirichlet: u(0,t)=0, u(L,t)=0
    - Neumann: u_x(0,t)=0, u_x(L,t)=0 (zero-flux)

    Args:
        u0: Initial displacement u(x,0), shape (nx,)
        v0: Initial velocity u_t(x,0), shape (nx,)
        c: Wave speed (>0)
        dx: Spatial spacing (>0)
        dt: Time step (>0)
        nt: Number of time steps
        bctype: "dirichlet" or "neumann" (static)

    Returns:
        u_final: Final displacement u(x, nt*dt), shape (nx,)
    """
    rho = c * dt / dx
    rho2 = rho**2

    # --- start step: u_1 = u_0 + dt*v_0 + (1/2)*c^2*dt^2*u_xx ---
    # u_xx includes /dx^2; multiply by c^2*dt^2 (NOT rho^2, which would
    # double-count 1/dx^2). Interior stencil only; BCs below (no roll wrap).
    u_0_xx_interior = (u0[2:] - 2.0 * u0[1:-1] + u0[:-2]) / dx**2
    u_first = u0 + dt * v0
    u_first = u_first.at[1:-1].add(0.5 * (c ** 2) * (dt ** 2) * u_0_xx_interior)

    # u_prev = u_0 (with BCs), u_curr = u_1 (with BCs)
    if bctype == "dirichlet":
        u_prev = u0.at[0].set(0.0).at[-1].set(0.0)
        u_first = u_first.at[0].set(0.0).at[-1].set(0.0)
    elif bctype == "neumann":
        u_prev = u0.at[0].set(u0[1]).at[-1].set(u0[-2])
        u_first = u_first.at[0].set(u_first[1]).at[-1].set(u_first[-2])

    u_curr = u_first

    # --- remaining steps (nt-1 of them) ---
    for _ in range(nt - 1):
        u_next = _wave_1d_step(u_curr, u_prev, c, dx, rho2, bctype)
        u_prev, u_curr = u_curr, u_next

    return u_curr


# ---------------------------------------------------------------------------
# Gradient helpers
# ---------------------------------------------------------------------------

def heat_1d_grad_D(
    u0: jnp.ndarray,
    D: float,
    dx: float,
    dt: float,
    nt: int,
    *,
    bctype: str = "dirichlet",
    beta: float = 0.0,
) -> float:
    """Compute gradient of ``sum(u_final**2)`` w.r.t. diffusion coefficient ``D``.

    Uses JAX ``grad`` through the finite-difference solver.

    Returns:
        Python float -- d(sum(u_final**2))/dD.
    """
    def loss(D_val):
        u_final = heat_1d_explicit_jit(u0, D_val, dx, dt, nt,
                                        bctype=bctype, beta=beta)
        return jnp.sum(u_final**2)

    return float(grad(loss)(D))


def heat_1d_grad_initial(
    u0: jnp.ndarray,
    D: float,
    dx: float,
    dt: float,
    nt: int,
    *,
    bctype: str = "dirichlet",
    beta: float = 0.0,
) -> jnp.ndarray:
    """Compute gradient of ``sum(u_final**2)`` w.r.t. initial condition ``u0``.

    Returns:
        array of shape (nx,) with d(sum(u_final**2))/d(u0_i).
    """
    def loss(u0_val):
        u_final = heat_1d_explicit_jit(u0_val, D, dx, dt, nt,
                                        bctype=bctype, beta=beta)
        return jnp.sum(u_final**2)

    return grad(loss)(u0)


# ---------------------------------------------------------------------------
# Arbitrary-dimensional solvers
# ---------------------------------------------------------------------------


def validate_nd_grid(D: float, spacing, dt: float, nt: int, *, wave_speed: float | None = None) -> tuple[float, ...]:
    """Validate an n-D explicit grid and return normalized spacings.

    Heat stability uses ``D*dt*sum(1/dx_i²) <= 1/2``; wave stability uses
    ``c*dt*sqrt(sum(1/dx_i²)) <= 1``. Validation occurs outside JIT so callers
    receive ordinary Python errors instead of tracer failures.
    """
    import math

    values = tuple(float(value) for value in spacing)
    if not values or any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("spacing must contain finite positive values")
    if not math.isfinite(float(D)) or D <= 0.0 or not math.isfinite(float(dt)) or dt <= 0.0:
        raise ValueError("D and dt must be finite and positive")
    if int(nt) < 1:
        raise ValueError("nt must be a positive integer")
    inverse_square_sum = sum(1.0 / value ** 2 for value in values)
    if wave_speed is None:
        ratio = float(D) * float(dt) * inverse_square_sum
        if ratio > 0.5:
            raise ValueError(f"unstable n-D heat ratio={ratio:g}; require D*dt*sum(dx^-2) <= 0.5")
    else:
        speed = float(wave_speed)
        courant = speed * float(dt) * inverse_square_sum ** 0.5
        if not math.isfinite(speed) or speed <= 0.0:
            raise ValueError("wave_speed must be finite and positive")
        if courant > 1.0:
            raise ValueError(f"unstable n-D wave CFL={courant:g}; require c*dt*sqrt(sum(dx^-2)) <= 1")
    return values


def _nd_laplacian(u: jnp.ndarray, spacing: jnp.ndarray) -> jnp.ndarray:
    lap = jnp.zeros_like(u)
    for axis in range(u.ndim):
        left = [slice(None)] * u.ndim
        right = [slice(None)] * u.ndim
        center = [slice(None)] * u.ndim
        left[axis], right[axis], center[axis] = slice(0, -2), slice(2, None), slice(1, -1)
        lap = lap.at[tuple(center)].add(
            (u[tuple(right)] - 2.0 * u[tuple(center)] + u[tuple(left)])
            / spacing[axis] ** 2
        )
    return lap


def _apply_nd_boundary(u: jnp.ndarray, bctype: str, beta: float) -> jnp.ndarray:
    for axis in range(u.ndim):
        # Heterogeneous by design: slice(None) placeholders replaced with ints
        # per axis below, so annotate as a plain list rather than list[slice].
        first: list = [slice(None)] * u.ndim
        last: list = [slice(None)] * u.ndim
        adjacent_first: list = [slice(None)] * u.ndim
        adjacent_last: list = [slice(None)] * u.ndim
        first[axis], last[axis] = 0, -1
        adjacent_first[axis], adjacent_last[axis] = 1, -2
        if bctype == "dirichlet":
            u = u.at[tuple(first)].set(beta)
            u = u.at[tuple(last)].set(beta)
        elif bctype == "neumann":
            u = u.at[tuple(first)].set(u[tuple(adjacent_first)])
            u = u.at[tuple(last)].set(u[tuple(adjacent_last)])
    return u


@jit(static_argnames=("bctype", "beta", "nt"))
def heat_nd_explicit_jit(
    u0: jnp.ndarray,
    D: float,
    spacing: jnp.ndarray,
    dt: float,
    nt: int,
    *,
    bctype: str = "dirichlet",
    beta: float = 0.0,
) -> jnp.ndarray:
    """Explicit heat solver for any spatial rank >= 1."""
    u0 = _apply_nd_boundary(u0, bctype, beta)
    spacing = jnp.asarray(spacing)

    def body(_, state):
        updated = state + D * dt * _nd_laplacian(state, spacing)
        return _apply_nd_boundary(updated, bctype, beta)

    return jax.lax.fori_loop(0, nt, body, u0)


@jit(static_argnames=("bctype", "nt"))
def wave_nd_explicit_jit(
    u0: jnp.ndarray,
    v0: jnp.ndarray,
    c: float,
    spacing: jnp.ndarray,
    dt: float,
    nt: int,
    *,
    bctype: str = "dirichlet",
) -> jnp.ndarray:
    """Explicit wave solver for any spatial rank >= 1."""
    spacing = jnp.asarray(spacing)
    u0 = _apply_nd_boundary(u0, bctype, 0.0)
    first = u0 + dt * v0 + 0.5 * c ** 2 * dt ** 2 * _nd_laplacian(u0, spacing)
    previous = _apply_nd_boundary(first, bctype, 0.0)
    current = previous

    def body(_, state):
        previous, current = state
        nxt = 2.0 * current - previous + c ** 2 * dt ** 2 * _nd_laplacian(current, spacing)
        return current, _apply_nd_boundary(nxt, bctype, 0.0)

    _, result = jax.lax.fori_loop(0, nt - 1, body, (previous, current))
    return result


def solve_heat_nd(
    u0: jnp.ndarray,
    D: float,
    spacing,
    total_time: float,
    nt: int,
    *,
    bctype: str = "dirichlet",
    beta: float = 0.0,
) -> jnp.ndarray:
    """Validated arbitrary-dimensional heat solve with a uniform final time."""
    validate_boundary_type(bctype)
    normalized = validate_nd_grid(D, spacing, float(total_time) / int(nt), nt)
    array = jnp.asarray(u0)
    if array.ndim != len(normalized) or min(array.shape) < 3:
        raise ValueError("u0 rank must match spacing length and every axis must have at least 3 points")
    return heat_nd_explicit_jit(array, D, jnp.asarray(normalized), float(total_time) / int(nt), nt,
                                bctype=bctype, beta=beta)


def solve_wave_nd(
    u0: jnp.ndarray,
    v0: jnp.ndarray,
    c: float,
    spacing,
    total_time: float,
    nt: int,
    *,
    bctype: str = "dirichlet",
) -> jnp.ndarray:
    """Validated arbitrary-dimensional wave solve with a uniform final time."""
    validate_boundary_type(bctype)
    normalized = validate_nd_grid(1.0, spacing, float(total_time) / int(nt), nt, wave_speed=c)
    displacement, velocity = jnp.asarray(u0), jnp.asarray(v0)
    if displacement.shape != velocity.shape or displacement.ndim != len(normalized) or min(displacement.shape) < 3:
        raise ValueError("u0 and v0 must have equal rank/shape matching spacing and at least 3 points per axis")
    return wave_nd_explicit_jit(displacement, velocity, c, jnp.asarray(normalized),
                                float(total_time) / int(nt), nt, bctype=bctype)


def heat_nd_grad_D(
    u0: jnp.ndarray,
    D: float,
    spacing,
    dt: float,
    nt: int,
    *,
    bctype: str = "dirichlet",
    beta: float = 0.0,
) -> float:
    """Gradient of the squared n-D heat solution with respect to ``D``."""
    normalized = validate_nd_grid(D, spacing, dt, nt)
    spacing_array = jnp.asarray(normalized)

    def loss(diffusion):
        result = heat_nd_explicit_jit(
            jnp.asarray(u0), diffusion, spacing_array, dt, nt,
            bctype=bctype, beta=beta,
        )
        return jnp.sum(result ** 2)

    return float(grad(loss)(D))


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------

def solve_heat_1d(
    u0: jnp.ndarray,
    D: float,
    L: float = 1.0,
    T: float = 1.0,
    nx: int = 50,
    nt: int = 100,
    bctype: str = "dirichlet",
) -> jnp.ndarray:
    """Solve 1D heat equation on ``[0, L]`` with diffusion ``D``.

    Convenience wrapper; sets up the grid and calls the JIT solver.
    """
    if not isinstance(nx, int) or not isinstance(nt, int) or nx < 3 or nt < 1:
        raise ValueError("nx must be >= 3 and nt must be >= 1")
    validate_boundary_type(bctype)
    dx = L / (nx - 1)
    dt = T / nt
    validate_explicit_grid(D, dx, dt, nt, nx=nx)

    if callable(u0):
        x = jnp.linspace(0.0, L, nx)
        u0 = u0(x)

    if u0.ndim == 0:          # scalar initial condition
        u0 = jnp.full(nx, float(u0))

    return heat_1d_explicit_jit(u0, D, dx, dt, nt, bctype=bctype)


# ---------------------------------------------------------------------------
# Test block
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== JAX PDE Solver Tests ===\n")

    # Test 1: Manufactured solution + solver-vs-analytic convergence
    print("Test 1: Heat equation manufactured solution")
    key = jax.random.PRNGKey(42)
    data = heat_1d_manufactured_solution(key, D=0.1, L=1.0, T=0.5, nx=30, nt=50)
    print(f"  x shape: {data['x'].shape}")
    print(f"  u_noisy shape: {data['u_noisy'].shape}")
    print(f"  u_exact shape: {data['u_exact'].shape}")
    rmse = jnp.sqrt(jnp.mean((data['u_noisy'] - data['u_exact'])**2))
    print(f"  RMSE vs exact (noiseless, expect 0): {rmse:.6f}")
    assert float(rmse) == 0.0

    # 1b: noisy generator recovers the noise level (non-trivial check)
    noisy = heat_1d_manufactured_solution(jax.random.PRNGKey(1), D=0.1,
                                          L=1.0, T=0.5, nx=30, nt=50,
                                          noise_level=0.05)
    rmse_n = jnp.sqrt(jnp.mean((noisy['u_noisy'] - noisy['u_exact'])**2))
    print(f"  RMSE vs exact (noise 0.05): {float(rmse_n):.4f}")
    assert 0.03 < float(rmse_n) < 0.07, rmse_n

    # 1c: explicit solver converges to the analytic final state
    nx_c, L_c, D_c, T_c, nt_c = 30, 1.0, 0.1, 0.1, 20
    dx_c, dt_c = L_c / (nx_c - 1), T_c / nt_c
    assert D_c * dt_c / dx_c**2 <= 0.5
    x_c = jnp.linspace(0.0, L_c, nx_c)
    u_num = heat_1d_explicit_jit(jnp.sin(jnp.pi * x_c / L_c), D_c, dx_c, dt_c,
                                 nt_c, bctype="dirichlet", beta=0.0)
    u_ana = (jnp.sin(jnp.pi * x_c / L_c)
             * jnp.exp(-D_c * jnp.pi**2 * T_c / L_c**2))
    rel = float(jnp.sqrt(jnp.mean((u_num - u_ana)**2)) / jnp.sqrt(jnp.mean(u_ana**2)))
    print(f"  solver-vs-analytic rel L2: {rel:.5f} (require < 0.02)")
    assert rel < 0.02, rel

    # Test 2: Gradient w.r.t. D (stable: r = D*dt/dx^2 <= 0.5)
    print("\nTest 2: Gradient w.r.t. diffusion coefficient D")
    u0 = data['u0']
    D_val = 0.1
    dx = 1.0 / 29.0
    nt_stable = 200
    dt = 0.5 / nt_stable  # r = 0.1*0.0025/(1/29)^2 ~= 0.21 (stable)
    print(f"  stability r = {D_val * dt / dx**2:.3f} (require <= 0.5)")
    grad_D = heat_1d_grad_D(u0, D_val, dx, dt, nt_stable)
    print(f"  d(loss)/dD at D={D_val}: {grad_D:.6f}")
    print(f"  Gradient is non-zero: {grad_D != 0}")
    print(f"  Gradient sign (should be negative): {grad_D < 0}")

    # Test 3: Gradient w.r.t. initial condition
    print("\nTest 3: Gradient w.r.t. initial condition u0")
    grad_u0 = heat_1d_grad_initial(u0, D_val, dx, dt, nt_stable)
    print(f"  d(loss)/d(u0) shape: {grad_u0.shape}")
    print(f"  Sample gradient values: min={grad_u0.min():.4f}, max={grad_u0.max():.4f}")

    # Test 4: Wave equation
    print("\nTest 4: Wave equation (CFL-stable)")
    nx_w = 40
    L_w = 1.0
    T_w = 0.5
    c_w = 1.0
    dx_w = L_w / (nx_w - 1)
    dt_w = 0.9 * dx_w / c_w
    nt_w = int(T_w / dt_w)

    u0_w = jnp.sin(2 * jnp.pi * jnp.linspace(0, L_w, nx_w))
    v0_w = jnp.zeros_like(u0_w)

    u_final = wave_1d_explicit_jit(u0_w, v0_w, c_w, dx_w, dt_w, nt_w, bctype="dirichlet")
    initial_energy = jnp.sum(u0_w**2)
    final_energy = jnp.sum(u_final**2)
    energy_ratio = final_energy / initial_energy
    print(f"  Initial energy: {initial_energy:.4f}")
    print(f"  Final energy: {final_energy:.4f}")
    print(f"  Energy ratio: {energy_ratio:.4f}")
    print(f"  Energy preserved within 20%: {abs(energy_ratio - 1.0) < 0.2}")

    # Test 5: Neumann boundary condition
    print("\nTest 5: Neumann boundary condition")
    u0_5 = jnp.ones(20)
    dx_5 = 1.0 / 19.0
    dt_5 = 0.01
    nt_5 = 10

    u_dirichlet = heat_1d_explicit_jit(u0_5, 0.1, dx_5, dt_5, nt_5, bctype="dirichlet", beta=0.0)
    u_neumann = heat_1d_explicit_jit(u0_5, 0.1, dx_5, dt_5, nt_5, bctype="neumann", beta=0.0)

    print(f"  Dirichlet: u[0]={u_dirichlet[0]:.4f}, u[-1]={u_dirichlet[-1]:.4f}")
    print(f"  Neumann:   sum={u_neumann.sum():.4f}  (expected ~20.0)")
    print(f"  Neumann preserves sum: {abs(u_neumann.sum() - 20.0) < 1e-3}")

    # Test 6: 2D heat vs analytic (exercises heat_nd_explicit_jit)
    print("\nTest 6: 2D heat vs analytic")
    n2, L2, D2, T2, nt2 = 12, 1.0, 0.1, 0.05, 5
    dx2 = L2 / (n2 - 1)
    dt2 = T2 / nt2
    ratio2 = D2 * dt2 * 2.0 / dx2**2
    print(f"  stability ratio = {ratio2:.3f} (require <= 0.5)")
    assert ratio2 <= 0.5
    g1 = jnp.linspace(0.0, L2, n2)
    X2, Y2 = jnp.meshgrid(g1, g1, indexing="ij")
    u0_2d = jnp.sin(jnp.pi * X2) * jnp.sin(jnp.pi * Y2)
    u_2d = solve_heat_nd(u0_2d, D2, (dx2, dx2), T2, nt2, bctype="dirichlet")
    u_2d_ana = u0_2d * jnp.exp(-D2 * 2.0 * jnp.pi**2 * T2)
    rel2 = float(jnp.sqrt(jnp.mean((u_2d - u_2d_ana)**2))
                 / jnp.sqrt(jnp.mean(u_2d_ana**2)))
    print(f"  2D solver-vs-analytic rel L2: {rel2:.5f} (require < 0.05)")
    assert rel2 < 0.05, rel2

    # Test 7: 2D wave stays bounded (exercises wave_nd_explicit_jit)
    print("\nTest 7: 2D wave boundedness")
    c7, T7, nt7 = 1.0, 0.1, 5
    dt7 = T7 / nt7
    cfl7 = c7 * dt7 * (2.0 / dx2**2) ** 0.5
    print(f"  CFL = {cfl7:.3f} (require <= 1)")
    assert cfl7 <= 1.0
    v0_2d = jnp.zeros_like(u0_2d)
    w_2d = solve_wave_nd(u0_2d, v0_2d, c7, (dx2, dx2), T7, nt7,
                         bctype="dirichlet")
    assert bool(jnp.all(jnp.isfinite(w_2d)))
    growth7 = float(jnp.max(jnp.abs(w_2d)) / jnp.max(jnp.abs(u0_2d)))
    print(f"  max|u| growth factor: {growth7:.3f} (require < 2)")
    assert growth7 < 2.0, growth7

    # Test 8: n-D gradient smoke (finite, negative in D)
    print("\nTest 8: heat_nd_grad_D smoke")
    gD_nd = heat_nd_grad_D(u0_2d, D2, (dx2, dx2), dt2, nt2, bctype="dirichlet")
    print(f"  d(loss)/dD = {gD_nd:.6f}")
    assert np.isfinite(gD_nd) and gD_nd < 0.0

    print("\n=== All tests passed ===")