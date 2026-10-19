r"""Limit-Order-Book Transport Discovery: advection-diffusion-replenishment dynamics.

Status: experimental

Scope & honesty ceiling:
- Models the *deviation of resting-order depth density around the mid-price* as a
  linear transport field ``u(x, t)`` on a periodic price-offset axis obeying

      u_t = -v * u_x + D * u_xx - lambda * u

  with quote drift ``v``, quote-flicker diffusion ``D`` and liquidity
  replenishment/decay ``lambda``. This is the standard linearised LOB density
  model; it deliberately excludes queue-position dynamics, order-size
  discreteness, and cross-asset coupling.
- Snapshots are synthetic: a spectral generator (exact mode evolution) and an
  explicit RK4 method-of-lines generator are both provided. No claim about real
  tick-level data is made here; wiring a live LOB feed would require a separate
  data-integrity experiment.
- Coefficients are recovered by two *independent* paths (finite-difference least
  squares and per-mode spectral decay/phase fits) and cross-checked against the
  statistical observables that define them (centre-of-mass drift speed, variance
  growth rate, total-mass decay rate) plus the repository PDE discovery engine.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_TWO_PI = 2.0 * np.pi
_TRANSPORT_TERMS = ("u", "u_x", "u_xx")


@dataclass
class LobTransportRecovery:
    """Coefficients recovered from snapshots via finite-difference least squares."""

    drift_velocity: float
    diffusion: float
    decay_rate: float
    r_squared: float
    n_time: int
    n_space: int
    terms: tuple[str, ...] = _TRANSPORT_TERMS


@dataclass
class SpectralTransportRecovery:
    """Coefficients recovered from per-mode spectral decay and phase slopes."""

    drift_velocity: float
    diffusion: float
    decay_rate: float
    modes_used: tuple[int, ...]


def simulate_lob_transport(
    n_x: int = 128,
    n_t: int = 100,
    domain_length: float = _TWO_PI,
    total_time: float = 1.0,
    drift_velocity: float = 0.4,
    diffusion: float = 0.02,
    decay_rate: float = 0.1,
    modes: tuple[int, ...] = (1, 2, 3),
    bump: bool = False,
    bump_centre: float | None = None,
    bump_width: float = 0.3,
    method: str = "spectral",
    seed: int = 2026,
) -> tuple[np.ndarray, float, float]:
    r"""Simulate LOB depth-deviation snapshots on a periodic price grid.

    Returns ``(snapshots, dx, dt)`` where ``snapshots`` has shape ``(n_t, n_x)``.
    ``method="spectral"`` propagates each Fourier mode exactly (ground-truth
    consistency generator); ``method="rk4"`` integrates the same PDE with an
    explicit fourth-order Runge-Kutta method of lines on periodic central
    differences (truncation-bearing generator).
    """
    if n_x < 8:
        raise ValueError("n_x must be at least 8")
    if n_t < 5:
        raise ValueError("n_t must be at least 5")
    if method not in {"spectral", "rk4"}:
        raise ValueError("method must be 'spectral' or 'rk4'")

    rng = np.random.default_rng(seed)
    dx = domain_length / n_x
    dt = total_time / (n_t - 1)
    x = np.arange(n_x) * dx

    if bump:
        centre = domain_length / 2.0 - 1.2 if bump_centre is None else bump_centre
        dist = np.abs(x - centre)
        dist = np.minimum(dist, domain_length - dist)
        u0 = np.exp(-0.5 * (dist / bump_width) ** 2)
    else:
        amplitudes = rng.uniform(0.5, 1.0, len(modes))
        phases = rng.uniform(0.0, _TWO_PI, len(modes))
        u0 = np.zeros(n_x)
        for amplitude, phase, mode in zip(amplitudes, phases, modes, strict=True):
            u0 += amplitude * np.cos(_TWO_PI * mode * x / domain_length + phase)

    if method == "spectral":
        snapshots = _spectral_evolution(u0, dx, dt, n_t, drift_velocity, diffusion, decay_rate)
    else:
        snapshots = _rk4_evolution(u0, dx, dt, n_t, drift_velocity, diffusion, decay_rate)
    return snapshots, dx, dt


def _spectral_evolution(
    u0: np.ndarray,
    dx: float,
    dt: float,
    n_t: int,
    drift_velocity: float,
    diffusion: float,
    decay_rate: float,
) -> np.ndarray:
    n_x = u0.size
    k = _TWO_PI * np.fft.fftfreq(n_x, d=dx)
    u0_hat = np.fft.fft(u0)
    decay = diffusion * k ** 2 + decay_rate
    snapshots = np.empty((n_t, n_x), dtype=np.float64)
    for i in range(n_t):
        t = i * dt
        snapshots[i] = np.real(np.fft.ifft(u0_hat * np.exp(-(decay + 1j * drift_velocity * k) * t)))
    return snapshots


def _rk4_evolution(
    u0: np.ndarray,
    dx: float,
    dt: float,
    n_t: int,
    drift_velocity: float,
    diffusion: float,
    decay_rate: float,
) -> np.ndarray:
    # Explicit RK4 on an advection-diffusion operator is only stable inside a
    # step-size window; without a guard, refining nx (or raising D) silently
    # pushes D*dt/dx^2 past its limit and the integrator explodes to NaN from
    # the first step. Refuse before generating unusable snapshots.
    cfl = diffusion * dt / (dx * dx)
    peclet = abs(drift_velocity) * dt / dx
    if cfl > 0.5 or peclet > 2.0:
        raise ValueError(
            f"RK4 timestep violates stability limits: diffusion number D*dt/dx^2={cfl:.3f} "
            f"(<= 0.5 required), transport number |v|*dt/dx={peclet:.3f} (<= 2.0 required). "
            "Reduce total_time/n_t or refine the spatial grid; alternatively use method='spectral'."
        )

    def rhs(u: np.ndarray) -> np.ndarray:
        u_x = (np.roll(u, -1) - np.roll(u, 1)) / (2.0 * dx)
        u_xx = (np.roll(u, -1) - 2.0 * u + np.roll(u, 1)) / (dx * dx)
        return -drift_velocity * u_x + diffusion * u_xx - decay_rate * u

    snapshots = np.empty((n_t, u0.size), dtype=np.float64)
    snapshots[0] = u0
    for i in range(n_t - 1):
        u = snapshots[i]
        k1 = rhs(u)
        k2 = rhs(u + 0.5 * dt * k1)
        k3 = rhs(u + 0.5 * dt * k2)
        k4 = rhs(u + dt * k3)
        snapshots[i + 1] = u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return snapshots


def recover_transport_coefficients(
    snapshots: np.ndarray,
    dx: float,
    dt: float,
) -> LobTransportRecovery:
    r"""Recover ``(v, D, lambda)`` by least squares on ``u_t = -v u_x + D u_xx - lambda u``.

    Time derivatives use central differences in the interior of the snapshot
    stack; spatial derivatives use periodic central differences.
    """
    snapshots = np.asarray(snapshots, dtype=np.float64)
    n_t, n_x = snapshots.shape
    if n_t < 3:
        raise ValueError("at least 3 snapshots are required")

    u_t = (snapshots[2:] - snapshots[:-2]) / (2.0 * dt)
    u_mid = snapshots[1:-1]
    u_x = (np.roll(u_mid, -1, axis=1) - np.roll(u_mid, 1, axis=1)) / (2.0 * dx)
    u_xx = (np.roll(u_mid, -1, axis=1) - 2.0 * u_mid + np.roll(u_mid, 1, axis=1)) / (dx * dx)

    design = np.column_stack([u_mid.ravel(), u_x.ravel(), u_xx.ravel()])
    target = u_t.ravel()
    coefficients, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    residual = target - design @ coefficients
    ss_res = float(residual @ residual)
    ss_tot = float(np.sum((target - target.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0

    return LobTransportRecovery(
        drift_velocity=float(-coefficients[1]),
        diffusion=float(coefficients[2]),
        decay_rate=float(-coefficients[0]),
        r_squared=r_squared,
        n_time=n_t,
        n_space=n_x,
    )


def spectral_transport_recovery(
    snapshots: np.ndarray,
    dx: float,
    dt: float,
    amplitude_floor: float = 1e-6,
) -> SpectralTransportRecovery:
    r"""Recover ``(v, D, lambda)`` from per-mode amplitude decay and phase drift.

    For each Fourier mode ``û_k(t)`` the model implies
    ``log|û_k(t)| = log|û_k(0)| - (D k^2 + lambda) t`` and
    ``arg û_k(t) = arg û_k(0) - v k t``. Fitting the positive-``k`` modes gives
    ``lambda`` and ``D`` from a linear fit against ``k^2`` and ``v`` from the
    phase-slope fit against ``k``.
    """
    snapshots = np.asarray(snapshots, dtype=np.float64)
    n_t, n_x = snapshots.shape
    spectrum = np.fft.fft(snapshots, axis=1)
    k = _TWO_PI * np.fft.fftfreq(n_x, d=dx)
    times = np.arange(n_t) * dt

    floor = amplitude_floor * float(np.max(np.abs(spectrum[0])))
    mode_indices = [m for m in range(1, n_x // 2) if abs(spectrum[0, m]) > floor]
    if len(mode_indices) < 2:
        raise ValueError("need at least two surviving positive modes for spectral recovery")

    amplitudes = np.abs(spectrum[:, mode_indices])
    log_amplitudes = np.log(amplitudes + 1e-300)
    decay_rates = np.empty(len(mode_indices))
    phase_slopes = np.empty(len(mode_indices))
    for column, mode in enumerate(mode_indices):
        log_fit = np.polyfit(times, log_amplitudes[:, column], 1)
        decay_rates[column] = -log_fit[0]
        phases = np.unwrap(np.angle(spectrum[:, mode]))
        phase_fit = np.polyfit(times, phases, 1)
        phase_slopes[column] = phase_fit[0]

    k_modes = k[mode_indices]
    decay_design = np.column_stack([np.ones(len(mode_indices)), k_modes ** 2])
    decay_coefficients, _, _, _ = np.linalg.lstsq(decay_design, decay_rates, rcond=None)
    decay_rate = float(decay_coefficients[0])
    diffusion = float(decay_coefficients[1])

    velocity_design = np.column_stack([k_modes])
    velocity_coefficients, _, _, _ = np.linalg.lstsq(velocity_design, -phase_slopes, rcond=None)
    drift_velocity = float(velocity_coefficients[0])

    return SpectralTransportRecovery(
        drift_velocity=drift_velocity,
        diffusion=diffusion,
        decay_rate=decay_rate,
        modes_used=tuple(mode_indices),
    )


def snapshot_mass(snapshots: np.ndarray, dx: float) -> np.ndarray:
    """Total depth mass ``integral u dx`` per snapshot."""
    return np.asarray(snapshots, dtype=np.float64).sum(axis=1) * dx


def snapshot_centre_of_mass(snapshots: np.ndarray, dx: float) -> np.ndarray:
    """Centre of mass per snapshot from the first circular moment.

    The linear first moment ``sum(x u)/sum(u)`` (a) jumps discontinuously by the
    domain length whenever a density packet crosses the periodic wrap point and
    (b) divides by the total mass, which is near zero for the default zero-mean
    Fourier-mode initial condition. Both are removed by computing the mean
    angle ``theta = atan2(sum u sin(q x), sum u cos(q x))`` with ``q = 2*pi/L``,
    guarding the degenerate zero-mass case with NaN, unwrapping the angle
    across time (so an advecting packet yields a continuous trajectory), and
    converting back to axis coordinates. The returned coordinate is
    ring-unwrapped (not mod L), by design: drift-speed cross-checks see a
    continuous trajectory that survives full wraps.
    """
    snapshots = np.asarray(snapshots, dtype=np.float64)
    if snapshots.ndim != 2:
        raise ValueError("snapshots must be a (n_t, n_x) array")
    n_x = snapshots.shape[1]
    domain_length = n_x * dx
    q = _TWO_PI / domain_length
    x = np.arange(n_x) * dx
    cos_moment = np.sum(snapshots * np.cos(q * x), axis=1)
    sin_moment = np.sum(snapshots * np.sin(q * x), axis=1)
    total_mass = np.sum(snapshots, axis=1) * dx
    mean_angle = np.arctan2(sin_moment, cos_moment)
    # A zero/negligible-mass field has no well-defined centre; represent that
    # honestly with NaN rather than atan2's silent 0/0 -> 0.
    mean_angle[np.abs(total_mass) < 1e-12] = np.nan
    unwrapped = np.unwrap(mean_angle)
    return unwrapped * domain_length / _TWO_PI


def snapshot_variance(snapshots: np.ndarray, dx: float) -> np.ndarray:
    """Variance about the centre of mass per snapshot (periodic-safe, exact).

    Mass-weighted variance of the angular deviation from the mean angle, with
    deviations wrapped into ``(-pi, pi]``::

        var_x = (L/2pi)**2 * sum_i w_i * wrap(theta_i - mu)**2 / sum_i w_i

    Unlike the mean-resultant approximation ``2*(L/2pi)**2*(1-|R|)`` (which is
    only first-order accurate for a narrow packet), this reproduces the usual
    linear variance for any packet width, so diffusion constants recovered
    from ``d var_x / dt`` stay exact. It never touches the periodic wrap
    point, and a zero-mass field yields NaN instead of dividing by zero.
    """
    snapshots = np.asarray(snapshots, dtype=np.float64)
    if snapshots.ndim != 2:
        raise ValueError("snapshots must be a (n_t, n_x) array")
    n_x = snapshots.shape[1]
    domain_length = n_x * dx
    q = _TWO_PI / domain_length
    x = np.arange(n_x) * dx
    theta = q * x
    cos_moment = np.sum(snapshots * np.cos(theta), axis=1)
    sin_moment = np.sum(snapshots * np.sin(theta), axis=1)
    total_mass = np.sum(snapshots, axis=1) * dx
    mean_angle = np.arctan2(sin_moment, cos_moment)
    deviation = theta[None, :] - mean_angle[:, None]
    deviation = (deviation + np.pi) % _TWO_PI - np.pi
    var_theta = np.sum(snapshots * deviation ** 2, axis=1) * dx / total_mass
    out = (domain_length / _TWO_PI) ** 2 * var_theta
    out[np.abs(total_mass) < 1e-12] = np.nan
    return out


def discover_transport_with_engine(
    snapshots: np.ndarray,
    dt: float,
    dx: float,
    **kwargs,
):
    """Run the repository 1D PDE discovery engine on LOB snapshots.

    The import is deliberately deferred so that importing
    ``apedp_dyn.market.lob_transport`` does not pull in the discovery stack.
    """
    from apedp_dyn.pde.dispatch import discover_pde_1d

    return discover_pde_1d(snapshots, dt, dx, **kwargs)
