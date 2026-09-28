"""D5-stage validators: energy conservation/balance laws (ported from apedp/dyn).

Principle (theirs, preserved): check the APPROPRIATE balance — never "everything
must be conserved". Undamped free oscillation conserves; damped decay obeys
dE/dt = F·v − c·v²; forced intervals obey the power balance. These validators
discriminate STRUCTURES (anti-damped impostors detected hard) but cannot detect
wrongly-valued parameters (a 2x-stiffness oscillator conserves its own energy
exactly) — parameter accuracy stays with trajectory validation (gate 4).
"""
from __future__ import annotations

import numpy as np


def mechanical_energy(x: np.ndarray, v: np.ndarray, m: float, k: float) -> np.ndarray:
    """E = m v^2/2 + k x^2/2."""
    return 0.5 * m * np.asarray(v, dtype=float) ** 2 + 0.5 * k * np.asarray(x, dtype=float) ** 2


def energy_drift(x: np.ndarray, v: np.ndarray, m: float, k: float) -> float:
    """Cumulative relative drift max|E − E0| / max|E|.

    Cumulative, not per-step: sustained exponential growth has an almost
    constant per-step relative change, which a per-step metric cannot see.
    """
    E = mechanical_energy(x, v, m, k)
    e_max = float(np.max(np.abs(E)))
    if e_max == 0.0:
        raise ValueError("degenerate energy trajectory")
    return float(np.max(np.abs(E - E[0])) / e_max)


def simulate_pair(model, m: float, k: float, t: np.ndarray,
                  u_of_t, x0: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Simulate a fitted model on a fixed grid; return (x, v).

    Integrator-precision tolerances: conservation is a structural property
    and must sit at solver noise, far below any parameter tolerance.
    """
    from apedp_dyn.simulation import simulate

    res = simulate(model, (float(t[0]), float(t[-1])), u_of_t,
                   np.asarray(x0, dtype=float), n_eval=len(t),
                   rtol=1e-10, atol=1e-12)
    if not res.success:
        raise ValueError(f"conservation sim failed: {res.detail}")
    return res.y[0], res.y[1]


def check_conservative_core(x: np.ndarray, v: np.ndarray, m: float, k: float,
                            tol: float = 1e-6) -> tuple[bool, float]:
    """Undamped free oscillation must conserve to integrator precision."""
    drift = energy_drift(x, v, m, k)
    return drift <= tol, drift


def check_damped_decay(x: np.ndarray, v: np.ndarray, m: float, k: float,
                       min_decay_fraction: float = 0.3,
                       slack: float = 0.02) -> tuple[bool, dict]:
    """Unforced dissipative interval: monotone non-increase + real decay."""
    E = mechanical_energy(x, v, m, k)
    info = {"decay_fraction": 1.0 - float(E[-1]) / float(E[0])}
    if np.any(np.diff(E) > slack * float(np.max(E))):
        return False, {**info, "reason": "energy increased in unforced interval"}
    if info["decay_fraction"] <= min_decay_fraction:
        return False, {**info, "reason": "damping did not act like damping"}
    return True, info
