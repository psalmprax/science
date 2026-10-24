r"""LOB transport discovery: spectral/FD coefficient recovery and engine cross-check.

Status: experimental
"""
from __future__ import annotations

import numpy as np

from apedp_dyn.market.lob_transport import (
    discover_transport_with_engine,
    recover_transport_coefficients,
    simulate_lob_transport,
    snapshot_centre_of_mass,
    snapshot_mass,
    snapshot_variance,
    spectral_transport_recovery,
)

TRUE_VELOCITY = 0.4
TRUE_DIFFUSION = 0.02
TRUE_DECAY = 0.1


def _relative_error(got: float, expected: float) -> float:
    return abs(got - expected) / abs(expected)


def test_spectral_recovery_is_machine_precision():
    snapshots, dx, dt = simulate_lob_transport(
        n_x=128, n_t=100, drift_velocity=TRUE_VELOCITY, diffusion=TRUE_DIFFUSION, decay_rate=TRUE_DECAY
    )
    recovery = spectral_transport_recovery(snapshots, dx, dt)
    assert recovery.modes_used == (1, 2, 3)
    assert _relative_error(recovery.drift_velocity, TRUE_VELOCITY) < 1e-8
    assert _relative_error(recovery.diffusion, TRUE_DIFFUSION) < 1e-8
    assert _relative_error(recovery.decay_rate, TRUE_DECAY) < 1e-8


def test_finite_difference_recovery_matches_generator_parameters():
    snapshots, dx, dt = simulate_lob_transport(
        n_x=128, n_t=100, drift_velocity=TRUE_VELOCITY, diffusion=TRUE_DIFFUSION, decay_rate=TRUE_DECAY
    )
    recovery = recover_transport_coefficients(snapshots, dx, dt)
    assert recovery.r_squared > 0.9999, recovery.r_squared
    assert _relative_error(recovery.drift_velocity, TRUE_VELOCITY) < 5e-3
    assert _relative_error(recovery.diffusion, TRUE_DIFFUSION) < 5e-3
    assert _relative_error(recovery.decay_rate, TRUE_DECAY) < 5e-3


def test_rk4_generator_recovery_matches_parameters():
    snapshots, dx, dt = simulate_lob_transport(
        n_x=128, n_t=100, drift_velocity=TRUE_VELOCITY, diffusion=TRUE_DIFFUSION,
        decay_rate=TRUE_DECAY, method="rk4",
    )
    recovery = recover_transport_coefficients(snapshots, dx, dt)
    assert recovery.r_squared > 0.9999, recovery.r_squared
    assert _relative_error(recovery.drift_velocity, TRUE_VELOCITY) < 1e-3
    assert _relative_error(recovery.diffusion, TRUE_DIFFUSION) < 1e-3
    assert _relative_error(recovery.decay_rate, TRUE_DECAY) < 1e-3


def test_statistical_observables_match_parameters():
    snapshots, dx, dt = simulate_lob_transport(
        bump=True, n_x=128, n_t=100, drift_velocity=TRUE_VELOCITY,
        diffusion=TRUE_DIFFUSION, decay_rate=TRUE_DECAY,
    )
    times = np.arange(snapshots.shape[0]) * dt
    centre = snapshot_centre_of_mass(snapshots, dx)
    mass = snapshot_mass(snapshots, dx)
    variance = snapshot_variance(snapshots, dx)

    velocity_from_centre = float(np.polyfit(times, centre, 1)[0])
    decay_from_mass = -float(np.polyfit(times, np.log(mass), 1)[0])
    diffusion_from_variance = float(np.polyfit(times, variance, 1)[0]) / 2.0

    assert _relative_error(velocity_from_centre, TRUE_VELOCITY) < 1e-6
    assert _relative_error(decay_from_mass, TRUE_DECAY) < 1e-6
    assert _relative_error(diffusion_from_variance, TRUE_DIFFUSION) < 1e-6


def test_engine_discovers_transport_terms():
    snapshots, dx, dt = simulate_lob_transport(
        n_x=128, n_t=64, drift_velocity=TRUE_VELOCITY, diffusion=TRUE_DIFFUSION, decay_rate=TRUE_DECAY
    )
    result = discover_transport_with_engine(snapshots, dt, dx)
    terms = list(result.active_terms)
    assert "u_x" in terms, terms
    assert "u_xx" in terms, terms
    coefficients = dict(zip(terms, result.coefficients, strict=True))
    assert _relative_error(coefficients["u_x"], -TRUE_VELOCITY) < 1e-3, coefficients
    assert _relative_error(coefficients["u_xx"], TRUE_DIFFUSION) < 1e-3, coefficients
    if "u" in terms:
        assert _relative_error(coefficients["u"], -TRUE_DECAY) < 1e-2, coefficients
    assert result.holdout_r2 > 0.999, result.holdout_r2
