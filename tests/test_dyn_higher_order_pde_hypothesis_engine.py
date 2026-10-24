"""Tests for Higher-Order Spatio-Temporal PDE Hypothesis Engine & JetFlow Bridge."""
from __future__ import annotations

import numpy as np

from apedp_dyn.epistemic.high_dim_pde import EpistemicPdeVerdict
from apedp_dyn.pde import (
    HigherOrderPdeDiscoveryResult,
    HigherOrderPdeProposer,
    HigherOrderPdeReferee,
    discover_higher_order_pde,
)


def _simulate_spectral_1d_first_order(
    u0: np.ndarray,
    dx: float,
    dt: float,
    n_steps: int,
    c_u: float = 0.0,
    c_ux: float = 0.0,
    c_uxx: float = 0.0,
    c_uxxx: float = 0.0,
    c_uxxxx: float = 0.0,
    c_uux: float = 0.0,
) -> np.ndarray:
    """Integrate 1D periodic higher-order PDE u_t = F(u, u_x, u_xx, u_xxx, u_xxxx, u*u_x) via RK4."""
    n_x = len(u0)
    k = 2.0 * np.pi * np.fft.fftfreq(n_x, d=dx)
    ik1 = 1j * k
    if n_x % 2 == 0:
        ik1[n_x // 2] = 0.0
    ik2 = (1j * k) ** 2
    ik3 = (1j * k) ** 3
    if n_x % 2 == 0:
        ik3[n_x // 2] = 0.0
    ik4 = (1j * k) ** 4

    lin_symbol = c_u + c_ux * ik1 + c_uxx * ik2 + c_uxxx * ik3 + c_uxxxx * ik4
    exp_half = np.exp(0.5 * dt * lin_symbol)
    exp_full = np.exp(dt * lin_symbol)

    def nonlin_hat(u_hat_state: np.ndarray) -> np.ndarray:
        if abs(c_uux) <= 1e-12:
            return np.zeros_like(u_hat_state)
        u_phys = np.fft.ifft(u_hat_state).real
        ux_phys = np.fft.ifft(ik1 * u_hat_state).real
        return np.fft.fft(c_uux * (u_phys * ux_phys))

    traj = np.zeros((n_steps, n_x), dtype=np.float64)
    u_hat = np.fft.fft(u0.copy())
    for t in range(n_steps):
        traj[t] = np.fft.ifft(u_hat).real
        a = dt * nonlin_hat(u_hat)
        b = dt * nonlin_hat(exp_half * (u_hat + 0.5 * a))
        c = dt * nonlin_hat(exp_half * u_hat + 0.5 * b)
        d = dt * nonlin_hat(exp_full * u_hat + exp_half * c)
        u_hat = exp_full * u_hat + (
            exp_full * a + 2.0 * exp_half * (b + c) + d
        ) / 6.0
    return traj


def test_kdv_third_order_dispersion_discovery_and_referee():
    """Verify discovery and sealed holdout confirmation of 3rd-order KdV equation: u_t = -1.0 u*u_x - 0.1 u_xxx."""
    n_x = 64
    length = 2.0 * np.pi
    dx = length / n_x
    x = np.linspace(0.0, length, n_x, endpoint=False)
    dt = 0.001

    u0 = np.sin(x) + 0.5 * np.cos(2.0 * x)
    full_traj = _simulate_spectral_1d_first_order(
        u0, dx=dx, dt=dt, n_steps=60, c_uxxx=-0.1, c_uux=-1.0
    )
    dev_u = full_traj[:30]
    holdout_u = full_traj[30:]

    proposer = HigherOrderPdeProposer(
        dt=dt,
        dx=dx,
        temporal_order=1,
        max_spatial_order=4,
        periodic=True,
        sparsity_threshold=0.04,
    )
    res, proposals, _ = proposer.propose_from_field(dev_u, compress_tt=False)
    assert len(proposals) == 1
    assert res.temporal_order == 1
    assert res.max_spatial_order == 3
    assert "u*u_x" in res.active_terms
    assert "u_xxx" in res.active_terms
    assert res.r2_dev > 0.999

    params = proposals[0].parameters
    assert abs(params["u*u_x"] - (-1.0)) < 0.02
    assert abs(params["u_xxx"] - (-0.1)) < 0.01

    referee = HigherOrderPdeReferee(min_holdout_r2=0.99)
    verdict = referee.adjudicate(res, holdout_u, dt=dt, dx=dx, periodic=True)
    assert verdict.verdict == EpistemicPdeVerdict.IDENTIFIED_CAUSAL_LAW
    assert verdict.frozen_oos_r2 > 0.999
    assert verdict.ultraviolet_well_posed


def test_kuramoto_sivashinsky_fourth_order_biharmonic_and_uv_guard():
    """Verify 4th-order biharmonic discovery (u_t = -0.4 u_xx - 0.05 u_xxxx) and UV blowup guard."""
    n_x = 64
    length = 2.0 * np.pi
    dx = length / n_x
    x = np.linspace(0.0, length, n_x, endpoint=False)
    dt = 0.0005

    u0 = np.cos(x) + 0.4 * np.sin(2.0 * x) + 0.2 * np.cos(3.0 * x)
    full_traj = _simulate_spectral_1d_first_order(
        u0, dx=dx, dt=dt, n_steps=50, c_uxx=-0.4, c_uxxxx=-0.05
    )
    dev_u = full_traj[:25]
    holdout_u = full_traj[25:]

    res = discover_higher_order_pde(
        dev_u,
        dt=dt,
        dx=dx,
        temporal_order=1,
        max_spatial_order=4,
        periodic=True,
        include_nonlinear=False,
        sparsity_threshold=0.03,
    )
    assert res.max_spatial_order == 4
    assert "u_xx" in res.active_terms
    assert "u_xxxx" in res.active_terms
    assert res.r2_dev > 0.999

    referee = HigherOrderPdeReferee(min_holdout_r2=0.99)
    verdict_good = referee.adjudicate(res, holdout_u, dt=dt, dx=dx, periodic=True)
    assert verdict_good.verdict == EpistemicPdeVerdict.IDENTIFIED_CAUSAL_LAW
    assert verdict_good.ultraviolet_well_posed

    # Now test that a positive +0.05 * u_xxxx coefficient is caught by the ultraviolet blowup guard
    ill_posed_res = HigherOrderPdeDiscoveryResult(
        temporal_order=1,
        target_lhs="u_t",
        formula="u_t = +0.0500*u_xxxx",
        active_terms=("u_xxxx",),
        coefficients=(+0.05,),
        r2_dev=0.99,
        residual_rmse=0.001,
        bic=-100.0,
        spatial_dimension=1,
        max_spatial_order=4,
    )
    verdict_bad = referee.adjudicate(ill_posed_res, holdout_u, dt=dt, dx=dx, periodic=True)
    assert not verdict_bad.ultraviolet_well_posed
    assert verdict_bad.verdict == EpistemicPdeVerdict.PREDICTIVE_WRONG_STRUCTURE


def test_second_order_temporal_klein_gordon_wave_discovery():
    """Verify automatic BIC selection of 2nd-order temporal PDE: u_tt = 1.5 u_xx - 0.5 u."""
    n_x = 64
    length = 2.0 * np.pi
    dx = length / n_x
    x = np.linspace(0.0, length, n_x, endpoint=False)
    dt = 0.01

    # Exact standing/traveling wave solution of u_tt = c^2 u_xx - m^2 u:
    # For mode k, omega_k = sqrt(c^2 k^2 + m^2)
    c_sq = 1.5
    m_sq = 0.5
    omega1 = np.sqrt(c_sq * (1.0 ** 2) + m_sq)
    omega2 = np.sqrt(c_sq * (2.0 ** 2) + m_sq)

    t_grid = np.arange(50) * dt
    u_traj = (
        np.outer(np.cos(omega1 * t_grid), np.sin(x))
        + 0.6 * np.outer(np.sin(omega2 * t_grid), np.cos(2.0 * x))
    )

    # Let discover_higher_order_pde choose between temporal_order=1 (u_t) and temporal_order=2 (u_tt)
    res = discover_higher_order_pde(
        u_traj[:30],
        dt=dt,
        dx=dx,
        temporal_order=None,
        max_spatial_order=4,
        periodic=True,
        include_nonlinear=False,
        sparsity_threshold=0.04,
    )
    assert res.temporal_order == 2
    assert res.target_lhs == "u_tt"
    assert "u_xx" in res.active_terms
    assert "u" in res.active_terms
    assert res.r2_dev > 0.999

    referee = HigherOrderPdeReferee(min_holdout_r2=0.99)
    verdict = referee.adjudicate(res, u_traj[25:], dt=dt, dx=dx, periodic=True)
    assert verdict.verdict == EpistemicPdeVerdict.IDENTIFIED_CAUSAL_LAW
    assert verdict.frozen_oos_r2 > 0.999


def test_2d_biharmonic_field_with_jetflow_tensor_train_compression():
    """Verify 2D higher-order PDE discovery coupled with 4D JetFlow Tensor-Train compression."""
    n_x, n_y = 16, 16
    length = 2.0 * np.pi
    dx = length / n_x
    x = np.linspace(0.0, length, n_x, endpoint=False)
    y = np.linspace(0.0, length, n_y, endpoint=False)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    dt = 0.01

    # Mode (kx=1, ky=1): laplacian(u) = -2 u, biharmonic_lap2(u) = +4 u
    # Mode (kx=2, ky=1): laplacian(u) = -5 u, biharmonic_lap2(u) = +25 u
    # Governing PDE: u_t = 0.2 * laplacian(u) - 0.02 * biharmonic_lap2(u)
    lam_11 = 0.2 * (-2.0) - 0.02 * 4.0
    lam_21 = 0.2 * (-5.0) - 0.02 * 25.0

    t_steps = 20
    field_2d = np.zeros((t_steps, n_y, n_x), dtype=np.float64)
    for ti in range(t_steps):
        t = ti * dt
        field_2d[ti] = (
            np.exp(lam_11 * t) * np.sin(xx) * np.cos(yy)
            + 0.7 * np.exp(lam_21 * t) * np.cos(2.0 * xx) * np.sin(yy)
        )

    # Stack into a 4D multi-channel video/field tensor (T, C=3, H, W)
    video_4d = np.stack([field_2d, 0.5 * field_2d, -0.3 * field_2d], axis=1)

    proposer = HigherOrderPdeProposer(
        dt=dt,
        dx=dx,
        temporal_order=1,
        max_spatial_order=4,
        include_nonlinear=False,
        sparsity_threshold=0.03,
    )
    res, proposals, tt_summary = proposer.propose_from_field(video_4d[:12], compress_tt=True)
    assert len(proposals) == 1
    assert tt_summary is not None
    assert tt_summary["compression_ratio"] > 1.0
    assert "laplacian(u)" in res.active_terms
    assert "biharmonic_lap2(u)" in res.active_terms
    assert res.r2_dev > 0.999

    referee = HigherOrderPdeReferee(min_holdout_r2=0.99)
    verdict = referee.adjudicate(res, field_2d[10:], dt=dt, dx=dx)
    assert verdict.verdict == EpistemicPdeVerdict.IDENTIFIED_CAUSAL_LAW
    assert verdict.frozen_oos_r2 > 0.999
