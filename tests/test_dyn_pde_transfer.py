"""Frozen-coefficient PDE transfer pins.

The existing adapter re-fits coefficients on the holdout partition, so its
`holdout_r2` is an optimistic re-fit metric. These tests pin the true
forward-transfer path (`apedp_dyn.epistemic.pde_transfer`): dev-discovered
coefficients are frozen and scored on the holdout window with the same
stencils as `rime/pde.rs`.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

import apedp_kernel as ak
from apedp_dyn.epistemic.pde_transfer import evaluate_frozen_transfer

DT = 0.002
DX = 0.1


def _simulate_burgers(nx: int = 60, nt: int = 50, nu: float = 0.1) -> np.ndarray:
    """Explicit scheme identical to the Rust Burgers test fixture."""
    u = np.zeros((nt, nx))
    x = np.arange(nx) * DX
    u[0] = np.sin(x)
    for i in range(nt - 1):
        adv = -u[i, 1:-1] * (u[i, 2:] - u[i, :-2]) / (2.0 * DX)
        diff = nu * (u[i, 2:] - 2.0 * u[i, 1:-1] + u[i, :-2]) / (DX * DX)
        u[i + 1, 1:-1] = u[i, 1:-1] + DT * (adv + diff)
        u[i + 1, 0] = u[i + 1, 1]
        u[i + 1, -1] = u[i + 1, -2]
    return u


def _discover(u: np.ndarray):
    cfg = {
        "max_spatial_order": 2,
        "max_polynomial_degree": 2,
        "include_advection_terms": True,
        "sparsity_threshold": 0.01,
        "ridge_alpha": 1e-6,
    }
    return json.loads(
        ak.rime_pde_discover_json(json.dumps(u.tolist()), DT, DX, json.dumps(cfg))
    )


def _heat_grid(nx: int = 60, nt: int = 50) -> np.ndarray:
    x = np.arange(nx) * DX
    t = np.arange(nt) * DT
    return np.exp(-t)[:, None] * np.sin(x)[None, :]


def test_heat_equation_frozen_transfer_exact():
    u = _heat_grid()
    n_dev = max(5, int(u.shape[0] * 0.75))
    dev = _discover(u[:n_dev])
    # u_t = -u = u_xx is degenerate, so the sparse solver may keep either or
    # both spellings; any active set must stay inside the equivalent pair.
    assert set(dev["active_terms"]) <= {"u", "u_xx"}, dev["active_terms"]
    hold = evaluate_frozen_transfer(
        u[n_dev - 1 :], DT, DX, dev["active_terms"], dev["coefficients"]
    )
    assert hold.r2 > 0.999, hold


def test_frozen_transfer_penalizes_wrong_coefficients():
    u = _heat_grid()
    n_dev = max(5, int(u.shape[0] * 0.75))
    dev = _discover(u[:n_dev])
    exact = evaluate_frozen_transfer(
        u[n_dev - 1 :], DT, DX, dev["active_terms"], dev["coefficients"]
    )
    doubled = evaluate_frozen_transfer(
        u[n_dev - 1 :], DT, DX, dev["active_terms"], [2.0 * c for c in dev["coefficients"]]
    )
    assert doubled.r2 < exact.r2 - 0.5, (doubled, exact)
    assert doubled.residual_rmse > exact.residual_rmse * 5.0


def test_burgers_frozen_transfer_survives_holdout_partition():
    u = _simulate_burgers()
    n_dev = max(5, int(u.shape[0] * 0.75))
    dev = _discover(u[:n_dev])
    assert "u*u_x" in dev["active_terms"], dev["active_terms"]
    assert "u_xx" in dev["active_terms"], dev["active_terms"]
    adv_idx = dev["active_terms"].index("u*u_x")
    diff_idx = dev["active_terms"].index("u_xx")
    assert abs(dev["coefficients"][adv_idx] - (-1.0)) < 0.15
    assert abs(dev["coefficients"][diff_idx] - 0.1) < 0.15

    frozen = evaluate_frozen_transfer(
        u[n_dev - 1 :], DT, DX, dev["active_terms"], dev["coefficients"]
    )
    refit_raw = _discover(u[n_dev - 1 :])
    assert frozen.r2 > 0.9, frozen
    assert frozen.n_samples > 100
    # The frozen metric is the honest one: report the re-fit alongside it in
    # failure messages so any future divergence is visible, not hidden.
    assert frozen.r2 <= refit_raw["r2"] + 0.02, (frozen.r2, refit_raw["r2"])


def test_unknown_term_rejected():
    u = _heat_grid()
    with pytest.raises(ValueError, match="unknown PDE term"):
        evaluate_frozen_transfer(u, DT, DX, ["u_bb"], [1.0])
