"""Planck radiance limitation pin.

Mathematical fact: for u(v) = v^3 / (e^v - 1),
    u'/u = 3/v - e^v / (e^v - 1).
If a polynomial-coefficient relation P(v) u' + Q(v) u = 0 held exactly, then
u'/u = -Q/P would be rational in v, forcing e^v into Q(v) -- contradicting
the transcendence of e^v over Q(v) (Hermite). So NO exact first-order
polynomial ODE exists for Planck radiance.

The honest engine behavior is therefore: return the best approximate jet
ODE, report a small jet residual, and set is_exact_recovery = False
(correct abstention from an exact-recovery claim). This test pins that the
engine does not manufacture an exact recovery where none exists.
"""
from __future__ import annotations

import json

import numpy as np

import apedp_kernel as ak


def test_planck_returns_approximate_jet_ode_without_exact_claim():
    v = np.linspace(0.1, 6.0, 60)
    u = (v ** 3) / (np.exp(v) - 1.0)
    cfg = {
        "max_x_degree": 2,
        "max_diff_order": 1,
        "smooth_window": 5,
        "sparsity_threshold": 0.05,
        "rcond_threshold": 1e-12,
    }
    res = json.loads(ak.rime_diff_pade_discover_json(
        json.dumps(v.tolist()), json.dumps(u.tolist()), json.dumps(cfg)
    ))
    # An approximate jet relation is found (small residual on the jet bundle).
    assert res["jet_rmse"] < 0.05
    # But no exact recovery may be claimed: none exists mathematically.
    assert res["is_exact_recovery"] is False
    assert res["integrated_trajectory_r2"] < 0.99
