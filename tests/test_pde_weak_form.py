"""Weak-form PDE identification (apedp_dyn/pde/weak_form.py) and its proposer option."""
from __future__ import annotations

import numpy as np
import pytest

from apedp_dyn.pde import weak_form


def _heat(noise: float, seed: int = 0, diffusion: float = 0.8):
    nx, nt, dx, dt = 64, 400, 0.1, 0.002
    x, t = np.arange(nx) * dx, np.arange(nt) * dt
    k1, k2 = 2 * np.pi / (nx * dx), 4 * np.pi / (nx * dx)
    u = (np.exp(-diffusion * k1 ** 2 * t)[:, None] * np.sin(k1 * x)[None]
         + 0.5 * np.exp(-diffusion * k2 ** 2 * t)[:, None] * np.cos(k2 * x)[None])
    return u + np.random.default_rng(seed).normal(0, 1, u.shape) * noise * u.std(), dt, dx


def test_the_heat_equation_is_identified_through_ten_percent_noise():
    for noise in (0.0, 0.01, 0.10):
        for seed in range(4):
            found = weak_form.identify(*_heat(noise, seed))
            assert set(found) == {"u_xx"}, (noise, seed, found)
            assert abs(found["u_xx"] - 0.8) < 0.05


def test_the_selection_does_not_depend_on_units():
    u, dt, dx = _heat(0.05)
    a = weak_form.identify(u, dt, dx)
    b = weak_form.identify(1000.0 * u, dt * 60.0, dx * 0.01)  # other units for u, t and x
    assert set(a) == set(b) == {"u_xx"}
    assert b["u_xx"] == pytest.approx(a["u_xx"] * 0.01 ** 2 / 60.0, rel=1e-6)


def test_integration_by_parts_is_exact_on_a_polynomial_field():
    """u = x^2 t: u_t = x^2 and u_xx = 2t, so the u_t and u_xx columns must match their closed forms."""
    x, t = np.linspace(0, 1, 41), np.linspace(0, 1, 81)
    u = t[:, None] * x[None, :] ** 2
    G, b = weak_form.weak_system(u, t[1] - t[0], x[1] - x[0], window_t=21, window_x=21, stride_t=20, stride_x=10)
    names = [n for n, _p, _a in weak_form.LIBRARY]
    G_dx2, _b_x2 = weak_form.weak_system(np.broadcast_to(x[None, :] ** 2, u.shape).copy(), t[1] - t[0], x[1] - x[0],
                                        window_t=21, window_x=21, stride_t=20, stride_x=10)
    assert np.allclose(b, G_dx2[:, names.index("u")], rtol=2e-3)  # -int(phi_t u) = int(phi x^2)
    G_2t, _ = weak_form.weak_system(np.broadcast_to(2 * t[:, None], u.shape).copy(), t[1] - t[0], x[1] - x[0],
                                    window_t=21, window_x=21, stride_t=20, stride_x=10)
    assert np.allclose(G[:, names.index("u_xx")], G_2t[:, names.index("u")], rtol=2e-3)


def test_a_field_too_small_for_a_test_function_is_refused():
    with pytest.raises(ValueError, match="too small"):
        weak_form.weak_system(np.ones((4, 4)), 0.1, 0.1)
    with pytest.raises(ValueError, match="times, positions"):
        weak_form.weak_system(np.ones(10), 0.1, 0.1)


def test_the_proposer_option_emits_the_weak_form_law_and_leaves_the_default_alone():
    pd = pytest.importorskip("pandas")
    from apedp_dyn.pde.proposer import PdeProposer

    u, dt, dx = _heat(0.05)
    frame = pd.DataFrame({"x": np.tile(np.arange(u.shape[1]) * dx, u.shape[0]), "u": u.ravel()})
    proposal = PdeProposer(dt=dt, dx=dx, estimator="weak_form").fit_and_propose(frame)[0]
    assert list(proposal.parameters) == ["u_xx"] and abs(proposal.parameters["u_xx"] - 0.8) < 0.05
    assert proposal.metadata["estimator"] == "weak_form" and proposal.metadata["r2_dev"] > 0.9
    assert proposal.formula.startswith("u_t = ")
    assert PdeProposer()._estimator == "kernel"
    with pytest.raises(ValueError, match="estimator"):
        PdeProposer(estimator="spline")
    no_grid = pd.DataFrame({"u": u.ravel()})
    assert PdeProposer(estimator="weak_form").fit_and_propose(no_grid) == ()
