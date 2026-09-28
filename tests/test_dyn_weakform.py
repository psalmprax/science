"""D2a weak-form tests: compactness, exactness, paired A/B advantage, gating."""
import numpy as np

from apedp_dyn import (
    enumerate_candidates,
    materialize,
    prune_admissible,
    run_all_gates,
)
from apedp_dyn.estimation import (
    boundary_term,
    bump,
    bump_deriv,
    make_windows,
    spline_coefficients,
    weak_coefficients,
)
from apedp_dyn.model import PhysicalModel
from apedp_dyn.problems import mass_spring_damper as msd


def _msd_setup():
    prob = msd.load()
    reg = prob.train
    t, Y = reg["truth_t"], reg["truth_y"]
    U = np.array([reg["u_of_t"](ti) for ti in t]).reshape(-1, 1)
    cands = enumerate_candidates(required=("u",), optional=("x", "v"), max_terms=3)
    truth = [s for s in prune_admissible(prob, cands)
             if set(s.candidate.term_set()) == {"u", "x", "v"}][0]
    return prob, t, Y, U, truth


def _noisy(Y, pct, seed=11):
    rng = np.random.default_rng(seed)
    return Y + rng.normal(0, pct / 100 * np.std(Y, axis=0), Y.shape)


def test_bump_is_compact_and_edges_vanish():
    tau = np.linspace(-2, 2, 401)
    assert np.all(bump(tau[np.abs(tau) >= 1.0]) == 0.0)
    assert np.all(bump_deriv(tau[np.abs(tau) >= 1.0]) == 0.0)
    prob, t, Y, U, _ = _msd_setup()
    w = make_windows((t[0], t[-1]))
    for j in (0, len(w.centers) - 1):
        assert abs(w.phi(t, j)[0]) == 0.0 and abs(w.phi(t, j)[-1]) == 0.0
        assert abs(w.phi_dot(t, j)[0]) == 0.0 and abs(w.phi_dot(t, j)[-1]) == 0.0


def test_boundary_term_retained_explicitly_when_nonzero():
    assert boundary_term((1.0, 2.0), (0.0, 0.0)) == 0.0
    assert boundary_term((1.0, 2.0), (0.5, 0.25)) == 2.0 * 0.25 - 1.0 * 0.5


def test_weak_exact_at_zero_noise():
    # Quadrature-limited exactness (measured 8.5e-08 on a0); assert with margin.
    prob, t, Y, U, truth = _msd_setup()
    est = weak_coefficients(prob, truth.candidate, t, Y, U)
    assert abs(est["a0"] - 1.0) < 1e-6, est
    assert abs(est["a1"] - (-4.0)) < 1e-6, est
    assert abs(est["a2"] - (-0.2)) < 1e-6, est


def test_weak_beats_spline_at_1pct_paired():
    # Same noise realization (seed 11); measured 2026-09-28: weak 0.0035,
    # spline 0.0108 max relative error. ~3x gap, asserted with margin.
    prob, t, Y, U, truth = _msd_setup()
    Yn = _noisy(Y, 1.0)
    s = spline_coefficients(prob, truth.candidate, t, Yn, U)
    w = weak_coefficients(prob, truth.candidate, t, Yn, U)
    se = max(abs(s["a0"] - 1), abs(s["a1"] + 4) / 4, abs(s["a2"] + 0.2) / 0.2)
    we = max(abs(w["a0"] - 1), abs(w["a1"] + 4) / 4, abs(w["a2"] + 0.2) / 0.2)
    assert we < 0.01, (we, w)
    assert we < se, (we, se)


def test_weak_fitted_model_passes_all_gates():
    # No bypass: weak-sized models run the full gate chain like any estimate.
    import dataclasses

    prob, t, Y, U, truth = _msd_setup()
    est = weak_coefficients(prob, truth.candidate, t, Y, U)
    model0 = materialize(prob, truth.candidate)
    model = dataclasses.replace(
        model0, parameters=tuple(dataclasses.replace(p, value=est[p.name])
                                 for p in model0.parameters))
    results = run_all_gates(model, prob)
    assert all(r.passed for r in results), [(r.name, r.detail) for r in results]
