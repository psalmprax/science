"""D1.2/D1.3/D1.6 fitting tests: frozen IR + bounded trajectory least squares.

D1.6 regression scenarios: deliberately incomplete candidates must not bypass
the gates merely by fitting the training trajectory.
"""
import numpy as np
import pytest

from apedp_dyn import (
    DynamicalEquation,
    Parameter,
    StateVariable,
    enumerate_candidates,
    fit_candidate,
    prune_admissible,
    select_by_train_loss,
)
from apedp_dyn.problems import mass_spring_damper
from apedp_dyn.problems.base import DynProblem, finalize


def _term_param_map(scored):
    return {t.name: t.parameter for t in scored.candidate.terms}


def test_msd_chain_truth_wins_train_and_holdout():
    prob = mass_spring_damper.load()
    cands = enumerate_candidates(required=("u",), optional=("x", "v"), max_terms=3)
    kept = prune_admissible(prob, cands)
    assert len(kept) == 4  # {u},{x,u},{v,u},{x,v,u}
    fitted = [fit_candidate(prob, s.candidate) for s in kept]
    assert all(s.fit is not None and s.fit.success for s in fitted)
    by_terms = {tuple(sorted(s.candidate.term_set())): s for s in fitted}
    truth, incomplete = by_terms[("u", "v", "x")], by_terms[("u", "x")]
    # Nested models: truth fits train at least as well (no holdout peeking here).
    assert truth.train_loss <= incomplete.train_loss
    # Incomplete fits train finitely but gate 4 exposes the missing damping
    # under the forcing intervention.
    assert np.isfinite(incomplete.train_loss)
    assert truth.provenance["gate4_status"]["passed"]
    assert not incomplete.provenance["gate4_status"]["passed"]
    assert truth.holdout_rmse < 0.01 * incomplete.holdout_rmse
    assert select_by_train_loss(fitted).candidate is truth.candidate


def test_msd_lumped_maps_to_physical_where_identifiable():
    from apedp_dyn.problems.mass_spring_damper import resolve_physical

    assert resolve_physical(4.0, 0.2, 1.0) == {"m": 1.0, "c": 0.2, "k": 4.0}
    with pytest.raises(ValueError):
        resolve_physical(4.0, 0.2, 0.0)  # unexcited input: ratios only, no claim


def test_fitted_truth_recovers_msd_physical_parameters():
    prob = mass_spring_damper.load()
    cands = enumerate_candidates(required=("u",), optional=("x", "v"), max_terms=3)
    truth = next(s for s in prune_admissible(prob, cands)
                 if set(s.candidate.term_set()) == {"u", "x", "v"})
    res = fit_candidate(prob, truth.candidate)
    assert res.provenance["gate4_status"]["passed"]
    from apedp_dyn.problems.mass_spring_damper import resolve_physical

    m = _term_param_map(res)
    # Grammar emits a1*x + a2*v + a0*u = dv/dt; physics reads -αx - βv + γu.
    phys = resolve_physical(-res.fit.param_values[m["x"]],
                            -res.fit.param_values[m["v"]],
                            res.fit.param_values[m["u"]])
    # Trajectory-fit recovery (optimizer stops at max_nfev): sub-percent.
    assert abs(phys["m"] - 1.0) < 5e-3, phys
    assert abs(phys["c"] - 0.2) < 5e-3, phys
    assert abs(phys["k"] - 4.0) < 5e-3, phys


def _undriven_pendulum(amplitude: float) -> DynProblem:
    """D1.6 pendulum: truth θ¨=-(g/L)sinθ, no damping, no forcing."""
    states = (StateVariable("theta", "radian", amplitude),
              StateVariable("omega", "radian/second", 0.0))
    params = (Parameter("g", "meter/second**2", 9.81),
              Parameter("L", "meter", 1.0))
    truth = (DynamicalEquation("theta", "omega"),
             DynamicalEquation("omega", "-(g/L)*sin(theta)"))
    reg = {"t_span": (0.0, 10.0),
           "u_of_t": lambda t: np.zeros(0),
           "x0": (amplitude, 0.0), "tolerance": 5e-2}
    return finalize(DynProblem("D16-PEND", "undriven-pendulum", states, (),
                               params, truth,
                               dict(reg), dict(reg, x0=(1.5, 0.0))))


def test_pendulum_linearized_exposed_by_amplitude():
    # Small-angle train: linearized θ¨=-(g/L)θ mimics truth. Large-amplitude
    # holdout (sinθ≠θ) must expose it at gate 4 while truth passes.
    train_prob = _undriven_pendulum(0.2)
    truth_c = enumerate_candidates(required=(), optional=("sin_x",),
                                   max_terms=1, id_prefix="t")[1]
    lin_c = enumerate_candidates(required=(), optional=("x",),
                                 max_terms=1, id_prefix="l")[1]
    assert truth_c.term_set() == ("sin_x",) and lin_c.term_set() == ("x",)
    truth = fit_candidate(train_prob, truth_c)
    lin = fit_candidate(train_prob, lin_c)
    assert truth.provenance["gate4_status"]["passed"], truth.provenance
    # Linearized fits the small-angle train well — that is the trap.
    assert lin.fit.success and np.isfinite(lin.train_loss)
    assert not lin.provenance["gate4_status"]["passed"], lin.provenance
    # Measured 2026-09-28: truth 0.045 vs linearized 3.68 (81x separation).
    assert truth.holdout_rmse < 0.05 * lin.holdout_rmse


def test_fit_refuses_inadmissible_without_integrating():
    prob = mass_spring_damper.load()
    bad = enumerate_candidates(required=("sin_x",), optional=(), max_terms=1)[0]
    res = fit_candidate(prob, bad)
    assert res.fit is None
    assert "sin" in res.provenance["rejection_reason"], res.provenance
    assert res.train_loss is None and res.holdout_rmse is None


def test_parameter_binding_refuses_length_mismatch():
    """Estimates must never bind to the wrong parameter names."""
    from apedp_dyn.fitting.parameter_fit import _bind

    assert _bind(["a", "b"], [1.0, 2.0]) == {"a": 1.0, "b": 2.0}
    # A bare zip() truncates to the shorter side, silently attaching numbers
    # to the wrong parameters or dropping them; the mismatch must be loud.
    with pytest.raises(ValueError, match="parameter binding mismatch"):
        _bind(["a", "b"], [1.0])
    with pytest.raises(ValueError, match="parameter binding mismatch"):
        _bind(["a"], [1.0, 2.0])


def test_fit_candidate_rejects_wrong_length_warm_start():
    prob = mass_spring_damper.load()
    truth = next(s for s in prune_admissible(prob, enumerate_candidates(
        required=("u",), optional=("x", "v"), max_terms=3))
        if set(s.candidate.term_set()) == {"u", "x", "v"})
    assert len(truth.candidate.parameter_names()) == 3
    with pytest.raises(ValueError, match="parameter binding mismatch"):
        fit_candidate(prob, truth.candidate, x0=np.array([1.0, 2.0]))


def test_no_param_se_field():
    """Negative result, pinned: Jacobian linearized SEs were measured and
    REJECTED (13x undercoverage on the clean truth fit; up to ~1.8e5x on a
    noisy bad-basin fit). Shipping them as diagnostics would manufacture
    false uncertainty structure, so FitResult carries no SE field until a
    validated UQ method (bootstrap/profile) lands."""
    from apedp_dyn.fitting.parameter_fit import FitResult

    assert "param_se" not in FitResult.__dataclass_fields__


def test_multistart_never_worse_than_single():
    from apedp_dyn import enumerate_candidates, fit_candidate, prune_admissible
    from apedp_dyn.fitting.parameter_fit import MULTISTART_SCALES, fit_candidate_multistart
    from apedp_dyn.problems import duffing

    assert len(MULTISTART_SCALES) == 5 and MULTISTART_SCALES[0] == 1.0
    prob = duffing.load()
    cands = enumerate_candidates(required=("u",), optional=("x", "x3", "v"), max_terms=4)
    kept = {tuple(sorted(s.candidate.term_set())): s for s in prune_admissible(prob, cands)}
    ir = kept[("u", "v", "x", "x3")].candidate
    single = fit_candidate(prob, ir)
    best, info = fit_candidate_multistart(prob, ir)
    assert info["n_starts"] == 5
    assert best.train_loss <= single.train_loss + 1e-12
    assert set(best.candidate.term_set()) == set(ir.term_set())
