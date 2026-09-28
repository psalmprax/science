"""Merge tests: ports from apedp/dyn into the unified apedp_dyn home.

Covers compiled RHS equivalence, the square-window third arm, multisine
excitation, noise-floor-aware holdout tolerance, and energy validators.
apedp/dyn itself is retired by this commit; its duplicated MSD/pendulum
roundtrips are superseded by test_dyn_problems/test_dyn_fitting.
"""
import numpy as np
from scipy.integrate import solve_ivp

from apedp_dyn import (
    DynamicalEquation,
    InputVariable,
    Parameter,
    PhysicalModel,
    StateVariable,
    auto_tolerance,
    compile_rhs,
    enumerate_candidates,
    make_signal,
    materialize,
    prune_admissible,
    simulate,
    square_window_coefficients,
)
from apedp_dyn.problems import mass_spring_damper as msd
from apedp_dyn.problems.base import DynProblem, finalize
from apedp_dyn.validation import (
    check_conservative_core,
    check_damped_decay,
    simulate_pair,
)


def _msd_truth():
    prob = msd.load()
    reg = prob.train
    t, Y = reg["truth_t"], reg["truth_y"]
    U = np.array([reg["u_of_t"](ti) for ti in t]).reshape(-1, 1)
    cands = enumerate_candidates(required=("u",), optional=("x", "v"), max_terms=3)
    truth = [s for s in prune_admissible(prob, cands)
             if set(s.candidate.term_set()) == {"u", "x", "v"}][0]
    return prob, t, Y, U, truth


def test_compiled_rhs_matches_reference():
    prob, t, Y, U, truth = _msd_truth()
    model = materialize(prob, truth.candidate)
    vals = {"a0": 1.0, "a1": -4.0, "a2": -0.2}
    import dataclasses

    model = dataclasses.replace(
        model, parameters=tuple(dataclasses.replace(p, value=vals[p.name])
                                for p in model.parameters))
    reg = prob.train
    ref = solve_ivp(lambda tt, x: model.rhs(tt, x, reg["u_of_t"]),
                    reg["t_span"], np.asarray(reg["x0"]), method="LSODA",
                    t_eval=t)
    got = simulate(model, reg["t_span"], reg["u_of_t"],
                   np.asarray(reg["x0"]), n_eval=len(t))
    assert got.success
    assert np.allclose(got.y, ref.y, rtol=1e-8, atol=1e-10)


def test_square_window_arm_recovers_msd():
    prob, t, Y, U, truth = _msd_truth()
    est = square_window_coefficients(prob, truth.candidate, t, Y, U)
    assert abs(est["a0"] - 1.0) < 0.05, est
    assert abs(est["a1"] - (-4.0)) / 4.0 < 0.05, est
    assert abs(est["a2"] - (-0.2)) / 0.2 < 0.05, est
    rng = np.random.default_rng(11)
    Yn = Y + rng.normal(0, 0.01 * np.std(Y, axis=0), Y.shape)
    est1 = square_window_coefficients(prob, truth.candidate, t, Yn, U)
    assert abs(est1["a0"] - 1.0) < 0.06, est1
    assert abs(est1["a1"] - (-4.0)) / 4.0 < 0.06, est1


def test_multisine_matches_formula():
    u = make_signal({"kind": "multisine",
                     "components": [[0.5, 2.0, 0.0], [0.4, 0.7, 1.0]]})
    for tt in (0.0, 1.0, 3.7):
        want = 0.5 * np.sin(2.0 * tt) + 0.4 * np.sin(0.7 * tt + 1.0)
        assert abs(float(u(tt)[0]) - want) < 1e-12


def test_auto_tolerance_tracks_noise_floor():
    t = np.linspace(0, 10, 400)
    clean = np.sin(t)
    assert auto_tolerance(clean) == max(0.10 * np.sqrt(np.mean(clean ** 2)), 0.0)
    rng = np.random.default_rng(3)
    noisy = clean + rng.normal(0, 0.05, t.shape)
    tol = auto_tolerance(noisy)
    assert tol >= 1.5 * 0.03, tol  # sees the floor
    assert tol <= 0.10 * np.sqrt(np.mean(noisy ** 2)) + 0.10, tol  # stays tight


def test_auto_tolerance_wiring_passes_truth():
    import dataclasses

    prob = msd.load()
    auto = dataclasses.replace(prob, holdout={**prob.holdout, "tolerance": "auto"})
    model = PhysicalModel(auto.states, auto.inputs, auto.parameters,
                          auto.truth_equations)
    from apedp_dyn import intervention_holdout

    res = intervention_holdout(model, auto)
    assert res.passed, res.detail


def _oscillator(c):
    states = (StateVariable("x", "meter", 0.5),
              StateVariable("v", "meter/second", 0.0))
    params = [Parameter("m", "kilogram", 1.0),
              Parameter("k", "kilogram/second**2", 4.0)]
    if c is not None:
        params.append(Parameter("c", "kilogram/second", c))
    eqs = (DynamicalEquation("x", "v"),
           DynamicalEquation("v", "(u - k*x)/m" if c is None else "(u - c*v - k*x)/m"))
    reg = {"t_span": (0.0, 10.0), "u_of_t": make_signal({"kind": "zero"}),
           "x0": (0.5, 0.0), "tolerance": 5e-2}
    return finalize(DynProblem("TM", "t", states,
                               (InputVariable("u", "kilogram*meter/second**2"),),
                               tuple(params), eqs, dict(reg), dict(reg)))


def test_conservation_core_and_impostor():
    und = _oscillator(None)
    t = und.train["truth_t"]
    xt, vt = und.train["truth_y"][:, 0], und.train["truth_y"][:, 1]
    ok, drift_truth = check_conservative_core(xt, vt, 1.0, 4.0)
    assert ok, drift_truth  # truth anchor at integrator precision
    core = PhysicalModel(und.states, und.inputs, und.parameters, und.truth_equations)
    x, v = simulate_pair(core, 1.0, 4.0, t, und.train["u_of_t"], (0.5, 0.0))
    ok, drift_core = check_conservative_core(x, v, 1.0, 4.0)
    assert ok, drift_core  # structural property, not parameter tolerance
    dmp = _oscillator(0.2)
    anti_params = tuple(Parameter(p.name, p.units,
                                  (-p.value if p.name == "c" else p.value))
                        for p in dmp.parameters)
    anti = PhysicalModel(dmp.states, dmp.inputs, anti_params, dmp.truth_equations)
    xa, va = simulate_pair(anti, 1.0, 4.0, t, dmp.train["u_of_t"], (0.5, 0.0))
    ok, drift_anti = check_conservative_core(xa, va, 1.0, 4.0)
    assert not ok and drift_anti > 0.1, drift_anti
    assert drift_anti > 100.0 * max(drift_core, 1e-15)


def test_damped_decay_balance():
    dmp = _oscillator(0.2)
    model = PhysicalModel(dmp.states, dmp.inputs, dmp.parameters, dmp.truth_equations)
    x, v = simulate_pair(model, 1.0, 4.0, dmp.train["truth_t"],
                         dmp.train["u_of_t"], (0.6, 0.0))
    ok, info = check_damped_decay(x, v, 1.0, 4.0)
    assert ok, info
