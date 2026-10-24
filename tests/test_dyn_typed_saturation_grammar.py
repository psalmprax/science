"""Typed-grammar saturation pins: dimensioned binding laws.

The saturation family exists in the IR vocabulary, but `sat_x`/`inv_x2` are
dimensionless-only: `1 + X` mixes a number with a length and gate 1 rejects it.
These pins cover the expansion that makes the *dimensioned* law reachable --
`xddot = a * x/(L + |x|)` (Langmuir/Michaelis-Menten) and its Hill n=2 form,
where `L` is a fitted reference level in the state's units.
"""
from __future__ import annotations

import pytest

import numpy as np

from apedp_dyn import fit_candidate
from apedp_dyn.grammar import (
    CandidateDynamics,
    Term,
    materialize,
    prune_admissible,
    saturation_candidate,
    validate,
)
from apedp_dyn.model import DynamicalEquation, InputVariable, Parameter, StateVariable
from apedp_dyn.problems import mass_spring_damper as msd
from apedp_dyn.problems.base import DynProblem, finalize, make_signal


def _planted_langmuir_problem() -> DynProblem:
    # Start inside the knee (x0 comparable to s) so the reference level is
    # constrained by the trajectory, not only by the saturated tail.
    states = (StateVariable("x", "meter", 0.3),
              StateVariable("v", "meter/second", 0.0))
    inputs = (InputVariable("u", "kilogram*meter/second**2"),)
    parameters = (Parameter("a", "1/second**2", 2.0),
                  Parameter("s", "meter", 0.3))
    truth = (DynamicalEquation("x", "v"),
             DynamicalEquation("v", "-a*x/(s + Abs(x))"))
    train = {"t_span": (0.0, 10.0), "u_of_t": make_signal({"kind": "zero"}),
             "x0": (0.3, 0.0)}
    holdout = {"t_span": (0.0, 10.0), "u_of_t": make_signal({"kind": "zero"}),
               "x0": (0.3, 0.0), "tolerance": "auto"}
    return finalize(DynProblem(
        "T-LANGMUIR", "langmuir drag", states, inputs, parameters,
        truth, train, holdout))


def test_saturation_term_carries_its_reference_scale_in_state_units():
    model = materialize(msd.load(), saturation_candidate())
    assert [p.name for p in model.parameters] == ["a0", "s0"]
    assert [p.role for p in model.parameters] == ["rate", "scale"]
    assert model.parameters[1].units == "meter"
    assert "Abs" in model.equations[1].expression
    assert "s0" in model.equations[1].expression


def test_saturation_candidate_passes_gate1_and_gate2():
    prob = msd.load()
    kept = prune_admissible(prob, [saturation_candidate()])
    assert len(kept) == 1
    assert kept[0].provenance["gate1_status"]["passed"]
    assert kept[0].provenance["gate2_status"]["passed"]


def test_hill_term_is_dimensionally_valid():
    model = materialize(msd.load(), saturation_candidate(("hill2_bind",)))
    assert [p.role for p in model.parameters] == ["rate", "scale"]
    assert model.parameters[1].units == "meter"


def test_planted_langmuir_law_is_fitted_by_the_trajectory_stage():
    """Reachability pin: the scale parameter is estimated, not left at a guess.

    One trajectory identifies (a, s) weakly (outside the knee, `s` only
    rescales the force), so this pins the fitted law and the scale's order of
    magnitude -- not machine-precision parameter recovery, which the
    proposer-level pins cover from noiseless data.
    """
    prob = _planted_langmuir_problem()
    scored = fit_candidate(prob, saturation_candidate())
    assert scored.train_loss is not None, scored.provenance
    assert scored.train_loss < 2e-4, scored.train_loss
    fitted = scored.fit.param_values
    assert set(fitted) == {"a0", "s0"}
    assert -3.0 < fitted["a0"] < -1.5, fitted
    assert 0.2 < fitted["s0"] < 0.8, fitted


def test_scaled_term_requires_its_scale_parameter():
    with pytest.raises(ValueError, match="requires a scale_parameter"):
        validate(CandidateDynamics(lhs="xddot", terms=(Term("langmuir_bind", "a0"),)))


def test_plain_term_rejects_a_scale_parameter():
    with pytest.raises(ValueError, match="does not take a scale_parameter"):
        validate(CandidateDynamics(
            lhs="xddot", terms=(Term("x", "a0", scale_parameter="s0"),)))


def test_dimensionless_saturation_terms_stay_out_of_dimensioned_problems():
    """`sat_x` is unchanged vocabulary: `1 + X` is still a gate-1 violation.

    This is why the expansion needed a reference scale rather than another
    fixed-constant saturation lambda.
    """
    cand = CandidateDynamics(lhs="xddot", terms=(Term("sat_x", "a0"),))
    kept = prune_admissible(msd.load(), [cand])
    assert kept == []
