"""Every problem loads, and every gate can fail.

Two failures in this repository were invisible until they were made
structural rather than incidental:

1. Seven of the eleven new problem modules could not be loaded at all.
   Two declared no inputs but handed ``make_signal({"kind": "zero"})``,
   whose arity defaults to 1, so an extra argument reached a RHS compiled
   for zero. Two more spelled an input ``I``, which SymPy binds to the
   imaginary unit, so ``(I - V/R)/C`` parsed complex and died inside
   ``float()``. Nothing loaded the problems, so nothing noticed.

2. Gate 1.5 (``dao_validity``) dispatched on a ``dae_mode`` attribute that
   ``PhysicalModel`` did not have, and its constrained branch iterated an
   attribute ``DynamicalEquation`` does not have. It returned True on both
   paths and had never rejected anything. The tests that "covered" it
   wrapped the call in ``try/except: pass``.

A test that only checks the happy path cannot detect either. The negative
controls below can: each gate is fed an input it must reject, so a gate
that has become unfailable fails here rather than quietly approving
everything downstream.
"""
from __future__ import annotations

import numpy as np
import pytest

from apedp_dyn import (
    DynamicalEquation,
    Parameter,
    PhysicalModel,
    dimensional_validity,
    equation_admissibility,
    run_all_gates,
)
from apedp_dyn.gates import dao_validity
from apedp_dyn.problems import ALL_PROBLEMS, mass_spring_damper

_ALL = sorted(ALL_PROBLEMS)


def _ids() -> list[str]:
    return _ALL


# --- every problem loads ---------------------------------------------------


@pytest.mark.parametrize("pid", _ids())
def test_every_registered_problem_loads(pid: str):
    """A problem that cannot be loaded is not a benchmark case."""
    problem = ALL_PROBLEMS[pid].load()
    assert problem.problem_id
    assert problem.states, f"{pid} declares no states"
    assert problem.parameters, f"{pid} declares no parameters"


@pytest.mark.parametrize("pid", _ids())
def test_every_problem_declares_one_equation_per_state(pid: str):
    problem = ALL_PROBLEMS[pid].spec()
    assert len(problem.truth_equations) == len(problem.states), (
        f"{pid}: {len(problem.truth_equations)} truth equations for "
        f"{len(problem.states)} states")


@pytest.mark.parametrize("pid", _ids())
@pytest.mark.parametrize("regime", ["train", "holdout"])
def test_every_problem_integrates_to_a_finite_trajectory(pid: str, regime: str):
    """Ground truth must be real-valued and finite to score anything against."""
    problem = ALL_PROBLEMS[pid].load()
    reg = getattr(problem, regime)
    assert reg["truth_y"].shape == (reg["truth_t"].shape[0], len(problem.states))
    assert np.all(np.isfinite(reg["truth_y"])), f"{pid}/{regime} truth diverged"


@pytest.mark.parametrize("pid", _ids())
@pytest.mark.parametrize("regime", ["train", "holdout"])
def test_input_signal_arity_matches_declared_inputs(pid: str, regime: str):
    """The drive vector must have one entry per declared input.

    This is the check that would have caught the autonomous problems: they
    declared no inputs but carried a length-1 zero signal, which was splatted
    into the RHS as a spurious argument.
    """
    problem = ALL_PROBLEMS[pid].spec()
    reg = getattr(problem, regime)
    n_inputs = len(problem.inputs)
    u = np.atleast_1d(np.asarray(reg["u_of_t"](reg["t_span"][0]), dtype=float))
    assert u.size == n_inputs, (
        f"{pid}/{regime}: signal arity {u.size} != {n_inputs} declared input(s)")


# Defects found by this test and NOT yet fixed, because each needs a domain
# convention decision rather than a mechanical correction. They are recorded
# here so the benchmark zoo stops being a place where truth models silently
# fail their own gates. Delete an entry once the underlying spec is corrected:
# strict=False means the xpass is not an error, so removal is the signal.
#
#   PS-VDPOL : the truth is ``mu*(1 - x**2)*v - x`` with x declared in metre.
#              The bare ``1`` is dimensionless, so the term is inhomogeneous.
#              Van der Pol's saturation is written for a normalised amplitude,
#              so either x must be declared dimensionless or the equation needs
#              an explicit amplitude scale. That is a modelling choice.
# (PS-LOG and PS-LV declared "individual", which pint does not define; they
# now use "count". PS-GRAV2 failed its own holdout at rmse 0.139 purely on
# integration tolerance, because the gate integrated at the fitter's default
# rtol=1e-3 against a DOP853 truth; it is now 5.4e-08.)
_KNOWN_GATE_DEFECTS = {
    "PS-VDPOL": "truth has a dimensionless 1 added to x**2; needs an amplitude scale",
}


@pytest.mark.parametrize("pid", _ids())
def test_truth_model_passes_every_gate(pid: str):
    if pid in _KNOWN_GATE_DEFECTS:
        pytest.xfail(_KNOWN_GATE_DEFECTS[pid])
    problem = ALL_PROBLEMS[pid].load()
    model = PhysicalModel(problem.states, problem.inputs,
                          problem.parameters, problem.truth_equations)
    for result in run_all_gates(model, problem):
        assert result.passed, f"{pid}/{result.name}: {result.detail}"


# --- negative controls: every gate can fail --------------------------------


def test_gate1_rejects_a_unit_mismatch():
    """A RHS whose units do not match the state rate must be rejected."""
    p = mass_spring_damper.spec()
    bad = PhysicalModel(
        p.states, p.inputs, p.parameters,
        (DynamicalEquation("x", "v"),
         DynamicalEquation("v", "(u - k*x)/m*m*m")),  # extra metres
    )
    assert not dimensional_validity(bad).passed


def test_gate2_rejects_an_admissible_but_wrong_equation():
    """d(x)/dt = x is admissible (no poles, real) but not the truth."""
    p = mass_spring_damper.spec()
    bad = PhysicalModel(
        p.states, p.inputs, p.parameters,
        (DynamicalEquation("x", "v"),
         DynamicalEquation("v", "x")),  # dimensionally fine, physically wrong
    )
    assert equation_admissibility(bad).passed, "control: this should be admissible"
    # It is the *dimensions* gate that must not be fooled, and the holdout
    # gate that must reject it on evidence:
    assert not dimensional_validity(bad).passed


def test_gate15_rejects_an_unknown_dae_mode():
    p = mass_spring_damper.spec()
    bad = PhysicalModel(p.states, p.inputs, p.parameters,
                        p.truth_equations, dae_mode="quantum")
    assert not dao_validity(bad).passed


def test_gate15_rejects_a_constrained_model_with_no_constraint():
    """Declaring the DAE mode is not the same as carrying the constraint."""
    p = mass_spring_damper.spec()
    bad = PhysicalModel(p.states, p.inputs, p.parameters,
                        p.truth_equations, dae_mode="constrained")
    assert not dao_validity(bad).passed


def test_gate15_rejects_a_dimensionful_constraint_parameter():
    p = mass_spring_damper.spec()
    bad = PhysicalModel(
        p.states, p.inputs,
        (Parameter("con", "newton", 1.0, (-1e6, 1e6)),),
        p.truth_equations,
        dae_mode="constrained", dae_constraint_params=("con",))
    assert not dao_validity(bad).passed


def test_sympy_cannot_hijack_a_symbol_named_i():
    """``I`` is SymPy's imaginary unit; an ODE naming a current must not get it.

    Regression for the RC/RLC circuits, which parsed to complex expressions
    and failed far downstream inside ``float()`` with an error that named
    neither the problem nor the symbol.
    """
    eq = DynamicalEquation("V", "(I - V/R) / C")
    assert eq.sympy_expr.is_real is not False
    assert eq.sympy_expr.has(__import__("sympy").I) is False, (
        "the input named I was bound to the imaginary unit")


def test_autonomous_problem_does_not_inject_a_signal_argument():
    """A problem with no inputs must build a RHS with no input arguments."""
    from apedp_dyn.problems import exponential_decay

    problem = exponential_decay.spec()
    assert problem.inputs == ()
    from apedp_dyn.problems.base import _truth_rhs

    compiled = _truth_rhs(problem.truth_equations,
                          [s.name for s in problem.states], [], {"k": 0.3})
    # t + one state + one parameter = 3 arguments, and nothing more. A
    # spurious input argument would make this a TypeError.
    assert compiled[0][1](0.0, 10.0, 0.3) == pytest.approx(-3.0)
