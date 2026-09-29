"""PS20 problem PS-RLC: second-order RLC circuit (coupled dynamics).

L dI/dt = V_in(t) - R*I - V_C
C dV_C/dt = I

First-order form (I, V_C):
    dI/dt   = (u - R*I - V_C) / L
    dV_C/dt = I / C

Train: step voltage at t=0.1.  Holdout: different step + changed R.
Exhibits underdamped oscillation when R < 2*sqrt(L/C).
"""
from apedp_dyn.model import DynamicalEquation, InputVariable, Parameter, StateVariable
from apedp_dyn.problems.base import DynProblem, finalize, make_signal

PROBLEM_ID = "PS-RLC"

_R = 10.0     # ohm
_L = 0.1      # henry
_C = 0.001    # farad


def spec() -> DynProblem:
    states = (
        StateVariable("I", "ampere", 0.0),
        StateVariable("Vc", "volt", 0.0),
    )
    inputs = (InputVariable("u", "volt"),)
    parameters = (
        Parameter("R", "ohm", _R, bounds=(0.1, 1e4)),
        Parameter("L", "henry", _L, bounds=(1e-4, 10.0)),
        Parameter("C", "farad", _C, bounds=(1e-6, 1.0)),
    )
    truth = (
        DynamicalEquation("I", "(u - R*I - Vc) / L"),
        DynamicalEquation("Vc", "I / C"),
    )
    train = {
        "t_span": (0.0, 1.0),
        "u_of_t": make_signal({"kind": "step", "amplitude": 5.0, "t0": 0.05}),
        "x0": (0.0, 0.0),
    }
    holdout = {
        "t_span": (0.0, 1.0),
        "u_of_t": make_signal({"kind": "step", "amplitude": 10.0, "t0": 0.05}),
        "x0": (0.0, 0.0),
        "tolerance": "auto",
    }
    return DynProblem(PROBLEM_ID, "rlc-circuit", states, inputs, parameters, truth,
                      train, holdout)


def load() -> DynProblem:
    return finalize(spec())
