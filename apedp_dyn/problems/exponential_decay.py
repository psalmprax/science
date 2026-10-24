"""PS20 problem PS-DECAY: exponential decay (simplest first-order ODE).

dx/dt = -k * x

The simplest possible dynamical system. Baseline sanity check.
Train: x0 = 10.  Holdout: x0 = 50 (5x larger initial condition).
"""
from apedp_dyn.model import DynamicalEquation, Parameter, StateVariable
from apedp_dyn.problems.base import DynProblem, finalize, make_signal

PROBLEM_ID = "PS-DECAY"

_K = 0.3  # 1/second


def spec() -> DynProblem:
    states = (StateVariable("x", "meter", 10.0),)
    inputs = ()  # autonomous
    parameters = (
        Parameter("k", "1/second", _K, bounds=(0.001, 10.0)),
    )
    truth = (DynamicalEquation("x", "-k * x"),)
    train = {
        "t_span": (0.0, 10.0),
        "u_of_t": make_signal({"kind": "zero", "n": 0}),
        "x0": (10.0,),
    }
    holdout = {
        "t_span": (0.0, 10.0),
        "u_of_t": make_signal({"kind": "zero", "n": 0}),
        "x0": (50.0,),
        "tolerance": "auto",
    }
    return DynProblem(PROBLEM_ID, "exponential-decay", states, inputs, parameters,
                      truth, train, holdout)


def load() -> DynProblem:
    return finalize(spec())
