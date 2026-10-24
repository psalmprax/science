"""PS20 problem PS-LOG: logistic population growth (nonlinear first-order ODE).

dx/dt = r * x * (1 - x/K)

where:
    x = population
    r = intrinsic growth rate [1/time]
    K = carrying capacity [count]

Autonomous system (no external input). Classic nonlinear dynamics benchmark.
Train: initial condition x0 = 10 with K = 100.
Holdout: different initial condition x0 = 1 (growth from near zero).
"""
from apedp_dyn.model import DynamicalEquation, Parameter, StateVariable
from apedp_dyn.problems.base import DynProblem, finalize, make_signal

PROBLEM_ID = "PS-LOG"

_R = 0.5    # 1/second
_K = 100.0  # counts


def spec() -> DynProblem:
    states = (StateVariable("x", "count", 10.0),)
    inputs = ()  # autonomous
    parameters = (
        Parameter("r", "1/second", _R, bounds=(0.01, 5.0)),
        Parameter("K", "count", _K, bounds=(1.0, 1e4)),
    )
    truth = (DynamicalEquation("x", "r * x * (1 - x/K)"),)
    train = {
        "t_span": (0.0, 20.0),
        "u_of_t": make_signal({"kind": "zero", "n": 0}),
        "x0": (10.0,),
    }
    holdout = {
        "t_span": (0.0, 20.0),
        "u_of_t": make_signal({"kind": "zero", "n": 0}),
        "x0": (1.0,),
        "tolerance": "auto",
    }
    return DynProblem(PROBLEM_ID, "logistic-growth", states, inputs, parameters, truth,
                      train, holdout)


def load() -> DynProblem:
    return finalize(spec())
