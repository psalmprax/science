"""PS20 problem PS-LV: Lotka-Volterra predator-prey (coupled nonlinear dynamics).

dx/dt =  a1 * x - b1 * x * y
dy/dt = -g1 * y + d1 * x * y

where:
    x  = prey population
    y  = predator population
    a1 = prey birth rate [1/time]
    b1 = predation rate [1/(time*count)]
    g1 = predator death rate [1/time]
    d1 = predator growth from predation [1/(time*count)]

Autonomous system. Exhibits oscillatory dynamics — neither species goes to
zero (in the undamped case). Classic coupled-nonlinear benchmark.

Train: moderate initial condition.
Holdout: different initial condition (shifted ecological balance).
Conservation: the system has a Hamiltonian invariant (V-integral).

Note: parameter names avoid SymPy builtins (alpha→a1, beta→b1, gamma→g1, delta→d1).
"""
from apedp_dyn.model import DynamicalEquation, Parameter, StateVariable
from apedp_dyn.problems.base import DynProblem, finalize, make_signal

PROBLEM_ID = "PS-LV"

_A1 = 1.1   # 1/second  (prey birth)
_B1 = 0.4   # 1/(second*count)  (predation)
_G1 = 0.4   # 1/second  (predator death)
_D1 = 0.1   # 1/(second*count)  (predator growth)


def spec() -> DynProblem:
    states = (
        StateVariable("x", "count", 10.0),
        StateVariable("y", "count", 5.0),
    )
    inputs = ()  # autonomous
    parameters = (
        Parameter("a1", "1/second", _A1, bounds=(0.01, 5.0)),
        Parameter("b1", "1/(second*count)", _B1, bounds=(0.001, 5.0)),
        Parameter("g1", "1/second", _G1, bounds=(0.01, 5.0)),
        Parameter("d1", "1/(second*count)", _D1, bounds=(0.001, 5.0)),
    )
    truth = (
        DynamicalEquation("x", "a1 * x - b1 * x * y"),
        DynamicalEquation("y", "-g1 * y + d1 * x * y"),
    )
    train = {
        "t_span": (0.0, 30.0),
        "u_of_t": make_signal({"kind": "zero", "n": 0}),
        "x0": (10.0, 5.0),
    }
    holdout = {
        "t_span": (0.0, 30.0),
        "u_of_t": make_signal({"kind": "zero", "n": 0}),
        "x0": (20.0, 3.0),
        "tolerance": "auto",
    }
    return DynProblem(PROBLEM_ID, "lotka-volterra", states, inputs, parameters, truth,
                      train, holdout)


def load() -> DynProblem:
    return finalize(spec())
