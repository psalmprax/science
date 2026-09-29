"""PS20 problem PS-DRAG: quadratic drag (nonlinear dissipation).

m dv/dt = F(t) - b * v * |v|

where drag force is proportional to v|v| (aerodynamic drag).
First-order in v; position x is a secondary state: dx/dt = v.

Train: constant force (terminal velocity approach).
Holdout: sine forcing (oscillatory regime, very different dynamics).
"""
from apedp_dyn.model import DynamicalEquation, InputVariable, Parameter, StateVariable
from apedp_dyn.problems.base import DynProblem, finalize, make_signal

PROBLEM_ID = "PS-DRAG"

_M = 1.0    # kilogram
_B = 0.5    # kilogram/meter  (drag coefficient)


def spec() -> DynProblem:
    states = (
        StateVariable("x", "meter", 0.0),
        StateVariable("v", "meter/second", 0.0),
    )
    inputs = (InputVariable("F", "kilogram*meter/second**2"),)
    parameters = (
        Parameter("m", "kilogram", _M, bounds=(0.01, 100.0)),
        Parameter("b", "kilogram/meter", _B, bounds=(0.001, 100.0)),
    )
    truth = (
        DynamicalEquation("x", "v"),
        DynamicalEquation("v", "(F - b * v * Abs(v)) / m"),
    )
    train = {
        "t_span": (0.0, 10.0),
        "u_of_t": make_signal({"kind": "step", "amplitude": 5.0, "t0": 0.5}),
        "x0": (0.0, 0.0),
    }
    holdout = {
        "t_span": (0.0, 10.0),
        "u_of_t": make_signal({"kind": "sine", "amplitude": 3.0, "frequency": 0.5}),
        "x0": (0.0, 1.0),
        "tolerance": "auto",
    }
    return DynProblem(PROBLEM_ID, "quadratic-drag", states, inputs, parameters, truth,
                      train, holdout)


def load() -> DynProblem:
    return finalize(spec())
