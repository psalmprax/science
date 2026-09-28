"""D1 problem D1-PEND: nonlinear pendulum with torque input.

theta'' + (g/L) sin(theta) + b theta' = u(t), first-order (theta, omega).
Train: mild drive (A=0.5, w=1.0) — persistent excitation is required to
identify the input gain (with u identically zero the u-term is structurally
unidentifiable from the train record). Holdout intervention: stronger, faster
drive A=1.5, w=2.0 — unreachable from the train record without the true
nonlinear law (linearized models diverge under drive).
"""
from apedp_dyn.problems.base import DynProblem, finalize, make_signal
from apedp_dyn.model import DynamicalEquation, InputVariable, Parameter, StateVariable

PROBLEM_ID = "D1-PEND"

_G = 9.81  # meter/second**2
_L = 1.0  # meter
_B = 0.1  # 1/second


def spec() -> DynProblem:
    states = (StateVariable("theta", "radian", 0.6),
              StateVariable("omega", "radian/second", 0.0))
    inputs = (InputVariable("u", "radian/second**2"),)
    parameters = (Parameter("g", "meter/second**2", _G),
                  Parameter("L", "meter", _L),
                  Parameter("b", "1/second", _B))
    truth = (DynamicalEquation("theta", "omega"),
             DynamicalEquation("omega", "-(g/L)*sin(theta) - b*omega + u"))
    train = {"t_span": (0.0, 10.0),
             "u_of_t": make_signal({"kind": "sine", "amplitude": 0.5, "frequency": 1.0}),
             "x0": (0.6, 0.0)}
    holdout = {"t_span": (0.0, 10.0),
               "u_of_t": make_signal({"kind": "sine", "amplitude": 1.5, "frequency": 2.0}),
               "x0": (0.6, 0.0),
               "tolerance": 5e-2}
    return DynProblem(PROBLEM_ID, "nonlinear-pendulum", states, inputs,
                      parameters, truth, train, holdout)


def load() -> DynProblem:
    return finalize(spec())
