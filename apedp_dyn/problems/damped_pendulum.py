"""PS20 problem PS-DAMPPEND: damped pendulum (nonlinear dissipative system).

theta'' + b*theta' + (g/L)*sin(theta) = 0

First-order form:
    dtheta/dt = omega
    domega/dt = -(g/L)*sin(theta) - b*omega

Autonomous system. Nonlinear + dissipative: no conserved energy, but
mechanical energy decays monotonically. Tests dissipation detection.

Train: theta0 = pi/3 (large initial angle).
Holdout: theta0 = pi/6, omega0 = 2.0 (different regime, initial velocity).
"""
from apedp_dyn.model import DynamicalEquation, Parameter, StateVariable
from apedp_dyn.problems.base import DynProblem, finalize, make_signal

PROBLEM_ID = "PS-DAMPPEND"

_G = 9.81  # meter/second**2
_L = 1.0   # meter
_B = 0.3   # 1/second (stronger damping than D1-PEND)


def spec() -> DynProblem:
    states = (
        StateVariable("theta", "radian", 1.047),  # pi/3
        StateVariable("omega", "radian/second", 0.0),
    )
    inputs = ()  # autonomous (no torque input)
    parameters = (
        Parameter("g", "meter/second**2", _G, bounds=(1.0, 20.0)),
        Parameter("L", "meter", _L, bounds=(0.1, 10.0)),
        Parameter("b", "1/second", _B, bounds=(0.01, 5.0)),
    )
    truth = (
        DynamicalEquation("theta", "omega"),
        DynamicalEquation("omega", "-(g/L)*sin(theta) - b*omega"),
    )
    train = {
        "t_span": (0.0, 15.0),
        "u_of_t": make_signal({"kind": "zero", "n": 0}),
        "x0": (1.047, 0.0),
    }
    holdout = {
        "t_span": (0.0, 15.0),
        "u_of_t": make_signal({"kind": "zero", "n": 0}),
        "x0": (0.524, 2.0),
        "tolerance": "auto",
    }
    return DynProblem(PROBLEM_ID, "damped-pendulum", states, inputs, parameters,
                      truth, train, holdout)


def load() -> DynProblem:
    return finalize(spec())
