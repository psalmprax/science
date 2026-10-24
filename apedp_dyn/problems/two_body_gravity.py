"""PS20 problem PS-GRAV2: two-body gravitational system (1D, radial).

d²r/dt² = -G*M/r² + L²/(m*r³)

Reduced 1D radial problem with effective potential (includes centrifugal term).
First-order form (r, v_r):
    dr/dt    = v_r
    dv_r/dt  = -GM/r**2 + h**2/r**3

where h = L/m = specific angular momentum (constant).

This is a nonlinear ODE with 1/r² and 1/r³ terms — a significant grammar
challenge. Tests whether APEDP can discover inverse-power-law dynamics.

Train: near-circular orbit (small radial oscillation).
Holdout: larger initial velocity (elliptical orbit).
"""
from apedp_dyn.model import DynamicalEquation, Parameter, StateVariable
from apedp_dyn.problems.base import DynProblem, finalize, make_signal

PROBLEM_ID = "PS-GRAV2"

_GM = 10.0  # meter**3/second**2
_H = 3.0    # meter**2/second  (specific angular momentum)


def spec() -> DynProblem:
    states = (
        StateVariable("r", "meter", 1.0),
        StateVariable("vr", "meter/second", 0.0),
    )
    inputs = ()  # autonomous
    parameters = (
        Parameter("GM", "meter**3/second**2", _GM, bounds=(0.1, 1000.0)),
        Parameter("h", "meter**2/second", _H, bounds=(0.1, 100.0)),
    )
    truth = (
        DynamicalEquation("r", "vr"),
        DynamicalEquation("vr", "-GM/r**2 + h**2/r**3"),
    )
    train = {
        "t_span": (0.0, 10.0),
        "u_of_t": make_signal({"kind": "zero", "n": 0}),
        "x0": (1.0, 0.0),
    }
    holdout = {
        "t_span": (0.0, 10.0),
        "u_of_t": make_signal({"kind": "zero", "n": 0}),
        "x0": (1.0, 1.5),
        "tolerance": "auto",
    }
    return DynProblem(PROBLEM_ID, "two-body-gravity", states, inputs, parameters,
                      truth, train, holdout)


def load() -> DynProblem:
    return finalize(spec())
