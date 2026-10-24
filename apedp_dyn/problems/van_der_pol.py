"""PS20 problem PS-VDPOL: Van der Pol oscillator (self-sustained oscillations).

Second-order form, written with the two scales made explicit so it is
dimensionally well posed:

    A * x'' + mu*(1 - (x/A)**2)*x' + w**2*x = 0

First-order form:
    dx/dt = v
    dv/dt = mu*(1 - (x/A)**2)*v - w**2*x

with A the amplitude at which the nonlinear damping changes sign [m] and
w the linear natural frequency [1/s]. Both are declared parameters rather
than implicit, because the textbook form mu*(1 - x**2)*v - x only makes
sense in normalised coordinates: written against physical units it is not
dimensionally homogeneous. `1 - x**2` adds a dimensionless 1 to a length
squared, and the bare `-x` has units of length where dv/dt needs an
acceleration. Naming the scales is what makes the dimensional gate
meaningful here rather than a formality.

Nonlinear damping: energy is injected when |x| < A and dissipated when
|x| > A, producing a limit cycle. Tests discovery of state-dependent
damping and of two competing scales.

Train: x0 = 0.5, v0 = 0 (start inside limit cycle).
Holdout: x0 = 4.0, v0 = 0 (start outside limit cycle — must converge to same cycle).
"""
from apedp_dyn.model import DynamicalEquation, Parameter, StateVariable
from apedp_dyn.problems.base import DynProblem, finalize, make_signal

PROBLEM_ID = "PS-VDPOL"

_MU = 1.0  # 1/second, nonlinear damping strength
_A = 1.0   # meter, amplitude at which damping changes sign
_W = 1.0   # 1/second, linear natural frequency


def spec() -> DynProblem:
    states = (
        StateVariable("x", "meter", 0.5),
        StateVariable("v", "meter/second", 0.0),
    )
    inputs = ()  # autonomous
    parameters = (
        Parameter("mu", "1/second", _MU, bounds=(0.01, 10.0)),
        Parameter("A", "meter", _A, bounds=(0.1, 10.0)),
        Parameter("w", "1/second", _W, bounds=(0.1, 10.0)),
    )
    truth = (
        DynamicalEquation("x", "v"),
        DynamicalEquation("v", "mu*(1 - (x/A)**2)*v - w**2*x"),
    )
    train = {
        "t_span": (0.0, 20.0),
        "u_of_t": make_signal({"kind": "zero", "n": 0}),
        "x0": (0.5, 0.0),
    }
    holdout = {
        "t_span": (0.0, 20.0),
        "u_of_t": make_signal({"kind": "zero", "n": 0}),
        "x0": (4.0, 0.0),
        "tolerance": "auto",
    }
    return DynProblem(PROBLEM_ID, "van-der-pol", states, inputs, parameters, truth,
                      train, holdout)


def load() -> DynProblem:
    return finalize(spec())
