"""D1 problem D1-MSD: damped harmonic oscillator with forced input.

m x'' + c x' + k x = u(t), first-order form (x, v).
Train: sine forcing A=1.0, w=1.0. Holdout intervention: stronger, faster
forcing A=2.0, w=2.5 — a memorized response cannot transfer; the law must.
"""
from apedp_dyn.model import DynamicalEquation, InputVariable, Parameter, StateVariable
from apedp_dyn.problems.base import DynProblem, finalize, make_signal

PROBLEM_ID = "D1-MSD"

_M = 1.0  # kg
_C = 0.2  # kg/s
_K = 4.0  # kg/s^2


def spec() -> DynProblem:
    states = (StateVariable("x", "meter", 0.5),
              StateVariable("v", "meter/second", 0.0))
    inputs = (InputVariable("u", "kilogram*meter/second**2"),)
    parameters = (Parameter("m", "kilogram", _M),
                  Parameter("c", "kilogram/second", _C),
                  Parameter("k", "kilogram/second**2", _K))
    truth = (DynamicalEquation("x", "v"),
             DynamicalEquation("v", "(u - c*v - k*x)/m"))
    train = {"t_span": (0.0, 10.0),
             "u_of_t": make_signal({"kind": "sine", "amplitude": 1.0, "frequency": 1.0}),
             "x0": (0.5, 0.0)}
    holdout = {"t_span": (0.0, 10.0),
               "u_of_t": make_signal({"kind": "sine", "amplitude": 2.0, "frequency": 2.5}),
               "x0": (0.5, 0.0),
               "tolerance": "auto"}
    return DynProblem(PROBLEM_ID, "mass-spring-damper", states, inputs,
                      parameters, truth, train, holdout)


def load() -> DynProblem:
    return finalize(spec())


def resolve_physical(alpha: float, beta: float, gamma: float) -> dict[str, float]:
    """Normalized fit (alpha=k/m, beta=c/m, gamma=1/m) -> physical (m, c, k).

    APEDP claims m/c/k separately only when the input gain identifies m:
    |gamma| must clear zero, i.e. the experiment must excite through u.
    Otherwise the data identify ratios, and this raises instead of guessing.
    """
    if abs(gamma) < 1e-9:
        raise ValueError(
            "input gain unidentifiable (|gamma| ~ 0): data constrain ratios "
            "k/m, c/m only; no separate m/c/k claim permitted")
    m = 1.0 / gamma
    return {"m": m, "c": beta * m, "k": alpha * m}
