"""D2b problem D2-DUFF: Duffing oscillator (nonlinear restoring force).

xddot = -alpha*x - beta*x^3 - delta*v + gamma*u.
Train: small drive (A=0.3) — cubic term negligible, linear rival fits.
Holdout intervention: large drive (A=2.5) — cubic engaged; only the true
nonlinear law transfers. Mirrors the pendulum amplitude story (D1.6).
"""
from apedp_dyn.problems.base import DynProblem, finalize, make_signal
from apedp_dyn.model import DynamicalEquation, InputVariable, Parameter, StateVariable

PROBLEM_ID = "D2-DUFF"

_ALPHA = 4.0  # 1/second**2  (linear stiffness / m)
_BNL = 2.0  # 1/(second**2*meter**2)  (cubic stiffness / m; "bnl" avoids sympy beta)
_DELTA = 0.2  # 1/second  (damping / m)
_GMU = 1.0  # 1/kilogram  (input gain; "gmu" avoids sympy gamma)


def spec() -> DynProblem:
    states = (StateVariable("x", "meter", 0.15),
              StateVariable("v", "meter/second", 0.0))
    inputs = (InputVariable("u", "kilogram*meter/second**2"),)
    parameters = (Parameter("alpha", "1/second**2", _ALPHA),
                  Parameter("bnl", "1/(second**2*meter**2)", _BNL),
                  Parameter("delta", "1/second", _DELTA),
                  Parameter("gmu", "1/kilogram", _GMU))
    truth = (DynamicalEquation("x", "v"),
             DynamicalEquation("v", "-alpha*x - bnl*x**3 - delta*v + gmu*u"))
    train = {"t_span": (0.0, 10.0),
             "u_of_t": make_signal({"kind": "sine", "amplitude": 0.3, "frequency": 1.0}),
             "x0": (0.15, 0.0)}
    holdout = {"t_span": (0.0, 10.0),
               "u_of_t": make_signal({"kind": "sine", "amplitude": 2.5, "frequency": 1.0}),
               "x0": (0.15, 0.0),
               "tolerance": 5e-2}
    return DynProblem(PROBLEM_ID, "duffing", states, inputs,
                      parameters, truth, train, holdout)


def load() -> DynProblem:
    return finalize(spec())
