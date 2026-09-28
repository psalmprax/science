"""D2b problem D2-COUPLED: 2-DOF spring-coupled oscillators (coupling topology).

m1 x1'' = -k1 x1 - kc (x1 - x2) - c1 v1 + u1
m2 x2'' = -k2 x2 - kc (x2 - x1) - c2 v2 + u2    (all masses normalized to 1).

Train drives both masses at distinct frequencies; holdout changes both
drives and perturbs the ICs. The uncoupled rival (no xr terms) can absorb
the coupling into shifted local stiffness on train, but cannot transfer —
coupling topology is what gate 4 tests.
"""
from apedp_dyn.model import DynamicalEquation, InputVariable, Parameter, StateVariable
from apedp_dyn.problems.base import DynProblem, finalize, make_signal

PROBLEM_ID = "D2-COUPLED"

_K1, _K2, _KC = 4.0, 6.0, 1.5  # 1/second**2 (normalized by unit masses)
_C1, _C2 = 0.15, 0.15  # 1/second

_TRAIN_SIGNALS = ({"kind": "sine", "amplitude": 1.0, "frequency": 1.0},
                  {"kind": "sine", "amplitude": 0.8, "frequency": 0.6})
_HOLDOUT_SIGNALS = ({"kind": "sine", "amplitude": 1.5, "frequency": 2.2},
                    {"kind": "sine", "amplitude": 1.2, "frequency": 1.3})


def spec() -> DynProblem:
    states = (StateVariable("x1", "meter", 0.4),
              StateVariable("v1", "meter/second", 0.0),
              StateVariable("x2", "meter", -0.3),
              StateVariable("v2", "meter/second", 0.0))
    inputs = (InputVariable("u1", "kilogram*meter/second**2"),
              InputVariable("u2", "kilogram*meter/second**2"))
    parameters = (Parameter("k1", "1/second**2", _K1),
                  Parameter("k2", "1/second**2", _K2),
                  Parameter("kc", "1/second**2", _KC),
                  Parameter("c1", "1/second", _C1),
                  Parameter("c2", "1/second", _C2),
                  Parameter("g1", "1/kilogram", 1.0),
                  Parameter("g2", "1/kilogram", 1.0))
    truth = (DynamicalEquation("x1", "v1"),
             DynamicalEquation("v1", "-k1*x1 - kc*(x1 - x2) - c1*v1 + g1*u1"),
             DynamicalEquation("x2", "v2"),
             DynamicalEquation("v2", "-k2*x2 - kc*(x2 - x1) - c2*v2 + g2*u2"))
    # Declarative, serializable two-input excitation: each mass is driven at a
    # distinct frequency on train, and the holdout changes both drives.
    train = {"t_span": (0.0, 10.0),
             "u_of_t": make_signal(_TRAIN_SIGNALS),
             "x0": (0.4, 0.0, -0.3, 0.0)}
    holdout = {"t_span": (0.0, 10.0),
               "u_of_t": make_signal(_HOLDOUT_SIGNALS),
               "x0": (0.7, 0.0, -0.5, 0.1),
               "tolerance": "auto"}
    return DynProblem(PROBLEM_ID, "coupled-2dof", states, inputs,
                      parameters, truth, train, holdout)


def load() -> DynProblem:
    return finalize(spec())
