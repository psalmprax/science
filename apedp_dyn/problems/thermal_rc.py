"""PS20 problem PS-THERM: first-order thermal RC (Newton's law of cooling).

C dT/dt = (Tenv - T) / R + Qh(t)

where:
    T    = object temperature [kelvin]
    Tenv = ambient temperature [kelvin] (constant parameter)
    R    = thermal resistance [kelvin*second/joule]
    C    = thermal capacitance [joule/kelvin]
    Qh   = external heat input [watt = joule/second]

Train: step heat input. Holdout: sine heat input at different frequency.

Note: parameter names avoid SymPy builtins (Q → Qh, T_env → Tenv).
"""
from apedp_dyn.model import DynamicalEquation, InputVariable, Parameter, StateVariable
from apedp_dyn.problems.base import DynProblem, finalize, make_signal

PROBLEM_ID = "PS-THERM"

_R = 5.0      # kelvin*second/watt  (thermal resistance)
_C = 2.0      # joule/kelvin        (thermal capacitance)
_TENV = 293.0  # kelvin  (20°C ambient)


def spec() -> DynProblem:
    states = (StateVariable("T", "kelvin", 293.0),)
    inputs = (InputVariable("Qh", "watt"),)
    parameters = (
        # Thermal resistance is kelvin/watt. The extra "*second" made R*C carry
        # time^2, so (Tenv-T)/(R*C) evaluated to kelvin/second^2 and gate 1
        # rejected the truth model as dimensionally inhomogeneous.
        Parameter("R", "kelvin/watt", _R, bounds=(0.1, 100.0)),
        Parameter("C", "joule/kelvin", _C, bounds=(0.01, 100.0)),
        Parameter("Tenv", "kelvin", _TENV, bounds=(200.0, 400.0)),
    )
    truth = (DynamicalEquation("T", "(Tenv - T) / (R * C) + Qh / C"),)
    train = {
        "t_span": (0.0, 60.0),
        "u_of_t": make_signal({"kind": "step", "amplitude": 10.0, "t0": 5.0}),
        "x0": (293.0,),
    }
    holdout = {
        "t_span": (0.0, 60.0),
        "u_of_t": make_signal({"kind": "sine", "amplitude": 5.0, "frequency": 0.1}),
        "x0": (290.0,),
        "tolerance": "auto",
    }
    return DynProblem(PROBLEM_ID, "thermal-rc", states, inputs, parameters, truth,
                      train, holdout)


def load() -> DynProblem:
    return finalize(spec())
