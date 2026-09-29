"""PS20 problem PS-RC: first-order RC circuit (electrical).

C dV/dt = I(t) - V/R,  i.e.  dV/dt = (u - V/R) / C  =  (u - V*alpha) * beta,
where alpha = 1/R [1/ohm], beta = 1/C [1/farad].

Train: step input at t=1 (charge transient).
Holdout intervention: sine input — different waveform shape.
Conservation: charge balance ∫I dt = C ΔV + ∫(V/R) dt.
"""
from apedp_dyn.model import DynamicalEquation, InputVariable, Parameter, StateVariable
from apedp_dyn.problems.base import DynProblem, finalize, make_signal

PROBLEM_ID = "PS-RC"

_R = 100.0    # ohm
_C = 0.001    # farad  (1 mF)


def spec() -> DynProblem:
    states = (StateVariable("V", "volt", 0.0),)
    inputs = (InputVariable("I", "ampere"),)
    parameters = (
        Parameter("R", "ohm", _R, bounds=(1.0, 1e4)),
        Parameter("C", "farad", _C, bounds=(1e-6, 1.0)),
    )
    truth = (DynamicalEquation("V", "(I - V/R) / C"),)
    train = {
        "t_span": (0.0, 2.0),
        "u_of_t": make_signal({"kind": "step", "amplitude": 0.01, "t0": 0.1}),
        "x0": (0.0,),
    }
    holdout = {
        "t_span": (0.0, 2.0),
        "u_of_t": make_signal({"kind": "sine", "amplitude": 0.01, "frequency": 10.0}),
        "x0": (0.0,),
        "tolerance": "auto",
    }
    return DynProblem(PROBLEM_ID, "rc-circuit", states, inputs, parameters, truth,
                      train, holdout)


def load() -> DynProblem:
    return finalize(spec())
