"""PS20 problem PS-RXNK: first-order reaction kinetics A -> B.

dA/dt = -k1 * A + k2 * B    (forward + reverse)
dB/dt =  k1 * A - k2 * B

Conservation: A + B = const (mass conservation).
Equilibrium: A_eq/B_eq = k2/k1.

Train: A0=1.0, B0=0.0 (pure A, approach equilibrium).
Holdout: A0=0.3, B0=0.7 (reverse direction, approach from B side).

This tests both coupled dynamics discovery AND conservation-law checking:
any valid candidate must satisfy d(A+B)/dt = 0.
"""
from apedp_dyn.model import DynamicalEquation, Parameter, StateVariable
from apedp_dyn.problems.base import DynProblem, finalize, make_signal

PROBLEM_ID = "PS-RXNK"

_K1 = 0.5   # 1/second  (forward rate)
_K2 = 0.2   # 1/second  (reverse rate)


def spec() -> DynProblem:
    states = (
        StateVariable("A", "mol/liter", 1.0),
        StateVariable("B", "mol/liter", 0.0),
    )
    inputs = ()  # autonomous
    parameters = (
        Parameter("k1", "1/second", _K1, bounds=(0.001, 10.0)),
        Parameter("k2", "1/second", _K2, bounds=(0.001, 10.0)),
    )
    truth = (
        DynamicalEquation("A", "-k1 * A + k2 * B"),
        DynamicalEquation("B", "k1 * A - k2 * B"),
    )
    train = {
        "t_span": (0.0, 15.0),
        "u_of_t": make_signal({"kind": "zero", "n": 0}),
        "x0": (1.0, 0.0),
    }
    holdout = {
        "t_span": (0.0, 15.0),
        "u_of_t": make_signal({"kind": "zero", "n": 0}),
        "x0": (0.3, 0.7),
        "tolerance": "auto",
    }
    return DynProblem(PROBLEM_ID, "reaction-kinetics", states, inputs, parameters,
                      truth, train, holdout)


def load() -> DynProblem:
    return finalize(spec())
