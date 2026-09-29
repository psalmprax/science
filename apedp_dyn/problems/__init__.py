"""APEDP-DYN benchmark problems.

PS20 Physical Systems Benchmark: 20 controlled dynamical systems where
the ground truth is known. Divided into four groups:

Group A — Classical mechanics (5 cases)
    mass_spring_damper, pendulum, duffing, damped_pendulum, van_der_pol

Group B — Electrical/thermal (5 cases)
    rc_circuit, rlc_circuit, thermal_rc, coupled (2-DOF), quadratic_drag

Group C — Population/reaction (3 cases)
    logistic, lotka_volterra, reaction_kinetics

Group D — Adversarial physical identification (2 cases)
    two_body_gravity, exponential_decay
"""
from apedp_dyn.problems import (
    coupled,
    damped_pendulum,
    duffing,
    exponential_decay,
    logistic,
    lotka_volterra,
    mass_spring_damper,
    pendulum,
    quadratic_drag,
    rc_circuit,
    reaction_kinetics,
    rlc_circuit,
    thermal_rc,
    two_body_gravity,
    van_der_pol,
)
from apedp_dyn.problems.base import DynProblem, finalize, make_signal

# Registry of all PS20 problems
ALL_PROBLEMS = {
    # Group A — Classical mechanics
    "D1-MSD": mass_spring_damper,
    "D1-PEND": pendulum,
    "D2-DUFF": duffing,
    "PS-DAMPPEND": damped_pendulum,
    "PS-VDPOL": van_der_pol,
    # Group B — Electrical/thermal/multi-DOF
    "PS-RC": rc_circuit,
    "PS-RLC": rlc_circuit,
    "PS-THERM": thermal_rc,
    "D2-COUPLED": coupled,
    "PS-DRAG": quadratic_drag,
    # Group C — Population/reaction
    "PS-LOG": logistic,
    "PS-LV": lotka_volterra,
    "PS-RXNK": reaction_kinetics,
    # Group D — Adversarial/challenging
    "PS-GRAV2": two_body_gravity,
    "PS-DECAY": exponential_decay,
}


def load_all():
    """Load and finalize all benchmark problems."""
    return {pid: mod.load() for pid, mod in ALL_PROBLEMS.items()}


__all__ = [
    "ALL_PROBLEMS",
    "DynProblem",
    "coupled",
    "damped_pendulum",
    "duffing",
    "exponential_decay",
    "finalize",
    "load_all",
    "logistic",
    "lotka_volterra",
    "make_signal",
    "mass_spring_damper",
    "pendulum",
    "quadratic_drag",
    "rc_circuit",
    "reaction_kinetics",
    "rlc_circuit",
    "thermal_rc",
    "two_body_gravity",
    "van_der_pol",
]
