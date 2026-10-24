"""APEDP-DYN benchmark problems and 1000+ equations catalog.

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

Extended 1000+ Catalog:
    Over 1,000 equations across classical, feynman, kinetics, circuits,
    astrophysics, biology, and nonlinear ODEs available via get_catalog_modules().
"""
from __future__ import annotations

from typing import Any

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
from apedp_dyn.problems.catalog import (
    build_catalog,
    count_catalog_equations,
    get_catalog_modules,
    get_problem,
)

ProblemSpec = DynProblem

# Registry of all PS20 benchmark problems (frozen core benchmark)
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


def load_all() -> dict[str, DynProblem]:
    """Load and finalize all PS20 benchmark problems."""
    return {pid: mod.load() for pid, mod in ALL_PROBLEMS.items()}


def get_all_problems_extended() -> dict[str, Any]:
    """Registry containing both PS20 benchmark problems and the 1000+ catalog."""
    return {**ALL_PROBLEMS, **get_catalog_modules()}


def __getattr__(name: str) -> Any:
    cat = get_catalog_modules()
    if name in cat:
        return cat[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ALL_PROBLEMS",
    "DynProblem",
    "ProblemSpec",
    "build_catalog",
    "count_catalog_equations",
    "coupled",
    "damped_pendulum",
    "duffing",
    "exponential_decay",
    "finalize",
    "get_all_problems_extended",
    "get_catalog_modules",
    "get_problem",
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
