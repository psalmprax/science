"""Post-fit validators (D5-stage physics checks, run after gates 1-4).

The conserved quantity is derived from the model's own equations
(:func:`derive_energy_law`), so the model-driven checks cover any second-order
mechanical form — including cubic-restoring and multi-DOF coupled systems. The
explicit ``(x, v, m, k)`` helpers remain for the analytic 1-DOF case.
"""
from apedp_dyn.validation.energy import (
    EnergyLaw,
    check_conservative,
    check_conservative_core,
    check_damped_decay,
    check_dissipative,
    derive_energy_law,
    energy_drift,
    energy_drift_from_law,
    mechanical_energy,
    simulate_pair,
    simulate_trajectory,
    total_energy,
)

__all__ = [
    "EnergyLaw",
    "check_conservative",
    "check_conservative_core",
    "check_damped_decay",
    "check_dissipative",
    "derive_energy_law",
    "energy_drift",
    "energy_drift_from_law",
    "mechanical_energy",
    "simulate_pair",
    "simulate_trajectory",
    "total_energy",
]
