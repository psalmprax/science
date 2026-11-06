"""PS-TMDD: Two-Compartment Target-Mediated Drug Disposition (TMDD) Pharmacokinetics & Receptor Occupancy.

Status: experimental

Models coupled nonlinear receptor binding and peripheral tissue distribution:
  dCp/dt = -k_el * Cp - k_on * Cp * R + k_off * CR - k12 * Cp + k21 * Ct
  dCt/dt =  k12 * Cp - k21 * Ct
  dR/dt  = -k_on * Cp * R + k_off * CR
  dCR/dt =  k_on * Cp * R - k_off * CR

Exact receptor conservation invariant:
  d(R + CR)/dt = 0  ==>  R(t) + CR(t) = R_total = const.
"""
from __future__ import annotations

import numpy as np

from apedp_dyn.model import DynamicalEquation, Parameter, StateVariable
from apedp_dyn.problems.base import DynProblem, finalize, make_signal

PROBLEM_ID = "PS-TMDD"

_K_EL = 0.08   # 1/hour (systemic elimination rate)
_K_ON = 0.45   # liter/(nmol*hour) (second-order target association rate)
_K_OFF = 0.12  # 1/hour (first-order complex dissociation rate)
_K12 = 0.15    # 1/hour (central to peripheral distribution)
_K21 = 0.10    # 1/hour (peripheral to central redistribution)


def spec() -> DynProblem:
    """Return the preregistered 2-compartment TMDD PK/PD dynamical problem specification."""
    states = (
        StateVariable("Cp", "nmol/liter", 2.0),
        StateVariable("Ct", "nmol/liter", 0.0),
        StateVariable("R", "nmol/liter", 1.0),
        StateVariable("CR", "nmol/liter", 0.0),
    )
    inputs = ()
    parameters = (
        Parameter("k_el", "1/hour", _K_EL, bounds=(0.001, 5.0)),
        Parameter("k_on", "liter/(nmol*hour)", _K_ON, bounds=(0.001, 10.0)),
        Parameter("k_off", "1/hour", _K_OFF, bounds=(0.001, 5.0)),
        Parameter("k12", "1/hour", _K12, bounds=(0.001, 5.0)),
        Parameter("k21", "1/hour", _K21, bounds=(0.001, 5.0)),
    )
    truth = (
        DynamicalEquation("Cp", "-k_el * Cp - k_on * Cp * R + k_off * CR - k12 * Cp + k21 * Ct"),
        DynamicalEquation("Ct", "k12 * Cp - k21 * Ct"),
        DynamicalEquation("R", "-k_on * Cp * R + k_off * CR"),
        DynamicalEquation("CR", "k_on * Cp * R - k_off * CR"),
    )
    train = {
        "t_span": (0.0, 24.0),
        "u_of_t": make_signal({"kind": "zero", "n": 0}),
        "x0": (2.0, 0.0, 1.0, 0.0),
    }
    holdout = {
        "t_span": (0.0, 24.0),
        "u_of_t": make_signal({"kind": "zero", "n": 0}),
        "x0": (1.2, 0.3, 0.8, 0.2),
        "tolerance": "auto",
    }
    return DynProblem(
        PROBLEM_ID,
        "tmdd-pkpd-2comp",
        states,
        inputs,
        parameters,
        truth,
        train,
        holdout,
    )


def load() -> DynProblem:
    """Finalize and load the 2-compartment TMDD PK/PD problem."""
    return finalize(spec())


def simulate_tmdd_trajectory(
    dose_nmol_l: float,
    k_el: float = _K_EL,
    k_on: float = _K_ON,
    k_off: float = _K_OFF,
    k12: float = _K12,
    k21: float = _K21,
    r_total: float = 1.0,
    t_hours: float = 24.0,
    n_steps: int = 240,
) -> dict[str, object]:
    """Integrate the 2-compartment TMDD ODEs via 4th-order Runge-Kutta and compute receptor occupancy."""
    if dose_nmol_l < 0.0:
        raise ValueError("dose_nmol_l must be non-negative")
    dt = t_hours / max(1, n_steps)
    t_grid = np.linspace(0.0, t_hours, n_steps + 1)
    y = np.zeros((n_steps + 1, 4), dtype=np.float64)
    y[0] = np.array([dose_nmol_l, 0.0, r_total, 0.0], dtype=np.float64)

    def rhs(state: np.ndarray) -> np.ndarray:
        cp, ct, r, cr = state
        binding = k_on * cp * r - k_off * cr
        dcp = -k_el * cp - binding - k12 * cp + k21 * ct
        dct = k12 * cp - k21 * ct
        dr = -binding
        dcr = binding
        return np.array([dcp, dct, dr, dcr], dtype=np.float64)

    for idx in range(n_steps):
        cur = y[idx]
        s1 = rhs(cur)
        s2 = rhs(cur + 0.5 * dt * s1)
        s3 = rhs(cur + 0.5 * dt * s2)
        s4 = rhs(cur + dt * s3)
        y[idx + 1] = cur + (dt / 6.0) * (s1 + 2.0 * s2 + 2.0 * s3 + s4)

    receptor_sum = y[:, 2] + y[:, 3]
    max_conservation_drift = float(np.max(np.abs(receptor_sum - r_total)))
    occupancy = y[:, 3] / max(1e-12, r_total)
    mean_occupancy = float(np.mean(occupancy))
    auc_cp = float(np.trapezoid(y[:, 0], t_grid))

    return {
        "time_hours": t_grid,
        "states": y,
        "receptor_occupancy": occupancy,
        "mean_receptor_occupancy": round(mean_occupancy, 4),
        "plasma_auc_nmol_h_l": round(auc_cp, 4),
        "max_receptor_conservation_drift": max_conservation_drift,
    }
