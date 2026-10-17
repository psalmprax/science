"""APEDP Continuous ODE & Coupled Dynamical System Discovery API.

Re-exports the core continuous ODE discovery engines from `apedp_dyn.ode`.
"""
from __future__ import annotations

from apedp_dyn.ode import (
    CoupledEquationInfo,
    CoupledOdeProposer,
    CoupledSystemDiscoveryResult,
    EpistemicOdeAdjudication,
    EpistemicOdeVerdict,
    OdeReferee,
    OdeTrajectoryData,
    discover_coupled_ode,
    discover_coupled_system,
    discover_ode,
)

__all__ = [
    "CoupledEquationInfo",
    "CoupledOdeProposer",
    "CoupledSystemDiscoveryResult",
    "EpistemicOdeAdjudication",
    "EpistemicOdeVerdict",
    "OdeReferee",
    "OdeTrajectoryData",
    "discover_coupled_ode",
    "discover_coupled_system",
    "discover_ode",
]
