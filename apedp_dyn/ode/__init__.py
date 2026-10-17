"""Continuous ODE & Coupled Dynamical System Discovery Package.

Provides autonomous non-linear coupled ODE discovery (Lotka-Volterra, predator-prey, SIR),
1st-order rational ODEs via Differential Padé, and epistemic stability referee adjudication.
"""
from __future__ import annotations

from apedp_dyn.ode.data import OdeTrajectoryData
from apedp_dyn.ode.dispatch import (
    CoupledEquationInfo,
    CoupledSystemDiscoveryResult,
    discover_coupled_ode,
    discover_coupled_system,
    discover_ode,
)
from apedp_dyn.ode.proposer import CoupledOdeProposer
from apedp_dyn.ode.referee import (
    EpistemicOdeAdjudication,
    EpistemicOdeVerdict,
    OdeReferee,
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
