"""APEDP-Continuum: Autonomous High-Dimensional Continuous PDE & Coupled ODE Discovery Platform."""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure root package can be imported if running within monorepo
_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from apedp_dyn.pde import (
    EpistemicPdeAdjudication,
    EpistemicPdeVerdict,
    HighDimPdeDiscoveryResult,
    PdeGridData,
    PdePointCloudData,
    PdeProposer,
    PdeReferee,
    adjudicate_high_dim_pde_discovery,
    discover_pde,
    discover_pde_1d,
    discover_pde_2d,
    discover_pde_allen_cahn_hd,
    discover_pde_black_scholes_barenblatt_100d,
    discover_pde_hjb_100d,
    discover_pde_nd_cartesian,
    discover_pde_neural_pinn_hd,
    discover_pde_nonuniform,
    discover_pde_point_cloud_hd,
    discover_pde_radial_hd,
    discover_pde_rbf_point_cloud_hd,
    discover_pde_separable_hd,
    discover_pde_tensor_train_hd,
    evaluate_point_cloud_pde_frozen_oos,
    tikhonov_smooth_trajectory,
)
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
from apedp_dyn import pde, ode

__version__ = "0.1.0"

__all__ = [
    "CoupledEquationInfo",
    "CoupledOdeProposer",
    "CoupledSystemDiscoveryResult",
    "EpistemicOdeAdjudication",
    "EpistemicOdeVerdict",
    "EpistemicPdeAdjudication",
    "EpistemicPdeVerdict",
    "HighDimPdeDiscoveryResult",
    "OdeReferee",
    "OdeTrajectoryData",
    "PdeGridData",
    "PdePointCloudData",
    "PdeProposer",
    "PdeReferee",
    "adjudicate_high_dim_pde_discovery",
    "discover_coupled_ode",
    "discover_coupled_system",
    "discover_ode",
    "discover_pde",
    "discover_pde_1d",
    "discover_pde_2d",
    "discover_pde_allen_cahn_hd",
    "discover_pde_black_scholes_barenblatt_100d",
    "discover_pde_hjb_100d",
    "discover_pde_nd_cartesian",
    "discover_pde_neural_pinn_hd",
    "discover_pde_nonuniform",
    "discover_pde_point_cloud_hd",
    "discover_pde_radial_hd",
    "discover_pde_rbf_point_cloud_hd",
    "discover_pde_separable_hd",
    "discover_pde_tensor_train_hd",
    "evaluate_point_cloud_pde_frozen_oos",
    "ode",
    "pde",
    "tikhonov_smooth_trajectory",
]
