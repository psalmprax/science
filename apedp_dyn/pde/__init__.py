"""Continuous Spatio-Temporal PDE Discovery & Epistemic Referee Package.

Provides high-dimensional PDE discovery from 1D to 100D+, hybrid Eulerian/Lagrangian
dispatch, mesh-free RBF point clouds, neural PINNs, Tensor-Train MPS cores,
and strict epistemic referee adjudication.
"""
from __future__ import annotations

from apedp_dyn.pde.data import PdeGridData, PdePointCloudData
from apedp_dyn.pde.dispatch import (
    EpistemicPdeAdjudication,
    EpistemicPdeVerdict,
    HighDimPdeDiscoveryResult,
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
from apedp_dyn.pde.proposer import PdeProposer
from apedp_dyn.pde.referee import PdeReferee

__all__ = [
    "EpistemicPdeAdjudication",
    "EpistemicPdeVerdict",
    "HighDimPdeDiscoveryResult",
    "PdeGridData",
    "PdePointCloudData",
    "PdeProposer",
    "PdeReferee",
    "adjudicate_high_dim_pde_discovery",
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
    "tikhonov_smooth_trajectory",
]
