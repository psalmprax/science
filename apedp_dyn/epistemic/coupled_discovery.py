r"""Coupled Multi-Variable Non-Linear Dynamical System Discovery.

Bridges the native Rust coupled Jet system engine (rime/coupled_system.rs) into APEDP-X:
- Discovers coupled ODEs \dot{\mathbf{x}} = \mathbf{F}(\mathbf{x}) across D >= 2 state variables.
- Recovers multi-variable nonlinear dynamics (e.g. Lotka-Volterra, FitzHugh-Nagumo, Lorenz).
- Enforces simultaneous system-level R^2 and worst-variable gating.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass

import apedp_kernel as ak


@dataclass(frozen=True)
class CoupledEquationInfo:
    state_variable: str
    formula: str
    active_terms: list[str]
    coefficients: list[float]
    r2: float
    residual_rmse: float


@dataclass(frozen=True)
class CoupledSystemDiscoveryResult:
    equations: list[CoupledEquationInfo]
    mean_system_r2: float
    worst_state_r2: float
    is_exact_recovery: bool


def discover_coupled_system(
    series_dict: dict[str, Sequence[float]] | Sequence[Sequence[float]],
    dt: float,
    *,
    labels: Sequence[str] | None = None,
    max_polynomial_degree: int = 2,
    include_cross_interactions: bool = True,
    sparsity_threshold: float = 0.02,
    ridge_alpha: float = 1e-6,
) -> CoupledSystemDiscoveryResult:
    """Discover coupled ODE system from multiple state variable time series."""
    if isinstance(series_dict, dict):
        state_labels = list(series_dict.keys())
        series_data = [list(series_dict[k]) for k in state_labels]
    else:
        series_data = [list(s) for s in series_dict]
        state_labels = list(labels) if labels else [f"x{i}" for i in range(len(series_data))]

    cfg = {
        "max_polynomial_degree": max_polynomial_degree,
        "include_cross_interactions": include_cross_interactions,
        "sparsity_threshold": sparsity_threshold,
        "ridge_alpha": ridge_alpha,
    }

    raw = json.loads(
        ak.rime_coupled_discover_json(
            json.dumps(series_data),
            float(dt),
            state_labels,
            json.dumps(cfg),
        )
    )

    eqs = [
        CoupledEquationInfo(
            state_variable=eq["state_variable"],
            formula=eq["formula"],
            active_terms=eq["active_terms"],
            coefficients=eq["coefficients"],
            r2=eq["r2"],
            residual_rmse=eq["residual_rmse"],
        )
        for eq in raw["equations"]
    ]

    return CoupledSystemDiscoveryResult(
        equations=eqs,
        mean_system_r2=raw["mean_system_r2"],
        worst_state_r2=raw["worst_state_r2"],
        is_exact_recovery=raw["is_exact_recovery"],
    )
