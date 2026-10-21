"""Unified dispatch and functional API for continuous ODE discovery."""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

try:  # pragma: no cover - exercised only on kernel-less installs
    import apedp_kernel as ak
except ImportError:  # kernel-only engine: fails at call time with an actionable error
    from apedp_dyn.kernel_shim import missing_kernel as ak  # type: ignore[assignment]
from apedp_dyn.epistemic.coupled_discovery import (
    CoupledEquationInfo as CoupledEquationInfo,  # explicit re-export surface
)
from apedp_dyn.epistemic.coupled_discovery import (
    CoupledSystemDiscoveryResult,
    discover_coupled_system,
)
from apedp_dyn.ode.data import OdeTrajectoryData


def discover_coupled_ode(
    data: OdeTrajectoryData | Mapping[str, Sequence[float]] | Sequence[Sequence[float]],
    dt: float | None = None,
    *,
    labels: Sequence[str] | None = None,
    max_polynomial_degree: int = 2,
    include_cross_interactions: bool = True,
    sparsity_threshold: float = 0.02,
    ridge_alpha: float = 1e-6,
) -> CoupledSystemDiscoveryResult:
    """Discover coupled non-linear multi-state ODE system dx/dt = F(x)."""
    series_input: dict[str, Sequence[float]]
    if isinstance(data, OdeTrajectoryData):
        step_dt = data.dt
        series_input = {str(k): list(v) for k, v in data.states.items()}
    else:
        if dt is None:
            raise ValueError("dt must be provided when passing raw sequence or dict")
        step_dt = float(dt)
        if isinstance(data, Mapping):
            series_input = {str(k): list(v) for k, v in data.items()}
        else:
            series_input = {f"series_{i:02d}": list(row) for i, row in enumerate(data)}

    return discover_coupled_system(
        series_input,
        step_dt,
        labels=labels,
        max_polynomial_degree=max_polynomial_degree,
        include_cross_interactions=include_cross_interactions,
        sparsity_threshold=sparsity_threshold,
        ridge_alpha=ridge_alpha,
    )


def discover_ode(
    x: Sequence[float],
    y: Sequence[float],
    *,
    max_degree: int = 2,
) -> dict[str, Any]:
    """Discover 1st-order rational ODE u'/u = P(x)/Q(x) via Differential Padé & Jet algebra.

    The kernel seam is ``rime_diff_pade_discover_json(x_json, y_json,
    config_json)`` — three JSON STRINGS (matching every other caller in the
    repository, e.g. tests/test_dyn_planck_limitation.py); the previous
    wrapper passed raw Python lists plus an int, which no kernel call site
    ever exercised end-to-end.
    """
    x_arr = np.ascontiguousarray(x, dtype=np.float64)
    y_arr = np.ascontiguousarray(y, dtype=np.float64)
    if len(x_arr) != len(y_arr):
        raise ValueError(f"Length of x ({len(x_arr)}) != length of y ({len(y_arr)})")

    config = {
        # Rust DiffPadeConfig defaults with max_x_degree overridden:
        "max_x_degree": int(max_degree),
        "max_diff_order": 1,
        "smooth_window": 5,
        "sparsity_threshold": 0.05,
        "rcond_threshold": 1e-12,
        "include_riccati": False,
    }
    raw_json = ak.rime_diff_pade_discover_json(
        json.dumps(x_arr.tolist()),
        json.dumps(y_arr.tolist()),
        json.dumps(config),
    )
    parsed = json.loads(raw_json)
    if not isinstance(parsed, dict):
        raise TypeError("Rust kernel returned a non-object JSON payload")
    return parsed
