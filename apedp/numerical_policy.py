"""Centralized Numerical Policy and Threshold Configuration for APEDP.

Eliminates scattered magic numbers (0.999, 0.99, 0.90, 0.50, 0.30, 10.0, etc.)
and provides a single, auditable source of truth for numerical thresholds,
adequacy gates, overfit boundaries, and runtime engine selections.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class DiscoveryNumericalPolicy:
    """Centralized thresholds and tolerances for physical equation discovery."""

    # Overfit & Extreme Fit Gates
    OVERFIT_R2_THRESHOLD: float = 0.999
    OVERFIT_PARAM_RATIO: float = 0.50  # active_k > n * ratio

    # Fit Confidence Thresholds
    COMPOSITION_HIGH_CONFIDENCE_R2: float = 0.99
    POWER_LAW_CONFIDENT_R2: float = 0.90
    CANDIDATE_VIABILITY_R2: float = 0.50
    MIN_SIGNIFICANT_R2: float = 0.30

    # Data Adequacy Gates
    MIN_SAMPLE_SIZE_SURROGATE: int = 30
    MIN_SAMPLE_SIZE_FORWARD_SELECT: int = 5
    MIN_SAMPLE_SIZE_POWER_LAW: int = 3
    MIN_RANGE_RATIO_IDENTIFIABILITY: float = 10.0  # max(x) / min(x) for power laws
    MIN_OBS_PER_PARAMETER: int = 5

    # Numerical Stabilities
    VARIANCE_EPSILON: float = 1e-300
    COLLINEARITY_VIF_THRESHOLD: float = 10.0
    COLLINEARITY_CORR_THRESHOLD: float = 0.98
    RATIONAL_SNAP_TOLERANCE: float = 0.02
    KASS_RAFTERY_DELTA_BIC: float = 2.0  # Positive evidence boundary
    FORWARD_SELECT_BEAM_WIDTH: int = 6  # Breadth for beam forward subset selection (avoids myopic greedy trapping)

    # Minimum Description Length (MDL) & Structural Complexity Policy
    # Derivation (code lengths in units of ln n, calibrated at n ~= 30):
    # - 1 fitted continuous exponent ~= 1 DOF ~= 0.5 ln n (BIC).
    # - Canonical monomials (x^1, 1/x): no extra DOF beyond the amplitudes
    #   already counted in k -> 0.0.
    # - Integer powers: discrete choice among K ~= 5 grid options -> ln K;
    #   at n ~= 30, ln 5 ~= 1.61 ~= 0.5 ln n -> 0.5.
    # - Cross interactions: pair choice among ~n pairs -> ~ln n -> 1.0.
    # - Fractional powers: grid choice + continuous refinement ~= 1.5.
    # - Transcendentals (exp/log): family bit + functional form, most
    #   expressive -> 2.5 (conservative upper bound).
    # Calibration procedure: tests/test_mdl_calibration.py sweeps MDL_WEIGHT
    # over {0, 0.5, 1, 2, 4} on a fixed collinear library (y = 3x + 2/x,
    # n = 30) and asserts w = 0 admits soup while w = 1 recovers the truth.
    # Tune via MDL_WEIGHT rather than editing individual costs.
    MDL_PENALTY_ENABLED: bool = True
    MDL_WEIGHT: float = 1.0
    MDL_COST_TRANSCENDENTAL: float = 2.5
    MDL_COST_FRACTIONAL_POWER: float = 1.5
    MDL_COST_CROSS_INTERACTION: float = 1.0
    MDL_COST_INTEGER_POWER: float = 0.5
    MDL_COST_CANONICAL: float = 0.0
    BOUNDARY_HOLDOUT_GUARD: bool = True
    BOUNDARY_HOLDOUT_PERCENTILE: float = 12.0
    BOUNDARY_HOLDOUT_DEGRADATION_LIMIT: float = 1.50


    # Default Neural Engine Backend
    DEFAULT_NEURAL_BACKEND: Literal["sklearn", "jax"] = "sklearn"



# Global default instance
DEFAULT_POLICY = DiscoveryNumericalPolicy()
DEFAULT_DISCOVERY_POLICY = DEFAULT_POLICY


def get_active_policy() -> DiscoveryNumericalPolicy:
    """Returns the active numerical policy (can be customized or injected)."""
    return DEFAULT_POLICY


def get_neural_discovery_backend(override: str | None = None) -> Literal["sklearn", "jax"]:
    """Resolves neural discovery backend: override > environment variable > policy default.

    Avoids ad-hoc os.environ checks scattered across discovery functions.
    """
    if override in {"sklearn", "jax"}:
        return override  # type: ignore[return-value]

    env_val = os.environ.get("NEURAL_DISCOVERY_ENGINE", "").strip().lower()
    if env_val in {"1", "true", "jax"}:
        return "jax"
    elif env_val in {"0", "false", "sklearn"}:
        return "sklearn"

    return DEFAULT_POLICY.DEFAULT_NEURAL_BACKEND
