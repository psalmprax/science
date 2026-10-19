"""Canonical Preregistered Specification Loader and Integrity Verifier.

This module provides a single authoritative source of truth for preregistered
constants and gate thresholds across APEDP-X / CPMD discovery and adjudication.
It guarantees:
  1. The canonical JSON specification is hashed and verified against its pinned SHA-256.
  2. Runtime code imports these thresholds rather than defining ad-hoc literals.
  3. CI tests can detect any divergence between documentation, tests, and runtime behavior.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

SPEC_PATH = Path(__file__).resolve().parent.parent.parent / "specifications" / "canonical_preregistered_spec_v1.json"
PINNED_SPEC_SHA256 = "a11fd957c0b787856c62b698a06715e0acf58b393eaa52a70a905d5b0582d141"


class SpecificationIntegrityError(RuntimeError):
    """Raised when canonical specification hash does not match pinned immutable record."""
    pass


def load_and_verify_canonical_spec(enforce_hash: bool = True) -> dict[str, Any]:
    """Load canonical preregistered specification and verify SHA-256 hash."""
    if not SPEC_PATH.exists():
        raise FileNotFoundError(f"Canonical specification file not found: {SPEC_PATH}")

    content = SPEC_PATH.read_bytes()
    computed_hash = hashlib.sha256(content).hexdigest()

    if enforce_hash and computed_hash != PINNED_SPEC_SHA256:
        raise SpecificationIntegrityError(
            f"Canonical specification SHA-256 mismatch!\n"
            f"Expected: {PINNED_SPEC_SHA256}\n"
            f"Computed: {computed_hash}\n"
            f"Silent modifications to preregistered settings violate scientific integrity."
        )

    loaded = json.loads(content.decode("utf-8"))
    if not isinstance(loaded, dict):
        raise SpecificationIntegrityError("Canonical specification must be a JSON object")
    return loaded


# Load on module import
_SPEC = load_and_verify_canonical_spec(enforce_hash=True)
_THRESHOLDS = _SPEC["thresholds"]

# Authoritative Runtime Constants
TACTICAL_ACTIVE_MIN: float = float(_THRESHOLDS["tactical_active_fraction_min"])
TACTICAL_ACTIVE_MAX: float = float(_THRESHOLDS["tactical_active_fraction_max"])
CONTINUOUS_FACTOR_ACTIVE_MIN: float = float(_THRESHOLDS["continuous_factor_active_fraction_min"])
CONTINUOUS_FACTOR_ACTIVE_MAX: float = float(_THRESHOLDS["continuous_factor_active_fraction_max"])
TAIL_RISK_MIN_EVENTS: int = int(_THRESHOLDS["tail_risk_min_events"])

DSR_THRESHOLD: float = float(_THRESHOLDS["deflated_sharpe_ratio_threshold"])
DIEBOLD_MARIANO_P_VALUE_THRESHOLD: float = float(_THRESHOLDS["diebold_mariano_p_value_threshold"])
MAX_LITERATURE_R2: float = float(_THRESHOLDS["max_literature_r2"])

MIN_OOS_TRANSFER_RETENTION: float = float(_THRESHOLDS["min_oos_transfer_retention"])
MIN_NET_SHARPE: float = float(_THRESHOLDS["min_net_sharpe"])

REFEREE_MAX_FPR: float = float(_THRESHOLDS["referee_max_fpr"])
REFEREE_MIN_CAR: float = float(_THRESHOLDS["referee_min_car"])
REFEREE_MIN_VRR: float = float(_THRESHOLDS["referee_min_vrr"])

STOCK_SPLIT_INVARIANCE_TOLERANCE: float = float(_THRESHOLDS["stock_split_invariance_tolerance"])
PRUNING_HOLDOUT_R2_BASE_FLOOR: float = float(_THRESHOLDS["pruning_holdout_r2_base_floor"])
PRUNING_HOLDOUT_F_TEST_SIGNIFICANCE_LEVEL: float = float(
    _THRESHOLDS["pruning_holdout_f_test_significance_level"]
)
PRUNING_HOLDOUT_F_TEST_PREDICTOR_DOF: int = int(_THRESHOLDS["pruning_holdout_f_test_predictor_dof"])

LITERATURE_BASELINES: list[str] = list(_SPEC["literature_baselines"])
PROMOTION_STAGES: list[str] = list(_SPEC["promotion_stages"])
