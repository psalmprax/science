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
import warnings
from pathlib import Path
from typing import Any

SPEC_PATH = Path(__file__).resolve().parent.parent.parent / "specifications" / "canonical_preregistered_spec_v1.json"
MANIFEST_PATH = SPEC_PATH.parent / "governance_manifest.json"
PINNED_SPEC_SHA256 = "dbf67c5ac443073721421eba456519a0eda09ae07680d638c26f1d019b0b2343"
# Pin history (AGENTS.md §1.3 re-certification trail, stored in-code next to the
# pin):
#   a11fd957c0b787856c62b698a06715e0acf58b393eaa52a70a905d5b0582d141
#   → 5bb42f0341b7a024d21280ee69922bd7dbfebd33cf55506253e3f5248411ec04
#   (2026-10-21 external audit FLAW-13 remediation: added the ODE referee's
#   preregistered holdout thresholds ode_referee_min_mean_r2=0.95 and
#   ode_referee_min_worst_r2=0.85 — values unchanged in behavior, moved from
#   hardcoded literals into the canonical file per the specification-
#   centralization rule. Ed25519 governance manifest re-signing remains
#   trustee-blocked; see FLAW-01.)
#   → a5e13979529d002362cb132454c2a6899455af8367656a0f1ab226eeab0c09de
#   (2026-10-28 Hypothesis B preregistration: added the temporal-rate
#   modulation null-test thresholds. No existing threshold changed. The
#   governance manifest remains un-resigned; FLAW-01 is still OPEN.)
#   → dbf67c5ac443073721421eba456519a0eda09ae07680d638c26f1d019b0b2343
#   (2026-10-29 integrity hardening: added
#   temporal_rate_min_compositions=2 so the common-mode claim's requirement
#   for two physically distinct compositions is a canonical threshold rather
#   than a hardcoded literal. No existing threshold changed. The governance
#   manifest remains un-resigned; FLAW-01 is still OPEN.)


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


def governance_manifest_status() -> dict[str, Any]:
    """Report whether the signed governance manifest covers this spec revision.

    The hash check above proves the disk file matches this module's pin. It says
    nothing about the Ed25519 governance manifest, which seals the hash the
    trustees signed. The two are allowed to disagree — they do (FLAW-01) — and
    when they do the Python pin is green while the signed certification is
    stale. That disagreement is returned here instead of being left to a test a
    reader has to know to run.
    """
    sealed: str | None = None
    try:
        payload = json.loads(MANIFEST_PATH.read_bytes().decode("utf-8"))
        raw = payload.get("spec_sha256") if isinstance(payload, dict) else None
        sealed = raw if isinstance(raw, str) else None
    except (OSError, ValueError):
        sealed = None
    return {
        "manifest_path": str(MANIFEST_PATH),
        "manifest_present": sealed is not None,
        "sealed_sha256": sealed,
        "disk_sha256": PINNED_SPEC_SHA256,
        "drift": sealed is not None and sealed != PINNED_SPEC_SHA256,
    }


# Load on module import
_SPEC = load_and_verify_canonical_spec(enforce_hash=True)
_THRESHOLDS = _SPEC["thresholds"]

GOVERNANCE_STATUS: dict[str, Any] = governance_manifest_status()
GOVERNANCE_SPEC_SHA256: str | None = GOVERNANCE_STATUS["sealed_sha256"]
GOVERNANCE_DRIFT: bool = bool(GOVERNANCE_STATUS["drift"])

if GOVERNANCE_DRIFT:
    warnings.warn(
        "Governance manifest is sealed to a different canonical-spec hash "
        f"({str(GOVERNANCE_SPEC_SHA256)[:12]}…) than the one on disk "
        f"({PINNED_SPEC_SHA256[:12]}…). The Python pin passes, but the signed "
        "certification does not cover this spec revision; re-signing requires "
        "the registered trustees (AGENTS.md §1.3, FLAW-01).",
        RuntimeWarning,
        stacklevel=2,
    )

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
STAGE2_MIN_IC: float = float(_THRESHOLDS.get("stage2_min_ic", 0.02))
STAGE4_MIN_OOS_IC: float = float(_THRESHOLDS.get("stage4_min_oos_ic", 0.015))
STAGE4_OOS_ALPHA: float = float(_THRESHOLDS.get("stage4_oos_alpha", 0.05))
STAGE5_MIN_FEE_BPS: float = float(_THRESHOLDS.get("stage5_min_fee_bps", 3.0))
STAGE6_PROSPECTIVE_SHARPE_HURDLE: float = float(_THRESHOLDS.get("stage6_prospective_sharpe_hurdle", 0.70))
STAGE6_MAX_DRAWDOWN_CEILING: float = float(_THRESHOLDS.get("stage6_max_drawdown_ceiling", 0.25))

REFEREE_MAX_FPR: float = float(_THRESHOLDS["referee_max_fpr"])
REFEREE_MIN_CAR: float = float(_THRESHOLDS["referee_min_car"])
REFEREE_MIN_VRR: float = float(_THRESHOLDS["referee_min_vrr"])

ODE_REFEREE_MIN_MEAN_R2: float = float(_THRESHOLDS["ode_referee_min_mean_r2"])
ODE_REFEREE_MIN_WORST_R2: float = float(_THRESHOLDS["ode_referee_min_worst_r2"])

TEMPORAL_RATE_MIN_CLOCKS: int = int(_THRESHOLDS["temporal_rate_min_clocks"])
TEMPORAL_RATE_MIN_COMPOSITIONS: int = int(
    _THRESHOLDS["temporal_rate_min_compositions"]
)
TEMPORAL_RATE_MIN_DRIVE_CYCLES: float = float(_THRESHOLDS["temporal_rate_min_drive_cycles"])
TEMPORAL_RATE_MIN_SAMPLES_PER_CYCLE: float = float(
    _THRESHOLDS["temporal_rate_min_samples_per_cycle"]
)
TEMPORAL_RATE_HARMONICS: int = int(_THRESHOLDS["temporal_rate_harmonics"])
TEMPORAL_RATE_SURROGATE_DRAWS: int = int(_THRESHOLDS["temporal_rate_surrogate_draws"])
TEMPORAL_RATE_FAMILYWISE_ALPHA: float = float(_THRESHOLDS["temporal_rate_familywise_alpha"])
TEMPORAL_RATE_HETEROGENEITY_ALPHA: float = float(
    _THRESHOLDS["temporal_rate_heterogeneity_alpha"]
)
TEMPORAL_RATE_MIN_FRACTIONAL_AMPLITUDE: float = float(
    _THRESHOLDS["temporal_rate_min_fractional_amplitude"]
)
TEMPORAL_RATE_MIN_CALIBRATION_SNR: float = float(_THRESHOLDS["temporal_rate_min_calibration_snr"])
TEMPORAL_RATE_CALIBRATION_TRIALS: int = int(_THRESHOLDS["temporal_rate_calibration_trials"])
TEMPORAL_RATE_CALIBRATION_FPR_MAX: float = float(
    _THRESHOLDS["temporal_rate_calibration_fpr_max"]
)
TEMPORAL_RATE_CALIBRATION_POWER_MIN: float = float(
    _THRESHOLDS["temporal_rate_calibration_power_min"]
)
TEMPORAL_RATE_INJECTION_FRACTIONAL_AMPLITUDE: float = float(
    _THRESHOLDS["temporal_rate_injection_fractional_amplitude"]
)

STOCK_SPLIT_INVARIANCE_TOLERANCE: float = float(_THRESHOLDS["stock_split_invariance_tolerance"])
PRUNING_HOLDOUT_R2_BASE_FLOOR: float = float(_THRESHOLDS["pruning_holdout_r2_base_floor"])
PRUNING_HOLDOUT_F_TEST_SIGNIFICANCE_LEVEL: float = float(
    _THRESHOLDS["pruning_holdout_f_test_significance_level"]
)
PRUNING_HOLDOUT_F_TEST_PREDICTOR_DOF: int = int(_THRESHOLDS["pruning_holdout_f_test_predictor_dof"])

LITERATURE_BASELINES: list[str] = list(_SPEC["literature_baselines"])
PROMOTION_STAGES: list[str] = list(_SPEC["promotion_stages"])
