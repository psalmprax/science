"""Operational readiness, gate auditing and blind challenge harness."""
from __future__ import annotations

import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from apedp.epistemic import DataPartitionVault, ModelEvidenceRecord

from .contracts import (
    FailureSeverity,
    ScientificBenchmarkContract,
)


@dataclass(frozen=True)
class OperationalReadinessReport:
    """Formal audit report determining if a Contract_Validated model can enter Operational deployment."""
    model_id: str
    is_operational_ready: bool
    robustness_score: float
    drift_monitoring_enabled: bool
    reproducibility_hash_verified: bool
    resource_budget_passed: bool
    fail_safe_boundary_clamp_defined: bool
    unmet_requirements: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "is_operational_ready": self.is_operational_ready,
            "robustness_score": round(self.robustness_score, 4),
            "drift_monitoring_enabled": self.drift_monitoring_enabled,
            "reproducibility_hash_verified": self.reproducibility_hash_verified,
            "resource_budget_passed": self.resource_budget_passed,
            "fail_safe_boundary_clamp_defined": self.fail_safe_boundary_clamp_defined,
            "unmet_requirements": list(self.unmet_requirements),
        }


class OperationalGateAuditor:
    """Dedicated operational readiness gate: Contract_Validated -> Operational."""

    @staticmethod
    def audit_readiness(
        candidate_record: ModelEvidenceRecord,
        eval_fn: Callable[[np.ndarray], np.ndarray],
        sample_inputs: np.ndarray,
        max_latency_ms: float = 50.0,
        drift_monitoring_configured: bool = True,
        fail_safe_clamp_defined: bool = True,
    ) -> OperationalReadinessReport:
        unmet: list[str] = []

        # 1. Robustness under input perturbation
        pert_low = sample_inputs * 0.99
        pert_high = sample_inputs * 1.01
        try:
            _ = eval_fn(sample_inputs)
            res_low = eval_fn(pert_low)
            res_high = eval_fn(pert_high)
            grad_est = np.abs((res_high - res_low) / np.maximum(1e-12, np.abs(pert_high - pert_low)))
            robustness = float(np.mean(np.isfinite(grad_est) & (grad_est < 1e8)))
            if robustness < 0.90:
                unmet.append(f"Input perturbation robustness score too low ({robustness*100:.1f}% < 90%)")
        except (RuntimeError, TypeError, ValueError, ZeroDivisionError, ArithmeticError, OverflowError) as e:
            robustness = 0.0
            unmet.append(f"Perturbation test crashed: {e}")

        # 2. Resource constraints
        t0 = time.perf_counter()
        _ = eval_fn(sample_inputs)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        resource_passed = latency_ms <= max_latency_ms
        if not resource_passed:
            unmet.append(f"Execution latency {latency_ms:.2f}ms exceeds maximum {max_latency_ms:.2f}ms")

        # 3. Drift & safety
        if not drift_monitoring_configured:
            unmet.append("No active distribution drift monitoring configured.")
        if not fail_safe_clamp_defined:
            unmet.append("No fail-safe boundary clamp defined for out-of-domain runtime calls.")

        # 4. Content hash reproducibility check.
        #    The digest is recomputed from the record's content and compared, so a
        #    tampered or never-computed record is rejected. (A 64-character string
        #    is not evidence of anything.)
        if not candidate_record.record_hash:
            hash_verified = False
            unmet.append("Cryptographic content hash missing from the evidence record.")
        elif not candidate_record.verify_record_hash():
            hash_verified = False
            unmet.append(
                "Content hash mismatch: record_hash does not match a digest recomputed "
                "from this record, so the record was altered or the hash was never computed."
            )
        else:
            hash_verified = True

        is_ready = len(unmet) == 0
        return OperationalReadinessReport(
            model_id=candidate_record.model_id,
            is_operational_ready=is_ready,
            robustness_score=robustness,
            drift_monitoring_enabled=drift_monitoring_configured,
            reproducibility_hash_verified=hash_verified,
            resource_budget_passed=resource_passed,
            fail_safe_boundary_clamp_defined=fail_safe_clamp_defined,
            unmet_requirements=tuple(unmet),
        )


class BlindChallengeHarness:
    """Scientific Holdout Laboratory: Evaluates candidate models against sealed challenge datasets.

    Sealing and evaluation are separate operations on purpose:

    * :meth:`seal_challenge` registers a labeled partition during *setup* and
      returns an opaque content hash. Call it before any candidate exists.
    * :meth:`evaluate_blind` scores a candidate against an already-sealed
      partition. It never receives labels and only aggregate residual
      statistics leave the harness -- predictions and targets are never
      returned to the caller, so a holdout cannot be exfiltrated through the
      evaluation API and re-used for fitting.

    Sealed partitions are copied on registration, so a caller mutating the
    arrays afterwards cannot retroactively change what was sealed.
    """

    def __init__(self, vault: DataPartitionVault | None = None):
        self._sealed_challenges: dict[str, dict[str, Any]] = {}
        self.vault = vault or DataPartitionVault(
            development_data_hash="dev_seed_partition_42",
            selection_data_hash="sel_seed_partition_43",
            falsification_data_hash="fals_seed_partition_44",
            locked_external_test_hash="ext_seed_partition_45",
            is_falsification_locked=True,
        )

    @staticmethod
    def content_hash(features: np.ndarray, targets: np.ndarray) -> str:
        """Opaque SHA-256 content hash identifying a (features, targets) partition."""
        features_arr = np.ascontiguousarray(features)
        targets_arr = np.ascontiguousarray(targets)
        return hashlib.sha256(features_arr.tobytes() + targets_arr.tobytes()).hexdigest()

    def seal_challenge(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Seal a labeled partition and return its content hash.

        This is the registration half of the blind protocol and should run once,
        during partition setup, before any candidate is scored. Everything about
        the partition is only reachable through the returned hash.
        """
        dataset_hash = self.content_hash(features, targets)
        self.register_sealed_challenge(
            dataset_hash=dataset_hash,
            features=features,
            targets=targets,
            metadata=dict(metadata or {}),
        )
        return dataset_hash

    def register_sealed_challenge(
        self,
        dataset_hash: str,
        features: np.ndarray,
        targets: np.ndarray,
        metadata: dict[str, Any]
    ) -> None:
        """Stores challenge test partition securely indexed by SHA-256 hash.

        Prefer :meth:`seal_challenge`, which derives the index from the partition
        content rather than trusting a caller-supplied label. Arrays are copied
        so later mutation by the caller cannot alter the sealed partition.
        """
        self._sealed_challenges[dataset_hash] = {
            "features": np.array(features, copy=True),
            "targets": np.array(targets, copy=True),
            "metadata": dict(metadata),
            "locked": True,
        }

    def has_challenge(self, dataset_hash: str) -> bool:
        """True when a partition has already been sealed under ``dataset_hash``."""
        return dataset_hash in self._sealed_challenges

    def evaluate_blind(
        self,
        candidate_evaluator: Callable[[np.ndarray], np.ndarray],
        dataset_hash: str,
        contract: ScientificBenchmarkContract,
        caller_phase: str = "falsification"
    ) -> dict[str, Any]:
        """Blindly evaluates candidate model predictions without revealing dataset contents to caller.

        Returns aggregate residual statistics only. Neither the prediction vector
        nor the sealed targets are returned.
        """
        self.vault.verify_isolation(dataset_hash, caller_phase)

        if dataset_hash not in self._sealed_challenges:
            raise KeyError(
                f"Sealed challenge with hash '{dataset_hash}' not found in vault. "
                "Partitions must be sealed via seal_challenge() during setup; "
                "evaluate_blind never registers implicitly."
            )

        challenge = self._sealed_challenges[dataset_hash]
        features: np.ndarray = challenge["features"]
        targets: np.ndarray = challenge["targets"]

        try:
            predictions = candidate_evaluator(features)
        except (RuntimeError, TypeError, ValueError, ZeroDivisionError, ArithmeticError, OverflowError) as e:
            return {
                "status": FailureSeverity.IMPLEMENTATION_FAILURE,
                "error": f"Evaluation crashed: {e}",
                "coverage": 0.0,
                "max_relative_error": float("inf"),
            }

        if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
            return {
                "status": FailureSeverity.IMPLEMENTATION_FAILURE,
                "error": "Predictions contain NaN or Inf",
                "coverage": 0.0,
                "max_relative_error": float("inf"),
            }

        abs_err = np.abs(predictions - targets)
        denom = np.where(np.abs(targets) < 1e-12, 1.0, np.abs(targets))
        rel_err = abs_err / denom

        threshold = contract.falsification_thresholds.get("max_relative_error", 0.05)
        in_bounds = rel_err <= threshold
        coverage = float(np.mean(in_bounds))

        worst_severity = FailureSeverity.PASS
        for val_min, val_max in contract.prediction_intervals.values():
            if np.any(predictions < val_min) or np.any(predictions > val_max):
                worst_severity = FailureSeverity.PREDICTIVE_FAILURE

        if coverage < 0.90 and worst_severity == FailureSeverity.PASS:
            worst_severity = FailureSeverity.PREDICTIVE_FAILURE

        # Aggregate statistics only: returning ``predictions``/``targets`` here
        # would hand the sealed partition back to the caller, defeating the
        # holdout entirely.
        return {
            "status": worst_severity,
            "coverage": coverage,
            "mean_relative_error": float(np.mean(rel_err)),
            "max_relative_error": float(np.max(rel_err)),
            "n_samples": len(targets),
        }
