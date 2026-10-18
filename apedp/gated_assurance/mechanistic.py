"""Mechanistic adjudication of a candidate against its declared evidence."""
from __future__ import annotations

import numpy as np

from apedp.epistemic import ModelEvidenceRecord

from .contracts import (
    AdjudicationReport,
    AdjudicationVerdict,
    ReplicationIndependenceVector,
    ScientificBenchmarkContract,
)


class MechanisticAdjudicationEngine:
    """Adjudicates discrepancies between model predictions and observations before refuting."""

    @classmethod
    def adjudicate(
        cls,
        candidate_record: ModelEvidenceRecord,
        contract: ScientificBenchmarkContract,
        features: np.ndarray,
        predictions: np.ndarray,
        observations: np.ndarray,
        independence: ReplicationIndependenceVector | None = None,
        estimated_sensor_noise_rel: float = 0.02,
    ) -> AdjudicationReport:
        if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
            return AdjudicationReport(
                verdict=AdjudicationVerdict.IMPLEMENTATION_FAILURE,
                confidence=1.0,
                boundary_concentration_ratio=0.0,
                noise_to_signal_ratio=float("inf"),
                reproducibility_independence=0.0,
                recommended_lifecycle_action="RETAIN_IN_CANDIDATE_FOR_DEBUGGING",
                rationale="Evaluation produced NaN or infinite values indicating numerical overflow.",
            )

        abs_err = np.abs(predictions - observations)
        denom = np.where(np.abs(observations) < 1e-12, 1.0, np.abs(observations))
        rel_err = abs_err / denom
        threshold = contract.falsification_thresholds.get("max_relative_error", 0.05)
        failing_indices = np.where(rel_err > threshold)[0]

        if len(failing_indices) == 0:
            # Nothing to adjudicate: the empirical channel supplies no evidence
            # against the mechanism. Reporting MODEL_FAILURE here would map to
            # MECHANISTIC_CONTRADICTION -> REFUTED, i.e. refuting a model on data
            # that agree with it. Abstain and let the caller weigh the
            # non-empirical signal that triggered adjudication.
            return AdjudicationReport(
                verdict=AdjudicationVerdict.NO_CONTRADICTION,
                confidence=0.0,
                boundary_concentration_ratio=0.0,
                noise_to_signal_ratio=0.0,
                reproducibility_independence=1.0,
                recommended_lifecycle_action="NO_ADJUDICATION_REQUIRED",
                rationale=(
                    "No prediction exceeds the contract's relative-error threshold, so "
                    "the empirical channel provides no evidence against the mechanism; "
                    "mechanistic refutation is not available from this evaluation."
                ),
            )

        # 1. Boundary Concentration Test (Domain Failure vs Model Failure)
        boundary_violations = 0
        total_violations = len(failing_indices)
        for low, high in candidate_record.validated_domain.values():
            span = high - low
            margin = 0.15 * span
            boundary_mask = (features[:, 0] <= low + margin) | (features[:, 0] >= high - margin)
            boundary_violations = int(np.sum(boundary_mask[failing_indices]))

        boundary_ratio = boundary_violations / max(1, total_violations)

        if boundary_ratio >= 0.70:
            return AdjudicationReport(
                verdict=AdjudicationVerdict.DOMAIN_FAILURE,
                confidence=boundary_ratio,
                boundary_concentration_ratio=boundary_ratio,
                noise_to_signal_ratio=estimated_sensor_noise_rel,
                reproducibility_independence=independence.composite_strength if independence else 0.5,
                recommended_lifecycle_action="DOMAIN_RESTRICTED",
                rationale=f"Discrepancies concentrated at domain boundary ({boundary_ratio*100:.1f}%). Core mechanism valid inside restricted sub-regime.",
            )

        # 2. Evidence Noise & Confounder Test
        mean_violation_size = float(np.mean(rel_err[failing_indices]))
        noise_ratio = estimated_sensor_noise_rel / max(1e-6, mean_violation_size)
        rep_strength = independence.composite_strength if independence else 0.5

        if noise_ratio > 0.40 or rep_strength < 0.35:
            return AdjudicationReport(
                verdict=AdjudicationVerdict.EVIDENCE_PROBLEM,
                confidence=0.80,
                boundary_concentration_ratio=boundary_ratio,
                noise_to_signal_ratio=noise_ratio,
                reproducibility_independence=rep_strength,
                recommended_lifecycle_action="CHALLENGED",
                rationale=f"High noise-to-signal ratio ({noise_ratio:.2f}) or low replication independence ({rep_strength:.2f}). Evidence insufficient to refute.",
            )

        # 3. Genuine Constitutive Model Failure
        return AdjudicationReport(
            verdict=AdjudicationVerdict.MODEL_FAILURE,
            confidence=0.95,
            boundary_concentration_ratio=boundary_ratio,
            noise_to_signal_ratio=noise_ratio,
            reproducibility_independence=rep_strength,
            recommended_lifecycle_action="REFUTED",
            rationale="High-fidelity in-domain empirical observations decisively contradict constitutive mechanism.",
        )
