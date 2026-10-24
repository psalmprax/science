"""Adapter from quantum candidates to the APEDP-X experiment ledger.

Status: experimental

The existing :class:`FullLineageExperimentLedger` already provides append-only
persistence, candidate lineage, and anti-snooping holdout burning.  This module
maps the quantum-specific evidence categories onto that machinery instead of
creating a second, weaker ledger.

Two rules are enforced here:

* A proposer-side screen is recorded, but it can never promote a candidate.
* A salted full-domain verifier failure burns the ``(benchmark, verifier,
  salt)`` confirmation slice; a replacement candidate cannot retry the same
  final salt.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, cast

from apedp_dyn.epistemic.experiment_ledger import (
    CandidateLifecycleState,
    DatasetRole,
    FullLineageExperimentLedger,
    HoldoutContaminationError,
)
from apedp_dyn.quantum_compilation.candidates import (
    CandidateStatus,
    CircuitCandidate,
    VerifierReceipt,
)

__all__ = ["QuantumCandidateLedger"]

_STATUS_TO_CORE = {
    CandidateStatus.PROPOSED: CandidateLifecycleState.PROPOSED,
    CandidateStatus.SCREEN_REJECTED: CandidateLifecycleState.PRE_SCREEN_REJECTED,
    CandidateStatus.VERIFIER_REJECTED: CandidateLifecycleState.CONFIRMATION_REJECTED,
    CandidateStatus.VERIFIED_NONIMPROVING: CandidateLifecycleState.DEVELOPMENT_PASSED,
    CandidateStatus.VERIFIED_IMPROVEMENT: CandidateLifecycleState.CONFIRMED_VALID,
    CandidateStatus.INSUFFICIENT_EVIDENCE: CandidateLifecycleState.UNRESOLVED,
    CandidateStatus.UNRESOLVED: CandidateLifecycleState.UNRESOLVED,
}


class QuantumCandidateLedger:
    """Thin quantum-evidence adapter over the APEDP-X full-lineage ledger."""

    def __init__(self, persistence_path: Path | str | None = None) -> None:
        self._ledger = FullLineageExperimentLedger(persistence_path=persistence_path)
        self._path = Path(persistence_path) if persistence_path else None

    @property
    def persistence_path(self) -> Path | None:
        return self._path

    def register(self, candidate: CircuitCandidate) -> CircuitCandidate:
        """Register a candidate without granting any correctness status."""
        numeric_parameters = {
            name: float(value)
            for name, value in candidate.parameters.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        self._ledger.register_candidate(
            candidate_id=candidate.candidate_id,
            proposer_id=candidate.generator_id,
            formula=json.dumps(candidate.parameters, sort_keys=True, default=str),
            parent_candidate_id=candidate.parent_candidate_id,
            feature_set=tuple(sorted(candidate.parameters)),
            parameter_choices=numeric_parameters,
        )
        self._ledger.update_lifecycle_state(
            candidate.candidate_id,
            CandidateLifecycleState.PROPOSED,
            metrics={"quantum_candidate": candidate.to_dict()},
        )
        return candidate

    def record_screen(
        self,
        candidate: CircuitCandidate,
        *,
        passed: bool,
        metrics: dict[str, Any],
        score_estimate: float | None,
        rejection_reason: str | None = None,
    ) -> CircuitCandidate:
        """Record the proposer-side screen; only the screen may be rejected here."""
        status = CandidateStatus.PROPOSED if passed else CandidateStatus.SCREEN_REJECTED
        updated = dataclasses.replace(
            candidate,
            status=status,
            screen_metrics=metrics,
            screen_score_estimate=score_estimate,
        )
        self._ledger.update_lifecycle_state(
            candidate.candidate_id,
            _STATUS_TO_CORE[status],
            rejection_reason=rejection_reason,
            metrics={
                "quantum_candidate": updated.to_dict(),
                "screen_metrics": metrics,
                "screen_score_estimate": score_estimate,
                "screen_passed": passed,
            },
        )
        return updated

    def record_verification(
        self,
        candidate: CircuitCandidate,
        receipt: VerifierReceipt,
        *,
        incumbent_score: float | None,
    ) -> CircuitCandidate:
        """Record an official-verifier receipt and map it to a lifecycle state.

        Promotion to ``CONFIRMED_VALID`` requires a *salted full-domain* PASS
        whose measured score is strictly lower than the incumbent.  A
        development-prefix PASS is recorded as ``DEVELOPMENT_PASSED`` and is
        never treated as a leaderboard result.
        """
        scope = receipt.verification_scope
        is_final = scope == "full-domain"
        role = DatasetRole.CONFIRMATION_HOLDOUT if is_final else DatasetRole.TRAINING_DEVELOPMENT
        slice_id = f"{receipt.verifier_sha256}:{receipt.salt_id}"
        if is_final:
            self._assert_final_slice_unused(candidate.benchmark, scope, slice_id)
        self._ledger.record_data_exposure(
            candidate_id=candidate.candidate_id,
            dataset_id=f"{candidate.benchmark}:{scope}",
            slice_identifier=slice_id,
            role=role,
            outcome_status="PASSED" if receipt.status == "PASS" else "FAILED",
        )

        if receipt.status != "PASS":
            status = CandidateStatus.VERIFIER_REJECTED
            reason = str(receipt.metrics.get("error", "official verifier did not pass"))
        elif not is_final:
            status = CandidateStatus.VERIFIED_NONIMPROVING
            reason = "development-prefix verification passed; final salted run is still required"
        elif receipt.score is None or incumbent_score is None:
            status = CandidateStatus.INSUFFICIENT_EVIDENCE
            reason = "full-domain pass without a comparable measured score"
        elif receipt.score < incumbent_score:
            status = CandidateStatus.VERIFIED_IMPROVEMENT
            reason = None
        else:
            status = CandidateStatus.VERIFIED_NONIMPROVING
            reason = "full-domain verification passed but did not beat the incumbent"

        updated = candidate.with_status(status, receipt=receipt)
        self._ledger.update_lifecycle_state(
            candidate.candidate_id,
            _STATUS_TO_CORE[status],
            rejection_reason=reason,
            metrics={
                "quantum_candidate": updated.to_dict(),
                "verifier_receipt": receipt.to_dict(),
                "verified_score": receipt.score,
                "verification_scope": scope,
            },
        )
        return updated

    def _assert_final_slice_unused(
        self, benchmark: str, scope: str, slice_id: str
    ) -> None:
        """A final confirmation salt is one-shot, even if the first run passed.

        ``FullLineageExperimentLedger`` burns only failed confirmation slices.
        A passed slice has still been consumed: reusing it would turn the
        grader salt into a development set.  This check makes that reuse raise
        before the second verifier run can happen.
        """
        dataset_id = f"{benchmark}:{scope}"
        for node in self._ledger.get_all_candidates():
            for exposure in node.data_exposures:
                if exposure.role is not DatasetRole.CONFIRMATION_HOLDOUT:
                    continue
                if (exposure.dataset_id, exposure.slice_identifier) == (dataset_id, slice_id):
                    raise HoldoutContaminationError(
                        "Final confirmation slice was already consumed by "
                        f"candidate {exposure.candidate_id!r}; refusing to reuse "
                        "a secret-salt holdout."
                    )

    def incumbent(self, benchmark: str) -> tuple[str | None, float | None]:
        """Return the best fully verified candidate id and score for a benchmark."""
        best_id: str | None = None
        best_score: float | None = None
        for node in self._ledger.get_all_candidates():
            if not node.candidate_id.startswith(f"{benchmark}-"):
                continue
            if node.state is not CandidateLifecycleState.CONFIRMED_VALID:
                continue
            score = node.metrics.get("verified_score")
            if not isinstance(score, (int, float)):
                continue
            if best_score is None or score < best_score:
                best_id, best_score = node.candidate_id, float(score)
        return best_id, best_score

    def node(self, candidate_id: str) -> Any:
        return self._ledger.get_candidate(candidate_id)

    def all_nodes(self) -> list[Any]:
        return cast(list[Any], self._ledger.get_all_candidates())

    def burned_holdouts(self) -> dict[tuple[str, str], str]:
        return cast(dict[tuple[str, str], str], self._ledger.get_burned_holdouts())

    def summary(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for node in self._ledger.get_all_candidates():
            quantum = node.metrics.get("quantum_candidate", {})
            rows.append(
                {
                    "candidate_id": node.candidate_id,
                    "proposer_id": node.proposer_id,
                    "state": node.state.value,
                    "rejection_reason": node.rejection_reason,
                    "screen_score_estimate": node.metrics.get("screen_score_estimate"),
                    "verified_score": node.metrics.get("verified_score"),
                    "verification_scope": node.metrics.get("verification_scope"),
                    "parameters": quantum.get("parameters", node.parameter_choices),
                }
            )
        return rows
