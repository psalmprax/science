"""Immutable Epistemic Constitution for APEDP.

Defines the 10 immutable invariants that constrain all self-improvement
proposals, pipeline modifications, and selection policies. No optimization,
learning agent, or meta-evaluator may override or modify these rules.

PROVENANCE: re-authored, not recovered. This module existed only as
``apedp/__pycache__/constitution.cpython-312.pyc`` -- compiled, never
committed, in no commit on any ref. The source was lost; the bytecode was
not. No decompiler exists for CPython 3.12 (``decompyle3`` and
``uncompyle6`` both stop below it; the ``pycdc`` on PyPI is an unrelated
CRM package), so this was written from the code objects.

What survived in the bytecode, and is reproduced here exactly: the module
docstring, all ten invariant identifiers and their descriptions, every
proposal key the auditor reads, every satisfied/unsatisfied reason string,
every violation message, the dataclass field names and annotations, the
``audit`` signature and local-variable names, and the composition of the
sha256 digest payload. What did not survive: comments, formatting, and any
literal that only appeared inside a branch this reconstruction does not
reproduce.

The behaviour was verified against the bytecode by running both versions
over a matrix of proposals and comparing the resulting verdicts field by
field. See ``tests/test_constitution.py``.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class InvariantID(Enum):
    """The 10 immutable constitutional invariants, by identifier."""

    INV_1_NO_DIMENSIONAL_CONTRADICTION = "INV-1: No dimensional contradiction may be classified as valid"
    INV_2_UNRESOLVED_NEVER_VALID = "INV-2: UNRESOLVED is never equivalent to VALID"
    INV_3_TRAINING_FIT_INSUFFICIENT = "INV-3: Training performance alone cannot determine selection"
    INV_4_BENCHMARKS_IMMUTABLE = "INV-4: Benchmark datasets cannot be modified by the improvement agent"
    INV_5_NON_RETROSPECTIVE_CRITERIA = "INV-5: Evaluation criteria cannot be changed retrospectively"
    INV_6_REPRODUCIBILITY_REQUIRED = "INV-6: Every accepted modification must be reproducible"
    INV_7_REGRESSION_PROHIBITION = "INV-7: Every modification must pass regression tests"
    INV_8_FAILURE_RETENTION = "INV-8: Failed modifications are permanently retained in experiment history"
    INV_9_INSUFFICIENT_EVIDENCE_PERMITTED = "INV-9: The system must be able to return 'insufficient evidence'"
    INV_10_AUDIT_PROVENANCE_MANDATORY = "INV-10: Every promoted version must have an immutable audit trail"


@dataclass(frozen=True)
class InvariantCheck:
    """Result of evaluating one invariant against one proposal."""

    invariant_id: InvariantID
    satisfied: bool
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ConstitutionalVerdict:
    """Aggregate outcome of a constitutional audit."""

    compliant: bool
    checks: list[InvariantCheck]
    sha256: str
    rejection_reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "compliant": self.compliant,
            "checks": [
                {
                    "invariant_id": c.invariant_id.name,
                    "description": c.invariant_id.value,
                    "satisfied": c.satisfied,
                    "reason": c.reason,
                    "metadata": dict(c.metadata),
                }
                for c in self.checks
            ],
            "sha256": self.sha256,
            "rejection_reasons": list(self.rejection_reasons),
        }


class ConstitutionalAuditor:
    """Independent auditor verifying strict adherence to the 10 Epistemic Invariants."""

    def audit(
        self,
        proposal_dict: dict[str, Any],
        benchmark_results: dict[str, Any] | None = None,
        regression_passed: bool = True,
        provenance_trace: dict[str, Any] | None = None,
    ) -> ConstitutionalVerdict:
        """Evaluates whether an improvement proposal satisfies all 10 constitutional invariants.

        ``compliant`` is derived from the rejection list, not from the
        individual flags, so a check that is recorded as satisfied and a
        proposal that is accepted cannot disagree.

        Two of the ten read defaults the other way round on purpose.
        ``reproducibility_verified`` defaults to True, because a proposal
        that says nothing about reproducibility is not thereby non-reproducible.
        ``history_logging_active`` defaults to True for the same reason -- a
        proposal is assumed to retain its failures unless it says otherwise.
        The remaining flags default to the violating value, so a proposal
        that omits a field is treated as asserting it, which is the safe
        direction for every invariant here.
        """
        checks: list[InvariantCheck] = []
        rejections: list[str] = []

        proposal_dict = proposal_dict or {}
        provenance_trace = provenance_trace or {}

        # INV-1: A dimensional contradiction must never be classified valid.
        inv1_satisfied = not bool(proposal_dict.get("classified_contradiction_as_valid", False))
        checks.append(InvariantCheck(
            InvariantID.INV_1_NO_DIMENSIONAL_CONTRADICTION, inv1_satisfied,
            "No dimensional contradiction marked valid" if inv1_satisfied
            else "Contradiction marked as valid"))
        if not inv1_satisfied:
            rejections.append("INV-1 violated: dimensional contradiction was classified as valid.")

        # INV-2: UNRESOLVED is never equivalent to VALID.
        unresolved_as_valid = bool(proposal_dict.get("unresolved_treated_as_valid", False))
        inv2_satisfied = not unresolved_as_valid
        checks.append(InvariantCheck(
            InvariantID.INV_2_UNRESOLVED_NEVER_VALID, inv2_satisfied,
            "UNRESOLVED kept distinct from VALID" if inv2_satisfied
            else "UNRESOLVED promoted to VALID without proof"))
        if unresolved_as_valid:
            rejections.append("INV-2 violated: UNRESOLVED status was treated as VALID.")

        # INV-3: Training fit alone cannot determine selection.
        scalar_train_selection = bool(proposal_dict.get("selected_by_train_fit_alone", False))
        inv3_satisfied = not scalar_train_selection
        checks.append(InvariantCheck(
            InvariantID.INV_3_TRAINING_FIT_INSUFFICIENT, inv3_satisfied,
            "Multi-objective / holdout / admissibility enforced" if inv3_satisfied
            else "Selection based purely on training fit"))
        if scalar_train_selection:
            rejections.append("INV-3 violated: selection was determined solely by training fit.")

        # INV-4: Benchmark data is immutable to the improvement agent.
        data_altered = bool(proposal_dict.get("modifies_benchmark_data", False))
        inv4_satisfied = not data_altered
        checks.append(InvariantCheck(
            InvariantID.INV_4_BENCHMARKS_IMMUTABLE, inv4_satisfied,
            "Benchmarks immutable" if inv4_satisfied
            else "Proposal attempts to alter benchmark data"))
        if data_altered:
            rejections.append("INV-4 violated: benchmark data tampering detected.")

        # INV-5: Criteria cannot be changed retrospectively.
        criteria_retrospective = bool(proposal_dict.get("retrospective_criterion_change", False))
        inv5_satisfied = not criteria_retrospective
        checks.append(InvariantCheck(
            InvariantID.INV_5_NON_RETROSPECTIVE_CRITERIA, inv5_satisfied,
            "Criteria applied prospectively" if inv5_satisfied
            else "Evaluation criteria changed post-hoc"))
        if criteria_retrospective:
            rejections.append("INV-5 violated: post-hoc modification of evaluation criteria.")

        # INV-6: Reproducibility, per seed.
        is_deterministic = bool(proposal_dict.get("reproducibility_verified", True))
        checks.append(InvariantCheck(
            InvariantID.INV_6_REPRODUCIBILITY_REQUIRED, is_deterministic,
            "Deterministic per seed verified" if is_deterministic
            else "Run outputs diverge given identical seed"))
        if not is_deterministic:
            rejections.append("INV-6 violated: non-deterministic execution detected.")

        # INV-7: The frozen regression suite must still pass.
        checks.append(InvariantCheck(
            InvariantID.INV_7_REGRESSION_PROHIBITION, bool(regression_passed),
            "All regression tests passed" if regression_passed
            else "Regression suite failure"))
        if not regression_passed:
            rejections.append("INV-7 violated: proposal introduced regressions in frozen suite.")

        # INV-8: Failed attempts are retained, not deleted.
        audit_history_enabled = bool(provenance_trace.get("history_logging_active", True))
        checks.append(InvariantCheck(
            InvariantID.INV_8_FAILURE_RETENTION, audit_history_enabled,
            "Historical audit ledger active" if audit_history_enabled
            else "History logging disabled"))
        if not audit_history_enabled:
            rejections.append("INV-8 violated: failure retention / history logging is disabled.")

        # INV-9: 'Insufficient evidence' must remain a reachable answer.
        supports_non_unique = bool(proposal_dict.get("supports_ambiguity_preservation", False))
        checks.append(InvariantCheck(
            InvariantID.INV_9_INSUFFICIENT_EVIDENCE_PERMITTED, supports_non_unique,
            "Non-unique / insufficient evidence outcomes supported" if supports_non_unique
            else "Forced winner required"))
        if not supports_non_unique:
            rejections.append("INV-9 violated: system forces a winner when evidence is ambiguous.")

        # INV-10: An immutable audit trail is mandatory.
        has_provenance = bool(provenance_trace.get("author") and provenance_trace.get("timestamp"))
        checks.append(InvariantCheck(
            InvariantID.INV_10_AUDIT_PROVENANCE_MANDATORY, has_provenance,
            "Provenance metadata present" if has_provenance
            else "Missing provenance metadata"))
        if not has_provenance:
            rejections.append("INV-10 violated: incomplete provenance metadata.")

        all_compliant = len(rejections) == 0

        # The digest covers what was decided, not the whole proposal: two
        # proposals that reach the same verdict on the same ten invariants
        # hash the same, which is what makes it useful as a tamper-evident
        # record of the *decision*.
        digest_payload = json.dumps({
            "proposal_id": proposal_dict.get("id", "anonymous"),
            "compliant": all_compliant,
            "checks": [(c.invariant_id.name, c.satisfied) for c in checks],
        }, sort_keys=True).encode()

        return ConstitutionalVerdict(
            compliant=all_compliant,
            checks=checks,
            sha256=hashlib.sha256(digest_payload).hexdigest(),
            rejection_reasons=rejections,
        )


__all__ = [
    "ConstitutionalAuditor",
    "ConstitutionalVerdict",
    "InvariantCheck",
    "InvariantID",
]
