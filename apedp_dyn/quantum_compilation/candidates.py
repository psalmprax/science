"""Typed candidate and verifier-receipt records for APEDP-Q.

Status: experimental

A circuit candidate is deliberately *not* an equation candidate.  Its identity
is the pair ``(contract_sha256, artifact_sha256)``: the same parameters rebuilt
against a different verifier are a different candidate, and the same parameters
rebuilt with different bytes are a different artifact.  The proposer emits this
record; only the official verifier receipt can change its status.
"""
from __future__ import annotations

import dataclasses
import datetime
import hashlib
import json
from enum import Enum
from pathlib import Path
from typing import Any

__all__ = [
    "CORRECTNESS_CHECKS",
    "CORRECTNESS_CLAIMING_STATUSES",
    "CandidateStatus",
    "CircuitCandidate",
    "EvidenceIncompleteError",
    "VerifierReceipt",
    "candidate_id_for",
    "compute_provenance_fingerprint",
    "compute_spec_fingerprint",
    "compute_structural_fingerprint",
    "evidence_from_receipt",
    "missing_correctness_checks",
    "normalise_correctness_evidence",
    "sha256_file",
]


class CandidateStatus(str, Enum):
    """Quantum-specific lifecycle states.

    These are broader than the existing trading hypothesis states and are
    mapped explicitly in :mod:`apedp_dyn.quantum_compilation.ledger`.
    """

    PROPOSED = "PROPOSED"
    SCREEN_REJECTED = "SCREEN_REJECTED"
    VERIFIER_REJECTED = "VERIFIER_REJECTED"
    #: The verifier produced no measured verdict (timeout, crash, unparseable
    #: summary). A capability gap, not evidence against the circuit.
    UNVERIFIABLE = "UNVERIFIABLE"
    VERIFIED_NONIMPROVING = "VERIFIED_NONIMPROVING"
    VERIFIED_IMPROVEMENT = "VERIFIED_IMPROVEMENT"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    UNRESOLVED = "UNRESOLVED"


#: The three checks a correctness claim has to account for. Value is the circuit's
#: computational output, junk is the workspace it must leave as it found it, and phase
#: is the tracked global phase the DQI verifier reports as ``phase_flipped``.
CORRECTNESS_CHECKS: tuple[str, ...] = ("value", "junk", "phase")

#: Statuses that assert the circuit is *correct*, not merely built or priced.
CORRECTNESS_CLAIMING_STATUSES: frozenset[Any] = frozenset({
    CandidateStatus.VERIFIED_NONIMPROVING,
    CandidateStatus.VERIFIED_IMPROVEMENT,
})


class EvidenceIncompleteError(RuntimeError):
    """Raised when a correctness claim has no evidence for one of the three checks."""


#: What each benchmark's verifier can actually establish about correctness. A verifier
#: that simulates the circuit per shot can supply value/phase evidence; one that only
#: prices it cannot, and says so rather than leaving the checks silently absent.
BENCHMARK_EVIDENCE_MODEL: dict[str, str | None] = {
    "dqi": "per-shot",
    "invsqrt": "cost-model-only",
    "dfthcsos": "cost-model-only",
}

_COST_MODEL_REASON = (
    "the {benchmark} verifier reports `status` and `score` only: it builds the circuit "
    "and prices it, and does not simulate it on shots, so this check is not something "
    "its PASS establishes"
)


def _benchmark_family(benchmark: str) -> str:
    """`DQI-...`/`dqi_v6`/`invsqrt` all map to their verifier family."""
    lowered = (benchmark or "").lower()
    for family in BENCHMARK_EVIDENCE_MODEL:
        if lowered.startswith(family):
            return family
    return ""


def normalise_correctness_evidence(evidence: dict[str, Any] | None, *,
                                   benchmark: str = "") -> dict[str, dict[str, Any]]:
    """Validate a declared evidence mapping against what the verifier can establish.

    Each of :data:`CORRECTNESS_CHECKS` must be accounted for exactly once, as one of:

    ``{"measured": True, "passed": bool, "source": str}``
        the check was executed; ``source`` names where the figure came from.
    ``{"covered_by": <check>, "reason": str}``
        another *measured and passed* check subsumes this one (the DQI verifier folds
        the ancilla read into ``value_errors``, for instance).
    ``{"not_applicable": str}``
        the verifier cannot establish it, and the reason says why. This is a statement
        about the verifier, not a waiver: the audit behind
        `docs/REPRESENTATION_AWARE_UNCOMPUTATION.md` found 13 of 21 records claiming
        correctness with these checks silently absent. For a *per-shot* verifier that
        will not do — at least one check must have been measured and passed — and only
        a cost-model verifier, which has no per-shot semantics to measure, may return
        all three as inapplicable.
    """
    evidence = dict(evidence or {})
    unknown = sorted(set(evidence) - set(CORRECTNESS_CHECKS))
    if unknown:
        raise EvidenceIncompleteError(
            f"unknown correctness check(s) {unknown}; expected {list(CORRECTNESS_CHECKS)}"
        )
    normalised: dict[str, dict[str, Any]] = {}
    for check in CORRECTNESS_CHECKS:
        entry = evidence.get(check)
        if not isinstance(entry, dict):
            raise EvidenceIncompleteError(
                f"no {check!r} evidence; a correctness claim needs value, junk and phase "
                f"accounted for (`not_applicable` with a reason is allowed)"
            )
        if "measured" in entry:
            if not entry.get("source"):
                raise EvidenceIncompleteError(
                    f"{check!r} evidence has no `source`; say which figure it is"
                )
            normalised[check] = {"measured": True,
                                 "passed": bool(entry.get("passed")),
                                 "source": str(entry["source"])}
        elif "covered_by" in entry:
            covering = str(entry["covered_by"])
            if covering not in CORRECTNESS_CHECKS or covering == check:
                raise EvidenceIncompleteError(
                    f"{check!r} is covered by {covering!r}, which is not another check"
                )
            if not entry.get("reason"):
                raise EvidenceIncompleteError(
                    f"{check!r} covered by {covering!r} without a reason"
                )
            normalised[check] = {"covered_by": covering, "reason": str(entry["reason"])}
        elif "not_applicable" in entry:
            if not entry.get("not_applicable"):
                raise EvidenceIncompleteError(f"{check!r}: `not_applicable` needs a reason")
            normalised[check] = {"not_applicable": str(entry["not_applicable"])}
        else:
            raise EvidenceIncompleteError(
                f"{check!r} evidence must be measured, covered_by or not_applicable"
            )
    for check, entry in normalised.items():
        covered_by_id = str(entry.get("covered_by") or "") or None
        if covered_by_id is None:
            continue
        target = normalised[covered_by_id]
        if not target.get("measured") or not target.get("passed"):
            raise EvidenceIncompleteError(
                f"{check!r} is covered by {covered_by_id!r}, which is not a passed "
                f"measurement ({target})"
            )
        if not entry["reason"]:
            raise EvidenceIncompleteError(f"{check!r}: covering needs a reason")
    per_shot = BENCHMARK_EVIDENCE_MODEL.get(_benchmark_family(benchmark)) == "per-shot"
    if per_shot and not any(entry.get("measured") and entry.get("passed")
                            for entry in normalised.values()):
        raise EvidenceIncompleteError(
            "no check was measured and passed; for a per-shot verifier "
            "`not_applicable` entries alone do not establish that a circuit is correct"
        )
    for check, entry in normalised.items():
        if entry.get("measured") and not entry.get("passed"):
            raise EvidenceIncompleteError(
                f"{check!r} was measured and failed; a failing check cannot sit under a "
                f"correctness claim"
            )
    return normalised


def missing_correctness_checks(evidence: dict[str, Any] | None, *,
                               benchmark: str = "") -> list[str]:
    """Which checks a correctness claim would be missing, without raising."""
    try:
        normalise_correctness_evidence(evidence, benchmark=benchmark)
    except EvidenceIncompleteError as error:
        return [str(error)]
    return []


def evidence_from_receipt(receipt: VerifierReceipt, benchmark: str = "") -> dict[str, dict[str, Any]]:
    """The correctness evidence a verifier receipt actually carries.

    For a per-shot verifier (DQI) the receipt's own error counts are the evidence, and
    the ancilla read is folded into ``value_errors`` by the verifier's own rule
    (``val_err = reg_diff.any(axis=0) | junk.any(axis=0)``), so junk is `covered_by`
    value with that reason. A cost-model verifier supplies no per-shot evidence, and the
    three checks come back `not_applicable` with that stated as the reason.
    """
    metrics = dict(receipt.metrics or {})
    family = _benchmark_family(benchmark)
    evidence: dict[str, dict[str, Any]] = {}
    if "value_errors" in metrics:
        evidence["value"] = {"measured": True,
                             "passed": metrics["value_errors"] == 0,
                             "source": "receipt.metrics['value_errors']"}
    if "phase_errors" in metrics:
        evidence["phase"] = {"measured": True,
                             "passed": metrics["phase_errors"] == 0,
                             "source": "receipt.metrics['phase_errors']"}
    if "junk" in metrics:
        evidence["junk"] = {"measured": True,
                            "passed": not metrics["junk"],
                            "source": "receipt.metrics['junk']"}
    elif "value_errors" in metrics:
        evidence["junk"] = {
            "covered_by": "value",
            "reason": "verify_submission_dqi folds the ancilla read into `value_errors` "
                      "(`val_err = reg_diff.any(axis=0) | junk.any(axis=0)`), so a zero "
                      "value count is also a clean workspace",
        }
    if not evidence and BENCHMARK_EVIDENCE_MODEL.get(family) == "cost-model-only":
        reason = _COST_MODEL_REASON.format(benchmark=family)
        evidence = {check: {"not_applicable": reason} for check in CORRECTNESS_CHECKS}
    return evidence


def sha256_file(path: Path, *, chunk_size: int = 1 << 20) -> str:
    """Return the SHA-256 hex digest of a candidate artifact."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def compute_spec_fingerprint(
    benchmark: str,
    contract_sha256: str,
    *,
    parameters: dict[str, Any] | None = None,
    domain_bounds: tuple[int, int] | None = None,
) -> str:
    """Compute the specification fingerprint enforcing functional problem equivalence.

    Any valid candidate must solve the exact preregistered problem contract.
    """
    payload = {
        "benchmark": benchmark,
        "contract_sha256": contract_sha256,
        "parameters": parameters or {},
        "domain_bounds": list(domain_bounds) if domain_bounds is not None else None,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def compute_structural_fingerprint(
    metrics_or_dag: dict[str, Any] | str,
) -> str:
    """Compute normalized structural DAG fingerprint to detect cloned or renamed circuits.

    Encodes normalized gate distributions, qubit layout, rotation count, and uncomputation schedule.
    """
    if isinstance(metrics_or_dag, str):
        payload = {"circuit_raw_repr": metrics_or_dag}
    else:
        # Normalize keys for canonical hashing
        payload = {
            "num_qubits": metrics_or_dag.get("num_qubits"),
            "toffoli_count": metrics_or_dag.get("toffoli_count") or metrics_or_dag.get("toffolis"),
            "rotations": metrics_or_dag.get("rotations") or metrics_or_dag.get("rotation_count"),
            "depth": metrics_or_dag.get("depth"),
            "gate_multiset": metrics_or_dag.get("gate_multiset") or metrics_or_dag.get("operations"),
            "wire_roles": metrics_or_dag.get("wire_roles"),
        }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def compute_provenance_fingerprint(
    generator_id: str,
    parameters: dict[str, Any],
    *,
    parent_id: str | None = None,
    verifier_sha256: str = "",
    timestamp: str = "",
) -> str:
    """Compute immutable provenance fingerprint documenting candidate creation lineage."""
    payload = {
        "generator_id": generator_id,
        "parameters": parameters,
        "parent_id": parent_id,
        "verifier_sha256": verifier_sha256,
        "timestamp": timestamp,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def candidate_id_for(
    benchmark: str,
    generator_id: str,
    parameters: dict[str, Any],
    *,
    contract_sha256: str,
) -> str:
    """Return a deterministic candidate id.

    The contract digest is part of the id so that a parameter point verified
    against one verifier version cannot silently inherit the status of the
    same point verified against another version.
    """
    payload = json.dumps(
        {
            "benchmark": benchmark,
            "generator_id": generator_id,
            "parameters": parameters,
            "contract_sha256": contract_sha256,
        },
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return f"{benchmark}-{generator_id}-{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:20]}"


@dataclasses.dataclass(frozen=True)
class VerifierReceipt:
    """A parsed official-verifier run.

    ``metrics`` is the full JSON payload.  The raw streams are truncated for the
    ledger but never discarded silently: the truncation length is recorded.
    """

    status: str
    score: float | None
    toffoli_equivalent: float | None
    num_qubits: int | None
    metrics: dict[str, Any]
    exit_code: int
    command: tuple[str, ...]
    salt_id: str
    verification_scope: str
    verifier_sha256: str
    artifact_sha256: str
    contract_sha256: str
    duration_s: float
    stdout_tail: str = ""
    stderr_tail: str = ""
    created_at: str = dataclasses.field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class CircuitCandidate:
    """A proposed circuit artifact plus its proposer-side evidence."""

    candidate_id: str
    benchmark: str
    generator_id: str
    parameters: dict[str, Any]
    artifact_path: str
    artifact_sha256: str
    contract_sha256: str
    parent_candidate_id: str | None = None
    status: CandidateStatus = CandidateStatus.PROPOSED
    screen_metrics: dict[str, Any] = dataclasses.field(default_factory=dict)
    screen_score_estimate: float | None = None
    verifier_receipt: VerifierReceipt | None = None
    #: Which of :data:`CORRECTNESS_CHECKS` the candidate has evidence for, and on what
    #: basis. A correctness-claiming status cannot be set without it (see
    #: :func:`normalise_correctness_evidence`).
    correctness_evidence: dict[str, Any] = dataclasses.field(default_factory=dict)
    spec_fingerprint: str = ""
    structural_fingerprint: str = ""
    provenance_fingerprint: str = ""
    created_at: str = dataclasses.field(default_factory=_now)

    def __post_init__(self) -> None:
        # Auto-derive fingerprints if not explicitly provided
        if not self.spec_fingerprint:
            object.__setattr__(
                self,
                "spec_fingerprint",
                compute_spec_fingerprint(
                    self.benchmark,
                    self.contract_sha256,
                    parameters=self.parameters,
                ),
            )
        if not self.structural_fingerprint:
            object.__setattr__(
                self,
                "structural_fingerprint",
                compute_structural_fingerprint(self.screen_metrics),
            )
        if not self.provenance_fingerprint:
            object.__setattr__(
                self,
                "provenance_fingerprint",
                compute_provenance_fingerprint(
                    self.generator_id,
                    self.parameters,
                    parent_id=self.parent_candidate_id,
                    timestamp=self.created_at,
                ),
            )

    def to_dict(self) -> dict[str, Any]:
        payload = dataclasses.asdict(self)
        payload["status"] = self.status.value
        if self.verifier_receipt is not None:
            payload["verifier_receipt"] = self.verifier_receipt.to_dict()
        return payload

    def with_status(
        self,
        status: CandidateStatus,
        *,
        receipt: VerifierReceipt | None = None,
        evidence: dict[str, Any] | None = None,
    ) -> CircuitCandidate:
        """Move the candidate to `status`, refusing a correctness claim with no evidence.

        The evidence is the candidate's own, plus whatever the receipt can establish,
        plus anything passed explicitly. A status in
        :data:`CORRECTNESS_CLAIMING_STATUSES` raises
        :class:`EvidenceIncompleteError` unless value, junk and phase are each
        measured-and-passed, covered by another passed check, or explicitly
        `not_applicable` with a reason the verifier supports.
        """
        merged = dict(self.correctness_evidence)
        if receipt is not None:
            for check, entry in evidence_from_receipt(receipt, self.benchmark).items():
                merged.setdefault(check, entry)
        if evidence is not None:
            merged.update(evidence)
        if status in CORRECTNESS_CLAIMING_STATUSES:
            merged = normalise_correctness_evidence(merged, benchmark=self.benchmark)
        return dataclasses.replace(
            self,
            status=status,
            verifier_receipt=receipt if receipt is not None else self.verifier_receipt,
            correctness_evidence=merged,
        )
