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
    "CandidateStatus",
    "CircuitCandidate",
    "VerifierReceipt",
    "candidate_id_for",
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
    VERIFIED_NONIMPROVING = "VERIFIED_NONIMPROVING"
    VERIFIED_IMPROVEMENT = "VERIFIED_IMPROVEMENT"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    UNRESOLVED = "UNRESOLVED"


def sha256_file(path: Path, *, chunk_size: int = 1 << 20) -> str:
    """Return the SHA-256 hex digest of a candidate artifact."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


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
    created_at: str = dataclasses.field(default_factory=_now)

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
    ) -> CircuitCandidate:
        return dataclasses.replace(
            self,
            status=status,
            verifier_receipt=receipt if receipt is not None else self.verifier_receipt,
        )
