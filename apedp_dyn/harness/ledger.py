"""Cryptographic Event Ledger and EEDR Repair Invariant Auditor for AEH.

Maintains an append-only, tamper-evident hash chain of all pipeline events.
Formally verifies the 8 EEDR Repair Invariants:
  1. H1 existed before the repair.
  2. H1's evidence was frozen (evidence conservation).
  3. The repair operator was authorized.
  4. H1* received a fresh candidate identity.
  5. The repair did not access prohibited evidence.
  6. H1* was independently re-evaluated de novo.
  7. H1's failure was retained permanently.
  8. Final promotion cannot overwrite the original failure record.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from apedp_dyn.harness.roles import ComponentRole


@dataclass(frozen=True)
class AuditEvent:
    """Tamper-evident record of a discrete action in the experiment."""
    event_id: str
    timestamp_utc: str
    role: ComponentRole
    action: str
    payload_summary: dict[str, Any]
    payload_hash: str
    prev_event_hash: str
    event_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp_utc": self.timestamp_utc,
            "role": self.role.value,
            "action": self.action,
            "payload_summary": self.payload_summary,
            "payload_hash": self.payload_hash,
            "prev_event_hash": self.prev_event_hash,
            "event_hash": self.event_hash,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AuditEvent:
        return cls(
            event_id=d["event_id"],
            timestamp_utc=d["timestamp_utc"],
            role=ComponentRole(d["role"]),
            action=d["action"],
            payload_summary=d["payload_summary"],
            payload_hash=d["payload_hash"],
            prev_event_hash=d["prev_event_hash"],
            event_hash=d["event_hash"],
        )

    @classmethod
    def create(
        cls,
        event_id: str,
        role: ComponentRole,
        action: str,
        payload_summary: dict[str, Any],
        prev_event_hash: str,
    ) -> AuditEvent:
        ts = datetime.now(timezone.utc).isoformat()
        payload_json = json.dumps(payload_summary, sort_keys=True)
        p_hash = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()

        event_str = f"{event_id}|{ts}|{role.value}|{action}|{p_hash}|{prev_event_hash}"
        e_hash = hashlib.sha256(event_str.encode("utf-8")).hexdigest()

        return cls(
            event_id=event_id,
            timestamp_utc=ts,
            role=role,
            action=action,
            payload_summary=payload_summary,
            payload_hash=p_hash,
            prev_event_hash=prev_event_hash,
            event_hash=e_hash,
        )


class EventLedger:
    """Append-only cryptographic ledger tracking all events across stages."""

    GENESIS_HASH = "0" * 64

    def __init__(self, journal_path: Path | str | None = None) -> None:
        self._events: list[AuditEvent] = []
        self._event_counter: int = 0
        self.journal_path: Path | None = Path(journal_path) if journal_path else None
        if self.journal_path:
            self.journal_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def events(self) -> list[AuditEvent]:
        return list(self._events)

    def record_event(
        self,
        role: ComponentRole,
        action: str,
        payload_summary: dict[str, Any],
    ) -> AuditEvent:
        """Appends a new event linked to the previous event's hash with optional WAL persistence."""
        self._event_counter += 1
        event_id = f"EVT_{self._event_counter:06d}"
        prev_hash = self._events[-1].event_hash if self._events else self.GENESIS_HASH

        event = AuditEvent.create(
            event_id=event_id,
            role=role,
            action=action,
            payload_summary=payload_summary,
            prev_event_hash=prev_hash,
        )
        self._events.append(event)

        # Write-Ahead Logging (WAL) to disk
        if self.journal_path:
            try:
                with open(self.journal_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(event.to_dict(), sort_keys=True) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
            except Exception:  # noqa: BLE001, S110 -- best-effort WAL append; a failed
                # journal write must not abort the in-memory event chain
                pass

        return event

    @classmethod
    def replay_from_journal(cls, journal_path: Path | str) -> EventLedger:
        """Reconstruct ledger from disk journal file and verify chain integrity."""
        path = Path(journal_path)
        ledger = cls(journal_path=path)
        if not path.exists():
            return ledger

        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    event_data = json.loads(line)
                    event = AuditEvent.from_dict(event_data)
                    ledger._events.append(event)
                    ledger._event_counter += 1

        if not ledger.verify_chain_integrity():
            raise RuntimeError(f"DISK JOURNAL CORRUPT: Hash chain verification failed for {journal_path}")
        return ledger

    def get_ledger_hash(self) -> str:
        """Returns the hash of the latest event or genesis hash if empty."""
        return self._events[-1].event_hash if self._events else self.GENESIS_HASH


    def verify_chain_integrity(self) -> bool:
        """Verifies mathematical unbrokenness of the SHA-256 hash chain."""
        if not self._events:
            return True

        expected_prev = self.GENESIS_HASH
        for e in self._events:
            if e.prev_event_hash != expected_prev:
                return False

            payload_json = json.dumps(e.payload_summary, sort_keys=True)
            p_hash = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
            if p_hash != e.payload_hash:
                return False

            event_str = f"{e.event_id}|{e.timestamp_utc}|{e.role.value}|{e.action}|{p_hash}|{e.prev_event_hash}"
            expected_hash = hashlib.sha256(event_str.encode("utf-8")).hexdigest()
            if expected_hash != e.event_hash:
                return False

            expected_prev = e.event_hash

        return True

    def audit_eedr_invariants(
        self,
        original_candidate_id: str,
        repaired_candidate_id: str | None,
        initial_evidence_hash: str,
        post_repair_evidence_hash: str,
        operator_applied: str | None,
        authorized_operators: tuple[str, ...],
    ) -> dict[str, bool]:
        """Formally verifies the 8 EEDR Invariants against the ledger history."""
        # 1. H1 existed before repair
        orig_logged = any(
            e.action in ("CANDIDATE_PROPOSED", "INITIAL_ADJUDICATION_FAILED")
            and e.payload_summary.get("candidate_id") == original_candidate_id
            for e in self._events
        )

        # 2. H1's evidence was frozen
        evidence_conserved = (initial_evidence_hash == post_repair_evidence_hash)

        # 3. Repair operator was authorized
        operator_valid = (
            operator_applied is None
            or operator_applied in authorized_operators
        )

        # 4. H1* received a new identity
        distinct_identity = (
            repaired_candidate_id is None
            or (repaired_candidate_id != original_candidate_id and original_candidate_id in repaired_candidate_id)
        )

        # 5. Repair did not access prohibited evidence
        repair_no_forbidden_access = True
        for e in self._events:
            if e.role == ComponentRole.REPAIRER:
                if e.payload_summary.get("accessed_answer_key", False):
                    repair_no_forbidden_access = False

        # 6. H1* was independently re-evaluated
        denovo_evaluated = (
            repaired_candidate_id is None
            or any(
                e.action == "DE_NOVO_ADJUDICATION_EXECUTED"
                and e.payload_summary.get("candidate_id") == repaired_candidate_id
                for e in self._events
            )
        )

        # 7. H1's failure was retained permanently
        original_failure_retained = any(
            e.action == "INITIAL_ADJUDICATION_FAILED"
            and e.payload_summary.get("candidate_id") == original_candidate_id
            for e in self._events
        )

        # 8. Final promotion cannot overwrite original failure
        failure_not_overwritten = True
        # Check that original candidate was never marked as PROMOTED_ORIGINAL
        for e in self._events:
            if (
                e.action == "CANDIDATE_PROMOTED"
                and e.payload_summary.get("candidate_id") == original_candidate_id
                and e.payload_summary.get("status") == "PROMOTED_ORIGINAL"
                and orig_logged
                and original_failure_retained
            ):
                failure_not_overwritten = False

        invariants = {
            "inv1_h1_existed_before_repair": orig_logged,
            "inv2_evidence_frozen_conserved": evidence_conserved,
            "inv3_repair_operator_authorized": operator_valid,
            "inv4_fresh_candidate_identity": distinct_identity,
            "inv5_no_prohibited_data_in_repair": repair_no_forbidden_access,
            "inv6_de_novo_re_evaluation": denovo_evaluated,
            "inv7_original_failure_retained": original_failure_retained,
            "inv8_failure_not_overwritten": failure_not_overwritten,
        }
        return invariants
