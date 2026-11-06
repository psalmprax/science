"""Append-only JSONL of referee verdicts.

The budget ledger records what the models cost; it says nothing about what the
referee decided and why. Without that second record, "which candidates were
refused, on which pinned thresholds, for what reason" lives only in a return
value that is printed once and lost — and a rejected candidate cannot be audited
after the fact.

Rows are written to their own file rather than mixed into the budget ledger, so
cost aggregation never has to filter foreign rows. Each row carries the kind,
the candidate identity, the preregistration and protocol digests it was judged
under, the reasons, and whatever measurement record the caller supplies.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

VERDICT_KIND = "phase0_verdict"
DEFAULT_VERDICT_LEDGER = Path("build/mmlm_matter_001/verdicts.jsonl")

#: Keys this ledger owns; a caller record may not overwrite them.
RESERVED_KEYS = frozenset({"kind", "sequence", "candidate_id", "recorded_utc"})


class VerdictLedgerError(RuntimeError):
    """A verdict row that cannot be written or read back."""


class VerdictLedger:
    """Append-only JSONL of verdicts, one row per adjudicated candidate."""

    def __init__(self, path: Path | str = DEFAULT_VERDICT_LEDGER) -> None:
        self.path = Path(path)

    def append(
        self,
        *,
        candidate_id: str,
        record: Mapping[str, Any],
        recorded_utc: str | None = None,
    ) -> dict[str, Any]:
        """Write one verdict row and return it as stored."""
        if not isinstance(candidate_id, str) or not candidate_id:
            raise VerdictLedgerError(
                f"candidate_id must be a non-empty string, got {candidate_id!r}")
        if not isinstance(record, Mapping):
            raise VerdictLedgerError(
                f"verdict record must be a mapping, got {type(record).__name__}")
        clash = RESERVED_KEYS & set(record)
        if clash:
            raise VerdictLedgerError(
                f"verdict record may not set reserved keys: {sorted(clash)}")
        row: dict[str, Any] = {
            "kind": VERDICT_KIND,
            "sequence": len(self.rows()) + 1,
            "candidate_id": candidate_id,
            "recorded_utc": recorded_utc
            or dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        }
        row.update(dict(record))
        try:
            line = json.dumps(row, sort_keys=True)
        except TypeError as exc:
            raise VerdictLedgerError(
                f"verdict for {candidate_id!r} is not JSON-serialisable: {exc}") from exc
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        return row

    def rows(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows: list[dict[str, Any]] = []
        for number, line in enumerate(
                self.path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise VerdictLedgerError(
                    f"verdict ledger {str(self.path)!r} line {number} is not JSON: "
                    f"{exc.msg}") from exc
            if not isinstance(row, dict):
                raise VerdictLedgerError(
                    f"verdict ledger {str(self.path)!r} line {number} is not an object")
            rows.append(row)
        return rows

    def verdicts(self) -> list[dict[str, Any]]:
        """Only the rows this ledger owns, skipping any foreign row."""
        return [row for row in self.rows() if row.get("kind") == VERDICT_KIND]

    def summary(self) -> dict[str, Any]:
        """Accepted/refused/unverifiable counts and the reasons, for an audit read-out.

        ``unverifiable`` is counted apart from ``refused`` on purpose: a candidate
        the checkout has no checker for is a capability gap, and folding it into
        the failure count would silently bias any pass-rate comparison that uses
        these rows as its denominator. ``eligible`` is the denominator to use.
        """
        verdicts = self.verdicts()
        reasons: dict[str, int] = {}
        for row in verdicts:
            for reason in row.get("reasons") or ():
                reasons[reason] = reasons.get(reason, 0) + 1
        statuses = [
            str(row.get("status") or ("accepted" if row.get("passed") is True else "refused"))
            for row in verdicts
        ]
        unverifiable = statuses.count("unverifiable")
        return {
            "verdicts": len(verdicts),
            "accepted": statuses.count("accepted"),
            "refused": statuses.count("refused"),
            "unverifiable": unverifiable,
            "eligible": len(statuses) - unverifiable,
            "reasons": dict(sorted(reasons.items())),
            "candidates": [row.get("candidate_id") for row in verdicts],
        }


__all__ = [
    "DEFAULT_VERDICT_LEDGER",
    "RESERVED_KEYS",
    "VERDICT_KIND",
    "VerdictLedger",
    "VerdictLedgerError",
]
