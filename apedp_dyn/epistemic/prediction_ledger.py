"""A ledger of dated predictions: what was claimed, before the data existed, and how it came out.

Status: experimental

**Why.** Every measured result of this repository so far is a benchmark with a
known answer. A law reported on real data has no answer key; what can be
checked is whether it predicts data that did not exist when it was stated.
This ledger records such a prediction before that data arrives and its
outcome after, so the system's record on real claims is a count anyone can
recompute, misses included.

**What.** An append-only JSON-lines file. A ``claim`` entry fixes, in advance,
the law, what it predicts, the data it will be scored on, the score, the
bound the score must meet, and the first date the outcome may be recorded. An
``outcome`` entry gives the observed score and the hash of the data used.
Every entry carries the SHA-256 of the entry before it, so removing or
editing an earlier entry breaks the chain (``verify``).

**Rules it enforces.** A claim cannot be changed or withdrawn. It gets one
outcome, not before its due date. Whether it held is computed here from the
bound stated in the claim, never supplied by the caller. A claim past its due
date with no outcome is reported as overdue, not dropped.

**What it does not do.** It does not prove when an entry was written; a
public commit of the file (or any outside timestamp) does that. It decides
nothing about a law: the referee and the confirmation step do.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

GENESIS = "0" * 64
DIRECTIONS = ("at_most", "at_least")


class LedgerError(ValueError):
    """An entry the ledger's rules refuse, or a chain that does not verify."""


def _digest(body: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class TrackRecord:
    claims: int
    held: int
    missed: int
    pending: int
    overdue: int


class PredictionLedger:
    """Dated predictions and their outcomes, hash-chained in one file."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def entries(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def verify(self) -> int:
        """Check the chain and the rules over the whole file; return the number of entries."""
        previous = GENESIS
        claims: dict[str, dict[str, Any]] = {}
        resolved: set[str] = set()
        for index, entry in enumerate(self.entries()):
            body = {k: v for k, v in entry.items() if k != "sha256"}
            if entry.get("previous") != previous or entry.get("sha256") != _digest(body):
                raise LedgerError(f"entry {index} does not continue the chain")
            previous = entry["sha256"]
            if entry["kind"] == "claim":
                if entry["claim_id"] in claims:
                    raise LedgerError(f"entry {index}: claim {entry['claim_id']} stated twice")
                claims[entry["claim_id"]] = entry
            else:
                claim = claims.get(entry["claim_id"])
                if claim is None or entry["claim_id"] in resolved:
                    raise LedgerError(f"entry {index}: outcome without a claim, or a second outcome")
                if entry["recorded_on"] < claim["due"] or entry["held"] != self._held(claim, entry["observed"]):
                    raise LedgerError(f"entry {index}: outcome before its due date, or a verdict the bound does not give")
                resolved.add(entry["claim_id"])
        return len(claims) + len(resolved)

    @staticmethod
    def _held(claim: dict[str, Any], observed: float) -> bool:
        return bool(observed <= claim["bound"] if claim["direction"] == "at_most" else observed >= claim["bound"])

    def _append(self, body: dict[str, Any]) -> dict[str, Any]:
        self.verify()
        existing = self.entries()
        body = {**body, "previous": existing[-1]["sha256"] if existing else GENESIS}
        entry = {**body, "sha256": _digest(body)}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
        return entry

    def claim(self, claim_id: str, *, law: str, predicts: str, data_source: str, score: str, direction: str,
              bound: float, stated_on: date, due: date, evidence: str = "") -> dict[str, Any]:
        """State a prediction. ``score`` names the number that will be computed; it holds if ``direction`` ``bound``."""
        if direction not in DIRECTIONS:
            raise LedgerError(f"direction must be one of {DIRECTIONS}")
        if due <= stated_on:
            raise LedgerError("a prediction is due after the day it is stated")
        if any(e["kind"] == "claim" and e["claim_id"] == claim_id for e in self.entries()):
            raise LedgerError(f"claim {claim_id} is already stated and cannot be restated")
        if not all(str(v).strip() for v in (claim_id, law, predicts, data_source, score)):
            raise LedgerError("a claim names its law, what it predicts, its data source and its score")
        return self._append({"kind": "claim", "claim_id": claim_id, "law": law, "predicts": predicts,
                             "data_source": data_source, "score": score, "direction": direction, "bound": float(bound),
                             "stated_on": stated_on.isoformat(), "due": due.isoformat(), "evidence": evidence})

    def resolve(self, claim_id: str, *, observed: float, data_sha256: str, recorded_on: date, note: str = "") -> dict[str, Any]:
        """Record the outcome of a claim. Whether it held follows from the claim's own bound."""
        entries = self.entries()
        claim = next((e for e in entries if e["kind"] == "claim" and e["claim_id"] == claim_id), None)
        if claim is None:
            raise LedgerError(f"no claim {claim_id}")
        if any(e["kind"] == "outcome" and e["claim_id"] == claim_id for e in entries):
            raise LedgerError(f"claim {claim_id} already has its outcome")
        if recorded_on.isoformat() < claim["due"]:
            raise LedgerError(f"claim {claim_id} is due {claim['due']}; an outcome cannot be recorded before")
        if len(data_sha256) != 64:
            raise LedgerError("an outcome names the SHA-256 of the data it was scored on")
        return self._append({"kind": "outcome", "claim_id": claim_id, "observed": float(observed),
                             "held": self._held(claim, float(observed)), "data_sha256": data_sha256,
                             "recorded_on": recorded_on.isoformat(), "note": note})

    def track_record(self, today: date) -> TrackRecord:
        """Held, missed, pending and overdue claims as of ``today``."""
        entries = self.entries()
        outcomes = {e["claim_id"]: e for e in entries if e["kind"] == "outcome"}
        claims = [e for e in entries if e["kind"] == "claim"]
        open_claims = [c for c in claims if c["claim_id"] not in outcomes]
        overdue = sum(c["due"] < today.isoformat() for c in open_claims)
        return TrackRecord(claims=len(claims), held=sum(o["held"] for o in outcomes.values()),
                           missed=sum(not o["held"] for o in outcomes.values()),
                           pending=len(open_claims) - overdue, overdue=overdue)
