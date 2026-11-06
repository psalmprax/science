"""Patent prior-art search over a local, provenance-checked corpus.

:class:`apedp.epistemic.PriorArtAuditor` searches 193 textbook laws. That corpus
answers "is this a known physical law", not "is this claimed in a patent", and the
two are different questions: patents claim applications, apparatus and methods, and
the same algebraic form can appear in one under a completely different vocabulary.

This module adds the missing source without weakening the integrity rules the rest
of the platform enforces:

* **Provenance is mandatory.** A record is either a real patent with a source URL
  and a retrieval date, or it is explicitly marked ``synthetic``. A record that
  claims to be a real patent without attribution is refused at load time, so a
  hand-written formula cannot enter the corpus wearing a patent number.
* **Claims are not formulas.** A patent record may carry a normalised
  ``expression`` for what its claim computes. Only those records can be matched
  symbolically; the rest are counted and reported as not searchable rather than
  silently ignored, because "0 of 412 claims carried a formula" and "no claim
  matched" are different findings.
* **A verdict is corpus-relative.** ``NOT_FOUND_IN_PATENT_CORPUS`` says the corpus
  was searched and this form was not in it. It is not a novelty opinion, not a
  freedom-to-operate opinion, and not legal advice.

Loading a corpus::

    from apedp.patent_prior_art import PatentPriorArtIndex

    index = PatentPriorArtIndex.from_jsonl("data/patents/corpus.jsonl")
    result = index.audit("a * x / (1 + b * x)")

Each JSONL line is one record::

    {"patent_id": "US-1234567-B2", "jurisdiction": "US",
     "title": "...", "claim": "...", "expression": "a*x/(1+b*x)",
     "source": "https://...", "retrieved": "2026-10-29"}

Candidate public sources for building a corpus (none is bundled here, and no
access terms are asserted): USPTO PatentsView API, USPTO Bulk Data, EPO Open
Patent Services (OPS), Google Patents public datasets. Whichever is used, record
the URL and retrieval date per patent: the loader requires them.
"""
from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import sympy as sp

#: Statuses a patent audit can return.
KNOWN_IN_PATENT_CORPUS = "KNOWN_IN_PATENT_CORPUS"
RELATED_STRUCTURE_IN_PATENT_CORPUS = "RELATED_STRUCTURE_IN_PATENT_CORPUS"
NOT_FOUND_IN_PATENT_CORPUS = "NOT_FOUND_IN_PATENT_CORPUS"
NO_SEARCHABLE_CLAIMS = "NO_SEARCHABLE_CLAIMS"


class PatentProvenanceError(ValueError):
    """Raised when a patent record cannot be attributed to a real source."""


@dataclass(frozen=True)
class PatentRecord:
    """One patent document, or one explicitly-labelled synthetic fixture."""

    patent_id: str
    jurisdiction: str
    title: str
    claim: str
    expression: str | None
    source: str
    retrieved: str
    synthetic: bool = False

    def validate(self) -> None:
        """Refuse unattributed records; a patent number is an attribution claim."""
        if not self.patent_id.strip():
            raise PatentProvenanceError("patent_id is empty")
        if not self.synthetic:
            if not self.source.startswith(("http://", "https://")):
                raise PatentProvenanceError(
                    f"{self.patent_id}: a non-synthetic record needs an http(s) source URL, "
                    f"got {self.source!r}")
            if not self.retrieved.strip():
                raise PatentProvenanceError(
                    f"{self.patent_id}: a non-synthetic record needs a retrieval date")
        if self.expression is not None:
            try:
                sp.sympify(self.expression)
            except (sp.SympifyError, TypeError, ValueError) as e:
                raise PatentProvenanceError(
                    f"{self.patent_id}: expression {self.expression!r} does not parse: {e}") from e

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> PatentRecord:
        try:
            record = cls(
                patent_id=str(raw["patent_id"]),
                jurisdiction=str(raw.get("jurisdiction", "")),
                title=str(raw.get("title", "")),
                claim=str(raw.get("claim", "")),
                expression=(None if raw.get("expression") in (None, "")
                            else str(raw["expression"])),
                source=str(raw.get("source", "")),
                retrieved=str(raw.get("retrieved", "")),
                synthetic=bool(raw.get("synthetic", False)),
            )
        except KeyError as e:
            raise PatentProvenanceError(f"record missing required field {e}") from e
        record.validate()
        return record


def load_patent_corpus(path: str | Path) -> list[PatentRecord]:
    """Read a JSONL patent corpus, validating every record and refusing duplicates."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"patent corpus not found: {source}")
    records: list[PatentRecord] = []
    seen: set[str] = set()
    for line_number, line in enumerate(source.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as e:
            raise PatentProvenanceError(f"{source}:{line_number}: not JSON: {e}") from e
        record = PatentRecord.from_dict(raw)
        if record.patent_id in seen:
            raise PatentProvenanceError(
                f"{source}:{line_number}: duplicate patent_id {record.patent_id}")
        seen.add(record.patent_id)
        records.append(record)
    return records


@dataclass(frozen=True)
class PatentAuditResult:
    """Corpus-relative outcome of a patent prior-art search."""

    candidate: str
    status: str
    closest_patent_id: str | None
    closest_claim: str | None
    matched_expression: str | None
    isomorphism: str | None
    searched_patents: int
    searchable_claims: int
    synthetic_records: int
    epistemic_disclaimer: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate,
            "status": self.status,
            "closest_patent_id": self.closest_patent_id,
            "closest_claim": self.closest_claim,
            "matched_expression": self.matched_expression,
            "isomorphism": self.isomorphism,
            "searched_patents": self.searched_patents,
            "searchable_claims": self.searchable_claims,
            "synthetic_records": self.synthetic_records,
            "epistemic_disclaimer": self.epistemic_disclaimer,
        }


class PatentPriorArtIndex:
    """Symbolic search of normalised patent claims, with provenance accounting."""

    def __init__(self, records: Iterable[PatentRecord]):
        self.records: tuple[PatentRecord, ...] = tuple(records)
        for record in self.records:
            record.validate()
        with_expression = [r for r in self.records if r.expression is not None]
        self.searchable: tuple[PatentRecord, ...] = tuple(with_expression)
        self.synthetic_records = sum(1 for r in self.records if r.synthetic)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> PatentPriorArtIndex:
        return cls(load_patent_corpus(path))

    def __len__(self) -> int:
        return len(self.records)

    @property
    def is_empty(self) -> bool:
        return not self.records

    def audit(self, candidate_expr: str | sp.Expr) -> PatentAuditResult:
        """Search the corpus for the candidate's algebraic form.

        Exact or scalar-multiple equivalence is reported first; otherwise a shared
        structural signature (function names, polynomial degree, symbol names) is
        reported as related. Neither is a legal conclusion.
        """
        candidate = str(candidate_expr)
        base_disclaimer = (
            f"Searched {len(self.records)} patent records "
            f"({len(self.searchable)} with a normalised claim formula, "
            f"{self.synthetic_records} synthetic). A patent corpus verdict is "
            "corpus-relative, is not a novelty opinion, and is not legal advice; "
            "a freedom-to-operate search requires a professional patent search.")

        if not self.searchable:
            return PatentAuditResult(
                candidate=candidate,
                status=NO_SEARCHABLE_CLAIMS,
                closest_patent_id=None,
                closest_claim=None,
                matched_expression=None,
                isomorphism=None,
                searched_patents=len(self.records),
                searchable_claims=0,
                synthetic_records=self.synthetic_records,
                epistemic_disclaimer=(
                    f"{base_disclaimer} No record carried a normalised claim formula, so no "
                    "symbolic comparison was possible."),
            )

        from apedp.epistemic import PriorArtAuditor

        # Rename-insensitive first: a patent's symbol names are its draughtsman's
        # notation, not a claim about which quantity is which, so `a*x/(1+b*x)` and
        # `alpha*T/(1+beta*T)` are the same claimed form.
        for record in self.searchable:
            equivalent, message = PriorArtAuditor.check_rename_isomorphism(
                candidate, record.expression)
            if equivalent:
                return PatentAuditResult(
                    candidate=candidate,
                    status=KNOWN_IN_PATENT_CORPUS,
                    closest_patent_id=record.patent_id,
                    closest_claim=record.claim,
                    matched_expression=record.expression,
                    isomorphism=message,
                    searched_patents=len(self.records),
                    searchable_claims=len(self.searchable),
                    synthetic_records=self.synthetic_records,
                    epistemic_disclaimer=(
                        f"{base_disclaimer} The candidate is algebraically isomorphic to the "
                        f"formula in claim-bearing record {record.patent_id}."),
                )

        related: tuple[PatentRecord, str] | None = None

        def _signature(expression: str) -> tuple[tuple[str, ...], int, tuple[str, ...]] | None:
            try:
                parsed = sp.sympify(expression)
                return PriorArtAuditor.pi_signature(
                    PriorArtAuditor._canonical_rename(parsed))
            except (sp.SympifyError, TypeError, ValueError, AttributeError):
                return None

        candidate_signature = _signature(candidate_expr) if isinstance(candidate_expr, str) \
            else _signature(str(candidate_expr))
        if candidate_signature is not None:
            for record in self.searchable:
                signature = _signature(str(record.expression))
                if signature is not None and signature == candidate_signature:
                    related = (record, f"STRUCTURAL_SIGNATURE_MATCH: {signature}")
                    break

        if related is not None:
            record, message = related
            return PatentAuditResult(
                candidate=candidate,
                status=RELATED_STRUCTURE_IN_PATENT_CORPUS,
                closest_patent_id=record.patent_id,
                closest_claim=record.claim,
                matched_expression=record.expression,
                isomorphism=message,
                searched_patents=len(self.records),
                searchable_claims=len(self.searchable),
                synthetic_records=self.synthetic_records,
                epistemic_disclaimer=(
                    f"{base_disclaimer} The candidate shares a structural signature with "
                    f"record {record.patent_id} but is not algebraically equivalent to it."),
            )

        return PatentAuditResult(
            candidate=candidate,
            status=NOT_FOUND_IN_PATENT_CORPUS,
            closest_patent_id=None,
            closest_claim=None,
            matched_expression=None,
            isomorphism=None,
            searched_patents=len(self.records),
            searchable_claims=len(self.searchable),
            synthetic_records=self.synthetic_records,
            epistemic_disclaimer=base_disclaimer,
        )


def audit_against_patents(
    candidate_expr: str | sp.Expr,
    records: Sequence[PatentRecord],
) -> PatentAuditResult:
    """Convenience wrapper: audit one candidate against an in-memory record list."""
    return PatentPriorArtIndex(records).audit(candidate_expr)


__all__ = [
    "KNOWN_IN_PATENT_CORPUS",
    "NOT_FOUND_IN_PATENT_CORPUS",
    "NO_SEARCHABLE_CLAIMS",
    "RELATED_STRUCTURE_IN_PATENT_CORPUS",
    "PatentAuditResult",
    "PatentPriorArtIndex",
    "PatentProvenanceError",
    "PatentRecord",
    "audit_against_patents",
    "load_patent_corpus",
]
