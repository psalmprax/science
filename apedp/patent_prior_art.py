"""Patent prior-art search over a bundled, provenance-checked corpus.

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
  ``expression`` for what one of its claims computes. Only those claims can be
  matched symbolically; the rest are counted and reported as not searchable rather
  than silently ignored, because "0 of 501 claims carried a formula" and "no claim
  matched" are different findings.
* **Operand-only claims cannot decide a verdict.** A claim whose normalised form
  names its operands and one operation (``x/y``, ``a*b``, a bare symbol) is
  isomorphic to any expression of that class, so a match is a coincidence of form
  class, not evidence that the patent claims the candidate. Such claims are
  **refused**: reported as ``CLAIM_STATES_OPERANDS_ONLY`` and never as known.
* **A verdict is corpus-relative.** ``NOT_FOUND_IN_PATENT_CORPUS`` says the corpus
  was searched and this form was not in it. It is not a novelty opinion, not a
  freedom-to-operate opinion, and not legal advice. The corpus is small, so every
  verdict also says that the index is thin.

Loading the bundled corpus::

    from apedp.patent_prior_art import BUNDLED_CORPUS_PATH, PatentPriorArtIndex

    index = PatentPriorArtIndex.from_jsonl(BUNDLED_CORPUS_PATH)
    result = index.audit("a * x / (1 + b * x)")

Loading a caller's own corpus::

    index = PatentPriorArtIndex.from_jsonl("data/patents/corpus.jsonl")

Each JSONL line is one record::

    {"patent_id": "US-1234567-B2", "jurisdiction": "US",
     "title": "...", "claim": "...", "expression": "a*x/(1+b*x)",
     "source": "https://...", "retrieved": "2026-10-29",
     "claims": [{"claim_number": "1", "claim": "...", "expression": "..."}]}

Status: production
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import sympy as sp

#: Statuses a patent audit can return.
KNOWN_IN_PATENT_CORPUS = "KNOWN_IN_PATENT_CORPUS"
RELATED_STRUCTURE_IN_PATENT_CORPUS = "RELATED_STRUCTURE_IN_PATENT_CORPUS"
NOT_FOUND_IN_PATENT_CORPUS = "NOT_FOUND_IN_PATENT_CORPUS"
NO_SEARCHABLE_CLAIMS = "NO_SEARCHABLE_CLAIMS"
CLAIM_STATES_OPERANDS_ONLY = "CLAIM_STATES_OPERANDS_ONLY"

#: The corpus that ships with the repository. ``None`` would mean "not bundled";
#: it is a path because the records and their raw artefacts are on disk.
BUNDLED_CORPUS_PATH: Path | None = (
    Path(__file__).resolve().parents[1] / "data" / "patents" / "corpus.jsonl"
)

#: Below this many decisive claims the index is declared thin in every verdict.
DECISIVE_CLAIMS_FLOOR = 50

_STATUS_DATA = Path(__file__).resolve().parents[1] / "data" / "patents"

_REFUSAL_SENTENCES: dict[str, str] = {
    "bare product": (
        "bare product — the claim names two operands and their product and says "
        "nothing about either, so any product is isomorphic to it"
    ),
    "bare quotient": (
        "bare quotient — the claim names two operands and their quotient and says "
        "nothing about either, so any quotient is isomorphic to it"
    ),
    "bare symbol (an equality of names)": (
        "bare symbol (an equality of names) — the claim equates a name with a name "
        "and states no operation, so any single symbol matches it"
    ),
}


class PatentProvenanceError(ValueError):
    """Raised when a patent record cannot be attributed to a real source."""


def _parse(expression: str) -> sp.Expr:
    """Parse a normalised claim expression, refusing anything sympy cannot read."""
    try:
        return sp.sympify(expression)
    except (sp.SympifyError, TypeError, ValueError, SyntaxError) as error:
        raise PatentProvenanceError(
            f"expression {expression!r} does not parse: {error}") from error


def operand_only_form_class(expression: str) -> str | None:
    """The operand-only class of ``expression``, or ``None`` when it has structure.

    A form is operand-only when it names symbols and at most one operation and no
    coefficient: ``x/y`` (bare quotient), ``a*b`` (bare product) or a lone symbol
    (an equality of names). A number anywhere in the expression — a coefficient, a
    constant term, an exponent other than -1 — is structure, so it is decisive.
    """
    parsed = _parse(expression)
    if parsed.is_Symbol:
        return "bare symbol (an equality of names)"

    def _is_reciprocal(node: sp.Expr) -> bool:
        return bool(node.is_Pow and node.exp == -1 and node.base.is_Symbol)

    if parsed.is_Mul:
        arguments = list(parsed.args)
        reciprocals = [argument for argument in arguments if _is_reciprocal(argument)]
        symbols = [argument for argument in arguments if argument.is_Symbol]
        if len(reciprocals) + len(symbols) != len(arguments):
            return None
        if not reciprocals and len(symbols) >= 2:
            return "bare product"
        if len(reciprocals) == 1:
            return "bare quotient"
    return None


def is_operand_only(expression: str) -> bool:
    """True when a normalised claim states operands and one operation, no form."""
    return operand_only_form_class(expression) is not None


@dataclass(frozen=True)
class PatentClaim:
    """One claim of a patent record, with its normalised form when it has one."""

    claim_number: str
    text: str
    expression: str | None = None
    expression_note: str = ""
    patent_id: str = ""

    @classmethod
    def from_dict(cls, raw: dict[str, Any], *, patent_id: str = "") -> PatentClaim:
        expression = raw.get("expression")
        return cls(
            claim_number=str(raw.get("claim_number", "")),
            text=str(raw.get("claim", raw.get("text", ""))),
            expression=None if expression in (None, "") else str(expression),
            expression_note=str(raw.get("expression_note", "")),
            patent_id=patent_id,
        )


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
    claims: tuple[PatentClaim, ...] = ()

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
            _parse(self.expression)
        for claim in self.claims:
            if claim.expression is not None:
                try:
                    _parse(claim.expression)
                except PatentProvenanceError as error:
                    raise PatentProvenanceError(
                        f"{self.patent_id} claim {claim.claim_number}: {error}") from error

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> PatentRecord:
        try:
            patent_id = str(raw["patent_id"])
        except KeyError as error:
            raise PatentProvenanceError(f"record missing required field {error}") from error
        expression = raw.get("expression")
        claim_text = str(raw.get("claim", ""))
        raw_claims = raw.get("claims")
        if raw_claims is None:
            claims: tuple[PatentClaim, ...] = ()
            if claim_text or expression not in (None, ""):
                claims = (PatentClaim(
                    claim_number=str(raw.get("claim_number", "1")),
                    text=claim_text,
                    expression=None if expression in (None, "") else str(expression),
                    expression_note=str(raw.get("expression_note", "")),
                    patent_id=patent_id,
                ),)
        else:
            claims = tuple(
                replace(PatentClaim.from_dict(entry, patent_id=patent_id),
                        patent_id=patent_id)
                for entry in raw_claims
            )
        record = cls(
            patent_id=patent_id,
            jurisdiction=str(raw.get("jurisdiction", "")),
            title=str(raw.get("title", "")),
            claim=claim_text,
            expression=None if expression in (None, "") else str(expression),
            source=str(raw.get("source", "")),
            retrieved=str(raw.get("retrieved", "")),
            synthetic=bool(raw.get("synthetic", False)),
            claims=claims,
        )
        record.validate()
        return record


@dataclass(frozen=True)
class PatentRefusal:
    """A normalised claim held apart because it states no functional form."""

    patent_id: str
    claim_number: str
    expression: str
    form_class: str
    why: str
    claim_states: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "patent_id": self.patent_id,
            "claim_number": self.claim_number,
            "expression": self.expression,
            "form_class": self.form_class,
            "why": self.why,
            "claim_states": self.claim_states,
        }


@dataclass(frozen=True)
class PatentAuditResult:
    """Corpus-relative outcome of a patent prior-art search."""

    candidate: str
    status: str
    closest_patent_id: str | None
    closest_claim: str | None
    matched_expression: str | None
    matched_claim_number: str | None
    isomorphism: str | None
    searched_patents: int
    searchable_claims: int
    decisive_claims: int
    operand_only_claims: int
    synthetic_records: int
    epistemic_disclaimer: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate,
            "status": self.status,
            "closest_patent_id": self.closest_patent_id,
            "closest_claim": self.closest_claim,
            "matched_expression": self.matched_expression,
            "matched_claim_number": self.matched_claim_number,
            "isomorphism": self.isomorphism,
            "searched_patents": self.searched_patents,
            "searchable_claims": self.searchable_claims,
            "decisive_claims": self.decisive_claims,
            "operand_only_claims": self.operand_only_claims,
            "synthetic_records": self.synthetic_records,
            "epistemic_disclaimer": self.epistemic_disclaimer,
        }


def _shape_node(node: sp.Expr) -> Any:
    """A canonical operation tree with every number and symbol abstracted.

    Coefficients are structure in the *verdict* sense (a scaled quotient is not a
    bare quotient), but for the "related form" tier they are parameters: two linear
    laws with different slopes share an operation skeleton. Symbols are abstracted
    too, because a patent's notation is its draughtsman's, not a claim about which
    quantity is which.
    """
    if node.is_Number:
        return ("number",)
    if node.is_Symbol:
        return ("symbol",)
    if isinstance(node, sp.Add):
        return ("add", tuple(sorted((_shape_node(a) for a in node.args), key=repr)))
    if isinstance(node, sp.Mul):
        return ("mul", tuple(sorted((_shape_node(a) for a in node.args), key=repr)))
    if isinstance(node, sp.Pow):
        return ("pow", _shape_node(node.base), _shape_node(node.exp))
    if isinstance(node, sp.Function):
        return ("function", type(node).__name__,
                tuple(_shape_node(a) for a in node.args))
    return ("other", type(node).__name__)


def _operation_shape(expression: str) -> Any:
    """The operation tree of ``expression``, or ``None`` when it cannot be parsed."""
    try:
        from apedp.epistemic import PriorArtAuditor

        return _shape_node(PriorArtAuditor._parse_physics(expression))
    except (sp.SympifyError, TypeError, ValueError, AttributeError, ArithmeticError):
        return None


def _bundled_rows() -> list[dict[str, Any]]:
    if BUNDLED_CORPUS_PATH is None or not BUNDLED_CORPUS_PATH.exists():
        raise FileNotFoundError(
            f"patent corpus not found: {BUNDLED_CORPUS_PATH}; no corpus is bundled")
    return [json.loads(line) for line in
            BUNDLED_CORPUS_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def load_patent_corpus(path: str | Path) -> list[PatentRecord]:
    """Read a JSONL patent corpus, validating every record and refusing duplicates."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(
            f"patent corpus not found: {source}; the corpus bundled at "
            f"{BUNDLED_CORPUS_PATH} is the one this module ships")
    records: list[PatentRecord] = []
    seen: set[str] = set()
    for line_number, line in enumerate(source.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as error:
            raise PatentProvenanceError(f"{source}:{line_number}: not JSON: {error}") from error
        record = PatentRecord.from_dict(raw)
        if record.patent_id in seen:
            raise PatentProvenanceError(
                f"{source}:{line_number}: duplicate patent_id {record.patent_id}")
        seen.add(record.patent_id)
        records.append(record)
    return records


def operand_only_refusals(
    records: Iterable[PatentRecord] | None = None,
) -> list[PatentRefusal]:
    """Every normalised claim held apart as operand-only, in record/claim order."""
    if records is None:
        records = load_patent_corpus(BUNDLED_CORPUS_PATH)
    refusals: list[PatentRefusal] = []
    for record in records:
        for claim in record.claims:
            if not claim.expression:
                continue
            form_class = operand_only_form_class(claim.expression)
            if form_class is None:
                continue
            refusals.append(PatentRefusal(
                patent_id=record.patent_id,
                claim_number=claim.claim_number,
                expression=claim.expression,
                form_class=form_class,
                why=_REFUSAL_SENTENCES[form_class],
                claim_states=claim.expression_note,
            ))
    return sorted(
        refusals,
        key=lambda refusal: (
            refusal.patent_id,
            int(refusal.claim_number) if refusal.claim_number.isdigit() else 0,
        ),
    )


def refusals_markdown(records: Iterable[PatentRecord] | None = None) -> str:
    """The refusal listing as the markdown document kept beside the corpus."""
    if records is None:
        records = load_patent_corpus(BUNDLED_CORPUS_PATH)
    records = list(records)
    refusals = operand_only_refusals(records)
    searchable = sum(1 for record in records for claim in record.claims if claim.expression)
    by_record: dict[str, list[PatentRefusal]] = {}
    for refusal in refusals:
        by_record.setdefault(refusal.patent_id, []).append(refusal)
    claim_totals = {
        record.patent_id: sum(1 for claim in record.claims if claim.expression)
        for record in records
    }

    lines = [
        "# Claims refused as operand-only",
        "",
        "`is_operand_only()` holds a claim apart when its normalised form names its",
        "operands and one operation and nothing about them. Any expression built the same",
        "way is isomorphic to such a form, so a match is a coincidence of form class rather",
        "than evidence that the patent claims the candidate: the engine returns",
        "`CLAIM_STATES_OPERANDS_ONLY` and never `KNOWN_IN_PATENT_CORPUS`.",
        "",
        f"{len(refusals)} of {searchable} normalised claims are refused this way, "
        f"across {len(by_record)} of the {len(records)} records; every other normalised "
        "claim is decisive and can carry a verdict.",
    ]
    for patent_id, entries in by_record.items():
        normalised = claim_totals[patent_id]
        lines.extend([
            "",
            f"## {patent_id}",
            "",
            f"{len(entries)} of {normalised} normalised claims refused; "
            f"{normalised - len(entries)} decisive.",
            "",
            "| claim | normalised form | refused because | what the claim states |",
            "|---|---|---|---|",
        ])
        for entry in entries:
            lines.append(
                f"| {entry.claim_number} | `{entry.expression}` | {entry.why} | "
                f"{entry.claim_states} |")
    return "\n".join(lines) + "\n"


def corpus_status() -> dict[str, Any]:
    """What the bundled corpus is, how much of it is searchable, and its limits.

    A boolean "bundled" was not enough: a record count says whether a verdict
    searched a corpus or a corner of one, and the decisive/operand-only split says
    how much of the searchable set could carry a conclusion.
    """
    records = load_patent_corpus(BUNDLED_CORPUS_PATH) if BUNDLED_CORPUS_PATH else []
    index = PatentPriorArtIndex(records)
    refusals = operand_only_refusals(records)
    claims_scanned = 0
    formulas_found = 0
    yield_path = _STATUS_DATA / "extraction_yield.json"
    if yield_path.exists():
        measured = json.loads(yield_path.read_text(encoding="utf-8"))
        claims_scanned = int(measured.get("claims_scanned", 0))
        formulas_found = int(measured.get("claims_with_a_parsed_formula", 0))
    max_claims = max((len([claim for claim in record.claims if claim.expression])
                      for record in records), default=0)
    sha256 = (hashlib.sha256(BUNDLED_CORPUS_PATH.read_bytes()).hexdigest()
              if BUNDLED_CORPUS_PATH and BUNDLED_CORPUS_PATH.exists() else None)
    return {
        "bundled": bool(BUNDLED_CORPUS_PATH and BUNDLED_CORPUS_PATH.exists()),
        "corpus_path": (str(BUNDLED_CORPUS_PATH) if BUNDLED_CORPUS_PATH else None),
        "corpus_sha256": sha256,
        "requires_caller_supplied_corpus": False,
        "records": len(index.records),
        "searchable_claims": len(index.searchable),
        "decisive_claims": index.decisive_claims,
        "operand_only_claims": index.operand_only_claims,
        "operand_only": [refusal.to_dict() for refusal in refusals],
        "records_with_formula": index.records_with_formula,
        "max_claims_per_record": max_claims,
        "unsearchable_records": len(index.records) - index.records_with_formula,
        "synthetic_records": index.synthetic_records,
        "decisive_claims_floor": DECISIVE_CLAIMS_FLOOR,
        "meets_decisive_floor": index.decisive_claims >= DECISIVE_CLAIMS_FLOOR,
        "thin": index.decisive_claims < DECISIVE_CLAIMS_FLOOR,
        "claims_scanned_for_formulas": claims_scanned,
        "formulas_found_in_claim_text": formulas_found,
        "reason": (
            f"The bundled corpus holds {len(index.records)} records with one to four "
            f"per record normalised claims across {index.records_with_formula} of them; "
            f"the other {len(index.records) - index.records_with_formula} state no "
            "relation in any claim and are counted as unsearchable, not as searched "
            f"and absent. {index.decisive_claims} of the {len(index.searchable)} "
            "normalised claims are decisive; the rest name their operands and one "
            "operation and are refused as operand-only."
        ),
    }


class PatentPriorArtIndex:
    """Symbolic search of normalised patent claims, with provenance accounting."""

    def __init__(self, records: Iterable[PatentRecord]):
        self.records: tuple[PatentRecord, ...] = tuple(records)
        for record in self.records:
            record.validate()
        self.searchable: tuple[PatentClaim, ...] = tuple(
            claim for record in self.records for claim in record.claims
            if claim.expression
        )
        self.records_with_formula = sum(
            1 for record in self.records
            if any(claim.expression for claim in record.claims)
        )
        self.synthetic_records = sum(1 for record in self.records if record.synthetic)
        self.operand_only_claims = sum(
            1 for claim in self.searchable if is_operand_only(claim.expression or ""))
        self.decisive_claims = len(self.searchable) - self.operand_only_claims
        self._decisive = tuple(
            claim for claim in self.searchable
            if not is_operand_only(claim.expression or ""))

    @classmethod
    def from_jsonl(cls, path: str | Path) -> PatentPriorArtIndex:
        return cls(load_patent_corpus(path))

    @classmethod
    def bundled(cls) -> PatentPriorArtIndex:
        """The corpus that ships with the repository."""
        return cls(load_patent_corpus(BUNDLED_CORPUS_PATH))

    def __len__(self) -> int:
        return len(self.records)

    @property
    def is_empty(self) -> bool:
        return not self.records

    def _disclaimer(self) -> str:
        base = (
            f"Searched {len(self.records)} patent records "
            f"({len(self.searchable)} with a normalised claim formula, "
            f"{self.synthetic_records} synthetic). A patent corpus verdict is "
            "corpus-relative, is not a novelty opinion, and is not legal advice; "
            "a freedom-to-operate search requires a professional patent search."
        )
        thin = (
            f" The index is thin: {self.decisive_claims} decisive claims is below "
            f"the declared floor of {DECISIVE_CLAIMS_FLOOR}, so absence here is "
            "absence from a thin index."
        )
        relation = (
            f" {self.records_with_formula} records with a stated relation, "
            f"{self.operand_only_claims} of their {len(self.searchable)} normalised "
            "claims held apart as operand-only."
        )
        return base + thin + relation

    def _result(self, candidate: str, status: str, *, claim: PatentClaim | None = None,
                isomorphism: str | None = None, extra: str = "") -> PatentAuditResult:
        return PatentAuditResult(
            candidate=candidate,
            status=status,
            closest_patent_id=claim.patent_id if claim else None,
            closest_claim=claim.text if claim else None,
            matched_expression=claim.expression if claim else None,
            matched_claim_number=claim.claim_number if claim else None,
            isomorphism=isomorphism,
            searched_patents=len(self.records),
            searchable_claims=len(self.searchable),
            decisive_claims=self.decisive_claims,
            operand_only_claims=self.operand_only_claims,
            synthetic_records=self.synthetic_records,
            epistemic_disclaimer=self._disclaimer() + extra,
        )

    def audit(self, candidate_expr: str | sp.Expr) -> PatentAuditResult:
        """Search the corpus for the candidate's algebraic form.

        Exact or scalar-multiple equivalence is reported first; otherwise a shared
        operation skeleton is reported as related. A candidate that is itself
        operand-only can only ever match a claim of the same operand-only class,
        and that match is refused rather than reported as known. None of it is a
        legal conclusion.
        """
        candidate = str(candidate_expr)
        if not self.searchable:
            return self._result(
                candidate, NO_SEARCHABLE_CLAIMS,
                extra=(" No record carried a normalised claim formula, so no "
                       "symbolic comparison was possible."),
            )

        candidate_class = operand_only_form_class(candidate)
        if candidate_class is not None:
            for claim in self.searchable:
                if (claim.expression
                        and operand_only_form_class(claim.expression) == candidate_class):
                    return self._result(
                        candidate, CLAIM_STATES_OPERANDS_ONLY, claim=claim,
                        extra=(
                            " Both the candidate and this claim name their operands "
                            "and one operation and nothing about them, so the match is "
                            "a coincidence of form class and not evidence that the "
                            "patent claims the candidate: it is refused rather than "
                            "reported as known."),
                    )

        from apedp.epistemic import PriorArtAuditor

        # Rename-insensitive first: a patent's symbol names are its draughtsman's
        # notation, not a claim about which quantity is which, so `a*x/(1+b*x)` and
        # `alpha*T/(1+beta*T)` are the same claimed form.
        for claim in self._decisive:
            equivalent, message = PriorArtAuditor.check_rename_isomorphism(
                candidate, claim.expression)
            if equivalent:
                return self._result(
                    candidate, KNOWN_IN_PATENT_CORPUS, claim=claim,
                    isomorphism=message,
                    extra=(
                        f" The candidate is algebraically isomorphic to the formula in "
                        f"claim {claim.claim_number} of {claim.patent_id}."),
                )

        shape = _operation_shape(candidate)
        if shape is not None:
            for claim in self._decisive:
                if claim.expression and _operation_shape(claim.expression) == shape:
                    return self._result(
                        candidate, RELATED_STRUCTURE_IN_PATENT_CORPUS, claim=claim,
                        isomorphism=f"STRUCTURAL_SIGNATURE_MATCH: {shape}",
                        extra=(
                            f" The candidate shares an operation skeleton with claim "
                            f"{claim.claim_number} of {claim.patent_id} but is not "
                            "algebraically equivalent to it, so it is related "
                            "structure rather than a known claim."),
                    )

        return self._result(candidate, NOT_FOUND_IN_PATENT_CORPUS)


def audit_against_patents(
    candidate_expr: str | sp.Expr,
    records: Sequence[PatentRecord],
) -> PatentAuditResult:
    """Convenience wrapper: audit one candidate against an in-memory record list."""
    return PatentPriorArtIndex(records).audit(candidate_expr)


__all__ = [
    "BUNDLED_CORPUS_PATH",
    "CLAIM_STATES_OPERANDS_ONLY",
    "DECISIVE_CLAIMS_FLOOR",
    "KNOWN_IN_PATENT_CORPUS",
    "NOT_FOUND_IN_PATENT_CORPUS",
    "NO_SEARCHABLE_CLAIMS",
    "RELATED_STRUCTURE_IN_PATENT_CORPUS",
    "PatentAuditResult",
    "PatentClaim",
    "PatentPriorArtIndex",
    "PatentProvenanceError",
    "PatentRecord",
    "PatentRefusal",
    "audit_against_patents",
    "corpus_status",
    "is_operand_only",
    "load_patent_corpus",
    "operand_only_form_class",
    "operand_only_refusals",
    "refusals_markdown",
]
