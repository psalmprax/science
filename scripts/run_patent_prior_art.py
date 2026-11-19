"""Run the patent prior-art engine against the bundled real corpus.

The corpus is small and real: twenty-nine publications fetched from Google Patents,
each with a source URL, a retrieval date and the SHA-256 of the bytes it came from
(``data/patents/raw/``). Seven of them carry fifteen normalised claims between them
(one to four per record) and twenty-two state no relation in any claim, so the run
exercises both verdict shapes, and a verdict is not bounded by one claim per patent.

Every candidate below is declared with its expectation, and the script **fails**
when the engine disagrees with the declaration. An expectation is a claim about
the corpus, and the declarations are what make a change in matching visible
instead of quietly plausible.

Two candidates exist to expose limits rather than to demonstrate reach:

* ``rename_equivalent`` shows that symbol names are not evidence — the same claimed
  form in different notation matches, which is correct for a form-level index and
  says nothing about infringement.
* ``generic_ratio`` shows an operands-and-one-operation claim. ``US8965721B2``
  claim 1 names a ratio of two impedances without their functional form, so its
  expression is ``Z_meas/Z_ref`` and *any* quotient is isomorphic to it. The engine
  now refuses that conclusion — the expected status is
  ``CLAIM_STATES_OPERANDS_ONLY``, and ``self_check_efficiency`` (``eta_C*i_ch``) is a
  bare product refused for the same reason.

Usage::

    python3 scripts/run_patent_prior_art.py
    python3 scripts/run_patent_prior_art.py --corpus data/patents/corpus.jsonl

Writes ``benchmarks/patent_prior_art/prior_art_run.json``.

Status: audit-only (records a measurement; concludes nothing legal)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apedp.patent_prior_art import (  # noqa: E402
    CLAIM_STATES_OPERANDS_ONLY,
    KNOWN_IN_PATENT_CORPUS,
    NOT_FOUND_IN_PATENT_CORPUS,
    RELATED_STRUCTURE_IN_PATENT_CORPUS,
    PatentPriorArtIndex,
    PatentRecord,
    is_operand_only,
    operand_only_refusals,
    refusals_markdown,
)

OUT = ROOT / "benchmarks" / "patent_prior_art" / "prior_art_run.json"
REFUSALS = ROOT / "data" / "patents" / "REFUSED_CLAIMS.md"
CORPUS = ROOT / "data" / "patents" / "corpus.jsonl"

#: (identifier, candidate expression, expected status, why this candidate is here)
CANDIDATES: tuple[tuple[str, str, str, str], ...] = (
    ("self_check_exponential", "AF*exp(t_A)", KNOWN_IN_PATENT_CORPUS,
     "verbatim form of US7120566B1 claim 1; the index must find its own record"),
    ("rename_equivalent", "k*exp(tau)", KNOWN_IN_PATENT_CORPUS,
     "the same claimed form in different notation, to show that symbol names "
     "are not evidence of anything"),
    ("self_check_efficiency", "eta_C*i_ch", CLAIM_STATES_OPERANDS_ONLY,
     "REFUSAL PROBE: a bare product of two symbols. The claim states the operands "
     "and one operation and nothing about them, so any product matches it and the "
     "engine refuses to report that as known"),
    ("saturating_response", "a*x/(1+b*x)", NOT_FOUND_IN_PATENT_CORPUS,
     "a Langmuir-type saturation, common in sensor claims, absent from these "
     "five normalised claims"),
    ("structured_not_swallowed", "2*Z_meas/Z_ref", NOT_FOUND_IN_PATENT_CORPUS,
     "ORDERING PROBE: a coefficient is structure, so the bare-quotient claim "
     "Z_meas/Z_ref must not swallow this candidate. Before the tier guard the "
     "scalar-multiple isomorphism made this row return CLAIM_STATES_OPERANDS_ONLY, "
     "which reads as coverage the claim does not have. It is also the case that made "
     "the structural tier stop comparing degree and symbol count: a scaled quotient "
     "and `1 - SS_res/SS_tot` agree on both and share no functional form, so they now "
     "read NOT_FOUND"),
    ("affine_slope_differs", "0.5*b_SEI*SOH + b_SEI", RELATED_STRUCTURE_IN_PATENT_CORPUS,
     "STRUCTURAL TIER PROBE: the same functional form as US11474159B2 claim 6 "
     "(`-0.01*b_SEI*SOH + b_SEI`) with a different slope. They share an operation "
     "skeleton — linear in SOH plus a symbol intercept — and are not algebraically "
     "equivalent, which is exactly what RELATED_STRUCTURE means now that the tier "
     "compares the operation tree and not degree and symbol count"),
    ("generic_ratio", "x/y", CLAIM_STATES_OPERANDS_ONLY,
     "REFUSAL PROBE: four normalised claims are quotients (EP4334731B1 claims 7 "
     "and 8, US12021452B2 claim 1, US8965721B2 claim 1). Any quotient is isomorphic "
     "to the first of them, so the match is refused rather than returned as KNOWN"),
    ("arrhenius_rate", "A*exp(-Ea/(R*T))", NOT_FOUND_IN_PATENT_CORPUS,
     "STRUCTURAL-TIER LIMIT: an Arrhenius law does not share an operation skeleton "
     "with US11474159B2 claim 7 (`A_SEI*exp(B_SEI/T) + C_SEI`): one is a product, the "
     "other a sum. It read RELATED when the tier compared degree and symbol count, "
     "which is why the tier now compares the operation tree. This row stays as the "
     "pin for 'not related just because both use exp'"),
)


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def index_records(corpus_path: pathlib.Path) -> list[PatentRecord]:
    """The corpus as validated records, for the refusal listing."""
    rows = [json.loads(line) for line in corpus_path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    return [PatentRecord.from_dict(row) for row in rows]


def build_record(corpus_path: pathlib.Path) -> dict:
    rows = [json.loads(line) for line in corpus_path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    index = PatentPriorArtIndex(PatentRecord.from_dict(row) for row in rows)
    results = []
    mismatches = []
    ordering_violations = []
    for identifier, expression, expected, why in CANDIDATES:
        outcome = index.audit(expression)
        agrees = outcome.status == expected
        if not agrees:
            mismatches.append((identifier, expected, outcome.status))
        # The invariant the ordering probe exists for: a form with structure is never
        # refused as operand-only, whatever an operand-only claim is isomorphic to.
        if not is_operand_only(expression) and outcome.status == CLAIM_STATES_OPERANDS_ONLY:
            ordering_violations.append({"id": identifier, "status": outcome.status})
        results.append({
            "id": identifier,
            "expression": expression,
            "expected_status": expected,
            "observed_status": outcome.status,
            "expectation_met": agrees,
            "why": why,
            "closest_patent_id": outcome.closest_patent_id,
            "matched_claim_number": outcome.matched_claim_number,
            "decisive_claims": outcome.decisive_claims,
            "operand_only_claims": outcome.operand_only_claims,
            "matched_expression": outcome.matched_expression,
            "matched_claim_is_decisive": (
                None if outcome.matched_expression is None
                else not is_operand_only(outcome.matched_expression)),
            "isomorphism": outcome.isomorphism,
            "searched_patents": outcome.searched_patents,
            "searchable_claims": outcome.searchable_claims,
            "synthetic_records": outcome.synthetic_records,
            "epistemic_disclaimer": outcome.epistemic_disclaimer,
        })

    # A count says six claims were subtracted; the list says which six, on what form,
    # and why — the refusals are the part of the verdict a reader cannot reconstruct.
    refusals = operand_only_refusals(index.records)
    refusals_by_record: dict[str, list[dict]] = {}
    for refusal in refusals:
        refusals_by_record.setdefault(refusal.patent_id, []).append(refusal.to_dict())
    form_classes: dict[str, int] = {}
    for refusal in refusals:
        form_classes[refusal.form_class] = form_classes.get(refusal.form_class, 0) + 1

    records = []
    by_id = {row["patent_id"]: row for row in rows}
    for record in index.records:
        row = by_id[record.patent_id]
        raw = ROOT / "data" / "patents" / "raw" / f"{record.patent_id}.json"
        artefact = json.loads(raw.read_text(encoding="utf-8"))
        records.append({
            "patent_id": record.patent_id,
            "jurisdiction": record.jurisdiction,
            "title": record.title,
            "claim_number": row["claim_number"],
            "normalised_expression": record.expression,
            "expression_note": row.get("expression_note", ""),
            "claims": row.get("claims", []),
            "searchable_claims": len(record.claims),
            "source": record.source,
            "retrieved": record.retrieved,
            "recorded_html_sha256": artefact["html_sha256"],
            "raw_artefact": str(raw.relative_to(ROOT)),
            "raw_artefact_sha256": _sha256(raw),
            "searchable": bool(record.claims),
            "refused_claims": refusals_by_record.get(record.patent_id, []),
            "decisive_claims": (len(record.claims)
                                - len(refusals_by_record.get(record.patent_id, []))),
        })

    return {
        "schema": "apedp.patent_prior_art.run/v1",
        "corpus_path": str(corpus_path.relative_to(ROOT)),
        "corpus_sha256": _sha256(corpus_path),
        "corpus": {
            "records": len(index.records),
            "searchable_claims": len(index.searchable),
            "decisive_claims": index.decisive_claims,
            "operand_only_claims": index.operand_only_claims,
            "records_with_formula": index.records_with_formula,
            "max_claims_per_record": max((len(r.claims) for r in index.records), default=0),
            "unsearchable_records": len(index.records) - index.records_with_formula,
            "synthetic_records": index.synthetic_records,
            "jurisdictions": sorted({r.jurisdiction for r in index.records}),
        },
        "records": records,
        "refusals": {
            "total": len(refusals),
            "by_form_class": dict(sorted(form_classes.items())),
            "by_record": {patent_id: [entry["claim_number"] for entry in entries]
                          for patent_id, entries in sorted(refusals_by_record.items())},
            "detail": [refusal.to_dict() for refusal in refusals],
            "markdown": str(REFUSALS.relative_to(ROOT)),
        },
        "candidates": results,
        "expectations_met": len(mismatches) == 0,
        "mismatches": [
            {"id": i, "expected": e, "observed": o} for i, e, o in mismatches
        ],
        "ordering_violations": ordering_violations,
        "scope": (
            "Every claim whose text states an explicit relation was normalised by "
            "hand — one to three per record, ten claims across five records — so a "
            "verdict is not bounded by a single claim per patent. Claims that state "
            "no relation are counted as unsearchable; claims that were read but not "
            "listed either state no relation or repeat one already recorded. "
            "Absence here is corpus-relative and is not a novelty or "
            "freedom-to-operate opinion."),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", default=str(CORPUS))
    parser.add_argument("--write", action="store_true", default=True)
    args = parser.parse_args(argv)

    corpus_path = pathlib.Path(args.corpus)
    record = build_record(corpus_path)

    print(f"corpus {record['corpus_path']}: {record['corpus']['records']} records, "
          f"{record['corpus']['searchable_claims']} searchable claims across "
          f"{record['corpus']['records_with_formula']} records, "
          f"{record['corpus']['synthetic_records']} synthetic")
    print()
    for candidate in record["candidates"]:
        mark = "ok " if candidate["expectation_met"] else "MISMATCH"
        print(f"{mark} {candidate['id']:<22} {candidate['observed_status']:<34} "
              f"{candidate['closest_patent_id'] or '-'}"
              f"{'' if not candidate['matched_claim_number'] else ' claim ' + str(candidate['matched_claim_number'])}")
        if candidate["observed_status"] == RELATED_STRUCTURE_IN_PATENT_CORPUS:
            print(f"       {candidate['isomorphism']}")

    if args.write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        REFUSALS.write_text(refusals_markdown(index_records(corpus_path)), encoding="utf-8")
        print(f"\nwrote {OUT.relative_to(ROOT)} and {REFUSALS.relative_to(ROOT)}")
        print(f"refused claims: {record['refusals']['total']} "
              f"({', '.join(f'{k}: {v}' for k, v in record['refusals']['by_form_class'].items())})")
        for patent_id, claim_numbers in record["refusals"]["by_record"].items():
            print(f"  {patent_id}: claims {', '.join(claim_numbers)} refused")

    if record["ordering_violations"]:
        print("\nA STRUCTURED CANDIDATE WAS REFUSED AS OPERAND-ONLY:", file=sys.stderr)
        for violation in record["ordering_violations"]:
            print(f"  {violation['id']}: {violation['status']}", file=sys.stderr)
        return 1

    if not record["expectations_met"]:
        print("\nENGINE DISAGREED WITH A DECLARED EXPECTATION:", file=sys.stderr)
        for mismatch in record["mismatches"]:
            print(f"  {mismatch['id']}: expected {mismatch['expected']}, "
                  f"observed {mismatch['observed']}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
