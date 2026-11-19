"""Pins for the bundled real patent corpus and the recorded prior-art run.

The corpus is small on purpose. What is pinned is not coverage but *traceability*:
each corpus row must be reproducible from a fetched artefact whose digest is
recorded, each expression must be the curated one, and the run's declared
expectations must hold — including the one that documents a transcription defect.
"""
from __future__ import annotations

import hashlib
import json
import pathlib

import pytest

from apedp.patent_prior_art import (
    BUNDLED_CORPUS_PATH,
    CLAIM_STATES_OPERANDS_ONLY,
    KNOWN_IN_PATENT_CORPUS,
    NOT_FOUND_IN_PATENT_CORPUS,
    RELATED_STRUCTURE_IN_PATENT_CORPUS,
    PatentPriorArtIndex,
    corpus_status,
    load_patent_corpus,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data" / "patents"
CORPUS = DATA / "corpus.jsonl"
CURATED = DATA / "normalised_claims.json"
RUN = REPO_ROOT / "benchmarks" / "patent_prior_art" / "prior_art_run.json"


def _rows() -> list[dict]:
    return [json.loads(line) for line in CORPUS.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def test_corpus_is_bundled_and_its_coverage_is_stated():
    status = corpus_status()
    assert status["bundled"] is True
    assert status["requires_caller_supplied_corpus"] is False
    assert status["records"] == 29
    assert status["searchable_claims"] == 15
    assert status["decisive_claims"] == 8
    assert status["operand_only_claims"] == 7
    assert status["records_with_formula"] == 7
    assert status["max_claims_per_record"] == 4
    assert status["unsearchable_records"] == 22
    assert status["synthetic_records"] == 0
    assert status["claims_scanned_for_formulas"] == 501
    assert status["formulas_found_in_claim_text"] == 0
    assert "one to four per record" in status["reason"].lower()
    assert BUNDLED_CORPUS_PATH is not None and BUNDLED_CORPUS_PATH.exists()


    assert status["decisive_claims_floor"] == 50
    assert status["meets_decisive_floor"] is False
    assert status["thin"] is True
    # Every verdict says the index is thin, so absence cannot be read as novelty.
    from apedp.patent_prior_art import PatentPriorArtIndex

    verdict = PatentPriorArtIndex.from_jsonl(CORPUS).audit("a*exp(-b*x)")
    assert "declared floor of 50" in verdict.epistemic_disclaimer
    assert "thin index" in verdict.epistemic_disclaimer


def test_missing_corpus_still_names_the_absence_and_the_bundled_path():
    with pytest.raises(FileNotFoundError, match="patent corpus not found") as excinfo:
        load_patent_corpus(DATA / "no-such-corpus.jsonl")
    assert "bundled at" in str(excinfo.value)


def test_every_record_is_an_attributed_real_patent():
    rows = _rows()
    assert len({row["patent_id"] for row in rows}) == len(rows)
    for row in rows:
        assert row["synthetic"] is False
        assert row["source"].startswith("https://patents.google.com/patent/")
        assert row["retrieved"] in {"2026-10-31", "2026-11-01"}
        assert row["jurisdiction"] in {"US", "EP"}
        assert row["claim"].strip()
        assert len(row["html_sha256"]) == 64
        int(row["html_sha256"], 16)


def test_claim_text_is_byte_identical_to_the_fetched_artefact():
    """The corpus claim is quoted from the retrieval, not retyped."""
    for row in _rows():
        raw = json.loads((DATA / "raw" / f"{row['patent_id']}.json").read_text(encoding="utf-8"))
        assert raw["source"] == row["source"]
        assert raw["html_sha256"] == row["html_sha256"]
        claims = {claim["num"]: claim["text"] for claim in raw["claims"]}
        assert claims[row["claim_number"]] == row["claim"], row["patent_id"]


def test_expressions_are_exactly_the_curated_ones():
    curated = json.loads(CURATED.read_text(encoding="utf-8"))
    assert curated.get("_schema", "").endswith("/v2")
    curated_rows = 0
    curated_claims = 0
    for row in _rows():
        if row["patent_id"] not in curated:
            continue
        entry = curated[row["patent_id"]]
        expected = [(str(item["claim_number"]), item.get("expression") or None,
                     item.get("note", "")) for item in entry.get("claims", [])]
        observed = [(claim["claim_number"], claim["expression"], claim["expression_note"])
                    for claim in row["claims"]]
        assert observed == expected, row["patent_id"]
        curated_claims += sum(1 for _, expression, _ in expected if expression)
        curated_rows += 1
    assert curated_rows == 29, "the curated layer covers every record, seven of them searchable"
    assert curated_claims == 15


def test_every_record_that_states_a_relation_holds_two_or_three_claims():
    """The one-claim-per-record limit no longer bounds a verdict."""
    per_record = {row["patent_id"]: [c for c in row["claims"] if c["expression"]]
                  for row in _rows()}
    assert len(per_record["EP4334731B1"]) == 3
    # Four here, not three: claim 1 defers to three relation functions, and the
    # simplified second one (claim 6) states a form distinct from claims 5 and 7.
    assert len(per_record["US11474159B2"]) == 4
    for patent_id in ("US8965721B2", "US20250321289A1", "US7120566B1"):
        assert len(per_record[patent_id]) == 2, patent_id
    # No record with a stated relation is left at a single claim, and only the one
    # whose claim 1 points at several relation functions exceeds three.
    # Harvested records contribute one claim each (the one their text states), so the
    # floor is now one; four is still the ceiling, and only one record reaches it.
    counts = [len(claims) for claims in per_record.values() if claims]
    assert counts and min(counts) >= 1 and max(counts) <= 4
    assert sum(1 for count in counts if count > 3) == 1
    assert per_record["US11496054B2"][0]["expression"] == "(d2*U2 - d1*U1)/d3"
    index = PatentPriorArtIndex.from_jsonl(CORPUS)
    assert len(index.searchable) == 15
    assert index.records_with_formula == 7


def test_harvested_records_record_what_claim_1_states():
    """Claim 1 of each harvested patent was read; none of them states a relation."""
    curated = json.loads(CURATED.read_text(encoding="utf-8"))
    for patent_id in ("US11474159B2", "US11921163B2", "US12130335B2",
                      "US12372585B2", "US20250321289A1"):
        reading = curated[patent_id]["claim_1"]
        assert reading["states_relation"] is False, patent_id
        assert len(reading["reason"]) > 80, patent_id
    assert "claim 6" in curated["US11474159B2"]["claim_1"]["reason"]
    rows = {row["patent_id"]: row for row in _rows()}
    expressions = {claim["expression"] for claim in rows["US11474159B2"]["claims"]}
    assert "-0.01*b_SEI*SOH + b_SEI" in expressions


def test_relation_free_records_are_counted_with_their_claim_sets_read():
    """Twenty-two records state no relation in any claim; recorded, not dropped."""
    curated = json.loads(CURATED.read_text(encoding="utf-8"))
    automatic = json.loads((DATA / "auto_expressions.json").read_text(encoding="utf-8"))
    assert automatic == {}, "the extractor found no formula; an entry here would need review"
    empty = [row for row in _rows() if not row["claims"]]
    assert len(empty) == 22
    for row in empty:
        assert row["expression"] is None
        entry = curated[row["patent_id"]]
        assert entry["claims"] == []
        assert "claims were read and scanned" in entry["_note"]
        assert str(len(json.loads(
            (DATA / "raw" / f"{row['patent_id']}.json").read_text(encoding="utf-8"))[
                "claims"])) in entry["_note"]
        raw = json.loads((DATA / "raw" / f"{row['patent_id']}.json").read_text(encoding="utf-8"))
        assert raw["claims"]


def test_extraction_yield_artefact_is_current_and_zero():
    """The growth step's own limit is a measurement, and it must stay reproducible."""
    report = json.loads((DATA / "extraction_yield.json").read_text(encoding="utf-8"))
    assert report["records_scanned"] == len(list((DATA / "raw").glob("*.json")))
    assert report["claims_scanned"] == sum(
        len(json.loads(path.read_text(encoding="utf-8"))["claims"])
        for path in (DATA / "raw").glob("*.json"))
    assert report["claims_with_a_parsed_formula"] == 0
    assert report["records_with_at_least_one_formula"] == 0


def test_records_without_a_stated_relation_are_counted_not_dropped():
    index = PatentPriorArtIndex.from_jsonl(CORPUS)
    assert len(index.records) == 29
    assert len(index.searchable) == 15
    result = index.audit("a*x/(1+b*x)")
    assert result.searched_patents == 29
    assert result.searchable_claims == 15
    assert "7 records with a stated relation" in result.epistemic_disclaimer


def test_declared_expectations_hold_in_a_fresh_run():
    """The recorded run must reproduce, and every expectation must be met."""
    from scripts.run_patent_prior_art import CANDIDATES, build_record

    fresh = build_record(CORPUS)
    assert fresh["expectations_met"] is True, fresh["mismatches"]
    observed = {candidate["id"]: candidate["observed_status"] for candidate in fresh["candidates"]}
    assert observed == {
        "self_check_exponential": KNOWN_IN_PATENT_CORPUS,
        "rename_equivalent": KNOWN_IN_PATENT_CORPUS,
        # An operands-and-one-operation claim cannot carry a KNOWN verdict.
        "self_check_efficiency": CLAIM_STATES_OPERANDS_ONLY,
        "saturating_response": NOT_FOUND_IN_PATENT_CORPUS,
        "generic_ratio": CLAIM_STATES_OPERANDS_ONLY,
        # A coefficient is structure, so the bare-quotient claim must not swallow
        # this candidate, and a scaled quotient shares no functional form with
        # `1 - SS_res/SS_tot` now that the tier compares the operation tree.
        "structured_not_swallowed": NOT_FOUND_IN_PATENT_CORPUS,
        # Same functional form as claim 6 with a different slope: related, not known.
        "affine_slope_differs": RELATED_STRUCTURE_IN_PATENT_CORPUS,
        # The structural tier no longer matches on degree and symbol count alone, so
        # an Arrhenius law is absent rather than 'related' to an exp-plus-constant
        # temperature law.
        "arrhenius_rate": NOT_FOUND_IN_PATENT_CORPUS,
    }
    assert {c[0] for c in CANDIDATES} == set(observed)

    # The ordering invariant, checked on the recorded run as well as in the engine:
    # no candidate with structure may be refused as operand-only, and the structured
    # probe that used to be swallowed must land on a decisive claim.
    from apedp.patent_prior_art import is_operand_only

    recorded = json.loads(RUN.read_text(encoding="utf-8"))
    assert recorded["ordering_violations"] == []
    for candidate in recorded["candidates"]:
        if not is_operand_only(candidate["expression"]):
            assert candidate["observed_status"] != CLAIM_STATES_OPERANDS_ONLY
    structured = next(c for c in recorded["candidates"]
                      if c["id"] == "structured_not_swallowed")
    assert structured["observed_status"] == NOT_FOUND_IN_PATENT_CORPUS
    assert structured["matched_claim_is_decisive"] is None
    # The structural tier still fires — on a shared functional form, not a shared
    # degree and symbol count.
    affine = next(c for c in recorded["candidates"] if c["id"] == "affine_slope_differs")
    assert affine["observed_status"] == RELATED_STRUCTURE_IN_PATENT_CORPUS
    assert affine["matched_claim_is_decisive"] is True
    assert affine["isomorphism"].startswith("STRUCTURAL_SHAPE_MATCH")


def test_refused_claims_are_listed_per_record_with_a_reason():
    """The six operand-only refusals are enumerated, not only counted."""
    from apedp.patent_prior_art import (
        is_operand_only,
        operand_only_form_class,
        operand_only_refusals,
        refusals_markdown,
    )

    status = corpus_status()
    refusals = operand_only_refusals()
    assert len(refusals) == status["operand_only_claims"] == 7
    assert len(status["operand_only"]) == 7

    by_record: dict[str, list[str]] = {}
    for refusal in refusals:
        by_record.setdefault(refusal.patent_id, []).append(refusal.claim_number)
        assert is_operand_only(refusal.expression), refusal.expression
        assert refusal.form_class == operand_only_form_class(refusal.expression)
        assert refusal.form_class in {
            "bare quotient", "bare product", "bare symbol (an equality of names)"}
        assert refusal.why

    # Pinned per record, because the point is which claims are refused, not how many.
    assert by_record == {
        "EP4334731B1": ["2", "7", "8"],
        "US12021452B2": ["1"],
        "US20250321289A1": ["11"],
        "US8965721B2": ["1", "8"],
    }

    markdown = refusals_markdown()
    for patent_id, claim_numbers in by_record.items():
        section = markdown.split(f"## {patent_id}")[1]
        assert f"{len(claim_numbers)} of" in section
        for number in claim_numbers:
            assert f"| {number} |" in section

    on_disk = (DATA / "REFUSED_CLAIMS.md").read_text(encoding="utf-8")
    assert on_disk == markdown, (
        "data/patents/REFUSED_CLAIMS.md is stale; re-run scripts/run_patent_prior_art.py")

    recorded = json.loads(RUN.read_text(encoding="utf-8"))
    assert recorded["refusals"]["total"] == 7
    assert recorded["refusals"]["by_record"] == by_record
    assert recorded["refusals"]["markdown"] == "data/patents/REFUSED_CLAIMS.md"
    for record in recorded["records"]:
        assert len(record["refused_claims"]) == len(
            by_record.get(record["patent_id"], []))
        assert record["decisive_claims"] == record["searchable_claims"] - len(
            record["refused_claims"])


def test_recorded_run_is_not_stale():
    recorded = json.loads(RUN.read_text(encoding="utf-8"))
    assert recorded["corpus_sha256"] == hashlib.sha256(CORPUS.read_bytes()).hexdigest()
    assert recorded["corpus"] == {
        "records": 29,
        "searchable_claims": 15,
        "decisive_claims": 8,
        "operand_only_claims": 7,
        "records_with_formula": 7,
        "max_claims_per_record": 4,
        "unsearchable_records": 22,
        "synthetic_records": 0,
        "jurisdictions": ["EP", "US"],
    }
    for record in recorded["records"]:
        raw = REPO_ROOT / record["raw_artefact"]
        assert hashlib.sha256(raw.read_bytes()).hexdigest() == record["raw_artefact_sha256"]


def test_a_coarse_transcription_matches_any_quotient_and_says_so():
    """The documented defect: a bare ratio record matches an unrelated quotient.

    Pinned rather than fixed, because the fix is a better transcription (or a
    matching rule that refuses to conclude from an operand-free form), not a
    silence. If this test ever fails because the record was sharpened, update the
    note in ``data/patents/normalised_claims.json`` and this pin together.
    """
    index = PatentPriorArtIndex.from_jsonl(CORPUS)
    result = index.audit("speed_of_light/speed_of_sound")
    # A bare quotient is no longer a conclusion: the claim names its operands and
    # nothing about them, so the match is refused rather than reported as known.
    assert result.status == CLAIM_STATES_OPERANDS_ONLY
    assert result.matched_expression in {"C_t_I/C_0_I", "C_t_E/C_0_E", "V_in/V_out",
                                       "Z_meas/Z_ref"}
    assert "/" in (result.matched_expression or "")
    assert result.closest_patent_id in {"EP4334731B1", "US12021452B2", "US8965721B2"}
    assert "not evidence" in result.epistemic_disclaimer
    assert "refused rather than reported as known" in result.epistemic_disclaimer
